import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

from utils_common import swin_transform


# TODO: Factor more of approx_grad and approx_grad_conditional into reusable
#       functions to reduce repetition.
#       Alternatively combine both functions, not sure if better


def get_evaluation_points(x, N, C, L, S, m, dim, epsilon, pre_x, G_activation):
    # Generate random points from normal distribution
    u = np.random.randn(N * m * dim).reshape(-1, m, dim)

    # Map to a uniform distribution on a unit sphere
    d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, m, 1)
    u = torch.Tensor(u / d).view(-1, m, C, L, S, S)
    u = torch.cat((u, torch.zeros(N, 1, C, L, S, S)), dim=1)

    # Shape N, m + 1, C, L, S, S
    u = u.view(-1, m + 1, C, L, S, S)

    evaluation_points = (x.view(-1, 1, C, L, S, S).cpu() + epsilon * u)
    evaluation_points = evaluation_points.view(-1, C, L, S, S)

    # Apply Generator activation function if not done
    if pre_x:
        evaluation_points = G_activation(evaluation_points)
    return u, evaluation_points


def get_swint_pts(victim_model, pts):
    swin_pts = swin_transform(pts)
    pred_victim_pts = victim_model(swin_pts.cuda(), return_loss=False)
    return torch.tensor(pred_victim_pts)


def get_movinet_pts(victim_model, pts, N, C, L, S, device, device_tf):
    # Movinet expects: b, f, h, w, c = N, L, S, S, C
    pts_tf = pts.reshape(-1, L, S, S, C).detach().cpu().numpy()
    with tf.device(device_tf):
        tf_tensor = tf.convert_to_tensor(pts_tf)
        pred_victim_pts = victim_model(tf_tensor).numpy()
    pred_victim_pts = torch.tensor(pred_victim_pts).to(device)
    return pred_victim_pts


def get_grad_estimates(args, loss_vals, dim, epsilon, u, m, C, L, S):
    # Compute difference following each direction
    differences = loss_vals[:, :-1] - loss_vals[:, -1].view(-1, 1)
    differences = differences.view(-1, m, 1, 1, 1, 1)

    # Formula for Forward Finite Differences
    grad_estimates = 1 / epsilon * differences * u[:, :-1]
    if args.forward_differences:
        grad_estimates *= dim

    # TODO: Decide if we're keeping l1 or other funcs, add conditionals for it
    if args.loss == "kl":
        grad_estimates = grad_estimates.mean(dim=1).view(-1, C, L, S, S)
    else:
        grad_estimates = grad_estimates.mean(dim=1).view(-1, C, L, S, S) / (args.num_classes * grad_estimates.size(0))

    return grad_estimates.detach()


def approximate_gradients(
        args, victim_model, threat_model, x, epsilon=1e-7, m=5,
        device='cpu', device_tf='/device:GPU:0', pre_x=False):
    # x shape: (N, C, L, S, S)

    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    with torch.no_grad():
        # Sample unit noise vector
        N, C, L, S = x.shape[:4]
        dim = S ** 2 * C * L

        # Get points to evaluate model at
        u, evaluation_points = get_evaluation_points(
            x, N, C, L, S, m, dim, epsilon, pre_x, args.G_activation)

        # Compute the approximation sequentially to allow large values of m
        pred_victim, pred_threat = [], []
        # Value to split the large evaluation_points tensor to fit in GPU
        max_points = args.max_points

        for i in (range(N * m // max_points + 1)):
            pts = evaluation_points[i * max_points:(i + 1) * max_points]

            # TODO: Update this to work with cfg parser
            if args.victim_model == 'swin-t':
                pred_victim_pts = get_swint_pts(victim_model, pts)
            else:
                pred_victim_pts = get_movinet_pts(
                    victim_model, pts, N, C, L, S, device=device,
                    device_tf=device_tf)

            pred_threat_pts = threat_model(pts)

            pred_victim.append(pred_victim_pts)
            pred_threat.append(pred_threat_pts)

        pred_victim = torch.cat(pred_victim, dim=0).to(device)
        pred_threat = torch.cat(pred_threat, dim=0).to(device)

        u = u.to(device)

        # TODO: Remove this if we're not adding l1 below
        if args.loss == "l1":
            loss_fn = F.l1_loss
            if args.no_logits:
                pred_victim = torch.log(pred_victim)
                if args.logit_correction == 'min':
                    pred_victim -= pred_victim.min(dim=1).values.view(-1, 1)
                elif args.logit_correction == 'mean':
                    pred_victim -= pred_victim.mean(dim=1).view(-1, 1)
            pred_victim = pred_victim.detach()

        elif args.loss == "kl":
            loss_fn = F.kl_div
            pred_threat = F.log_softmax(pred_threat, dim=1)
            pred_victim = pred_victim.detach()

        else:
            raise ValueError(args.loss)

        # Compute loss
        if args.loss == "kl":
            loss_vals = - loss_fn(pred_threat, pred_victim, reduction='none')
            loss_vals = loss_vals.sum(dim=1).view(-1, m + 1)
        else:
            loss_vals = - loss_fn(pred_threat, pred_victim, reduction='none').mean(dim=1).view(-1, m + 1)

        grad_estimates = get_grad_estimates(
            args, loss_vals, dim, epsilon, u, m, C, L, S)
        loss_G = loss_vals[:, -1].mean()
        return grad_estimates, loss_G


def approximate_gradients_conditional(
        args, victim_model, x, labels, epsilon=1e-7, m=5,
        device='cpu', device_tf='/device:GPU:0', pre_x=False):
    # x shape: (N, C, L, S, S)

    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    with torch.no_grad():
        # Sample unit noise vector
        N, C, L, S, _ = x.shape
        dim = S ** 2 * C * L

        labels = labels.unsqueeze(0).repeat(m + 1, 1).transpose(0, 1).reshape(-1)

        # Get points to evaluate model at
        u, evaluation_points = get_evaluation_points(
            x, N, C, L, S, m, dim, epsilon, pre_x, args.G_activation)

        # Compute the approximation sequentially to allow large values of m
        pred_victim, pred_threat = [], []
        # Value to split the large evaluation_points tensor to fit in GPU
        max_points = args.max_points

        for i in (range(N * m // max_points + 1)):
            pts = evaluation_points[i * max_points:(i + 1) * max_points]

            # TODO: Update this to work with cfg parser
            if args.victim_model == 'swin-t':
                pred_victim_pts = get_swint_pts(victim_model, pts)
            else:
                pred_victim_pts = get_movinet_pts(
                    victim_model, pts, N, C, L, S, device=device,
                    device_tf=device_tf)

            pred_victim.append(pred_victim_pts)
        pred_victim = torch.cat(pred_victim, dim=0).to(device)

        u = u.to(device)

        # TODO: Remove this if we're not adding l1 below
        # if args.loss == "l1":
        #     loss_fn = F.l1_loss
        #     if args.no_logits:
        #         pred_victim = torch.log(pred_victim)
        #         if args.logit_correction == 'min':
        #             pred_victim -= pred_victim.min(dim=1).values.view(-1, 1)
        #         elif args.logit_correction == 'mean':
        #             pred_victim -= pred_victim.mean(dim=1).view(-1, 1)
        #         pred_victim = pred_victim.detach()

        # # TODO: Verify that KL Div doesn't have an issue with values being 0.
        # #       labels is a one-hot vector, can't use this if
        # #       log_softmax/kldiv/anything has an issue with it
        # elif args.loss == "kl":
        #     loss_fn = F.kl_div
        #     pred_threat = F.log_softmax(labels, dim=1)
        #     pred_victim = pred_victim.detach()

        if args.loss == "cross-entropy":
            loss_fn = F.cross_entropy
        else:
            raise ValueError(args.loss)

        # Compute loss
        # TODO: Decide if we're keeping l1 or other funcs, add conditionals for it
        if args.loss == "kl":
            loss_vals = loss_fn(pred_victim, labels, reduction='none')
            loss_vals = loss_vals.sum(dim=1).view(-1, m + 1)
        else:
            loss_vals = loss_fn(pred_victim, labels, reduction='none').view(-1, m + 1)

        grad_estimates = get_grad_estimates(
            args, loss_vals, dim, epsilon, u, m, C, L, S)
        loss_G = loss_vals[:, -1].mean()
        return grad_estimates, loss_G
