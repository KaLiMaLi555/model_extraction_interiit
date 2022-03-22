import network
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F


def get_evaluation_points(x, N, C, L, S, m, dim, epsilon, pre_x, G_activation):
    u = np.random.randn(N * m * dim).reshape(-1, m, dim)  # generate random points from normal distribution

    d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
    u = torch.Tensor(u / d).view(-1, m, C, L, S, S)
    u = torch.cat((u, torch.zeros(N, 1, C, L, S, S)), dim=1)  # Shape N, m + 1, S^2

    u = u.view(-1, m + 1, C, L, S, S)

    evaluation_points = (x.view(-1, 1, C, L, S, S).cpu() + epsilon * u).view(-1, C, L, S, S)

    # Apply Generator activation function if not done
    if pre_x:
        evaluation_points = G_activation(evaluation_points)
    return u, evaluation_points


def get_swint_pts(victim_model, pts):
    swin_pts = network.swin.swin_transform(pts)
    pred_victim_pts = victim_model(swin_pts).detach()
    return pred_victim_pts


def get_movinet_pts(victim_model, pts, N, C, L, S, device='cpu', device_tf='/device:GPU:0'):
    # Movinet expects: b, f, h, w, c = N, L, S, S, C
    pts_tf = pts.reshape(N, L, S, S, C).detach().cpu().numpy()
    with tf.device(device_tf):
        tf_tensor = tf.convert_to_tensor(pts_tf)
        pred_victim_pts = victim_model(tf_tensor).numpy()
    pred_victim_pts = torch.tensor(pred_victim_pts).to(device)
    return pred_victim_pts


def approximate_gradients(args, victim_model, threat_model, x, epsilon=1e-7, m=5, device='cpu', device_tf='/device:GPU:0', pre_x=False):
    # x shape: (N, C, L, S, S)

    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    # TODO: Update this to work with cfg parser
    threat_model.eval()
    if args.model == 'swin-t':
        victim_model.eval()

    with torch.no_grad():
        # Sample unit noise vector
        N, C, L, S, _ = x.shape
        dim = S ** 2 * C * L

        # Get points to evaluate model at
        u, evaluation_points = get_evaluation_points(N, C, L, S, m, dim, epsilon, pre_x, x, args)

        # Compute the approximation sequentially to allow large values of m
        pred_victim, pred_threat = [], []
        max_number_points = 32  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

        for i in (range(N * m // max_number_points + 1)):
            pts = evaluation_points[i * max_number_points: (i + 1) * max_number_points]

            # TODO: Update this to work with cfg parser
            if args.model == 'swin-t':
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
                pred_victim = torch.log(pred_victim).detach()
                if args.logit_correction == 'min':
                    pred_victim -= pred_victim.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    pred_victim -= pred_victim.mean(dim=1).view(-1, 1).detach()

        elif args.loss == "kl":
            loss_fn = F.kl_div
            pred_threat = F.log_softmax(pred_threat, dim=1)
            pred_victim = pred_victim.detach()

        else:
            raise ValueError(args.loss)

        # Compute loss
        # TODO: Decide if we're keeping l1 or other funcs, add conditionals for it
        if args.loss == "kl":
            loss_values = - loss_fn(pred_threat, pred_victim, reduction='none').sum(dim=1).view(-1, m + 1)
        else:
            raise ValueError(args.loss)
            # loss_values = - loss_fn(pred_threat, pred_victim, reduction='none').mean(dim = 1).view(-1, m + 1)

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        if args.forward_differences:
            gradient_estimates *= dim

        # TODO: Decide if we're keeping l1 or other funcs, add conditionals for it
        if args.loss == "kl":
            gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, L, S, S)
        else:
            raise ValueError(args.loss)
            # gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, L, S, S) / (num_classes * N)

        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G


def approximate_gradients_conditional(args, victim_model, labels, x, epsilon=1e-7, m=5, device='cpu', device_tf='/device:GPU:0', pre_x=False):
    # x shape: (N, C, L, S, S)

    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    # TODO: Update this to work with cfg parser
    if args.model == 'swin-t':
        victim_model.eval()

    with torch.no_grad():
        # Sample unit noise vector
        N, C, L, S, _ = x.shape
        dim = S ** 2 * C * L

        # Get points to evaluate model at
        u, evaluation_points = get_evaluation_points(N, C, L, S, m, dim, epsilon, pre_x, x, args)
        print('DEBUG, remove before submission:', evaluation_points.shape, labels.shape)
        print('verify that the first dim of the above 2 are the same, if not then changes needed')

        # Compute the approximation sequentially to allow large values of m
        pred_victim = []
        max_number_points = 32  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

        for i in (range(N * m // max_number_points + 1)):
            pts = evaluation_points[i * max_number_points: (i + 1) * max_number_points]

            # TODO: Update this to work with cfg parser
            if args.model == 'swin-t':
                pred_victim_pts = get_swint_pts(victim_model, pts)
            else:
                pred_victim_pts = get_movinet_pts(
                    victim_model, pts, N, C, L, S, device=device,
                    device_tf=device_tf)

            pred_victim.append(pred_victim_pts)

        pred_victim = torch.cat(pred_victim, dim=0).to(device)

        u = u.to(device)

        # TODO: Remove this if we're not adding l1 below
        if args.loss == "l1":
            loss_fn = F.l1_loss
            if args.no_logits:
                pred_victim = torch.log(pred_victim).detach()
                if args.logit_correction == 'min':
                    pred_victim -= pred_victim.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    pred_victim -= pred_victim.mean(dim=1).view(-1, 1).detach()

        # TODO: Verify that KL Div doesn't have an issue with values being 0.
        #       labels is a one-hot vector, can't use this if
        #       log_softmax/kldiv/anything has an issue with it
        elif args.loss == "kl":
            loss_fn = F.kl_div
            pred_threat = F.log_softmax(labels, dim=1)
            pred_victim = pred_victim.detach()

        else:
            raise ValueError(args.loss)

        # Compute loss
        # TODO: Decide if we're keeping l1 or other funcs, add conditionals for it
        if args.loss == "kl":
            loss_values = - loss_fn(pred_threat, pred_victim, reduction='none').sum(dim=1).view(-1, m + 1)
        else:
            raise ValueError(args.loss)

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        if args.forward_differences:
            gradient_estimates *= dim

        # TODO: Decide if we're keeping l1 or other funcs, add conditionals for it
        if args.loss == "kl":
            gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, L, S, S)
        else:
            raise ValueError(args.loss)
            # gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, L, S, S) / (num_classes * N)

        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G
