import network
import numpy as np
import torch
import torch.nn.functional as F


def approximate_gradients_swint(args, victim_model, clone_model, x, epsilon=1e-7, m=5, verb=False, num_classes=400, device="cpu", pre_x=False):
    # Sampling from unit sphere is the method 3 from this website:
    #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    # x = torch.Tensor(np.arange(2*1*7*7).reshape(-1, 1, 7, 7))
    # x (N, C, L, S, S)

    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    clone_model.eval()
    victim_model.eval()
    with torch.no_grad():
        # Sample unit noise vector
        N = x.size(0)
        C = x.size(1)
        L = x.size(2)
        S = x.size(3)
        dim = S ** 2 * C * L

        u = np.random.randn(N * m * dim).reshape(-1, m, dim)  # generate random points from normal distribution

        d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).view(-1, m, C, L, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, L, S, S)), dim=1)  # Shape N, m + 1, S^2

        u = u.view(-1, m + 1, C, L, S, S)

        evaluation_points = (x.view(-1, 1, C, L, S, S).cpu() + epsilon * u).view(-1, C, L, S, S)
        if pre_x:
            evaluation_points = args.G_activation(evaluation_points)  # Apply args.G_activation function

        # Compute the approximation sequentially to allow large values of m
        pred_victim = []
        pred_clone = []
        max_number_points = 32  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

        for i in (range(N * m // max_number_points + 1)):
            pts = evaluation_points[i * max_number_points: (i + 1) * max_number_points]
            pts = pts.to(device)

            swin_pts = network.swin.swin_transform(pts)

            pred_victim_pts = victim_model(swin_pts).detach()
            pred_clone_pts = clone_model(pts)

            pred_victim.append(pred_victim_pts)
            pred_clone.append(pred_clone_pts)

        pred_victim = torch.cat(pred_victim, dim=0).to(device)
        pred_clone = torch.cat(pred_clone, dim=0).to(device)

        u = u.to(device)

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
            pred_clone = F.log_softmax(pred_clone, dim=1)
            pred_victim = pred_victim.detach()

        else:
            raise ValueError(args.loss)

        # Compute loss
        if args.loss == "kl":
            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').sum(dim=1).view(-1, m + 1)
        else:
            raise ValueError(args.loss)
            # loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').mean(dim = 1).view(-1, m + 1)

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        if args.forward_differences:
            gradient_estimates *= dim

        if args.loss == "kl":
            gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, L, S, S)
        else:
            raise ValueError(args.loss)
            # gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, L, S, S) / (num_classes * N)

        clone_model.train()
        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G


class Args(dict):
    def __init__(self, **args):
        for k, v in args.items():
            self[k] = v

