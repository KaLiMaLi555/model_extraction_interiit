import numpy as np
import torch
import torch.nn.functional as F
import wandb

from torchmetrics.functional import accuracy
import network


def estimate_gradient_objective(args, teacher, x, labels=None, epsilon=1e-7, m=5, num_classes=400, device="cpu", pre_x=False):
    # Sampling from unit sphere is the method 3 from this website:
    #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    # x = torch.Tensor(np.arange(2*1*7*7).reshape(-1, 1, 7, 7))
    # x (N, C, L, S, S)

    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    teacher.eval()
    with torch.no_grad():
        # Sample unit noise vector
        N = x.size(0)
        C = x.size(1)
        L = x.size(2)
        S = x.size(3)
        dim = S ** 2 * C * L

        labels = labels.to(device).unsqueeze(0).repeat(m + 1, 1).transpose(0, 1).reshape(-1)

        u = np.random.randn(N * m * dim).reshape(-1, m, dim)  # generate random points from normal distribution

        d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).to(device).view(-1, m, C, L, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, L, S, S).to(device)), dim=1)  # Shape N, m + 1, S^2

        u = u.view(-1, m + 1, C, L, S, S)

        evaluation_points = (x.view(-1, 1, C, L, S, S).to(device) + epsilon * u).view(-1, C, L, S, S)
        if pre_x:
            evaluation_points = args.G_activation(evaluation_points)  # Apply args.G_activation function

        # Compute the approximation sequentially to allow large values of m
        pred_teacher = []
        exp_labels = []
        max_number_points = 32  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

        # print()
        # print(evaluation_points.shape)
        # print(labels.shape)
        for i in (range(N * m // max_number_points + 1)):
            pts = evaluation_points[i * max_number_points: (i + 1) * max_number_points]
            pts = pts.to(device)

            swin_pts = network.swin.swin_transform(pts.detach())

            pred_teacher_pts = teacher(swin_pts, return_loss=False)
            pred_teacher.append(torch.Tensor(pred_teacher_pts).to(device))
            exp_labels.append(labels[i * max_number_points: (i + 1) * max_number_points])

        pred_teacher = torch.cat(pred_teacher, dim=0).to(device)
        exp_labels = torch.cat(exp_labels, dim=0).to(device)

        # print()
        # print(pred_teacher.shape)
        # print(exp_labels.shape)
        # u = u.to(device)

        conditional_loss = F.cross_entropy(pred_teacher, exp_labels, reduction='none').view(-1, m + 1).to(device)

        with torch.no_grad():
            print('Expected Labels')
            print(labels)
            print('Teacher predictions')
            print(torch.argmax(pred_teacher, dim=1))

            t1 = 100 * accuracy(pred_teacher, labels, top_k=1)
            t5 = 100 * accuracy(pred_teacher, labels, top_k=5)
            print('T1 accuracy')
            print(t1)
            wandb.log({'T1': t1.detach().cpu().numpy()})
            print('T5 accuracy')
            print(t5)
            wandb.log({'T5': t5.detach().cpu().numpy()})

            print(f'Conditional Loss: {conditional_loss[:, -1].mean().item()}')
            wandb.log({'loss_G_conditional': conditional_loss[:, -1].mean().item()})
        loss_values = conditional_loss

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
            # raise ValueError(args.loss)
            gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, L, S, S) / (num_classes * N)

        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G


def compute_gradient(args, teacher, x, labels=None, device="cpu", pre_x=False):
    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    x_copy = x.clone().detach().requires_grad_(True)
    x_ = x_copy.to(device)

    if pre_x:
        x_ = args.G_activation(x_)

    x_swin = network.swin.swin_transform(x_)
    pred_teacher, loss = teacher(x_swin, torch.nn.functional.one_hot(labels, 400))

    conditional_loss = F.cross_entropy(pred_teacher, labels, reduction='none')
    loss_values = conditional_loss
    # print("True mean loss", loss_values)
    loss_values.backward()

    return x_copy.grad, loss_values


class Args(dict):
    def __init__(self, **args):
        for k, v in args.items():
            self[k] = v
