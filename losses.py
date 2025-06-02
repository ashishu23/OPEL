
import torch
from torch.nn import functional as F
from scipy.spatial.distance import cdist
import pdb
import math
from utils import get_lav_weights, generate_unique_video_steps


def calculate_similarity(embeddings1, embeddings2, temperature):
    nc = embeddings1.size(1)
    # L2 distance
    emb1_norm = (embeddings1**2).sum(dim=1)
    emb2_norm = (embeddings2**2).sum(dim=1)
    # Trick used to calculate the distance matrix without any loops. It uses
    # the fact that (a - b)^2 = a^2 + b^2 - 2ab.
    dist = torch.max(
        emb1_norm + emb2_norm - 2.0 * torch.matmul(
            embeddings1,
            embeddings2.t()
        ),
        torch.tensor(
            0.0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    )
    # similarity: (N, M)
    similarity = -1.0 * dist
    similarity /= embeddings1.size(1)
    similarity /= temperature
    return similarity


def embeddings_similarity(embeddings1, embeddings2, temperature):
    '''
    embeddings1: (N, D). in paper, U. u_i, i=1 to N
    embeddings2: (M, D). in paper, V. v_j, j=1 to M
    '''
    max_num_frames = embeddings1.size(0)

    similarity = calculate_similarity(embeddings1, embeddings2, temperature)
    similarity = F.softmax(similarity, dim=1)
    # v_tilda
    soft_nearest_neighbor = torch.matmul(similarity, embeddings2)

    # logits for Beta_k
    logits = calculate_similarity(
        soft_nearest_neighbor,
        embeddings1,
        temperature
    )
    # labels = F.one_hot(
    #     torch.tensor(range(max_num_frames)),
    #     num_classes=max_num_frames
    # )
    labels = torch.eye(max_num_frames)[torch.tensor(range(max_num_frames))]

    return logits, labels


def contrastive_idm_loss(
    embeddings,
    steps,
    steps_norm,
    lambda_=2.0,
    sigma=15.0
):
    lambda_ = torch.tensor(lambda_, requires_grad=True).cuda()
    sigma = torch.tensor(sigma, requires_grad=True).cuda()
    unique_vid_steps = generate_unique_video_steps(embeddings, steps)
    unique_vid_steps_norm = generate_unique_video_steps(embeddings, steps_norm)
    assert embeddings.shape[0] == len(unique_vid_steps)
    losses = torch.tensor(()).to('cuda')
    for video_count, (single_video_embds, video_steps) in enumerate(
        zip(embeddings, unique_vid_steps_norm)
    ):
        loss = torch.tensor([0.0]).to('cuda')
        w, w_dash = get_lav_weights(video_steps)
        w = torch.tensor(w, requires_grad=True).to('cuda')
        w_dash = torch.tensor(w_dash, requires_grad=True).to('cuda')
        temporal_dist = torch.abs(
            unique_vid_steps[video_count].unsqueeze(dim=0) - \
                unique_vid_steps[video_count].unsqueeze(dim=1)
        )
        y_ = (
            torch.ones(temporal_dist.shape, requires_grad=True).cuda() * \
                (temporal_dist.cuda() > sigma.to(torch.int64)).to(torch.float32)
        )
        ## Calculating the self-distance matrix
        dist_calculation = single_video_embds.detach().cpu().numpy()
        self_dist_D = torch.tensor(cdist(
            dist_calculation,
            dist_calculation,
        )).to('cuda').to(torch.float32)
        max_values = torch.max(
            torch.tensor(0, dtype=torch.float32).to('cuda'),
            lambda_ - self_dist_D
        )
        loss = y_ * w_dash.to(torch.float32) * max_values + (torch.tensor(1.0).cuda() - y_) * w.to(torch.float32) * self_dist_D
        losses = torch.cat((losses, loss.sum().unsqueeze(dim=0)), 0)
    return losses.mean()


def cycleback_regression_loss(
    logits,
    labels,
    num_frames,
    steps,
    seq_lens,
    normalize_indices,
    variance_lambda,
):
    labels = labels.detach().cuda()  # (bs, ts)
    steps = steps.detach().cuda()  # (bs, ts)
    steps = steps.float().cuda()
    seq_lens = seq_lens.float().cuda()

    seq_lens = seq_lens.unsqueeze(1).repeat(1, num_frames).cuda()
    steps = steps / seq_lens

    # After using torch.nn.DataParallel, logits are on 'cuda' and rest of the
    # things are on 'cpu'. Moving beta to 'cuda' fixes the issue.
    beta = F.softmax(logits, dim=1).to('cuda')
    true_timesteps = (labels * steps).sum(dim=1)
    pred_timesteps = (beta * steps).sum(dim=1)
    pred_timesteps_repeated = pred_timesteps.unsqueeze(1).repeat(1, num_frames)
    pred_timesteps_var = (
        (steps - pred_timesteps_repeated)**2 * beta
    ).sum(dim=1)
    pred_timesteps_log_var = pred_timesteps_var.log()
    squared_error = (true_timesteps - pred_timesteps)**2
    loss = torch.mean(
        (-pred_timesteps_log_var).exp() * squared_error + variance_lambda * \
            pred_timesteps_log_var
    )
    return loss


def temporal_cycle_consistency_loss(
    embeddings,
    steps,
    seq_lens,
    cfg,
    num_frames,
    batch_size,
    temperature,
    variance_lambda,
    normalize_indices,
    writer=None,
    iter_count=None
):
    logits_list = []
    labels_list = []
    steps_list = []
    seq_lens_list = []
    # print(embeddings.shape)  # torch.Size([2, 32, 128])
    # print(batch_size)
    # pdb.set_trace()
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                logits, labels = embeddings_similarity(
                    embeddings[i],
                    embeddings[j],
                    temperature
                )
                logits_list.append(logits)
                labels_list.append(labels)
                steps_list.append(steps[i:i+1].repeat(num_frames, 1))
                seq_lens_list.append(seq_lens[i:i+1].repeat(num_frames))
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    steps = torch.cat(steps_list, dim=0)
    seq_lens = torch.cat(seq_lens_list, dim=0)

    loss = cycleback_regression_loss(
        logits,
        labels,
        num_frames,
        steps,
        seq_lens,
        normalize_indices,
        variance_lambda
    )

    if cfg.LAV.USE_CIDM:
        contrastive_loss = contrastive_idm_loss(
            embeddings,
            steps,
            steps / seq_lens.unsqueeze(1).repeat(1, num_frames),
            lambda_ = cfg.LAV.LAMBDA,
            sigma=cfg.LAV.SIGMA
        )
        if writer is not None:
            writer.add_scalar(
                'Loss/C-IDM',
                (cfg.LAV.CONTRIB_PERCENT * contrastive_loss).item(),
                iter_count
            )
            writer.add_scalar(
                'Loss/TCC',
                loss.item(),
                iter_count
            )
        return cfg.LAV.CONTRIB_PERCENT * contrastive_loss + loss

    return loss


def otuprel_gauss_loss(X, Y, maxIter=20, lambda1=1.0, lambda2=0.1, virtual_distance=5.0, zeta=0.5, sigma=2.0, epoch=None, device='cuda'):
  
    N, _ = X.shape
    M, _ = Y.shape

    N = torch.tensor(N).cuda()
    M = torch.tensor(M).cuda()
    # Adjust lambda1 and lambda2 according to the formula
    lambda1 = lambda1 * (N + M)
    lambda2 = lambda2 * (N * M) / 4.0

    D_x_y = torch.mean((X.unsqueeze(1) - Y.unsqueeze(0))**2, dim=2)
    min_index = torch.argmin(D_x_y, dim=1).float()

    N += 1
    M += 1

    power = int(torch.sqrt(100*torch.tensor(epoch) + 1.0))
        
    phi = 0.999 ** power
    phi = min(phi, 0.999)
    phi = max(phi, 0.001)

    P = torch.zeros((N, M), device=device)
    S = torch.zeros((N, M), device=device)

    mid_para = 1.0 / (N**2) + 1 / (M**2)
    mid_para = math.sqrt(mid_para)
    pi = math.pi
    pi = torch.tensor(pi)
    threshold_value = 2.0 * virtual_distance / (N + M)

    for i in range(1, N+1):
        for j in range(1, M+1):
            # the distance to diagonal
            d_prior = torch.abs(i/N - j/M)
            d_prior = d_prior / mid_para
            # the distance to the most similar matching for a given i, adding extra 1 for virtual frame
            if i > 1:
                d_similarity = torch.abs(j/M - (min_index[i-2]+1)/M)
            else:
                d_similarity = torch.abs(j/M - 1.0/M)
            d_similarity = d_similarity / mid_para
            p_consistency = torch.exp(-d_prior**2.0 / (2.0 * sigma**2)) / (sigma * torch.sqrt(2.0 * pi))
            p_optimal = torch.exp(-d_similarity**2.0 / (2.0 * sigma**2)) / (sigma * torch.sqrt(2.0 * pi))
            P[i-1, j-1] = phi * p_consistency + (1.0 - phi) * p_optimal
            # virtual frame prior value
            if (i == 1 or j == 1) and not(i == j):
                d = threshold_value * 1.5 / mid_para
                P[i-1, j-1] = torch.exp(-d**2.0 / (2.0 * sigma**2)) / (sigma * torch.sqrt(2.0 * pi))

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            s_consistency = torch.abs(i/N - j/M)

            if i > 1:
                s_optimal = torch.abs(j/M - (min_index[i-2] + 1)/M)
            else:
                s_optimal = torch.abs(j/M - 1.0/M)

            s_consistency = lambda1 / (s_consistency**2 + 1.0)
            s_optimal = lambda1 / (s_optimal**2 + 1.0)

            S[i-1, j-1] = phi * s_consistency + (1.0 - phi) * s_optimal

            if (i == 1 or j == 1) and not(i == j):
                s = threshold_value * 1.5
                S[i-1, j-1] = lambda1 / (s**2 + 1.0)

    XX = torch.sum(X * X, dim=1, keepdims=True)
    YY = torch.sum(Y * Y, dim=1, keepdims=True).transpose(0, 1)
    D = XX + YY - 2.0 * torch.matmul(X, Y.transpose(0, 1))

    bin1 = torch.full((1, M - 1), zeta, device=device)
    bin2 = torch.full((N, 1), zeta, device=device)
    D = torch.cat([bin1, D], 0)
    D = torch.cat([bin2, D], 1)

    K = P * torch.exp((S - D) / lambda2)
    K = torch.clamp(K, 1e-15, 1.0e20)

    a = torch.full([N, 1], 1.0 / N, device=device)
    b = torch.full([M, 1], 1.0 / M, device=device)

    ainvK = K / a
    u = torch.full([N, 1], 1.0 / N, device=device)
    for _ in range(maxIter):
        Ktu = torch.matmul(K.T, u)
        aKtu = torch.matmul(ainvK, b / Ktu)
        u = 1.0 / aKtu

    new_Ktu = torch.matmul(K.T, u)
    v = b / new_Ktu

    U = K * D
    dis = torch.sum(u * torch.matmul(U, v))
    dis = dis/(N * M)

    return dis,U

# This is the proposed loss function used in OPEL
def otuprel_exp_loss(X, Y, maxIter=20, lambda1=1.0, lambda2=0.1, virtual_distance=5.0, zeta=0.5, b_laplace=2.0, epoch=None, device='cuda'):
  
    N, _ = X.shape
    M, _ = Y.shape

    N = torch.tensor(N).cuda()
    M = torch.tensor(M).cuda()
    # Adjust lambda1 and lambda2 according to the formula
    lambda1 = lambda1 * (N + M)
    lambda2 = lambda2 * (N * M) / 4.0

    D_x_y = torch.mean((X.unsqueeze(1) - Y.unsqueeze(0))**2, dim=2)
    min_index = torch.argmin(D_x_y, dim=1).float()

    N += 1
    M += 1

    power = int(torch.sqrt(100*torch.tensor(epoch) + 1.0))
        
    phi = 0.999 ** power
    phi = min(phi, 0.999)
    phi = max(phi, 0.001)

    P = torch.zeros((N, M), device=device)
    S = torch.zeros((N, M), device=device)

    mid_para = 1.0 / (N**2) + 1 / (M**2)
    mid_para = math.sqrt(mid_para)
    threshold_value = 2.0 * virtual_distance / (N + M)

    for i in range(1, N+1):
        for j in range(1, M+1):
            # the distance to diagonal
            d_prior = torch.abs(i/N - j/M)
            d_prior = d_prior / mid_para
            # the distance to the most similar matching for a given i, adding extra 1 for virtual frame
            if i > 1:
                d_similarity = torch.abs(j/M - (min_index[i-2]+1)/M)
            else:
                d_similarity = torch.abs(j/M - 1.0/M)
            d_similarity = d_similarity / mid_para
            p_consistency = torch.exp(-d_prior / b_laplace) / (2.0 * b_laplace)
            p_optimal = torch.exp(-d_similarity / b_laplace) / (2.0 * b_laplace)
            P[i-1, j-1] = phi * p_consistency + (1.0 - phi) * p_optimal
            # virtual frame prior value
            if (i == 1 or j == 1) and not(i == j):
                d = threshold_value * 1.5 / mid_para
                P[i-1, j-1] = torch.exp(-d / b_laplace) / (2.0 * b_laplace)

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            s_consistency = torch.abs(i/N - j/M)

            if i > 1:
                s_optimal = torch.abs(j/M - (min_index[i-2] + 1)/M)
            else:
                s_optimal = torch.abs(j/M - 1.0/M)

            s_consistency = lambda1 / (s_consistency**2 + 1.0)
            s_optimal = lambda1 / (s_optimal**2 + 1.0)

            S[i-1, j-1] = phi * s_consistency + (1.0 - phi) * s_optimal

            if (i == 1 or j == 1) and not(i == j):
                s = threshold_value * 1.5
                S[i-1, j-1] = lambda1 / (s**2 + 1.0)

    XX = torch.sum(X * X, dim=1, keepdims=True)
    YY = torch.sum(Y * Y, dim=1, keepdims=True).transpose(0, 1)
    D = XX + YY - 2.0 * torch.matmul(X, Y.transpose(0, 1))

    bin1 = torch.full((1, M - 1), zeta, device=device)
    bin2 = torch.full((N, 1), zeta, device=device)
    D = torch.cat([bin1, D], 0)
    D = torch.cat([bin2, D], 1)

    K = P * torch.exp((S - D) / lambda2)
    K = torch.clamp(K, 1e-15, 1.0e20)

    a = torch.full([N, 1], 1.0 / N, device=device)
    b = torch.full([M, 1], 1.0 / M, device=device)

    ainvK = K / a
    u = torch.full([N, 1], 1.0 / N, device=device)
    for _ in range(maxIter):
        Ktu = torch.matmul(K.T, u)
        aKtu = torch.matmul(ainvK, b / Ktu)
        u = 1.0 / aKtu

    new_Ktu = torch.matmul(K.T, u)
    v = b / new_Ktu

    U = K * D
    dis = torch.sum(u * torch.matmul(U, v))
    dis = dis/(N * M)

    return dis,U


def all_loss_otuprel_gauss(X, Y, lambda3=2.0, delta=15.0, epoch=None, temperature=0.5, norm_embeds=False, c=0.0001, sigma=2.0):

    N, _ = X.shape
    M, _ = Y.shape
    assert X.shape[1] == Y.shape[1], 'The dimensions of instances in the input sequences must be the same!'

    N = torch.tensor(N).cuda()
    M = torch.tensor(M).cuda()

    device = X.device  # Ensure computations are done on the same device as X and Y

    if norm_embeds:
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)

    # Efficiently compute weights for C(x) and C(y) using broadcasting and vectorized operations
    indices = torch.arange(N, device=device).unsqueeze(0)
    W_x_p = 1.0 / (torch.pow(indices - indices.T, 2) + 1.0)
    W_x_pb = (torch.pow(indices - indices.T, 2) + 1.0)
    # Create masks for positive and negative pairs
    y_x = (torch.abs(indices - indices.T) > delta).float()

    indices = torch.arange(M, device=device).unsqueeze(0)
    W_y_p = 1.0 / (torch.pow(indices - indices.T, 2) + 1.0)
    W_y_pb = (torch.pow(indices - indices.T, 2) + 1.0)

    y_y = (torch.abs(indices - indices.T) > delta).float()

    # Compute distance matrices
    D_x = torch.mean((X.unsqueeze(1) - X.unsqueeze(0)) ** 2, dim=2)
    D_y = torch.mean((Y.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=2)

    # Compute contrastive terms C_x and C_y
    C_x = torch.mean(y_x * W_x_pb * torch.clamp(lambda3 - D_x, min=0) + (1.0 - y_x) * W_x_p * D_x)
    C_y = torch.mean(y_y * W_y_pb * torch.clamp(lambda3 - D_y, min=0) + (1.0 - y_y) * W_y_p * D_y)

   
    otuprel_dis, U = otuprel_gauss_loss(X, Y, sigma=sigma, epoch=epoch)
    U = U[1:, 1:]  # Adjust U as necessary

    # Compute best and worst matches
    X_best, X_worst = torch.argmax(U, dim=1), torch.argmin(U, dim=1)
    Y_best, Y_worst = torch.argmax(U, dim=0), torch.argmin(U, dim=0)

    # Compute distances for inter-sequence contrastive loss
    best_distance = torch.mean((X - Y[X_best]) ** 2 + (Y - X[Y_best]) ** 2) / temperature
    worst_distance = torch.mean((X - Y[X_worst]) ** 2 + (Y - X[Y_worst]) ** 2) / temperature

    # Compute inter-sequence loss
    loss_inter = F.cross_entropy(torch.stack([best_distance, worst_distance]), torch.tensor([0, 1], device=device).float())

    # Compute overall loss
    overall = 0.5 * (C_x + C_y) + otuprel_dis / (N * M) + c * loss_inter
    # overall = 0.5 * (C_x + C_y) + otuprel_dis / (N * M) 
    # Things to tune:
    # W_pb with and without
    # otuprel_dis / (N * M) or otuprel_dis in overall loss
    # loss_inter coefficient: 0.5, 0.01, 0.0001
    # phi vary from [1->0.5] or [1->0]
    
    return overall


def all_loss_otuprel_exp(X, Y, lambda3=2.0, delta=15.0, epoch=None, temperature=0.5, norm_embeds=False, c=0.0001, b_laplace=2.0):

    N, _ = X.shape
    M, _ = Y.shape
    assert X.shape[1] == Y.shape[1], 'The dimensions of instances in the input sequences must be the same!'

    N = torch.tensor(N).cuda()
    M = torch.tensor(M).cuda()

    device = X.device  # Ensure computations are done on the same device as X and Y

    if norm_embeds:
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)

    # Efficiently compute weights for C(x) and C(y) using broadcasting and vectorized operations
    indices = torch.arange(N, device=device).unsqueeze(0)
    W_x_p = 1.0 / (torch.pow(indices - indices.T, 2) + 1.0)
    W_x_pb = (torch.pow(indices - indices.T, 2) + 1.0)
    # Create masks for positive and negative pairs
    y_x = (torch.abs(indices - indices.T) > delta).float()

    indices = torch.arange(M, device=device).unsqueeze(0)
    W_y_p = 1.0 / (torch.pow(indices - indices.T, 2) + 1.0)
    W_y_pb = (torch.pow(indices - indices.T, 2) + 1.0)

    y_y = (torch.abs(indices - indices.T) > delta).float()

    # Compute distance matrices
    D_x = torch.mean((X.unsqueeze(1) - X.unsqueeze(0)) ** 2, dim=2)
    D_y = torch.mean((Y.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=2)

    # Compute contrastive terms C_x and C_y
    C_x = torch.mean(y_x * W_x_pb * torch.clamp(lambda3 - D_x, min=0) + (1.0 - y_x) * W_x_p * D_x)
    C_y = torch.mean(y_y * W_y_pb * torch.clamp(lambda3 - D_y, min=0) + (1.0 - y_y) * W_y_p * D_y)

    # Assuming otuprel_loss is optimized for PyTorch and returns a loss compatible with CUDA tensors
    otuprel_dis, U = otuprel_exp_loss(X, Y, b_laplace=b_laplace, epoch=epoch)
    U = U[1:, 1:]  # Adjust U as necessary

    # Compute best and worst matches
    X_best, X_worst = torch.argmax(U, dim=1), torch.argmin(U, dim=1)
    Y_best, Y_worst = torch.argmax(U, dim=0), torch.argmin(U, dim=0)

    # Compute distances for inter-sequence contrastive loss
    best_distance = torch.mean((X - Y[X_best]) ** 2 + (Y - X[Y_best]) ** 2) / temperature
    worst_distance = torch.mean((X - Y[X_worst]) ** 2 + (Y - X[Y_worst]) ** 2) / temperature

    # Compute inter-sequence loss
    loss_inter = F.cross_entropy(torch.stack([best_distance, worst_distance]), torch.tensor([0, 1], device=device).float())

    # Compute overall loss
    overall = 0.5 * (C_x + C_y) + otuprel_dis / (N * M) + c * loss_inter
    
    return overall
