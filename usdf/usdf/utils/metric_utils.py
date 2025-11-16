import torch
from pytorch3d.loss import chamfer_distance

def minimal_matching_distance(A, B):
    """
    Compute the minimal matching distance between two sets of point clouds.
    :param A: (M_1, N, 3) tensor containing M_1 point clouds of size N (test shape set)
    :param B: (M_2, N, 3) tensor containing M_2 point clouds of size N (completion set)
    :return:
        mmd_value: <scalar> the minimal matching distance between A and B
    """
    M_1, M_2 = A.shape[0], B.shape[0]
    # compute for each point in
    # bthe chamfer

    chamfer_distances = []
    for i in range(M_2):
        B_i = B[i].unsqueeze(0).repeat(M_1, 1, 1)
        cd_i = chamfer_distance(A, B_i)[0]
        chamfer_distances.append(cd_i)
    chamfer_distances = torch.stack(chamfer_distances, dim=0)
    mmd_value = torch.min(chamfer_distances)
    return mmd_value


def total_matching_distance(B):
    """
    Compute the total matching distance between a set of point clouds.
    :param B: (M, N, 3) tensor containing M point clouds of size N (completion set)
    :return:
        tmd_value: <scalar> the total matching distance between B
    """
    M = B.shape[0]
    # compute for each point in
    # bthe chamfer

    chamfer_distances = []
    for i in range(M):
        B_i = B[i].unsqueeze(0).repeat(M-1, 1, 1) # (M-1, N, 3)
        # get the B subset without B_i
        B_subset = B[torch.arange(M) != i] # (M-1, N, 3)
        cd_i = chamfer_distance(B_subset, B_i, batch_reduction="mean")[0]
        chamfer_distances.append(cd_i)
    chamfer_distances = torch.stack(chamfer_distances, dim=0)
    tmd_value = torch.mean(chamfer_distances)
    return tmd_value


# TEST:
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    A = torch.randn(10, 100, 3).to(device)
    B = torch.randn(5, 100, 3).to(device)
    mmd_value = minimal_matching_distance(A, B)
    print(mmd_value)
    tmd_value = total_matching_distance(B)
    print(tmd_value)

    # test a set where they are all the same points
    A = torch.ones((10, 100, 3), device=device)
    B = torch.ones((5, 100, 3), device=device)
    mmd_value = minimal_matching_distance(A, B)
    print(mmd_value)
    tmd_value = total_matching_distance(B)
    print(tmd_value)