import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from geopy.distance import geodesic
from ..trainer import predict


def evaluate(config,
             model,
             model_path,
             reference_dataloader,
             query_dataloader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    print("\nExtract Features:")
    reference_features, reference_labels, reference_locs = predict(config, model, reference_dataloader, model_path)
    query_features, query_labels, query_locs = predict(config, model, query_dataloader, model_path)

    print("Compute Scores:")
    r1 = calculate_scores(query_features, reference_features, query_labels, reference_labels,
                          query_locs, reference_locs, step_size=step_size, ranks=ranks)

    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()

    return r1


def calc_sim(config,
             model,
             model_path,
             reference_dataloader,
             query_dataloader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    print("\nExtract Features:")
    reference_features, reference_labels, _ = predict(config, model, reference_dataloader, model_path)
    query_features, query_labels, _ = predict(config, model, query_dataloader, model_path)

    print("Compute Scores Train:")
    r1 = calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=step_size,
                                ranks=ranks)

    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)

    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()

    return r1, near_dict


def calculate_scores(query_features, reference_features, query_labels, reference_labels, query_locs, reference_locs,
                     step_size=1000,
                     ref_step_size=5000, ranks=[1, 5, 10]):
    topk = ranks.copy()
    Q = len(query_features)
    R = len(reference_features)

    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()

    ref2index = {idx: i for i, idx in enumerate(reference_labels_np)}

    topk.append(R // 100)

    results = np.zeros([len(topk)])

    hit_rate = 0.0

    R1_100m = 0.0  # Retrieval success ratio

    # Iterate over query in steps with progress bar
    with tqdm(total=Q) as pbar:
        for q_start in range(0, Q, step_size):
            q_end = min(q_start + step_size, Q)
            query_batch = query_features[q_start:q_end]

            similarity = []

            # Iterate over reference in steps
            for r_start in range(0, R, ref_step_size):
                r_end = min(r_start + ref_step_size, R)
                reference_batch = reference_features[r_start:r_end]

                # Compute similarity for the current batch
                sim_tmp = query_batch @ reference_batch.T
                similarity.append(sim_tmp.cpu())

            # Concatenate similarity results for the current query batch
            similarity = torch.cat(similarity, dim=1)

            for i in range(q_start, q_end):
                # similarity value of gt reference
                gt_sim = similarity[i - q_start, ref2index[query_labels_np[i][0]]]

                # number of references with higher similarity as gt
                higher_sim = similarity[i - q_start, :] > gt_sim
                ranking = higher_sim.sum().item()

                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.


                # # mask for semi pos
                # mask = torch.ones(R)
                # for near_pos in query_labels_np[i][1:]:
                #     mask[ref2index[near_pos]] = 0

                # # calculate hit rate
                # hit = (higher_sim * mask).sum().item()
                # if hit < 1:
                #     hit_rate += 1.0

                # calculate hit rate (corrected version)
                top1_id = torch.argmax(similarity[i - q_start, :])
                positive_ids = []
                for near_pos in query_labels_np[i][:]:
                    positive_ids.append(ref2index[near_pos])

                if top1_id in positive_ids:
                    hit_rate += 1.0


                # calculate R@1-100m
                recall_1_id = torch.argmax(similarity[i - q_start, :])
                query_loc, reference_loc = query_locs[i], reference_locs[recall_1_id]
                distance = geodesic((query_loc[0], query_loc[1]), (reference_loc[0], reference_loc[1])).kilometers
                if distance <= 0.1: # 100m
                    R1_100m += 1.0

            pbar.update(q_end - q_start)

    # wait to close pbar
    time.sleep(0.1)

    results = results / Q * 100.
    hit_rate = hit_rate / Q * 100.
    R1_100m = R1_100m / Q * 100.

    # Output results
    string = []
    for i in range(len(topk) - 1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))

    string.append('Recall@top1: {:.4f}'.format(results[-1]))

    string.append('Hit_Rate: {:.4f}'.format(hit_rate))

    string.append('R@1_100m: {:.4f}'.format(R1_100m))

    print(' - '.join(string))

    return results[0]


def calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=1000,
                           ranks=[1, 5, 10]):
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)

    steps = Q // step_size + 1

    query_labels_np = query_labels[:, 0].cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()

    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i

    similarity = []

    for i in range(steps):
        start = step_size * i

        end = start + step_size

        sim_tmp = query_features[start:end] @ reference_features.T

        similarity.append(sim_tmp.cpu())

    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk.append(R // 100)

    results = np.zeros([len(topk)])

    bar = tqdm(range(Q))

    for i in bar:

        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]

        # number of references with higher similiarity as gt
        higher_sim = similarity[i, :] > gt_sim

        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.

    results = results / Q * 100.

    bar.close()

    # wait to close pbar
    time.sleep(0.1)

    string = []
    for i in range(len(topk) - 1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))

    string.append('Recall@top1: {:.4f}'.format(results[-1]))

    print(' - '.join(string))

    return results[0]


def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64,
                      step_size=1000):
    query_labels = query_labels[:, 0]

    Q = len(query_features)

    steps = Q // step_size + 1

    similarity = []

    for i in range(steps):
        start = step_size * i

        end = start + step_size

        sim_tmp = query_features[start:end] @ reference_features.T

        similarity.append(sim_tmp.cpu())

    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    # there might be more ground views for same sat view
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range + 2, dim=1)

    topk_references = []

    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i, :]])

    topk_references = torch.stack(topk_references, dim=0)

    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)

    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()

    for i in range(len(topk_references)):
        nearest = topk_references[i][mask[i]][:neighbour_range]

        nearest_dict[query_labels[i].item()] = list(nearest)

    return nearest_dict
