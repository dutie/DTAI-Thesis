import numpy as np
from numba import njit
from numba.typed import List
from subsequence_dtw import compute_cost_matrix, subsequence_dtw

import oit

from tqdm import tqdm

magnitude_type = list[float]
test_type      = list[list[float]]
templates_type = list[list[list[float]]]
sensor_type    = list[templates_type, test_type]
exercise_type  = list[sensor_type]
subject_type   = list[exercise_type]

def update_progress(pbar, start, end, beta, multiplier=1):
    nstart = round(((1-beta)*start + beta*end))
    nend   = round((beta*start + (1-beta)*end))
    removed = (nend - nstart) * multiplier
    pbar.update(removed)

def calc_non_inf_vector_magnitude(tests, tempsLen):
    for idx, test in enumerate(tests):
        idxInf = []
        
        for idxs, vector in enumerate(test):
            if np.inf == vector:
                idxInf.append(idxs)
        if len(idxInf) == 0:
            return True
        prev = idxInf[0]
        start = prev
        consec = []
        for idInf in idxInf[1:]:
            if idInf != prev+1:
                consec.append([start, prev])
                start = idInf
            prev = idInf
        consec.append([start, prev])
        gaps = []
        prev = 0
        for con in consec:
            gaps.append(con[0] - prev)
            prev = con[1]
        gaps.append(len(test) - prev)
        if max(gaps) > tempsLen[idx] * 0.9:
            return True
    return False

@njit
def get_inf_ids(test):
    inf_ids = []
    for inf_id, vector in enumerate(test):
        if np.inf in vector:
            inf_ids.append(inf_id)
    return inf_ids

def get_range_of_non_infs(inf_ids, test_len):
    non_inf_gaps = []
    start_gap = -1
    end_gap = -1
    for time_step in range(test_len):
        if time_step not in inf_ids:
            if start_gap < 0:
                start_gap = time_step
        else:
            if start_gap >= 0:
                end_gap = time_step
                non_inf_gaps.append([start_gap, end_gap])
                start_gap, end_gap = -1, -1
    if test_len - inf_ids[-1] > 1:
        non_inf_gaps.append([inf_ids[-1]+1, test_len])
    return non_inf_gaps

def calc_non_inf_vector_general(tests, tempsLen):
    for idx, test in enumerate(tests):
        idxInf = get_inf_ids(test)
        if len(idxInf) == 0:
            return True
        gaps = get_range_of_non_infs(idxInf, len(test))
        gap_sizes = list(map(lambda x: x[1] - x[0], gaps))
        if len(gap_sizes) > 0:
            max_gap_size = np.max(gap_sizes)
            if max_gap_size >= tempsLen[idx] * 0.9:
                return True
    return False


def sufficiently_many(tests, tempsLen: list[int]):
    if np.ndim(tests[0]) > 1:
        return calc_non_inf_vector_general(tests, tempsLen)
    else:
        return calc_non_inf_vector_magnitude(tests, tempsLen)

def match_long_enough(start: int, end: int, alpha: float, tempLen: int):
    return (end - start) >= tempLen * alpha

def set_to_inf(start: int, end: int, test_, beta: float):
    # EDITED:
    test = np.array(test_, dtype=np.float64)
    ###
    nstart = round(((1-beta)*start + beta*end))
    nend   = round((beta*start + (1-beta)*end))
    if np.ndim(test) > 1:
        test[nstart: nend, :] = np.inf
    else:
        test[nstart: nend] = np.inf
    return test

def set_all_to_inf(start, end, tests, beta):
    return [set_to_inf(start, end, test, beta) for test in tests]

def mtmmdtw_step(tests, templates, tempsLen: list[int]):
    zipped   = zip(templates, tests)
    costMatrices    = [compute_cost_matrix(template, test) for template, test in zipped]
    sdtw            = [subsequence_dtw(costMatrix) for costMatrix in costMatrices]
    normalizedCosts = np.divide([acc_cost for acc_cost, _ , _ in sdtw], tempsLen)
    
    bestFitId       = np.argmin(normalizedCosts)
    start, end      = sdtw[bestFitId][1], sdtw[bestFitId][2]
    norm_acc_cost   = normalizedCosts[bestFitId]
    
    return norm_acc_cost, start, end, bestFitId

def mtmmdtw(save_at, test, templates, alpha: float = 0.5, beta: float = 0.02, test_is_list = False):
    if not test_is_list:
        tests = [test.copy() for _ in range(3)]
    else:
        tests = test.copy()
        
    tempsLen = [len(template) for template in templates]
    
    results = []
    pbar_total = len(test)*3 + 150
    pbar = tqdm(total=pbar_total)
    while sufficiently_many(tests, tempsLen):
        cost, start, end, detectionId = mtmmdtw_step(tests, templates, tempsLen)
        
        if match_long_enough(start, end, alpha, tempsLen[detectionId]):
            update_progress(pbar, start, end, beta, 3)
            results.append((detectionId, tests[detectionId].copy(), start, end, cost))
            tests = set_all_to_inf(start, end, tests, beta)
        else:
            update_progress(pbar,start, end, beta)
            tests[detectionId] = set_to_inf(start, end, tests[detectionId].copy(), beta)
    pbar.close()
    with open(save_at, 'wb') as f:
        np.save(f, results)
    return results

def mtmmdtw_all_templates(save_at, test, templates, alpha: float = 0.5, beta: float = 0.02, test_is_list = False):
    if not test_is_list:
        tests = [test.copy() for _ in range(len(templates))]
    else:
        tests = test.copy()
    assert len(tests) == len(templates)
    
    tempsLen = [len(template) for template in templates]
    
    results = []
    pbar_total = len(test)*(len(templates)) + 150
    pbar = tqdm(total=pbar_total)
    while sufficiently_many(tests, tempsLen):
        cost, start, end, detectionId = mtmmdtw_step(tests, templates, tempsLen)
        
        if match_long_enough(start, end, alpha, tempsLen[detectionId]):
            update_progress(pbar, start, end, beta, 3)
            results.append((detectionId, tests[detectionId].copy(), start, end, cost))
            tests = set_all_to_inf(start, end, tests, beta)
        else:
            update_progress(pbar,start, end, beta)
            tests[detectionId] = set_to_inf(start, end, tests[detectionId].copy(), beta)
    pbar.close()
    with open(save_at, 'wb') as f:
        np.save(f, results)
    return results
    


def novel_mtmmdtw_wsvd(save_at, exercise, alpha: float = 0.5, beta: float = 0.02, gamma: float = 1.0): 
    templates = []
    tests = []
    pbar_total = len(exercise[0][1])*3 + 150
    pbar = tqdm(total=pbar_total)
    for sensor in exercise:
        templates_, test_ = sensor
        tempsLen = [len(template) for template in templates_] 
        if len(templates) == 0:
            templates = [oit.svd(template.copy()) for template in templates_]
            tests     = [oit.wsvd(test_.copy(),int(tempLen*gamma)) for tempLen in tempsLen ]
        else:
            templates = [append_time_series(template, oit.svd(template_)) for template, template_ in zip(templates, templates_)]
            tests     = [append_time_series(test, oit.wsvd(test_.copy(), int(tempLen*gamma))) for test, tempLen in zip(tests, tempsLen)]
    
    results = []
    
    while sufficiently_many(tests, tempsLen):
        cost, start, end, detectionId = mtmmdtw_step(tests, templates, tempsLen)
        
        if match_long_enough(start, end, alpha, tempsLen[detectionId]):
            update_progress(pbar, start, end, beta, 3)
            results.append((detectionId, tests[detectionId].copy(), start, end, cost))
            tests = set_all_to_inf(start, end, tests, beta)
        else:
            update_progress(pbar, start, end, beta, 1)
            tests[detectionId] = set_to_inf(start, end, tests[detectionId].copy(), beta)

    with open(save_at, 'wb') as f:
        np.save(f, results)
    return results

def novel_mtmmdtw_abs(save_at, exercise, alpha: float = 0.5, beta: float = 0.02, gamma: float = 1.0):    
    
    templates = []
    tests = []
    pbar_total = len(exercise[0][1])*3 + 150
    pbar = tqdm(total=pbar_total)
    for sensor in exercise:
        templates_, test_ = sensor
        
        tempsLen = [len(template) for template in templates_] 
        if len(templates) == 0:
            templates = [oit.usvd_abs(template.copy()) for template in templates_]
            tests     = [oit.wusvd_absolute(test_.copy(),int(tempLen*gamma)) for tempLen in tempsLen ]
        else:
            templates = [append_time_series(template, oit.usvd_abs(template_.copy())) for template, template_ in zip(templates, templates_)]
            tests     = [append_time_series(test, oit.wusvd_absolute(test_.copy(), int(tempLen*gamma))) for test, tempLen in zip(tests, tempsLen)]
            
    results = []
    
    while sufficiently_many(tests, tempsLen):
        cost, start, end, detectionId = mtmmdtw_step(tests, templates, tempsLen)
        
        if match_long_enough(start, end, alpha, tempsLen[detectionId]):
            update_progress(pbar, start, end, beta, 3)
            results.append((detectionId, tests[detectionId].copy(), start, end, cost))
            tests = set_all_to_inf(start, end, tests, beta)
        else:
            update_progress(pbar, start, end, beta)
            tests[detectionId] = set_to_inf(start, end, tests[detectionId].copy(), beta)
            
    with open(save_at, 'wb') as f:
        np.save(f, results)
    return results

def novel_mtmmdtw_skew(save_at, exercise, alpha: float = 0.5, beta: float = 0.02, gamma: float = 1.0):    
    
    templates = []
    tests = []
    pbar_total = len(exercise[0][1])*3 + 150
    pbar = tqdm(total=pbar_total)
    for sensor in exercise:
        templates_, test_ = sensor
        
        tempsLen = [len(template) for template in templates_] 
        if len(templates) == 0:
            templates = [oit.usvd_skew(template.copy()) for template in templates_]
            tests     = [oit.wusvd_skew(test_.copy(),int(tempLen*gamma)) for tempLen in tempsLen ]
        else:
            templates = [append_time_series(template, oit.usvd_skew(template_.copy())) for template, template_ in zip(templates, templates_)]
            tests     = [append_time_series(test, oit.wusvd_skew(test_.copy(), int(tempLen*gamma))) for test, tempLen in zip(tests, tempsLen)]
            
    results = []
    
    while sufficiently_many(tests, tempsLen):
        cost, start, end, detectionId = mtmmdtw_step(tests, templates, tempsLen)
        
        if match_long_enough(start, end, alpha, tempsLen[detectionId]):
            update_progress(pbar, start, end, beta, 3)
            results.append((detectionId, tests[detectionId].copy(), start, end, cost))
            tests = set_all_to_inf(start, end, tests, beta)
        else:
            update_progress(pbar, start, end, beta)
            tests[detectionId] = set_to_inf(start, end, tests[detectionId].copy(), beta)
            
    with open(save_at, 'wb') as f:
        np.save(f, results)
    return results
def novel_mtmmdtw_mean(save_at, exercise, alpha: float = 0.5, beta: float = 0.02, gamma: float = 1.0):    
    templates = []
    tests = []
    pbar_total = len(exercise[0][1])*3 + 150
    pbar = tqdm(total=pbar_total)
    for sensor in exercise:
        templates_, test_ = sensor
        
        tempsLen = [len(template) for template in templates_] 
        if len(templates) == 0:
            templates = [oit.usvd_mean(template.copy()) for template in templates_]
            tests     = [oit.wusvd_mean(test_.copy(),int(tempLen*gamma)) for tempLen in tempsLen ]
        else:
            templates = [append_time_series(template, oit.usvd_mean(template_.copy())) for template, template_ in zip(templates, templates_)]
            tests     = [append_time_series(test, oit.wusvd_mean(test_.copy(), int(tempLen*gamma))) for test, tempLen in zip(tests, tempsLen)]
            
    results = []
    
    while sufficiently_many(tests, tempsLen):
        cost, start, end, detectionId = mtmmdtw_step(tests, templates, tempsLen)
        
        if match_long_enough(start, end, alpha, tempsLen[detectionId]):
            update_progress(pbar, start, end, beta, 3)
            results.append((detectionId, tests[detectionId].copy(), start, end, cost))
            tests = set_all_to_inf(start, end, tests, beta)
        else:
            update_progress(pbar, start, end, beta, 1)
            tests[detectionId] = set_to_inf(start, end, tests[detectionId].copy(), beta)
            
    with open(save_at, 'wb') as f:
        np.save(f, results)
    return results

