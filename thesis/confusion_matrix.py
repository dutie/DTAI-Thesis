import numpy as np
import matplotlib.pyplot as plt

class CM:
    def __init__(self, time_stamps):
        self.time_stamps = time_stamps

    def query_time_stamps(self, subjectId, exerciseId):
        # subjectId: int, 1-5
        # exerciseId: int, 1-8
        res = list(filter(lambda exercise: exercise['id'] == f"s{subjectId}:e{exerciseId}", self.time_stamps))
        return res

def data_cleaning_annotated(subject, exercise, cm, sortBy=0):
    annotated = cm.query_time_stamps(subject, exercise)[0]
    cleaned   = list(map(lambda vals: list(map(lambda val: int(val), vals)), annotated['data']))
    cleaned = sorted(cleaned, key=lambda d: d[sortBy], reverse=False)
    return cleaned
def data_cleaning_matches(results, sortBy: int = 2):
    """
    results: List[List]: [[exerciseTypeId: int, timeSeries: TimeSeries, start: int, end: int]]
    sortBy: int: 2 = start, 3 = end
    """
    sorted_res = sorted(results, key=lambda d: d[sortBy], reverse=False)
    return sorted_res

def calc_ratio(gt, mt):
    gt_start, gt_end, _ = gt
    mt_start, mt_end, _ = mt
#     print(f"Start (mt: {mt_start}) (gt: {gt_start}))")
#     print(f"End (mt: {mt_end}) (gt: {gt_end}))")
    overlap = max(0, (min(gt_end, mt_end)- max(gt_start, mt_start)))
    total   = abs(min(gt_start, mt_start) - max(gt_end, mt_end))
#     print("Overlap: ", overlap)
#     print("Total: ", total)

    return overlap / total

def calc_match(gt, mts, alpha):
    max_ratio = -1
    max_mt = None
    for mt in mts:
        ratio = calc_ratio(gt, mt)
        if ratio > max_ratio:
            max_ratio = ratio
            max_mt = mt
    if max_mt and max_ratio > alpha:
            return (gt, max_mt, max_ratio)

    return None

def calc_matches(gts, mts, alpha, results, accum, number_of_templates=0):
    if gts:
        gt = gts.pop(0)
        temp_result = calc_match(gt, mts, alpha)
        if not temp_result:
            accum.append(gt)
        else:
            mistake = False
            for (cnt, (_, _max_mt, _max_ratio)) in enumerate(results):
                if _max_mt[0] == temp_result[1][0] and _max_mt[1] == temp_result[1][1]:
                    mistake = True
                    if _max_ratio < temp_result[2]:
                        results[cnt] = temp_result
                        accum.append(results[cnt][1])
            if not mistake:
                results.append(temp_result)
        return calc_matches(gts, mts, alpha, results, accum)
    else:
        gts.extend(accum)
        return results

def find_fd(matched, matches, cm):
    for mt in matches:
        if (mt[0],mt[1]) not in [(match[0], match[1]) for (_, match, _) in matched]:
            cm[-1][mt[2]] += 1

def clean(detections, subjectId, exerciseId, cm):
    if len(detections) > 0 and len(detections[0]) == 5:
        dts = [[dts, dte, dtt] for dtt, _, dts, dte, _ in detections]
    else:
        dts = [[dts, dte, dtt] for dtt, _, dts, dte in detections]
    gts = data_cleaning_annotated(subjectId, exerciseId, cm)
    dts = data_cleaning_matches(dts, 0)
    return gts, dts

def evaluate_algorithm(ground_truth, matches, alpha=0.5):
    confusion_matrix = np.zeros((4, 4))

    # Step 1: Matches
    matched = calc_matches(ground_truth, matches, alpha, [], [])
    # Step 2: False Detections
    false_detections = find_fd(matched, matches, confusion_matrix)
    # Step 3: Missed Detections
    for gt in ground_truth:
        confusion_matrix[gt[2]][-1] += 1   

    for m in matched:
        confusion_matrix[m[0][2]][m[1][2]] += 1
    return confusion_matrix

def evaluate_large_algorithm(ground_truth, matches, alpha=0.5, number_of_templates=3):
    confusion_matrix = np.zeros((number_of_templates+1, number_of_templates+1))
    # Step 1: Matches
    matched = calc_matches(ground_truth, matches, alpha, [], [], number_of_templates)
    # Step 2: False Detections
    false_detections = find_fd(matched, matches, confusion_matrix)
    # Step 3: Missed Detections
    for gt in ground_truth:
        confusion_matrix[gt[2]][-1] += 1   

    for m in matched:
        confusion_matrix[m[0][2]][m[1][2]] += 1
    return confusion_matrix

def get_results(detections, min_ratio, subjectId, exerciseId, cm):
    clean_gts, clean_detections = clean(detections, subjectId, exerciseId, cm)
    confusion_matrix = evaluate_algorithm(clean_gts, clean_detections, min_ratio)
    return confusion_matrix

def get_results_large(detections, min_ratio, subjectId, exerciseId, cm, number_of_templates=3):
    clean_gts, clean_detections = clean(detections, subjectId, exerciseId, cm)
    confusion_matrix = evaluate_large_algorithm(clean_gts, clean_detections, min_ratio, number_of_templates)
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, title, fig=[], ax=[], isReturned=False, max_bar=None):
    if ax == []:
        fig, ax = plt.subplots(figsize=(18,16))
    fig.suptitle(title)
    # Plot the heatmap
    im = ax.imshow(confusion_matrix, cmap='Reds')

    # Add the colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    if max_bar:
        im.set_clim(0.0, max_bar)

    xticklabels = list(map(str, np.arange(confusion_matrix.shape[1]-1)))
    xticklabels.append('Missed')
    yticklabels = list(map(str, np.arange(confusion_matrix.shape[0]-1)))
    yticklabels.append('False')

    # Set the tick labels and positions
    ax.set_xticks(np.arange(confusion_matrix.shape[1]))
    ax.set_yticks(np.arange(confusion_matrix.shape[0]))
    ax.set_xticklabels(xticklabels, fontsize=12)
    ax.set_yticklabels(yticklabels, fontsize=12)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="black", fontsize=14)

    # Add the axis labels and the title
    ax.set_xlabel('Guessed Exercise Type', fontsize=14)
    ax.set_ylabel('Actual Exercise Type', fontsize=14)
    ax.set_title(title, fontsize=16)

    if isReturned:
        return fig, ax
    else:
        # Show the plot
        plt.show()   

def generate_full_cumulative_overview(list_subjects_cms: list[list[np.ndarray]]):
    " [subject[exercise result], subject[..]..] "
    size = np.shape(list_subjects_cms[0][0])
    cms = np.zeros(size)
    for subject in list_subjects_cms:
        for result in subject:
            cms += result
    # plot_confusion_matrix(cms, "All Accumulated")
    return cms

def generate_cumulative_overview(list_subjects_cms: list[list[np.ndarray]], title='All Subjects Accumulated'):
    cms = generate_overview(list_subjects_cms[0])
    # plot_confusion_matrix(cms, 'Subject 1')
    for ids, su in enumerate(list_subjects_cms[1:]):
        # plot_confusion_matrix(generate_overview(su), f'Subject {ids + 2}')
        cms += generate_overview(su)
    # plot_confusion_matrix(cms, title)
    return cms

def generate_overview(list_cms: list[np.ndarray]):
    size = len(list_cms)
    cms = np.zeros((size*3+1, size*3+1))
    current = 0
    for conf in list_cms:
        current_matrix = current * 3
        # Detections
        if len(list_cms[current]) > 0:
            cms[current_matrix:current_matrix+3, current_matrix: current_matrix+3] = list_cms[current][0:3, 0:3]
            # Missed Detections
            cms[current_matrix:current_matrix+3, -1] = list_cms[current][0:3, -1]
            # False Detections
            cms[-1, current_matrix: current_matrix+3] = list_cms[current][-1, 0:3]
        else:
            # Missed Detections
            cms[current_matrix:current_matrix+3, -1] = np.ones(3)*10

        current += 1
    return cms

def impose_detection_template_on_result(time_series, templates, annotated, time_series_range=[0,500]):
    x = np.arange(time_series_range[0], time_series_range[1])
    plt.plot(x, time_series[time_series_range[0]:time_series_range[1]])
    for idx in range(len(templates)):
        temp_x = np.arange(annotated[idx][0], annotated[idx][1])
        plt.plot(templates[idx], 'r+')
    plt.show()

def plot_per_subject(data):
    fig, axs = plt.subplots(8, 5, figsize=(32, 8))
    for ids, subject in enumerate(data):
        for ide, exercise in enumerate(subject):
            sensor = exercise[0]
            axs[ide, ids].plot(sensor[1])
    plt.tight_layout()
    plt.show()
    
def plot_per_exercise(data):
    fig, axs = plt.subplots(5, 8, figsize=(32, 8))
    for ids, subject in enumerate(data):
        for ide, exercise in enumerate(subject):
            sensor = exercise[0]
            axs[ids, ide].plot(sensor[1])
    plt.tight_layout()
    plt.show()
    
def set_size(width='thesis', fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    width_pt = 455.24408

    # Width of figure (in pts)
    # fig_width_pt = width_pt * fraction
    fig_width_pt = width_pt * 1
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)