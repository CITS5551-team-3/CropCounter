import csv
from typing import Optional

from matplotlib import pyplot as plt
from scipy.stats import sem
import numpy as np

from params import Params
from count import im_count

def load_counts(filename = './images/counts.csv', *, delimiter = '\t') -> tuple[dict[str, list[Optional[int]]], dict[str, str]]:
    # sources in all lists are in the same order
    image_counts: dict[str, list[Optional[int]]] = {}
    image_locs: dict[str, str] = {}
    
    with open(filename, newline='') as file:
        csv_reader = csv.reader(file, delimiter=delimiter)

        for row in csv_reader:
            image_loc, image_id, *count_strs = row
            counts: list[Optional[int]] = [int(count_str) if count_str else None for count_str in count_strs]

            image_counts[image_id] = counts
            image_locs[image_id] = image_loc
        
    return image_counts, image_locs

def main():
    show = False

    manual_counts, image_locs = load_counts()

    # could filter by those that have 2 or more counts

    # start ticks at 1 because of box plot
    image_name_x_range = np.array(list(range(len(manual_counts)))) + 1
    image_names = list(manual_counts.keys())

    manual_count_values: list[list[Optional[int]]] = list(manual_counts.values())


    # plot a comparison of the computed points and the manual counts

    params = Params()

    # read cached computed values
    cache_delimiter = ','
    cached_computed_counts: dict[str, int] = {}
    try:
        with open('./images/computed_counts.csv', mode='r', newline='') as file:
            print('cache found')
            csv_reader = csv.reader(file, delimiter=cache_delimiter)
            for row in csv_reader:
                image_id, count_str = row
                cached_computed_counts[image_id] = int(count_str)
            print(f'cache loaded ({len(cached_computed_counts)} entries)')
    except FileNotFoundError:
        print('cache not found, continuing...')
    
    print('')

    computed_counts: list[int] = []
    for i, image_name in enumerate(manual_counts.keys()):
        print(f'{i + 1} of {len(manual_counts)} ({image_name}): counting...')

        if image_name in cached_computed_counts:
            count = cached_computed_counts[image_name]
            computed_counts.append(count)
            print(f'{i + 1} of {len(manual_counts)} ({image_name}): loaded count from cache ({count})')
        else:
            filename = f'./images/{image_name}.JPG'
            count = im_count(params, filename)
            computed_counts.append(count)
            print(f'{i + 1} of {len(manual_counts)} ({image_name}): finished count ({count})')
        
        print('')
    
    # store results
    with open('./images/computed_counts.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file, delimiter=cache_delimiter)
        csv_writer.writerows(zip(manual_counts.keys(), computed_counts))

    plt.scatter(image_name_x_range, np.array(computed_counts), marker=r'$\times$', s=128, label='Computed Counts')

    
    # box plot

    clean_manual_count_values = [[value for value in values if value] for values in manual_count_values]
    plt.boxplot(clean_manual_count_values)
    

    plt.xticks(image_name_x_range, image_names)
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.ylim(0, 700) # TODO don't hard-code
    plt.grid(axis='y', which='major')
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)

    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('./figures/computed_est_box_plot')
    if show: plt.show()
    else: plt.close()



    # plot the residuals for each image

    # estimate the true value for each image
    true_counts = [float(np.mean([value for value in values if value])) for values in manual_count_values]

    residuals = [computed_count - true_count for computed_count, true_count in zip(computed_counts, true_counts)]

    plt.xticks(image_name_x_range, image_names)
    plt.xticks(rotation=90)

    locations: list[str] = sorted(list(set(image_locs.values())))
    location_image_names: dict[str, list[str]] = {loc: [image_name for image_name in image_names if image_locs[image_name] == loc] for loc in locations}
    image_indices: dict[str, int] = {image_name: i for i, image_name in enumerate(image_names)}
    
    loc_residuals: dict[str, list[float]] = {}
    for loc in locations:
        loc_x_indices = [image_indices[image_name] for image_name in location_image_names[loc]]
        loc_residuals[loc] = [residuals[i] for i in loc_x_indices]

        plt.scatter([image_name_x_range[i] for i in loc_x_indices], loc_residuals[loc], label=loc)
    
    print(f'Avg error (abs residuals): {np.mean(np.abs(residuals))}')
    print()

    plt.ylim(-150, 150) # TODO don't hard-code
    plt.grid(axis='y', which='major')
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)
    
    plt.legend()

    plt.tight_layout()
    plt.savefig('./figures/raw_residuals')
    if show: plt.show()
    else: plt.close()



    # plot relative residuals for each location

    loc_x_range = [i + 1 for i in range(len(locations))]

    loc_rel_residuals: dict[str, list[float]] = {
        loc: [
            residuals[image_indices[image_name]] / true_counts[image_indices[image_name]] for image_name in loc_image_names
        ] for loc, loc_image_names in location_image_names.items()
    }
    # plt.boxplot([loc_rel_residuals[loc] for loc in locations])
    for i, loc in enumerate(locations):
        plt.scatter([loc_x_range[i]] * len(loc_rel_residuals[loc]), loc_rel_residuals[loc])

    print('Location avg relative residual:')
    for loc in locations:
        print(f'{loc}: {np.mean(loc_rel_residuals[loc])}')
    print()
    print('Location avg relative error (abs residuals):')
    for loc in locations:
        print(f'{loc}: {np.mean(np.abs(loc_rel_residuals[loc]))}')
    print()

    plt.xticks(loc_x_range, locations)

    plt.ylim(-1, 1) # TODO don't hard-code
    plt.grid(axis='y', which='major')
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)

    # plt.legend()

    plt.tight_layout()
    plt.savefig('./figures/loc_rel_residuals')
    if show: plt.show()
    else: plt.close()



    # plot std error for the manual counts of each image

    plt.xticks(image_name_x_range, image_names)
    plt.xticks(rotation=90)

    manual_count_std_errs = [sem(manual_counts) for manual_counts in clean_manual_count_values]
    plt.scatter(image_name_x_range, manual_count_std_errs)

    print(f'Avg std_err: {np.mean(manual_count_std_errs)}')
    print()

    plt.ylim(0, 120) # TODO don't hard-code
    plt.grid(axis='y', which='major')
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)

    plt.tight_layout()
    plt.savefig('./figures/manual_std_errs')
    if show: plt.show()
    else: plt.close()



    # plot relative std error for the manual counts of each image

    plt.xticks(image_name_x_range, image_names)
    plt.xticks(rotation=90)

    manual_count_rel_std_errs = [
        std_err / np.mean(manual_counts) for manual_counts, std_err in zip(clean_manual_count_values, manual_count_std_errs)
    ]
    plt.scatter(image_name_x_range, manual_count_rel_std_errs)

    print(f'Avg rel std_err: {np.mean(manual_count_rel_std_errs)}')
    print()

    plt.ylim(0, 0.5) # TODO don't hard-code
    plt.grid(axis='y', which='major')
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)

    plt.tight_layout()
    plt.savefig('./figures/manual_rel_std_errs')
    if show: plt.show()
    else: plt.close()



    # plot relative residuals vs err for each location

    bar_offset = 0.05
    loc_indices: dict[str, int] = {loc: i for i, loc in enumerate(locations)}
    plt.scatter([
        loc_x_range[loc_indices[image_locs[image_name]]] - bar_offset for image_name in image_names
    ], [
        abs(residuals[image_indices[image_name]] / true_counts[image_indices[image_name]]) for image_name in image_names
    ], label='Computed Count')

    plt.scatter([
        loc_x_range[loc_indices[image_locs[image_name]]] + bar_offset for image_name in image_names
    ], manual_count_rel_std_errs, label='Manual Count')

    plt.xticks(loc_x_range, locations)

    plt.ylabel('Relative Error')
    plt.ylim(0, 1) # TODO don't hard-code
    plt.grid(axis='y', which='major')
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)

    plt.legend()

    plt.tight_layout()
    plt.savefig('./figures/loc_rel_err_comparison')
    if show: plt.show()
    else: plt.close()


if __name__ == '__main__':
    main()
