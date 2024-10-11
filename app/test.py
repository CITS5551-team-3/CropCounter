import csv
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np

from params import Params
from count import im_count

def load_counts(filename = './images/counts.csv', *, delimiter = '\t') -> dict[str, list[Optional[int]]]:
    # sources in all lists are in the same order
    image_counts: dict[str, list[Optional[int]]] = {}
    
    with open(filename, newline='') as file:
        csv_reader = csv.reader(file, delimiter=delimiter)

        for row in csv_reader:
            image_id, *count_strs = row
            counts: list[Optional[int]] = [int(count_str) if count_str else None for count_str in count_strs]

            image_counts[image_id] = counts
        
    return image_counts

def main():
    manual_counts = load_counts()

    # could filter by those that have 2 or more counts

    # start ticks at 1 because of box plot
    x_range = np.array(list(range(len(manual_counts)))) + 1
    x_ticks = list(manual_counts.keys())

    manual_count_values: list[list[Optional[int]]] = list(manual_counts.values())


    # plot the computed points

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

    plt.scatter(x_range, np.array(computed_counts), marker=r'$\times$', label='Computed Counts')


    # plot each point

    n_sources = len(manual_count_values[0])
    source_x_values: list[list[int]] = [[] for _ in range(n_sources)]
    source_y_values: list[list[int]] = [[] for _ in range(n_sources)]

    for image_index, image_counts in enumerate(manual_counts.values()):
        for source_index, count in enumerate(image_counts):
            if count is None: continue

            source_x_values[source_index].append(image_index)
            source_y_values[source_index].append(count)
    
    for x_values, y_values in zip(source_x_values, source_y_values):
        # start x at 1 because of box plot
        plt.scatter(np.array(x_values) + 1, np.array(y_values))

    
    # box plot

    boxplot_values = [[value for value in values if value] + [computed_value] for values, computed_value in zip(manual_count_values, computed_counts)]
    plt.boxplot(boxplot_values)
    

    plt.xticks(x_range, x_ticks)
    plt.xticks(rotation=90)

    # plt.yticks(range(0, 600, 20)) # TODO don't hard-code
    plt.grid(axis='y', which='major')
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)

    plt.legend(loc='upper right')
    
    plt.show()

if __name__ == '__main__':
    main()
