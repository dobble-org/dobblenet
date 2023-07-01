import csv
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt 
from tap import Tap


class PlotCLI(Tap):
    yolo_results_file: Path
    output: Path


@dataclass
class Results:
    box_loss: float
    obj_loss: float
    cls_loss: float


@dataclass
class Metrics:
    precision: float
    recall: float
    map: float


@dataclass
class EpochResult:
    number: int
    train: Results
    val: Results
    metrics: Metrics


def read_results(results_file_path: Path):
    epochs_results = []
    with results_file_path.open() as results_file:
        results = csv.reader(results_file, delimiter=',', quotechar='|')
        next(results) # read header

        for result in results:
            epoch_naumber = int(result[0])
            train_results = Results(
                box_loss=float(result[1]),
                obj_loss=float(result[2]),
                cls_loss=float(result[3]),
            )
            metrics_results = Metrics(
                precision=float(result[4]),
                recall=float(result[5]),
                map=float(result[7]),
            )
            val_results = Results(
                box_loss=float(result[8]),
                obj_loss=float(result[9]),
                cls_loss=float(result[10]),
            )
            epochs_results.append(
                EpochResult(
                    number= epoch_naumber,
                    train = train_results,
                    metrics=metrics_results,
                    val=val_results
                )
            )
    return epochs_results

def get_epoch_numbers_as_list(epochs_results: list[EpochResult]) -> list[int]:
    return [epoch.number for epoch in epochs_results]

def get_map_as_list(epochs_results: list[EpochResult]) -> list[float]:
    return [epoch.metrics.map for epoch in epochs_results]

def plot_map(epochs_results: list[EpochResult], output: Path):
    epoch_numbers = get_epoch_numbers_as_list(epochs_results)
    maps = get_map_as_list(epochs_results)
    ax = plt.gca()
    ax.plot(epoch_numbers, maps)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP 0.5:0.95')
    plt.savefig(output, dpi=800)


def main(results_file_path: Path, output: Path):
    epoch_results = read_results(results_file_path)
    plot_map(epoch_results, output)

if __name__ == '__main__':
    ARGS = PlotCLI(underscores_to_dashes=True).parse_args()
    main(ARGS.yolo_results_file, ARGS.output)