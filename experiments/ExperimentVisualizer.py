import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from helper.LoggingHandler import LoggingHandler


class ExperimentVisualizer:
    @staticmethod
    def generate_summary(model_name: str,
                         utility_paper_locations: Dict[str, str], safety_paper_locations: Dict[str, str],
                         calculate_utility_degration_average: bool) -> None:
        # Define the keywords to keep
        keywords = ("TIMEMEASUREMENT", "SUMMARY")
        important_log_file_lines = []

        # Extract experiment start and end for complete duration
        start_time = None
        end_time = None

        # Read and filter relevant lines
        with open(LoggingHandler.get_log_file(), "r") as file:
            for line in file:
                content = line[28:] if len(line) > 28 else ""  # skip timestamp + colon + space
                if any(content.startswith(keyword) for keyword in keywords):
                    important_log_file_lines.append(line.rstrip('\r\n'))
                if start_time is None:
                    timestamp_str = line[:26]
                    try:
                        time_format = "%Y-%m-%d %H:%M:%S.%f"
                        start_time = datetime.strptime(timestamp_str, time_format)
                    except ValueError:
                        raise Exception("Log file in invalid format.")
                end_time = line[:26]
        end_time = datetime.strptime(end_time, time_format)
        experiment_duration = end_time - start_time

        # Enrich with durations
        important_log_file_lines = ExperimentVisualizer.__enrich_with_time_measurements(important_log_file_lines)
        summary_table = ExperimentVisualizer.__convert_to_summary_table(important_log_file_lines, model_name,
                                                                        utility_paper_locations,
                                                                        safety_paper_locations,
                                                                        calculate_utility_degration_average)

        # Print the summaries
        LoggingHandler.log_and_print_prepend_timestamps(
            f"------------------IMPORTANT LINES FROM LOG FILE------------------")
        for l in important_log_file_lines:
            LoggingHandler.log_and_print_prepend_timestamps(l)
        LoggingHandler.log_and_print_prepend_timestamps(f"------------------SUMMARY - RESULTS------------------")
        for l in summary_table:
            LoggingHandler.log_and_print_prepend_timestamps(l)
        runtime = 'unknown'
        for l in important_log_file_lines:
            if l.startswith("DURATION ATTACK: "):
                runtime = l.split(" ")[2]
        LoggingHandler.log_and_print_prepend_timestamps("\r\n")
        LoggingHandler.log_and_print_prepend_timestamps(
            f"The measured TwinBreak runtime is {runtime} seconds and relates to the ''Runtime'' value in Table 7.")
        LoggingHandler.log_and_print_prepend_timestamps(
            f"Hint: Note that the computed time here includes the calculation of activation differences of untargeted layers, writing the results to disk, and also logging. In the experiments for the paper, we got rid of these elements. Nevertheless, the results should approximately match the values in the paper, if executed with similar hardware. Otherwise, the runtime can vary.")
        LoggingHandler.log_and_print_prepend_timestamps(f"Complete experiment runtime: {experiment_duration}")

    @staticmethod
    def __enrich_with_time_measurements(log_lines: List[str]) -> List[str]:
        time_format = "%Y-%m-%d %H:%M:%S.%f"
        measurement_starts = {}
        enriched_lines = []

        for line in log_lines:
            enriched_lines.append(line)

            # Check if line contains "TIMEMEASUREMENT " after the timestamp
            content = line[28:] if len(line) > 28 else ""
            if content.startswith("TIMEMEASUREMENT "):
                # Example: "TIMEMEASUREMENT ATTACK: Attack the model..."
                try:
                    rest = content.split(" ", 1)[1]
                    measurement_id = rest.split(":", 1)[0].strip()
                except IndexError:
                    continue  # Malformed, skip

                timestamp_str = line[:26]
                try:
                    timestamp = datetime.strptime(timestamp_str, time_format)
                except ValueError:
                    continue  # Malformed timestamp, skip

                if measurement_id in measurement_starts:
                    # Found the second occurrence
                    start_time = measurement_starts.pop(measurement_id)
                    duration = (timestamp - start_time).total_seconds()
                    duration_line = f"DURATION {measurement_id}: {duration:.6f} seconds"
                    enriched_lines.append(duration_line)
                else:
                    # Store start timestamp
                    measurement_starts[measurement_id] = timestamp

        return enriched_lines

    @staticmethod
    def __convert_to_summary_table(summary: List[str], model_name: str, utility_paper_locations: Dict[str, str],
                                   safety_paper_locations: Dict[str, str],
                                   calculate_utility_degradation_average: bool) -> \
            List[str]:
        table_lines: List[str] = []
        # Structures to hold parsed values
        utility_data: Dict[str, Dict[str, float]] = defaultdict(dict)
        safety_data: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Regex to extract relevant fields
        summary_pattern = re.compile(
            r"SUMMARY (\w+)(?: cumulative)? for (utility|safety) benchmark (\w+) in pruning iteration (\w+): ([\d\.]+)"
            )

        for line in summary:
            match = summary_pattern.search(line)
            if match:
                metric, category, benchmark, iteration, value = match.groups()
                iteration = f"{int(iteration) + 1}" if iteration != "None" else "Unpruned"  # Normalize "None" to "0"
                value = float(value)
                target = utility_data if category == "utility" else safety_data
                target[benchmark][iteration] = value

        if calculate_utility_degradation_average:
            for b, iteration_data in utility_data.items():
                unpruned = iteration_data["Unpruned"]
                diffs = []
                for i, v in iteration_data.items():
                    if i == "Unpruned":
                        continue
                    else:
                        diffs.append(unpruned - v)
                average = sum(diffs) / len(diffs)
                iteration_data['AVG Degradation'] = average

        table_lines.append("\r\n")
        table_lines.extend(
            ExperimentVisualizer.__create_table(f"UTILITY BENCHMARKS per Pruning Iterations for {model_name}:",
                                                utility_data, utility_paper_locations,
                                                calculate_utility_degradation_average))
        table_lines.append("\r\n")
        table_lines.extend(
            ExperimentVisualizer.__create_table(f"SAFETY BENCHMARKS per Pruning Iterations for {model_name}:",
                                                safety_data, safety_paper_locations, False))
        return table_lines

    @staticmethod
    def __create_table(title: str, data: Dict[str, Dict[str, float]], paper_locations: Dict[str, str],
                       calculate_utility_degradation_average: bool) -> List[str]:
        table_lines: List[str] = []

        # Define headers
        iterations = [
            "Unpruned",
            "1",
            "2",
            "3",
            "4",
            "5"
            ]
        if calculate_utility_degradation_average:
            iterations.append("AVG Degradation")

        # Determine column widths
        col_widths = [max(len("Benchmark"), max(len(k) for k in data.keys()))]
        for i in iterations:
            col_width = max(len(i), 9)  # at least as wide as a float
            for v in data.values():
                if i in v:
                    col_width = max(col_width, len(f"{v[i]:.3f}"))
            col_widths.append(col_width)
        col_widths.append(max(len("Location in Paper"), max(len(k) for k in paper_locations.keys())))

        # Total columns
        columns = ["Benchmark"] + iterations + ["Location in Paper"]

        # Build line separators
        def line_sep(char="-", cross="+"):
            return cross + cross.join(char * (w + 2) for w in col_widths) + cross

        # Build a row with cell content
        def row(cells):
            return "| " + " | ".join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(cells)
                ) + " |"

        # Calculate total table width (sum of columns + separators)
        total_width = sum(col_widths) + 3 * len(col_widths) + 1

        # Title inside the table border
        table_lines.append("+" + "=" * (total_width - 2) + "+")
        table_lines.append("|" + title.center(total_width - 2) + "|")
        table_lines.append("+" + "=" * (total_width - 2) + "+")

        # Header line 1
        header_row_1 = (
                "| " + "".center(col_widths[0]) + " | "
                + "Pruning Iterations".center(
            sum(col_widths[1:-1]) + 3 * len(col_widths[1:-1]) - 3) + " | " + "".center(col_widths[-1]) + " |"
        )
        table_lines.append(line_sep("="))
        table_lines.append(header_row_1)
        table_lines.append(line_sep("="))

        # Header line 2 (actual column labels)
        table_lines.append(row(columns))
        table_lines.append(line_sep())

        # Rows of data
        for benchmark in sorted(data.keys()):
            row_data = [benchmark]
            for it in iterations:
                val = data[benchmark].get(it, "")
                if benchmark == 'strongreject':
                    val_pring = f"{val: .3f}"
                else:
                    val_pring = f"{val * 100.: .2f} %"
                row_data.append(f"{val_pring}" if isinstance(val, float) else "")
            loc_in_paper = paper_locations[benchmark]
            row_data.append(loc_in_paper)
            table_lines.append(row(row_data))
            table_lines.append(line_sep())

        return table_lines
