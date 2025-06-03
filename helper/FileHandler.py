from __future__ import annotations

import os
from datetime import datetime
from typing import List

import numpy as np
from numpy import ndarray


class FileHandler:
    """Deals with folders, files and interaction with file content."""
    __POSTFIX_FOR_DATA_SIZE: str = '_DataSize.txt'

    # region Done file
    @staticmethod
    def create_done_file(directory: str) -> None:
        with open(f'{directory}done.txt', 'w') as output_file:
            current_time = str(datetime.now())
            output_file.write(current_time)

    @staticmethod
    def check_file_exists(file_path: str) -> bool:
        return os.path.exists(f'{file_path}')

    @staticmethod
    def check_done_file_exists(directory: str) -> bool:
        return os.path.exists(f'{directory}done.txt')

    # endregion

    # region Directories
    @staticmethod
    def ensure_dir_exists(*path_components: str) -> str:
        if len(path_components) == 0:
            return './'
        current_path = '/'
        for path_component in path_components:
            splitted_path_components = path_component.split('/')
            for splitted_component in splitted_path_components:
                current_path += splitted_component
                if not current_path[-1] == '/':
                    current_path += '/'
                if not os.path.exists(current_path):
                    os.mkdir(current_path)
                    if not os.path.exists(current_path):
                        raise Exception(f'mkdir failed for {current_path}')
                elif not os.path.isdir(current_path):
                    raise Exception('No directory: {0}'.format(current_path))
        assert os.path.exists(current_path)
        return current_path

    @staticmethod
    def dir_is_empty(path: str) -> bool:
        return os.listdir(path) == []

    # endregion

    # region Read / Write to File
    @staticmethod
    def write_list_to_file(file_path: str, list_to_save: list) -> None:
        list_to_save = [str(x) for x in list_to_save]
        with open(file_path, 'w') as output_file:
            output_file.write('\n'.join(list_to_save))

    @staticmethod
    def write_to_file(file_path: str, str_to_save: str) -> None:
        with open(file_path, 'w') as output_file:
            output_file.write(str_to_save)

    @staticmethod
    def append_to_file_as_row(file_path: str, value: float | int | str) -> None:
        with open(file_path, 'a') as output_file:
            output_file.write(f'{value}\n')

    @staticmethod
    def get_lines_from_file_as_list(file_path: str) -> list:
        with open(file_path) as input_file:
            lines = input_file.read()
            if lines[-1] == '\n':
                lines = lines[:-1]
            lines = lines.split('\n')
            lines = [line[:-1] if line[-1] == '\n' else line for line in lines]
            return lines

    @staticmethod
    def get_lines_from_file_as_float_list(file_path: str) -> List[float]:
        with open(file_path) as input_file:
            lines = input_file.read()
            if lines[-1] == '\n':
                lines = lines[:-1]
            lines = lines.split('\n')
            lines = [float(line[:-1]) if line[-1] == '\n' else float(line) for line in lines]
            return lines

    @staticmethod
    def get_lines_from_file_as_int_list(file_path: str) -> List[int]:
        with open(file_path) as input_file:
            lines = input_file.read()
            if lines[-1] == '\n':
                lines = lines[:-1]
            lines = lines.split('\n')
            lines = [int(line[:-1]) if line[-1] == '\n' else int(line) for line in lines]
            return lines

    @staticmethod
    def create_model_data_size_file(filename: str, dataset_size: int) -> None:
        with open(filename + FileHandler.__POSTFIX_FOR_DATA_SIZE, 'w') as file:
            size = str(dataset_size)
            file.write(size)
            file.close()

    @staticmethod
    def load_model_data_size_file(filename: str) -> int | None:
        if not os.path.exists(filename + FileHandler.__POSTFIX_FOR_DATA_SIZE):
            return None

        with open(filename + FileHandler.__POSTFIX_FOR_DATA_SIZE, 'r') as file:
            file_content = file.read()
            file.close()
        return int(file_content)

    @staticmethod
    def write_2dim_ndarray_matrix_to_file(filename: str, values: ndarray) -> None:
        assert len(values.shape) == 2
        assert values.shape[0] == values.shape[1]

        with open(filename, 'w') as file:
            for index, dim_one in enumerate(values):
                file.write(",".join([str(value) for value in dim_one]))
                if index < len(values):
                    file.write('\n')

    @staticmethod
    def delete_file(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        else:
            print(f"Unable to delete file, since it does not exist: {path}")

    # noinspection PyTypeChecker
    @staticmethod
    def np_save_txt(file_name: str, values: List[float] | ndarray, fmt=None) -> None:
        if fmt is None:
            return np.savetxt(file_name, values)
        else:
            return np.savetxt(file_name, values, fmt=fmt)

    # endregion

    @staticmethod
    def get_new_numbered_file(folder: str, file_name: str, file_extension: str) -> str:
        file_list = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        identifier = 0
        for logfile in file_list:
            if logfile[-len(file_extension):] != file_extension:
                continue
            if logfile[:len(file_name)] != file_name:
                continue
            number = logfile[len(file_name):-len(file_extension)]
            if not number.isnumeric():
                continue
            number = int(number)
            if number >= identifier:
                identifier = number + 1

        return f'{file_name}{identifier}{file_extension}'

    @staticmethod
    def get_new_numbered_folder(parent_folder: str, folder: str) -> str:
        folder_list = [f for f in os.listdir(parent_folder) if not os.path.isfile(os.path.join(parent_folder, f))]

        identifier = 0
        for existing_folder in folder_list:
            if existing_folder[:len(folder)] != folder:
                continue
            number = existing_folder[len(folder):]
            if not number.isnumeric():
                continue
            number = int(number)
            if number >= identifier:
                identifier = number + 1

        return f'{folder}{identifier}'
