import os
from pathlib import Path

import pyafar_utils  # apt-get update && apt-get install -y libgl1 libglib2.0-0

input_paths_infant = [
    # "data/test_video/test_format/Kamera 1/test_video_child.avi",
    # "data/diti_02_04_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 1/23.10.2024 11_55_09 (UTC+02_00).avi",
    # "data/diti_06_05_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 1/11.01.2025 15_27_45 (UTC+01_00).avi",
    "data/diti_11_04_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 1/25.01.2025 11_17_44 (UTC+01_00).avi"
    # "data/diti_12_04_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 1/22.01.2025 12_42_34 (UTC+01_00).avi",
    # "data/diti_19_04_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 1/11.02.2025 10_34_53 (UTC+01_00).avi",
    # "data/diti_21_04_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 1/17.02.2025 10_01_23 (UTC+01_00).avi"
]

input_paths_adult = [
    # "data/test_video/test_format/Kamera 2/test_video_adult.avi",
    # "data/diti_02_04_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 2/23.10.2024 11_55_09 (UTC+02_00).avi",
    # "data/diti_06_05_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 2/11.01.2025 15_27_45 (UTC+01_00).avi",
    "data/diti_11_04_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 2/25.01.2025 11_17_44 (UTC+01_00).avi"
    # "data/diti_12_04_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 2/22.01.2025 12_42_34 (UTC+01_00).avi",
    # "data/diti_19_04_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 2/11.02.2025 10_34_53 (UTC+01_00).avi",
    # "data/diti_21_04_kocyk_bez_zabawek/Format odtwarzacza mediów/Kamera 2/17.02.2025 10_01_23 (UTC+01_00).avi"
]

full_visualisation = False


def get_output_path(input_path, suffix):
    path = Path(input_path)
    parts = path.parts

    experiment_dir = parts[1]
    kamera_dir = parts[3]
    kamera_number = kamera_dir.split()[1]

    output_dir = Path('results') / experiment_dir / f'Kamera_{kamera_number}_{suffix}'
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    return str(output_dir)


def create_csv(df, output_path):
    # df = pyafar_utils.fix_dataframe(df)
    pyafar_utils.export_au_to_csv(df, output_path + "_short.csv", type='infant')


for input_path_i, input_path_a in zip(input_paths_infant, input_paths_adult):
    output_path_i = get_output_path(input_path_i, 'infant')
    output_path_a = get_output_path(input_path_a, 'adult')

    print("Processing", input_path_i)
    df_i = pyafar_utils.run_infant(input_path_i, AUs=pyafar_utils.AUs_infant)
    create_csv(df_i, output_path_i)

    print("Processing", input_path_a)
    df_a = pyafar_utils.run_adult(input_path_a, AUs=pyafar_utils.AUs_infant, AU_Int=[])
    create_csv(df_a, output_path_a)

    try:
        if full_visualisation:
            video_output_path_i = output_path_i + ".avi"
            video_output_path_a = output_path_a + ".avi"

            print("Visualisation", video_output_path_i)
            pyafar_utils.save_video_with_au(df_i, input_path_i, video_output_path_i, landmarks=True)
            print("Visualisation", video_output_path_a)
            pyafar_utils.save_video_with_au(df_a, input_path_a, video_output_path_a, landmarks=True)

            merged_output_path = output_path_i + "_merged.avi"
            pyafar_utils.merge_videos_vertically([video_output_path_i, video_output_path_a], merged_output_path)
            os.remove(video_output_path_i)
            os.remove(video_output_path_a)
    except Exception as e:  # Avoid bare 'except'
        # Log the error for debugging
        print(f"Error during video processing: {str(e)}")
