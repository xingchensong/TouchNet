import json
import subprocess

import pytest
import torch
import torchaudio

from touchnet.data import DataConfig
from touchnet.data.datapipe import TouchDatapipe


@pytest.fixture
def run_shell():
    def _run(cmd, check=True):
        return subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
    return _run


@pytest.mark.parametrize("num, expected_md5", [
    (1, "52aa63d2a8bb241047c12b9006c3b647"),
    (2, "a77adbbe26ee4bc6c1aefc38ec858273")
])
def test_make_data(run_shell, num, expected_md5):
    result = run_shell(
        f"""
            python touchnet/bin/make_data.py \
                --save_dir tests/tmp/{num}sample_per_shard \
                --jsonl_path tests/assets/dataset/data.jsonl  \
                --num_utt_per_shard {num} \
                --audio_resample 16000 \
                --num_workers 1 \
                --datatypes 'audio+metainfo'
        """
    )
    assert result.returncode == 0
    md5 = run_shell(
        f"""
        find tests/tmp/{num}sample_per_shard -type f -exec md5sum {{}} \\; | sort | cut -d ' ' -f1 | md5sum | awk '{{print $1}}'
        """
    )
    assert md5.stdout.strip() == expected_md5
    orig_data = {}
    with open("tests/assets/dataset/data.jsonl", "r") as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            orig_data[data['key']] = data
    data_config = DataConfig()
    data_config.datalist_path = f"tests/tmp/{num}sample_per_shard/data.list"
    data_config.datalist_shuffling = False
    data_config.datalist_sharding = False
    data_config.datalist_epoch = 1
    data_config.dataset_shuffling = False
    data_config.audio_speed_perturb = False
    data_config.audiofeat_spec_aug = False
    data_config.audiofeat_spec_sub = False
    data_config.audiofeat_spec_trim = False
    data_config.audiofeat_dither = 0.0
    datapipe = TouchDatapipe(data_config, 0, 1)
    for data in datapipe:
        key = data['key']
        assert key in orig_data
        assert data['wav'] == orig_data[key]['wav']
        assert data['txt'] == orig_data[key]['txt']
        orig_waveform = torchaudio.load(data['wav'])[0]
        waveform = data['waveform']
        assert torch.allclose(orig_waveform, waveform)
    run_shell(f"rm -rf tests/tmp/{num}sample_per_shard")
