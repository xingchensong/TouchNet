import subprocess

import pytest


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
    (1, "b6288ac0dcdbb316f54d3b3a1302a991"),
    (2, "ce21a98e84a57e75c3a754167e4843df")
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
                --datatypes 'audio+text'
        """
    )
    assert result.returncode == 0
    md5 = run_shell(
        f"""
        find tests/tmp/{num}sample_per_shard -type f -exec md5sum {{}} \\; | sort | cut -d ' ' -f1 | md5sum | awk '{{print $1}}'
        """
    )
    assert md5.stdout.strip() == expected_md5
    run_shell(f"rm -rf tests/tmp/{num}sample_per_shard")
