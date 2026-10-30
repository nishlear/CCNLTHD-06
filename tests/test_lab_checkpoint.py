import json
import subprocess
import sys


def test_lab_resumes_same_thread_in_fresh_process_and_replay_is_stable(tmp_path):
    def run(phase,thread="lab-review-1"):
        return subprocess.run([sys.executable,"-m","labs.state_checkpoint.solution","--db",str(tmp_path/"lab.db"),
            "--thread-id",thread,"--phase",phase],capture_output=True,text=True,timeout=30)
    first=run("start");assert first.returncode==0,first.stderr
    before=json.loads(first.stdout)
    assert before["collected"]==["A","B"] and before["next"]==["review"] and before["output"]==[]
    resumed=run("resume");assert resumed.returncode==0,resumed.stderr
    after=json.loads(resumed.stdout)
    assert after["next"]==[] and after["output"]==["A","B"] and after["approved"]
    assert json.loads(run("resume").stdout)==after
    assert run("resume","wrong-thread").returncode!=0
