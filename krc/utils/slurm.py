import os
import subprocess
import time

def submitjob(changecards, binary, version, cwd):
    for c in changecards:
        #KRC PATH support is bad, binary and file must be colocated...
        c = os.path.basename(c)
        proc = subprocess.Popen('sbatch', stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        command = '/home/jlaura/anaconda3/envs/krc/bin/python -u /home/jlaura/krc_changecards/submit_to_cluster/spawnkrc.py'

        job_string="""\
#!/bin/bash -l
#SBATCH -n 1
#SBATCH -p longall
#SBATCH --job-name krc
#SBATCH -t 12:00:00
#SBATCH --workdir {}
pwd
{} {} {} {}""".format(cwd, command, binary, c.strip(), version)
        proc.stdin.write(str.encode(job_string))
        out, err = proc.communicate()
        if version < 352:
            time.sleep(60)
        else:
            time.sleep(0.25) # For 3.4.4