#!/usr/bin/env python

# Inspired by Nate Odell's (naodell@gmail.com) BatchMaster.py for condor
# https://github.com/NWUHEP/BLT/blob/topic_wbranch/BLTAnalysis/python/BatchMaster.py

import sys
import os
import shutil
import random
from pathlib import Path
import argparse
import configparser
import json
import ast
import subprocess
import itertools
from datetime import datetime


class SlurmMaster:
    def __init__(self, config):
        self.date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        self.config = Path(config)
        self.parse_config()

        self.log_dir = Path(self.setup['log_path'], self.date)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.notes_dir = Path(self.setup['notes_path'], self.date)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.date)
        # self.wave_eq_mre_id = random.randint(10000, 90000)
        self.staging_dir= Path(self.setup['stage_path'], self.date)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.core_dir = Path(self.setup['core_path'])
        self.core_package = self.setup['core_package']
        self.setup_staging()

    def setup_staging(self):
        print('Staging Code')
        shutil.copytree(Path(self.core_dir, self.core_package),
                        Path(self.staging_dir, self.core_package+self.date))
        shutil.copy(Path(self.core_dir, 'setup.py'), Path(self.staging_dir, 'setup.py'))
        shutil.rmtree(Path(self.staging_dir, self.core_package+self.date, '__pycache__'))
        os.system("sed -i " +
                  f"'s/import {self.core_package}/import {self.core_package}{self.date}/g' " +
                  str(self.staging_dir) + '/' + self.core_package + self.date + '/*py')
        os.system("sed -i " +
                  f"'s/from {self.core_package}/from {self.core_package}{self.date}/g' " +
                  str(self.staging_dir) + '/' + self.core_package + self.date + '/*py')
        os.system("sed -i " +
                  f"'s/{self.core_package}/{self.core_package}{self.date}/g' " +
                  str(self.staging_dir) + '/setup.py')

        with open(Path(self.notes_dir, 'notes.txt'), 'w') as f:
            f.writelines(self.notes+'\n')

        os.chdir(self.staging_dir)
        os.system('python setup.py develop')

    def parse_config(self):
        config = configparser.ConfigParser()
        # config.read('config_inis/test_config.ini')
        config.read(str(self.config))
        sections = config.sections()
        self.config_dict = {}
        self.subj_list = []
        self.only_group = []
        self.project = None

        if 'Project' in sections:
            self.project = config['Project']['task']

        if 'Notes' in sections:
            self.notes = config['Notes']['note']

        if 'Node' in sections:
            self.node = config['Node']

        if 'Setup' in sections:
            self.setup = config['Setup']
            for c in self.setup:
                self.setup[c] = self.setup[c].strip('"')
                self.setup[c] = self.setup[c].strip("'")

        # Iterate through config and convert all scalars to lists
        for c in config['Hyper']:
            print(c)
            print(config['Hyper'][c])
            val = ast.literal_eval(config['Hyper'][c])
            if c == 'subject_id':
                if type(val) == list:
                    self.subj_list  = val
                else:
                    self.subj_list.append(val)
            elif c == 'subj_group':
                self.subj_list = val
            elif c == 'only_group':
                self.only_group = val
            else:
                if type(val) == list:
                    self.config_dict[c] = val
                else:
                    self.config_dict[c] = [val]
        if len(self.subj_list) == 0:
            raise KeyError('No subject_id defined!')
        if len(self.only_group) == 0:
            self.only_group = list(range(len(self.subj_list)))

        # Make every possible combo of config items
        self.config_combos = product_dict(**self.config_dict)

    def generate_slurm_script(self, number, conf, subj, subj_num, project):
        '''Make a slurm submission script.'''
        if project == 'GenWave':
            module = 'gen_phantom_wave.py'
            self.gpu = True
        if project == 'MRE':
            module = 'train_mre_model.py'
            self.gpu = True
        else:
            raise ValueError(f'"{project}" is not a valid project')

        if type(subj) is list:
            subj_name = f'GROUP{subj_num}'
            subj = ' '.join(subj)
        else:
            subj_name = subj

        print(conf)
        arg_string = f'--subject_id {subj}'
        for i in conf:
            if type(conf[i]) is list:
                print(conf[i])
                clean_vals = ' '.join(conf[i])
                arg_string += f' --{i} {clean_vals}'
            else:
                arg_string += f' --{i}={conf[i]}'
        script_name = str(self.staging_dir)+f'/slurm_script_{self.date}_n{number}_subj{subj_name}'
        script = open(script_name, 'w')
        script.write('#!/bin/bash\n')
        script.write(f'#SBATCH --partition={self.node["partition"]}\n')
        script.write(f'#SBATCH --gpus={self.node["ngpus"]}\n')
        script.write('#SBATCH --nodes=1\n')
        script.write('#SBATCH --time=24:00:00\n')
        script.write('#SBATCH --mail-user=brianleepollack@gmail.com\n')
        script.write(f'#SBATCH --output={str(self.log_dir)}/job_n{number}_subj{subj_name}.stdout\n')
        script.write(f'#SBATCH --error={str(self.log_dir)}/job_n{number}_subj{subj_name}.stderr\n')
        script.write('\n')

        script.write('set -x\n')
        script.write('echo "$@"\n')
        script.write('source /jet/home/bpollack/anaconda3/etc/profile.d/conda.sh\n')
        script.write('conda activate mre_ai\n')
        script.write('\n')
        script.write('nvidia-smi\n')

        script.write(f'python {str(self.staging_dir)}/{self.core_package}{self.date}/{module}'
                     f' {arg_string}\n')

        script.close()
        print(arg_string)
        return script_name

    def submit_scripts(self):
        if self.project != 'XR':
            for i, conf in enumerate(self.config_combos):
                # for j, subj in enumerate(self.subj_list):
                for j in self.only_group:
                    subj = self.subj_list[j]
                    script_name = self.generate_slurm_script(i, conf, subj, j, self.project)
                    print(script_name)
                    subprocess.call(f'sbatch {script_name}', shell=True)
        else:
            # for j, subj in enumerate(self.subj_list):
            for j in self.only_group:
                subj = self.subj_list[j]
                script_name = self.generate_slurm_script(0, self.config_dict, subj, j, self.project)
                print(script_name)
                subprocess.call(f'sbatch {script_name}', shell=True)


def product_dict(**kwargs):
    '''From https://stackoverflow.com/a/5228294/4942417,
    Produce all combos of configs for list-like items.'''
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Submit a series of SLURM jobs.')
    parser.add_argument('config', type=str, help='Path to config_file.')
    args = parser.parse_args()

    SM = SlurmMaster(args.config)
    SM.submit_scripts()
    print(SM.notes)
