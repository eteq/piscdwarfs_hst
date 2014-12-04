from __future__ import division

import os
import sys
import subprocess


def find_dolphot(dirstocheck=None):
    if dirstocheck is None:
        return subprocess.check_output(['which', 'dolphot'])
    else:
        for dr in dirstocheck:
            path = os.path.join(dr, 'dolphot')
            if os.path.isfile(os.path.join(dr, 'dolphot')):
                return path
        raise ValueError('Could not find dolphot in {0}'.format(dirstocheck))


class DolphotRunner(object):
    def __init__(self, cmd='dolphot', logfile='auto', paramfile='auto',
                       execpathordirs=None, workingdir='.', params={}):
        """
        if `logfile` is "auto", it means use the first argument
        """
        if isinstance(execpathordirs, basestring):
            dolphot_path = execpathordirs
            if not os.path.isfile(dolphot_path):
                raise ValueError('could not find dolphot path "{0}"'.format(dolphot_path))
        else:
            dolphot_path = find_dolphot(execpathordirs)
        self.dolphot_bin_dir = os.path.abspath(os.path.split(dolphot_path)[0])

        self.workingdir = workingdir
        self.cmd = cmd
        self.logfile = logfile
        self.paramfile = paramfile
        self.params = params

    @property
    def workingdir(self):
        return self._workingdir
    @workingdir.setter
    def workingdir(self, value):
        self._workingdir = os.path.abspath(value)

    def __call__(self, *args):
        from time import sleep, time

        exec_path = os.path.join(self.dolphot_bin_dir, self.cmd)
        if self.logfile == 'auto':
            logfile = args[0] + '_' + self.cmd + '.log'
        else:
            logfile = self.logfile

        if self.paramfile == 'auto':
            paramfile = args[0] + '_' + self.cmd + '.param'
        else:
            paramfile = self.paramfile

        args = list(args)
        args.insert(0, exec_path)

        if paramfile:
            if self.params:
                with open(paramfile, 'w') as f:
                    for k, v in self.params.items():
                        f.write('{0} = {1}\n'.format(k, v))
                args.append('-p' + paramfile)
        else:
            for k, v in self.params.items():
                args.append('{0}={1}'.format(k, v))

        p = subprocess.Popen(args, cwd=self.workingdir, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)

        output = []
        stt = time()
        while p.poll() is None:
            print (time() - stt, '0')
            p.stdout.flush()
            line = p.stdout.readline()
            if line == '':
                print (time() - stt, '1')
                sleep(.05)  # 50 ms pause
            else:
                print (time() - stt, '2')
                output.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
        print (time() - stt, '3')
        output.append(p.stdout.read())
        self.last_out = ''.join(output)

        if logfile is not None:
            with open(logfile, 'w') as f:
                f.write(self.last_out)

        if p.returncode != 0:
            raise ValueError('command "{0}" returned {1} instead of '
                             '0'.format(self.cmd, p.returncode), self.last_out)
        return self.last_out
