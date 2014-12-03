from __future__ import division

import os
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
    def __init__(self, cmd, logfile, execpathordirs=None, workingdir='.'):
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

    @property
    def workingdir(self):
        return self._workingdir
    @workingdir.setter
    def workingdir(self, value):
        self._workingdir = os.path.abspath(value)

    def __call__(self, *args):
        exec_path = os.path.join(self.dolphot_bin_dir, self.cmd)
        if self.logfile == 'auto':
            logfile = args[0] + '_' + self.cmd + '.log'
        else:
            logfile = self.logfile

        args = list(args)
        args.insert(0, exec_path)
        p = subprocess.Popen(args, cwd=self.workingdir, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        self.last_out = p.communicate()[0]
        if logfile is not None:
            with open(logfile, 'w') as f:
                f.write(self.last_out)

        if p.returncode != 0:
            raise ValueError('command "{0}" returned {1} instead of '
                             '0'.format(self.cmd, p.returncode), self.last_out)
        return self.last_out
