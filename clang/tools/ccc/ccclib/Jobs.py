import Arguments
import Util

class Job(object):
    """Job - A set of commands to execute as a single task."""

    def iterjobs(self):
        abstract

class Command(Job):
    """Command - Represent the information needed to execute a single
    process."""

    def __init__(self, executable, args):
        assert Util.all_true(args, lambda x: isinstance(x, Arguments.Arg))
        self.executable = executable
        self.args = args

    def __repr__(self):
        return Util.prefixAndPPrint(self.__class__.__name__,
                                    (self.executable, self.args))
    
    def render(self, args):
        argv = [self.executable]
        for arg in self.args:
            argv.extend(args.render(arg))
        return argv

    def iterjobs(self):
        yield self
    
class PipedJob(Job):
    """PipedJob - A sequence of piped commands."""

    def __init__(self, commands):
        assert all_true(args, lambda x: isinstance(x, Arguments.Command))
        self.commands = list(commands)

    def addJob(self, job):
        assert isinstance(job, Command)
        self.commands.append(job)

    def __repr__(self):
        return Util.prefixAndPPrint(self.__class__.__name__, (self.commands,))

class JobList(Job):
    """JobList - A sequence of jobs to perform."""

    def __init__(self, jobs=[]):
        self.jobs = list(jobs)
    
    def addJob(self, job):
        self.jobs.append(job)

    def __repr__(self):
        return Util.prefixAndPPrint(self.__class__.__name__, (self.jobs,))

    def iterjobs(self):
        for j in self.jobs:
            yield j
