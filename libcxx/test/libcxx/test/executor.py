import os

from libcxx.test import tracing

from lit.util import executeCommand  # pylint: disable=import-error


class Executor(object):
    def run(self, exe_path, cmd, local_cwd, env=None):
        """Execute a command.
            Be very careful not to change shared state in this function.
            Executor objects are shared between python processes in `lit -jN`.
        Args:
            exe_path: str:   Local path to the executable to be run
            cmd: [str]:      subprocess.call style command
            local_cwd: str:  Local path to the working directory
            env: {str: str}: Environment variables to execute under
        Returns:
            out, err, exitCode
        """
        raise NotImplementedError


class LocalExecutor(Executor):
    def __init__(self):
        super(LocalExecutor, self).__init__()

    def run(self, exe_path, cmd=None, work_dir='.', env=None):
        cmd = cmd or [exe_path]
        env_cmd = []
        if env:
            env_cmd += ['env']
            env_cmd += ['%s=%s' % (k, v) for k, v in env.items()]
        if work_dir == '.':
            work_dir = os.getcwd()
        return executeCommand(env_cmd + cmd, cwd=work_dir)


class PrefixExecutor(Executor):
    """Prefix an executor with some other command wrapper.

    Most useful for setting ulimits on commands, or running an emulator like
    qemu and valgrind.
    """
    def __init__(self, commandPrefix, chain):
        super(PrefixExecutor, self).__init__()

        self.commandPrefix = commandPrefix
        self.chain = chain

    def run(self, exe_path, cmd=None, work_dir='.', env=None):
        cmd = cmd or [exe_path]
        return self.chain.run(exe_path, self.commandPrefix + cmd, work_dir,
                              env=env)


class PostfixExecutor(Executor):
    """Postfix an executor with some args."""
    def __init__(self, commandPostfix, chain):
        super(PostfixExecutor, self).__init__()

        self.commandPostfix = commandPostfix
        self.chain = chain

    def run(self, exe_path, cmd=None, work_dir='.', env=None):
        cmd = cmd or [exe_path]
        return self.chain.run(cmd + self.commandPostfix, work_dir, env=env)



class TimeoutExecutor(PrefixExecutor):
    """Execute another action under a timeout.

    Deprecated. http://reviews.llvm.org/D6584 adds timeouts to LIT.
    """
    def __init__(self, duration, chain):
        super(TimeoutExecutor, self).__init__(
            ['timeout', duration], chain)


class SSHExecutor(Executor):
    def __init__(self, host, username=None):
        super(SSHExecutor, self).__init__()

        self.user_prefix = username + '@' if username else ''
        self.host = host
        self.scp_command = 'scp'
        self.ssh_command = 'ssh'

        self.local_run = executeCommand
        # TODO(jroelofs): switch this on some -super-verbose-debug config flag
        if False:
            self.local_run = tracing.trace_function(
                self.local_run, log_calls=True, log_results=True,
                label='ssh_local')

    def remote_temp_dir(self):
        return self._remote_temp(True)

    def remote_temp_file(self):
        return self._remote_temp(False)

    def _remote_temp(self, is_dir):
        # TODO: detect what the target system is, and use the correct
        # mktemp command for it. (linux and darwin differ here, and I'm
        # sure windows has another way to do it)

        # Not sure how to do suffix on osx yet
        dir_arg = '-d' if is_dir else ''
        cmd = 'mktemp -q {} /tmp/libcxx.XXXXXXXXXX'.format(dir_arg)
        temp_path, err, exitCode = self.__execute_command_remote([cmd])
        temp_path = temp_path.strip()
        if exitCode != 0:
            raise RuntimeError(err)
        return temp_path

    def copy_in(self, local_srcs, remote_dsts):
        scp = self.scp_command
        remote = self.host
        remote = self.user_prefix + remote

        # This could be wrapped up in a tar->scp->untar for performance
        # if there are lots of files to be copied/moved
        for src, dst in zip(local_srcs, remote_dsts):
            cmd = [scp, '-p', src, remote + ':' + dst]
            self.local_run(cmd)

    def delete_remote(self, remote):
        try:
            self.__execute_command_remote(['rm', '-rf', remote])
        except OSError:
            # TODO: Log failure to delete?
            pass

    def run(self, exe_path, cmd=None, work_dir='.', env=None):
        if work_dir != '.':
            raise NotImplementedError(
                'work_dir arg is not supported for SSHExecutor')
        target_exe_path = None
        target_cwd = None
        try:
            target_exe_path = self.remote_temp_file()
            target_cwd = self.remote_temp_dir()
            if cmd:
                # Replace exe_path with target_exe_path.
                cmd = [c if c != exe_path else target_exe_path for c in cmd]
            else:
                cmd = [target_exe_path]
            self.copy_in([exe_path], [target_exe_path])
            return self.__execute_command_remote(cmd, target_cwd, env)
        except:
            raise
        finally:
            if target_exe_path:
                self.delete_remote(target_exe_path)
            if target_cwd:
                self.delete_remote(target_cwd)

    def __execute_command_remote(self, cmd, remote_work_dir='.', env=None):
        remote = self.user_prefix + self.host
        ssh_cmd = [self.ssh_command, '-oBatchMode=yes', remote]
        if env:
            env_cmd = ['env'] + ['%s=%s' % (k, v) for k, v in env.items()]
        else:
            env_cmd = []
        remote_cmd = ' '.join(env_cmd + cmd)
        if remote_work_dir != '.':
            remote_cmd = 'cd ' + remote_work_dir + ' && ' + remote_cmd
        return self.local_run(ssh_cmd + [remote_cmd])
