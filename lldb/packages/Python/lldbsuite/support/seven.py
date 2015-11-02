import six

if six.PY2:
    import commands
    get_command_output = commands.getoutput
    get_command_status_output = commands.getstatusoutput

else:
    def get_command_status_output(command):
        try:
            import subprocess
            return (0, subprocess.check_output(command, shell=True))
        except subprocess.CalledProcessError as e:
            return (e.returncode, e.output)

    def get_command_output(command):
        return get_command_status_output(command)[1]
