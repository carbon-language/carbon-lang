"""Utility for changing directories and execution of commands in a subshell."""

import os, shlex, subprocess

def chdir(debugger, args, result, dict):
    """Change the working directory, or cd to ${HOME}."""
    dir = args.strip()
    if dir:
        os.chdir(args)
    else:
        os.chdir(os.path.expanduser('~'))
    print "Current working directory: %s" % os.getcwd()

def system(debugger, command_line, result, dict):
    """Execute the command (a string) in a subshell."""
    args = shlex.split(command_line)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    retcode = process.poll()
    if output and error:
        print "stdout=>\n", output
        print "stderr=>\n", error
    elif output:
        print output
    elif error:
        print error
    print "retcode:", retcode
