#!/usr/bin/python

#----------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
#
# To use this in the embedded python interpreter using "lldb":
#   % cd /path/containing/cmdtemplate.py
#   % lldb
#   (lldb) script import cmdtemplate
#
# For the shells csh, tcsh:
#   ( setenv PYTHONPATH /path/to/LLDB.framework/Resources/Python ; ./cmdtemplate.py )
#
# For the shells sh, bash:
#   PYTHONPATH=/path/to/LLDB.framework/Resources/Python ./cmdtemplate.py 
#----------------------------------------------------------------------

import lldb
import commands
import optparse
import shlex

def ls(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <PATH> [PATH ...]"
    description='''This command lets you run the /bin/ls command from within lldb as a quick and easy example.'''
    parser = optparse.OptionParser(description=description, prog='ls',usage=usage)
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return
    
    for arg in args:
        if options.verbose:
            print commands.getoutput('/bin/ls "%s"' % arg)
        else:
            print commands.getoutput('/bin/ls -lAF "%s"' % arg)

if __name__ == '__main__':
    # This script is being run from the command line, create a debugger in case we are
    # going to use any debugger functions in our function.
    lldb.debugger = lldb.SBDebugger.Create()
    ls (sys.argv)
elif lldb.debugger:
    # This script is being run from LLDB in the emabedded command interpreter
    # Add any commands contained in this module to LLDB
    lldb.debugger.HandleCommand('command script add -f cmdtemplate.ls ls')
    print '"ls" command installed, type "ls --help" for detailed help'
