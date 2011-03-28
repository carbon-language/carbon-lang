#!/usr/bin/env python

"""
Run lldb to disassemble all the available functions for an executable image.

"""

import os
import sys
from optparse import OptionParser

def setupSysPath():
    """
    Add LLDB.framework/Resources/Python to the search paths for modules.
    """
    # Get the directory containing the current script.
    scriptPath = sys.path[0]
    if not scriptPath.endswith(os.path.join('utils', 'test')):
        print "This script expects to reside in lldb's utils/test directory."
        sys.exit(-1)

    # This is our base name component.
    base = os.path.abspath(os.path.join(scriptPath, os.pardir, os.pardir))

    # This is for the goodies in the test directory under base.
    sys.path.append(os.path.join(base,'test'))

    # These are for xcode build directories.
    xcode3_build_dir = ['build']
    xcode4_build_dir = ['build', 'lldb', 'Build', 'Products']
    dbg = ['Debug']
    rel = ['Release']
    bai = ['BuildAndIntegration']
    python_resource_dir = ['LLDB.framework', 'Resources', 'Python']

    dbgPath  = os.path.join(base, *(xcode3_build_dir + dbg + python_resource_dir))
    dbgPath2 = os.path.join(base, *(xcode4_build_dir + dbg + python_resource_dir))
    relPath  = os.path.join(base, *(xcode3_build_dir + rel + python_resource_dir))
    relPath2 = os.path.join(base, *(xcode4_build_dir + rel + python_resource_dir))
    baiPath  = os.path.join(base, *(xcode3_build_dir + bai + python_resource_dir))
    baiPath2 = os.path.join(base, *(xcode4_build_dir + bai + python_resource_dir))

    lldbPath = None
    if os.path.isfile(os.path.join(dbgPath, 'lldb.py')):
        lldbPath = dbgPath
    elif os.path.isfile(os.path.join(dbgPath2, 'lldb.py')):
        lldbPath = dbgPath2
    elif os.path.isfile(os.path.join(relPath, 'lldb.py')):
        lldbPath = relPath
    elif os.path.isfile(os.path.join(relPath2, 'lldb.py')):
        lldbPath = relPath2
    elif os.path.isfile(os.path.join(baiPath, 'lldb.py')):
        lldbPath = baiPath
    elif os.path.isfile(os.path.join(baiPath2, 'lldb.py')):
        lldbPath = baiPath2

    if not lldbPath:
        print 'This script requires lldb.py to be in either ' + dbgPath + ',',
        print relPath + ', or ' + baiPath
        sys.exit(-1)

    # This is to locate the lldb.py module.  Insert it right after sys.path[0].
    sys.path[1:1] = [lldbPath]
    print "sys.path:", sys.path


def run_command(ci, cmd, res):
    print "run command:", cmd
    ci.HandleCommand(cmd, res)
    if res.Succeeded():
        print "output:", res.GetOutput()
    else:
        print "run command failed!"
        print "error:", res.GetError()

def do_lldb_disassembly(lldb_commands, lldb_options, exe):
    import lldb, lldbutil, atexit

    # Create the debugger instance now.
    dbg = lldb.SBDebugger.Create()
    if not dbg.IsValid():
            raise Exception('Invalid debugger instance')

    # Register an exit callback.
    atexit.register(lambda: lldb.SBDebugger.Terminate())

    # We want our debugger to be synchronous.
    dbg.SetAsync(False)

    # Get the command interpreter from the debugger.
    ci = dbg.GetCommandInterpreter()
    if not ci:
        raise Exception('Could not get the command interpreter')

    # And the associated result object.
    res = lldb.SBCommandReturnObject()

    # See if there any extra command(s) to execute before we issue the file command.
    for cmd in lldb_commands:
        run_command(ci, cmd, res)

    # Now issue the file command.
    run_command(ci, 'file %s' % exe, res)

    # Send the 'image dump symtab' command.
    run_command(ci, 'image dump symtab', res)    

def main():
    # This is to set up the Python path to include the pexpect-2.4 dir.
    # Remember to update this when/if things change.
    scriptPath = sys.path[0]
    sys.path.append(os.path.join(scriptPath, os.pardir, os.pardir, 'test', 'pexpect-2.4'))

    parser = OptionParser(usage="""\
Run lldb to disassemble all the available functions for an executable image.

Usage: %prog [options]
""")
    parser.add_option('-C', '--lldb-command',
                      type='string', action='append', metavar='COMMAND',
                      default=[], dest='lldb_commands',
                      help='Command(s) lldb executes after starting up (can be empty)')
    parser.add_option('-O', '--lldb-options',
                      type='string', action='store',
                      dest='lldb_options',
                      help="""The options passed to 'lldb' command if specified.""")
    parser.add_option('-e', '--executable',
                      type='string', action='store',
                      dest='executable',
                      help="""The executable to do disassembly on.""")

    opts, args = parser.parse_args()

    lldb_commands = opts.lldb_commands
    lldb_options = opts.lldb_options

    if not opts.executable:
        parser.print_help()
        sys.exit(1)
    executable = opts.executable

    # We have parsed the options.
    print "lldb commands:", lldb_commands
    print "lldb options:", lldb_options
    print "executable:", executable

    setupSysPath()
    do_lldb_disassembly(lldb_commands, lldb_options, executable)

if __name__ == '__main__':
    main()
