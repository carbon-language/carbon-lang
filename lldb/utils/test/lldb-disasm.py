#!/usr/bin/env python

"""
Run lldb to disassemble all the available functions for an executable image.

"""

import os
import sys
from optparse import OptionParser

def setupSysPath():
    """
    Add LLDB.framework/Resources/Python and the test dir to the sys.path.
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


def run_command(ci, cmd, res, echoInput=True, echoOutput=True):
    if echoInput:
        print "run command:", cmd
    ci.HandleCommand(cmd, res)
    if res.Succeeded():
        if echoOutput:
            print "run_command output:", res.GetOutput()
    else:
        if echoOutput:
            print "run command failed!"
            print "run_command error:", res.GetError()

def do_lldb_disassembly(lldb_commands, exe, disassemble_options, num_symbols, symbols_to_disassemble):
    import lldb, lldbutil, atexit, re

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
    run_command(ci, 'image dump symtab', res, echoOutput=False)

    if not res.Succeeded():
        print "Symbol table dump failed!"
        sys.exit(-2)

    # Do disassembly on the symbols.
    # The following line from the 'image dump symtab' gives us a hint as to the
    # starting char position of the symbol name.
    # Index   UserID DSX Type         File Address/Value Load Address       Size               Flags      Name
    # ------- ------ --- ------------ ------------------ ------------------ ------------------ ---------- ----------------------------------
    # [    0]      0     Code         0x0000000000000820                    0x0000000000000000 0x000e0008 sandbox_init_internal
    symtab_dump = res.GetOutput()
    symbol_pos = -1
    code_type_pos = -1
    code_type_end = -1

    # Heuristics: the first 50 lines should give us the answer for symbol_pos and code_type_pos.
    for line in symtab_dump.splitlines()[:50]:
        print "line:", line
        if re.match("^Index.*Name$", line):
            symbol_pos = line.rfind('Name')
            #print "symbol_pos:", symbol_pos
            code_type_pos = line.find('Type')
            code_type_end = code_type_pos + 4
            #print "code_type_pos:", code_type_pos
            break

    # Define a generator for the symbols to disassemble.
    def symbol_iter(num, symbols, symtab_dump):
        # If we specify the symbols to disassemble, ignore symbol table dump.
        if symbols:
            for i in range(len(symbols)):
                print "symbol:", symbols[i]
                yield symbols[i]
        else:
            limited = True if num != -1 else False
            if limited:
                count = 0
            for line in symtab_dump.splitlines():
                if limited and count >= num:
                    return
                if line[code_type_pos:code_type_end] == 'Code':
                    symbol = line[symbol_pos:]
                    print "symbol:", symbol
                    if limited:
                        count = count + 1
                        print "symbol count:", count
                        yield symbol

    # Disassembly time.
    for symbol in symbol_iter(num_symbols, symbols_to_disassemble, symtab_dump):
        cmd = "disassemble %s '%s'" % (disassemble_options, symbol)
        run_command(ci, cmd, res)


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
    parser.add_option('-e', '--executable',
                      type='string', action='store',
                      dest='executable',
                      help="""Mandatory: the executable to do disassembly on.""")
    parser.add_option('-o', '--options',
                      type='string', action='store',
                      dest='disassemble_options',
                      help="""Mandatory: the options passed to lldb's 'disassemble' command.""")
    parser.add_option('-n', '--num-symbols',
                      type='int', action='store', default=-1,
                      dest='num_symbols',
                      help="""The number of symbols to disassemble, if specified.""")
    parser.add_option('-s', '--symbol',
                      type='string', action='append', metavar='SYMBOL', default=[],
                      dest='symbols_to_disassemble',
                      help="""The symbol(s) to invoke lldb's 'disassemble' command on, if specified.""")
    
    opts, args = parser.parse_args()

    lldb_commands = opts.lldb_commands

    if not opts.executable or not opts.disassemble_options:
        parser.print_help()
        sys.exit(1)

    executable = opts.executable
    disassemble_options = opts.disassemble_options
    num_symbols = opts.num_symbols
    symbols_to_disassemble = opts.symbols_to_disassemble

    # We have parsed the options.
    print "lldb commands:", lldb_commands
    print "executable:", executable
    print "disassemble options:", disassemble_options
    print "num of symbols to disassemble:", num_symbols
    print "symbols to disassemble:", symbols_to_disassemble

    setupSysPath()
    do_lldb_disassembly(lldb_commands, executable, disassemble_options, num_symbols, symbols_to_disassemble)

if __name__ == '__main__':
    main()
