#!/usr/bin/python

#----------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
#
# # To use this in the embedded python interpreter using "lldb" just
# import it with the full path using the "command script import" 
# command
#   (lldb) command script import /path/to/heap.py
#
# For the shells csh, tcsh:
#   ( setenv PYTHONPATH /path/to/LLDB.framework/Resources/Python ; ./heap.py )
#
# For the shells sh, bash:
#   PYTHONPATH=/path/to/LLDB.framework/Resources/Python ./heap.py 
#----------------------------------------------------------------------

import lldb
import commands
import optparse
import shlex

def heap_search(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <PATH> [PATH ...]"
    description='''This command lets you run the /bin/ls command from within lldb as a quick and easy example.'''
    parser = optparse.OptionParser(description=description, prog='heap_search',usage=usage)
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    parser.add_option('-t', '--type', type='string', dest='type', help='the type of data to search for (defaults to "pointer")', default='pointer')
    parser.add_option('-o', '--po', action='store_true', dest='po', default='print the object descriptions for any matches')
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return
    
    if args:
        
        for data in args:
            if options.type == 'pointer':
                ptr = int(data, 0)
                expr = 'find_pointer_in_heap(0x%x)' % ptr
                #print 'expr: %s' % expr
                expr_sbvalue = lldb.frame.EvaluateExpression (expr)
                if expr_sbvalue.error.Success():
                    if expr_sbvalue.unsigned:
                        match_value = lldb.value(expr_sbvalue)  
                        i = 0
                        while 1:
                            match_entry = match_value[i]; i += 1
                            malloc_addr = int(match_entry.addr)
                            if malloc_addr == 0:
                                break
                            malloc_size = int(match_entry.size)
                            offset = int(match_entry.offset)
                            dynamic_value = match_entry.addr.sbvalue.dynamic
                            # If the type is still 'void *' then we weren't able to figure
                            # out a dynamic type for the malloc_addr
                            type_name = dynamic_value.type.name
                            if type_name == 'void *':
                                if malloc_size == 4096:
                                    error = lldb.SBError()
                                    data = bytearray(lldb.process.ReadMemory(malloc_addr, 16, error))
                                    if data == '\xa1\xa1\xa1\xa1AUTORELEASE!':
                                        print 'found %s 0x%x in (autorelease object pool) 0x%x, malloc_size = %u, offset = %u' % (options.type, ptr, malloc_addr, malloc_size, offset)
                                        continue
                                print 'found %s 0x%x in malloc block 0x%x, malloc_size = %u, offset = %u' % (options.type, ptr, malloc_addr, malloc_size, offset)
                            else:
                                print 'found %s 0x%x in (%s) 0x%x, malloc_size = %u, offset = %u' % (options.type, ptr, type_name, malloc_addr, malloc_size, offset)
                                if options.po:
                                    desc = dynamic_value.GetObjectDescription()
                                    if desc:
                                        print '  (%s) 0x%x %s\n' % (type_name, malloc_addr, desc)
                    else:
                        print '%s 0x%x was not found in any malloc blocks' % (options.type, ptr)
                else:
                    print expr_sbvalue.error
            elif options.type == 'cstring':
                expr = 'find_cstring_in_heap("%s")' % data
                #print 'expr: %s' % expr
                expr_sbvalue = lldb.frame.EvaluateExpression (expr)
                if expr_sbvalue.error.Success():
                    if expr_sbvalue.unsigned:
                        match_value = lldb.value(expr_sbvalue)  
                        print match_value
                        i = 0
                        while 1:
                            match_entry = match_value[i]; i += 1
                            malloc_addr = int(match_entry.addr)
                            if malloc_addr == 0:
                                break
                            malloc_size = int(match_entry.size)
                            offset = int(match_entry.offset)
                            dynamic_value = match_entry.addr.sbvalue.dynamic
                            # If the type is still 'void *' then we weren't able to figure
                            # out a dynamic type for the malloc_addr
                            type_name = dynamic_value.type.name
                            if type_name == 'void *':
                                print 'found %s "%s" in malloc block 0x%x, malloc_size = %u, offset = %u' % (options.type, data, malloc_addr, malloc_size, offset)
                            else:
                                print 'found %s "%s" in (%s) 0x%x, malloc_size = %u, offset = %u' % (options.type, data, type_name, malloc_addr, malloc_size, offset)
                                if options.po:
                                    desc = dynamic_value.GetObjectDescription()
                                    if desc:
                                        print '  (%s) 0x%x %s\n' % (type_name, malloc_addr, desc)
                    else:
                        print '%s "%s" was not found in any malloc blocks' % (options.type, data)
                else:
                    print expr_sbvalue.error
                
            else:
                print 'error: invalid type "%s"\nvalid values are "pointer", "cstring"' % options.type
                sys.exit(1)
    else:
        print 'error: no arguments were given'

if __name__ == '__main__':
    # This script is being run from the command line, create a debugger in case we are
    # going to use any debugger functions in our function.
    lldb.debugger = lldb.SBDebugger.Create()
    ls (sys.argv)

def __lldb_init_module (debugger, dict):
    # This initializer is being run from LLDB in the embedded command interpreter
    # Add any commands contained in this module to LLDB
    debugger.HandleCommand('process load libheap.dylib')
    debugger.HandleCommand('command script add -f heap.heap_search heap_search')
    print '"heap_search" command installed, type "heap_search --help" for detailed help'




