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
import os
import shlex

def heap_search(options, arg_str):
    expr = None
    if options.type == 'pointer':
        ptr = int(arg_str, 0)
        expr = 'find_pointer_in_heap(0x%x)' % ptr
    elif options.type == 'cstr':
        expr = 'find_cstring_in_heap("%s")' % arg_str
    else:
        print 'error: invalid type "%s"\nvalid values are "pointer", "cstr"' % options.type
        return
    
    expr_sbvalue = lldb.frame.EvaluateExpression (expr)
    if expr_sbvalue.error.Success():
        if expr_sbvalue.unsigned:
            match_value = lldb.value(expr_sbvalue)  
            i = 0
            while 1:
                match_entry = match_value[i]; i += 1
                malloc_addr = match_entry.addr.sbvalue.unsigned
                if malloc_addr == 0:
                    break
                malloc_size = int(match_entry.size)
                offset = int(match_entry.offset)
                dynamic_value = match_entry.addr.sbvalue.GetDynamicValue(lldb.eDynamicCanRunTarget)
                # If the type is still 'void *' then we weren't able to figure
                # out a dynamic type for the malloc_addr
                type_name = dynamic_value.type.name
                if type_name == 'void *':
                    if options.type == 'pointer' and malloc_size == 4096:
                        error = lldb.SBError()
                        data = bytearray(lldb.process.ReadMemory(malloc_addr, 16, error))
                        if data == '\xa1\xa1\xa1\xa1AUTORELEASE!':
                            print 'found %s %s: block = 0x%x, size = %u, offset = %u, type = (autorelease object pool)' % (options.type, arg_str, malloc_addr, malloc_size, offset)
                            continue
                
                print 'found %s %s: block = 0x%x, size = %u, offset = %u, type = \'%s\'' % (options.type, arg_str, malloc_addr, malloc_size, offset, type_name),
                derefed_dynamic_value = dynamic_value.deref
                ivar_member = None
                if derefed_dynamic_value:
                    derefed_dynamic_type = derefed_dynamic_value.type
                    member = derefed_dynamic_type.GetFieldAtIndex(0)
                    search_bases = False
                    if member:
                        if member.GetOffsetInBytes() <= offset:
                            for field_idx in range (derefed_dynamic_type.GetNumberOfFields()):
                                member = derefed_dynamic_type.GetFieldAtIndex(field_idx)
                                member_byte_offset = member.GetOffsetInBytes()
                                if member_byte_offset == offset:
                                    ivar_member = member
                                    break
                        else:
                            search_bases = True
                    else:
                        search_bases = True

                    if not ivar_member and search_bases:
                        for field_idx in range (derefed_dynamic_type.GetNumberOfDirectBaseClasses()):
                            member = derefed_dynamic_type.GetDirectBaseClassAtIndex(field_idx)
                            member_byte_offset = member.GetOffsetInBytes()
                            if member_byte_offset == offset:
                                ivar_member = member
                                break
                        if not ivar_member:
                            for field_idx in range (derefed_dynamic_type.GetNumberOfVirtualBaseClasses()):
                                member = derefed_dynamic_type.GetVirtualBaseClassAtIndex(field_idx)
                                member_byte_offset = member.GetOffsetInBytes()
                                if member_byte_offset == offset:
                                    ivar_member = member
                                    break
                    if ivar_member:
                        print ", ivar = %s" % ivar_member.name,
                    print "\n", dynamic_value.deref
                else:
                    print
                if options.print_object_description:
                    desc = dynamic_value.GetObjectDescription()
                    if desc:
                        print '  (%s) 0x%x %s\n' % (type_name, malloc_addr, desc)
        else:
            print '%s %s was not found in any malloc blocks' % (options.type, arg_str)
    else:
        print expr_sbvalue.error        
    
def heap_ptr_refs(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <PATH> [PATH ...]"
    description='''Searches the heap for pointer references on darwin user space programs. 
    
    Any matches that were found will dump the malloc blocks that contain the pointers 
    and might be able to print what kind of objects the pointers are contained in using 
    dynamic type information from the program.'''
    parser = optparse.OptionParser(description=description, prog='heap_ptr_refs',usage=usage)
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    parser.add_option('-o', '--po', action='store_true', dest='print_object_description', help='print the object descriptions for any matches', default=False)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return

    options.type = 'pointer'
    
    if args:
        
        for data in args:
            heap_search (options, data)
    else:
        print 'error: no pointer arguments were given'

def heap_cstr_refs(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <PATH> [PATH ...]"
    description='''Searches the heap for C string references on darwin user space programs. 
    
    Any matches that were found will dump the malloc blocks that contain the C strings 
    and might be able to print what kind of objects the pointers are contained in using 
    dynamic type information from the program.'''
    parser = optparse.OptionParser(description=description, prog='heap_cstr_refs',usage=usage)
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    parser.add_option('-o', '--po', action='store_true', dest='print_object_description', help='print the object descriptions for any matches', default=False)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return

    options.type = 'cstr'

    if args:

        for data in args:
            heap_search (options, data)
    else:
        print 'error: no c string arguments were given to search for'

def __lldb_init_module (debugger, dict):
    # This initializer is being run from LLDB in the embedded command interpreter
    # Add any commands contained in this module to LLDB
    libheap_dylib_path = os.path.dirname(__file__) + '/libheap.dylib'
    debugger.HandleCommand('process load "%s"' % libheap_dylib_path)
    debugger.HandleCommand('command script add -f heap.heap_ptr_refs heap_ptr_refs')
    debugger.HandleCommand('command script add -f heap.heap_cstr_refs heap_cstr_refs')
    print '"heap_ptr_refs" and "heap_cstr_refs" commands have been installed, use the "--help" options on these commands for detailed help.'




