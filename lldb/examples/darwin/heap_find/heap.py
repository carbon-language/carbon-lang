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
import lldb.utils.symbolication

def load_dylib():
    if lldb.target:
        python_module_directory = os.path.dirname(__file__)
        heap_code_directory = python_module_directory + '/heap'
        libheap_dylib_path = heap_code_directory + '/heap/libheap.dylib'
        if not os.path.exists(libheap_dylib_path):
            make_command = '(cd "%s" ; make)' % heap_code_directory
            print make_command
            print commands.getoutput(make_command)
        if os.path.exists(libheap_dylib_path):
            libheap_dylib_spec = lldb.SBFileSpec(libheap_dylib_path)
            if lldb.target.FindModule(libheap_dylib_spec):
                return None # success, 'libheap.dylib' already loaded
            if lldb.process:
                state = lldb.process.state
                if state == lldb.eStateStopped:
                    (libheap_dylib_path)
                    error = lldb.SBError()
                    image_idx = lldb.process.LoadImage(libheap_dylib_spec, error)
                    if error.Success():
                        return None
                    else:
                        if error:
                            return 'error: %s' % error
                        else:
                            return 'error: "process load \'%s\'" failed' % libheap_dylib_spec
                else:
                    return 'error: process is not stopped'
            else:
                return 'error: invalid process'
        else:
            return 'error: file does not exist "%s"' % libheap_dylib_path
    else:
        return 'error: invalid target'
        
    debugger.HandleCommand('process load "%s"' % libheap_dylib_path)
    
def add_common_options(parser):
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    parser.add_option('-o', '--po', action='store_true', dest='print_object_description', help='print the object descriptions for any matches', default=False)
    parser.add_option('-m', '--memory', action='store_true', dest='memory', help='dump the memory for each matching block', default=False)
    parser.add_option('-f', '--format', type='string', dest='format', help='the format to use when dumping memory if --memory is specified', default=None)
    parser.add_option('-s', '--stack', action='store_true', dest='stack', help='gets the stack that allocated each malloc block if MallocStackLogging is enabled', default=False)
    #parser.add_option('-S', '--stack-history', action='store_true', dest='stack_history', help='gets the stack history for all allocations whose start address matches each malloc block if MallocStackLogging is enabled', default=False)
    
def heap_search(options, arg_str):
    dylid_load_err = load_dylib()
    if dylid_load_err:
        print dylid_load_err
        return
    expr = None
    arg_str_description = arg_str
    default_memory_format = "Y" # 'Y' is "bytes with ASCII" format
    #memory_chunk_size = 1
    if options.type == 'pointer':
        expr = 'find_pointer_in_heap((void *)%s)' % arg_str
        arg_str_description = 'malloc block containing pointer %s' % arg_str
        default_memory_format = "A" # 'A' is "address" format
        #memory_chunk_size = lldb.process.GetAddressByteSize()
    elif options.type == 'cstr':
        expr = 'find_cstring_in_heap("%s")' % arg_str
        arg_str_description = 'malloc block containing "%s"' % arg_str
    elif options.type == 'addr':
        expr = 'find_block_for_address((void *)%s)' % arg_str
        arg_str_description = 'malloc block for %s' % arg_str
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
                description = '[%u] %s: addr = 0x%x' % (i, arg_str_description, malloc_addr)
                if offset != 0:
                    description += ' + %u' % (offset)
                description += ', size = %u' % (malloc_size)
                if type_name == 'void *':
                    if options.type == 'pointer' and malloc_size == 4096:
                        error = lldb.SBError()
                        data = bytearray(lldb.process.ReadMemory(malloc_addr, 16, error))
                        if data == '\xa1\xa1\xa1\xa1AUTORELEASE!':
                            description += ', type = (AUTORELEASE!)'
                    print description
                else:
                    description += ', type = %s' % (type_name)
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
                        description +=', ivar = %s' % (ivar_member.name)

                    print description
                    if derefed_dynamic_value:
                        print derefed_dynamic_value
                    if options.print_object_description:
                        desc = dynamic_value.GetObjectDescription()
                        if desc:
                            print '  (%s) 0x%x %s\n' % (type_name, malloc_addr, desc)
                if options.memory:
                    memory_format = options.format
                    if not memory_format:
                        memory_format = default_memory_format
                    cmd_result = lldb.SBCommandReturnObject()
                    #count = malloc_size / memory_chunk_size
                    memory_command = "memory read -f %s 0x%x 0x%x" % (memory_format, malloc_addr, malloc_addr + malloc_size)
                    lldb.debugger.GetCommandInterpreter().HandleCommand(memory_command, cmd_result)
                    print cmd_result.GetOutput()
                if options.stack:
                    symbolicator = lldb.utils.symbolication.Symbolicator()
                    symbolicator.target = lldb.target
                    expr_str = "g_stack_frames_count = sizeof(g_stack_frames)/sizeof(uint64_t); (int)__mach_stack_logging_get_frames((unsigned)mach_task_self(), 0x%xull, g_stack_frames, g_stack_frames_count, &g_stack_frames_count)" % (malloc_addr)
                    #print expr_str
                    expr = lldb.frame.EvaluateExpression (expr_str);
                    expr_error = expr.GetError()
                    if expr_error.Success():
                        err = expr.unsigned
                        if err:
                            print 'error: __mach_stack_logging_get_frames() returned error %i' % (err)
                        else:
                            count_expr = lldb.frame.EvaluateExpression ("g_stack_frames_count")
                            count = count_expr.unsigned
                            #print 'g_stack_frames_count is %u' % (count)
                            if count > 0:
                                frame_idx = 0
                                frames_expr = lldb.value(lldb.frame.EvaluateExpression ("g_stack_frames"))
                                done = False
                                for stack_frame_idx in range(count):
                                    if not done:
                                        frame_load_addr = int(frames_expr[stack_frame_idx])
                                        if frame_load_addr >= 0x1000:
                                            frames = symbolicator.symbolicate(frame_load_addr)
                                            if frames:
                                                for frame in frames:
                                                    print '[%3u] %s' % (frame_idx, frame)
                                                    frame_idx += 1
                                            else:
                                                print '[%3u] 0x%x' % (frame_idx, frame_load_addr)
                                                frame_idx += 1
                                        else:
                                            done = True
                    else:
                        print 'error: %s' % (expr_error)
        else:
            print '%s %s was not found in any malloc blocks' % (options.type, arg_str)
    else:
        print expr_sbvalue.error
    print     
    
def ptr_refs(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <PTR> [PTR ...]"
    description='''Searches the heap for pointer references on darwin user space programs. 
    
    Any matches that were found will dump the malloc blocks that contain the pointers 
    and might be able to print what kind of objects the pointers are contained in using 
    dynamic type information in the program.'''
    parser = optparse.OptionParser(description=description, prog='ptr_refs',usage=usage)
    add_common_options(parser)
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

def cstr_refs(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <CSTR> [CSTR ...]"
    description='''Searches the heap for C string references on darwin user space programs. 
    
    Any matches that were found will dump the malloc blocks that contain the C strings 
    and might be able to print what kind of objects the pointers are contained in using 
    dynamic type information in the program.'''
    parser = optparse.OptionParser(description=description, prog='cstr_refs',usage=usage)
    add_common_options(parser)
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

def malloc_info(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <ADDR> [ADDR ...]"
    description='''Searches the heap a malloc block that contains the addresses specified as arguments. 

    Any matches that were found will dump the malloc blocks that match or contain
    the specified address. The matching blocks might be able to show what kind 
    of objects they are using dynamic type information in the program.'''
    parser = optparse.OptionParser(description=description, prog='cstr_refs',usage=usage)
    add_common_options(parser)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return

    options.type = 'addr'

    if args:

        for data in args:
            heap_search (options, data)
    else:
        print 'error: no c string arguments were given to search for'

def __lldb_init_module (debugger, dict):
    # This initializer is being run from LLDB in the embedded command interpreter
    # Add any commands contained in this module to LLDB
    debugger.HandleCommand('command script add -f heap.ptr_refs ptr_refs')
    debugger.HandleCommand('command script add -f heap.cstr_refs cstr_refs')
    debugger.HandleCommand('command script add -f heap.malloc_info malloc_info')
    print '"ptr_refs", "cstr_refs", and "malloc_info" commands have been installed, use the "--help" options on these commands for detailed help.'




