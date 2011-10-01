#!/usr/bin/python

#----------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
# On MacOSX csh, tcsh:
#   setenv PYTHONPATH /Developer/Library/PrivateFrameworks/LLDB.framework/Resources/Python
# On MacOSX sh, bash:
#   export PYTHONPATH=/Developer/Library/PrivateFrameworks/LLDB.framework/Resources/Python
#----------------------------------------------------------------------

import lldb
import optparse
import os
import re
import sys
import time

PARSE_MODE_NORMAL = 0
PARSE_MODE_THREAD = 1
PARSE_MODE_IMAGES = 2
PARSE_MODE_THREGS = 3
PARSE_MODE_SYSTEM = 4

class CrashLog:
    """Class that does parses darwin crash logs"""
    thread_state_regex = re.compile('^Thread ([0-9]+) crashed with')
    thread_regex = re.compile('^Thread ([0-9]+)([^:]*):(.*)')
    frame_regex = re.compile('^([0-9]+).*\t(0x[0-9a-fA-F]+) +(.*)')
    image_regex_uuid = re.compile('(0x[0-9a-fA-F]+)[- ]+(0x[0-9a-fA-F]+) +([^ ]+) +([^<]+)<([-0-9a-fA-F]+)> (.*)');
    image_regex_no_uuid = re.compile('(0x[0-9a-fA-F]+)[- ]+(0x[0-9a-fA-F]+) +([^ ]+) +([^/]+)/(.*)');
    empty_line_regex = re.compile('^$')
    
    class Thread:
        """Class that represents a thread in a darwin crash log"""
        def __init__(self, index):
            self.index = index
            self.frames = list()
            self.registers = dict()
            self.reason = None
            self.queue = None
        
        def dump(self, prefix):
            print "%sThread[%u] %s" % (prefix, self.index, self.reason)
            if self.frames:
                print "%s  Frames:" % (prefix)
                for frame in self.frames:
                    frame.dump(prefix + '    ')
            if self.registers:
                print "%s  Registers:" % (prefix)
                for reg in self.registers.keys():
                    print "%s    %-5s = %#16.16x" % (prefix, reg, self.registers[reg])
        
        def did_crash(self):
            return self.reason != None
        
        def __str__(self):
            s = "Thread[%u]" % self.index
            if self.reason:
                s += ' %s' % self.reason
            return s

    class Frame:
        """Class that represents a stack frame in a thread in a darwin crash log"""
        def __init__(self, index, pc, details):
            self.index = index
            self.pc = pc
            self.sym_ctx = None
            self.details = details

        def __str__(self):
            return "[%2u] %#16.16x %s" % (self.index, self.pc, self.details)
            
        def dump(self, prefix):
            print "%s%s" % (prefix, self)
    
    class Image:
        """Class that represents a binary images in a darwin crash log"""
        def __init__(self, text_addr_lo, text_addr_hi, ident, version, uuid, path):
            self.text_addr_lo = text_addr_lo
            self.text_addr_hi = text_addr_hi
            self.ident = ident
            self.version = version
            self.uuid = uuid
            self.path = path
 
        def dump(self, prefix):
            print "%s%s" % (prefix, self)

        def __str__(self):
            return "%#16.16x %s %s" % (self.text_addr_lo, self.uuid, self.path)
    
    
    def __init__(self, path):
        """CrashLog constructor that take a path to a darwin crash log file"""
        self.path = path;
        self.info_lines = list()
        self.system_profile = list()
        self.threads = list()
        self.images = list()
        self.crashed_thread_idx = -1
        self.version = -1
        f = open(self.path)
        self.file_lines = f.read().splitlines()
        parse_mode = PARSE_MODE_NORMAL
        thread = None
        for line in self.file_lines:
            # print line
            line_len = len(line)
            if line_len == 0:
                if thread:
                    if parse_mode == PARSE_MODE_THREAD:
                        if thread.index == self.crashed_thread_idx:
                            thread.reason = ''
                            if self.thread_exception:
                                thread.reason += self.thread_exception
                            if self.thread_exception_data:
                                thread.reason += " (%s)" % self.thread_exception_data                                
                        self.threads.append(thread)
                    thread = None
                else:
                    # only append an extra empty line if the previous line 
                    # in the info_lines wasn't empty
                    if len(self.info_lines) > 0 and len(self.info_lines[-1]):
                        self.info_lines.append(line)
                parse_mode = PARSE_MODE_NORMAL
                # print 'PARSE_MODE_NORMAL'
            elif parse_mode == PARSE_MODE_NORMAL:
                if line.startswith ('Process:'):
                    (self.process_name, pid_with_brackets) = line[8:].strip().split()
                    self.process_id = pid_with_brackets.strip('[]')
                elif line.startswith ('Path:'):
                    self.process_path = line[5:].strip()
                elif line.startswith ('Identifier:'):
                    self.process_identifier = line[11:].strip()
                elif line.startswith ('Version:'):
                    (self.process_version, compatability_version) = line[8:].strip().split()
                    self.process_compatability_version = compatability_version.strip('()')
                elif line.startswith ('Parent Process:'):
                    (self.parent_process_name, pid_with_brackets) = line[15:].strip().split()
                    self.parent_process_id = pid_with_brackets.strip('[]') 
                elif line.startswith ('Exception Type:'):
                    self.thread_exception = line[15:].strip()
                    continue
                elif line.startswith ('Exception Codes:'):
                    self.thread_exception_data = line[16:].strip()
                    continue
                elif line.startswith ('Crashed Thread:'):
                    self.crashed_thread_idx = int(line[15:].strip().split()[0])
                    continue
                elif line.startswith ('Report Version:'):
                    self.version = int(line[15:].strip())
                    continue
                elif line.startswith ('System Profile:'):
                    parse_mode = PARSE_MODE_SYSTEM
                    continue
                elif (line.startswith ('Interval Since Last Report:') or
                      line.startswith ('Crashes Since Last Report:') or
                      line.startswith ('Per-App Interval Since Last Report:') or
                      line.startswith ('Per-App Crashes Since Last Report:') or
                      line.startswith ('Sleep/Wake UUID:') or
                      line.startswith ('Anonymous UUID:')):
                    # ignore these
                    continue  
                elif line.startswith ('Thread'):
                    thread_state_match = self.thread_state_regex.search (line)
                    if thread_state_match:
                        thread_state_match = self.thread_regex.search (line)
                        thread_idx = int(thread_state_match.group(1))
                        parse_mode = PARSE_MODE_THREGS
                        thread = self.threads[thread_idx]
                    else:
                        thread_match = self.thread_regex.search (line)
                        if thread_match:
                            # print 'PARSE_MODE_THREAD'
                            parse_mode = PARSE_MODE_THREAD
                            thread_idx = int(thread_match.group(1))
                            thread = CrashLog.Thread(thread_idx)
                    continue
                elif line.startswith ('Binary Images:'):
                    parse_mode = PARSE_MODE_IMAGES
                    continue
                self.info_lines.append(line.strip())
            elif parse_mode == PARSE_MODE_THREAD:
                frame_match = self.frame_regex.search(line)
                if frame_match:
                    thread.frames.append (CrashLog.Frame(int(frame_match.group(1)), int(frame_match.group(2), 0), frame_match.group(3)))
                else:
                    print "error: frame regex failed"
            elif parse_mode == PARSE_MODE_IMAGES:
                image_match = self.image_regex_uuid.search (line)
                if image_match:
                    image = CrashLog.Image (int(image_match.group(1),0), 
                                            int(image_match.group(2),0), 
                                            image_match.group(3).strip(), 
                                            image_match.group(4).strip(), 
                                            image_match.group(5), 
                                            image_match.group(6))
                    self.images.append (image)
                else:
                    image_match = self.image_regex_no_uuid.search (line)
                    if image_match:
                        image = CrashLog.Image (int(image_match.group(1),0), 
                                                int(image_match.group(2),0), 
                                                image_match.group(3).strip(), 
                                                image_match.group(4).strip(), 
                                                None,
                                                image_match.group(5))
                        self.images.append (image)
                    else:
                        print "error: image regex failed for: %s" % line

            elif parse_mode == PARSE_MODE_THREGS:
                stripped_line = line.strip()
                reg_values = stripped_line.split('  ')
                for reg_value in reg_values:
                    (reg, value) = reg_value.split(': ')
                    thread.registers[reg.strip()] = int(value, 0)
            elif parse_mode == PARSE_MODE_SYSTEM:
                self.system_profile.append(line)
        f.close()
        
    def dump(self):
        print "Crash Log File: %s" % (self.path)
        print "\nThreads:"
        for thread in self.threads:
            thread.dump('  ')
        print "\nImages:"
        for image in self.images:
            image.dump('  ')

def disassemble_instructions (target, instructions, pc, insts_before_pc, insts_after_pc):
    lines = list()
    pc_index = -1
    comment_column = 50
    for inst_idx, inst in enumerate(instructions):
        inst_pc = inst.GetAddress().GetLoadAddress(target);
        if pc == inst_pc:
            pc_index = inst_idx
        mnemonic = inst.GetMnemonic (target)
        operands =  inst.GetOperands (target)
        comment =  inst.GetComment (target)
        #data = inst.GetData (target)
        lines.append ("%#16.16x: %8s %s" % (inst_pc, mnemonic, operands))
        if comment:
            line_len = len(lines[-1])
            if line_len < comment_column:
                lines[-1] += ' ' * (comment_column - line_len)
                lines[-1] += "; %s" % comment

    if pc_index >= 0:
        if pc_index >= insts_before_pc:
            start_idx = pc_index - insts_before_pc
        else:
            start_idx = 0
        end_idx = pc_index + insts_after_pc
        if end_idx > inst_idx:
            end_idx = inst_idx
        for i in range(start_idx, end_idx+1):
            if i == pc_index:
                print ' -> ', lines[i]
            else:
                print '    ', lines[i]

def print_module_section_data (section):
    print section
    section_data = section.GetSectionData()
    if section_data:
        ostream = lldb.SBStream()
        section_data.GetDescription (ostream, section.GetFileAddress())
        print ostream.GetData()

def print_module_section (section, depth):
    print section
            
    if depth > 0:
        num_sub_sections = section.GetNumSubSections()
        for sect_idx in range(num_sub_sections):
            print_module_section (section.GetSubSectionAtIndex(sect_idx), depth - 1)

def print_module_sections (module, depth):
    num_sections = module.GetNumSections()

    for sect_idx in range(num_sections):
        section = module.GetSectionAtIndex(sect_idx)
        print_module_section (section, depth)

def print_module_symbols (module):
    n = module.GetNumSymbols()

    for i in range(n):
        print module.GetSymbolAtIndex(i)

def usage():
    print "Usage: lldb-symbolicate.py [-n name] executable-image"
    sys.exit(0)


if __name__ == '__main__':
    parser = optparse.OptionParser(description='A script that parses skinny and universal mach-o files.')
    parser.add_option('--arch', type='string', metavar='arch', dest='triple', help='specify one architecture or target triple')
    parser.add_option('--platform', type='string', metavar='platform', dest='platform', help='specify one platform by name')
    parser.add_option('--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    parser.add_option('--interactive', action='store_true', dest='interactive', help='enable interactive mode', default=False)
    parser.add_option('--no-images', action='store_false', dest='show_images', help='don\'t show images in stack frames', default=True)
    parser.add_option('--no-dependents', action='store_false', dest='dependents', help='skip loading dependent modules', default=True)
    parser.add_option('--sections', action='store_true', dest='dump_sections', help='show module sections', default=False)
    parser.add_option('--symbols', action='store_true', dest='dump_symbols', help='show module symbols', default=False)
    parser.add_option('--image-list', action='store_true', dest='dump_image_list', help='show image list', default=False)
    parser.add_option('--debug-delay', type='int', dest='debug_delay', metavar='NSEC', help='pause for NSEC seconds for debugger', default=0)
    parser.add_option('--section-depth', type='int', dest='section_depth', help='set the section depth to use when showing sections', default=0)
    parser.add_option('--section-data', type='string', action='append', dest='sect_data_names', help='specify sections by name to display data for')
    parser.add_option('--address', type='int', action='append', dest='addresses', help='specify addresses to lookup')
    parser.add_option('--crash-log', type='string', action='append', dest='crash_log_files', help='specify crash log files to symbolicate')
    parser.add_option('--crashed-only', action='store_true', dest='crashed_only', help='only show the crashed thread', default=False)
    loaded_addresses = False
    (options, args) = parser.parse_args()
    if options.verbose:
        print 'options', options

    if options.debug_delay > 0:
        print "Waiting %u seconds for debugger to attach..." % options.debug_delay
        time.sleep(options.debug_delay)

    # Create a new debugger instance
    debugger = lldb.SBDebugger.Create()

    # When we step or continue, don't return from the function until the process 
    # stops. We do this by setting the async mode to false.
    debugger.SetAsync (False)
    error = lldb.SBError()
    
    if options.crash_log_files:
        options.dependents = False
        for crash_log_file in options.crash_log_files:
            triple = "x86_64"
            crash_log = CrashLog(crash_log_file)
            #crash_log.dump()
            target = debugger.CreateTarget (crash_log.process_path, options.triple, options.platform, options.dependents, error);
            exe_module = target.GetModuleAtIndex(0)
            image_paths = list()
            for image in crash_log.images:
                if image.path == crash_log.process_path:
                    module = exe_module
                else:
                    module = target.AddModule (image.path, options.triple, image.uuid)
                if image.path in image_paths:
                    print "warning: skipping %s loaded at %#16.16x duplicate entry (probably commpage)" % (image.path, image.text_addr_lo)
                else:
                    image_paths.append(image.path)
                    
                    if not module and image.uuid != module.GetUUIDString():
                        if image.uuid:
                            print "warning: couldn't locate %s %s" % (image.uuid, image.path)
                        else:
                            print "warning: couldn't locate %s" % (image.path)
                    else:
                        target.SetSectionLoadAddress (module.FindSection ("__TEXT"), image.text_addr_lo)
            for line in crash_log.info_lines:
                print line
            # Reconstruct inlined frames for all threads for anything that has debug info
            for thread in crash_log.threads:
                if options.crashed_only and thread.did_crash() == False:
                    continue
                # start a new frame list that we will fixup for each thread
                new_thread_frames = list()
                # Iterate through all concrete frames for a thread and resolve
                # any parent frames of inlined functions
                for frame_idx, frame in enumerate(thread.frames):
                    # Resolve the frame's pc into a section + offset address 'pc_addr'
                    pc_addr = target.ResolveLoadAddress (frame.pc)
                    # Check to see if we were able to resolve the address
                    if pc_addr:
                        # We were able to resolve the frame's PC into a section offset
                        # address.
                        
                        # Resolve the frame's PC value into a symbol context. A symbol
                        # context can resolve a module, compile unit, function, block,
                        # line table entry and/or symbol. If the frame has a block, then
                        # we can look for inlined frames, which are represented by blocks
                        # that have inlined information in them
                        frame.sym_ctx = target.ResolveSymbolContextForAddress (pc_addr, lldb.eSymbolContextEverything);
                        
                        # dump if the verbose option was specified
                        if options.verbose:
                            print "frame.pc = %#16.16x (file_addr = %#16.16x)" % (frame.pc, pc_addr.GetFileAddress())
                            print "frame.pc_addr = ", pc_addr
                            print "frame.sym_ctx = "
                            print frame.sym_ctx
                            print
                        
                        # Append the frame we already had from the crash log to the new
                        # frames list
                        new_thread_frames.append(frame)

                        new_frame = CrashLog.Frame (frame.index, -1, None)

                        # Try and use the current frame's symbol context to calculate a 
                        # parent frame for an inlined function. If the curent frame is
                        # inlined, it will return a valid symbol context for the parent 
                        # frame of the current inlined function
                        parent_pc_addr = lldb.SBAddress()
                        new_frame.sym_ctx = frame.sym_ctx.GetParentOfInlinedScope (pc_addr, parent_pc_addr)

                        # See if we were able to reconstruct anything?
                        while new_frame.sym_ctx:
                            # We have a parent frame of an inlined frame, create a new frame
                            # Convert the section + offset 'parent_pc_addr' to a load address 
                            new_frame.pc = parent_pc_addr.GetLoadAddress(target)
                            # push the new frame onto the new frame stack
                            new_thread_frames.append (new_frame)
                            # dump if the verbose option was specified
                            if options.verbose:
                                print "new_frame.pc = %#16.16x (%s)" % (new_frame.pc, parent_pc_addr)
                                print "new_frame.sym_ctx = "
                                print new_frame.sym_ctx
                                print
                            # Create another new frame in case we have multiple inlined frames
                            prev_new_frame = new_frame
                            new_frame = CrashLog.Frame (frame.index, -1, None)
                            # Swap the addresses so we can try another inlined lookup
                            pc_addr = parent_pc_addr;
                            new_frame.sym_ctx = prev_new_frame.sym_ctx.GetParentOfInlinedScope (pc_addr, parent_pc_addr)
                # Replace our thread frames with our new list that includes parent
                # frames for inlined functions
                thread.frames = new_thread_frames
            # Now iterate through all threads and display our richer stack backtraces
            for thread in crash_log.threads:
                this_thread_crashed = thread.did_crash()
                if options.crashed_only and this_thread_crashed == False:
                    continue
                print "%s" % thread
                prev_frame_index = -1
                for frame_idx, frame in enumerate(thread.frames):
                    details = '          %s' % frame.details
                    module = frame.sym_ctx.GetModule()
                    instructions = None
                    if module:
                        module_basename = module.GetFileSpec().GetFilename();
                        function_start_load_addr = -1
                        function_name = None
                        function = frame.sym_ctx.GetFunction()
                        block = frame.sym_ctx.GetBlock()
                        line_entry = frame.sym_ctx.GetLineEntry()
                        symbol = frame.sym_ctx.GetSymbol()
                        inlined_block = block.GetContainingInlinedBlock();
                        if inlined_block:
                            function_name = inlined_block.GetInlinedName();
                            block_range_idx = inlined_block.GetRangeIndexForBlockAddress (target.ResolveLoadAddress (frame.pc))
                            if block_range_idx < lldb.UINT32_MAX:
                                block_range_start_addr = inlined_block.GetRangeStartAddress (block_range_idx)
                                function_start_load_addr = block_range_start_addr.GetLoadAddress (target)
                            else:
                                function_start_load_addr = frame.pc
                            if this_thread_crashed and frame_idx == 0:
                                instructions = function.GetInstructions(target)
                        elif function:
                            function_name = function.GetName()
                            function_start_load_addr = function.GetStartAddress().GetLoadAddress (target)
                            if this_thread_crashed and frame_idx == 0:
                                instructions = function.GetInstructions(target)
                        elif symbol:
                            function_name = symbol.GetName()
                            function_start_load_addr = symbol.GetStartAddress().GetLoadAddress (target)
                            if this_thread_crashed and frame_idx == 0:
                                instructions = symbol.GetInstructions(target)

                        if function_name:
                            # Print the function or symbol name and annotate if it was inlined
                            inline_suffix = ''
                            if inlined_block: 
                                inline_suffix = '[inlined] '
                            else:
                                inline_suffix = '          '
                            if options.show_images:
                                details = "%s%s`%s" % (inline_suffix, module_basename, function_name)
                            else:
                                details = "%s" % (function_name)
                            # Dump the offset from the current function or symbol if it is non zero
                            function_offset = frame.pc - function_start_load_addr
                            if function_offset > 0:
                                details += " + %u" % (function_offset)
                            elif function_offset < 0:
                                defaults += " %i (invalid negative offset, file a bug) " % function_offset
                            # Print out any line information if any is available
                            if line_entry.GetFileSpec():
                                details += ' at %s' % line_entry.GetFileSpec().GetFilename()
                                details += ':%u' % line_entry.GetLine ()
                                column = line_entry.GetColumn()
                                if column > 0:
                                    details += ':%u' % column
                            
                        
                    # Only print out the concrete frame index if it changes.
                    # if prev_frame_index != frame.index:
                    #     print "[%2u] %#16.16x %s" % (frame.index, frame.pc, details)
                    # else:
                    #     print "     %#16.16x %s" % (frame.pc, details)
                    print "[%2u] %#16.16x %s" % (frame.index, frame.pc, details)
                    prev_frame_index = frame.index
                    if instructions:
                        print
                        disassemble_instructions (target, instructions, frame.pc, 4, 4)
                        print
                        
                print                

            if options.dump_image_list:
                print "Binary Images:"
                for image in crash_log.images:
                    print image
    else:
        for exe_file in args:

            # Create a target from a file and arch
            print "Creating a target for '%s'" % exe_file

            target = debugger.CreateTarget (exe_file, options.triple, options.platform, options.dependents, error);
        
            if target:
                exe_module = None;
                module_count = target.GetNumModules();
                for module_idx in range(module_count):
                    module = target.GetModuleAtIndex (module_idx)
                    if module_idx == 0:
                        exe_module = module
                    print "module[%u] = %s" % (module_idx, module)
            
                    if options.dump_symbols:
                        print_module_symbols (module)
                    if options.dump_sections:
                        print_module_sections (module, options.section_depth)
                    if options.sect_data_names:
                        for sect_name in options.sect_data_names:
                            section = module.FindSection (sect_name)
                            if section:
                                print_module_section_data (section)
                            else:
                                print "No section was found in '%s' named '%s'" % (module, sect_name)
                if options.addresses:
                    for address in options.addresses:
                        if loaded_addresses:
                            so_address = target.ResolveLoadAddress (address)
                            if so_address:
                                print so_address
                                so_address_sc = exe_module.ResolveSymbolContextForAddress (so_address, lldb.eSymbolContextEverything);
                                print so_address_sc
                            else:
                                print "error: 0x%8.8x failed to resolve as a load address" % (address)
                        else:
                            so_address = exe_module.ResolveFileAddress (address)
                
                            if so_address:
                                print so_address
                                so_address_sc = exe_module.ResolveSymbolContextForAddress (so_address, lldb.eSymbolContextEverything);
                                print so_address_sc
                            else:
                                print "error: 0x%8.8x failed to resolve as a file address in %s" % (address, exe_module)
                
                    # text_base_addr = 0x10000
                    # load_addr = 0x10bb0
                    # text_segment = exe_module.FindSection ("__TEXT")
                    # if text_segment:
                    #     target.SetSectionLoadAddress (text_segment, text_base_addr)
                    # 
                    #     load_so_addr = target.ResolveLoadAddress (load_addr)
                    # 
                    #     if load_so_addr:
                    #         sc = target.ResolveSymbolContextForAddress (so_addr, lldb.eSymbolContextEverything);
                    #         print sc
                    #     else:
                    #         print "error: 0x%8.8x failed to resolve as a load address" % (load_addr)
            else:
                print "error: ", error


    lldb.SBDebugger.Terminate()
        
