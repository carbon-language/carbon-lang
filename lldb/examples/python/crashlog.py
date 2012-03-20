#!/usr/bin/python

#----------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
#
# To use this in the embedded python interpreter using "lldb":
#
#   cd /path/containing/crashlog.py
#   lldb
#   (lldb) script import crashlog
#   "crashlog" command installed, type "crashlog --help" for detailed help
#   (lldb) crashlog ~/Library/Logs/DiagnosticReports/a.crash
#
# The benefit of running the crashlog command inside lldb in the 
# embedded python interpreter is when the command completes, there 
# will be a target with all of the files loaded at the locations
# described in the crash log. Only the files that have stack frames
# in the backtrace will be loaded unless the "--load-all" option
# has been specified. This allows users to explore the program in the
# state it was in right at crash time. 
#
# On MacOSX csh, tcsh:
#   ( setenv PYTHONPATH /path/to/LLDB.framework/Resources/Python ; ./crashlog.py ~/Library/Logs/DiagnosticReports/a.crash )
#
# On MacOSX sh, bash:
#   PYTHONPATH=/path/to/LLDB.framework/Resources/Python ./crashlog.py ~/Library/Logs/DiagnosticReports/a.crash
#----------------------------------------------------------------------

import lldb
import commands
import optparse
import os
import plistlib
#import pprint # pp = pprint.PrettyPrinter(indent=4); pp.pprint(command_args)
import re
import shlex
import sys
import time
import uuid


PARSE_MODE_NORMAL = 0
PARSE_MODE_THREAD = 1
PARSE_MODE_IMAGES = 2
PARSE_MODE_THREGS = 3
PARSE_MODE_SYSTEM = 4

class CrashLog:
    """Class that does parses darwin crash logs"""
    thread_state_regex = re.compile('^Thread ([0-9]+) crashed with')
    thread_regex = re.compile('^Thread ([0-9]+)([^:]*):(.*)')
    frame_regex = re.compile('^([0-9]+) +([^ ]+) *\t(0x[0-9a-fA-F]+) +(.*)')
    image_regex_uuid = re.compile('(0x[0-9a-fA-F]+)[- ]+(0x[0-9a-fA-F]+) +[+]?([^ ]+) +([^<]+)<([-0-9a-fA-F]+)> (.*)');
    image_regex_no_uuid = re.compile('(0x[0-9a-fA-F]+)[- ]+(0x[0-9a-fA-F]+) +[+]?([^ ]+) +([^/]+)/(.*)');
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
        dsymForUUIDBinary = os.path.expanduser('~rc/bin/dsymForUUID')
        if not os.path.exists(dsymForUUIDBinary):
            dsymForUUIDBinary = commands.getoutput('which dsymForUUID')
            
        dwarfdump_uuid_regex = re.compile('UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*')
        
        def __init__(self, text_addr_lo, text_addr_hi, ident, version, uuid, path):
            self.text_addr_lo = text_addr_lo
            self.text_addr_hi = text_addr_hi
            self.ident = ident
            self.version = version
            self.arch = None
            self.uuid = uuid
            self.path = path
            self.resolved_path = None
            self.dsym = None
            self.module = None
        
        def dump(self, prefix):
            print "%s%s" % (prefix, self)
        
        def __str__(self):
            return "%#16.16x %s %s" % (self.text_addr_lo, self.uuid, self.get_resolved_path())
        
        def get_resolved_path(self):
            if self.resolved_path:
                return self.resolved_path
            elif self.path:
                return self.path
            return None

        def get_resolved_path_basename(self):
            path = self.get_resolved_path()
            if path:
                return os.path.basename(path)
            return None

        def dsym_basename(self):
            if self.dsym:
                return os.path.basename(self.dsym)
            return None
        
        def fetch_symboled_executable_and_dsym(self):
            if self.resolved_path:
                # Don't load a module twice...
                return 0
            print 'Locating %s %s...' % (self.uuid, self.path),
            if os.path.exists(self.dsymForUUIDBinary):
                dsym_for_uuid_command = '%s %s' % (self.dsymForUUIDBinary, self.uuid)
                s = commands.getoutput(dsym_for_uuid_command)
                if s:
                    plist_root = plistlib.readPlistFromString (s)
                    if plist_root:
                        plist = plist_root[self.uuid]
                        if plist:
                            if 'DBGArchitecture' in plist:
                                self.arch = plist['DBGArchitecture']
                            if 'DBGDSYMPath' in plist:
                                self.dsym = os.path.realpath(plist['DBGDSYMPath'])
                            if 'DBGSymbolRichExecutable' in plist:
                                self.resolved_path = os.path.expanduser (plist['DBGSymbolRichExecutable'])
            if not self.resolved_path and os.path.exists(self.path):
                dwarfdump_cmd_output = commands.getoutput('dwarfdump --uuid "%s"' % self.path)
                self_uuid = uuid.UUID(self.uuid)
                for line in dwarfdump_cmd_output.splitlines():
                    match = self.dwarfdump_uuid_regex.search (line)
                    if match:
                        dwarf_uuid_str = match.group(1)
                        dwarf_uuid = uuid.UUID(dwarf_uuid_str)
                        if self_uuid == dwarf_uuid:
                            self.resolved_path = self.path
                            self.arch = match.group(2)
                            break;
                if not self.resolved_path:
                    print "error: file %s '%s' doesn't match the UUID in the installed file" % (self.uuid, self.path)
                    return 0
            if (self.resolved_path and os.path.exists(self.resolved_path)) or (self.path and os.path.exists(self.path)):
                print 'ok'
                if self.path != self.resolved_path:
                    print '  exe = "%s"' % self.resolved_path 
                if self.dsym:
                    print ' dsym = "%s"' % self.dsym
                return 1
            else:
                return 0
        
        def load_module(self):
            if not lldb.target:
                return 'error: no target'
            if self.module:
                text_section = self.module.FindSection ("__TEXT")
                if text_section:
                    error = lldb.target.SetSectionLoadAddress (text_section, self.text_addr_lo)
                    if error.Success():
                        #print 'Success: loaded %s.__TEXT = 0x%x' % (self.get_resolved_path_basename(), self.text_addr_lo)
                        return None
                    else:
                        return 'error: %s' % error.GetCString()
                else:
                    return 'error: unable to find "__TEXT" section in "%s"' % self.get_resolved_path()
            else:
                return 'error: invalid module'
        
        def create_target(self):
            if self.fetch_symboled_executable_and_dsym ():
                resolved_path = self.get_resolved_path();
                path_spec = lldb.SBFileSpec (resolved_path)
                #result.PutCString ('plist[%s] = %s' % (uuid, self.plist))
                error = lldb.SBError()
                lldb.target = lldb.debugger.CreateTarget (resolved_path, self.arch, None, False, error);
                if lldb.target:
                    self.module = lldb.target.FindModule (path_spec)
                    if self.module:
                        err = self.load_module()
                        if err:
                            print err
                        else:
                            return None
                    else:
                        return 'error: unable to get module for (%s) "%s"' % (self.arch, resolved_path)
                else:
                    return 'error: unable to create target for (%s) "%s"' % (self.arch, resolved_path)
            else:
                return 'error: unable to locate main executable (%s) "%s"' % (self.arch, self.path)
        
        def add_target_module(self):
            if lldb.target:
                # Check for the module by UUID first in case it has been already loaded in LLDB
                self.module = lldb.target.AddModule (None, None, str(self.uuid))
                if not self.module:
                    if self.fetch_symboled_executable_and_dsym ():
                        resolved_path = self.get_resolved_path();
                        path_spec = lldb.SBFileSpec (resolved_path)
                        #print 'target.AddModule (path="%s", arch="%s", uuid=%s)' % (resolved_path, self.arch, self.uuid)
                        self.module = lldb.target.AddModule (resolved_path, self.arch, self.uuid)
                if self.module:
                    err = self.load_module()
                    if err:
                        print err;
                    else:
                        return None
                else:
                    return 'error: unable to get module for (%s) "%s"' % (self.arch, resolved_path)
            else:
                return 'error: invalid target'
        
    def __init__(self, path):
        """CrashLog constructor that take a path to a darwin crash log file"""
        self.path = os.path.expanduser(path);
        self.info_lines = list()
        self.system_profile = list()
        self.threads = list()
        self.images = list()
        self.idents = list() # A list of the required identifiers for doing all stack backtraces
        self.crashed_thread_idx = -1
        self.version = -1
        self.error = None
        # With possible initial component of ~ or ~user replaced by that user's home directory.
        try:
            f = open(self.path)
        except IOError:
            self.error = 'error: cannot open "%s"' % self.path
            return

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
                    ident = frame_match.group(2)
                    if not ident in self.idents:
                        self.idents.append(ident)
                    thread.frames.append (CrashLog.Frame(int(frame_match.group(1)), int(frame_match.group(3), 0), frame_match.group(4)))
                else:
                    print 'error: frame regex failed for line: "%s"' % line
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
    
    def find_image_with_identifier(self, ident):
        for image in self.images:
            if image.ident == ident:
                return image
        return None
    
    def create_target(self):
        if not self.images:
            return 'error: no images in crash log'
        exe_path = self.images[0].get_resolved_path()
        err = self.images[0].create_target ()
        if not err:
            return None # success
        # We weren't able to open the main executable as, but we can still symbolicate
        if self.idents:
            for ident in self.idents:
                image = self.find_image_with_identifier (ident)
                if image:
                    err = image.create_target ()
                    if not err:
                        return None # success
        for image in self.images:
            err = image.create_target ()
            if not err:
                return None # success
        return 'error: unable to locate any executables from the crash log'

def disassemble_instructions (instructions, pc, options, non_zeroeth_frame):
    lines = list()
    pc_index = -1
    comment_column = 50
    for inst_idx, inst in enumerate(instructions):
        inst_pc = inst.GetAddress().GetLoadAddress(lldb.target);
        if pc == inst_pc:
            pc_index = inst_idx
        mnemonic = inst.GetMnemonic (lldb.target)
        operands =  inst.GetOperands (lldb.target)
        comment =  inst.GetComment (lldb.target)
        #data = inst.GetData (lldb.target)
        lines.append ("%#16.16x: %8s %s" % (inst_pc, mnemonic, operands))
        if comment:
            line_len = len(lines[-1])
            if line_len < comment_column:
                lines[-1] += ' ' * (comment_column - line_len)
                lines[-1] += "; %s" % comment

    if pc_index >= 0:
        # If we are disassembling the non-zeroeth frame, we need to backup the PC by 1
        if non_zeroeth_frame and pc_index > 0:
            pc_index = pc_index - 1
        if options.disassemble_before == -1:
            start_idx = 0
        else:
            start_idx = pc_index - options.disassemble_before
        if start_idx < 0:
            start_idx = 0
        if options.disassemble_before == -1:
            end_idx = inst_idx
        else:
            end_idx = pc_index + options.disassemble_after
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
    for sect in module.section_iter():
        print_module_section (sect, depth)

def print_module_symbols (module):
    for sym in module:
        print sym

def usage():
    print "Usage: lldb-symbolicate.py [-n name] executable-image"
    sys.exit(0)

def Symbolicate(debugger, command, result, dict):
    try:
        SymbolicateCrashLog (shlex.split(command))
    except:
        result.PutCString ("error: python exception %s" % sys.exc_info()[0])
                
def SymbolicateCrashLog(command_args):
    usage = "usage: %prog [options] <FILE> [FILE ...]"
    description='''Symbolicate one or more darwin crash log files to provide source file and line information,
inlined stack frames back to the concrete functions, and disassemble the location of the crash
for the first frame of the crashed thread.
If this script is imported into the LLDB command interpreter, a "crashlog" command will be added to the interpreter
for use at the LLDB command line. After a crash log has been parsed and symbolicated, a target will have been
created that has all of the shared libraries loaded at the load addresses found in the crash log file. This allows
you to explore the program as if it were stopped at the locations described in the crash log and functions can 
be disassembled and lookups can be performed using the addresses found in the crash log.'''
    parser = optparse.OptionParser(description=description, prog='crashlog.py',usage=usage)
    parser.add_option('--platform', type='string', metavar='platform', dest='platform', help='specify one platform by name')
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    parser.add_option('--no-images', action='store_false', dest='show_images', help='don\'t show images in stack frames', default=True)
    parser.add_option('-a', '--load-all', action='store_true', dest='load_all_images', help='load all executable images, not just the images found in the crashed stack frames', default=False)
    parser.add_option('--image-list', action='store_true', dest='dump_image_list', help='show image list', default=False)
    parser.add_option('-g', '--debug-delay', type='int', dest='debug_delay', metavar='NSEC', help='pause for NSEC seconds for debugger', default=0)
    parser.add_option('-c', '--crashed-only', action='store_true', dest='crashed_only', help='only symbolicate the crashed thread', default=False)
    parser.add_option('-d', '--disasm-depth', type='int', dest='disassemble_depth', help='set the depth in stack frames that should be disassembled (default is 1)', default=1)
    parser.add_option('-D', '--disasm-all', action='store_true', dest='disassemble_all_threads', help='enabled disassembly of frames on all threads (not just the crashed thread)', default=False)
    parser.add_option('-B', '--disasm-before', type='int', dest='disassemble_before', help='the number of instructions to disassemble before the frame PC', default=4)
    parser.add_option('-A', '--disasm-after', type='int', dest='disassemble_after', help='the number of instructions to disassemble after the frame PC', default=4)
    loaded_addresses = False
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return
        
    if options.verbose:
        print 'command_args = %s' % command_args
        print 'options', options
        print 'args', args
        
    if options.debug_delay > 0:
        print "Waiting %u seconds for debugger to attach..." % options.debug_delay
        time.sleep(options.debug_delay)
    error = lldb.SBError()
    if args:
        for crash_log_file in args:
            crash_log = CrashLog(crash_log_file)
            if crash_log.error:
                print crash_log.error
                return
            if options.verbose:
                crash_log.dump()
            if not crash_log.images:
                print 'error: no images in crash log'
                return

            err = crash_log.create_target ()
            if err:
                print err
                return
                
            exe_module = lldb.target.GetModuleAtIndex(0)
            images_to_load = list()
            loaded_image_paths = list()
            if options.load_all_images:
                # --load-all option was specified, load everything up
                for image in crash_log.images:
                    images_to_load.append(image)
            else:
                # Only load the images found in stack frames for the crashed threads
                for ident in crash_log.idents:
                    image = crash_log.find_image_with_identifier (ident)
                    if image:
                        images_to_load.append(image)
                    else:
                        print 'error: can\'t find image for identifier "%s"' % ident
            
            for image in images_to_load:
                if image.path in loaded_image_paths:
                    print "warning: skipping %s loaded at %#16.16x duplicate entry (probably commpage)" % (image.path, image.text_addr_lo)
                else:
                    err = image.add_target_module ()
                    if err:
                        print err
                    else:
                        loaded_image_paths.append(image.path)
            
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
                    pc_addr = lldb.target.ResolveLoadAddress (frame.pc)
                    # Check to see if we were able to resolve the address
                    if pc_addr:
                        # We were able to resolve the frame's PC into a section offset
                        # address.

                        # Resolve the frame's PC value into a symbol context. A symbol
                        # context can resolve a module, compile unit, function, block,
                        # line table entry and/or symbol. If the frame has a block, then
                        # we can look for inlined frames, which are represented by blocks
                        # that have inlined information in them
                        frame.sym_ctx = lldb.target.ResolveSymbolContextForAddress (pc_addr, lldb.eSymbolContextEverything);

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
                            new_frame.pc = parent_pc_addr.GetLoadAddress(lldb.target)
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
                        disassemble = (this_thread_crashed or options.disassemble_all_threads) and frame_idx < options.disassemble_depth;
                        if inlined_block:
                            function_name = inlined_block.GetInlinedName();
                            block_range_idx = inlined_block.GetRangeIndexForBlockAddress (lldb.target.ResolveLoadAddress (frame.pc))
                            if block_range_idx < lldb.UINT32_MAX:
                                block_range_start_addr = inlined_block.GetRangeStartAddress (block_range_idx)
                                function_start_load_addr = block_range_start_addr.GetLoadAddress (lldb.target)
                            else:
                                function_start_load_addr = frame.pc
                            if disassemble:
                                instructions = function.GetInstructions(lldb.target)
                        elif function:
                            function_name = function.GetName()
                            function_start_load_addr = function.GetStartAddress().GetLoadAddress (lldb.target)
                            if disassemble:
                                instructions = function.GetInstructions(lldb.target)
                        elif symbol:
                            function_name = symbol.GetName()
                            function_start_load_addr = symbol.GetStartAddress().GetLoadAddress (lldb.target)
                            if disassemble:
                                instructions = symbol.GetInstructions(lldb.target)

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
                        disassemble_instructions (instructions, frame.pc, options, frame.index > 0)
                        print

                print                

            if options.dump_image_list:
                print "Binary Images:"
                for image in crash_log.images:
                    print image


if __name__ == '__main__':
    # Create a new debugger instance
    lldb.debugger = lldb.SBDebugger.Create()
    SymbolicateCrashLog (sys.argv)
elif lldb.debugger:
    lldb.debugger.HandleCommand('command script add -f crashlog.Symbolicate crashlog')
    print '"crashlog" command installed, type "crashlog --help" for detailed help'

