#!/usr/bin/python

import argparse, re, subprocess, sys

#-- This code fragment loads LLDB --

def AddLLDBToSysPathOnMacOSX():
    def GetLLDBFrameworkPath():
        lldb_path = subprocess.check_output(["xcrun", "-find", "lldb"])
        re_result = re.match("(.*)/Developer/usr/bin/lldb", lldb_path)
        if re_result == None:
            return None
        xcode_contents_path = re_result.group(1)
        return xcode_contents_path + "/SharedFrameworks/LLDB.framework"
    
    lldb_framework_path = GetLLDBFrameworkPath()
    
    if lldb_framework_path == None:
        print "Couldn't find LLDB.framework"
        sys.exit(-1)
    
    sys.path.append(lldb_framework_path + "/Resources/Python")

AddLLDBToSysPathOnMacOSX()

import lldb

#-- End of code fragment that loads LLDB --

parser = argparse.ArgumentParser(description="Run an exhaustive test of the LLDB disassembler for a specific architecture.")

parser.add_argument('--arch', required=True, action='store', help='The architecture whose disassembler is to be tested')
parser.add_argument('--bytes', required=True, action='store', type=int, help='The byte width of instructions for that architecture')
parser.add_argument('--random', required=False, action='store_true', help='Enables non-sequential testing')
parser.add_argument('--start', required=False, action='store', type=int, help='The first instruction value to test')
parser.add_argument('--skip', required=False, action='store', type=int, help='The interval between instructions to test')

arguments = sys.argv[1:]

arg_ns = parser.parse_args(arguments)

debugger = lldb.SBDebugger.Create()

if debugger.IsValid() == False:
    print "Couldn't create an SBDebugger"
    sys.exit(-1)

target = debugger.CreateTargetWithFileAndArch(None, arg_ns.arch)

if target.IsValid() == False:
    print "Couldn't create an SBTarget for architecture " + arg_ns.arch
    sys.exit(-1)

class SequentialInstructionProvider:
    def __init__(self, byte_width, start=0, skip=1):
        self.m_byte_width = byte_width
        self.m_start = start
        self.m_skip = skip
        self.m_value = start
        self.m_last = (1 << (byte_width * 8)) - 1
    def GetNextInstruction(self):
        # Print current state
        print self.m_value
        # Done printing current state
        if self.m_value > self.m_last:
            return None
        ret = bytearray(self.m_byte_width)
        for i in range(self.m_byte_width):
            ret[self.m_byte_width - (i + 1)] = (self.m_value >> (i * 8)) & 255 
        self.m_value += self.m_skip
        return ret
    def __iter__(self):
        return self
    def next(self):
        ret = self.GetNextInstruction()
        if ret == None:
            raise StopIteration
        return ret

class RandomInstructionProvider:
    def __init__(self, byte_width):
        self.m_byte_width = byte_width
        self.m_random_file = open("/dev/random", 'r')
    def GetNextInstruction(self):
        ret = bytearray(self.m_byte_width)
        for i in range(self.m_byte_width):
            ret[i] = self.m_random_file.read(1)
        # Print current state
        for ret_byte in ret:
            print hex(ret_byte) + " ",
        print
        # Done printing current state
        return ret
    def __iter__(self):
        return self
    def next(self):
        ret = self.GetNextInstruction()
        if ret == None:
            raise StopIteration
        return ret

def GetProviderWithArguments(args):
    instruction_provider = None
    if args.random == True:
        instruction_provider = RandomInstructionProvider(args.bytes)
    else:
        start = 0
        skip = 1
        if args.start != None:
            start = args.start
        if args.skip != None:
            skip = args.skip
        instruction_provider = SequentialInstructionProvider(args.bytes, start, skip)
    return instruction_provider

instruction_provider = GetProviderWithArguments(arg_ns)

fake_address = lldb.SBAddress()

for inst_bytes in instruction_provider:
    inst_list = target.GetInstructions(fake_address, inst_bytes)
    if not inst_list.IsValid():
        print "Invalid instruction list"
        continue
    inst = inst_list.GetInstructionAtIndex(0)
    if not inst.IsValid():
        print "Invalid instruction"
        continue
    instr_output_stream = lldb.SBStream()
    inst.GetDescription(instr_output_stream)
    print instr_output_stream.GetData()
