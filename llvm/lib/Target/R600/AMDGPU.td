//===-- AMDIL.td - AMDIL Tablegen files --*- tablegen -*-------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//

// Include AMDIL TD files
include "AMDILBase.td"


def AMDGPUInstrInfo : InstrInfo {
  let guessInstructionProperties = 1;
}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//
def AMDGPUAsmWriter : AsmWriter {
    string AsmWriterClassName = "InstPrinter";
    int Variant = 0;
    bit isMCAsmWriter = 1;
}

def AMDGPU : Target {
  // Pull in Instruction Info:
  let InstructionSet = AMDGPUInstrInfo;
  let AssemblyWriters = [AMDGPUAsmWriter];
}

// Include AMDGPU TD files
include "R600Schedule.td"
include "SISchedule.td"
include "Processors.td"
include "AMDGPUInstrInfo.td"
include "AMDGPUIntrinsics.td"
include "AMDGPURegisterInfo.td"
include "AMDGPUInstructions.td"
include "AMDGPUCallingConv.td"
