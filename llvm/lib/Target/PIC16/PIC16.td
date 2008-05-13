//===- PIC16.td - Describe the PIC16 Target Machine -----------*- tblgen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the PIC16 target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "../Target.td"

//===----------------------------------------------------------------------===//
// Descriptions
//===----------------------------------------------------------------------===//

include "PIC16RegisterInfo.td"
include "PIC16CallingConv.td"
include "PIC16InstrInfo.td"

def PIC16InstrInfo : InstrInfo {
  let TSFlagsFields = [];
  let TSFlagsShifts = [];
}



// Not currently supported, but work as SubtargetFeature placeholder.
def FeaturePIC16Old : SubtargetFeature<"pic16old", "IsPIC16Old", "true",
                                      "PIC16 Old ISA Support">;

//===----------------------------------------------------------------------===//
// PIC16 processors supported.
//===----------------------------------------------------------------------===//

def : Processor<"generic", NoItineraries, []>;

def PIC16 : Target {
  let InstructionSet = PIC16InstrInfo;
}

