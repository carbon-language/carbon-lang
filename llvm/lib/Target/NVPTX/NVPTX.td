//===- NVPTX.td - Describe the NVPTX Target Machine -----------*- tblgen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the NVPTX target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

include "NVPTXRegisterInfo.td"
include "NVPTXInstrInfo.td"

//===----------------------------------------------------------------------===//
// Subtarget Features.
// - We use the SM version number instead of explicit feature table.
// - Need at least one feature to avoid generating zero sized array by
//   TableGen in NVPTXGenSubtarget.inc.
//===----------------------------------------------------------------------===//

// SM Versions
def SM20 : SubtargetFeature<"sm_20", "SmVersion", "20",
                            "Target SM 2.0">;
def SM21 : SubtargetFeature<"sm_21", "SmVersion", "21",
                            "Target SM 2.1">;
def SM30 : SubtargetFeature<"sm_30", "SmVersion", "30",
                            "Target SM 3.0">;
def SM35 : SubtargetFeature<"sm_35", "SmVersion", "35",
                            "Target SM 3.5">;
def SM50 : SubtargetFeature<"sm_50", "SmVersion", "50",
                            "Target SM 5.0">;

// PTX Versions
def PTX30 : SubtargetFeature<"ptx30", "PTXVersion", "30",
                             "Use PTX version 3.0">;
def PTX31 : SubtargetFeature<"ptx31", "PTXVersion", "31",
                             "Use PTX version 3.1">;
def PTX32 : SubtargetFeature<"ptx32", "PTXVersion", "32",
                             "Use PTX version 3.2">;
def PTX40 : SubtargetFeature<"ptx40", "PTXVersion", "40",
                             "Use PTX version 4.0">;

//===----------------------------------------------------------------------===//
// NVPTX supported processors.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"sm_20", [SM20]>;
def : Proc<"sm_21", [SM21]>;
def : Proc<"sm_30", [SM30]>;
def : Proc<"sm_35", [SM35]>;
def : Proc<"sm_50", [SM50]>;


def NVPTXInstrInfo : InstrInfo {
}

def NVPTX : Target {
  let InstructionSet = NVPTXInstrInfo;
}
