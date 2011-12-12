//===- Hexagon.td - Describe the Hexagon Target Machine ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Hexagon Subtarget features.
//


// Hexagon Archtectures
def ArchV2       : SubtargetFeature<"v2", "HexagonArchVersion", "V2",
                                    "Hexagon v2">;
def ArchV3       : SubtargetFeature<"v3", "HexagonArchVersion", "V3",
                                    "Hexagon v3">;
def ArchV4       : SubtargetFeature<"v4", "HexagonArchVersion", "V4",
                                    "Hexagon v4">;

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//
include "HexagonSchedule.td"
include "HexagonRegisterInfo.td"
include "HexagonCallingConv.td"
include "HexagonInstrInfo.td"
include "HexagonIntrinsics.td"
include "HexagonIntrinsicsDerived.td"


def HexagonInstrInfo : InstrInfo {
  // Define how we want to layout our target-specific information field.
}

//===----------------------------------------------------------------------===//
// Hexagon processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, ProcessorItineraries Itin,
           list<SubtargetFeature> Features>
 : Processor<Name, Itin, Features>;

def : Proc<"hexagonv2", HexagonItineraries,   [ArchV2]>;
def : Proc<"hexagonv3", HexagonItineraries,   [ArchV2, ArchV3]>;
def : Proc<"hexagonv4", HexagonItinerariesV4, [ArchV2, ArchV3, ArchV4]>;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def Hexagon : Target {
  // Pull in Instruction Info:
  let InstructionSet = HexagonInstrInfo;
}
