//===-- Hexagon.td - Describe the Hexagon Target Machine --*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the top level entry point for the Hexagon target.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Hexagon Subtarget features.
//===----------------------------------------------------------------------===//

// Hexagon Archtectures
def ArchV2       : SubtargetFeature<"v2", "HexagonArchVersion", "V2",
                                    "Hexagon v2">;
def ArchV3       : SubtargetFeature<"v3", "HexagonArchVersion", "V3",
                                    "Hexagon v3">;
def ArchV4       : SubtargetFeature<"v4", "HexagonArchVersion", "V4",
                                    "Hexagon v4">;
def ArchV5       : SubtargetFeature<"v5", "HexagonArchVersion", "V5",
                                    "Hexagon v5">;

//===----------------------------------------------------------------------===//
// Hexagon Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//
def HasV2T                      : Predicate<"Subtarget.hasV2TOps()">;
def HasV2TOnly                  : Predicate<"Subtarget.hasV2TOpsOnly()">;
def NoV2T                       : Predicate<"!Subtarget.hasV2TOps()">;
def HasV3T                      : Predicate<"Subtarget.hasV3TOps()">;
def HasV3TOnly                  : Predicate<"Subtarget.hasV3TOpsOnly()">;
def NoV3T                       : Predicate<"!Subtarget.hasV3TOps()">;
def HasV4T                      : Predicate<"Subtarget.hasV4TOps()">;
def NoV4T                       : Predicate<"!Subtarget.hasV4TOps()">;
def HasV5T                      : Predicate<"Subtarget.hasV5TOps()">;
def NoV5T                       : Predicate<"!Subtarget.hasV5TOps()">;
def UseMEMOP                    : Predicate<"Subtarget.useMemOps()">;
def IEEERndNearV5T              : Predicate<"Subtarget.modeIEEERndNear()">;

//===----------------------------------------------------------------------===//
// Classes used for relation maps.
//===----------------------------------------------------------------------===//

class ImmRegShl;
// PredRel - Filter class used to relate non-predicated instructions with their
// predicated forms.
class PredRel;
// PredNewRel - Filter class used to relate predicated instructions with their
// predicate-new forms.
class PredNewRel: PredRel;
// ImmRegRel - Filter class used to relate instructions having reg-reg form
// with their reg-imm counterparts.
class ImmRegRel;
// NewValueRel - Filter class used to relate regular store instructions with
// their new-value store form.
class NewValueRel: PredNewRel;
// NewValueRel - Filter class used to relate load/store instructions having
// different addressing modes with each other.
class AddrModeRel: NewValueRel;

//===----------------------------------------------------------------------===//
// Generate mapping table to relate non-predicate instructions with their
// predicated formats - true and false.
//

def getPredOpcode : InstrMapping {
  let FilterClass = "PredRel";
  // Instructions with the same BaseOpcode and isNVStore values form a row.
  let RowFields = ["BaseOpcode", "isNVStore", "PNewValue"];
  // Instructions with the same predicate sense form a column.
  let ColFields = ["PredSense"];
  // The key column is the unpredicated instructions.
  let KeyCol = [""];
  // Value columns are PredSense=true and PredSense=false
  let ValueCols = [["true"], ["false"]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate predicate-true instructions with their
// predicate-false forms
//
def getFalsePredOpcode : InstrMapping {
  let FilterClass = "PredRel";
  let RowFields = ["BaseOpcode", "PNewValue", "isNVStore", "isBrTaken"];
  let ColFields = ["PredSense"];
  let KeyCol = ["true"];
  let ValueCols = [["false"]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate predicate-false instructions with their
// predicate-true forms
//
def getTruePredOpcode : InstrMapping {
  let FilterClass = "PredRel";
  let RowFields = ["BaseOpcode", "PNewValue", "isNVStore", "isBrTaken"];
  let ColFields = ["PredSense"];
  let KeyCol = ["false"];
  let ValueCols = [["true"]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate predicated instructions with their .new
// format.
//
def getPredNewOpcode : InstrMapping {
  let FilterClass = "PredNewRel";
  let RowFields = ["BaseOpcode", "PredSense", "isNVStore", "isBrTaken"];
  let ColFields = ["PNewValue"];
  let KeyCol = [""];
  let ValueCols = [["new"]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate .new predicated instructions with their old
// format.
//
def getPredOldOpcode : InstrMapping {
  let FilterClass = "PredNewRel";
  let RowFields = ["BaseOpcode", "PredSense", "isNVStore"];
  let ColFields = ["PNewValue"];
  let KeyCol = ["new"];
  let ValueCols = [[""]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate store instructions with their new-value
// format.
//
def getNewValueOpcode : InstrMapping {
  let FilterClass = "NewValueRel";
  let RowFields = ["BaseOpcode", "PredSense", "PNewValue", "addrMode"];
  let ColFields = ["NValueST"];
  let KeyCol = ["false"];
  let ValueCols = [["true"]];
}

//===----------------------------------------------------------------------===//
// Generate mapping table to relate new-value store instructions with their old
// format.
//
def getNonNVStore : InstrMapping {
  let FilterClass = "NewValueRel";
  let RowFields = ["BaseOpcode", "PredSense", "PNewValue", "addrMode"];
  let ColFields = ["NValueST"];
  let KeyCol = ["true"];
  let ValueCols = [["false"]];
}

def getBasedWithImmOffset : InstrMapping {
  let FilterClass = "AddrModeRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore",
                   "isMEMri", "isFloat"];
  let ColFields = ["addrMode"];
  let KeyCol = ["Absolute"];
  let ValueCols = [["BaseImmOffset"]];
}

def getBaseWithRegOffset : InstrMapping {
  let FilterClass = "AddrModeRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["addrMode"];
  let KeyCol = ["BaseImmOffset"];
  let ValueCols = [["BaseRegOffset"]];
}

def getRegForm : InstrMapping {
  let FilterClass = "ImmRegRel";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue"];
  let ColFields = ["InputType"];
  let KeyCol = ["imm"];
  let ValueCols = [["reg"]];
}

def getRegShlForm : InstrMapping {
  let FilterClass = "ImmRegShl";
  let RowFields = ["CextOpcode", "PredSense", "PNewValue", "isNVStore"];
  let ColFields = ["InputType"];
  let KeyCol = ["imm"];
  let ValueCols = [["reg"]];
}

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//
include "HexagonSchedule.td"
include "HexagonRegisterInfo.td"
include "HexagonCallingConv.td"
include "HexagonInstrInfo.td"
include "HexagonIntrinsics.td"
include "HexagonIntrinsicsDerived.td"

def HexagonInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Hexagon processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, SchedMachineModel Model,
           list<SubtargetFeature> Features>
 : ProcessorModel<Name, Model, Features>;

def : Proc<"hexagonv4", HexagonModelV4, [ArchV2, ArchV3, ArchV4]>;
def : Proc<"hexagonv5", HexagonModelV4, [ArchV2, ArchV3, ArchV4, ArchV5]>;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def Hexagon : Target {
  // Pull in Instruction Info:
  let InstructionSet = HexagonInstrInfo;
}
