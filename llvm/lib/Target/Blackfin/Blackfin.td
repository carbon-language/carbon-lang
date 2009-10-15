//===- Blackfin.td - Describe the Blackfin Target Machine --*- tablegen -*-===//
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
// Blackfin Subtarget features.
//===----------------------------------------------------------------------===//

def FeatureSDRAM : SubtargetFeature<"sdram", "sdram", "true",
    "Build for SDRAM">;

def FeatureICPLB : SubtargetFeature<"icplb", "icplb", "true",
    "Assume instruction cache lookaside buffers are enabled at runtime">;

//===----------------------------------------------------------------------===//
// Bugs in the silicon becomes workarounds in the compiler.
// See http://www.analog.com/ for the full list of IC anomalies.
//===----------------------------------------------------------------------===//

def WA_MI_SHIFT : SubtargetFeature<"mi-shift-anomaly","wa_mi_shift", "true",
    "Work around 05000074 - "
    "Multi-Issue Instruction with dsp32shiftimm and P-reg Store">;

def WA_CSYNC : SubtargetFeature<"csync-anomaly","wa_csync", "true",
    "Work around 05000244 - "
    "If I-Cache Is On, CSYNC/SSYNC/IDLE Around Change of Control">;

def WA_SPECLD : SubtargetFeature<"specld-anomaly","wa_specld", "true",
    "Work around 05000245 - "
    "Access in the Shadow of a Conditional Branch">;

def WA_HWLOOP : SubtargetFeature<"hwloop-anomaly","wa_hwloop", "true",
    "Work around 05000257 - "
    "Interrupt/Exception During Short Hardware Loop">;

def WA_MMR_STALL : SubtargetFeature<"mmr-stall-anomaly","wa_mmr_stall", "true",
    "Work around 05000283 - "
    "System MMR Write Is Stalled Indefinitely when Killed">;

def WA_LCREGS : SubtargetFeature<"lcregs-anomaly","wa_lcregs", "true",
    "Work around 05000312 - "
    "SSYNC, CSYNC, or Loads to LT, LB and LC Registers Are Interrupted">;

def WA_KILLED_MMR : SubtargetFeature<"killed-mmr-anomaly",
                                     "wa_killed_mmr", "true",
    "Work around 05000315 - "
    "Killed System MMR Write Completes Erroneously on Next System MMR Access">;

def WA_RETS : SubtargetFeature<"rets-anomaly", "wa_rets", "true",
    "Work around 05000371 - "
    "Possible RETS Register Corruption when Subroutine Is under 5 Cycles">;

def WA_IND_CALL : SubtargetFeature<"ind-call-anomaly", "wa_ind_call", "true",
    "Work around 05000426 - "
    "Speculative Fetches of Indirect-Pointer Instructions">;

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "BlackfinRegisterInfo.td"
include "BlackfinCallingConv.td"
include "BlackfinIntrinsics.td"
include "BlackfinInstrInfo.td"

def BlackfinInstrInfo : InstrInfo {}

//===----------------------------------------------------------------------===//
// Blackfin processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, string Suffix, list<SubtargetFeature> Features>
 : Processor<!strconcat(Name, Suffix), NoItineraries, Features>;

def : Proc<"generic", "", []>;

multiclass Core<string Name,string Suffix,
                list<SubtargetFeature> Features> {
  def : Proc<Name, Suffix, Features>;
  def : Proc<Name, "", Features>;
  def : Proc<Name, "-none", []>;
}

multiclass CoreEdinburgh<string Name>
      : Core<Name, "-0.6", [WA_MI_SHIFT, WA_SPECLD, WA_LCREGS]> {
  def : Proc<Name, "-0.5",
        [WA_MI_SHIFT, WA_SPECLD, WA_MMR_STALL, WA_LCREGS, WA_KILLED_MMR,
         WA_RETS]>;
  def : Proc<Name, "-0.4",
        [WA_MI_SHIFT, WA_CSYNC, WA_SPECLD, WA_HWLOOP, WA_MMR_STALL, WA_LCREGS,
         WA_KILLED_MMR, WA_RETS]>;
  def : Proc<Name, "-0.3",
        [WA_MI_SHIFT, WA_CSYNC, WA_SPECLD, WA_HWLOOP, WA_MMR_STALL, WA_LCREGS,
         WA_KILLED_MMR, WA_RETS]>;
  def : Proc<Name, "-any",
        [WA_MI_SHIFT, WA_CSYNC, WA_SPECLD, WA_HWLOOP, WA_MMR_STALL, WA_LCREGS,
         WA_KILLED_MMR, WA_RETS]>;
}
multiclass CoreBraemar<string Name>
       : Core<Name, "-0.3",
         [WA_MI_SHIFT, WA_SPECLD, WA_LCREGS, WA_RETS, WA_IND_CALL]> {
  def  : Proc<Name, "-0.2",
         [WA_MI_SHIFT, WA_CSYNC, WA_SPECLD, WA_HWLOOP, WA_MMR_STALL, WA_LCREGS,
          WA_KILLED_MMR, WA_RETS, WA_IND_CALL]>;
  def  : Proc<Name, "-any",
         [WA_MI_SHIFT, WA_CSYNC, WA_SPECLD, WA_HWLOOP, WA_MMR_STALL, WA_LCREGS,
          WA_KILLED_MMR, WA_RETS, WA_IND_CALL]>;
}
multiclass CoreStirling<string Name>
      : Core<Name, "-0.5", [WA_MI_SHIFT, WA_SPECLD, WA_IND_CALL]> {
  def : Proc<Name, "-0.4",
        [WA_MI_SHIFT, WA_SPECLD, WA_LCREGS, WA_RETS, WA_IND_CALL]>;
  def : Proc<Name, "-0.3",
        [WA_MI_SHIFT, WA_SPECLD, WA_MMR_STALL, WA_LCREGS, WA_KILLED_MMR,
         WA_RETS, WA_IND_CALL]>;
  def : Proc<Name, "-any",
        [WA_MI_SHIFT, WA_SPECLD, WA_MMR_STALL, WA_LCREGS, WA_KILLED_MMR,
         WA_RETS, WA_IND_CALL]>;
}
multiclass CoreMoab<string Name>
      : Core<Name, "-0.3", [WA_MI_SHIFT, WA_SPECLD, WA_IND_CALL]> {
  def : Proc<Name, "-0.2", [WA_MI_SHIFT, WA_SPECLD, WA_IND_CALL]>;
  def : Proc<Name, "-0.1", [WA_MI_SHIFT, WA_SPECLD, WA_RETS, WA_IND_CALL]>;
  def : Proc<Name, "-0.0",
        [WA_MI_SHIFT, WA_SPECLD, WA_LCREGS, WA_RETS, WA_IND_CALL]>;
  def : Proc<Name, "-any",
        [WA_MI_SHIFT, WA_SPECLD, WA_LCREGS, WA_RETS, WA_IND_CALL]>;
}
multiclass CoreTeton<string Name>
      : Core<Name, "-0.5",
        [WA_MI_SHIFT, WA_SPECLD, WA_MMR_STALL, WA_LCREGS, WA_KILLED_MMR,
         WA_RETS, WA_IND_CALL]> {
  def : Proc<Name, "-0.3",
        [WA_MI_SHIFT, WA_CSYNC, WA_SPECLD, WA_HWLOOP, WA_MMR_STALL, WA_LCREGS,
         WA_KILLED_MMR, WA_RETS, WA_IND_CALL]>;
  def : Proc<Name, "-any",
        [WA_MI_SHIFT, WA_CSYNC, WA_SPECLD, WA_HWLOOP, WA_MMR_STALL, WA_LCREGS,
         WA_KILLED_MMR, WA_RETS, WA_IND_CALL]>;
}
multiclass CoreKookaburra<string Name>
      : Core<Name, "-0.2", [WA_MI_SHIFT, WA_SPECLD, WA_IND_CALL]> {
  def : Proc<Name, "-0.1", [WA_MI_SHIFT, WA_SPECLD, WA_RETS, WA_IND_CALL]>;
  def : Proc<Name, "-0.0", [WA_MI_SHIFT, WA_SPECLD, WA_RETS, WA_IND_CALL]>;
  def : Proc<Name, "-any", [WA_MI_SHIFT, WA_SPECLD, WA_RETS, WA_IND_CALL]>;
}
multiclass CoreMockingbird<string Name>
      : Core<Name, "-0.1", [WA_MI_SHIFT, WA_SPECLD, WA_IND_CALL]> {
  def : Proc<Name, "-0.0", [WA_MI_SHIFT, WA_SPECLD, WA_IND_CALL]>;
  def : Proc<Name, "-any", [WA_MI_SHIFT, WA_SPECLD, WA_IND_CALL]>;
}
multiclass CoreBrodie<string Name>
      : Core<Name, "-0.1", [WA_MI_SHIFT, WA_SPECLD, WA_IND_CALL]> {
  def : Proc<Name, "-0.0", [WA_MI_SHIFT, WA_SPECLD, WA_IND_CALL]>;
  def : Proc<Name, "-any", [WA_MI_SHIFT, WA_SPECLD, WA_IND_CALL]>;
}

defm BF512 : CoreBrodie<"bf512">;
defm BF514 : CoreBrodie<"bf514">;
defm BF516 : CoreBrodie<"bf516">;
defm BF518 : CoreBrodie<"bf518">;
defm BF522 : CoreMockingbird<"bf522">;
defm BF523 : CoreKookaburra<"bf523">;
defm BF524 : CoreMockingbird<"bf524">;
defm BF525 : CoreKookaburra<"bf525">;
defm BF526 : CoreMockingbird<"bf526">;
defm BF527 : CoreKookaburra<"bf527">;
defm BF531 : CoreEdinburgh<"bf531">;
defm BF532 : CoreEdinburgh<"bf532">;
defm BF533 : CoreEdinburgh<"bf533">;
defm BF534 : CoreBraemar<"bf534">;
defm BF536 : CoreBraemar<"bf536">;
defm BF537 : CoreBraemar<"bf537">;
defm BF538 : CoreStirling<"bf538">;
defm BF539 : CoreStirling<"bf539">;
defm BF542 : CoreMoab<"bf542">;
defm BF544 : CoreMoab<"bf544">;
defm BF548 : CoreMoab<"bf548">;
defm BF549 : CoreMoab<"bf549">;
defm BF561 : CoreTeton<"bf561">;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def Blackfin : Target {
  // Pull in Instruction Info:
  let InstructionSet = BlackfinInstrInfo;
}
