//===-- MipsSubtarget.cpp - Mips Subtarget Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Mips specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-subtarget"

#include "MipsMachineFunction.h"
#include "MipsSubtarget.h"
#include "MipsTargetMachine.h"
#include "Mips.h"
#include "MipsRegisterInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "MipsGenSubtargetInfo.inc"


using namespace llvm;

// FIXME: Maybe this should be on by default when Mips16 is specified
//
static cl::opt<bool> Mixed16_32(
  "mips-mixed-16-32",
  cl::init(false),
  cl::desc("Allow for a mixture of Mips16 "
           "and Mips32 code in a single source file"),
  cl::Hidden);

static cl::opt<bool> Mips_Os16(
  "mips-os16",
  cl::init(false),
  cl::desc("Compile all functions that don' use "
           "floating point as Mips 16"),
  cl::Hidden);

static cl::opt<bool>
Mips16HardFloat("mips16-hard-float", cl::NotHidden,
                cl::desc("MIPS: mips16 hard float enable."),
                cl::init(false));

static cl::opt<bool>
Mips16ConstantIslands(
  "mips16-constant-islands", cl::Hidden,
  cl::desc("MIPS: mips16 constant islands enable. experimental feature"),
  cl::init(false));

void MipsSubtarget::anchor() { }

MipsSubtarget::MipsSubtarget(const std::string &TT, const std::string &CPU,
                             const std::string &FS, bool little,
                             Reloc::Model _RM, MipsTargetMachine *_TM) :
  MipsGenSubtargetInfo(TT, CPU, FS),
  MipsArchVersion(Mips32), MipsABI(UnknownABI), IsLittle(little),
  IsSingleFloat(false), IsFP64bit(false), IsGP64bit(false), HasVFPU(false),
  IsLinux(true), HasSEInReg(false), HasCondMov(false), HasSwap(false),
  HasBitCount(false), HasFPIdx(false),
  InMips16Mode(false), InMips16HardFloat(Mips16HardFloat),
  InMicroMipsMode(false), HasDSP(false), HasDSPR2(false),
  AllowMixed16_32(Mixed16_32 | Mips_Os16), Os16(Mips_Os16), HasMSA(false),
  RM(_RM), OverrideMode(NoOverride), TM(_TM)
{
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "mips32";

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);

  PreviousInMips16Mode = InMips16Mode;

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  // Set MipsABI if it hasn't been set yet.
  if (MipsABI == UnknownABI)
    MipsABI = hasMips64() ? N64 : O32;

  // Check if Architecture and ABI are compatible.
  assert(((!hasMips64() && (isABI_O32() || isABI_EABI())) ||
          (hasMips64() && (isABI_N32() || isABI_N64()))) &&
         "Invalid  Arch & ABI pair.");

  if (hasMSA() && !isFP64bit())
    report_fatal_error("MSA requires a 64-bit FPU register file (FR=1 mode). "
                       "See -mattr=+fp64.",
                       false);

  // Is the target system Linux ?
  if (TT.find("linux") == std::string::npos)
    IsLinux = false;

  // Set UseSmallSection.
  UseSmallSection = !IsLinux && (RM == Reloc::Static);
  // set some subtarget specific features
  if (inMips16Mode())
    HasBitCount=false;
}

bool
MipsSubtarget::enablePostRAScheduler(CodeGenOpt::Level OptLevel,
                                    TargetSubtargetInfo::AntiDepBreakMode &Mode,
                                     RegClassVector &CriticalPathRCs) const {
  Mode = TargetSubtargetInfo::ANTIDEP_NONE;
  CriticalPathRCs.clear();
  CriticalPathRCs.push_back(hasMips64() ?
                            &Mips::GPR64RegClass : &Mips::GPR32RegClass);
  return OptLevel >= CodeGenOpt::Aggressive;
}

//FIXME: This logic for reseting the subtarget along with
// the helper classes can probably be simplified but there are a lot of
// cases so we will defer rewriting this to later.
//
void MipsSubtarget::resetSubtarget(MachineFunction *MF) {
  bool ChangeToMips16 = false, ChangeToNoMips16 = false;
  DEBUG(dbgs() << "resetSubtargetFeatures" << "\n");
  AttributeSet FnAttrs = MF->getFunction()->getAttributes();
  ChangeToMips16 = FnAttrs.hasAttribute(AttributeSet::FunctionIndex,
                                        "mips16");
  ChangeToNoMips16 = FnAttrs.hasAttribute(AttributeSet::FunctionIndex,
                                        "nomips16");
  assert (!(ChangeToMips16 & ChangeToNoMips16) &&
          "mips16 and nomips16 specified on the same function");
  if (ChangeToMips16) {
    if (PreviousInMips16Mode)
      return;
    OverrideMode = Mips16Override;
    PreviousInMips16Mode = true;
    TM->setHelperClassesMips16();
    return;
  } else if (ChangeToNoMips16) {
    if (!PreviousInMips16Mode)
      return;
    OverrideMode = NoMips16Override;
    PreviousInMips16Mode = false;
    TM->setHelperClassesMipsSE();
    return;
  } else {
    if (OverrideMode == NoOverride)
      return;
    OverrideMode = NoOverride;
    DEBUG(dbgs() << "back to default" << "\n");
    if (inMips16Mode() && !PreviousInMips16Mode) {
      TM->setHelperClassesMips16();
      PreviousInMips16Mode = true;
    } else if (!inMips16Mode() && PreviousInMips16Mode) {
      TM->setHelperClassesMipsSE();
      PreviousInMips16Mode = false;
    }
    return;
  }
}

bool MipsSubtarget::mipsSEUsesSoftFloat() const {
  return TM->Options.UseSoftFloat && !InMips16HardFloat;
}

bool MipsSubtarget::useConstantIslands() {
  DEBUG(dbgs() << "use constant islands " << Mips16ConstantIslands << "\n");
  return Mips16ConstantIslands;
}
