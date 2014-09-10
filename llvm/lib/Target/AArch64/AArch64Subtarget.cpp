//===-- AArch64Subtarget.cpp - AArch64 Subtarget Information ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-subtarget"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "AArch64GenSubtargetInfo.inc"

static cl::opt<bool>
EnableEarlyIfConvert("aarch64-early-ifcvt", cl::desc("Enable the early if "
                     "converter pass"), cl::init(true), cl::Hidden);

AArch64Subtarget &
AArch64Subtarget::initializeSubtargetDependencies(StringRef FS) {
  // Determine default and user-specified characteristics

  if (CPUString.empty())
    CPUString = "generic";

  ParseSubtargetFeatures(CPUString, FS);
  return *this;
}

AArch64Subtarget::AArch64Subtarget(const std::string &TT,
                                   const std::string &CPU,
                                   const std::string &FS, TargetMachine &TM,
                                   bool LittleEndian)
    : AArch64GenSubtargetInfo(TT, CPU, FS), ARMProcFamily(Others),
      HasFPARMv8(false), HasNEON(false), HasCrypto(false), HasCRC(false),
      HasZeroCycleRegMove(false), HasZeroCycleZeroing(false), CPUString(CPU),
      TargetTriple(TT),
      // This nested ternary is horrible, but DL needs to be properly
      // initialized
      // before TLInfo is constructed.
      DL(isTargetMachO()
             ? "e-m:o-i64:64-i128:128-n32:64-S128"
             : (LittleEndian ? "e-m:e-i64:64-i128:128-n32:64-S128"
                             : "E-m:e-i64:64-i128:128-n32:64-S128")),
      FrameLowering(), InstrInfo(initializeSubtargetDependencies(FS)),
      TSInfo(&DL), TLInfo(TM) {}

/// ClassifyGlobalReference - Find the target operand flags that describe
/// how a global value should be referenced for the current subtarget.
unsigned char
AArch64Subtarget::ClassifyGlobalReference(const GlobalValue *GV,
                                        const TargetMachine &TM) const {

  // Determine whether this is a reference to a definition or a declaration.
  // Materializable GVs (in JIT lazy compilation mode) do not require an extra
  // load from stub.
  bool isDecl = GV->hasAvailableExternallyLinkage();
  if (GV->isDeclaration() && !GV->isMaterializable())
    isDecl = true;

  // MachO large model always goes via a GOT, simply to get a single 8-byte
  // absolute relocation on all global addresses.
  if (TM.getCodeModel() == CodeModel::Large && isTargetMachO())
    return AArch64II::MO_GOT;

  // The small code mode's direct accesses use ADRP, which cannot necessarily
  // produce the value 0 (if the code is above 4GB).
  if (TM.getCodeModel() == CodeModel::Small &&
      GV->isWeakForLinker() && isDecl) {
    // In PIC mode use the GOT, but in absolute mode use a constant pool load.
    if (TM.getRelocationModel() == Reloc::Static)
        return AArch64II::MO_CONSTPOOL;
    else
        return AArch64II::MO_GOT;
  }

  // If symbol visibility is hidden, the extra load is not needed if
  // the symbol is definitely defined in the current translation unit.

  // The handling of non-hidden symbols in PIC mode is rather target-dependent:
  //   + On MachO, if the symbol is defined in this module the GOT can be
  //     skipped.
  //   + On ELF, the R_AARCH64_COPY relocation means that even symbols actually
  //     defined could end up in unexpected places. Use a GOT.
  if (TM.getRelocationModel() != Reloc::Static && GV->hasDefaultVisibility()) {
    if (isTargetMachO())
      return (isDecl || GV->isWeakForLinker()) ? AArch64II::MO_GOT
                                               : AArch64II::MO_NO_FLAG;
    else
      // No need to go through the GOT for local symbols on ELF.
      return GV->hasLocalLinkage() ? AArch64II::MO_NO_FLAG : AArch64II::MO_GOT;
  }

  return AArch64II::MO_NO_FLAG;
}

/// This function returns the name of a function which has an interface
/// like the non-standard bzero function, if such a function exists on
/// the current subtarget and it is considered prefereable over
/// memset with zero passed as the second argument. Otherwise it
/// returns null.
const char *AArch64Subtarget::getBZeroEntry() const {
  // Prefer bzero on Darwin only.
  if(isTargetDarwin())
    return "bzero";

  return nullptr;
}

void AArch64Subtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                         MachineInstr *begin, MachineInstr *end,
                                         unsigned NumRegionInstrs) const {
  // LNT run (at least on Cyclone) showed reasonably significant gains for
  // bi-directional scheduling. 253.perlbmk.
  Policy.OnlyTopDown = false;
  Policy.OnlyBottomUp = false;
}

bool AArch64Subtarget::enableEarlyIfConversion() const {
  return EnableEarlyIfConvert;
}
