//===-- ARM64Subtarget.cpp - ARM64 Subtarget Information --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM64 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "ARM64InstrInfo.h"
#include "ARM64Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "ARM64GenSubtargetInfo.inc"

using namespace llvm;

ARM64Subtarget::ARM64Subtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS)
    : ARM64GenSubtargetInfo(TT, CPU, FS), HasZeroCycleRegMove(false),
      HasZeroCycleZeroing(false), CPUString(CPU), TargetTriple(TT) {
  // Determine default and user-specified characteristics

  if (CPUString.empty())
    // We default to Cyclone for now.
    CPUString = "cyclone";

  ParseSubtargetFeatures(CPUString, FS);
}

/// ClassifyGlobalReference - Find the target operand flags that describe
/// how a global value should be referenced for the current subtarget.
unsigned char
ARM64Subtarget::ClassifyGlobalReference(const GlobalValue *GV,
                                        const TargetMachine &TM) const {

  // Determine whether this is a reference to a definition or a declaration.
  // Materializable GVs (in JIT lazy compilation mode) do not require an extra
  // load from stub.
  bool isDecl = GV->hasAvailableExternallyLinkage();
  if (GV->isDeclaration() && !GV->isMaterializable())
    isDecl = true;

  // If symbol visibility is hidden, the extra load is not needed if
  // the symbol is definitely defined in the current translation unit.
  if (TM.getRelocationModel() != Reloc::Static && GV->hasDefaultVisibility() &&
      (isDecl || GV->isWeakForLinker()))
    return ARM64II::MO_GOT;

  if (TM.getCodeModel() == CodeModel::Large && isTargetMachO())
    return ARM64II::MO_GOT;

  // FIXME: this will fail on static ELF for weak symbols.
  return ARM64II::MO_NO_FLAG;
}

/// This function returns the name of a function which has an interface
/// like the non-standard bzero function, if such a function exists on
/// the current subtarget and it is considered prefereable over
/// memset with zero passed as the second argument. Otherwise it
/// returns null.
const char *ARM64Subtarget::getBZeroEntry() const {
  // At the moment, always prefer bzero.
  return "bzero";
}

void ARM64Subtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                         MachineInstr *begin, MachineInstr *end,
                                         unsigned NumRegionInstrs) const {
  // LNT run (at least on Cyclone) showed reasonably significant gains for
  // bi-directional scheduling. 253.perlbmk.
  Policy.OnlyTopDown = false;
  Policy.OnlyBottomUp = false;
}
