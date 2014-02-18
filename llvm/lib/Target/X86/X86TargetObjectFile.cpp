//===-- X86TargetObjectFile.cpp - X86 Object Info -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86TargetObjectFile.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Operator.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Target/TargetLowering.h"

using namespace llvm;
using namespace dwarf;

const MCExpr *X86_64MachoTargetObjectFile::getTTypeGlobalReference(
    const GlobalValue *GV, unsigned Encoding, Mangler &Mang,
    const TargetMachine &TM, MachineModuleInfo *MMI,
    MCStreamer &Streamer) const {

  // On Darwin/X86-64, we can reference dwarf symbols with foo@GOTPCREL+4, which
  // is an indirect pc-relative reference.
  if (Encoding & (DW_EH_PE_indirect | DW_EH_PE_pcrel)) {
    const MCSymbol *Sym = TM.getTargetLowering()->getSymbol(GV, Mang);
    const MCExpr *Res =
      MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_GOTPCREL, getContext());
    const MCExpr *Four = MCConstantExpr::Create(4, getContext());
    return MCBinaryExpr::CreateAdd(Res, Four, getContext());
  }

  return TargetLoweringObjectFileMachO::getTTypeGlobalReference(
      GV, Encoding, Mang, TM, MMI, Streamer);
}

MCSymbol *X86_64MachoTargetObjectFile::getCFIPersonalitySymbol(
    const GlobalValue *GV, Mangler &Mang, const TargetMachine &TM,
    MachineModuleInfo *MMI) const {
  return TM.getTargetLowering()->getSymbol(GV, Mang);
}

void
X86LinuxTargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);
}

const MCExpr *
X86LinuxTargetObjectFile::getDebugThreadLocalSymbol(
    const MCSymbol *Sym) const {
  return MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_DTPOFF, getContext());
}

const MCExpr *X86WindowsTargetObjectFile::getExecutableRelativeSymbol(
    const ConstantExpr *CE, Mangler &Mang, const TargetMachine &TM) const {
  // We are looking for the difference of two symbols, need a subtraction
  // operation.
  const SubOperator *Sub = dyn_cast<SubOperator>(CE);
  if (!Sub)
    return 0;

  // Symbols must first be numbers before we can subtract them, we need to see a
  // ptrtoint on both subtraction operands.
  const PtrToIntOperator *SubLHS =
      dyn_cast<PtrToIntOperator>(Sub->getOperand(0));
  const PtrToIntOperator *SubRHS =
      dyn_cast<PtrToIntOperator>(Sub->getOperand(1));
  if (!SubLHS || !SubRHS)
    return 0;

  // Our symbols should exist in address space zero, cowardly no-op if
  // otherwise.
  if (SubLHS->getPointerAddressSpace() != 0 ||
      SubRHS->getPointerAddressSpace() != 0)
    return 0;

  // Both ptrtoint instructions must wrap global variables:
  // - Only global variables are eligible for image relative relocations.
  // - The subtrahend refers to the special symbol __ImageBase, a global.
  const GlobalVariable *GVLHS =
      dyn_cast<GlobalVariable>(SubLHS->getPointerOperand());
  const GlobalVariable *GVRHS =
      dyn_cast<GlobalVariable>(SubRHS->getPointerOperand());
  if (!GVLHS || !GVRHS)
    return 0;

  // We expect __ImageBase to be a global variable without a section, externally
  // defined.
  //
  // It should look something like this: @__ImageBase = external constant i8
  if (GVRHS->isThreadLocal() || GVRHS->getName() != "__ImageBase" ||
      !GVRHS->hasExternalLinkage() || GVRHS->hasInitializer() ||
      GVRHS->hasSection())
    return 0;

  // An image-relative, thread-local, symbol makes no sense.
  if (GVLHS->isThreadLocal())
    return 0;

  return MCSymbolRefExpr::Create(TM.getTargetLowering()->getSymbol(GVLHS, Mang),
                                 MCSymbolRefExpr::VK_COFF_IMGREL32,
                                 getContext());
}
