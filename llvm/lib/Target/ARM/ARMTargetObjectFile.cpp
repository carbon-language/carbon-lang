//===-- llvm/Target/ARMTargetObjectFile.cpp - ARM Object Info Impl --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARMTargetObjectFile.h"
#include "ARMSubtarget.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ELF.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;
using namespace dwarf;

//===----------------------------------------------------------------------===//
//                               ELF Target
//===----------------------------------------------------------------------===//

void ARMElfTargetObjectFile::Initialize(MCContext &Ctx,
                                        const TargetMachine &TM) {
  bool isAAPCS_ABI = TM.getSubtarget<ARMSubtarget>().isAAPCS_ABI();
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(isAAPCS_ABI);

  if (isAAPCS_ABI) {
    LSDASection = NULL;
  }

  AttributesSection =
    getContext().getELFSection(".ARM.attributes",
                               ELF::SHT_ARM_ATTRIBUTES,
                               0,
                               SectionKind::getMetadata());
}

const MCExpr *ARMElfTargetObjectFile::
getTTypeGlobalReference(const GlobalValue *GV, Mangler *Mang,
                        MachineModuleInfo *MMI, unsigned Encoding,
                        MCStreamer &Streamer) const {
  assert(Encoding == DW_EH_PE_absptr && "Can handle absptr encoding only");

  return MCSymbolRefExpr::Create(getSymbol(*Mang, GV),
                                 MCSymbolRefExpr::VK_ARM_TARGET2,
                                 getContext());
}
