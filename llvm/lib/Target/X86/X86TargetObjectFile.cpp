//===-- llvm/Target/X86/X86TargetObjectFile.cpp - X86 Object Info ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86TargetObjectFile.h"
#include "X86TargetMachine.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Target/Mangler.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Dwarf.h"
using namespace llvm;
using namespace dwarf;

const MCExpr *X8664_MachoTargetObjectFile::
getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                               MachineModuleInfo *MMI, unsigned Encoding,
                               MCStreamer &Streamer) const {

  // On Darwin/X86-64, we can reference dwarf symbols with foo@GOTPCREL+4, which
  // is an indirect pc-relative reference.
  if (Encoding & (DW_EH_PE_indirect | DW_EH_PE_pcrel)) {
    const MCSymbol *Sym = Mang->getSymbol(GV);
    const MCExpr *Res =
      MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_GOTPCREL, getContext());
    const MCExpr *Four = MCConstantExpr::Create(4, getContext());
    return MCBinaryExpr::CreateAdd(Res, Four, getContext());
  }

  return TargetLoweringObjectFileMachO::
    getExprForDwarfGlobalReference(GV, Mang, MMI, Encoding, Streamer);
}

MCSymbol *X8664_MachoTargetObjectFile::
getCFIPersonalitySymbol(const GlobalValue *GV, Mangler *Mang,
                        MachineModuleInfo *MMI) const {
  return Mang->getSymbol(GV);
}

unsigned X8632_ELFTargetObjectFile::getPersonalityEncoding() const {
  if (TM.getRelocationModel() == Reloc::PIC_)
    return DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4;
  else
    return DW_EH_PE_absptr;
}

unsigned X8632_ELFTargetObjectFile::getLSDAEncoding() const {
  if (TM.getRelocationModel() == Reloc::PIC_)
    return DW_EH_PE_pcrel | DW_EH_PE_sdata4;
  else
    return DW_EH_PE_absptr;
}

unsigned X8632_ELFTargetObjectFile::getFDEEncoding(bool FDE) const {
  if (TM.getRelocationModel() == Reloc::PIC_)
    return DW_EH_PE_pcrel | DW_EH_PE_sdata4;
  else
    return DW_EH_PE_absptr;
}

unsigned X8632_ELFTargetObjectFile::getTTypeEncoding() const {
  if (TM.getRelocationModel() == Reloc::PIC_)
    return DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4;
  else
    return DW_EH_PE_absptr;
}

unsigned X8664_ELFTargetObjectFile::getPersonalityEncoding() const {
  CodeModel::Model Model = TM.getCodeModel();
  if (TM.getRelocationModel() == Reloc::PIC_)
    return DW_EH_PE_indirect | DW_EH_PE_pcrel | (Model == CodeModel::Small ||
                                                 Model == CodeModel::Medium ?
                                            DW_EH_PE_sdata4 : DW_EH_PE_sdata8);

  if (Model == CodeModel::Small || Model == CodeModel::Medium)
    return DW_EH_PE_udata4;

  return DW_EH_PE_absptr;
}

unsigned X8664_ELFTargetObjectFile::getLSDAEncoding() const {
  CodeModel::Model Model = TM.getCodeModel();
  if (TM.getRelocationModel() == Reloc::PIC_)
    return DW_EH_PE_pcrel | (Model == CodeModel::Small ?
                             DW_EH_PE_sdata4 : DW_EH_PE_sdata8);

  if (Model == CodeModel::Small)
    return DW_EH_PE_udata4;

  return DW_EH_PE_absptr;
}

unsigned X8664_ELFTargetObjectFile::getFDEEncoding(bool CFI) const {
  if (CFI)
    return DW_EH_PE_pcrel | DW_EH_PE_sdata4;

  CodeModel::Model Model = TM.getCodeModel();
  if (TM.getRelocationModel() == Reloc::PIC_)
    return DW_EH_PE_pcrel | DW_EH_PE_sdata4;

  return DW_EH_PE_udata4;
}

unsigned X8664_ELFTargetObjectFile::getTTypeEncoding() const {
  CodeModel::Model Model = TM.getCodeModel();
  if (TM.getRelocationModel() == Reloc::PIC_)
    return DW_EH_PE_indirect | DW_EH_PE_pcrel | (Model == CodeModel::Small ||
                                                 Model == CodeModel::Medium ?
                                            DW_EH_PE_sdata4 : DW_EH_PE_sdata8);

  if (Model == CodeModel::Small)
    return DW_EH_PE_udata4;

  return DW_EH_PE_absptr;
}
