//===-- llvm/Target/X86/X86TargetObjectFile.cpp - X86 Object Info ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86TargetObjectFile.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Mangler.h"
#include "llvm/MC/MCExpr.h"
using namespace llvm;

const MCExpr *X8664_MachoTargetObjectFile::
getSymbolForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                                 MachineModuleInfo *MMI,
                                 bool &IsIndirect, bool &IsPCRel) const {
  
  // On Darwin/X86-64, we can reference dwarf symbols with foo@GOTPCREL+4, which
  // is an indirect pc-relative reference.
  IsIndirect = true;
  IsPCRel    = true;
  
  SmallString<128> Name;
  Mang->getNameWithPrefix(Name, GV, false);
  Name += "@GOTPCREL";
  const MCExpr *Res =
    MCSymbolRefExpr::Create(Name.str(), getContext());
  const MCExpr *Four = MCConstantExpr::Create(4, getContext());
  return MCBinaryExpr::CreateAdd(Res, Four, getContext());
}

