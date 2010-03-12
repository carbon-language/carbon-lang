//===-- llvm/Target/PPCTargetObjectFile.cpp - PPC Object Info Impl --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetObjectFile.h"
#include "PPCSubtarget.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;
using namespace dwarf;

//===----------------------------------------------------------------------===//
//                              Mach-O Target
//===----------------------------------------------------------------------===//

void PPCMachOTargetObjectFile::Initialize(MCContext &Ctx,
                                          const TargetMachine &TM) {
  TargetLoweringObjectFileMachO::Initialize(Ctx, TM);

  // Exception Handling.
  LSDASection = getMachOSection("__TEXT", "__gcc_except_tab", 0,
                                SectionKind::getReadOnlyWithRel());
}

unsigned PPCMachOTargetObjectFile::getTTypeEncoding() const {
  return DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4;
}
