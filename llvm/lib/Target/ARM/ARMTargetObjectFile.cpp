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
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;
using namespace dwarf;

//===----------------------------------------------------------------------===//
//                               ELF Target
//===----------------------------------------------------------------------===//

void ARMElfTargetObjectFile::Initialize(MCContext &Ctx,
                                        const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);

  if (TM.getSubtarget<ARMSubtarget>().isAAPCS_ABI()) {
    StaticCtorSection =
      getELFSection(".init_array", MCSectionELF::SHT_INIT_ARRAY,
                    MCSectionELF::SHF_WRITE | MCSectionELF::SHF_ALLOC,
                    SectionKind::getDataRel());
    StaticDtorSection =
      getELFSection(".fini_array", MCSectionELF::SHT_FINI_ARRAY,
                    MCSectionELF::SHF_WRITE | MCSectionELF::SHF_ALLOC,
                    SectionKind::getDataRel());
  }
}

//===----------------------------------------------------------------------===//
//                              Mach-O Target
//===----------------------------------------------------------------------===//

void ARMMachOTargetObjectFile::Initialize(MCContext &Ctx,
                                          const TargetMachine &TM) {
  TargetLoweringObjectFileMachO::Initialize(Ctx, TM);

  // Exception Handling.
  LSDASection = getMachOSection("__TEXT", "__gcc_except_tab", 0,
                                SectionKind::getReadOnlyWithRel());
}

unsigned ARMMachOTargetObjectFile::getTTypeEncoding() const {
  return DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4;
}
