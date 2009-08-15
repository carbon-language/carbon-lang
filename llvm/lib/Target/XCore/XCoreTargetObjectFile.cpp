//===-- XCoreTargetObjectFile.cpp - XCore object files --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCoreTargetObjectFile.h"
#include "XCoreSubtarget.h"
#include "MCSectionXCore.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;


void XCoreTargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &TM){
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);

  DataSection =
    MCSectionXCore::Create(".dp.data", MCSectionELF::SHT_PROGBITS, 
                           MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_WRITE |
                           MCSectionXCore::SHF_DP_SECTION,
                           SectionKind::getDataRel(), false, getContext());
  BSSSection =
    MCSectionXCore::Create(".dp.bss", MCSectionELF::SHT_NOBITS,
                           MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_WRITE |
                           MCSectionXCore::SHF_DP_SECTION,
                           SectionKind::getBSS(), false, getContext());
  
  // For now, disable lowering of mergable sections, just drop everything into
  // ReadOnly.
  MergeableConst4Section = 0;
  MergeableConst8Section = 0;
  MergeableConst16Section = 0;
  
  // TLS globals are lowered in the backend to arrays indexed by the current
  // thread id. After lowering they require no special handling by the linker
  // and can be placed in the standard data / bss sections.
  TLSDataSection = DataSection;
  TLSBSSSection = BSSSection;
  
  if (TM.getSubtarget<XCoreSubtarget>().isXS1A())
    ReadOnlySection =   // FIXME: Why is this a writable section for XS1A?
      MCSectionXCore::Create(".dp.rodata", MCSectionELF::SHT_PROGBITS,
                             MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_WRITE |
                             MCSectionXCore::SHF_DP_SECTION,
                             SectionKind::getDataRel(), false, getContext());
  else
    ReadOnlySection = 
      MCSectionXCore::Create(".cp.rodata", MCSectionELF::SHT_PROGBITS,
                             MCSectionELF::SHF_ALLOC |
                             MCSectionXCore::SHF_CP_SECTION,
                             SectionKind::getReadOnly(), false, getContext());
}
