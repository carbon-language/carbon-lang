//===-- llvm/Target/ARMTargetObjectFile.h - ARM Object Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ARM_TARGETOBJECTFILE_H
#define LLVM_TARGET_ARM_TARGETOBJECTFILE_H

#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/MC/MCSectionELF.h"

namespace llvm {

  class ARMElfTargetObjectFile : public TargetLoweringObjectFileELF {
  public:
    ARMElfTargetObjectFile() : TargetLoweringObjectFileELF() {}

    void Initialize(MCContext &Ctx, const TargetMachine &TM) {
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
  };
} // end namespace llvm

#endif
