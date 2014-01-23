//===-- XCoreTargetObjectFile.h - XCore Object Info -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_XCORE_TARGETOBJECTFILE_H
#define LLVM_TARGET_XCORE_TARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

static const unsigned CodeModelLargeSize = 256;

  class XCoreTargetObjectFile : public TargetLoweringObjectFileELF {
   const MCSection *BSSSectionLarge;
   const MCSection *DataSectionLarge;
   const MCSection *ReadOnlySectionLarge;
  public:
    void Initialize(MCContext &Ctx, const TargetMachine &TM);

    virtual const MCSection *
    getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                             Mangler *Mang, const TargetMachine &TM) const;

    virtual const MCSection *SelectSectionForGlobal(const GlobalValue *GV,
                                                    SectionKind Kind,
                                                    Mangler *Mang,
                                                    TargetMachine &TM) const;

    virtual const MCSection *getSectionForConstant(SectionKind Kind) const;
  };
} // end namespace llvm

#endif
