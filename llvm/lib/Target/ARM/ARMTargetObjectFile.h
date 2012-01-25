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

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

class MCContext;
class TargetMachine;

class ARMElfTargetObjectFile : public TargetLoweringObjectFileELF {
protected:
  const MCSection *AttributesSection;
  bool isAAPCS_ABI;
public:
  ARMElfTargetObjectFile() :
    TargetLoweringObjectFileELF(),
    AttributesSection(NULL)
  {}

  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);

  virtual const MCSection *getAttributesSection() const {
    return AttributesSection;
  }

  const MCSection * getStaticCtorSection(unsigned Priority) const;
  const MCSection * getStaticDtorSection(unsigned Priority) const;
};

} // end namespace llvm

#endif
