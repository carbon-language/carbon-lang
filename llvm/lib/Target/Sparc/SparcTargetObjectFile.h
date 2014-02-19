//===-- SparcTargetObjectFile.h - Sparc Object Info -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_SPARC_TARGETOBJECTFILE_H
#define LLVM_TARGET_SPARC_TARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

class MCContext;
class TargetMachine;

class SparcELFTargetObjectFile : public TargetLoweringObjectFileELF {
public:
  SparcELFTargetObjectFile() :
    TargetLoweringObjectFileELF()
  {}

  const MCExpr *
  getTTypeGlobalReference(const GlobalValue *GV, unsigned Encoding,
                          Mangler &Mang, const TargetMachine &TM,
                          MachineModuleInfo *MMI, MCStreamer &Streamer) const
      LLVM_OVERRIDE;
};

} // end namespace llvm

#endif
