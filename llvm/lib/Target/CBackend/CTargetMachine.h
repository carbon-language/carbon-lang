//===-- CTargetMachine.h - TargetMachine for the C backend ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TargetMachine that is used by the C backend.
//
//===----------------------------------------------------------------------===//

#ifndef CTARGETMACHINE_H
#define CTARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {

struct CTargetMachine : public TargetMachine {
  const TargetData DataLayout;       // Calculates type size & alignment

  CTargetMachine(const Module &M, const std::string &FS)
    : TargetMachine("CBackend", M),
      DataLayout("CBackend", &M) {}

  // This is the only thing that actually does anything here.
  virtual bool addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                   CodeGenFileType FileType, bool Fast);

  // This class always works, but shouldn't be the default in most cases.
  static unsigned getModuleMatchQuality(const Module &M) { return 1; }
  
  virtual const TargetData       *getTargetData() const { return &DataLayout; }
};

} // End llvm namespace


#endif
