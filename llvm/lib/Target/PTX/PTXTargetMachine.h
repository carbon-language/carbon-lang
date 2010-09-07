//===-- PTXTargetMachine.h - Define TargetMachine for PTX -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PTX specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_TARGET_MACHINE_H
#define PTX_TARGET_MACHINE_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class PTXTargetMachine : public LLVMTargetMachine {
    public:
      PTXTargetMachine(const Target &T, const std::string &TT,
                       const std::string &FS);
  }; // class PTXTargetMachine
} // namespace llvm

#endif // PTX_TARGET_MACHINE_H
