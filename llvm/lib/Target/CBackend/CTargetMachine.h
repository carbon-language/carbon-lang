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
class IntrinsicLowering;

struct CTargetMachine : public TargetMachine {
  CTargetMachine(const Module &M, IntrinsicLowering *IL) :
    TargetMachine("CBackend", IL) {}

  virtual const TargetInstrInfo &getInstrInfo() const { abort(); }
  virtual const TargetFrameInfo &getFrameInfo() const { abort(); }
  virtual const TargetSchedInfo &getSchedInfo() const { abort(); }
  virtual const TargetRegInfo   &getRegInfo()   const { abort(); }

  // This is the only thing that actually does anything here.
  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);
};

} // End llvm namespace


#endif
