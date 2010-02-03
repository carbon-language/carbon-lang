//===-- X86/X86MCCodeEmitter.cpp - Convert X86 code to machine code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86MCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-emitter"
#include "X86.h"
#include "X86TargetMachine.h"
#include "llvm/MC/MCCodeEmitter.h"
using namespace llvm;

namespace {
class X86MCCodeEmitter : public MCCodeEmitter {
  X86MCCodeEmitter(const X86MCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const X86MCCodeEmitter &); // DO NOT IMPLEMENT
  X86TargetMachine &TM;
public:
  X86MCCodeEmitter(X86TargetMachine &tm) : TM(tm) {
  }

  ~X86MCCodeEmitter() {}
  
  void EncodeInstruction(const MCInst &MI, raw_ostream &OS) const {
  }
};

} // end anonymous namespace


MCCodeEmitter *llvm::createX86MCCodeEmitter(const Target &,
                                            TargetMachine &TM) {
  return new X86MCCodeEmitter(static_cast<X86TargetMachine&>(TM));
}
