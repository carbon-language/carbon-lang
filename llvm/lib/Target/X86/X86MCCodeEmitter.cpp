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
#include "X86InstrInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
class X86MCCodeEmitter : public MCCodeEmitter {
  X86MCCodeEmitter(const X86MCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const X86MCCodeEmitter &); // DO NOT IMPLEMENT
  const TargetMachine &TM;
  const TargetInstrInfo &TII;
public:
  X86MCCodeEmitter(TargetMachine &tm) 
    : TM(tm), TII(*TM.getInstrInfo()) {
  }

  ~X86MCCodeEmitter() {}
  
  void EmitByte(unsigned char C, raw_ostream &OS) const {
    OS << (char)C;
  }
  
  void EncodeInstruction(const MCInst &MI, raw_ostream &OS) const;
  
};

} // end anonymous namespace


MCCodeEmitter *llvm::createX86MCCodeEmitter(const Target &,
                                            TargetMachine &TM) {
  return new X86MCCodeEmitter(TM);
}



void X86MCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS) const {
  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = TII.get(Opcode);
  
  // Emit the lock opcode prefix as needed.
  if (Desc.TSFlags & X86II::LOCK)
    EmitByte(0xF0, OS);
  
  // Emit segment override opcode prefix as needed.
  switch (Desc.TSFlags & X86II::SegOvrMask) {
  default: assert(0 && "Invalid segment!");
  case 0: break;  // No segment override!
  case X86II::FS:
    EmitByte(0x64, OS);
    break;
  case X86II::GS:
    EmitByte(0x65, OS);
    break;
  }
  
  
}
