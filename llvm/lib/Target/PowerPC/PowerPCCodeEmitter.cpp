//===-- PowerPCCodeEmitter.cpp - JIT Code Emitter for PowerPC -----*- C++ -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
//
//===----------------------------------------------------------------------===//

#include "PowerPCTargetMachine.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "Support/Debug.h"

namespace llvm {

namespace {
  class PowerPCCodeEmitter : public MachineFunctionPass {
    TargetMachine &TM;
    MachineCodeEmitter &MCE;

  public:
    PowerPCCodeEmitter(TargetMachine &T, MachineCodeEmitter &M) 
      : TM(T), MCE(M) {}

    const char *getPassName() const { return "PowerPC Machine Code Emitter"; }

    /// runOnMachineFunction - emits the given MachineFunction to memory
    ///
    bool runOnMachineFunction(MachineFunction &MF);

    /// emitBasicBlock - emits the given MachineBasicBlock to memory
    ///
    void emitBasicBlock(MachineBasicBlock &MBB);

    /// emitWord - write a 32-bit word to memory at the current PC
    ///
    void emitWord(unsigned w) { MCE.emitWord(w); }

    unsigned getValueBit(int64_t Val, unsigned bit);

    /// getBinaryCodeForInstr - returns the assembled code for an instruction
    ///
    unsigned getBinaryCodeForInstr(MachineInstr &MI) { return 0; }
  };
}

/// addPassesToEmitMachineCode - Add passes to the specified pass manager to get
/// machine code emitted.  This uses a MachineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool PowerPCTargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                      MachineCodeEmitter &MCE) {
  // Machine code emitter pass for PowerPC
  PM.add(new PowerPCCodeEmitter(*this, MCE)); 
  // Delete machine code for this function after emitting it:
  PM.add(createMachineCodeDeleter());
  // We don't yet support machine code emission
  return true;
}

bool PowerPCCodeEmitter::runOnMachineFunction(MachineFunction &MF) {
  MCE.startFunction(MF);
  MCE.emitConstantPool(MF.getConstantPool());
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    emitBasicBlock(*I);
  MCE.finishFunction(MF);
  return false;
}

void PowerPCCodeEmitter::emitBasicBlock(MachineBasicBlock &MBB) {
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ++I)
    emitWord(getBinaryCodeForInstr(*I));
}

unsigned PowerPCCodeEmitter::getValueBit(int64_t Val, unsigned bit) {
  Val >>= bit;
  return (Val & 1);
}

void *PowerPCJITInfo::getJITStubForFunction(Function *F,
                                            MachineCodeEmitter &MCE) {
  assert (0 && "PowerPCJITInfo::getJITStubForFunction not implemented");
  return 0;
}

void PowerPCJITInfo::replaceMachineCodeForFunction (void *Old, void *New) {
  assert (0 && "PowerPCJITInfo::replaceMachineCodeForFunction not implemented");
}

//#include "PowerPCGenCodeEmitter.inc"

} // end llvm namespace

