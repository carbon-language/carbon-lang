//===-- PPC32CodeEmitter.cpp - JIT Code Emitter for PowerPC32 -----*- C++ -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file defines the PowerPC 32-bit CodeEmitter and associated machinery to
// JIT-compile bytecode to native PowerPC.
//
//===----------------------------------------------------------------------===//

#include "PPC32JITInfo.h"
#include "PPC32TargetMachine.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"

namespace llvm {

namespace {
  class PPC32CodeEmitter : public MachineFunctionPass {
    TargetMachine &TM;
    MachineCodeEmitter &MCE;

    int64_t getMachineOpValue(MachineInstr &MI, MachineOperand &MO);

  public:
    PPC32CodeEmitter(TargetMachine &T, MachineCodeEmitter &M) 
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
    
    /// getValueBit - return the particular bit of Val
    ///
    unsigned getValueBit(int64_t Val, unsigned bit) { return (Val >> bit) & 1; }

    /// getBinaryCodeForInstr - returns the assembled code for an instruction
    ///
    unsigned getBinaryCodeForInstr(MachineInstr &MI);
  };
}

/// addPassesToEmitMachineCode - Add passes to the specified pass manager to get
/// machine code emitted.  This uses a MachineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool PPC32TargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                    MachineCodeEmitter &MCE) {
  // Machine code emitter pass for PowerPC
  PM.add(new PPC32CodeEmitter(*this, MCE)); 
  // Delete machine code for this function after emitting it
  PM.add(createMachineCodeDeleter());
  return false;
}

bool PPC32CodeEmitter::runOnMachineFunction(MachineFunction &MF) {
  MCE.startFunction(MF);
  MCE.emitConstantPool(MF.getConstantPool());
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    emitBasicBlock(*I);
  MCE.finishFunction(MF);
  return false;
}

void PPC32CodeEmitter::emitBasicBlock(MachineBasicBlock &MBB) {
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ++I)
    emitWord(getBinaryCodeForInstr(*I));
}

int64_t PPC32CodeEmitter::getMachineOpValue(MachineInstr &MI, 
                                            MachineOperand &MO) {
  int64_t rv = 0; // Return value; defaults to 0 for unhandled cases
                  // or things that get fixed up later by the JIT.
  if (MO.isPCRelativeDisp()) {
    std::cerr << "PPC32CodeEmitter: PC-relative disp unhandled\n";
    abort();
  } else if (MO.isRegister()) {
    rv = MO.getReg();
  } else if (MO.isImmediate()) {
    rv = MO.getImmedValue();
#if 0
  } else if (MO.isGlobalAddress()) {
  } else if (MO.isMachineBasicBlock()) {
    MachineBasicBlock *MBB = MO.getMachineBasicBlock();
  } else if (MO.isExternalSymbol()) {
  } else if (MO.isFrameIndex()) {
    unsigned index = MO.getFrameIndex();
  } else if (MO.isConstantPoolIndex()) {
    unsigned index = MO.getCosntantPoolIndex();
#endif
  } else {
    std::cerr << "ERROR: Unknown type of MachineOperand: " << MO << "\n";
    abort();
  }

  return rv;
}


void *PPC32JITInfo::getJITStubForFunction(Function *F,
                                          MachineCodeEmitter &MCE) {
  std::cerr << "PPC32JITInfo::getJITStubForFunction not implemented\n";
  abort();
  return 0;
}

void PPC32JITInfo::replaceMachineCodeForFunction (void *Old, void *New) {
  std::cerr << "PPC32JITInfo::replaceMachineCodeForFunction not implemented\n";
  abort();
}

#include "PPC32GenCodeEmitter.inc"

} // end llvm namespace

