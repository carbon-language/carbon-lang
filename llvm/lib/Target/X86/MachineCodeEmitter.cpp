//===-- X86/MachineCodeEmitter.cpp - Convert X86 code to machine code -----===//
//
// This file contains the pass that transforms the X86 machine instructions into
// actual executable machine code.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"

namespace {
  struct Emitter : public FunctionPass {
    X86TargetMachine    &TM;
    const X86InstrInfo  &II;
    MachineCodeEmitter  &MCE;

    Emitter(X86TargetMachine &tm, MachineCodeEmitter &mce)
      : TM(tm), II(TM.getInstrInfo()), MCE(mce) {}

    bool runOnFunction(Function &F);

    void emitBasicBlock(MachineBasicBlock &MBB);
    void emitInstruction(MachineInstr &MI);
  };
}


/// addPassesToEmitMachineCode - Add passes to the specified pass manager to get
/// machine code emitted.  This uses a MAchineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool X86TargetMachine::addPassesToEmitMachineCode(PassManager &PM,
                                                  MachineCodeEmitter &MCE) {
  PM.add(new Emitter(*this, MCE));
  return false;
}

bool Emitter::runOnFunction(Function &F) {
  MachineFunction &MF = MachineFunction::get(&F);

  MCE.startFunction(MF);
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    emitBasicBlock(*I);
  MCE.finishFunction(MF);
  return false;
}

void Emitter::emitBasicBlock(MachineBasicBlock &MBB) {
  MCE.startBasicBlock(MBB);
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ++I)
    emitInstruction(**I);
}

void Emitter::emitInstruction(MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  const MachineInstrDescriptor &Desc = II.get(Opcode);

  // Emit instruction prefixes if neccesary
  if (Desc.TSFlags & X86II::OpSize) MCE.emitByte(0x66);// Operand size...
  if (Desc.TSFlags & X86II::TB)     MCE.emitByte(0x0F);// Two-byte opcode prefix

  switch (Desc.TSFlags & X86II::FormMask) {
  case X86II::RawFrm:
    MCE.emitByte(II.getBaseOpcodeFor(Opcode));

    if (MI.getNumOperands() == 1) {
      assert(MI.getOperand(0).getType() == MachineOperand::MO_PCRelativeDisp);
      MCE.emitPCRelativeDisp(MI.getOperand(0).getVRegValue());
    }

    break;
  }
}
