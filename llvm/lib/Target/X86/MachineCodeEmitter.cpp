//===-- X86/MachineCodeEmitter.cpp - Convert X86 code to machine code -----===//
//
// This file contains the pass that transforms the X86 machine instructions into
// actual executable machine code.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"

namespace {
  struct Emitter : public FunctionPass {
    TargetMachine &TM;
    MachineCodeEmitter &MCE;

    Emitter(TargetMachine &tm, MachineCodeEmitter &mce) : TM(tm), MCE(mce) {}
    ~Emitter() {
    }

    bool runOnFunction(Function &F) { return false; }
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
