//===-- X86/Printer.cpp - Convert X86 code to human readable rep. ---------===//
//
// This file contains a printer that converts from our internal representation
// of LLVM code to a nice human readable form that is suitable for debuggging.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include <iostream>

namespace {
  struct Printer : public FunctionPass {
    TargetMachine &TM;
    std::ostream &O;

    Printer(TargetMachine &tm, std::ostream &o) : TM(tm), O(o) {}

    bool runOnFunction(Function &F);
  };
}

bool Printer::runOnFunction(Function &F) {
  MachineFunction &MF = MachineFunction::get(&F);
  O << "x86 printing not implemented yet!\n";
  
  // This should use the X86InstructionInfo::print method to print assembly
  // for each instruction
  return false;
}




/// createX86CodePrinterPass - Print out the specified machine code function to
/// the specified stream.  This function should work regardless of whether or
/// not the function is in SSA form or not.
///
Pass *createX86CodePrinterPass(TargetMachine &TM, std::ostream &O) {
  return new Printer(TM, O);
}
