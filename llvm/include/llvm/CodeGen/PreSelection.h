//===-- llvm/CodeGen/PreSelection.h ----------------------------*- C++ -*--===//
//
// External interface to pre-selection pass that specializes LLVM
// code for a target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PRE_SELECTION_H
#define LLVM_CODEGEN_PRE_SELECTION_H

class TargetMachine;
class Pass;

Pass *createPreSelectionPass(TargetMachine &Target);

#endif
