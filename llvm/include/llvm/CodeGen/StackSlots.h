//===-- llvm/CodeGen/StackSots.h -------------------------------*- C++ -*--===//
//
// External interface to stack-slots pass that enters 2 empty slots at the top
// of each function stack
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_STACKSLOTS_H
#define LLVM_CODEGEN_STACKSLOTS_H

class Pass;
class TargetMachine;
Pass *createStackSlotsPass(const TargetMachine &TM);

#endif
