//===-- llvm/CodeGen/PeepholeOpts.h ----------------------------*- C++ -*--===//
//
// External interface to peephole optimization pass operating on machine code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PEEPHOLE_OPTS_H
#define LLVM_CODEGEN_PEEPHOLE_OPTS_H

class TargetMachine;
class FunctionPass;

FunctionPass *createPeepholeOptsPass(TargetMachine &Target);

#endif
