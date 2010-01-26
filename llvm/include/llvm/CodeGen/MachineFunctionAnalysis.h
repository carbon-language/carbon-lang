//===-- MachineFunctionAnalysis.h - Owner of MachineFunctions ----*-C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachineFunctionAnalysis class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINE_FUNCTION_ANALYSIS_H
#define LLVM_CODEGEN_MACHINE_FUNCTION_ANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class MachineFunction;

/// MachineFunctionAnalysis - This class is a Pass that manages a
/// MachineFunction object.
struct MachineFunctionAnalysis : public FunctionPass {
private:
  const TargetMachine &TM;
  CodeGenOpt::Level OptLevel;
  MachineFunction *MF;
  unsigned NextFnNum;
public:
  static char ID;
  explicit MachineFunctionAnalysis(const TargetMachine &tm,
                                   CodeGenOpt::Level OL = CodeGenOpt::Default);
  ~MachineFunctionAnalysis();

  MachineFunction &getMF() const { return *MF; }
  CodeGenOpt::Level getOptLevel() const { return OptLevel; }

private:
  virtual bool doInitialization(Module &) { NextFnNum = 1; return false; }
  virtual bool runOnFunction(Function &F);
  virtual void releaseMemory();
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
};

} // End llvm namespace

#endif
