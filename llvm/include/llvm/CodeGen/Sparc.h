//===-- llvm/CodeGen/Sparc.h - Sparc Target Description ----------*- C++ -*--=//
//
// This file defines the Sparc processor targets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SPARC_H
#define LLVM_CODEGEN_SPARC_H

#include "llvm/CodeGen/TargetMachine.h"

//---------------------------------------------------------------------------
// class UltraSparcMachine 
// 
// Purpose:
//   Primary interface to machine description for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class TargetMachine, and creates machine-dependent subclasses
//   for classes such as MachineInstrInfo. 
//---------------------------------------------------------------------------

class UltraSparc : public TargetMachine {
public:
  UltraSparc();
  virtual ~UltraSparc();

  // compileMethod - For the sparc, we do instruction selection, followed by
  // delay slot scheduling, then register allocation.
  //
  virtual bool compileMethod(Method *M);
};

#endif
