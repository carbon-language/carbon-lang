//===-- llvm/CodeGen/AsmPrinter.h - AsmPrinter Framework --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This class is intended to be used as a base class for target-specific
// asmwriters.  This class primarily takes care of printing global constants,
// which are printed in a very similar way across all targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ASMPRINTER_H
#define LLVM_CODEGEN_ASMPRINTER_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {
  class Constant;
  class Mangler;

  class AsmPrinter : public MachineFunctionPass {
  protected:
    /// Output stream on which we're printing assembly code.
    ///
    std::ostream &O;

    /// Target machine description.
    ///
    TargetMachine &TM;

    /// Name-mangler for global names.
    ///
    Mangler *Mang;

    /// Cache of mangled name for current function. This is recalculated at the
    /// beginning of each call to runOnMachineFunction().
    ///
    std::string CurrentFnName;

    AsmPrinter(std::ostream &o, TargetMachine &tm) : O(o), TM(tm) { }

    /// doInitialization - Set up the AsmPrinter when we are working on a new
    /// module.  If your pass overrides this, it must make sure to explicitly
    /// call this implementation.
    bool doInitialization(Module &M);

    /// doFinalization - Shut down the asmprinter.  If you override this in your
    /// pass, you must make sure to call it explicitly.
    bool doFinalization(Module &M);

    /// setupMachineFunction - This should be called when a new MachineFunction
    /// is being processed from runOnMachineFunction.
    void setupMachineFunction(MachineFunction &MF);

    /// emitConstantValueOnly - Print out the specified constant, without a
    /// storage class.  Only constants of first-class type are allowed here.
    void emitConstantValueOnly(const Constant *CV);
  };
}

#endif
