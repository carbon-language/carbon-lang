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

    //===------------------------------------------------------------------===//
    // Properties to be set by the derived class ctor, used to configure the
    // asmwriter.

    /// GlobalPrefix - If this is set to a non-empty string, it is prepended
    /// onto all global symbols.  This is often used for "_" or ".".
    const char *GlobalPrefix;

    /// ZeroDirective - this should be set to the directive used to get some
    /// number of zero bytes emitted to the current section.  Common cases are
    /// "\t.zero\t" and "\t.space\t".  If this is set to null, the
    /// Data*bitsDirective's will be used to emit zero bytes.
    const char *ZeroDirective;   // Defaults to "\t.zero\t"

    /// AsciiDirective - This directive allows emission of an ascii string with
    /// the standard C escape characters embedded into it.
    const char *AsciiDirective;
    
    /// DataDirectives - These directives are used to output some unit of
    /// integer data to the current section.  If a data directive is set to
    /// null, smaller data directives will be used to emit the large sizes.
    const char *Data8bitsDirective;   // Defaults to "\t.byte\t"
    const char *Data16bitsDirective;  // Defaults to "\t.short\t"
    const char *Data32bitsDirective;  // Defaults to "\t.long\t"
    const char *Data64bitsDirective;  // Defaults to "\t.quad\t"

    AsmPrinter(std::ostream &o, TargetMachine &tm)
      : O(o), TM(tm),
        GlobalPrefix(""),
        ZeroDirective("\t.zero\t"),
        AsciiDirective("\t.ascii\t"),
        Data8bitsDirective("\t.byte\t"),
        Data16bitsDirective("\t.short\t"),
        Data32bitsDirective("\t.long\t"),
        Data64bitsDirective("\t.quad\t") { }

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

    /// emitGlobalConstant - Print a general LLVM constant to the .s file.
    ///
    void emitGlobalConstant(const Constant* CV);
  };
}

#endif
