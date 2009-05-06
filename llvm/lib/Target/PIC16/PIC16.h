//===-- PIC16.h - Top-level interface for PIC16 representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in 
// the LLVM PIC16 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_PIC16_H
#define LLVM_TARGET_PIC16_H

#include "llvm/Target/TargetMachine.h"
#include <iosfwd>
#include <cassert>
#include <string>

namespace llvm {
  class PIC16TargetMachine;
  class FunctionPass;
  class MachineCodeEmitter;
  class raw_ostream;

namespace PIC16CC {
  enum CondCodes {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    ULT,
    UGT,
    ULE,
    UGE
  };
}
  // A Central object to manage all ABI naming conventions.
  class PIC16ABINames {
    public:
    // Map the name of the symbol to its section name.
    // Current ABI:
    // ------------------------------------------------------
    // Global variables do not have any '.' in their names.
    // they are prefixed with @
    // These are maily function names and global variable names.
    // -------------------------------------------------------
    // Functions and auto variables.
    // Names are mangled as <prefix><funcname>.<id>.<varname>
    // Where prefix is a special char '@' and id is any one of
    // the following
    // .auto. - an automatic var of a function.
    // .temp. - temproray data of a function.
    // .ret.  - return value label for a function.
    // .frame. - Frame label for a function where retval, args
    //           and temps are stored.
    // .args. - Label used to pass arguments to a direct call.
    // Example - Function name:   @foo
    //           Its frame:       @foo.frame.
    //           Its retval:      @foo.ret.
    //           Its local vars:  @foo.auto.a
    //           Its temp data:   @foo.temp.
    //           Its arg passing: @foo.args.
    //----------------------------------------------
    // Libcall - compiler generated libcall names must have a .lib.
    //           This id will be used to emit extern decls for libcalls.
    // Example - libcall name:   @sra_i8.lib.
    //           To pass args:   @sra_i8.args.
    //           To return val:  @sra_i8.ret.
    //----------------------------------------------
    
    enum IDs {
      PREFIX_SYMBOL,

      FUNC_AUTOS,
      FUNC_FRAME,
      FUNC_RET,
      FUNC_ARGS,
      FUNC_TEMPS,
      
      LIBCALL,
      
      FRAME_SECTION,
      AUTOS_SECTION
   };

  };

  inline static const char *getIDName(PIC16ABINames::IDs id) {
    switch (id) {
    default: assert(0 && "Unknown id");
    case PIC16ABINames::PREFIX_SYMBOL:    return "@";
    case PIC16ABINames::FUNC_AUTOS:       return ".auto.";
    case PIC16ABINames::FUNC_FRAME:       return ".frame.";
    case PIC16ABINames::FUNC_TEMPS:       return ".temp.";
    case PIC16ABINames::FUNC_ARGS:       return ".args.";
    case PIC16ABINames::FUNC_RET:       return ".ret.";
    case PIC16ABINames::FRAME_SECTION:       return "fpdata";
    case PIC16ABINames::AUTOS_SECTION:       return "fadata";
    }
  }

  inline static PIC16ABINames::IDs getID(const std::string &Sym) {
    if (Sym.find(getIDName(PIC16ABINames::FUNC_TEMPS)))
     return PIC16ABINames::FUNC_TEMPS;

    if (Sym.find(getIDName(PIC16ABINames::FUNC_FRAME)))
     return PIC16ABINames::FUNC_FRAME;

    if (Sym.find(getIDName(PIC16ABINames::FUNC_RET)))
     return PIC16ABINames::FUNC_RET;

    if (Sym.find(getIDName(PIC16ABINames::FUNC_ARGS)))
     return PIC16ABINames::FUNC_ARGS;

    if (Sym.find(getIDName(PIC16ABINames::FUNC_AUTOS)))
     return PIC16ABINames::FUNC_AUTOS;

    if (Sym.find(getIDName(PIC16ABINames::LIBCALL)))
     return PIC16ABINames::LIBCALL;

    // It does not have any ID. So its a global.
    assert (0 && "Could not determine ID symbol type");
  }


  inline static const char *PIC16CondCodeToString(PIC16CC::CondCodes CC) {
    switch (CC) {
    default: assert(0 && "Unknown condition code");
    case PIC16CC::NE:  return "ne";
    case PIC16CC::EQ:   return "eq";
    case PIC16CC::LT:   return "lt";
    case PIC16CC::ULT:   return "lt";
    case PIC16CC::LE:  return "le";
    case PIC16CC::GT:  return "gt";
    case PIC16CC::UGT:  return "gt";
    case PIC16CC::GE:   return "ge";
    }
  }

  inline static bool isSignedComparison(PIC16CC::CondCodes CC) {
    switch (CC) {
    default: assert(0 && "Unknown condition code");
    case PIC16CC::NE:  
    case PIC16CC::EQ: 
    case PIC16CC::LT:
    case PIC16CC::LE:
    case PIC16CC::GE:
    case PIC16CC::GT:
      return true;
    case PIC16CC::ULT:
    case PIC16CC::UGT:
    case PIC16CC::ULE:
    case PIC16CC::UGE:
      return false;   // condition codes for unsigned comparison. 
    }
  }



  FunctionPass *createPIC16ISelDag(PIC16TargetMachine &TM);
  FunctionPass *createPIC16CodePrinterPass(raw_ostream &OS, 
                                           PIC16TargetMachine &TM,
                                           CodeGenOpt::Level OptLevel,
                                           bool Verbose);
  // Banksel optimzer pass.
  FunctionPass *createPIC16MemSelOptimizerPass();
  std::string getSectionNameForSym(const std::string &Sym);
} // end namespace llvm;

// Defines symbolic names for PIC16 registers.  This defines a mapping from
// register name to register number.
#include "PIC16GenRegisterNames.inc"

// Defines symbolic names for the PIC16 instructions.
#include "PIC16GenInstrNames.inc"

#endif
