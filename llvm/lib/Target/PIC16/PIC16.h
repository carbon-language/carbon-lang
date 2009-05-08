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
#include <cstring>
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

    inline static const char *getIDName(IDs id) {
      switch (id) {
      default: assert(0 && "Unknown id");
      case PREFIX_SYMBOL:    return "@";
      case FUNC_AUTOS:       return ".auto.";
      case FUNC_FRAME:       return ".frame.";
      case FUNC_TEMPS:       return ".temp.";
      case FUNC_ARGS:       return ".args.";
      case FUNC_RET:       return ".ret.";
      case FRAME_SECTION:       return "fpdata";
      case AUTOS_SECTION:       return "fadata";
      }
    }

    inline static IDs getID(const std::string &Sym) {
      if (Sym.find(getIDName(FUNC_TEMPS)))
        return FUNC_TEMPS;

      if (Sym.find(getIDName(FUNC_FRAME)))
        return FUNC_FRAME;

      if (Sym.find(getIDName(FUNC_RET)))
        return FUNC_RET;

      if (Sym.find(getIDName(FUNC_ARGS)))
        return FUNC_ARGS;

      if (Sym.find(getIDName(FUNC_AUTOS)))
        return FUNC_AUTOS;

      if (Sym.find(getIDName(LIBCALL)))
        return LIBCALL;

      // It does not have any ID. So its a global.
      assert (0 && "Could not determine ID symbol type");
    }

    // Get func name from a mangled name.
    // In all cases func name is the first component before a '.'.
    static inline std::string getFuncNameForSym(const std::string &Sym) {
      const char *prefix = getIDName (PREFIX_SYMBOL);

      // If this name has a prefix, func name start after prfix in that case.
      size_t func_name_start = 0;
      if (Sym.find(prefix, 0, strlen(prefix)) != std::string::npos)
        func_name_start = strlen(prefix);

      // Position of the . after func name. That's where func name ends.
      size_t func_name_end = Sym.find ('.', func_name_start);

      return Sym.substr (func_name_start, func_name_end);
    }

    // Form a section name given the section type and func name.
    static std::string
    getSectionNameForFunc (const std::string &Fname, const IDs sec_id) {
      std::string sec_id_string = getIDName(sec_id);
      return sec_id_string + "." + Fname + ".#";
    }

    // Get the section for the given external symbol names.
    // This tries to find the type (ID) of the symbol from its mangled name
    // and return appropriate section name for it.
    static inline std::string getSectionNameForSym(const std::string &Sym) {
      std::string SectionName;
 
      IDs id = getID (Sym);
      std::string Fname = getFuncNameForSym (Sym);

      switch (id) {
        default : assert (0 && "Could not determine external symbol type");
        case FUNC_FRAME:
        case FUNC_RET:
        case FUNC_TEMPS:
        case FUNC_ARGS:  {
          return getSectionNameForFunc (Fname, FRAME_SECTION);
        }
        case FUNC_AUTOS: {
          return getSectionNameForFunc (Fname, AUTOS_SECTION);
        }
      }
    }
  }; // class PIC16ABINames.




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
