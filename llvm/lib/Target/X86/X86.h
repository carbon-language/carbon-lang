//===-- X86.h - Top-level interface for X86 representation ------*- C++ -*-===//
//
// This file contains the entry points for global functions defined in the x86
// target library, as used by the LLVM JIT.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_X86_H
#define TARGET_X86_H

#include <iosfwd>
class TargetMachine;
class Pass;

/// createSimpleX86InstructionSelector - This pass converts an LLVM function
/// into a machine code representation is a very simple peep-hole fashion.  The
/// generated code sucks but the implementation is nice and simple.
///
Pass *createSimpleX86InstructionSelector(TargetMachine &TM);

/// createX86PeepholeOptimizer - Create a pass to perform X86 specific peephole
/// optimizations.
///
Pass *createX86PeepholeOptimizerPass();

/// createX86FloatingPointStackifierPass - This function returns a pass which
/// converts floating point register references and pseudo instructions into
/// floating point stack references and physical instructions.
///
Pass *createX86FloatingPointStackifierPass();

/// createX86CodePrinterPass - Print out the specified machine code function to
/// the specified stream.  This function should work regardless of whether or
/// not the function is in SSA form or not.
///
Pass *createX86CodePrinterPass(std::ostream &O);

/// X86EmitCodeToMemory - This function converts a register allocated function
/// into raw machine code in a dynamically allocated chunk of memory.  A pointer
/// to the start of the function is returned.
///
Pass *createEmitX86CodeToMemory();

/// X86 namespace - This namespace contains all of the register and opcode enums
/// used by the X86 backend.
///
namespace X86 {
  // Defines a large number of symbolic names for X86 registers.  This defines a
  // mapping from register name to register number.
  //
  enum Register {
#define R(ENUM, NAME, FLAGS, TSFLAGS, ALIAS_SET) ENUM,
#include "X86RegisterInfo.def"
  };

  // This defines a large number of symbolic names for X86 instruction opcodes.
  enum Opcode {
#define I(ENUM, NAME, BASEOPCODE, FLAGS, TSFLAGS, IMPDEFS, IMPUSES) ENUM,
#include "X86InstrInfo.def"
  };
}

#endif
