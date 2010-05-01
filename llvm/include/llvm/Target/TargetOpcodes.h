//===-- llvm/Target/TargetOpcodes.h - Target Indep Opcodes ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the target independent instruction opcodes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETOPCODES_H
#define LLVM_TARGET_TARGETOPCODES_H

namespace llvm {
  
/// Invariant opcodes: All instruction sets have these as their low opcodes.
namespace TargetOpcode {
  enum { 
    PHI = 0,
    INLINEASM = 1,
    DBG_LABEL = 2,
    EH_LABEL = 3,
    GC_LABEL = 4,
    
    /// KILL - This instruction is a noop that is used only to adjust the
    /// liveness of registers. This can be useful when dealing with
    /// sub-registers.
    KILL = 5,
    
    /// EXTRACT_SUBREG - This instruction takes two operands: a register
    /// that has subregisters, and a subregister index. It returns the
    /// extracted subregister value. This is commonly used to implement
    /// truncation operations on target architectures which support it.
    EXTRACT_SUBREG = 6,
    
    /// INSERT_SUBREG - This instruction takes three operands: a register
    /// that has subregisters, a register providing an insert value, and a
    /// subregister index. It returns the value of the first register with
    /// the value of the second register inserted. The first register is
    /// often defined by an IMPLICIT_DEF, as is commonly used to implement
    /// anyext operations on target architectures which support it.
    INSERT_SUBREG = 7,
    
    /// IMPLICIT_DEF - This is the MachineInstr-level equivalent of undef.
    IMPLICIT_DEF = 8,
    
    /// SUBREG_TO_REG - This instruction is similar to INSERT_SUBREG except
    /// that the first operand is an immediate integer constant. This constant
    /// is often zero, as is commonly used to implement zext operations on
    /// target architectures which support it, such as with x86-64 (with
    /// zext from i32 to i64 via implicit zero-extension).
    SUBREG_TO_REG = 9,
    
    /// COPY_TO_REGCLASS - This instruction is a placeholder for a plain
    /// register-to-register copy into a specific register class. This is only
    /// used between instruction selection and MachineInstr creation, before
    /// virtual registers have been created for all the instructions, and it's
    /// only needed in cases where the register classes implied by the
    /// instructions are insufficient. The actual MachineInstrs to perform
    /// the copy are emitted with the TargetInstrInfo::copyRegToReg hook.
    COPY_TO_REGCLASS = 10,

    /// DBG_VALUE - a mapping of the llvm.dbg.value intrinsic
    DBG_VALUE = 11,

    /// REG_SEQUENCE - This variadic instruction is used to form a register that
    /// represent a consecutive sequence of sub-registers. It's used as register
    /// coalescing / allocation aid and must be eliminated before code emission.
    /// e.g. v1027 = REG_SEQUENCE v1024, 3, v1025, 4, v1026, 5
    /// After register coalescing references of v1024 should be replace with
    /// v1027:3, v1025 with v1027:4, etc.
    REG_SEQUENCE = 12
  };
} // end namespace TargetOpcode
} // end namespace llvm

#endif
