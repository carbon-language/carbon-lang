//===- X86InstrInfo.h - X86 Instruction Information ------------*- C++ -*- ===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86INSTRUCTIONINFO_H
#define X86INSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "X86RegisterInfo.h"

namespace llvm {

/// X86II - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace X86II {
  enum {
    //===------------------------------------------------------------------===//
    // Instruction types.  These are the standard/most common forms for X86
    // instructions.
    //

    // PseudoFrm - This represents an instruction that is a pseudo instruction
    // or one that has not been implemented yet.  It is illegal to code generate
    // it, but tolerated for intermediate implementation stages.
    Pseudo         = 0,

    /// Raw - This form is for instructions that don't have any operands, so
    /// they are just a fixed opcode value, like 'leave'.
    RawFrm         = 1,
    
    /// AddRegFrm - This form is used for instructions like 'push r32' that have
    /// their one register operand added to their opcode.
    AddRegFrm      = 2,

    /// MRMDestReg - This form is used for instructions that use the Mod/RM byte
    /// to specify a destination, which in this case is a register.
    ///
    MRMDestReg     = 3,

    /// MRMDestMem - This form is used for instructions that use the Mod/RM byte
    /// to specify a destination, which in this case is memory.
    ///
    MRMDestMem     = 4,

    /// MRMSrcReg - This form is used for instructions that use the Mod/RM byte
    /// to specify a source, which in this case is a register.
    ///
    MRMSrcReg      = 5,

    /// MRMSrcMem - This form is used for instructions that use the Mod/RM byte
    /// to specify a source, which in this case is memory.
    ///
    MRMSrcMem      = 6,
  
    /// MRMS[0-7][rm] - These forms are used to represent instructions that use
    /// a Mod/RM byte, and use the middle field to hold extended opcode
    /// information.  In the intel manual these are represented as /0, /1, ...
    ///

    // First, instructions that operate on a register r/m operand...
    MRMS0r = 16,  MRMS1r = 17,  MRMS2r = 18,  MRMS3r = 19, // Format /0 /1 /2 /3
    MRMS4r = 20,  MRMS5r = 21,  MRMS6r = 22,  MRMS7r = 23, // Format /4 /5 /6 /7

    // Next, instructions that operate on a memory r/m operand...
    MRMS0m = 24,  MRMS1m = 25,  MRMS2m = 26,  MRMS3m = 27, // Format /0 /1 /2 /3
    MRMS4m = 28,  MRMS5m = 29,  MRMS6m = 30,  MRMS7m = 31, // Format /4 /5 /6 /7

    FormMask       = 31,

    //===------------------------------------------------------------------===//
    // Actual flags...

    // OpSize - Set if this instruction requires an operand size prefix (0x66),
    // which most often indicates that the instruction operates on 16 bit data
    // instead of 32 bit data.
    OpSize      = 1 << 5,

    // Op0Mask - There are several prefix bytes that are used to form two byte
    // opcodes.  These are currently 0x0F, and 0xD8-0xDF.  This mask is used to
    // obtain the setting of this field.  If no bits in this field is set, there
    // is no prefix byte for obtaining a multibyte opcode.
    //
    Op0Shift    = 6,
    Op0Mask     = 0xF << Op0Shift,

    // TB - TwoByte - Set if this instruction has a two byte opcode, which
    // starts with a 0x0F byte before the real opcode.
    TB          = 1 << Op0Shift,

    // D8-DF - These escape opcodes are used by the floating point unit.  These
    // values must remain sequential.
    D8 = 2 << Op0Shift,   D9 = 3 << Op0Shift,
    DA = 4 << Op0Shift,   DB = 5 << Op0Shift,
    DC = 6 << Op0Shift,   DD = 7 << Op0Shift,
    DE = 8 << Op0Shift,   DF = 9 << Op0Shift,

    //===------------------------------------------------------------------===//
    // This three-bit field describes the size of a memory operand.  Zero is
    // unused so that we can tell if we forgot to set a value.
    ArgShift = 10,
    ArgMask  = 7 << ArgShift,
    Arg8     = 1 << ArgShift,
    Arg16    = 2 << ArgShift,
    Arg32    = 3 << ArgShift,
    Arg64    = 4 << ArgShift,  // 64 bit int argument for FILD64
    ArgF32   = 5 << ArgShift,
    ArgF64   = 6 << ArgShift,
    ArgF80   = 7 << ArgShift,

    //===------------------------------------------------------------------===//
    // FP Instruction Classification...  Zero is non-fp instruction.

    // FPTypeMask - Mask for all of the FP types...
    FPTypeShift = 13,
    FPTypeMask  = 7 << FPTypeShift,

    // ZeroArgFP - 0 arg FP instruction which implicitly pushes ST(0), f.e. fld0
    ZeroArgFP  = 1 << FPTypeShift,

    // OneArgFP - 1 arg FP instructions which implicitly read ST(0), such as fst
    OneArgFP   = 2 << FPTypeShift,

    // OneArgFPRW - 1 arg FP instruction which implicitly read ST(0) and write a
    // result back to ST(0).  For example, fcos, fsqrt, etc.
    //
    OneArgFPRW = 3 << FPTypeShift,

    // TwoArgFP - 2 arg FP instructions which implicitly read ST(0), and an
    // explicit argument, storing the result to either ST(0) or the implicit
    // argument.  For example: fadd, fsub, fmul, etc...
    TwoArgFP   = 4 << FPTypeShift,

    // SpecialFP - Special instruction forms.  Dispatch by opcode explicitly.
    SpecialFP  = 5 << FPTypeShift,

    // PrintImplUses - Print out implicit uses in the assembly output.
    PrintImplUses = 1 << 16,

    OpcodeShift   = 17,
    OpcodeMask    = 0xFF << OpcodeShift,
    // Bits 25 -> 31 are unused
  };
}

class X86InstrInfo : public TargetInstrInfo {
  const X86RegisterInfo RI;
public:
  X86InstrInfo();

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  /// createNOPinstr - returns the target's implementation of NOP, which is
  /// usually a pseudo-instruction, implemented by a degenerate version of
  /// another instruction, e.g. X86: `xchg ax, ax'; SparcV9: `sethi r0, r0, r0'
  ///
  MachineInstr* createNOPinstr() const;

  /// isNOPinstr - not having a special NOP opcode, we need to know if a given
  /// instruction is interpreted as an `official' NOP instr, i.e., there may be
  /// more than one way to `do nothing' but only one canonical way to slack off.
  ///
  bool isNOPinstr(const MachineInstr &MI) const;

  // getBaseOpcodeFor - This function returns the "base" X86 opcode for the
  // specified opcode number.
  //
  unsigned char getBaseOpcodeFor(unsigned Opcode) const {
    return get(Opcode).TSFlags >> X86II::OpcodeShift;
  }
};

} // End llvm namespace

#endif
