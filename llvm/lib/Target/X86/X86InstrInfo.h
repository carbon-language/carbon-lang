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
  
    /// MRM[0-7][rm] - These forms are used to represent instructions that use
    /// a Mod/RM byte, and use the middle field to hold extended opcode
    /// information.  In the intel manual these are represented as /0, /1, ...
    ///

    // First, instructions that operate on a register r/m operand...
    MRM0r = 16,  MRM1r = 17,  MRM2r = 18,  MRM3r = 19, // Format /0 /1 /2 /3
    MRM4r = 20,  MRM5r = 21,  MRM6r = 22,  MRM7r = 23, // Format /4 /5 /6 /7

    // Next, instructions that operate on a memory r/m operand...
    MRM0m = 24,  MRM1m = 25,  MRM2m = 26,  MRM3m = 27, // Format /0 /1 /2 /3
    MRM4m = 28,  MRM5m = 29,  MRM6m = 30,  MRM7m = 31, // Format /4 /5 /6 /7

    FormMask       = 31,

    //===------------------------------------------------------------------===//
    // Actual flags...

    // OpSize - Set if this instruction requires an operand size prefix (0x66),
    // which most often indicates that the instruction operates on 16 bit data
    // instead of 32 bit data.
    OpSize      = 1 << 5,

    // Op0Mask - There are several prefix bytes that are used to form two byte
    // opcodes.  These are currently 0x0F, 0xF3, and 0xD8-0xDF.  This mask is
    // used to obtain the setting of this field.  If no bits in this field is
    // set, there is no prefix byte for obtaining a multibyte opcode.
    //
    Op0Shift    = 6,
    Op0Mask     = 0xF << Op0Shift,

    // TB - TwoByte - Set if this instruction has a two byte opcode, which
    // starts with a 0x0F byte before the real opcode.
    TB          = 1 << Op0Shift,

    // REP - The 0xF3 prefix byte indicating repetition of the following
    // instruction.
    REP         = 2 << Op0Shift,

    // D8-DF - These escape opcodes are used by the floating point unit.  These
    // values must remain sequential.
    D8 = 3 << Op0Shift,   D9 = 4 << Op0Shift,
    DA = 5 << Op0Shift,   DB = 6 << Op0Shift,
    DC = 7 << Op0Shift,   DD = 8 << Op0Shift,
    DE = 9 << Op0Shift,   DF = 10 << Op0Shift,

    //===------------------------------------------------------------------===//
    // This three-bit field describes the size of a memory operand.  Zero is
    // unused so that we can tell if we forgot to set a value.
    MemShift = 10,
    MemMask  = 7 << MemShift,
    Mem8     = 1 << MemShift,
    Mem16    = 2 << MemShift,
    Mem32    = 3 << MemShift,
    Mem64    = 4 << MemShift,
    Mem80    = 5 << MemShift,
    Mem128   = 6 << MemShift,

    //===------------------------------------------------------------------===//
    // This tow-bit field describes the size of an immediate operand.  Zero is
    // unused so that we can tell if we forgot to set a value.
    ImmShift = 13,
    ImmMask  = 7 << ImmShift,
    Imm8     = 1 << ImmShift,
    Imm16    = 2 << ImmShift,
    Imm32    = 3 << ImmShift,

    //===------------------------------------------------------------------===//
    // FP Instruction Classification...  Zero is non-fp instruction.

    // FPTypeMask - Mask for all of the FP types...
    FPTypeShift = 15,
    FPTypeMask  = 7 << FPTypeShift,

    // NotFP - The default, set for instructions that do not use FP registers.
    NotFP      = 0 << FPTypeShift,

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

    // CondMovFP - "2 operand" floating point conditional move instructions.
    CondMovFP  = 5 << FPTypeShift,

    // SpecialFP - Special instruction forms.  Dispatch by opcode explicitly.
    SpecialFP  = 6 << FPTypeShift,

    // PrintImplUsesAfter - Print out implicit uses in the assembly output after
    // the normal operands.
    PrintImplUsesAfter = 1 << 18,

    // PrintImplUsesBefore - Print out implicit uses in the assembly output before
    // the normal operands.
    PrintImplUsesBefore = 1 << 19,

    OpcodeShift   = 20,
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

  //
  // Return true if the instruction is a register to register move and
  // leave the source and dest operands in the passed parameters.
  //
  virtual bool isMoveInstr(const MachineInstr& MI,
                           unsigned& sourceReg,
                           unsigned& destReg) const;

  // getBaseOpcodeFor - This function returns the "base" X86 opcode for the
  // specified opcode number.
  //
  unsigned char getBaseOpcodeFor(unsigned Opcode) const {
    return get(Opcode).TSFlags >> X86II::OpcodeShift;
  }
};

} // End llvm namespace

#endif
