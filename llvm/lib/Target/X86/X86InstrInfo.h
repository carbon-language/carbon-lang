//===- X86InstructionInfo.h - X86 Instruction Information ---------*-C++-*-===//
//
// This file contains the X86 implementation of the MachineInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86INSTRUCTIONINFO_H
#define X86INSTRUCTIONINFO_H

#include "llvm/Target/MachineInstrInfo.h"
#include "X86RegisterInfo.h"

/// X86II - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace X86II {
  enum {
    //===------------------------------------------------------------------===//
    // Instruction types.  These are the standard/most common forms for X86
    // instructions.
    //

    /// Raw - This form is for instructions that don't have any operands, so
    /// they are just a fixed opcode value, like 'leave'.
    RawFrm         = 0,
    
    /// AddRegFrm - This form is used for instructions like 'push r32' that have
    /// their one register operand added to their opcode.
    AddRegFrm      = 1,

    /// MRMDestReg - This form is used for instructions that use the Mod/RM byte
    /// to specify a destination, which in this case is a register.
    ///
    MRMDestReg     = 2,

    /// MRMDestMem - This form is used for instructions that use the Mod/RM byte
    /// to specify a destination, which in this case is memory.
    ///
    MRMDestMem     = 3,

    /// MRMSrcReg - This form is used for instructions that use the Mod/RM byte
    /// to specify a source, which in this case is a register.
    ///
    MRMSrcReg      = 4,

    /// MRMSrcMem - This form is used for instructions that use the Mod/RM byte
    /// to specify a source, which in this case is memory.
    ///
    MRMSrcMem      = 5,
  
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

    /// Void - Set if this instruction produces no value
    Void        = 1 << 5,

    // TB - TwoByte - Set if this instruction has a two byte opcode, which
    // starts with a 0x0F byte before the real opcode.
    TB          = 1 << 6,

    // FIXME: There are several more two byte opcode escapes: D8-DF
    // Handle this.

    // OpSize - Set if this instruction requires an operand size prefix (0x66),
    // which most often indicates that the instruction operates on 16 bit data
    // instead of 32 bit data.
    OpSize      = 1 << 7,

    // This three-bit field describes the size of a memory operand.
    // I'm just being paranoid not using the zero value; there's 
    // probably no reason you couldn't use it.
    Arg8     = 0x1 << 8,
    Arg16    = 0x2 << 8,
    Arg32    = 0x3 << 8,
    Arg64    = 0x4 << 8,
    Arg80    = 0x5 << 8,
    Arg128   = 0x6 << 8,
    ArgMask  = 0x7 << 8,
  };
}

class X86InstrInfo : public MachineInstrInfo {
  const X86RegisterInfo RI;
public:
  X86InstrInfo();

  /// getRegisterInfo - MachineInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  /// print - Print out an x86 instruction in intel syntax
  ///
  virtual void print(const MachineInstr *MI, std::ostream &O,
                     const TargetMachine &TM) const;

  // getBaseOpcodeFor - This function returns the "base" X86 opcode for the
  // specified opcode number.
  //
  unsigned char getBaseOpcodeFor(unsigned Opcode) const;



  //===--------------------------------------------------------------------===//
  //
  // These are stubs for pure virtual methods that should be factored out of
  // MachineInstrInfo.  We never call them, we don't want them, but we need
  // stubs so that we can instatiate our class.
  //
  MachineOpCode getNOPOpCode() const { abort(); }
  void CreateCodeToLoadConst(const TargetMachine& target, Function* F,
                             Value *V, Instruction *I,
                             std::vector<MachineInstr*>& mvec,
                             MachineCodeForInstruction& mcfi) const { abort(); }
  void CreateCodeToCopyIntToFloat(const TargetMachine& target,
                                  Function* F, Value* val, Instruction* dest,
                                  std::vector<MachineInstr*>& mvec,
                                  MachineCodeForInstruction& mcfi) const {
    abort();
  }
  void CreateCodeToCopyFloatToInt(const TargetMachine& target, Function* F,
                                  Value* val, Instruction* dest,
                                  std::vector<MachineInstr*>& mvec,
                                  MachineCodeForInstruction& mcfi)const {
    abort();
  }
  void CreateCopyInstructionsByType(const TargetMachine& target,
                                    Function* F, Value* src,
                                    Instruction* dest,
                                    std::vector<MachineInstr*>& mvec,
                                    MachineCodeForInstruction& mcfi)const {
    abort();
  }
  
  void CreateSignExtensionInstructions(const TargetMachine& target,
                                       Function* F, Value* srcVal,
                                       Value* destVal, unsigned numLowBits,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const {
    abort();
  }

  void CreateZeroExtensionInstructions(const TargetMachine& target,
                                       Function* F, Value* srcVal,
                                       Value* destVal, unsigned srcSizeInBits,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const {
    abort();
  }
};


#endif
