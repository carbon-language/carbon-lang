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

    /// Other - An instruction gets this form if it doesn't fit any of the
    /// catagories below.
    OtherFrm       = 0,

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
  
    /// TODO: Mod/RM that uses a fixed opcode extension, like /0

    FormMask       = 7,

    //===------------------------------------------------------------------===//
    // Actual flags...

    /// Void - Set if this instruction produces no value
    Void        = 1 << 3,

    // TB - TwoByte - Set if this instruction has a two byte opcode, which
    // starts with a 0x0F byte before the real opcode.
    TB          = 1 << 4,

    // OpSize - Set if this instruction requires an operand size prefix (0x66),
    // which most often indicates that the instruction operates on 16 bit data
    // instead of 32 bit data.
    OpSize      = 1 << 5,
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
