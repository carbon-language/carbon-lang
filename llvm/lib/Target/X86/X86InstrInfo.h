//===- X86InstructionInfo.h - X86 Instruction Information ---------*-C++-*-===//
//
// This file contains the X86 implementation of the MInstructionInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86INSTRUCTIONINFO_H
#define X86INSTRUCTIONINFO_H

#include "llvm/Target/MachineInstrInfo.h"
#include "X86RegisterInfo.h"

class X86InstructionInfo : public MachineInstrInfo {
  const X86RegisterInfo RI;
public:
  X86InstructionInfo();

  /// getRegisterInfo - MInstructionInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  /// print - Print out an x86 instruction in GAS syntax
  ///
  virtual void print(const MachineInstr *MI, std::ostream &O) const;


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
