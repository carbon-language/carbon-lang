//===-- SparcV9InstrInfo.h - Define TargetInstrInfo for SparcV9 -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This class contains information about individual instructions.
// Most information is stored in the SparcV9MachineInstrDesc array above.
// Other information is computed on demand, and most such functions
// default to member functions in base class TargetInstrInfo. 
//
//===----------------------------------------------------------------------===//

#ifndef SPARC_INSTRINFO_H
#define SPARC_INSTRINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "SparcV9Internals.h"

namespace llvm {

struct SparcV9InstrInfo : public TargetInstrInfo {
  SparcV9InstrInfo();

  // All immediate constants are in position 1 except the
  // store instructions and SETxx.
  // 
  virtual int getImmedConstantPos(MachineOpCode opCode) const {
    bool ignore;
    if (this->maxImmedConstant(opCode, ignore) != 0) {
      // 1st store opcode
      assert(! this->isStore((MachineOpCode) V9::STBr - 1));
      // last store opcode
      assert(! this->isStore((MachineOpCode) V9::STXFSRi + 1));

      if (opCode == V9::SETSW || opCode == V9::SETUW ||
          opCode == V9::SETX  || opCode == V9::SETHI)
        return 0;
      if (opCode >= V9::STBr && opCode <= V9::STXFSRi)
        return 2;
      return 1;
    }
    else
      return -1;
  }

  virtual bool hasResultInterlock(MachineOpCode opCode) const
  {
    // All UltraSPARC instructions have interlocks (note that delay slots
    // are not considered here).
    // However, instructions that use the result of an FCMP produce a
    // 9-cycle stall if they are issued less than 3 cycles after the FCMP.
    // Force the compiler to insert a software interlock (i.e., gap of
    // 2 other groups, including NOPs if necessary).
    return (opCode == V9::FCMPS || opCode == V9::FCMPD || opCode == V9::FCMPQ);
  }

  //-------------------------------------------------------------------------
  // Queries about representation of LLVM quantities (e.g., constants)
  //-------------------------------------------------------------------------

  virtual bool ConstantMayNotFitInImmedField(const Constant* CV,
                                             const Instruction* I) const;

  //-------------------------------------------------------------------------
  // Code generation support for creating individual machine instructions
  //-------------------------------------------------------------------------

  // Get certain common op codes for the current target.  This and all the
  // Create* methods below should be moved to a machine code generation class
  // 
  virtual MachineOpCode getNOPOpCode() const { return V9::NOP; }

  // Get the value of an integral constant in the form that must
  // be put into the machine register.  The specified constant is interpreted
  // as (i.e., converted if necessary to) the specified destination type.  The
  // result is always returned as an uint64_t, since the representation of
  // int64_t and uint64_t are identical.  The argument can be any known const.
  // 
  // isValidConstant is set to true if a valid constant was found.
  // 
  virtual uint64_t ConvertConstantToIntType(const TargetMachine &target,
                                            const Value *V,
                                            const Type *destType,
                                            bool  &isValidConstant) const;

  // Create an instruction sequence to put the constant `val' into
  // the virtual register `dest'.  `val' may be a Constant or a
  // GlobalValue, viz., the constant address of a global variable or function.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void  CreateCodeToLoadConst(const TargetMachine& target,
                                      Function* F,
                                      Value* val,
                                      Instruction* dest,
                                      std::vector<MachineInstr*>& mvec,
                                      MachineCodeForInstruction& mcfi) const;

  // Create an instruction sequence to copy an integer value `val'
  // to a floating point value `dest' by copying to memory and back.
  // val must be an integral type.  dest must be a Float or Double.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void  CreateCodeToCopyIntToFloat(const TargetMachine& target,
                                       Function* F,
                                       Value* val,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const;

  // Similarly, create an instruction sequence to copy an FP value
  // `val' to an integer value `dest' by copying to memory and back.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void  CreateCodeToCopyFloatToInt(const TargetMachine& target,
                                       Function* F,
                                       Value* val,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const;
  
  // Create instruction(s) to copy src to dest, for arbitrary types
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateCopyInstructionsByType(const TargetMachine& target,
                                       Function* F,
                                       Value* src,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const;

  // Create instruction sequence to produce a sign-extended register value
  // from an arbitrary sized value (sized in bits, not bytes).
  // The generated instructions are appended to `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateSignExtensionInstructions(const TargetMachine& target,
                                       Function* F,
                                       Value* srcVal,
                                       Value* destVal,
                                       unsigned int numLowBits,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const;

  // Create instruction sequence to produce a zero-extended register value
  // from an arbitrary sized value (sized in bits, not bytes).
  // The generated instructions are appended to `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateZeroExtensionInstructions(const TargetMachine& target,
                                       Function* F,
                                       Value* srcVal,
                                       Value* destVal,
                                       unsigned int numLowBits,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const;
};

} // End llvm namespace

#endif
