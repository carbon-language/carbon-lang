//===-- InstSelectSimple.cpp - A simple instruction selector for x86 ------===//
//
// This file defines a simple peephole instruction selector for the x86 platform
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstructionInfo.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/InstVisitor.h"
#include <map>

namespace {
  struct ISel : public InstVisitor<ISel> { // eventually will be a FunctionPass
    MachineFunction *F;                    // The function we are compiling into
    MachineBasicBlock *BB;                 // The current MBB we are compiling

    unsigned CurReg;
    std::map<Value*, unsigned> RegMap;  // Mapping between Val's and SSA Regs

    ISel(MachineFunction *f)
      : F(f), BB(0), CurReg(MRegisterInfo::FirstVirtualRegister) {}

    /// runOnFunction - Top level implementation of instruction selection for
    /// the entire function.
    ///
    bool runOnFunction(Function &F) {
      visit(F);
      RegMap.clear();
      return false;  // We never modify the LLVM itself.
    }

    /// visitBasicBlock - This method is called when we are visiting a new basic
    /// block.  This simply creates a new MachineBasicBlock to emit code into
    /// and adds it to the current MachineFunction.  Subsequent visit* for
    /// instructions will be invoked for all instructions in the basic block.
    ///
    void visitBasicBlock(BasicBlock &LLVM_BB) {
      BB = new MachineBasicBlock();
      // FIXME: Use the auto-insert form when it's available
      F->getBasicBlockList().push_back(BB);
    }

    // Visitation methods for various instructions.  These methods simply emit
    // fixed X86 code for each instruction.
    //
    void visitReturnInst(ReturnInst &RI);
    void visitAdd(BinaryOperator &B);

    void visitInstruction(Instruction &I) {
      std::cerr << "Cannot instruction select: " << I;
      abort();
    }

    
    /// copyConstantToRegister - Output the instructions required to put the
    /// specified constant into the specified register.
    ///
    void copyConstantToRegister(Constant *C, unsigned Reg);

    /// getReg - This method turns an LLVM value into a register number.  This
    /// is guaranteed to produce the same register number for a particular value
    /// every time it is queried.
    ///
    unsigned getReg(Value &V) { return getReg(&V); }  // Allow references
    unsigned getReg(Value *V) {
      unsigned &Reg = RegMap[V];
      if (Reg == 0)
        Reg = CurReg++;

      // If this operand is a constant, emit the code to copy the constant into
      // the register here...
      //
      if (Constant *C = dyn_cast<Constant>(V))
        copyConstantToRegister(C, Reg);

      return Reg;
    }

  };
}


/// copyConstantToRegister - Output the instructions required to put the
/// specified constant into the specified register.
///
void ISel::copyConstantToRegister(Constant *C, unsigned R) {
  assert (!isa<ConstantExpr>(C) && "Constant expressions not yet handled!\n");

  switch (C->getType()->getPrimitiveID()) {
  case Type::SByteTyID:
    BuildMI(BB, X86::MOVir8, R).addSImm(cast<ConstantSInt>(C)->getValue());
    break;
  case Type::UByteTyID:
    BuildMI(BB, X86::MOVir8, R).addZImm(cast<ConstantUInt>(C)->getValue());
    break;
  case Type::ShortTyID:
    BuildMI(BB, X86::MOVir16, R).addSImm(cast<ConstantSInt>(C)->getValue());
    break;
  case Type::UShortTyID:
    BuildMI(BB, X86::MOVir16, R).addZImm(cast<ConstantUInt>(C)->getValue());
    break;
  case Type::IntTyID:
    BuildMI(BB, X86::MOVir32, R).addSImm(cast<ConstantSInt>(C)->getValue());
    break;
  case Type::UIntTyID:
    BuildMI(BB, X86::MOVir32, R).addZImm(cast<ConstantUInt>(C)->getValue());
    break;
  default: assert(0 && "Type not handled yet!");      
  }
}


/// 'ret' instruction - Here we are interested in meeting the x86 ABI.  As such,
/// we have the following possibilities:
///
///   ret void: No return value, simply emit a 'ret' instruction
///   ret sbyte, ubyte : Extend value into EAX and return
///   ret short, ushort: Extend value into EAX and return
///   ret int, uint    : Move value into EAX and return
///   ret pointer      : Move value into EAX and return
///   ret long, ulong  : Move value into EAX/EDX (?) and return
///   ret float/double : ?  Top of FP stack?  XMM0?
///
void ISel::visitReturnInst(ReturnInst &I) {
  if (I.getNumOperands() != 0) {  // Not 'ret void'?
    // Move result into a hard register... then emit a ret
    visitInstruction(I);  // abort
  }

  // Emit a simple 'ret' instruction... appending it to the end of the basic
  // block
  BuildMI(BB, X86::RET, 0);
}


/// 'add' instruction - Simply turn this into an x86 reg,reg add instruction.
void ISel::visitAdd(BinaryOperator &B) {
  unsigned Op0r = getReg(B.getOperand(0)), Op1r = getReg(B.getOperand(1));
  unsigned DestReg = getReg(B);

  switch (B.getType()->getPrimitiveSize()) {
  case 1:   // UByte, SByte
    BuildMI(BB, X86::ADDrr8, DestReg).addReg(Op0r).addReg(Op1r);
    break;
  case 2:   // UShort, Short
    BuildMI(BB, X86::ADDrr16, DestReg).addReg(Op0r).addReg(Op1r);
    break;
  case 4:   // UInt, Int
    BuildMI(BB, X86::ADDrr32, DestReg).addReg(Op0r).addReg(Op1r);
    break;

  case 8:   // ULong, Long
  default:
    visitInstruction(B);  // abort
  }
}



/// X86SimpleInstructionSelection - This function converts an LLVM function into
/// a machine code representation is a very simple peep-hole fashion.  The
/// generated code sucks but the implementation is nice and simple.
///
MachineFunction *X86SimpleInstructionSelection(Function &F, TargetMachine &TM) {
  MachineFunction *Result = new MachineFunction(&F, TM);
  ISel(Result).runOnFunction(F);
  return Result;
}
