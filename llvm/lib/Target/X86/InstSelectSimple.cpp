//===-- InstSelectSimple.cpp - A simple instruction selector for x86 ------===//
//
// This file defines a simple peephole instruction selector for the x86 platform
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/InstVisitor.h"
#include <map>

namespace {
  struct ISel : public FunctionPass, InstVisitor<ISel> {
    TargetMachine &TM;
    MachineFunction *F;                    // The function we are compiling into
    MachineBasicBlock *BB;                 // The current MBB we are compiling

    unsigned CurReg;
    std::map<Value*, unsigned> RegMap;  // Mapping between Val's and SSA Regs

    ISel(TargetMachine &tm)
      : TM(tm), F(0), BB(0), CurReg(MRegisterInfo::FirstVirtualRegister) {}

    /// runOnFunction - Top level implementation of instruction selection for
    /// the entire function.
    ///
    bool runOnFunction(Function &Fn) {
      F = &MachineFunction::construct(&Fn, TM);
      visit(Fn);
      RegMap.clear();
      F = 0;
      return false;  // We never modify the LLVM itself.
    }

    /// visitBasicBlock - This method is called when we are visiting a new basic
    /// block.  This simply creates a new MachineBasicBlock to emit code into
    /// and adds it to the current MachineFunction.  Subsequent visit* for
    /// instructions will be invoked for all instructions in the basic block.
    ///
    void visitBasicBlock(BasicBlock &LLVM_BB) {
      BB = new MachineBasicBlock(&LLVM_BB);
      // FIXME: Use the auto-insert form when it's available
      F->getBasicBlockList().push_back(BB);
    }

    // Visitation methods for various instructions.  These methods simply emit
    // fixed X86 code for each instruction.
    //
    void visitReturnInst(ReturnInst &RI);
    void visitAdd(BinaryOperator &B);
    void visitShiftInst(ShiftInst &I);

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
    BuildMI(BB, X86::MOVir8, 1, R).addSImm(cast<ConstantSInt>(C)->getValue());
    break;
  case Type::UByteTyID:
    BuildMI(BB, X86::MOVir8, 1, R).addZImm(cast<ConstantUInt>(C)->getValue());
    break;
  case Type::ShortTyID:
    BuildMI(BB, X86::MOVir16, 1, R).addSImm(cast<ConstantSInt>(C)->getValue());
    break;
  case Type::UShortTyID:
    BuildMI(BB, X86::MOVir16, 1, R).addZImm(cast<ConstantUInt>(C)->getValue());
    break;
  case Type::IntTyID:
    BuildMI(BB, X86::MOVir32, 1, R).addSImm(cast<ConstantSInt>(C)->getValue());
    break;
  case Type::UIntTyID:
    BuildMI(BB, X86::MOVir32, 1, R).addZImm(cast<ConstantUInt>(C)->getValue());
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

/// SimpleLog2 - Compute and return Log2 of the input, valid only for inputs 1,
/// 2, 4, & 8.  Used to convert operand size into dense classes.
///
static inline unsigned SimpleLog2(unsigned N) {
  switch (N) {
  case 1: return 0;
  case 2: return 1;
  case 4: return 2;
  case 8: return 3;
  default: assert(0 && "Invalid operand to SimpleLog2!");
  }
  return 0;  // not reached
}

/// Shift instructions: 'shl', 'sar', 'shr' - Some special cases here
/// for constant immediate shift values, and for constant immediate
/// shift values equal to 1. Even the general case is sort of special,
/// because the shift amount has to be in CL, not just any old register.
///
void
ISel::visitShiftInst (ShiftInst & I)
{
  unsigned Op0r = getReg (I.getOperand (0));
  unsigned DestReg = getReg (I);
  bool isRightShift = (I.getOpcode () == Instruction::Shr);
  bool isOperandUnsigned = I.getType ()->isUnsigned ();
  unsigned OperandClass = SimpleLog2(I.getType()->getPrimitiveSize());

  if (ConstantUInt *CUI = dyn_cast <ConstantUInt> (I.getOperand (1)))
    {
      // The shift amount is constant, guaranteed to be a ubyte. Get its value.
      assert(CUI->getType() == Type::UByteTy && "Shift amount not a ubyte?");
      unsigned char shAmt = CUI->getValue();

      // Emit: <insn> reg, shamt  (shift-by-immediate opcode "ir" form.)
      if (isRightShift)
	{
	  if (isOperandUnsigned)
	    {
	      // This is a shift right logical (SHR).
	      switch (OperandClass)
		{
		case 0:
		  BuildMI (BB, X86::SHRir8, 2,
			   DestReg).addReg (Op0r).addZImm (shAmt);
		  break;
		case 1:
		  BuildMI (BB, X86::SHRir16, 2,
			   DestReg).addReg (Op0r).addZImm (shAmt);
		  break;
		case 2:
		  BuildMI (BB, X86::SHRir32, 2,
			   DestReg).addReg (Op0r).addZImm (shAmt);
		  break;
		case 3:
		default:
		  visitInstruction (I);
		  break;
		}
	    }
	  else
	    {
	      // This is a shift right arithmetic (SAR).
	      switch (OperandClass)
		{
		case 0:
		  BuildMI (BB, X86::SARir8, 2,
			   DestReg).addReg (Op0r).addZImm (shAmt);
		  break;
		case 1:
		  BuildMI (BB, X86::SARir16, 2,
			   DestReg).addReg (Op0r).addZImm (shAmt);
		  break;
		case 2:
		  BuildMI (BB, X86::SARir32, 2,
			   DestReg).addReg (Op0r).addZImm (shAmt);
		  break;
		case 3:
		default:
		  visitInstruction (I);
		  break;
		}
	    }
	}
      else
	{
	  // This is a left shift (SHL).
	  switch (OperandClass)
	    {
	    case 0:
	      BuildMI (BB, X86::SHLir8, 2,
		       DestReg).addReg (Op0r).addZImm (shAmt);
	      break;
	    case 1:
	      BuildMI (BB, X86::SHLir16, 2,
		       DestReg).addReg (Op0r).addZImm (shAmt);
	      break;
	    case 2:
	      BuildMI (BB, X86::SHLir32, 2,
		       DestReg).addReg (Op0r).addZImm (shAmt);
	      break;
	    case 3:
	    default:
	      visitInstruction (I);
	      break;
	    }
	}
    }
  else
    {
      // The shift amount is non-constant.
      //
      // In fact, you can only shift with a variable shift amount if
      // that amount is already in the CL register, so we have to put it
      // there first.
      //
      // Get it from the register it's in.
      unsigned Op1r = getReg (I.getOperand (1));
      // Emit: move cl, shiftAmount (put the shift amount in CL.)
      BuildMI (BB, X86::MOVrr8, 2, X86::CL).addReg (Op1r);
      // Emit: <insn> reg, cl       (shift-by-CL opcode; "rr" form.)
      if (isRightShift)
	{
	  if (OperandClass)
	    {
	      // This is a shift right logical (SHR).
	      switch (OperandClass)
		{
		case 0:
		  BuildMI (BB, X86::SHRrr8, 2,
			   DestReg).addReg (Op0r).addReg (X86::CL);
		  break;
		case 1:
		  BuildMI (BB, X86::SHRrr16, 2,
			   DestReg).addReg (Op0r).addReg (X86::CL);
		  break;
		case 2:
		  BuildMI (BB, X86::SHRrr32, 2,
			   DestReg).addReg (Op0r).addReg (X86::CL);
		  break;
		case 3:
		default:
		  visitInstruction (I);
		  break;
		}
	    }
	  else
	    {
	      // This is a shift right arithmetic (SAR).
	      switch (OperandClass)
		{
		case 0:
		  BuildMI (BB, X86::SARrr8, 2,
			   DestReg).addReg (Op0r).addReg (X86::CL);
		  break;
		case 1:
		  BuildMI (BB, X86::SARrr16, 2,
			   DestReg).addReg (Op0r).addReg (X86::CL);
		  break;
		case 2:
		  BuildMI (BB, X86::SARrr32, 2,
			   DestReg).addReg (Op0r).addReg (X86::CL);
		  break;
		case 3:
		default:
		  visitInstruction (I);
		  break;
		}
	    }
	}
      else
	{
	  // This is a left shift (SHL).
	  switch (OperandClass)
	    {
	    case 0:
	      BuildMI (BB, X86::SHLrr8, 2,
		       DestReg).addReg (Op0r).addReg (X86::CL);
	      break;
	    case 1:
	      BuildMI (BB, X86::SHLrr16, 2,
		       DestReg).addReg (Op0r).addReg (X86::CL);
	      break;
	    case 2:
	      BuildMI (BB, X86::SHLrr32, 2,
		       DestReg).addReg (Op0r).addReg (X86::CL);
	      break;
	    case 3:
	    default:
	      visitInstruction (I);
	      break;
	    }
	}
    }
}


/// 'add' instruction - Simply turn this into an x86 reg,reg add instruction.
void ISel::visitAdd(BinaryOperator &B) {
  unsigned Op0r = getReg(B.getOperand(0)), Op1r = getReg(B.getOperand(1));
  unsigned DestReg = getReg(B);

  switch (B.getType()->getPrimitiveSize()) {
  case 1:   // UByte, SByte
    BuildMI(BB, X86::ADDrr8, 2, DestReg).addReg(Op0r).addReg(Op1r);
    break;
  case 2:   // UShort, Short
    BuildMI(BB, X86::ADDrr16, 2, DestReg).addReg(Op0r).addReg(Op1r);
    break;
  case 4:   // UInt, Int
    BuildMI(BB, X86::ADDrr32, 2, DestReg).addReg(Op0r).addReg(Op1r);
    break;
  case 8:   // ULong, Long
    // Here we have a pair of operands each occupying a pair of registers.
    // We need to do an ADDrr32 of the least-significant pair immediately
    // followed by an ADCrr32 (Add with Carry) of the most-significant pair.
    // I don't know how we are representing these multi-register arguments.
  default:
    visitInstruction(B);  // abort
  }
}



/// createSimpleX86InstructionSelector - This pass converts an LLVM function
/// into a machine code representation is a very simple peep-hole fashion.  The
/// generated code sucks but the implementation is nice and simple.
///
Pass *createSimpleX86InstructionSelector(TargetMachine &TM) {
  return new ISel(TM);
}
