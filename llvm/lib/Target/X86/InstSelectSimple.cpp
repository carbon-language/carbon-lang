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
#include "llvm/iPHINode.h"
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
    void visitBranchInst(BranchInst &BI);

    // Arithmetic operators
    void visitAdd(BinaryOperator &B);

    // Bitwise operators
    void visitAnd(BinaryOperator &B) { visitBitwise(B, 0); }
    void visitOr (BinaryOperator &B) { visitBitwise(B, 1); }
    void visitXor(BinaryOperator &B) { visitBitwise(B, 2); }
    void visitBitwise(BinaryOperator &B, unsigned OpcodeClass);

    // Binary comparison operators

    // Other operators
    void visitShiftInst(ShiftInst &I);
    void visitPHINode(PHINode &I);

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

/// getClass - Turn a primitive type into a "class" number which is based on the
/// size of the type, and whether or not it is floating point.
///
static inline unsigned getClass(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::SByteTyID:
  case Type::UByteTyID:   return 0;          // Byte operands are class #0
  case Type::ShortTyID:
  case Type::UShortTyID:  return 1;          // Short operands are class #1
  case Type::IntTyID:
  case Type::UIntTyID:
  case Type::PointerTyID: return 2;          // Int's and pointers are class #2

  case Type::LongTyID:
  case Type::ULongTyID:   return 3;          // Longs are class #3
  case Type::FloatTyID:   return 4;          // Float is class #4
  case Type::DoubleTyID:  return 5;          // Doubles are class #5
  default:
    assert(0 && "Invalid type to getClass!");
    return 0;  // not reached
  }
}

/// copyConstantToRegister - Output the instructions required to put the
/// specified constant into the specified register.
///
void ISel::copyConstantToRegister(Constant *C, unsigned R) {
  assert (!isa<ConstantExpr>(C) && "Constant expressions not yet handled!\n");

  if (C->getType()->isIntegral()) {
    unsigned Class = getClass(C->getType());
    assert(Class != 3 && "Type not handled yet!");

    static const unsigned IntegralOpcodeTab[] = {
      X86::MOVir8, X86::MOVir16, X86::MOVir32
    };

    if (C->getType()->isSigned()) {
      ConstantSInt *CSI = cast<ConstantSInt>(C);
      BuildMI(BB, IntegralOpcodeTab[Class], 1, R).addSImm(CSI->getValue());
    } else {
      ConstantUInt *CUI = cast<ConstantUInt>(C);
      BuildMI(BB, IntegralOpcodeTab[Class], 1, R).addZImm(CUI->getValue());
    }
  } else {
    assert(0 && "Type not handled yet!");
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

/// visitBranchInst - Handle conditional and unconditional branches here.  Note
/// that since code layout is frozen at this point, that if we are trying to
/// jump to a block that is the immediate successor of the current block, we can
/// just make a fall-through. (but we don't currently).
///
void ISel::visitBranchInst(BranchInst &BI) {
  if (BI.isConditional())   // Only handles unconditional branches so far...
    visitInstruction(BI);

  BuildMI(BB, X86::JMP, 1).addPCDisp(BI.getSuccessor(0));
}



/// 'add' instruction - Simply turn this into an x86 reg,reg add instruction.
void ISel::visitAdd(BinaryOperator &B) {
  unsigned Op0r = getReg(B.getOperand(0)), Op1r = getReg(B.getOperand(1));
  unsigned DestReg = getReg(B);
  unsigned Class = getClass(B.getType());

  static const unsigned Opcodes[] = { X86::ADDrr8, X86::ADDrr16, X86::ADDrr32 };

  if (Class >= sizeof(Opcodes)/sizeof(Opcodes[0]))
    visitInstruction(B);  // Not handled class yet...

  BuildMI(BB, Opcodes[Class], 2, DestReg).addReg(Op0r).addReg(Op1r);

  // For Longs: Here we have a pair of operands each occupying a pair of
  // registers.  We need to do an ADDrr32 of the least-significant pair
  // immediately followed by an ADCrr32 (Add with Carry) of the most-significant
  // pair.  I don't know how we are representing these multi-register arguments.
}

/// visitBitwise - Implement the three bitwise operators for integral types...
/// OperatorClass is one of: 0 for And, 1 for Or, 2 for Xor.
void ISel::visitBitwise(BinaryOperator &B, unsigned OperatorClass) {
  if (B.getType() == Type::BoolTy)  // FIXME: Handle bools
    visitInstruction(B);

  unsigned Class = getClass(B.getType());
  if (Class > 2)  // FIXME: Handle longs
    visitInstruction(B);

  static const unsigned OpcodeTab[][4] = {
    { X86::ANDrr8, X86::ANDrr16, X86::ANDrr32, 0 },  // AND
    { X86:: ORrr8, X86:: ORrr16, X86:: ORrr32, 0 },  // OR
    { X86::XORrr8, X86::XORrr16, X86::XORrr32, 0 },  // XOR
  };
  
  unsigned Opcode = OpcodeTab[OperatorClass][Class];
  unsigned Op0r = getReg(B.getOperand(0));
  unsigned Op1r = getReg(B.getOperand(1));
  BuildMI(BB, Opcode, 2, getReg(B)).addReg(Op0r).addReg(Op1r);
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
  bool isLeftShift = I.getOpcode() == Instruction::Shl;
  bool isOperandSigned = I.getType()->isUnsigned();
  unsigned OperandClass = getClass(I.getType());

  if (OperandClass > 2)
    visitInstruction(I); // Can't handle longs yet!

  if (ConstantUInt *CUI = dyn_cast <ConstantUInt> (I.getOperand (1)))
    {
      // The shift amount is constant, guaranteed to be a ubyte. Get its value.
      assert(CUI->getType() == Type::UByteTy && "Shift amount not a ubyte?");
      unsigned char shAmt = CUI->getValue();

      static const unsigned ConstantOperand[][4] = {
        { X86::SHRir8, X86::SHRir16, X86::SHRir32, 0 },  // SHR
        { X86::SARir8, X86::SARir16, X86::SARir32, 0 },  // SAR
        { X86::SHLir8, X86::SHLir16, X86::SHLir32, 0 },  // SHL
        { X86::SHLir8, X86::SHLir16, X86::SHLir32, 0 },  // SAL = SHL
      };

      const unsigned *OpTab = // Figure out the operand table to use
        ConstantOperand[isLeftShift*2+isOperandSigned];

      // Emit: <insn> reg, shamt  (shift-by-immediate opcode "ir" form.)
      BuildMI(BB, OpTab[OperandClass], 2, DestReg).addReg(Op0r).addZImm(shAmt);
    }
  else
    {
      // The shift amount is non-constant.
      //
      // In fact, you can only shift with a variable shift amount if
      // that amount is already in the CL register, so we have to put it
      // there first.
      //

      // Emit: move cl, shiftAmount (put the shift amount in CL.)
      BuildMI (BB, X86::MOVrr8, 2, X86::CL).addReg(getReg(I.getOperand(1)));

      // This is a shift right (SHR).
      static const unsigned NonConstantOperand[][4] = {
        { X86::SHRrr8, X86::SHRrr16, X86::SHRrr32, 0 },  // SHR
        { X86::SARrr8, X86::SARrr16, X86::SARrr32, 0 },  // SAR
        { X86::SHLrr8, X86::SHLrr16, X86::SHLrr32, 0 },  // SHL
        { X86::SHLrr8, X86::SHLrr16, X86::SHLrr32, 0 },  // SAL = SHL
      };

      const unsigned *OpTab = // Figure out the operand table to use
        NonConstantOperand[isLeftShift*2+isOperandSigned];

      BuildMI(BB, OpTab[OperandClass], 2, DestReg).addReg(Op0r).addReg(X86::CL);
    }
}

/// visitPHINode - Turn an LLVM PHI node into an X86 PHI node...
///
void ISel::visitPHINode(PHINode &PN) {
  MachineInstr *MI = BuildMI(BB, X86::PHI, PN.getNumOperands(), getReg(PN));

  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    // FIXME: This will put constants after the PHI nodes in the block, which
    // is invalid.  They should be put inline into the PHI node eventually.
    //
    MI->addRegOperand(getReg(PN.getIncomingValue(i)));
    MI->addPCDispOperand(PN.getIncomingBlock(i));
  }
}


/// createSimpleX86InstructionSelector - This pass converts an LLVM function
/// into a machine code representation is a very simple peep-hole fashion.  The
/// generated code sucks but the implementation is nice and simple.
///
Pass *createSimpleX86InstructionSelector(TargetMachine &TM) {
  return new ISel(TM);
}
