//===-- InstSelectSimple.cpp - A simple instruction selector for x86 ------===//
//
// This file defines a simple peephole instruction selector for the x86 platform
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86InstrBuilder.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/iOperators.h"
#include "llvm/iOther.h"
#include "llvm/iPHINode.h"
#include "llvm/iMemory.h"
#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Target/MRegisterInfo.h"
#include <map>

using namespace MOTy;  // Get Use, Def, UseAndDef

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
    void visitSimpleBinary(BinaryOperator &B, unsigned OpcodeClass);
    void visitAdd(BinaryOperator &B) { visitSimpleBinary(B, 0); }
    void visitSub(BinaryOperator &B) { visitSimpleBinary(B, 1); }
    void visitMul(BinaryOperator &B);

    void visitDiv(BinaryOperator &B) { visitDivRem(B); }
    void visitRem(BinaryOperator &B) { visitDivRem(B); }
    void visitDivRem(BinaryOperator &B);

    // Bitwise operators
    void visitAnd(BinaryOperator &B) { visitSimpleBinary(B, 2); }
    void visitOr (BinaryOperator &B) { visitSimpleBinary(B, 3); }
    void visitXor(BinaryOperator &B) { visitSimpleBinary(B, 4); }

    // Binary comparison operators
    void visitSetCCInst(SetCondInst &I, unsigned OpNum);
    void visitSetEQ(SetCondInst &I) { visitSetCCInst(I, 0); }
    void visitSetNE(SetCondInst &I) { visitSetCCInst(I, 1); }
    void visitSetLT(SetCondInst &I) { visitSetCCInst(I, 2); }
    void visitSetGT(SetCondInst &I) { visitSetCCInst(I, 3); }
    void visitSetLE(SetCondInst &I) { visitSetCCInst(I, 4); }
    void visitSetGE(SetCondInst &I) { visitSetCCInst(I, 5); }

    // Memory Instructions
    void visitLoadInst(LoadInst &I);
    void visitStoreInst(StoreInst &I);

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
      if (Reg == 0) {
        Reg = CurReg++;
        RegMap[V] = Reg;

        // Add the mapping of regnumber => reg class to MachineFunction
        F->addRegMap(Reg,
                     TM.getRegisterInfo()->getRegClassForType(V->getType()));
      }

      // If this operand is a constant, emit the code to copy the constant into
      // the register here...
      //
      if (Constant *C = dyn_cast<Constant>(V))
        copyConstantToRegister(C, Reg);

      return Reg;
    }
  };
}

/// TypeClass - Used by the X86 backend to group LLVM types by their basic X86
/// Representation.
///
enum TypeClass {
  cByte, cShort, cInt, cLong, cFloat, cDouble
};

/// getClass - Turn a primitive type into a "class" number which is based on the
/// size of the type, and whether or not it is floating point.
///
static inline TypeClass getClass(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::SByteTyID:
  case Type::UByteTyID:   return cByte;      // Byte operands are class #0
  case Type::ShortTyID:
  case Type::UShortTyID:  return cShort;     // Short operands are class #1
  case Type::IntTyID:
  case Type::UIntTyID:
  case Type::PointerTyID: return cInt;       // Int's and pointers are class #2

  case Type::LongTyID:
  case Type::ULongTyID:   return cLong;      // Longs are class #3
  case Type::FloatTyID:   return cFloat;     // Float is class #4
  case Type::DoubleTyID:  return cDouble;    // Doubles are class #5
  default:
    assert(0 && "Invalid type to getClass!");
    return cByte;  // not reached
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


/// SetCC instructions - Here we just emit boilerplate code to set a byte-sized
/// register, then move it to wherever the result should be. 
/// We handle FP setcc instructions by pushing them, doing a
/// compare-and-pop-twice, and then copying the concodes to the main
/// processor's concodes (I didn't make this up, it's in the Intel manual)
///
void ISel::visitSetCCInst(SetCondInst &I, unsigned OpNum) {
  // The arguments are already supposed to be of the same type.
  const Type *CompTy = I.getOperand(0)->getType();
  unsigned reg1 = getReg(I.getOperand(0));
  unsigned reg2 = getReg(I.getOperand(1));

  unsigned Class = getClass(CompTy);
  switch (Class) {
    // Emit: cmp <var1>, <var2> (do the comparison).  We can
    // compare 8-bit with 8-bit, 16-bit with 16-bit, 32-bit with
    // 32-bit.
  case cByte:
    BuildMI (BB, X86::CMPrr8, 2).addReg (reg1).addReg (reg2);
    break;
  case cShort:
    BuildMI (BB, X86::CMPrr16, 2).addReg (reg1).addReg (reg2);
    break;
  case cInt:
    BuildMI (BB, X86::CMPrr32, 2).addReg (reg1).addReg (reg2);
    break;

    // Push the variables on the stack with fldl opcodes.
    // FIXME: assuming var1, var2 are in memory, if not, spill to
    // stack first
  case cFloat:  // Floats
    BuildMI (BB, X86::FLDr4, 1, X86::NoReg).addReg (reg1);
    BuildMI (BB, X86::FLDr4, 1, X86::NoReg).addReg (reg2);
    break;
  case cDouble:  // Doubles
    BuildMI (BB, X86::FLDr8, 1, X86::NoReg).addReg (reg1);
    BuildMI (BB, X86::FLDr8, 1, X86::NoReg).addReg (reg2);
    break;
  case cLong:
  default:
    visitInstruction(I);
  }

  if (CompTy->isFloatingPoint()) {
    // (Non-trapping) compare and pop twice.
    BuildMI (BB, X86::FUCOMPP, 0);
    // Move fp status word (concodes) to ax.
    BuildMI (BB, X86::FNSTSWr8, 1, X86::AX);
    // Load real concodes from ax.
    BuildMI (BB, X86::SAHF, 1).addReg(X86::AH);
  }

  // Emit setOp instruction (extract concode; clobbers ax),
  // using the following mapping:
  // LLVM  -> X86 signed  X86 unsigned
  // -----    -----       -----
  // seteq -> sete        sete
  // setne -> setne       setne
  // setlt -> setl        setb
  // setgt -> setg        seta
  // setle -> setle       setbe
  // setge -> setge       setae

  static const unsigned OpcodeTab[2][6] = {
    {X86::SETEr, X86::SETNEr, X86::SETBr, X86::SETAr, X86::SETBEr, X86::SETAEr},
    {X86::SETEr, X86::SETNEr, X86::SETLr, X86::SETGr, X86::SETLEr, X86::SETGEr},
  };

  BuildMI(BB, OpcodeTab[CompTy->isSigned()][OpNum], 0, X86::AL);
  
  // Put it in the result using a move.
  BuildMI (BB, X86::MOVrr8, 1, getReg(I)).addReg(X86::AL);
}


/// 'ret' instruction - Here we are interested in meeting the x86 ABI.  As such,
/// we have the following possibilities:
///
///   ret void: No return value, simply emit a 'ret' instruction
///   ret sbyte, ubyte : Extend value into EAX and return
///   ret short, ushort: Extend value into EAX and return
///   ret int, uint    : Move value into EAX and return
///   ret pointer      : Move value into EAX and return
///   ret long, ulong  : Move value into EAX/EDX and return
///   ret float/double : Top of FP stack
///
void ISel::visitReturnInst (ReturnInst &I) {
  if (I.getNumOperands() == 0) {
    // Emit a 'ret' instruction
    BuildMI(BB, X86::RET, 0);
    return;
  }

  unsigned val = getReg(I.getOperand(0));
  unsigned Class = getClass(I.getOperand(0)->getType());
  bool isUnsigned = I.getOperand(0)->getType()->isUnsigned();
  switch (Class) {
  case cByte:
    // ret sbyte, ubyte: Extend value into EAX and return
    if (isUnsigned)
      BuildMI (BB, X86::MOVZXr32r8, 1, X86::EAX).addReg (val);
    else
      BuildMI (BB, X86::MOVSXr32r8, 1, X86::EAX).addReg (val);
    break;
  case cShort:
    // ret short, ushort: Extend value into EAX and return
    if (isUnsigned)
      BuildMI (BB, X86::MOVZXr32r16, 1, X86::EAX).addReg (val);
    else
      BuildMI (BB, X86::MOVSXr32r16, 1, X86::EAX).addReg (val);
    break;
  case cInt:
    // ret int, uint, ptr: Move value into EAX and return
    // MOV EAX, <val>
    BuildMI(BB, X86::MOVrr32, 1, X86::EAX).addReg(val);
    break;

    // ret float/double: top of FP stack
    // FLD <val>
  case cFloat:  // Floats
    BuildMI(BB, X86::FLDr4, 1).addReg(val);
    break;
  case cDouble:  // Doubles
    BuildMI(BB, X86::FLDr8, 1).addReg(val);
    break;
  case cLong:
    // ret long: use EAX(least significant 32 bits)/EDX (most
    // significant 32)...uh, I think so Brain, but how do i call
    // up the two parts of the value from inside this mouse
    // cage? *zort*
  default:
    visitInstruction(I);
  }

  // Emit a 'ret' instruction
  BuildMI(BB, X86::RET, 0);
}

/// visitBranchInst - Handle conditional and unconditional branches here.  Note
/// that since code layout is frozen at this point, that if we are trying to
/// jump to a block that is the immediate successor of the current block, we can
/// just make a fall-through. (but we don't currently).
///
void
ISel::visitBranchInst (BranchInst & BI)
{
  if (BI.isConditional ())
    {
      BasicBlock *ifTrue = BI.getSuccessor (0);
      BasicBlock *ifFalse = BI.getSuccessor (1); // this is really unobvious 

      // simplest thing I can think of: compare condition with zero,
      // followed by jump-if-equal to ifFalse, and jump-if-nonequal to
      // ifTrue
      unsigned int condReg = getReg (BI.getCondition ());
      BuildMI (BB, X86::CMPri8, 2).addReg (condReg).addZImm (0);
      BuildMI (BB, X86::JNE, 1).addPCDisp (BI.getSuccessor (0));
      BuildMI (BB, X86::JE, 1).addPCDisp (BI.getSuccessor (1));
    }
  else // unconditional branch
    {
      BuildMI (BB, X86::JMP, 1).addPCDisp (BI.getSuccessor (0));
    }
}


/// visitSimpleBinary - Implement simple binary operators for integral types...
/// OperatorClass is one of: 0 for Add, 1 for Sub, 2 for And, 3 for Or,
/// 4 for Xor.
///
void ISel::visitSimpleBinary(BinaryOperator &B, unsigned OperatorClass) {
  if (B.getType() == Type::BoolTy)  // FIXME: Handle bools for logicals
    visitInstruction(B);

  unsigned Class = getClass(B.getType());
  if (Class > 2)  // FIXME: Handle longs
    visitInstruction(B);

  static const unsigned OpcodeTab[][4] = {
    // Arithmetic operators
    { X86::ADDrr8, X86::ADDrr16, X86::ADDrr32, 0 },  // ADD
    { X86::SUBrr8, X86::SUBrr16, X86::SUBrr32, 0 },  // SUB

    // Bitwise operators
    { X86::ANDrr8, X86::ANDrr16, X86::ANDrr32, 0 },  // AND
    { X86:: ORrr8, X86:: ORrr16, X86:: ORrr32, 0 },  // OR
    { X86::XORrr8, X86::XORrr16, X86::XORrr32, 0 },  // XOR
  };
  
  unsigned Opcode = OpcodeTab[OperatorClass][Class];
  unsigned Op0r = getReg(B.getOperand(0));
  unsigned Op1r = getReg(B.getOperand(1));
  BuildMI(BB, Opcode, 2, getReg(B)).addReg(Op0r).addReg(Op1r);
}

/// visitMul - Multiplies are not simple binary operators because they must deal
/// with the EAX register explicitly.
///
void ISel::visitMul(BinaryOperator &I) {
  unsigned Class = getClass(I.getType());
  if (Class > 2)  // FIXME: Handle longs
    visitInstruction(I);

  static const unsigned Regs[]     ={ X86::AL    , X86::AX     , X86::EAX     };
  static const unsigned Clobbers[] ={ X86::AH    , X86::DX     , X86::EDX     };
  static const unsigned MulOpcode[]={ X86::MULrr8, X86::MULrr16, X86::MULrr32 };
  static const unsigned MovOpcode[]={ X86::MOVrr8, X86::MOVrr16, X86::MOVrr32 };

  unsigned Reg     = Regs[Class];
  unsigned Clobber = Clobbers[Class];
  unsigned Op0Reg  = getReg(I.getOperand(0));
  unsigned Op1Reg  = getReg(I.getOperand(1));

  // Put the first operand into one of the A registers...
  BuildMI(BB, MovOpcode[Class], 1, Reg).addReg(Op0Reg);
  
  // Emit the appropriate multiply instruction...
  BuildMI(BB, MulOpcode[Class], 3)
    .addReg(Reg, UseAndDef).addReg(Op1Reg).addClobber(Clobber);

  // Put the result into the destination register...
  BuildMI(BB, MovOpcode[Class], 1, getReg(I)).addReg(Reg);
}


/// visitDivRem - Handle division and remainder instructions... these
/// instruction both require the same instructions to be generated, they just
/// select the result from a different register.  Note that both of these
/// instructions work differently for signed and unsigned operands.
///
void ISel::visitDivRem(BinaryOperator &I) {
  unsigned Class = getClass(I.getType());
  if (Class > 2)  // FIXME: Handle longs
    visitInstruction(I);

  static const unsigned Regs[]     ={ X86::AL    , X86::AX     , X86::EAX     };
  static const unsigned MovOpcode[]={ X86::MOVrr8, X86::MOVrr16, X86::MOVrr32 };
  static const unsigned ExtOpcode[]={ X86::CBW   , X86::CWD    , X86::CDQ     };
  static const unsigned ClrOpcode[]={ X86::XORrr8, X86::XORrr16, X86::XORrr32 };
  static const unsigned ExtRegs[]  ={ X86::AH    , X86::DX     , X86::EDX     };

  static const unsigned DivOpcode[][4] = {
    { X86::DIVrr8 , X86::DIVrr16 , X86::DIVrr32 , 0 },  // Unsigned division
    { X86::IDIVrr8, X86::IDIVrr16, X86::IDIVrr32, 0 },  // Signed division
  };

  bool isSigned   = I.getType()->isSigned();
  unsigned Reg    = Regs[Class];
  unsigned ExtReg = ExtRegs[Class];
  unsigned Op0Reg = getReg(I.getOperand(0));
  unsigned Op1Reg = getReg(I.getOperand(1));

  // Put the first operand into one of the A registers...
  BuildMI(BB, MovOpcode[Class], 1, Reg).addReg(Op0Reg);

  if (isSigned) {
    // Emit a sign extension instruction...
    BuildMI(BB, ExtOpcode[Class], 1, ExtReg).addReg(Reg);
  } else {
    // If unsigned, emit a zeroing instruction... (reg = xor reg, reg)
    BuildMI(BB, ClrOpcode[Class], 2, ExtReg).addReg(ExtReg).addReg(ExtReg);
  }

  // Emit the appropriate divide or remainder instruction...
  BuildMI(BB, DivOpcode[isSigned][Class], 2)
    .addReg(Reg, UseAndDef).addReg(ExtReg, UseAndDef).addReg(Op1Reg);

  // Figure out which register we want to pick the result out of...
  unsigned DestReg = (I.getOpcode() == Instruction::Div) ? Reg : ExtReg;
  
  // Put the result into the destination register...
  BuildMI(BB, MovOpcode[Class], 1, getReg(I)).addReg(DestReg);
}


/// Shift instructions: 'shl', 'sar', 'shr' - Some special cases here
/// for constant immediate shift values, and for constant immediate
/// shift values equal to 1. Even the general case is sort of special,
/// because the shift amount has to be in CL, not just any old register.
///
void ISel::visitShiftInst (ShiftInst &I) {
  unsigned Op0r = getReg (I.getOperand(0));
  unsigned DestReg = getReg(I);
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
      BuildMI(BB, X86::MOVrr8, 1, X86::CL).addReg(getReg(I.getOperand(1)));

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


/// visitLoadInst - Implement LLVM load instructions in terms of the x86 'mov'
/// instruction.
///
void ISel::visitLoadInst(LoadInst &I) {
  unsigned Class = getClass(I.getType());
  if (Class > 2)  // FIXME: Handle longs and others...
    visitInstruction(I);

  static const unsigned Opcode[] = { X86::MOVmr8, X86::MOVmr16, X86::MOVmr32 };

  unsigned AddressReg = getReg(I.getOperand(0));
  addDirectMem(BuildMI(BB, Opcode[Class], 4, getReg(I)), AddressReg);
}


/// visitStoreInst - Implement LLVM store instructions in terms of the x86 'mov'
/// instruction.
///
void ISel::visitStoreInst(StoreInst &I) {
  unsigned Class = getClass(I.getOperand(0)->getType());
  if (Class > 2)  // FIXME: Handle longs and others...
    visitInstruction(I);

  static const unsigned Opcode[] = { X86::MOVrm8, X86::MOVrm16, X86::MOVrm32 };

  unsigned ValReg = getReg(I.getOperand(0));
  unsigned AddressReg = getReg(I.getOperand(1));
  addDirectMem(BuildMI(BB, Opcode[Class], 1+4), AddressReg).addReg(ValReg);
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
