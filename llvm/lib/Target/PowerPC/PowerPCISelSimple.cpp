//===-- InstSelectSimple.cpp - A simple instruction selector for PowerPC --===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

#include "PowerPC.h"
#include "PowerPCInstrBuilder.h"
#include "PowerPCInstrInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicLowering.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
using namespace llvm;

namespace {
  /// TypeClass - Used by the PowerPC backend to group LLVM types by their basic PPC
  /// Representation.
  ///
  enum TypeClass {
    cByte, cShort, cInt, cFP, cLong
  };
}

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

  case Type::FloatTyID:
  case Type::DoubleTyID:  return cFP;        // Floating Point is #3

  case Type::LongTyID:
  case Type::ULongTyID:   return cLong;      // Longs are class #4
  default:
    assert(0 && "Invalid type to getClass!");
    return cByte;  // not reached
  }
}

// getClassB - Just like getClass, but treat boolean values as ints.
static inline TypeClass getClassB(const Type *Ty) {
  if (Ty == Type::BoolTy) return cInt;
  return getClass(Ty);
}

namespace {
  struct ISel : public FunctionPass, InstVisitor<ISel> {
    TargetMachine &TM;
    MachineFunction *F;                 // The function we are compiling into
    MachineBasicBlock *BB;              // The current MBB we are compiling
    int VarArgsFrameIndex;              // FrameIndex for start of varargs area
    int ReturnAddressIndex;             // FrameIndex for the return address

    std::map<Value*, unsigned> RegMap;  // Mapping between Val's and SSA Regs

    // MBBMap - Mapping between LLVM BB -> Machine BB
    std::map<const BasicBlock*, MachineBasicBlock*> MBBMap;

    // AllocaMap - Mapping from fixed sized alloca instructions to the
    // FrameIndex for the alloca.
    std::map<AllocaInst*, unsigned> AllocaMap;

    ISel(TargetMachine &tm) : TM(tm), F(0), BB(0) {}

    /// runOnFunction - Top level implementation of instruction selection for
    /// the entire function.
    ///
    bool runOnFunction(Function &Fn) {
      // First pass over the function, lower any unknown intrinsic functions
      // with the IntrinsicLowering class.
      LowerUnknownIntrinsicFunctionCalls(Fn);

      F = &MachineFunction::construct(&Fn, TM);

      // Create all of the machine basic blocks for the function...
      for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
        F->getBasicBlockList().push_back(MBBMap[I] = new MachineBasicBlock(I));

      BB = &F->front();

      // Set up a frame object for the return address.  This is used by the
      // llvm.returnaddress & llvm.frameaddress intrinisics.
      ReturnAddressIndex = F->getFrameInfo()->CreateFixedObject(4, -4);

      // Copy incoming arguments off of the stack...
      LoadArgumentsToVirtualRegs(Fn);

      // Instruction select everything except PHI nodes
      visit(Fn);

      // Select the PHI nodes
      SelectPHINodes();

      RegMap.clear();
      MBBMap.clear();
      AllocaMap.clear();
      F = 0;
      // We always build a machine code representation for the function
      return true;
    }

    virtual const char *getPassName() const {
      return "PowerPC Simple Instruction Selection";
    }

    /// visitBasicBlock - This method is called when we are visiting a new basic
    /// block.  This simply creates a new MachineBasicBlock to emit code into
    /// and adds it to the current MachineFunction.  Subsequent visit* for
    /// instructions will be invoked for all instructions in the basic block.
    ///
    void visitBasicBlock(BasicBlock &LLVM_BB) {
      BB = MBBMap[&LLVM_BB];
    }

    /// LowerUnknownIntrinsicFunctionCalls - This performs a prepass over the
    /// function, lowering any calls to unknown intrinsic functions into the
    /// equivalent LLVM code.
    ///
    void LowerUnknownIntrinsicFunctionCalls(Function &F);

    /// LoadArgumentsToVirtualRegs - Load all of the arguments to this function
    /// from the stack into virtual registers.
    ///
    void LoadArgumentsToVirtualRegs(Function &F);

    /// SelectPHINodes - Insert machine code to generate phis.  This is tricky
    /// because we have to generate our sources into the source basic blocks,
    /// not the current one.
    ///
    void SelectPHINodes();

    // Visitation methods for various instructions.  These methods simply emit
    // fixed PowerPC code for each instruction.

    // Control flow operators
    void visitReturnInst(ReturnInst &RI);
    void visitBranchInst(BranchInst &BI);

    struct ValueRecord {
      Value *Val;
      unsigned Reg;
      const Type *Ty;
      ValueRecord(unsigned R, const Type *T) : Val(0), Reg(R), Ty(T) {}
      ValueRecord(Value *V) : Val(V), Reg(0), Ty(V->getType()) {}
    };
    void doCall(const ValueRecord &Ret, MachineInstr *CallMI,
                const std::vector<ValueRecord> &Args);
    void visitCallInst(CallInst &I);
    void visitIntrinsicCall(Intrinsic::ID ID, CallInst &I);

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

    // Comparison operators...
    void visitSetCondInst(SetCondInst &I);
    unsigned EmitComparison(unsigned OpNum, Value *Op0, Value *Op1,
                            MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator MBBI);
    void visitSelectInst(SelectInst &SI);
    
    
    // Memory Instructions
    void visitLoadInst(LoadInst &I);
    void visitStoreInst(StoreInst &I);
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitAllocaInst(AllocaInst &I);
    void visitMallocInst(MallocInst &I);
    void visitFreeInst(FreeInst &I);
    
    // Other operators
    void visitShiftInst(ShiftInst &I);
    void visitPHINode(PHINode &I) {}      // PHI nodes handled by second pass
    void visitCastInst(CastInst &I);
    void visitVANextInst(VANextInst &I);
    void visitVAArgInst(VAArgInst &I);

    void visitInstruction(Instruction &I) {
      std::cerr << "Cannot instruction select: " << I;
      abort();
    }

    /// promote32 - Make a value 32-bits wide, and put it somewhere.
    ///
    void promote32(unsigned targetReg, const ValueRecord &VR);

    /// emitGEPOperation - Common code shared between visitGetElementPtrInst and
    /// constant expression GEP support.
    ///
    void emitGEPOperation(MachineBasicBlock *BB, MachineBasicBlock::iterator IP,
                          Value *Src, User::op_iterator IdxBegin,
                          User::op_iterator IdxEnd, unsigned TargetReg);

    /// emitCastOperation - Common code shared between visitCastInst and
    /// constant expression cast support.
    ///
    void emitCastOperation(MachineBasicBlock *BB,MachineBasicBlock::iterator IP,
                           Value *Src, const Type *DestTy, unsigned TargetReg);

    /// emitSimpleBinaryOperation - Common code shared between visitSimpleBinary
    /// and constant expression support.
    ///
    void emitSimpleBinaryOperation(MachineBasicBlock *BB,
                                   MachineBasicBlock::iterator IP,
                                   Value *Op0, Value *Op1,
                                   unsigned OperatorClass, unsigned TargetReg);

    /// emitBinaryFPOperation - This method handles emission of floating point
    /// Add (0), Sub (1), Mul (2), and Div (3) operations.
    void emitBinaryFPOperation(MachineBasicBlock *BB,
                               MachineBasicBlock::iterator IP,
                               Value *Op0, Value *Op1,
                               unsigned OperatorClass, unsigned TargetReg);

    void emitMultiply(MachineBasicBlock *BB, MachineBasicBlock::iterator IP,
                      Value *Op0, Value *Op1, unsigned TargetReg);

    void doMultiply(MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBI,
                    unsigned DestReg, const Type *DestTy,
                    unsigned Op0Reg, unsigned Op1Reg);
    void doMultiplyConst(MachineBasicBlock *MBB, 
                         MachineBasicBlock::iterator MBBI,
                         unsigned DestReg, const Type *DestTy,
                         unsigned Op0Reg, unsigned Op1Val);

    void emitDivRemOperation(MachineBasicBlock *BB,
                             MachineBasicBlock::iterator IP,
                             Value *Op0, Value *Op1, bool isDiv,
                             unsigned TargetReg);

    /// emitSetCCOperation - Common code shared between visitSetCondInst and
    /// constant expression support.
    ///
    void emitSetCCOperation(MachineBasicBlock *BB,
                            MachineBasicBlock::iterator IP,
                            Value *Op0, Value *Op1, unsigned Opcode,
                            unsigned TargetReg);

    /// emitShiftOperation - Common code shared between visitShiftInst and
    /// constant expression support.
    ///
    void emitShiftOperation(MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator IP,
                            Value *Op, Value *ShiftAmount, bool isLeftShift,
                            const Type *ResultTy, unsigned DestReg);
      
    /// emitSelectOperation - Common code shared between visitSelectInst and the
    /// constant expression support.
    void emitSelectOperation(MachineBasicBlock *MBB,
                             MachineBasicBlock::iterator IP,
                             Value *Cond, Value *TrueVal, Value *FalseVal,
                             unsigned DestReg);

    /// copyConstantToRegister - Output the instructions required to put the
    /// specified constant into the specified register.
    ///
    void copyConstantToRegister(MachineBasicBlock *MBB,
                                MachineBasicBlock::iterator MBBI,
                                Constant *C, unsigned Reg);

    void emitUCOM(MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBI,
                   unsigned LHS, unsigned RHS);

    /// makeAnotherReg - This method returns the next register number we haven't
    /// yet used.
    ///
    /// Long values are handled somewhat specially.  They are always allocated
    /// as pairs of 32 bit integer values.  The register number returned is the
    /// lower 32 bits of the long value, and the regNum+1 is the upper 32 bits
    /// of the long value.
    ///
    unsigned makeAnotherReg(const Type *Ty) {
      assert(dynamic_cast<const PowerPCRegisterInfo*>(TM.getRegisterInfo()) &&
             "Current target doesn't have PPC reg info??");
      const PowerPCRegisterInfo *MRI =
        static_cast<const PowerPCRegisterInfo*>(TM.getRegisterInfo());
      if (Ty == Type::LongTy || Ty == Type::ULongTy) {
        const TargetRegisterClass *RC = MRI->getRegClassForType(Type::IntTy);
        // Create the lower part
        F->getSSARegMap()->createVirtualRegister(RC);
        // Create the upper part.
        return F->getSSARegMap()->createVirtualRegister(RC)-1;
      }

      // Add the mapping of regnumber => reg class to MachineFunction
      const TargetRegisterClass *RC = MRI->getRegClassForType(Ty);
      return F->getSSARegMap()->createVirtualRegister(RC);
    }

    /// getReg - This method turns an LLVM value into a register number.
    ///
    unsigned getReg(Value &V) { return getReg(&V); }  // Allow references
    unsigned getReg(Value *V) {
      // Just append to the end of the current bb.
      MachineBasicBlock::iterator It = BB->end();
      return getReg(V, BB, It);
    }
    unsigned getReg(Value *V, MachineBasicBlock *MBB,
                    MachineBasicBlock::iterator IPt);

    /// getFixedSizedAllocaFI - Return the frame index for a fixed sized alloca
    /// that is to be statically allocated with the initial stack frame
    /// adjustment.
    unsigned getFixedSizedAllocaFI(AllocaInst *AI);
  };
}

/// dyn_castFixedAlloca - If the specified value is a fixed size alloca
/// instruction in the entry block, return it.  Otherwise, return a null
/// pointer.
static AllocaInst *dyn_castFixedAlloca(Value *V) {
  if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
    BasicBlock *BB = AI->getParent();
    if (isa<ConstantUInt>(AI->getArraySize()) && BB ==&BB->getParent()->front())
      return AI;
  }
  return 0;
}

/// getReg - This method turns an LLVM value into a register number.
///
unsigned ISel::getReg(Value *V, MachineBasicBlock *MBB,
                      MachineBasicBlock::iterator IPt) {
  // If this operand is a constant, emit the code to copy the constant into
  // the register here...
  //
  if (Constant *C = dyn_cast<Constant>(V)) {
    unsigned Reg = makeAnotherReg(V->getType());
    copyConstantToRegister(MBB, IPt, C, Reg);
    return Reg;
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    unsigned Reg1 = makeAnotherReg(V->getType());
	unsigned Reg2 = makeAnotherReg(V->getType());
    // Move the address of the global into the register
    BuildMI(*MBB, IPt, PPC32::LOADHiAddr, 2, Reg1).addReg(PPC32::R0).addGlobalAddress(GV);
    BuildMI(*MBB, IPt, PPC32::LOADLoAddr, 2, Reg2).addReg(Reg1).addGlobalAddress(GV);
    return Reg2;
  } else if (CastInst *CI = dyn_cast<CastInst>(V)) {
    // Do not emit noop casts at all.
    if (getClassB(CI->getType()) == getClassB(CI->getOperand(0)->getType()))
      return getReg(CI->getOperand(0), MBB, IPt);
  } else if (AllocaInst *AI = dyn_castFixedAlloca(V)) {
    unsigned Reg = makeAnotherReg(V->getType());
    unsigned FI = getFixedSizedAllocaFI(AI);
    addFrameReference(BuildMI(*MBB, IPt, PPC32::ADDI, 2, Reg), FI, 0, false);
    return Reg;
  }

  unsigned &Reg = RegMap[V];
  if (Reg == 0) {
    Reg = makeAnotherReg(V->getType());
    RegMap[V] = Reg;
  }

  return Reg;
}

/// getFixedSizedAllocaFI - Return the frame index for a fixed sized alloca
/// that is to be statically allocated with the initial stack frame
/// adjustment.
unsigned ISel::getFixedSizedAllocaFI(AllocaInst *AI) {
  // Already computed this?
  std::map<AllocaInst*, unsigned>::iterator I = AllocaMap.lower_bound(AI);
  if (I != AllocaMap.end() && I->first == AI) return I->second;

  const Type *Ty = AI->getAllocatedType();
  ConstantUInt *CUI = cast<ConstantUInt>(AI->getArraySize());
  unsigned TySize = TM.getTargetData().getTypeSize(Ty);
  TySize *= CUI->getValue();   // Get total allocated size...
  unsigned Alignment = TM.getTargetData().getTypeAlignment(Ty);
      
  // Create a new stack object using the frame manager...
  int FrameIdx = F->getFrameInfo()->CreateStackObject(TySize, Alignment);
  AllocaMap.insert(I, std::make_pair(AI, FrameIdx));
  return FrameIdx;
}


/// copyConstantToRegister - Output the instructions required to put the
/// specified constant into the specified register.
///
void ISel::copyConstantToRegister(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator IP,
                                  Constant *C, unsigned R) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    unsigned Class = 0;
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      emitGEPOperation(MBB, IP, CE->getOperand(0),
                       CE->op_begin()+1, CE->op_end(), R);
      return;
    case Instruction::Cast:
      emitCastOperation(MBB, IP, CE->getOperand(0), CE->getType(), R);
      return;

    case Instruction::Xor: ++Class; // FALL THROUGH
    case Instruction::Or:  ++Class; // FALL THROUGH
    case Instruction::And: ++Class; // FALL THROUGH
    case Instruction::Sub: ++Class; // FALL THROUGH
    case Instruction::Add:
      emitSimpleBinaryOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                                Class, R);
      return;

    case Instruction::Mul:
      emitMultiply(MBB, IP, CE->getOperand(0), CE->getOperand(1), R);
      return;

    case Instruction::Div:
    case Instruction::Rem:
      emitDivRemOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                          CE->getOpcode() == Instruction::Div, R);
      return;

    case Instruction::SetNE:
    case Instruction::SetEQ:
    case Instruction::SetLT:
    case Instruction::SetGT:
    case Instruction::SetLE:
    case Instruction::SetGE:
      emitSetCCOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                         CE->getOpcode(), R);
      return;

    case Instruction::Shl:
    case Instruction::Shr:
      emitShiftOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                         CE->getOpcode() == Instruction::Shl, CE->getType(), R);
      return;

    case Instruction::Select:
      emitSelectOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                          CE->getOperand(2), R);
      return;

    default:
      std::cerr << "Offending expr: " << C << "\n";
      assert(0 && "Constant expression not yet handled!\n");
    }
  }

  if (C->getType()->isIntegral()) {
    unsigned Class = getClassB(C->getType());

    if (Class == cLong) {
      // Copy the value into the register pair.
      uint64_t Val = cast<ConstantInt>(C)->getRawValue();
	  unsigned hiTmp = makeAnotherReg(Type::IntTy);
	  unsigned loTmp = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, PPC32::ADDIS, 2, loTmp).addReg(PPC32::R0).addImm(Val >> 48);
      BuildMI(*MBB, IP, PPC32::ORI, 2, R).addReg(loTmp).addImm((Val >> 32) & 0xFFFF);
      BuildMI(*MBB, IP, PPC32::ADDIS, 2, hiTmp).addReg(PPC32::R0).addImm((Val >> 16) & 0xFFFF);
      BuildMI(*MBB, IP, PPC32::ORI, 2, R+1).addReg(hiTmp).addImm(Val & 0xFFFF);
      return;
    }

    assert(Class <= cInt && "Type not handled yet!");

    if (C->getType() == Type::BoolTy) {
      BuildMI(*MBB, IP, PPC32::ADDI, 2, R).addReg(PPC32::R0).addImm(C == ConstantBool::True);
    } else if (Class == cByte || Class == cShort) {
      ConstantInt *CI = cast<ConstantInt>(C);
      BuildMI(*MBB, IP, PPC32::ADDI, 2, R).addReg(PPC32::R0).addImm(CI->getRawValue());
    } else {
      ConstantInt *CI = cast<ConstantInt>(C);
      int TheVal = CI->getRawValue() & 0xFFFFFFFF;
      if (TheVal < 32768 && TheVal >= -32768) {
		BuildMI(*MBB, IP, PPC32::ADDI, 2, R).addReg(PPC32::R0).addImm(CI->getRawValue());
	  } else {
		unsigned TmpReg = makeAnotherReg(Type::IntTy);
		BuildMI(*MBB, IP, PPC32::ADDIS, 2, TmpReg).addReg(PPC32::R0).addImm(CI->getRawValue() >> 16);
		BuildMI(*MBB, IP, PPC32::ORI, 2, R).addReg(TmpReg).addImm(CI->getRawValue() & 0xFFFF);
	  }
    }
  } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
      // We need to spill the constant to memory...
      MachineConstantPool *CP = F->getConstantPool();
      unsigned CPI = CP->getConstantPoolIndex(CFP);
      const Type *Ty = CFP->getType();

      assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!"); 
      unsigned LoadOpcode = Ty == Type::FloatTy ? PPC32::LFS : PPC32::LFD;
      addConstantPoolReference(BuildMI(*MBB, IP, LoadOpcode, 2, R), CPI);
  } else if (isa<ConstantPointerNull>(C)) {
    // Copy zero (null pointer) to the register.
    BuildMI(*MBB, IP, PPC32::ADDI, 2, R).addReg(PPC32::R0).addImm(0);
  } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C)) {
    BuildMI(*MBB, IP, PPC32::ADDIS, 2, R).addReg(PPC32::R0).addGlobalAddress(CPR->getValue());
    BuildMI(*MBB, IP, PPC32::ORI, 2, R).addReg(PPC32::R0).addGlobalAddress(CPR->getValue());
  } else {
    std::cerr << "Offending constant: " << C << "\n";
    assert(0 && "Type not handled yet!");
  }
}

/// LoadArgumentsToVirtualRegs - Load all of the arguments to this function from
/// the stack into virtual registers.
///
/// FIXME: When we can calculate which args are coming in via registers
/// source them from there instead.
void ISel::LoadArgumentsToVirtualRegs(Function &Fn) {
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  unsigned GPR_remaining = 8;
  unsigned FPR_remaining = 13;
  unsigned GPR_idx = 3;
  unsigned FPR_idx = 1;
	
  MachineFrameInfo *MFI = F->getFrameInfo();

  for (Function::aiterator I = Fn.abegin(), E = Fn.aend(); I != E; ++I) {
    bool ArgLive = !I->use_empty();
    unsigned Reg = ArgLive ? getReg(*I) : 0;
    int FI;          // Frame object index

    switch (getClassB(I->getType())) {
    case cByte:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(1, ArgOffset);
		if (GPR_remaining > 0) {
			BuildMI(BB, PPC32::OR, 2, Reg).addReg(PPC32::R0+GPR_idx).addReg(PPC32::R0+GPR_idx);
		} else {
			addFrameReference(BuildMI(BB, PPC32::LBZ, 2, Reg), FI);
		}
	  }
      break;
    case cShort:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(2, ArgOffset);
		if (GPR_remaining > 0) {
			BuildMI(BB, PPC32::OR, 2, Reg).addReg(PPC32::R0+GPR_idx).addReg(PPC32::R0+GPR_idx);
		} else {
			addFrameReference(BuildMI(BB, PPC32::LHZ, 2, Reg), FI);
		}
	  }
      break;
    case cInt:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(4, ArgOffset);
		if (GPR_remaining > 0) {
			BuildMI(BB, PPC32::OR, 2, Reg).addReg(PPC32::R0+GPR_idx).addReg(PPC32::R0+GPR_idx);
		} else {
			addFrameReference(BuildMI(BB, PPC32::LWZ, 2, Reg), FI);
		}
	  }
      break;
    case cLong:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(8, ArgOffset);
		if (GPR_remaining > 1) {
			BuildMI(BB, PPC32::OR, 2, Reg).addReg(PPC32::R0+GPR_idx).addReg(PPC32::R0+GPR_idx);
			BuildMI(BB, PPC32::OR, 2, Reg+1).addReg(PPC32::R0+GPR_idx+1).addReg(PPC32::R0+GPR_idx+1);
		} else {
			addFrameReference(BuildMI(BB, PPC32::LWZ, 2, Reg), FI);
			addFrameReference(BuildMI(BB, PPC32::LWZ, 2, Reg+1), FI, 4);
		}
	  }
      ArgOffset += 4;   // longs require 4 additional bytes
	  if (GPR_remaining > 1) {
		GPR_remaining--;    // uses up 2 GPRs
		GPR_idx++;
	  }
      break;
    case cFP:
      if (ArgLive) {
        unsigned Opcode;
        if (I->getType() == Type::FloatTy) {
          Opcode = PPC32::LFS;
          FI = MFI->CreateFixedObject(4, ArgOffset);
        } else {
          Opcode = PPC32::LFD;
          FI = MFI->CreateFixedObject(8, ArgOffset);
        }
		if (FPR_remaining > 0) {
			BuildMI(BB, PPC32::FMR, 1, Reg).addReg(PPC32::F0+FPR_idx);
			FPR_remaining--;
			FPR_idx++;
		} else {
			addFrameReference(BuildMI(BB, Opcode, 2, Reg), FI);
		}
	  }
      if (I->getType() == Type::DoubleTy) {
        ArgOffset += 4;   // doubles require 4 additional bytes
		if (GPR_remaining > 0) {
			GPR_remaining--;    // uses up 2 GPRs
			GPR_idx++;
		}
	  }
      break;
    default:
      assert(0 && "Unhandled argument type!");
    }
    ArgOffset += 4;  // Each argument takes at least 4 bytes on the stack...
	if (GPR_remaining > 0) {
		GPR_remaining--;    // uses up 2 GPRs
		GPR_idx++;
	}
  }

  // If the function takes variable number of arguments, add a frame offset for
  // the start of the first vararg value... this is used to expand
  // llvm.va_start.
  if (Fn.getFunctionType()->isVarArg())
    VarArgsFrameIndex = MFI->CreateFixedObject(1, ArgOffset);
}


/// SelectPHINodes - Insert machine code to generate phis.  This is tricky
/// because we have to generate our sources into the source basic blocks, not
/// the current one.
///
void ISel::SelectPHINodes() {
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  const Function &LF = *F->getFunction();  // The LLVM function...
  for (Function::const_iterator I = LF.begin(), E = LF.end(); I != E; ++I) {
    const BasicBlock *BB = I;
    MachineBasicBlock &MBB = *MBBMap[I];

    // Loop over all of the PHI nodes in the LLVM basic block...
    MachineBasicBlock::iterator PHIInsertPoint = MBB.begin();
    for (BasicBlock::const_iterator I = BB->begin();
         PHINode *PN = const_cast<PHINode*>(dyn_cast<PHINode>(I)); ++I) {

      // Create a new machine instr PHI node, and insert it.
      unsigned PHIReg = getReg(*PN);
      MachineInstr *PhiMI = BuildMI(MBB, PHIInsertPoint,
                                    PPC32::PHI, PN->getNumOperands(), PHIReg);

      MachineInstr *LongPhiMI = 0;
      if (PN->getType() == Type::LongTy || PN->getType() == Type::ULongTy)
        LongPhiMI = BuildMI(MBB, PHIInsertPoint,
                            PPC32::PHI, PN->getNumOperands(), PHIReg+1);

      // PHIValues - Map of blocks to incoming virtual registers.  We use this
      // so that we only initialize one incoming value for a particular block,
      // even if the block has multiple entries in the PHI node.
      //
      std::map<MachineBasicBlock*, unsigned> PHIValues;

      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        MachineBasicBlock *PredMBB = MBBMap[PN->getIncomingBlock(i)];
        unsigned ValReg;
        std::map<MachineBasicBlock*, unsigned>::iterator EntryIt =
          PHIValues.lower_bound(PredMBB);

        if (EntryIt != PHIValues.end() && EntryIt->first == PredMBB) {
          // We already inserted an initialization of the register for this
          // predecessor.  Recycle it.
          ValReg = EntryIt->second;

        } else {        
          // Get the incoming value into a virtual register.
          //
          Value *Val = PN->getIncomingValue(i);

          // If this is a constant or GlobalValue, we may have to insert code
          // into the basic block to compute it into a virtual register.
          if ((isa<Constant>(Val) && !isa<ConstantExpr>(Val)) ||
              isa<GlobalValue>(Val)) {
            // Simple constants get emitted at the end of the basic block,
            // before any terminator instructions.  We "know" that the code to
            // move a constant into a register will never clobber any flags.
            ValReg = getReg(Val, PredMBB, PredMBB->getFirstTerminator());
          } else {
            // Because we don't want to clobber any values which might be in
            // physical registers with the computation of this constant (which
            // might be arbitrarily complex if it is a constant expression),
            // just insert the computation at the top of the basic block.
            MachineBasicBlock::iterator PI = PredMBB->begin();
            
            // Skip over any PHI nodes though!
            while (PI != PredMBB->end() && PI->getOpcode() == PPC32::PHI)
              ++PI;
            
            ValReg = getReg(Val, PredMBB, PI);
          }

          // Remember that we inserted a value for this PHI for this predecessor
          PHIValues.insert(EntryIt, std::make_pair(PredMBB, ValReg));
        }

        PhiMI->addRegOperand(ValReg);
        PhiMI->addMachineBasicBlockOperand(PredMBB);
        if (LongPhiMI) {
          LongPhiMI->addRegOperand(ValReg+1);
          LongPhiMI->addMachineBasicBlockOperand(PredMBB);
        }
      }

      // Now that we emitted all of the incoming values for the PHI node, make
      // sure to reposition the InsertPoint after the PHI that we just added.
      // This is needed because we might have inserted a constant into this
      // block, right after the PHI's which is before the old insert point!
      PHIInsertPoint = LongPhiMI ? LongPhiMI : PhiMI;
      ++PHIInsertPoint;
    }
  }
}


// canFoldSetCCIntoBranchOrSelect - Return the setcc instruction if we can fold
// it into the conditional branch or select instruction which is the only user
// of the cc instruction.  This is the case if the conditional branch is the
// only user of the setcc, and if the setcc is in the same basic block as the
// conditional branch.  We also don't handle long arguments below, so we reject
// them here as well.
//
static SetCondInst *canFoldSetCCIntoBranchOrSelect(Value *V) {
  if (SetCondInst *SCI = dyn_cast<SetCondInst>(V))
    if (SCI->hasOneUse()) {
      Instruction *User = cast<Instruction>(SCI->use_back());
      if ((isa<BranchInst>(User) || isa<SelectInst>(User)) &&
          SCI->getParent() == User->getParent() &&
          (getClassB(SCI->getOperand(0)->getType()) != cLong ||
           SCI->getOpcode() == Instruction::SetEQ ||
           SCI->getOpcode() == Instruction::SetNE))
        return SCI;
    }
  return 0;
}

// Return a fixed numbering for setcc instructions which does not depend on the
// order of the opcodes.
//
static unsigned getSetCCNumber(unsigned Opcode) {
  switch(Opcode) {
  default: assert(0 && "Unknown setcc instruction!");
  case Instruction::SetEQ: return 0;
  case Instruction::SetNE: return 1;
  case Instruction::SetLT: return 2;
  case Instruction::SetGE: return 3;
  case Instruction::SetGT: return 4;
  case Instruction::SetLE: return 5;
  }
}

/// emitUCOM - emits an unordered FP compare.
void ISel::emitUCOM(MachineBasicBlock *MBB, MachineBasicBlock::iterator IP,
                     unsigned LHS, unsigned RHS) {
	BuildMI(*MBB, IP, PPC32::FCMPU, 2, PPC32::CR0).addReg(LHS).addReg(RHS);
}

// EmitComparison - This function emits a comparison of the two operands,
// returning the extended setcc code to use.
unsigned ISel::EmitComparison(unsigned OpNum, Value *Op0, Value *Op1,
                              MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator IP) {
  // The arguments are already supposed to be of the same type.
  const Type *CompTy = Op0->getType();
  unsigned Class = getClassB(CompTy);
  unsigned Op0r = getReg(Op0, MBB, IP);

  // Special case handling of: cmp R, i
  if (isa<ConstantPointerNull>(Op1)) {
      BuildMI(*MBB, IP, PPC32::CMPI, 2, PPC32::CR0).addReg(Op0r).addImm(0);
  } else if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    if (Class == cByte || Class == cShort || Class == cInt) {
      unsigned Op1v = CI->getRawValue();

      // Mask off any upper bits of the constant, if there are any...
      Op1v &= (1ULL << (8 << Class)) - 1;

	  // Compare immediate or promote to reg?
	  if (Op1v <= 32767) {
		BuildMI(*MBB, IP, CompTy->isSigned() ? PPC32::CMPI : PPC32::CMPLI, 3, PPC32::CR0).addImm(0).addReg(Op0r).addImm(Op1v);
	  } else {
		unsigned Op1r = getReg(Op1, MBB, IP);
		BuildMI(*MBB, IP, CompTy->isSigned() ? PPC32::CMP : PPC32::CMPL, 3, PPC32::CR0).addImm(0).addReg(Op0r).addReg(Op1r);
	  }
      return OpNum;
    } else {
      assert(Class == cLong && "Unknown integer class!");
      unsigned LowCst = CI->getRawValue();
      unsigned HiCst = CI->getRawValue() >> 32;
      if (OpNum < 2) {    // seteq, setne
        unsigned LoTmp = Op0r;
        if (LowCst != 0) {
		  unsigned LoLow = makeAnotherReg(Type::IntTy);
          unsigned LoTmp = makeAnotherReg(Type::IntTy);
          BuildMI(*MBB, IP, PPC32::XORI, 2, LoLow).addReg(Op0r).addImm(LowCst);
          BuildMI(*MBB, IP, PPC32::XORIS, 2, LoTmp).addReg(LoLow).addImm(LowCst >> 16);
        }
        unsigned HiTmp = Op0r+1;
        if (HiCst != 0) {
		  unsigned HiLow = makeAnotherReg(Type::IntTy);
          unsigned HiTmp = makeAnotherReg(Type::IntTy);
          BuildMI(*MBB, IP, PPC32::XORI, 2, HiLow).addReg(Op0r+1).addImm(HiCst);
          BuildMI(*MBB, IP, PPC32::XORIS, 2, HiTmp).addReg(HiLow).addImm(HiCst >> 16);
        }
        unsigned FinalTmp = makeAnotherReg(Type::IntTy);
        BuildMI(*MBB, IP, PPC32::ORo, 2, FinalTmp).addReg(LoTmp).addReg(HiTmp);
        //BuildMI(*MBB, IP, PPC32::CMPLI, 2, PPC32::CR0).addReg(FinalTmp).addImm(0);
        return OpNum;
      } else {
        // Emit a sequence of code which compares the high and low parts once
        // each, then uses a conditional move to handle the overflow case.  For
        // example, a setlt for long would generate code like this:
        //
        // AL = lo(op1) < lo(op2)   // Always unsigned comparison
        // BL = hi(op1) < hi(op2)   // Signedness depends on operands
        // dest = hi(op1) == hi(op2) ? BL : AL;
        //

        // FIXME: Not Yet Implemented
		return OpNum;
      }
    }
  }

  unsigned Op1r = getReg(Op1, MBB, IP);
  switch (Class) {
  default: assert(0 && "Unknown type class!");
  case cByte:
  case cShort:
  case cInt:
	BuildMI(*MBB, IP, CompTy->isSigned() ? PPC32::CMP : PPC32::CMPL, 2, PPC32::CR0).addReg(Op0r).addReg(Op1r);
    break;
  case cFP:
    emitUCOM(MBB, IP, Op0r, Op1r);
    break;

  case cLong:
    if (OpNum < 2) {    // seteq, setne
      unsigned LoTmp = makeAnotherReg(Type::IntTy);
      unsigned HiTmp = makeAnotherReg(Type::IntTy);
      unsigned FinalTmp = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, PPC32::XOR, 2, LoTmp).addReg(Op0r).addReg(Op1r);
      BuildMI(*MBB, IP, PPC32::XOR, 2, HiTmp).addReg(Op0r+1).addReg(Op1r+1);
      BuildMI(*MBB, IP, PPC32::ORo,  2, FinalTmp).addReg(LoTmp).addReg(HiTmp);
      //BuildMI(*MBB, IP, PPC32::CMPLI, 2, PPC32::CR0).addReg(FinalTmp).addImm(0);
      break;  // Allow the sete or setne to be generated from flags set by OR
    } else {
      // Emit a sequence of code which compares the high and low parts once
      // each, then uses a conditional move to handle the overflow case.  For
      // example, a setlt for long would generate code like this:
      //
      // AL = lo(op1) < lo(op2)   // Signedness depends on operands
      // BL = hi(op1) < hi(op2)   // Always unsigned comparison
      // dest = hi(op1) == hi(op2) ? BL : AL;
      //

      // FIXME: Not Yet Implemented
      return OpNum;
    }
  }
  return OpNum;
}

/// SetCC instructions - Here we just emit boilerplate code to set a byte-sized
/// register, then move it to wherever the result should be. 
///
void ISel::visitSetCondInst(SetCondInst &I) {
  if (canFoldSetCCIntoBranchOrSelect(&I))
    return;  // Fold this into a branch or select.

  unsigned DestReg = getReg(I);
  MachineBasicBlock::iterator MII = BB->end();
  emitSetCCOperation(BB, MII, I.getOperand(0), I.getOperand(1), I.getOpcode(),DestReg);
}

/// emitSetCCOperation - Common code shared between visitSetCondInst and
/// constant expression support.
///
/// FIXME: this is wrong.  we should figure out a way to guarantee
/// TargetReg is a CR and then make it a no-op
void ISel::emitSetCCOperation(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator IP,
                              Value *Op0, Value *Op1, unsigned Opcode,
                              unsigned TargetReg) {
  unsigned OpNum = getSetCCNumber(Opcode);
  OpNum = EmitComparison(OpNum, Op0, Op1, MBB, IP);

  // The value is already in CR0 at this point, do nothing.
}


void ISel::visitSelectInst(SelectInst &SI) {
  unsigned DestReg = getReg(SI);
  MachineBasicBlock::iterator MII = BB->end();
  emitSelectOperation(BB, MII, SI.getCondition(), SI.getTrueValue(),SI.getFalseValue(), DestReg);
}
 
/// emitSelect - Common code shared between visitSelectInst and the constant
/// expression support.
/// FIXME: this is most likely broken in one or more ways.  Namely, PowerPC has
/// no select instruction.  FSEL only works for comparisons against zero.
void ISel::emitSelectOperation(MachineBasicBlock *MBB,
                               MachineBasicBlock::iterator IP,
                               Value *Cond, Value *TrueVal, Value *FalseVal,
                               unsigned DestReg) {
  unsigned SelectClass = getClassB(TrueVal->getType());

  unsigned TrueReg  = getReg(TrueVal, MBB, IP);
  unsigned FalseReg = getReg(FalseVal, MBB, IP);

  if (TrueReg == FalseReg) {
	if (SelectClass == cFP) {
		BuildMI(*MBB, IP, PPC32::FMR, 1, DestReg).addReg(TrueReg);
	} else {
		BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(TrueReg).addReg(TrueReg);
	}
	
    if (SelectClass == cLong)
		BuildMI(*MBB, IP, PPC32::OR, 2, DestReg+1).addReg(TrueReg+1).addReg(TrueReg+1);
    return;
  }

  unsigned CondReg = getReg(Cond, MBB, IP);
  unsigned numZeros = makeAnotherReg(Type::IntTy);
  unsigned falseHi = makeAnotherReg(Type::IntTy);
  unsigned falseAll = makeAnotherReg(Type::IntTy);
  unsigned trueAll = makeAnotherReg(Type::IntTy);
  unsigned Temp1 = makeAnotherReg(Type::IntTy);
  unsigned Temp2 = makeAnotherReg(Type::IntTy);

  BuildMI(*MBB, IP, PPC32::CNTLZW, 1, numZeros).addReg(CondReg);
  BuildMI(*MBB, IP, PPC32::RLWINM, 4, falseHi).addReg(numZeros).addImm(26).addImm(0).addImm(0);
  BuildMI(*MBB, IP, PPC32::SRAWI, 2, falseAll).addReg(falseHi).addImm(31);
  BuildMI(*MBB, IP, PPC32::NOR, 2, trueAll).addReg(falseAll).addReg(falseAll);
  BuildMI(*MBB, IP, PPC32::AND, 2, Temp1).addReg(TrueReg).addReg(trueAll);
  BuildMI(*MBB, IP, PPC32::AND, 2, Temp2).addReg(FalseReg).addReg(falseAll);
  BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(Temp1).addReg(Temp2);
  
  if (SelectClass == cLong) {
	unsigned Temp3 = makeAnotherReg(Type::IntTy);
	unsigned Temp4 = makeAnotherReg(Type::IntTy);
	BuildMI(*MBB, IP, PPC32::AND, 2, Temp3).addReg(TrueReg+1).addReg(trueAll);
	BuildMI(*MBB, IP, PPC32::AND, 2, Temp4).addReg(FalseReg+1).addReg(falseAll);
	BuildMI(*MBB, IP, PPC32::OR, 2, DestReg+1).addReg(Temp3).addReg(Temp4);
  }
  
  return;
}



/// promote32 - Emit instructions to turn a narrow operand into a 32-bit-wide
/// operand, in the specified target register.
///
void ISel::promote32(unsigned targetReg, const ValueRecord &VR) {
  bool isUnsigned = VR.Ty->isUnsigned() || VR.Ty == Type::BoolTy;

  Value *Val = VR.Val;
  const Type *Ty = VR.Ty;
  if (Val) {
    if (Constant *C = dyn_cast<Constant>(Val)) {
      Val = ConstantExpr::getCast(C, Type::IntTy);
      Ty = Type::IntTy;
    }

    // If this is a simple constant, just emit a load directly to avoid the copy.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
      int TheVal = CI->getRawValue() & 0xFFFFFFFF;

      if (TheVal < 32768 && TheVal >= -32768) {
		BuildMI(BB, PPC32::ADDI, 2, targetReg).addReg(PPC32::R0).addImm(TheVal);
	  } else {
		unsigned TmpReg = makeAnotherReg(Type::IntTy);
		BuildMI(BB, PPC32::ADDIS, 2, TmpReg).addReg(PPC32::R0).addImm(TheVal >> 16);
		BuildMI(BB, PPC32::ORI, 2, targetReg).addReg(TmpReg).addImm(TheVal & 0xFFFF);
	  }
      return;
    }
  }

  // Make sure we have the register number for this value...
  unsigned Reg = Val ? getReg(Val) : VR.Reg;

  switch (getClassB(Ty)) {
  case cByte:
    // Extend value into target register (8->32)
    if (isUnsigned)
      BuildMI(BB, PPC32::RLWINM, 4, targetReg).addReg(Reg).addZImm(0).addZImm(24).addZImm(31);
    else
      BuildMI(BB, PPC32::EXTSB, 1, targetReg).addReg(Reg);
    break;
  case cShort:
    // Extend value into target register (16->32)
    if (isUnsigned)
      BuildMI(BB, PPC32::RLWINM, 4, targetReg).addReg(Reg).addZImm(0).addZImm(16).addZImm(31);
    else
      BuildMI(BB, PPC32::EXTSH, 1, targetReg).addReg(Reg);
    break;
  case cInt:
    // Move value into target register (32->32)
    BuildMI(BB, PPC32::ORI, 2, targetReg).addReg(Reg).addReg(Reg);
    break;
  default:
    assert(0 && "Unpromotable operand class in promote32");
  }
}

// just emit blr.
void ISel::visitReturnInst(ReturnInst &I) {
  Value *RetVal = I.getOperand(0);

  switch (getClassB(RetVal->getType())) {
  case cByte:   // integral return values: extend or move into r3 and return
  case cShort:
  case cInt:
    promote32(PPC32::R3, ValueRecord(RetVal));
    break;
  case cFP: {   // Floats & Doubles: Return in f1
    unsigned RetReg = getReg(RetVal);
    BuildMI(BB, PPC32::FMR, 1, PPC32::F1).addReg(RetReg);
    break;
  }
  case cLong: {
    unsigned RetReg = getReg(RetVal);
    BuildMI(BB, PPC32::OR, 2, PPC32::R3).addReg(RetReg).addReg(RetReg);
    BuildMI(BB, PPC32::OR, 2, PPC32::R4).addReg(RetReg+1).addReg(RetReg+1);
    break;
  }
  default:
    visitInstruction(I);
  }
  BuildMI(BB, PPC32::BLR, 1).addImm(0);
}

// getBlockAfter - Return the basic block which occurs lexically after the
// specified one.
static inline BasicBlock *getBlockAfter(BasicBlock *BB) {
  Function::iterator I = BB; ++I;  // Get iterator to next block
  return I != BB->getParent()->end() ? &*I : 0;
}

/// visitBranchInst - Handle conditional and unconditional branches here.  Note
/// that since code layout is frozen at this point, that if we are trying to
/// jump to a block that is the immediate successor of the current block, we can
/// just make a fall-through (but we don't currently).
///
void ISel::visitBranchInst(BranchInst &BI) {
	// Update machine-CFG edges
	BB->addSuccessor (MBBMap[BI.getSuccessor(0)]);
	if (BI.isConditional())
		BB->addSuccessor (MBBMap[BI.getSuccessor(1)]);
	
	BasicBlock *NextBB = getBlockAfter(BI.getParent());  // BB after current one
	
	if (!BI.isConditional()) {  // Unconditional branch?
		if (BI.getSuccessor(0) != NextBB)
			BuildMI(BB, PPC32::B, 1).addMBB(MBBMap[BI.getSuccessor(0)]);
		return;
	}
	
  // See if we can fold the setcc into the branch itself...
  SetCondInst *SCI = canFoldSetCCIntoBranchOrSelect(BI.getCondition());
  if (SCI == 0) {
    // Nope, cannot fold setcc into this branch.  Emit a branch on a condition
    // computed some other way...
    unsigned condReg = getReg(BI.getCondition());
	BuildMI(BB, PPC32::CMPLI, 3, PPC32::CR0).addImm(0).addReg(condReg).addImm(0);
    if (BI.getSuccessor(1) == NextBB) {
      if (BI.getSuccessor(0) != NextBB)
        BuildMI(BB, PPC32::BC, 3).addImm(4).addImm(2).addMBB(MBBMap[BI.getSuccessor(0)]);
    } else {
	  BuildMI(BB, PPC32::BC, 3).addImm(12).addImm(2).addMBB(MBBMap[BI.getSuccessor(1)]);
      
      if (BI.getSuccessor(0) != NextBB)
        BuildMI(BB, PPC32::B, 1).addMBB(MBBMap[BI.getSuccessor(0)]);
    }
    return;
  }


  unsigned OpNum = getSetCCNumber(SCI->getOpcode());
  MachineBasicBlock::iterator MII = BB->end();
  OpNum = EmitComparison(OpNum, SCI->getOperand(0), SCI->getOperand(1), BB,MII);

  const Type *CompTy = SCI->getOperand(0)->getType();
  bool isSigned = CompTy->isSigned() && getClassB(CompTy) != cFP;
  
  // LLVM  -> X86 signed  X86 unsigned
  // -----    ----------  ------------
  // seteq -> je          je
  // setne -> jne         jne
  // setlt -> jl          jb
  // setge -> jge         jae
  // setgt -> jg          ja
  // setle -> jle         jbe

  static const unsigned BITab[6] = { 2, 2, 0, 0, 1, 1 };
  unsigned BO_true = (OpNum % 2 == 0) ? 12 : 4;
  unsigned BO_false = (OpNum % 2 == 0) ? 4 : 12;
  unsigned BIval = BITab[0];

  if (BI.getSuccessor(0) != NextBB) {
		BuildMI(BB, PPC32::BC, 3).addImm(BO_true).addImm(BIval).addMBB(MBBMap[BI.getSuccessor(0)]);
    if (BI.getSuccessor(1) != NextBB)
		BuildMI(BB, PPC32::B, 1).addMBB(MBBMap[BI.getSuccessor(1)]);
  } else {
    // Change to the inverse condition...
    if (BI.getSuccessor(1) != NextBB) {
	  BuildMI(BB, PPC32::BC, 3).addImm(BO_false).addImm(BIval).addMBB(MBBMap[BI.getSuccessor(1)]);
    }
  }
}


/// doCall - This emits an abstract call instruction, setting up the arguments
/// and the return value as appropriate.  For the actual function call itself,
/// it inserts the specified CallMI instruction into the stream.
///
/// FIXME: See Documentation at the following URL for "correct" behavior
/// <http://developer.apple.com/documentation/DeveloperTools/Conceptual/MachORuntime/2rt_powerpc_abi/chapter_9_section_5.html>
void ISel::doCall(const ValueRecord &Ret, MachineInstr *CallMI,
                  const std::vector<ValueRecord> &Args) {
  // Count how many bytes are to be pushed on the stack...
  unsigned NumBytes = 0;

  if (!Args.empty()) {
    for (unsigned i = 0, e = Args.size(); i != e; ++i)
      switch (getClassB(Args[i].Ty)) {
      case cByte: case cShort: case cInt:
        NumBytes += 4; break;
      case cLong:
        NumBytes += 8; break;
      case cFP:
        NumBytes += Args[i].Ty == Type::FloatTy ? 4 : 8;
        break;
      default: assert(0 && "Unknown class!");
      }

    // Adjust the stack pointer for the new arguments...
    BuildMI(BB, PPC32::ADJCALLSTACKDOWN, 1).addImm(NumBytes);

    // Arguments go on the stack in reverse order, as specified by the ABI.
    unsigned ArgOffset = 0;
	unsigned GPR_remaining = 8;
	unsigned FPR_remaining = 13;
	unsigned GPR_idx = 3;
	unsigned FPR_idx = 1;
	
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
      unsigned ArgReg;
      switch (getClassB(Args[i].Ty)) {
      case cByte:
      case cShort:
        // Promote arg to 32 bits wide into a temporary register...
        ArgReg = makeAnotherReg(Type::UIntTy);
        promote32(ArgReg, Args[i]);
		  
		// Reg or stack?
		if (GPR_remaining > 0) {
			BuildMI(BB, PPC32::OR, 2, PPC32::R0 + GPR_idx).addReg(ArgReg).addReg(ArgReg);
		} else {
			BuildMI(BB, PPC32::STW, 3).addReg(ArgReg).addImm(ArgOffset).addReg(PPC32::R1);
		}
		break;
      case cInt:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;

		// Reg or stack?
		if (GPR_remaining > 0) {
		    BuildMI(BB, PPC32::OR, 2, PPC32::R0 + GPR_idx).addReg(ArgReg).addReg(ArgReg);
		} else {
		    BuildMI(BB, PPC32::STW, 3).addReg(ArgReg).addImm(ArgOffset).addReg(PPC32::R1);
		}
		break;
      case cLong:
		ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;

		// Reg or stack?
		if (GPR_remaining > 1) {
		    BuildMI(BB, PPC32::OR, 2, PPC32::R0 + GPR_idx).addReg(ArgReg).addReg(ArgReg);
		    BuildMI(BB, PPC32::OR, 2, PPC32::R0 + GPR_idx + 1).addReg(ArgReg+1).addReg(ArgReg+1);
		} else {
		    BuildMI(BB, PPC32::STW, 3).addReg(ArgReg).addImm(ArgOffset).addReg(PPC32::R1);
		    BuildMI(BB, PPC32::STW, 3).addReg(ArgReg+1).addImm(ArgOffset+4).addReg(PPC32::R1);
		}

        ArgOffset += 4;        // 8 byte entry, not 4.
		if (GPR_remaining > 0) {
			GPR_remaining -= 1;    // uses up 2 GPRs
			GPR_idx += 1;
		}
        break;
      case cFP:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
        if (Args[i].Ty == Type::FloatTy) {
			// Reg or stack?
			if (FPR_remaining > 0) {
				BuildMI(BB, PPC32::FMR, 1, PPC32::F0 + FPR_idx).addReg(ArgReg);
				FPR_remaining--;
				FPR_idx++;
			} else {
				BuildMI(BB, PPC32::STFS, 3).addReg(ArgReg).addImm(ArgOffset).addReg(PPC32::R1);
			}
        } else {
          assert(Args[i].Ty == Type::DoubleTy && "Unknown FP type!");
			// Reg or stack?
			if (FPR_remaining > 0) {
				BuildMI(BB, PPC32::FMR, 1, PPC32::F0 + FPR_idx).addReg(ArgReg);
				FPR_remaining--;
				FPR_idx++;
			} else {
				BuildMI(BB, PPC32::STFD, 3).addReg(ArgReg).addImm(ArgOffset).addReg(PPC32::R1);
			}

			ArgOffset += 4;       // 8 byte entry, not 4.
			if (GPR_remaining > 0) {
				GPR_remaining--;    // uses up 2 GPRs
				GPR_idx++;
			}
        }
        break;

      default: assert(0 && "Unknown class!");
      }
      ArgOffset += 4;
	  if (GPR_remaining > 0) {
		GPR_remaining--;    // uses up 2 GPRs
		GPR_idx++;
	  }
    }
  } else {
    BuildMI(BB, PPC32::ADJCALLSTACKDOWN, 1).addImm(0);
  }

  BB->push_back(CallMI);

  BuildMI(BB, PPC32::ADJCALLSTACKUP, 1).addImm(NumBytes);

  // If there is a return value, scavenge the result from the location the call
  // leaves it in...
  //
  if (Ret.Ty != Type::VoidTy) {
    unsigned DestClass = getClassB(Ret.Ty);
    switch (DestClass) {
    case cByte:
    case cShort:
    case cInt:
      // Integral results are in r3
	  BuildMI(BB, PPC32::OR, 2, Ret.Reg).addReg(PPC32::R3).addReg(PPC32::R3);
    case cFP:     // Floating-point return values live in f1
      BuildMI(BB, PPC32::FMR, 1, Ret.Reg).addReg(PPC32::F1);
      break;
    case cLong:   // Long values are in r3:r4
	  BuildMI(BB, PPC32::OR, 2, Ret.Reg).addReg(PPC32::R3).addReg(PPC32::R3);
	  BuildMI(BB, PPC32::OR, 2, Ret.Reg+1).addReg(PPC32::R4).addReg(PPC32::R4);
      break;
    default: assert(0 && "Unknown class!");
    }
  }
}


/// visitCallInst - Push args on stack and do a procedure call instruction.
void ISel::visitCallInst(CallInst &CI) {
  MachineInstr *TheCall;
  if (Function *F = CI.getCalledFunction()) {
    // Is it an intrinsic function call?
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID()) {
      visitIntrinsicCall(ID, CI);   // Special intrinsics are not handled here
      return;
    }

    // Emit a CALL instruction with PC-relative displacement.
    TheCall = BuildMI(PPC32::CALLpcrel, 1).addGlobalAddress(F, true);
  } else {  // Emit an indirect call through the CTR
    unsigned Reg = getReg(CI.getCalledValue());
    BuildMI(PPC32::MTSPR, 2).addZImm(9).addReg(Reg);
    TheCall = BuildMI(PPC32::CALLindirect, 1).addZImm(20).addZImm(0);
  }

  std::vector<ValueRecord> Args;
  for (unsigned i = 1, e = CI.getNumOperands(); i != e; ++i)
    Args.push_back(ValueRecord(CI.getOperand(i)));

  unsigned DestReg = CI.getType() != Type::VoidTy ? getReg(CI) : 0;
  doCall(ValueRecord(DestReg, CI.getType()), TheCall, Args);
}         


/// dyncastIsNan - Return the operand of an isnan operation if this is an isnan.
///
static Value *dyncastIsNan(Value *V) {
  if (CallInst *CI = dyn_cast<CallInst>(V))
    if (Function *F = CI->getCalledFunction())
      if (F->getIntrinsicID() == Intrinsic::isnan)
        return CI->getOperand(1);
  return 0;
}

/// isOnlyUsedByUnorderedComparisons - Return true if this value is only used by
/// or's whos operands are all calls to the isnan predicate.
static bool isOnlyUsedByUnorderedComparisons(Value *V) {
  assert(dyncastIsNan(V) && "The value isn't an isnan call!");

  // Check all uses, which will be or's of isnans if this predicate is true.
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;++UI){
    Instruction *I = cast<Instruction>(*UI);
    if (I->getOpcode() != Instruction::Or) return false;
    if (I->getOperand(0) != V && !dyncastIsNan(I->getOperand(0))) return false;
    if (I->getOperand(1) != V && !dyncastIsNan(I->getOperand(1))) return false;
  }

  return true;
}

/// LowerUnknownIntrinsicFunctionCalls - This performs a prepass over the
/// function, lowering any calls to unknown intrinsic functions into the
/// equivalent LLVM code.
///
void ISel::LowerUnknownIntrinsicFunctionCalls(Function &F) {
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(I++))
        if (Function *F = CI->getCalledFunction())
          switch (F->getIntrinsicID()) {
          case Intrinsic::not_intrinsic:
          case Intrinsic::vastart:
          case Intrinsic::vacopy:
          case Intrinsic::vaend:
          case Intrinsic::returnaddress:
          case Intrinsic::frameaddress:
          case Intrinsic::isnan:
            // We directly implement these intrinsics
            break;
          case Intrinsic::readio: {
            // On PPC, memory operations are in-order.  Lower this intrinsic
            // into a volatile load.
            Instruction *Before = CI->getPrev();
            LoadInst * LI = new LoadInst(CI->getOperand(1), "", true, CI);
            CI->replaceAllUsesWith(LI);
            BB->getInstList().erase(CI);
            break;
          }
          case Intrinsic::writeio: {
            // On PPC, memory operations are in-order.  Lower this intrinsic
            // into a volatile store.
            Instruction *Before = CI->getPrev();
            StoreInst *LI = new StoreInst(CI->getOperand(1),
                                          CI->getOperand(2), true, CI);
            CI->replaceAllUsesWith(LI);
            BB->getInstList().erase(CI);
            break;
          }
          default:
            // All other intrinsic calls we must lower.
            Instruction *Before = CI->getPrev();
            TM.getIntrinsicLowering().LowerIntrinsicCall(CI);
            if (Before) {        // Move iterator to instruction after call
              I = Before; ++I;
            } else {
              I = BB->begin();
            }
          }
}

void ISel::visitIntrinsicCall(Intrinsic::ID ID, CallInst &CI) {
  unsigned TmpReg1, TmpReg2, TmpReg3;
  switch (ID) {
  case Intrinsic::vastart:
    // Get the address of the first vararg value...
    TmpReg1 = getReg(CI);
    addFrameReference(BuildMI(BB, PPC32::ADDI, 2, TmpReg1), VarArgsFrameIndex);
    return;

  case Intrinsic::vacopy:
    TmpReg1 = getReg(CI);
    TmpReg2 = getReg(CI.getOperand(1));
    BuildMI(BB, PPC32::OR, 2, TmpReg1).addReg(TmpReg2).addReg(TmpReg2);
    return;
  case Intrinsic::vaend: return;

  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
    TmpReg1 = getReg(CI);
    if (cast<Constant>(CI.getOperand(1))->isNullValue()) {
      if (ID == Intrinsic::returnaddress) {
        // Just load the return address
        addFrameReference(BuildMI(BB, PPC32::LWZ, 2, TmpReg1),
                          ReturnAddressIndex);
      } else {
        addFrameReference(BuildMI(BB, PPC32::ADDI, 2, TmpReg1),
                          ReturnAddressIndex, -4, false);
      }
    } else {
      // Values other than zero are not implemented yet.
      BuildMI(BB, PPC32::ADDI, 2, TmpReg1).addReg(PPC32::R0).addImm(0);
    }
    return;

  case Intrinsic::isnan:
    // If this is only used by 'isunordered' style comparisons, don't emit it.
    if (isOnlyUsedByUnorderedComparisons(&CI)) return;
    TmpReg1 = getReg(CI.getOperand(1));
    emitUCOM(BB, BB->end(), TmpReg1, TmpReg1);
	TmpReg2 = makeAnotherReg(Type::IntTy);
	BuildMI(BB, PPC32::MFCR, TmpReg2);
    TmpReg3 = getReg(CI);
    BuildMI(BB, PPC32::RLWINM, 4, TmpReg3).addReg(TmpReg2).addImm(4).addImm(31).addImm(31);
    return;

  default: assert(0 && "Error: unknown intrinsics should have been lowered!");
  }
}

/// visitSimpleBinary - Implement simple binary operators for integral types...
/// OperatorClass is one of: 0 for Add, 1 for Sub, 2 for And, 3 for Or, 4 for
/// Xor.
///
void ISel::visitSimpleBinary(BinaryOperator &B, unsigned OperatorClass) {
  unsigned DestReg = getReg(B);
  MachineBasicBlock::iterator MI = BB->end();
  Value *Op0 = B.getOperand(0), *Op1 = B.getOperand(1);
  unsigned Class = getClassB(B.getType());

  emitSimpleBinaryOperation(BB, MI, Op0, Op1, OperatorClass, DestReg);
}

/// emitBinaryFPOperation - This method handles emission of floating point
/// Add (0), Sub (1), Mul (2), and Div (3) operations.
void ISel::emitBinaryFPOperation(MachineBasicBlock *BB,
                                 MachineBasicBlock::iterator IP,
                                 Value *Op0, Value *Op1,
                                 unsigned OperatorClass, unsigned DestReg) {

  // Special case: op Reg, <const fp>
  if (ConstantFP *Op1C = dyn_cast<ConstantFP>(Op1)) {
      // Create a constant pool entry for this constant.
      MachineConstantPool *CP = F->getConstantPool();
      unsigned CPI = CP->getConstantPoolIndex(Op1C);
      const Type *Ty = Op1->getType();

      static const unsigned OpcodeTab[][4] = {
        { PPC32::FADDS, PPC32::FSUBS, PPC32::FMULS, PPC32::FDIVS },   // Float
        { PPC32::FADD, PPC32::FSUB, PPC32::FMUL, PPC32::FDIV },   // Double
      };

      assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");
	  unsigned TempReg = makeAnotherReg(Ty);
      unsigned LoadOpcode = Ty == Type::FloatTy ? PPC32::LFS : PPC32::LFD;
      addConstantPoolReference(BuildMI(*BB, IP, LoadOpcode, 2, TempReg), CPI);

      unsigned Opcode = OpcodeTab[Ty != Type::FloatTy][OperatorClass];
      unsigned Op0r = getReg(Op0, BB, IP);
	  BuildMI(*BB, IP, Opcode, DestReg).addReg(Op0r).addReg(TempReg);
      return;
    }
  
  // Special case: R1 = op <const fp>, R2
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(Op0))
    if (CFP->isExactlyValue(-0.0) && OperatorClass == 1) {
      // -0.0 - X === -X
      unsigned op1Reg = getReg(Op1, BB, IP);
      BuildMI(*BB, IP, PPC32::FNEG, 1, DestReg).addReg(op1Reg);
      return;
    } else {
      // R1 = op CST, R2  -->  R1 = opr R2, CST

      // Create a constant pool entry for this constant.
      MachineConstantPool *CP = F->getConstantPool();
      unsigned CPI = CP->getConstantPoolIndex(CFP);
      const Type *Ty = CFP->getType();

      static const unsigned OpcodeTab[][4] = {
        { PPC32::FADDS, PPC32::FSUBS, PPC32::FMULS, PPC32::FDIVS },   // Float
        { PPC32::FADD, PPC32::FSUB, PPC32::FMUL, PPC32::FDIV },   // Double
      };

      assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");
	  unsigned TempReg = makeAnotherReg(Ty);
      unsigned LoadOpcode = Ty == Type::FloatTy ? PPC32::LFS : PPC32::LFD;
      addConstantPoolReference(BuildMI(*BB, IP, LoadOpcode, 2, TempReg), CPI);

      unsigned Opcode = OpcodeTab[Ty != Type::FloatTy][OperatorClass];
      unsigned Op1r = getReg(Op1, BB, IP);
	  BuildMI(*BB, IP, Opcode, DestReg).addReg(TempReg).addReg(Op1r);
      return;
    }

  // General case.
  static const unsigned OpcodeTab[4] = {
    PPC32::FADD, PPC32::FSUB, PPC32::FMUL, PPC32::FDIV
  };

  unsigned Opcode = OpcodeTab[OperatorClass];
  unsigned Op0r = getReg(Op0, BB, IP);
  unsigned Op1r = getReg(Op1, BB, IP);
  BuildMI(*BB, IP, Opcode, 2, DestReg).addReg(Op0r).addReg(Op1r);
}

/// emitSimpleBinaryOperation - Implement simple binary operators for integral
/// types...  OperatorClass is one of: 0 for Add, 1 for Sub, 2 for And, 3 for
/// Or, 4 for Xor.
///
/// emitSimpleBinaryOperation - Common code shared between visitSimpleBinary
/// and constant expression support.
///
void ISel::emitSimpleBinaryOperation(MachineBasicBlock *MBB,
                                     MachineBasicBlock::iterator IP,
                                     Value *Op0, Value *Op1,
                                     unsigned OperatorClass, unsigned DestReg) {
  unsigned Class = getClassB(Op0->getType());

    // Arithmetic and Bitwise operators
    static const unsigned OpcodeTab[5] = {
	  PPC32::ADD, PPC32::SUB, PPC32::AND, PPC32::OR, PPC32::XOR
	};
    // Otherwise, code generate the full operation with a constant.
    static const unsigned BottomTab[] = {
      PPC32::ADDC, PPC32::SUBC, PPC32::AND, PPC32::OR, PPC32::XOR
    };
    static const unsigned TopTab[] = {
      PPC32::ADDE, PPC32::SUBFE, PPC32::AND, PPC32::OR, PPC32::XOR
    };
  
  if (Class == cFP) {
    assert(OperatorClass < 2 && "No logical ops for FP!");
    emitBinaryFPOperation(MBB, IP, Op0, Op1, OperatorClass, DestReg);
    return;
  }

  if (Op0->getType() == Type::BoolTy) {
    if (OperatorClass == 3)
      // If this is an or of two isnan's, emit an FP comparison directly instead
      // of or'ing two isnan's together.
      if (Value *LHS = dyncastIsNan(Op0))
        if (Value *RHS = dyncastIsNan(Op1)) {
          unsigned Op0Reg = getReg(RHS, MBB, IP), Op1Reg = getReg(LHS, MBB, IP);
		  unsigned TmpReg = makeAnotherReg(Type::IntTy);
          emitUCOM(MBB, IP, Op0Reg, Op1Reg);
		  BuildMI(*MBB, IP, PPC32::MFCR, TmpReg);
		  BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(TmpReg).addImm(4).addImm(31).addImm(31);
          return;
        }
  }

  // sub 0, X -> neg X
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op0))
    if (OperatorClass == 1 && CI->isNullValue()) {
      unsigned op1Reg = getReg(Op1, MBB, IP);
      BuildMI(*MBB, IP, PPC32::NEG, 1, DestReg).addReg(op1Reg);
      
      if (Class == cLong) {
		unsigned zeroes = makeAnotherReg(Type::IntTy);
		unsigned overflow = makeAnotherReg(Type::IntTy);
        unsigned T = makeAnotherReg(Type::IntTy);
		BuildMI(*MBB, IP, PPC32::CNTLZW, 1, zeroes).addReg(op1Reg);
		BuildMI(*MBB, IP, PPC32::RLWINM, 4, overflow).addReg(zeroes).addImm(27).addImm(5).addImm(31);
		BuildMI(*MBB, IP, PPC32::ADD, 2, T).addReg(op1Reg+1).addReg(overflow);
		BuildMI(*MBB, IP, PPC32::NEG, 1, DestReg+1).addReg(T);
      }
      return;
    }

  // Special case: op Reg, <const int>
  if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
    unsigned Op0r = getReg(Op0, MBB, IP);

    // xor X, -1 -> not X
    if (OperatorClass == 4 && Op1C->isAllOnesValue()) {
      BuildMI(*MBB, IP, PPC32::NOR, 2, DestReg).addReg(Op0r).addReg(Op0r);
      if (Class == cLong)  // Invert the top part too
        BuildMI(*MBB, IP, PPC32::NOR, 2, DestReg+1).addReg(Op0r+1).addReg(Op0r+1);
      return;
    }

    unsigned Opcode = OpcodeTab[OperatorClass];
    unsigned Op1r = getReg(Op1, MBB, IP);

    if (Class != cLong) {
      BuildMI(*MBB, IP, Opcode, 2, DestReg).addReg(Op0r).addReg(Op1r);
      return;
    }
    
    // If the constant is zero in the low 32-bits, just copy the low part
    // across and apply the normal 32-bit operation to the high parts.  There
    // will be no carry or borrow into the top.
    if (cast<ConstantInt>(Op1C)->getRawValue() == 0) {
      if (OperatorClass != 2) // All but and...
        BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(Op0r).addReg(Op0r);
      else
        BuildMI(*MBB, IP, PPC32::ADDI, 2, DestReg).addReg(PPC32::R0).addImm(0);
	  BuildMI(*MBB, IP, Opcode, 2, DestReg+1).addReg(Op0r+1).addReg(Op1r+1);
      return;
    }
    
    // If this is a long value and the high or low bits have a special
    // property, emit some special cases.
    unsigned Op1h = cast<ConstantInt>(Op1C)->getRawValue() >> 32LL;
    
    // If this is a logical operation and the top 32-bits are zero, just
    // operate on the lower 32.
    if (Op1h == 0 && OperatorClass > 1) {
      BuildMI(*MBB, IP, Opcode, 2, DestReg).addReg(Op0r).addReg(Op1r);
      if (OperatorClass != 2)  // All but and
        BuildMI(*MBB, IP, PPC32::OR, 2, DestReg+1).addReg(Op0r+1).addReg(Op0r+1);
      else
        BuildMI(*MBB, IP, PPC32::ADDI, 2, DestReg+1).addReg(PPC32::R0).addImm(0);
      return;
    }
    
    // TODO: We could handle lots of other special cases here, such as AND'ing
    // with 0xFFFFFFFF00000000 -> noop, etc.
    
    BuildMI(*MBB, IP, BottomTab[OperatorClass], 2, DestReg).addReg(Op0r).addImm(Op1r);
    BuildMI(*MBB, IP, TopTab[OperatorClass], 2, DestReg+1).addReg(Op0r+1).addImm(Op1r+1);
    return;
  }

  unsigned Op0r = getReg(Op0, MBB, IP);
  unsigned Op1r = getReg(Op1, MBB, IP);

  if (Class != cLong) {
	unsigned Opcode = OpcodeTab[OperatorClass];
	BuildMI(*MBB, IP, Opcode, 2, DestReg).addReg(Op0r).addReg(Op1r);
  } else {
    BuildMI(*MBB, IP, BottomTab[OperatorClass], 2, DestReg).addReg(Op0r).addImm(Op1r);
    BuildMI(*MBB, IP, TopTab[OperatorClass], 2, DestReg+1).addReg(Op0r+1).addImm(Op1r+1);
  }
  return;
}

/// doMultiply - Emit appropriate instructions to multiply together the
/// registers op0Reg and op1Reg, and put the result in DestReg.  The type of the
/// result should be given as DestTy.
///
void ISel::doMultiply(MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBI,
                      unsigned DestReg, const Type *DestTy,
                      unsigned op0Reg, unsigned op1Reg) {
  unsigned Class = getClass(DestTy);
  switch (Class) {
  case cLong:
    BuildMI(*MBB, MBBI, PPC32::MULHW, 2, DestReg+1).addReg(op0Reg+1).addReg(op1Reg+1);
  case cInt:
  case cShort:
  case cByte:
    BuildMI(*MBB, MBBI, PPC32::MULLW, 2, DestReg).addReg(op0Reg).addReg(op1Reg);
    return;
  default:
	assert(0 && "doMultiply cannot operate on unknown type!");
  }
}

// ExactLog2 - This function solves for (Val == 1 << (N-1)) and returns N.  It
// returns zero when the input is not exactly a power of two.
static unsigned ExactLog2(unsigned Val) {
  if (Val == 0 || (Val & (Val-1))) return 0;
  unsigned Count = 0;
  while (Val != 1) {
    Val >>= 1;
    ++Count;
  }
  return Count+1;
}


/// doMultiplyConst - This function is specialized to efficiently codegen an 8,
/// 16, or 32-bit integer multiply by a constant.
void ISel::doMultiplyConst(MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator IP,
                           unsigned DestReg, const Type *DestTy,
                           unsigned op0Reg, unsigned ConstRHS) {
  unsigned Class = getClass(DestTy);
  // Handle special cases here.
  switch (ConstRHS) {
  case 0:
    BuildMI(*MBB, IP, PPC32::ADDI, 2, DestReg).addReg(PPC32::R0).addImm(0);
    return;
  case 1:
    BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(op0Reg).addReg(op0Reg);
    return;
  case 2:
    BuildMI(*MBB, IP, PPC32::ADD, 2,DestReg).addReg(op0Reg).addReg(op0Reg);
    return;
  }

  // If the element size is exactly a power of 2, use a shift to get it.
  if (unsigned Shift = ExactLog2(ConstRHS)) {
    switch (Class) {
    default: assert(0 && "Unknown class for this function!");
    case cByte:
    case cShort:
    case cInt:
      BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(op0Reg).addImm(Shift-1).addImm(0).addImm(31-Shift-1);
      return;
    }
  }
  
  // Most general case, emit a normal multiply...
  unsigned TmpReg1 = makeAnotherReg(Type::IntTy);
  unsigned TmpReg2 = makeAnotherReg(Type::IntTy);
  BuildMI(*MBB, IP, PPC32::ADDIS, 2, TmpReg1).addReg(PPC32::R0).addImm(ConstRHS >> 16);
  BuildMI(*MBB, IP, PPC32::ORI, 2, TmpReg2).addReg(TmpReg1).addImm(ConstRHS);
  
  // Emit a MUL to multiply the register holding the index by
  // elementSize, putting the result in OffsetReg.
  doMultiply(MBB, IP, DestReg, DestTy, op0Reg, TmpReg2);
}

void ISel::visitMul(BinaryOperator &I) {
  unsigned ResultReg = getReg(I);

  Value *Op0 = I.getOperand(0);
  Value *Op1 = I.getOperand(1);

  MachineBasicBlock::iterator IP = BB->end();
  emitMultiply(BB, IP, Op0, Op1, ResultReg);
}

void ISel::emitMultiply(MachineBasicBlock *MBB, MachineBasicBlock::iterator IP,
                        Value *Op0, Value *Op1, unsigned DestReg) {
  MachineBasicBlock &BB = *MBB;
  TypeClass Class = getClass(Op0->getType());

  // Simple scalar multiply?
  unsigned Op0Reg  = getReg(Op0, &BB, IP);
  switch (Class) {
  case cByte:
  case cShort:
  case cInt:
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
      unsigned Val = (unsigned)CI->getRawValue(); // Isn't a 64-bit constant
      doMultiplyConst(&BB, IP, DestReg, Op0->getType(), Op0Reg, Val);
    } else {
      unsigned Op1Reg  = getReg(Op1, &BB, IP);
      doMultiply(&BB, IP, DestReg, Op1->getType(), Op0Reg, Op1Reg);
    }
    return;
  case cFP:
    emitBinaryFPOperation(MBB, IP, Op0, Op1, 2, DestReg);
    return;
  case cLong:
    break;
  }

  // Long value.  We have to do things the hard way...
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    unsigned CLow = CI->getRawValue();
    unsigned CHi  = CI->getRawValue() >> 32;
    
    if (CLow == 0) {
      // If the low part of the constant is all zeros, things are simple.
      BuildMI(BB, IP, PPC32::ADDI, 2, DestReg).addReg(PPC32::R0).addImm(0);
      doMultiplyConst(&BB, IP, DestReg+1, Type::UIntTy, Op0Reg, CHi);
      return;
    }
    
    // Multiply the two low parts
    unsigned OverflowReg = 0;
    if (CLow == 1) {
      BuildMI(BB, IP, PPC32::OR, 2, DestReg).addReg(Op0Reg).addReg(Op0Reg);
    } else {
	  unsigned TmpRegL = makeAnotherReg(Type::UIntTy);
      unsigned Op1RegL = makeAnotherReg(Type::UIntTy);
      OverflowReg = makeAnotherReg(Type::UIntTy);
	  BuildMI(BB, IP, PPC32::ADDIS, 2, TmpRegL).addReg(PPC32::R0).addImm(CLow >> 16);
	  BuildMI(BB, IP, PPC32::ORI, 2, Op1RegL).addReg(TmpRegL).addImm(CLow);
	  BuildMI(BB, IP, PPC32::MULLW, 2, DestReg).addReg(Op0Reg).addReg(Op1RegL);
	  BuildMI(BB, IP, PPC32::MULHW, 2, OverflowReg).addReg(Op0Reg).addReg(Op1RegL);
    }
    
    unsigned AHBLReg = makeAnotherReg(Type::UIntTy);
    doMultiplyConst(&BB, IP, AHBLReg, Type::UIntTy, Op0Reg+1, CLow);
    
    unsigned AHBLplusOverflowReg;
    if (OverflowReg) {
      AHBLplusOverflowReg = makeAnotherReg(Type::UIntTy);
      BuildMI(BB, IP, PPC32::ADD, 2,                // AH*BL+(AL*BL >> 32)
              AHBLplusOverflowReg).addReg(AHBLReg).addReg(OverflowReg);
    } else {
      AHBLplusOverflowReg = AHBLReg;
    }
    
    if (CHi == 0) {
      BuildMI(BB, IP, PPC32::OR, 2, DestReg+1).addReg(AHBLplusOverflowReg).addReg(AHBLplusOverflowReg);
    } else {
      unsigned ALBHReg = makeAnotherReg(Type::UIntTy); // AL*BH
      doMultiplyConst(&BB, IP, ALBHReg, Type::UIntTy, Op0Reg, CHi);
      
      BuildMI(BB, IP, PPC32::ADD, 2,      // AL*BH + AH*BL + (AL*BL >> 32)
              DestReg+1).addReg(AHBLplusOverflowReg).addReg(ALBHReg);
    }
    return;
  }

  // General 64x64 multiply

  unsigned Op1Reg  = getReg(Op1, &BB, IP);
  
  // Multiply the two low parts... capturing carry into EDX
  BuildMI(BB, IP, PPC32::MULLW, 2, DestReg).addReg(Op0Reg).addReg(Op1Reg);  // AL*BL
  
  unsigned OverflowReg = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, IP, PPC32::MULHW, 2, OverflowReg).addReg(Op0Reg).addReg(Op1Reg); // AL*BL >> 32
  
  unsigned AHBLReg = makeAnotherReg(Type::UIntTy);   // AH*BL
  BuildMI(BB, IP, PPC32::MULLW, 2, AHBLReg).addReg(Op0Reg+1).addReg(Op1Reg);
  
  unsigned AHBLplusOverflowReg = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, IP, PPC32::ADD, 2,                // AH*BL+(AL*BL >> 32)
          AHBLplusOverflowReg).addReg(AHBLReg).addReg(OverflowReg);
  
  unsigned ALBHReg = makeAnotherReg(Type::UIntTy); // AL*BH
  BuildMI(BB, IP, PPC32::MULLW, 2, ALBHReg).addReg(Op0Reg).addReg(Op1Reg+1);
  
  BuildMI(BB, IP, PPC32::ADD, 2,      // AL*BH + AH*BL + (AL*BL >> 32)
          DestReg+1).addReg(AHBLplusOverflowReg).addReg(ALBHReg);
}


/// visitDivRem - Handle division and remainder instructions... these
/// instruction both require the same instructions to be generated, they just
/// select the result from a different register.  Note that both of these
/// instructions work differently for signed and unsigned operands.
///
void ISel::visitDivRem(BinaryOperator &I) {
  unsigned ResultReg = getReg(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  MachineBasicBlock::iterator IP = BB->end();
  emitDivRemOperation(BB, IP, Op0, Op1, I.getOpcode() == Instruction::Div, ResultReg);
}

void ISel::emitDivRemOperation(MachineBasicBlock *BB,
                               MachineBasicBlock::iterator IP,
                               Value *Op0, Value *Op1, bool isDiv,
                               unsigned ResultReg) {
  const Type *Ty = Op0->getType();
  unsigned Class = getClass(Ty);
  switch (Class) {
  case cFP:              // Floating point divide
    if (isDiv) {
      emitBinaryFPOperation(BB, IP, Op0, Op1, 3, ResultReg);
      return;
    } else {               // Floating point remainder...
      unsigned Op0Reg = getReg(Op0, BB, IP);
      unsigned Op1Reg = getReg(Op1, BB, IP);
      MachineInstr *TheCall =
        BuildMI(PPC32::CALLpcrel, 1).addExternalSymbol("fmod", true);
      std::vector<ValueRecord> Args;
      Args.push_back(ValueRecord(Op0Reg, Type::DoubleTy));
      Args.push_back(ValueRecord(Op1Reg, Type::DoubleTy));
      doCall(ValueRecord(ResultReg, Type::DoubleTy), TheCall, Args);
    }
    return;
  case cLong: {
    static const char *FnName[] =
      { "__moddi3", "__divdi3", "__umoddi3", "__udivdi3" };
    unsigned Op0Reg = getReg(Op0, BB, IP);
    unsigned Op1Reg = getReg(Op1, BB, IP);
    unsigned NameIdx = Ty->isUnsigned()*2 + isDiv;
    MachineInstr *TheCall =
      BuildMI(PPC32::CALLpcrel, 1).addExternalSymbol(FnName[NameIdx], true);

    std::vector<ValueRecord> Args;
    Args.push_back(ValueRecord(Op0Reg, Type::LongTy));
    Args.push_back(ValueRecord(Op1Reg, Type::LongTy));
    doCall(ValueRecord(ResultReg, Type::LongTy), TheCall, Args);
    return;
  }
  case cByte: case cShort: case cInt:
    break;          // Small integrals, handled below...
  default: assert(0 && "Unknown class!");
  }

  // Special case signed division by power of 2.
  if (isDiv)
    if (ConstantSInt *CI = dyn_cast<ConstantSInt>(Op1)) {
      assert(Class != cLong && "This doesn't handle 64-bit divides!");
      int V = CI->getValue();

      if (V == 1) {       // X /s 1 => X
        unsigned Op0Reg = getReg(Op0, BB, IP);
        BuildMI(*BB, IP, PPC32::OR, 2, ResultReg).addReg(Op0Reg).addReg(Op0Reg);
        return;
      }

      if (V == -1) {      // X /s -1 => -X
        unsigned Op0Reg = getReg(Op0, BB, IP);
        BuildMI(*BB, IP, PPC32::NEG, 1, ResultReg).addReg(Op0Reg);
        return;
      }

      bool isNeg = false;
      if (V < 0) {         // Not a positive power of 2?
        V = -V;
        isNeg = true;      // Maybe it's a negative power of 2.
      }
      if (unsigned Log = ExactLog2(V)) {
        --Log;
        unsigned Op0Reg = getReg(Op0, BB, IP);
        unsigned TmpReg = makeAnotherReg(Op0->getType());
        if (Log != 1) 
          BuildMI(*BB, IP, PPC32::SRAWI, 2, TmpReg).addReg(Op0Reg).addImm(Log-1);
        else
          BuildMI(*BB, IP, PPC32::OR, 2, TmpReg).addReg(Op0Reg).addReg(Op0Reg);

        unsigned TmpReg2 = makeAnotherReg(Op0->getType());
        BuildMI(*BB, IP, PPC32::RLWINM, 4, TmpReg2).addReg(TmpReg).addImm(Log).addImm(32-Log).addImm(31);

        unsigned TmpReg3 = makeAnotherReg(Op0->getType());
        BuildMI(*BB, IP, PPC32::ADD, 2, TmpReg3).addReg(Op0Reg).addReg(TmpReg2);

        unsigned TmpReg4 = isNeg ? makeAnotherReg(Op0->getType()) : ResultReg;
        BuildMI(*BB, IP, PPC32::SRAWI, 2, TmpReg4).addReg(Op0Reg).addImm(Log);

        if (isNeg)
          BuildMI(*BB, IP, PPC32::NEG, 1, ResultReg).addReg(TmpReg4);
        return;
      }
    }

  unsigned Op0Reg = getReg(Op0, BB, IP);
  unsigned Op1Reg = getReg(Op1, BB, IP);

  if (isDiv) {
	if (Ty->isSigned()) {
		BuildMI(*BB, IP, PPC32::DIVW, 2, ResultReg).addReg(Op0Reg).addReg(Op1Reg);
	} else {
		BuildMI(*BB, IP, PPC32::DIVWU, 2, ResultReg).addReg(Op0Reg).addReg(Op1Reg);
	}
  } else { // Remainder
	unsigned TmpReg1 = makeAnotherReg(Op0->getType());
	unsigned TmpReg2 = makeAnotherReg(Op0->getType());
	
	if (Ty->isSigned()) {
		BuildMI(*BB, IP, PPC32::DIVW, 2, TmpReg1).addReg(Op0Reg).addReg(Op1Reg);
	} else {
		BuildMI(*BB, IP, PPC32::DIVWU, 2, TmpReg1).addReg(Op0Reg).addReg(Op1Reg);
	}
	BuildMI(*BB, IP, PPC32::MULLW, 2, TmpReg2).addReg(TmpReg1).addReg(Op1Reg);
	BuildMI(*BB, IP, PPC32::SUBF, 2, ResultReg).addReg(TmpReg2).addReg(Op0Reg);
  }
}


/// Shift instructions: 'shl', 'sar', 'shr' - Some special cases here
/// for constant immediate shift values, and for constant immediate
/// shift values equal to 1. Even the general case is sort of special,
/// because the shift amount has to be in CL, not just any old register.
///
void ISel::visitShiftInst(ShiftInst &I) {
  MachineBasicBlock::iterator IP = BB->end ();
  emitShiftOperation (BB, IP, I.getOperand (0), I.getOperand (1),
                      I.getOpcode () == Instruction::Shl, I.getType (),
                      getReg (I));
}

/// emitShiftOperation - Common code shared between visitShiftInst and
/// constant expression support.
void ISel::emitShiftOperation(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator IP,
                              Value *Op, Value *ShiftAmount, bool isLeftShift,
                              const Type *ResultTy, unsigned DestReg) {
  unsigned SrcReg = getReg (Op, MBB, IP);
  bool isSigned = ResultTy->isSigned ();
  unsigned Class = getClass (ResultTy);
  
  // Longs, as usual, are handled specially...
  if (Class == cLong) {
    // If we have a constant shift, we can generate much more efficient code
    // than otherwise...
    //
    if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(ShiftAmount)) {
      unsigned Amount = CUI->getValue();
      if (Amount < 32) {
        if (isLeftShift) {
		  // FIXME: RLWIMI is a use-and-def of DestReg+1, but that violates SSA
          BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg+1).addReg(SrcReg+1).addImm(Amount).addImm(0).addImm(31-Amount);
          BuildMI(*MBB, IP, PPC32::RLWIMI, 5).addReg(DestReg+1).addReg(SrcReg).addImm(Amount).addImm(32-Amount).addImm(31);
          BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg).addImm(Amount).addImm(0).addImm(31-Amount);
        } else {
		  // FIXME: RLWIMI is a use-and-def of DestReg, but that violates SSA
          BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg).addImm(32-Amount).addImm(Amount).addImm(31);
          BuildMI(*MBB, IP, PPC32::RLWIMI, 5).addReg(DestReg).addReg(SrcReg+1).addImm(32-Amount).addImm(0).addImm(Amount-1);
          BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg+1).addReg(SrcReg+1).addImm(32-Amount).addImm(Amount).addImm(31);
        }
      } else {                 // Shifting more than 32 bits
        Amount -= 32;
        if (isLeftShift) {
          if (Amount != 0) {
			BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg+1).addReg(SrcReg).addImm(Amount).addImm(0).addImm(31-Amount);
          } else {
            BuildMI(*MBB, IP, PPC32::OR, 2, DestReg+1).addReg(SrcReg).addReg(SrcReg);
          }
          BuildMI(*MBB, IP, PPC32::ADDI, 2, DestReg).addReg(PPC32::R0).addImm(0);
        } else {
          if (Amount != 0) {
			if (isSigned)
				BuildMI(*MBB, IP, PPC32::SRAWI, 2, DestReg).addReg(SrcReg+1).addImm(Amount);
			else
				BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg+1).addImm(32-Amount).addImm(Amount).addImm(31);
          } else {
            BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(SrcReg+1).addReg(SrcReg+1);
          }
          BuildMI(*MBB, IP, PPC32::ADDI, 2, DestReg+1).addReg(PPC32::R0).addImm(0);
        }
      }
    } else {
      unsigned TmpReg1 = makeAnotherReg(Type::IntTy);
      unsigned TmpReg2 = makeAnotherReg(Type::IntTy);
	  unsigned TmpReg3 = makeAnotherReg(Type::IntTy);
	  unsigned TmpReg4 = makeAnotherReg(Type::IntTy);
	  unsigned TmpReg5 = makeAnotherReg(Type::IntTy);
	  unsigned TmpReg6 = makeAnotherReg(Type::IntTy);
	  unsigned ShiftAmountReg = getReg (ShiftAmount, MBB, IP);
	  
      if (isLeftShift) {
		BuildMI(*MBB, IP, PPC32::SUBFIC, 2, TmpReg1).addReg(ShiftAmountReg).addImm(32);
		BuildMI(*MBB, IP, PPC32::SLW, 2, TmpReg2).addReg(SrcReg+1).addReg(ShiftAmountReg);
		BuildMI(*MBB, IP, PPC32::SRW, 2, TmpReg3).addReg(SrcReg).addReg(TmpReg1);
		BuildMI(*MBB, IP, PPC32::OR, 2, TmpReg4).addReg(TmpReg2).addReg(TmpReg3);
		BuildMI(*MBB, IP, PPC32::ADDI, 2, TmpReg5).addReg(ShiftAmountReg).addImm(-32);
		BuildMI(*MBB, IP, PPC32::SLW, 2, TmpReg6).addReg(SrcReg).addReg(TmpReg5);
		BuildMI(*MBB, IP, PPC32::OR, 2, DestReg+1).addReg(TmpReg4).addReg(TmpReg6);
		BuildMI(*MBB, IP, PPC32::SLW, 2, DestReg).addReg(SrcReg).addReg(ShiftAmountReg);
	  } else {
		if (isSigned) {
			// FIXME: Unimplmented
			// Page C-3 of the PowerPC 32bit Programming Environments Manual
		} else {
			BuildMI(*MBB, IP, PPC32::SUBFIC, 2, TmpReg1).addReg(ShiftAmountReg).addImm(32);
			BuildMI(*MBB, IP, PPC32::SRW, 2, TmpReg2).addReg(SrcReg).addReg(ShiftAmountReg);
			BuildMI(*MBB, IP, PPC32::SLW, 2, TmpReg3).addReg(SrcReg+1).addReg(TmpReg1);
			BuildMI(*MBB, IP, PPC32::OR, 2, TmpReg4).addReg(TmpReg2).addReg(TmpReg3);
			BuildMI(*MBB, IP, PPC32::ADDI, 2, TmpReg5).addReg(ShiftAmountReg).addImm(-32);
			BuildMI(*MBB, IP, PPC32::SRW, 2, TmpReg6).addReg(SrcReg+1).addReg(TmpReg5);
			BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(TmpReg4).addReg(TmpReg6);
			BuildMI(*MBB, IP, PPC32::SRW, 2, DestReg+1).addReg(SrcReg+1).addReg(ShiftAmountReg);
		}
	  }
    }
    return;
  }

  if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(ShiftAmount)) {
    // The shift amount is constant, guaranteed to be a ubyte. Get its value.
    assert(CUI->getType() == Type::UByteTy && "Shift amount not a ubyte?");
    unsigned Amount = CUI->getValue();

	if (isLeftShift) {
		BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg).addImm(Amount).addImm(0).addImm(31-Amount);
	} else {
		if (isSigned) {
			BuildMI(*MBB, IP, PPC32::SRAWI, 2, DestReg).addReg(SrcReg).addImm(Amount);
		} else {
			BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg).addImm(32-Amount).addImm(Amount).addImm(31);
		}
	}
  } else {                  // The shift amount is non-constant.
    unsigned ShiftAmountReg = getReg (ShiftAmount, MBB, IP);

	if (isLeftShift) {
		BuildMI(*MBB, IP, PPC32::SLW, 2, DestReg).addReg(SrcReg).addReg(ShiftAmountReg);
	} else {
		BuildMI(*MBB, IP, isSigned ? PPC32::SRAW : PPC32::SRW, 2, DestReg).addReg(SrcReg).addReg(ShiftAmountReg);
	}
  }
}


/// visitLoadInst - Implement LLVM load instructions
///
void ISel::visitLoadInst(LoadInst &I) {
  static const unsigned Opcodes[] = { PPC32::LBZ, PPC32::LHZ, PPC32::LWZ, PPC32::LFS };
  unsigned Class = getClassB(I.getType());
  unsigned Opcode = Opcodes[Class];
  if (I.getType() == Type::DoubleTy) Opcode = PPC32::LFD;

  unsigned DestReg = getReg(I);

  if (AllocaInst *AI = dyn_castFixedAlloca(I.getOperand(0))) {
	unsigned FI = getFixedSizedAllocaFI(AI);
    if (Class == cLong) {
		addFrameReference(BuildMI(BB, PPC32::LWZ, 2, DestReg), FI);
		addFrameReference(BuildMI(BB, PPC32::LWZ, 2, DestReg+1), FI, 4);
    } else {
		addFrameReference(BuildMI(BB, Opcode, 2, DestReg), FI);
	}
  } else {
	unsigned SrcAddrReg = getReg(I.getOperand(0));
    
    if (Class == cLong) {
      BuildMI(BB, PPC32::LWZ, 2, DestReg).addImm(0).addReg(SrcAddrReg);
      BuildMI(BB, PPC32::LWZ, 2, DestReg+1).addImm(4).addReg(SrcAddrReg);
    } else {
      BuildMI(BB, Opcode, 2, DestReg).addImm(0).addReg(SrcAddrReg);
    }
  }
}

/// visitStoreInst - Implement LLVM store instructions
///
void ISel::visitStoreInst(StoreInst &I) {
  unsigned ValReg      = getReg(I.getOperand(0));
  unsigned AddressReg  = getReg(I.getOperand(1));
 
  const Type *ValTy = I.getOperand(0)->getType();
  unsigned Class = getClassB(ValTy);

  if (Class == cLong) {
	BuildMI(BB, PPC32::STW, 3).addReg(ValReg).addImm(0).addReg(AddressReg);
   	BuildMI(BB, PPC32::STW, 3).addReg(ValReg+1).addImm(4).addReg(AddressReg);
    return;
  }

  static const unsigned Opcodes[] = {
    PPC32::STB, PPC32::STH, PPC32::STW, PPC32::STFS
  };
  unsigned Opcode = Opcodes[Class];
  if (ValTy == Type::DoubleTy) Opcode = PPC32::STFD;
  BuildMI(BB, Opcode, 3).addReg(ValReg).addImm(0).addReg(AddressReg);
}


/// visitCastInst - Here we have various kinds of copying with or without sign
/// extension going on.
///
void ISel::visitCastInst(CastInst &CI) {
  Value *Op = CI.getOperand(0);

  unsigned SrcClass = getClassB(Op->getType());
  unsigned DestClass = getClassB(CI.getType());
  // Noop casts are not emitted: getReg will return the source operand as the
  // register to use for any uses of the noop cast.
  if (DestClass == SrcClass)
    return;

  // If this is a cast from a 32-bit integer to a Long type, and the only uses
  // of the case are GEP instructions, then the cast does not need to be
  // generated explicitly, it will be folded into the GEP.
  if (DestClass == cLong && SrcClass == cInt) {
    bool AllUsesAreGEPs = true;
    for (Value::use_iterator I = CI.use_begin(), E = CI.use_end(); I != E; ++I)
      if (!isa<GetElementPtrInst>(*I)) {
        AllUsesAreGEPs = false;
        break;
      }        

    // No need to codegen this cast if all users are getelementptr instrs...
    if (AllUsesAreGEPs) return;
  }

  unsigned DestReg = getReg(CI);
  MachineBasicBlock::iterator MI = BB->end();
  emitCastOperation(BB, MI, Op, CI.getType(), DestReg);
}

/// emitCastOperation - Common code shared between visitCastInst and constant
/// expression cast support.
///
void ISel::emitCastOperation(MachineBasicBlock *BB,
                             MachineBasicBlock::iterator IP,
                             Value *Src, const Type *DestTy,
                             unsigned DestReg) {
  const Type *SrcTy = Src->getType();
  unsigned SrcClass = getClassB(SrcTy);
  unsigned DestClass = getClassB(DestTy);
  unsigned SrcReg = getReg(Src, BB, IP);

  // Implement casts to bool by using compare on the operand followed by set if
  // not zero on the result.
  if (DestTy == Type::BoolTy) {
    switch (SrcClass) {
    case cByte:
	case cShort:
    case cInt: {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
	  BuildMI(*BB, IP, PPC32::ADDIC, 2, TmpReg).addReg(SrcReg).addImm(-1);
	  BuildMI(*BB, IP, PPC32::SUBFE, 2, DestReg).addReg(TmpReg).addReg(SrcReg);
      break;
    }
    case cLong: {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      unsigned SrcReg2 = makeAnotherReg(Type::IntTy);
      BuildMI(*BB, IP, PPC32::OR, 2, SrcReg2).addReg(SrcReg).addReg(SrcReg+1);
	  BuildMI(*BB, IP, PPC32::ADDIC, 2, TmpReg).addReg(SrcReg2).addImm(-1);
	  BuildMI(*BB, IP, PPC32::SUBFE, 2, DestReg).addReg(TmpReg).addReg(SrcReg2);
      break;
    }
    case cFP:
      // FIXME
	  // Load -0.0
	  // Compare
	  // move to CR1
	  // Negate -0.0
	  // Compare
	  // CROR
	  // MFCR
	  // Left-align
	  // SRA ?
      break;
    }
    return;
  }

  // Implement casts between values of the same type class (as determined by
  // getClass) by using a register-to-register move.
  if (SrcClass == DestClass) {
	if (SrcClass <= cInt) {
	  BuildMI(*BB, IP, PPC32::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
	} else if (SrcClass == cFP && SrcTy == DestTy) {
      BuildMI(*BB, IP, PPC32::FMR, 1, DestReg).addReg(SrcReg);
    } else if (SrcClass == cFP) {
      if (SrcTy == Type::FloatTy) {  // float -> double
        assert(DestTy == Type::DoubleTy && "Unknown cFP member!");
        BuildMI(*BB, IP, PPC32::FMR, 1, DestReg).addReg(SrcReg);
      } else {                       // double -> float
        assert(SrcTy == Type::DoubleTy && DestTy == Type::FloatTy &&
               "Unknown cFP member!");
		BuildMI(*BB, IP, PPC32::FRSP, 1, DestReg).addReg(SrcReg);
      }
    } else if (SrcClass == cLong) {
	  BuildMI(*BB, IP, PPC32::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
	  BuildMI(*BB, IP, PPC32::OR, 2, DestReg+1).addReg(SrcReg+1).addReg(SrcReg+1);
    } else {
      assert(0 && "Cannot handle this type of cast instruction!");
      abort();
    }
    return;
  }

  // Handle cast of SMALLER int to LARGER int using a move with sign extension
  // or zero extension, depending on whether the source type was signed.
  if (SrcClass <= cInt && (DestClass <= cInt || DestClass == cLong) &&
      SrcClass < DestClass) {
    bool isLong = DestClass == cLong;
    if (isLong) DestClass = cInt;

    bool isUnsigned = SrcTy->isUnsigned() || SrcTy == Type::BoolTy;
    if (SrcClass < cInt) {
      if (isUnsigned) {
    	unsigned shift = (SrcClass == cByte) ? 24 : 16;
    	BuildMI(*BB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg).addZImm(0).addImm(shift).addImm(31);
      } else {
        BuildMI(*BB, IP, (SrcClass == cByte) ? PPC32::EXTSB : PPC32::EXTSH, 1, DestReg).addReg(SrcReg);
	  }
	} else {
	  BuildMI(*BB, IP, PPC32::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
	}

    if (isLong) {  // Handle upper 32 bits as appropriate...
      if (isUnsigned)     // Zero out top bits...
        BuildMI(*BB, IP, PPC32::ADDI, 2, DestReg+1).addReg(PPC32::R0).addImm(0);
      else                // Sign extend bottom half...
        BuildMI(*BB, IP, PPC32::SRAWI, 2, DestReg+1).addReg(DestReg).addImm(31);
    }
    return;
  }

  // Special case long -> int ...
  if (SrcClass == cLong && DestClass == cInt) {
    BuildMI(*BB, IP, PPC32::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
    return;
  }
  
  // Handle cast of LARGER int to SMALLER int with a clear or sign extend
  if ((SrcClass <= cInt || SrcClass == cLong) && DestClass <= cInt
      && SrcClass > DestClass) {
    bool isUnsigned = SrcTy->isUnsigned() || SrcTy == Type::BoolTy;
	if (isUnsigned) {
    	unsigned shift = (SrcClass == cByte) ? 24 : 16;
    	BuildMI(*BB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg).addZImm(0).addImm(shift).addImm(31);
	} else {
        BuildMI(*BB, IP, (SrcClass == cByte) ? PPC32::EXTSB : PPC32::EXTSH, 1, DestReg).addReg(SrcReg);
	}
    return;
  }

  // Handle casts from integer to floating point now...
  if (DestClass == cFP) {

	// Emit a library call for long to float conversion
	if (SrcClass == cLong) {
		std::vector<ValueRecord> Args;
		Args.push_back(ValueRecord(SrcReg, SrcTy));
		MachineInstr *TheCall = BuildMI(PPC32::CALLpcrel, 1).addExternalSymbol("__floatdidf", true);
		doCall(ValueRecord(DestReg, DestTy), TheCall, Args);
		return;
	}

    unsigned TmpReg = makeAnotherReg(Type::IntTy);
    switch (SrcTy->getPrimitiveID()) {
    case Type::BoolTyID:
    case Type::SByteTyID:
      BuildMI(*BB, IP, PPC32::EXTSB, 1, TmpReg).addReg(SrcReg);
      break;
    case Type::UByteTyID:
	  BuildMI(*BB, IP, PPC32::RLWINM, 4, TmpReg).addReg(SrcReg).addZImm(0).addImm(24).addImm(31);
      break;
    case Type::ShortTyID:
      BuildMI(*BB, IP, PPC32::EXTSB, 1, TmpReg).addReg(SrcReg);
      break;
    case Type::UShortTyID:
	  BuildMI(*BB, IP, PPC32::RLWINM, 4, TmpReg).addReg(SrcReg).addZImm(0).addImm(16).addImm(31);
      break;
	case Type::IntTyID:
	  BuildMI(*BB, IP, PPC32::OR, 2, TmpReg).addReg(SrcReg).addReg(SrcReg);
	  break;
	case Type::UIntTyID:
	  BuildMI(*BB, IP, PPC32::OR, 2, TmpReg).addReg(SrcReg).addReg(SrcReg);
	  break;
    default:  // No promotion needed...
      break;
    }
    
    SrcReg = TmpReg;
	
    // Spill the integer to memory and reload it from there.
	// Also spill room for a special conversion constant
	int ConstantFrameIndex = 
      F->getFrameInfo()->CreateStackObject(Type::DoubleTy, TM.getTargetData());
    int ValueFrameIdx =
      F->getFrameInfo()->CreateStackObject(Type::DoubleTy, TM.getTargetData());

	unsigned constantHi = makeAnotherReg(Type::IntTy);
	unsigned constantLo = makeAnotherReg(Type::IntTy);
	unsigned ConstF = makeAnotherReg(Type::DoubleTy);
	unsigned TempF = makeAnotherReg(Type::DoubleTy);
	
    if (!SrcTy->isSigned()) {
		BuildMI(*BB, IP, PPC32::ADDIS, 2, constantHi).addReg(PPC32::R0).addImm(0x4330);
		BuildMI(*BB, IP, PPC32::ADDI, 2, constantLo).addReg(PPC32::R0).addImm(0);
		addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantHi), ConstantFrameIndex);
		addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantLo), ConstantFrameIndex, 4);
		addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantHi), ValueFrameIdx);
		addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(SrcReg), ValueFrameIdx, 4);
		addFrameReference(BuildMI(*BB, IP, PPC32::LFD, 2, ConstF), ConstantFrameIndex);
		addFrameReference(BuildMI(*BB, IP, PPC32::LFD, 2, TempF), ValueFrameIdx);
		BuildMI(*BB, IP, PPC32::FSUB, 2, DestReg).addReg(TempF).addReg(ConstF);
	} else {
		unsigned TempLo = makeAnotherReg(Type::IntTy);
		BuildMI(*BB, IP, PPC32::ADDIS, 2, constantHi).addReg(PPC32::R0).addImm(0x4330);
		BuildMI(*BB, IP, PPC32::ADDIS, 2, constantLo).addReg(PPC32::R0).addImm(0x8000);
		addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantHi), ConstantFrameIndex);
		addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantLo), ConstantFrameIndex, 4);
		addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantHi), ValueFrameIdx);
		BuildMI(*BB, IP, PPC32::XORIS, 2, TempLo).addReg(SrcReg).addImm(0x8000);
		addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(TempLo), ValueFrameIdx, 4);
		addFrameReference(BuildMI(*BB, IP, PPC32::LFD, 2, ConstF), ConstantFrameIndex);
		addFrameReference(BuildMI(*BB, IP, PPC32::LFD, 2, TempF), ValueFrameIdx);
		BuildMI(*BB, IP, PPC32::FSUB, 2, DestReg).addReg(TempF).addReg(ConstF);
	}
    return;
  }

  // Handle casts from floating point to integer now...
  if (SrcClass == cFP) {

	// emit library call
	if (DestClass == cLong) {
		std::vector<ValueRecord> Args;
		Args.push_back(ValueRecord(SrcReg, SrcTy));
		MachineInstr *TheCall = BuildMI(PPC32::CALLpcrel, 1).addExternalSymbol("__fixdfdi", true);
		doCall(ValueRecord(DestReg, DestTy), TheCall, Args);
		return;
	}

    int ValueFrameIdx =
      F->getFrameInfo()->CreateStackObject(Type::DoubleTy, TM.getTargetData());

	// load into 32 bit value, and then truncate as necessary
	// FIXME: This is wrong for unsigned dest types
	//if (DestTy->isSigned()) {
		unsigned TempReg = makeAnotherReg(Type::DoubleTy);
		BuildMI(*BB, IP, PPC32::FCTIWZ, 1, TempReg).addReg(SrcReg);
		addFrameReference(BuildMI(*BB, IP, PPC32::STFD, 3).addReg(TempReg), ValueFrameIdx);
		addFrameReference(BuildMI(*BB, IP, PPC32::LWZ, 2, DestReg), ValueFrameIdx+4);
	//} else {
	//}
	
	// FIXME: Truncate return value
    return;
  }

  // Anything we haven't handled already, we can't (yet) handle at all.
  assert(0 && "Unhandled cast instruction!");
  abort();
}

/// visitVANextInst - Implement the va_next instruction...
///
void ISel::visitVANextInst(VANextInst &I) {
  unsigned VAList = getReg(I.getOperand(0));
  unsigned DestReg = getReg(I);

  unsigned Size;
  switch (I.getArgType()->getPrimitiveID()) {
  default:
    std::cerr << I;
    assert(0 && "Error: bad type for va_next instruction!");
    return;
  case Type::PointerTyID:
  case Type::UIntTyID:
  case Type::IntTyID:
    Size = 4;
    break;
  case Type::ULongTyID:
  case Type::LongTyID:
  case Type::DoubleTyID:
    Size = 8;
    break;
  }

  // Increment the VAList pointer...
  BuildMI(BB, PPC32::ADDI, 2, DestReg).addReg(VAList).addImm(Size);
}

void ISel::visitVAArgInst(VAArgInst &I) {
  unsigned VAList = getReg(I.getOperand(0));
  unsigned DestReg = getReg(I);

  switch (I.getType()->getPrimitiveID()) {
  default:
    std::cerr << I;
    assert(0 && "Error: bad type for va_next instruction!");
    return;
  case Type::PointerTyID:
  case Type::UIntTyID:
  case Type::IntTyID:
    BuildMI(BB, PPC32::LWZ, 2, DestReg).addImm(0).addReg(VAList);
    break;
  case Type::ULongTyID:
  case Type::LongTyID:
    BuildMI(BB, PPC32::LWZ, 2, DestReg).addImm(0).addReg(VAList);
    BuildMI(BB, PPC32::LWZ, 2, DestReg+1).addImm(4).addReg(VAList);
    break;
  case Type::DoubleTyID:
    BuildMI(BB, PPC32::LFD, 2, DestReg).addImm(0).addReg(VAList);
    break;
  }
}

/// visitGetElementPtrInst - instruction-select GEP instructions
///
void ISel::visitGetElementPtrInst(GetElementPtrInst &I) {
  unsigned outputReg = getReg(I);
  emitGEPOperation(BB, BB->end(), I.getOperand(0),I.op_begin()+1, I.op_end(), outputReg);
}

void ISel::emitGEPOperation(MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator IP,
                            Value *Src, User::op_iterator IdxBegin,
                            User::op_iterator IdxEnd, unsigned TargetReg) {
  const TargetData &TD = TM.getTargetData();
  if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Src))
    Src = CPR->getValue();

  std::vector<Value*> GEPOps;
  GEPOps.resize(IdxEnd-IdxBegin+1);
  GEPOps[0] = Src;
  std::copy(IdxBegin, IdxEnd, GEPOps.begin()+1);
  
  std::vector<const Type*> GEPTypes;
  GEPTypes.assign(gep_type_begin(Src->getType(), IdxBegin, IdxEnd),
                  gep_type_end(Src->getType(), IdxBegin, IdxEnd));

  // Keep emitting instructions until we consume the entire GEP instruction.
  while (!GEPOps.empty()) {
      // It's an array or pointer access: [ArraySize x ElementType].
      const SequentialType *SqTy = cast<SequentialType>(GEPTypes.back());
      Value *idx = GEPOps.back();
      GEPOps.pop_back();        // Consume a GEP operand
      GEPTypes.pop_back();

      // Many GEP instructions use a [cast (int/uint) to LongTy] as their
      // operand on X86.  Handle this case directly now...
      if (CastInst *CI = dyn_cast<CastInst>(idx))
        if (CI->getOperand(0)->getType() == Type::IntTy ||
            CI->getOperand(0)->getType() == Type::UIntTy)
          idx = CI->getOperand(0);

      // We want to add BaseReg to(idxReg * sizeof ElementType). First, we
      // must find the size of the pointed-to type (Not coincidentally, the next
      // type is the type of the elements in the array).
      const Type *ElTy = SqTy->getElementType();
      unsigned elementSize = TD.getTypeSize(ElTy);

     if (elementSize == 1) {
        // If the element size is 1, we don't have to multiply, just add
        unsigned idxReg = getReg(idx, MBB, IP);
        unsigned Reg = makeAnotherReg(Type::UIntTy);
        BuildMI(*MBB, IP, PPC32::ADD, 2,TargetReg).addReg(Reg).addReg(idxReg);
        --IP;            // Insert the next instruction before this one.
        TargetReg = Reg; // Codegen the rest of the GEP into this
      } else {
        unsigned idxReg = getReg(idx, MBB, IP);
        unsigned OffsetReg = makeAnotherReg(Type::UIntTy);

        // Make sure we can back the iterator up to point to the first
        // instruction emitted.
        MachineBasicBlock::iterator BeforeIt = IP;
        if (IP == MBB->begin())
          BeforeIt = MBB->end();
        else
          --BeforeIt;
        doMultiplyConst(MBB, IP, OffsetReg, Type::IntTy, idxReg, elementSize);

        // Emit an ADD to add OffsetReg to the basePtr.
        unsigned Reg = makeAnotherReg(Type::UIntTy);
        BuildMI(*MBB, IP, PPC32::ADD, 2, TargetReg).addReg(Reg).addReg(OffsetReg);

        // Step to the first instruction of the multiply.
        if (BeforeIt == MBB->end())
          IP = MBB->begin();
        else
          IP = ++BeforeIt;

        TargetReg = Reg; // Codegen the rest of the GEP into this
      }
    }
}

/// visitAllocaInst - If this is a fixed size alloca, allocate space from the
/// frame manager, otherwise do it the hard way.
///
void ISel::visitAllocaInst(AllocaInst &I) {
  // If this is a fixed size alloca in the entry block for the function, we
  // statically stack allocate the space, so we don't need to do anything here.
  //
  if (dyn_castFixedAlloca(&I)) return;
  
  // Find the data size of the alloca inst's getAllocatedType.
  const Type *Ty = I.getAllocatedType();
  unsigned TySize = TM.getTargetData().getTypeSize(Ty);

  // Create a register to hold the temporary result of multiplying the type size
  // constant by the variable amount.
  unsigned TotalSizeReg = makeAnotherReg(Type::UIntTy);
  unsigned SrcReg1 = getReg(I.getArraySize());
  
  // TotalSizeReg = mul <numelements>, <TypeSize>
  MachineBasicBlock::iterator MBBI = BB->end();
  doMultiplyConst(BB, MBBI, TotalSizeReg, Type::UIntTy, SrcReg1, TySize);

  // AddedSize = add <TotalSizeReg>, 15
  unsigned AddedSizeReg = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, PPC32::ADD, 2, AddedSizeReg).addReg(TotalSizeReg).addImm(15);

  // AlignedSize = and <AddedSize>, ~15
  unsigned AlignedSize = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, PPC32::RLWNM, 4, AlignedSize).addReg(AddedSizeReg).addImm(0).addImm(0).addImm(27);
  
  // Subtract size from stack pointer, thereby allocating some space.
  BuildMI(BB, PPC32::SUB, 2, PPC32::R1).addReg(PPC32::R1).addReg(AlignedSize);

  // Put a pointer to the space into the result register, by copying
  // the stack pointer.
  BuildMI(BB, PPC32::OR, 2, getReg(I)).addReg(PPC32::R1).addReg(PPC32::R1);

  // Inform the Frame Information that we have just allocated a variable-sized
  // object.
  F->getFrameInfo()->CreateVariableSizedObject();
}

/// visitMallocInst - Malloc instructions are code generated into direct calls
/// to the library malloc.
///
void ISel::visitMallocInst(MallocInst &I) {
  unsigned AllocSize = TM.getTargetData().getTypeSize(I.getAllocatedType());
  unsigned Arg;

  if (ConstantUInt *C = dyn_cast<ConstantUInt>(I.getOperand(0))) {
    Arg = getReg(ConstantUInt::get(Type::UIntTy, C->getValue() * AllocSize));
  } else {
    Arg = makeAnotherReg(Type::UIntTy);
    unsigned Op0Reg = getReg(I.getOperand(0));
    MachineBasicBlock::iterator MBBI = BB->end();
    doMultiplyConst(BB, MBBI, Arg, Type::UIntTy, Op0Reg, AllocSize);
  }

  std::vector<ValueRecord> Args;
  Args.push_back(ValueRecord(Arg, Type::UIntTy));
  MachineInstr *TheCall = BuildMI(PPC32::CALLpcrel, 1).addExternalSymbol("malloc", true);
  doCall(ValueRecord(getReg(I), I.getType()), TheCall, Args);
}


/// visitFreeInst - Free instructions are code gen'd to call the free libc
/// function.
///
void ISel::visitFreeInst(FreeInst &I) {
  std::vector<ValueRecord> Args;
  Args.push_back(ValueRecord(I.getOperand(0)));
  MachineInstr *TheCall = BuildMI(PPC32::CALLpcrel, 1).addExternalSymbol("free", true);
  doCall(ValueRecord(0, Type::VoidTy), TheCall, Args);
}
   
/// createPPC32SimpleInstructionSelector - This pass converts an LLVM function
/// into a machine code representation is a very simple peep-hole fashion.  The
/// generated code sucks but the implementation is nice and simple.
///
FunctionPass *llvm::createPPCSimpleInstructionSelector(TargetMachine &TM) {
  return new ISel(TM);
}
