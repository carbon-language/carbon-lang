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
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/FunctionFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Target/MRegisterInfo.h"
#include <map>

/// BMI - A special BuildMI variant that takes an iterator to insert the
/// instruction at as well as a basic block.
/// this is the version for when you have a destination register in mind.
inline static MachineInstrBuilder BMI(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator &I,
                                      MachineOpCode Opcode,
                                      unsigned NumOperands,
                                      unsigned DestReg) {
  assert(I >= MBB->begin() && I <= MBB->end() && "Bad iterator!");
  MachineInstr *MI = new MachineInstr(Opcode, NumOperands+1, true, true);
  I = MBB->insert(I, MI)+1;
  return MachineInstrBuilder(MI).addReg(DestReg, MOTy::Def);
}

/// BMI - A special BuildMI variant that takes an iterator to insert the
/// instruction at as well as a basic block.
inline static MachineInstrBuilder BMI(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator &I,
                                      MachineOpCode Opcode,
                                      unsigned NumOperands) {
  assert(I > MBB->begin() && I <= MBB->end() && "Bad iterator!");
  MachineInstr *MI = new MachineInstr(Opcode, NumOperands, true, true);
  I = MBB->insert(I, MI)+1;
  return MachineInstrBuilder(MI);
}


namespace {
  struct ISel : public FunctionPass, InstVisitor<ISel> {
    TargetMachine &TM;
    MachineFunction *F;                    // The function we are compiling into
    MachineBasicBlock *BB;                 // The current MBB we are compiling

    unsigned CurReg;
    std::map<Value*, unsigned> RegMap;  // Mapping between Val's and SSA Regs

    // MBBMap - Mapping between LLVM BB -> Machine BB
    std::map<const BasicBlock*, MachineBasicBlock*> MBBMap;

    ISel(TargetMachine &tm)
      : TM(tm), F(0), BB(0), CurReg(MRegisterInfo::FirstVirtualRegister) {}

    /// runOnFunction - Top level implementation of instruction selection for
    /// the entire function.
    ///
    bool runOnFunction(Function &Fn) {
      F = &MachineFunction::construct(&Fn, TM);

      // Create all of the machine basic blocks for the function...
      for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
        F->getBasicBlockList().push_back(MBBMap[I] = new MachineBasicBlock(I));

      BB = &F->front();
      LoadArgumentsToVirtualRegs(Fn);

      // Instruction select everything except PHI nodes
      visit(Fn);

      // Select the PHI nodes
      SelectPHINodes();

      RegMap.clear();
      MBBMap.clear();
      CurReg = MRegisterInfo::FirstVirtualRegister;
      F = 0;
      return false;  // We never modify the LLVM itself.
    }

    virtual const char *getPassName() const {
      return "X86 Simple Instruction Selection";
    }

    /// visitBasicBlock - This method is called when we are visiting a new basic
    /// block.  This simply creates a new MachineBasicBlock to emit code into
    /// and adds it to the current MachineFunction.  Subsequent visit* for
    /// instructions will be invoked for all instructions in the basic block.
    ///
    void visitBasicBlock(BasicBlock &LLVM_BB) {
      BB = MBBMap[&LLVM_BB];
    }

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
    // fixed X86 code for each instruction.
    //

    // Control flow operators
    void visitReturnInst(ReturnInst &RI);
    void visitBranchInst(BranchInst &BI);
    void visitCallInst(CallInst &I);

    // Arithmetic operators
    void visitSimpleBinary(BinaryOperator &B, unsigned OpcodeClass);
    void visitAdd(BinaryOperator &B) { visitSimpleBinary(B, 0); }
    void visitSub(BinaryOperator &B) { visitSimpleBinary(B, 1); }
    void doMultiply(MachineBasicBlock *MBB, MachineBasicBlock::iterator &MBBI,
                    unsigned destReg, const Type *resultType,
		    unsigned op0Reg, unsigned op1Reg);
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
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitAllocaInst(AllocaInst &I);

    // We assume that by this point, malloc instructions have been
    // lowered to calls, and dlsym will magically find malloc for us.
    void visitMallocInst(MallocInst &I) { visitInstruction (I); }
    void visitFreeInst(FreeInst &I) { visitInstruction(I); }
    
    // Other operators
    void visitShiftInst(ShiftInst &I);
    void visitPHINode(PHINode &I) {}      // PHI nodes handled by second pass
    void visitCastInst(CastInst &I);

    void visitInstruction(Instruction &I) {
      std::cerr << "Cannot instruction select: " << I;
      abort();
    }

    /// promote32 - Make a value 32-bits wide, and put it somewhere.
    void promote32 (const unsigned targetReg, Value *v);
    
    // emitGEPOperation - Common code shared between visitGetElementPtrInst and
    // constant expression GEP support.
    //
    void emitGEPOperation(MachineBasicBlock *BB, MachineBasicBlock::iterator&IP,
                          Value *Src, User::op_iterator IdxBegin,
                          User::op_iterator IdxEnd, unsigned TargetReg);

    /// copyConstantToRegister - Output the instructions required to put the
    /// specified constant into the specified register.
    ///
    void copyConstantToRegister(MachineBasicBlock *MBB,
                                MachineBasicBlock::iterator &MBBI,
                                Constant *C, unsigned Reg);

    /// makeAnotherReg - This method returns the next register number
    /// we haven't yet used.
    unsigned makeAnotherReg(const Type *Ty) {
      // Add the mapping of regnumber => reg class to MachineFunction
      const TargetRegisterClass *RC =
	TM.getRegisterInfo()->getRegClassForType(Ty);
      F->getSSARegMap()->addRegMap(CurReg, RC);
      return CurReg++;
    }

    /// getReg - This method turns an LLVM value into a register number.  This
    /// is guaranteed to produce the same register number for a particular value
    /// every time it is queried.
    ///
    unsigned getReg(Value &V) { return getReg(&V); }  // Allow references
    unsigned getReg(Value *V) {
      // Just append to the end of the current bb.
      MachineBasicBlock::iterator It = BB->end();
      return getReg(V, BB, It);
    }
    unsigned getReg(Value *V, MachineBasicBlock *MBB,
                    MachineBasicBlock::iterator &IPt) {
      unsigned &Reg = RegMap[V];
      if (Reg == 0) {
        Reg = makeAnotherReg(V->getType());
        RegMap[V] = Reg;
      }

      // If this operand is a constant, emit the code to copy the constant into
      // the register here...
      //
      if (Constant *C = dyn_cast<Constant>(V)) {
        copyConstantToRegister(MBB, IPt, C, Reg);
        RegMap.erase(V);  // Assign a new name to this constant if ref'd again
      } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
        // Move the address of the global into the register
        BMI(MBB, IPt, X86::MOVir32, 1, Reg).addReg(GV);
        RegMap.erase(V);  // Assign a new name to this address if ref'd again
      }

      return Reg;
    }
  };
}

/// TypeClass - Used by the X86 backend to group LLVM types by their basic X86
/// Representation.
///
enum TypeClass {
  cByte, cShort, cInt, cFP, cLong
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

  case Type::FloatTyID:
  case Type::DoubleTyID:  return cFP;        // Floating Point is #3
  case Type::LongTyID:
  case Type::ULongTyID:   //return cLong;      // Longs are class #3
    return cInt;          // FIXME: LONGS ARE TREATED AS INTS!
  default:
    assert(0 && "Invalid type to getClass!");
    return cByte;  // not reached
  }
}

// getClassB - Just like getClass, but treat boolean values as bytes.
static inline TypeClass getClassB(const Type *Ty) {
  if (Ty == Type::BoolTy) return cByte;
  return getClass(Ty);
}


/// copyConstantToRegister - Output the instructions required to put the
/// specified constant into the specified register.
///
void ISel::copyConstantToRegister(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator &IP,
                                  Constant *C, unsigned R) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    if (CE->getOpcode() == Instruction::GetElementPtr) {
      emitGEPOperation(MBB, IP, CE->getOperand(0),
                       CE->op_begin()+1, CE->op_end(), R);
      return;
    }

    std::cerr << "Offending expr: " << C << "\n";
    assert(0 && "Constant expressions not yet handled!\n");
  }

  if (C->getType()->isIntegral()) {
    unsigned Class = getClassB(C->getType());
    assert(Class <= cInt && "Type not handled yet!");

    static const unsigned IntegralOpcodeTab[] = {
      X86::MOVir8, X86::MOVir16, X86::MOVir32
    };

    if (C->getType() == Type::BoolTy) {
      BMI(MBB, IP, X86::MOVir8, 1, R).addZImm(C == ConstantBool::True);
    } else if (C->getType()->isSigned()) {
      ConstantSInt *CSI = cast<ConstantSInt>(C);
      BMI(MBB, IP, IntegralOpcodeTab[Class], 1, R).addSImm(CSI->getValue());
    } else {
      ConstantUInt *CUI = cast<ConstantUInt>(C);
      BMI(MBB, IP, IntegralOpcodeTab[Class], 1, R).addZImm(CUI->getValue());
    }
  } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
    double Value = CFP->getValue();
    if (Value == +0.0)
      BMI(MBB, IP, X86::FLD0, 0, R);
    else if (Value == +1.0)
      BMI(MBB, IP, X86::FLD1, 0, R);
    else {
      std::cerr << "Cannot load constant '" << Value << "'!\n";
      assert(0);
    }

  } else if (isa<ConstantPointerNull>(C)) {
    // Copy zero (null pointer) to the register.
    BMI(MBB, IP, X86::MOVir32, 1, R).addZImm(0);
  } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C)) {
    unsigned SrcReg = getReg(CPR->getValue(), MBB, IP);
    BMI(MBB, IP, X86::MOVrr32, 1, R).addReg(SrcReg);
  } else {
    std::cerr << "Offending constant: " << C << "\n";
    assert(0 && "Type not handled yet!");
  }
}

/// LoadArgumentsToVirtualRegs - Load all of the arguments to this function from
/// the stack into virtual registers.
///
void ISel::LoadArgumentsToVirtualRegs(Function &Fn) {
  // Emit instructions to load the arguments...  On entry to a function on the
  // X86, the stack frame looks like this:
  //
  // [ESP] -- return address
  // [ESP + 4] -- first argument (leftmost lexically) if four bytes in size
  // [ESP + 8] -- second argument, if four bytes in size
  //    ... 
  //
  unsigned ArgOffset = 0;
  FunctionFrameInfo *FFI = F->getFrameInfo();

  for (Function::aiterator I = Fn.abegin(), E = Fn.aend(); I != E; ++I) {
    unsigned Reg = getReg(*I);
    
    ArgOffset += 4;  // Each argument takes at least 4 bytes on the stack...
    int FI;          // Frame object index

    switch (getClassB(I->getType())) {
    case cByte:
      FI = FFI->CreateFixedObject(1, ArgOffset);
      addFrameReference(BuildMI(BB, X86::MOVmr8, 4, Reg), FI);
      break;
    case cShort:
      FI = FFI->CreateFixedObject(2, ArgOffset);
      addFrameReference(BuildMI(BB, X86::MOVmr16, 4, Reg), FI);
      break;
    case cInt:
      FI = FFI->CreateFixedObject(4, ArgOffset);
      addFrameReference(BuildMI(BB, X86::MOVmr32, 4, Reg), FI);
      break;
    case cFP:
      unsigned Opcode;
      if (I->getType() == Type::FloatTy) {
	Opcode = X86::FLDr32;
	FI = FFI->CreateFixedObject(4, ArgOffset);
      } else {
	Opcode = X86::FLDr64;
	ArgOffset += 4;   // doubles require 4 additional bytes
	FI = FFI->CreateFixedObject(8, ArgOffset);
      }
      addFrameReference(BuildMI(BB, Opcode, 4, Reg), FI);
      break;
    default:
      assert(0 && "Unhandled argument type!");
    }
  }
}


/// SelectPHINodes - Insert machine code to generate phis.  This is tricky
/// because we have to generate our sources into the source basic blocks, not
/// the current one.
///
void ISel::SelectPHINodes() {
  const Function &LF = *F->getFunction();  // The LLVM function...
  for (Function::const_iterator I = LF.begin(), E = LF.end(); I != E; ++I) {
    const BasicBlock *BB = I;
    MachineBasicBlock *MBB = MBBMap[I];

    // Loop over all of the PHI nodes in the LLVM basic block...
    unsigned NumPHIs = 0;
    for (BasicBlock::const_iterator I = BB->begin();
         PHINode *PN = (PHINode*)dyn_cast<PHINode>(&*I); ++I) {
      // Create a new machine instr PHI node, and insert it.
      MachineInstr *MI = BuildMI(X86::PHI, PN->getNumOperands(), getReg(*PN));
      MBB->insert(MBB->begin()+NumPHIs++, MI); // Insert it at the top of the BB

      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        MachineBasicBlock *PredMBB = MBBMap[PN->getIncomingBlock(i)];

        // Get the incoming value into a virtual register.  If it is not already
        // available in a virtual register, insert the computation code into
        // PredMBB
        //
	// FIXME: This should insert the code into the BOTTOM of the block, not
	// the top of the block.  This just makes for huge live ranges...
        MachineBasicBlock::iterator PI = PredMBB->begin();
        while ((*PI)->getOpcode() == X86::PHI) ++PI;
        
        MI->addRegOperand(getReg(PN->getIncomingValue(i), PredMBB, PI));
        MI->addMachineBasicBlockOperand(PredMBB);
      }
    }
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

#if 0
    // Push the variables on the stack with fldl opcodes.
    // FIXME: assuming var1, var2 are in memory, if not, spill to
    // stack first
  case cFP:  // Floats
    BuildMI (BB, X86::FLDr32, 1).addReg (reg1);
    BuildMI (BB, X86::FLDr32, 1).addReg (reg2);
    break;
  case cFP (doubles):  // Doubles
    BuildMI (BB, X86::FLDr64, 1).addReg (reg1);
    BuildMI (BB, X86::FLDr64, 1).addReg (reg2);
    break;
#endif
  case cLong:
  default:
    visitInstruction(I);
  }

#if 0
  if (CompTy->isFloatingPoint()) {
    // (Non-trapping) compare and pop twice.
    BuildMI (BB, X86::FUCOMPP, 0);
    // Move fp status word (concodes) to ax.
    BuildMI (BB, X86::FNSTSWr8, 1, X86::AX);
    // Load real concodes from ax.
    BuildMI (BB, X86::SAHF, 1).addReg(X86::AH);
  }
#endif

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

  BuildMI(BB, OpcodeTab[CompTy->isSigned()][OpNum], 0, getReg(I));
}

/// promote32 - Emit instructions to turn a narrow operand into a 32-bit-wide
/// operand, in the specified target register.
void ISel::promote32 (unsigned targetReg, Value *v) {
  unsigned vReg = getReg(v);
  bool isUnsigned = v->getType()->isUnsigned();
  switch (getClass(v->getType())) {
  case cByte:
    // Extend value into target register (8->32)
    if (isUnsigned)
      BuildMI(BB, X86::MOVZXr32r8, 1, targetReg).addReg(vReg);
    else
      BuildMI(BB, X86::MOVSXr32r8, 1, targetReg).addReg(vReg);
    break;
  case cShort:
    // Extend value into target register (16->32)
    if (isUnsigned)
      BuildMI(BB, X86::MOVZXr32r16, 1, targetReg).addReg(vReg);
    else
      BuildMI(BB, X86::MOVSXr32r16, 1, targetReg).addReg(vReg);
    break;
  case cInt:
    // Move value into target register (32->32)
    BuildMI(BB, X86::MOVrr32, 1, targetReg).addReg(vReg);
    break;
  default:
    assert(0 && "Unpromotable operand class in promote32");
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
///   ret long, ulong  : Move value into EAX/EDX and return
///   ret float/double : Top of FP stack
///
void ISel::visitReturnInst (ReturnInst &I) {
  if (I.getNumOperands() == 0) {
    BuildMI(BB, X86::RET, 0); // Just emit a 'ret' instruction
    return;
  }

  Value *RetVal = I.getOperand(0);
  switch (getClass(RetVal->getType())) {
  case cByte:   // integral return values: extend or move into EAX and return
  case cShort:
  case cInt:
    promote32(X86::EAX, RetVal);
    break;
  case cFP:                   // Floats & Doubles: Return in ST(0)
    BuildMI(BB, X86::FpMOV, 1, X86::ST0).addReg(getReg(RetVal));
    break;
  case cLong:
    // ret long: use EAX(least significant 32 bits)/EDX (most
    // significant 32)...
  default:
    visitInstruction (I);
  }
  // Emit a 'ret' instruction
  BuildMI(BB, X86::RET, 0);
}

/// visitBranchInst - Handle conditional and unconditional branches here.  Note
/// that since code layout is frozen at this point, that if we are trying to
/// jump to a block that is the immediate successor of the current block, we can
/// just make a fall-through. (but we don't currently).
///
void ISel::visitBranchInst(BranchInst &BI) {
  if (BI.isConditional()) {
    BasicBlock *ifTrue  = BI.getSuccessor(0);
    BasicBlock *ifFalse = BI.getSuccessor(1);

    // Compare condition with zero, followed by jump-if-equal to ifFalse, and
    // jump-if-nonequal to ifTrue
    unsigned condReg = getReg(BI.getCondition());
    BuildMI(BB, X86::CMPri8, 2).addReg(condReg).addZImm(0);
    BuildMI(BB, X86::JNE, 1).addPCDisp(BI.getSuccessor(0));
    BuildMI(BB, X86::JE, 1).addPCDisp(BI.getSuccessor(1));
  } else { // unconditional branch
    BuildMI(BB, X86::JMP, 1).addPCDisp(BI.getSuccessor(0));
  }
}

/// visitCallInst - Push args on stack and do a procedure call instruction.
void ISel::visitCallInst(CallInst &CI) {
  // Count how many bytes are to be pushed on the stack...
  unsigned NumBytes = 0;

  if (CI.getNumOperands() > 1) {
    for (unsigned i = 1, e = CI.getNumOperands(); i != e; ++i)
      switch (getClass(CI.getOperand(i)->getType())) {
      case cByte: case cShort: case cInt:
	NumBytes += 4;
	break;
      case cLong:
	NumBytes += 8;
	break;
      case cFP:
	NumBytes += CI.getOperand(i)->getType() == Type::FloatTy ? 4 : 8;
	break;
      default: assert(0 && "Unknown class!");
      }

    // Adjust the stack pointer for the new arguments...
    BuildMI(BB, X86::ADJCALLSTACKDOWN, 1).addZImm(NumBytes);

    // Arguments go on the stack in reverse order, as specified by the ABI.
    unsigned ArgOffset = 0;
    for (unsigned i = 1, e = CI.getNumOperands(); i != e; ++i) {
      Value *Arg = CI.getOperand(i);
      switch (getClass(Arg->getType())) {
      case cByte:
      case cShort: {
	// Promote arg to 32 bits wide into a temporary register...
	unsigned R = makeAnotherReg(Type::UIntTy);
	promote32(R, Arg);
	addRegOffset(BuildMI(BB, X86::MOVrm32, 5),
		     X86::ESP, ArgOffset).addReg(R);
	break;
      }
      case cInt:
	addRegOffset(BuildMI(BB, X86::MOVrm32, 5),
		     X86::ESP, ArgOffset).addReg(getReg(Arg));
	break;

      case cFP:
	if (Arg->getType() == Type::FloatTy) {
	  addRegOffset(BuildMI(BB, X86::FSTr32, 5),
		       X86::ESP, ArgOffset).addReg(getReg(Arg));
	} else {
	  assert(Arg->getType() == Type::DoubleTy && "Unknown FP type!");
	  ArgOffset += 4;
	  addRegOffset(BuildMI(BB, X86::FSTr32, 5),
		       X86::ESP, ArgOffset).addReg(getReg(Arg));
	}
	break;

      default:
	// FIXME: long/ulong/float/double args not handled.
	visitInstruction(CI);
	break;
      }
      ArgOffset += 4;
    }
  }

  if (Function *F = CI.getCalledFunction()) {
    // Emit a CALL instruction with PC-relative displacement.
    BuildMI(BB, X86::CALLpcrel32, 1).addPCDisp(F);
  } else {
    unsigned Reg = getReg(CI.getCalledValue());
    BuildMI(BB, X86::CALLr32, 1).addReg(Reg);
  }

  BuildMI(BB, X86::ADJCALLSTACKUP, 1).addZImm(NumBytes);

  // If there is a return value, scavenge the result from the location the call
  // leaves it in...
  //
  if (CI.getType() != Type::VoidTy) {
    unsigned resultTypeClass = getClass(CI.getType());
    switch (resultTypeClass) {
    case cByte:
    case cShort:
    case cInt: {
      // Integral results are in %eax, or the appropriate portion
      // thereof.
      static const unsigned regRegMove[] = {
	X86::MOVrr8, X86::MOVrr16, X86::MOVrr32
      };
      static const unsigned AReg[] = { X86::AL, X86::AX, X86::EAX };
      BuildMI(BB, regRegMove[resultTypeClass], 1, getReg(CI))
	         .addReg(AReg[resultTypeClass]);
      break;
    }
    case cFP:     // Floating-point return values live in %ST(0)
      BuildMI(BB, X86::FpMOV, 1, getReg(CI)).addReg(X86::ST0);
      break;
    default:
      std::cerr << "Cannot get return value for call of type '"
                << *CI.getType() << "'\n";
      visitInstruction(CI);
    }
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
  if (Class > cFP)  // FIXME: Handle longs
    visitInstruction(B);

  static const unsigned OpcodeTab[][4] = {
    // Arithmetic operators
    { X86::ADDrr8, X86::ADDrr16, X86::ADDrr32, X86::FpADD },  // ADD
    { X86::SUBrr8, X86::SUBrr16, X86::SUBrr32, X86::FpSUB },  // SUB

    // Bitwise operators
    { X86::ANDrr8, X86::ANDrr16, X86::ANDrr32, 0 },  // AND
    { X86:: ORrr8, X86:: ORrr16, X86:: ORrr32, 0 },  // OR
    { X86::XORrr8, X86::XORrr16, X86::XORrr32, 0 },  // XOR
  };
  
  unsigned Opcode = OpcodeTab[OperatorClass][Class];
  assert(Opcode && "Floating point arguments to logical inst?");
  unsigned Op0r = getReg(B.getOperand(0));
  unsigned Op1r = getReg(B.getOperand(1));
  BuildMI(BB, Opcode, 2, getReg(B)).addReg(Op0r).addReg(Op1r);
}

/// doMultiply - Emit appropriate instructions to multiply together
/// the registers op0Reg and op1Reg, and put the result in destReg.
/// The type of the result should be given as resultType.
void ISel::doMultiply(MachineBasicBlock *MBB, MachineBasicBlock::iterator &MBBI,
                      unsigned destReg, const Type *resultType,
                      unsigned op0Reg, unsigned op1Reg) {
  unsigned Class = getClass(resultType);
  switch (Class) {
  case cFP:              // Floating point multiply
    BuildMI(BB, X86::FpMUL, 2, destReg).addReg(op0Reg).addReg(op1Reg);
    return;
  default:
  case cLong:
    assert(0 && "doMultiply not implemented for this class yet!");
  case cByte:
  case cShort:
  case cInt:          // Small integerals, handled below...
    break;
  }
 
  static const unsigned Regs[]     ={ X86::AL    , X86::AX     , X86::EAX     };
  static const unsigned MulOpcode[]={ X86::MULrr8, X86::MULrr16, X86::MULrr32 };
  static const unsigned MovOpcode[]={ X86::MOVrr8, X86::MOVrr16, X86::MOVrr32 };
  unsigned Reg     = Regs[Class];

  // Emit a MOV to put the first operand into the appropriately-sized
  // subreg of EAX.
  BMI(MBB, MBBI, MovOpcode[Class], 1, Reg).addReg (op0Reg);
  
  // Emit the appropriate multiply instruction.
  BMI(MBB, MBBI, MulOpcode[Class], 1).addReg (op1Reg);

  // Emit another MOV to put the result into the destination register.
  BMI(MBB, MBBI, MovOpcode[Class], 1, destReg).addReg (Reg);
}

/// visitMul - Multiplies are not simple binary operators because they must deal
/// with the EAX register explicitly.
///
void ISel::visitMul(BinaryOperator &I) {
  unsigned DestReg = getReg(I);
  unsigned Op0Reg  = getReg(I.getOperand(0));
  unsigned Op1Reg  = getReg(I.getOperand(1));
  MachineBasicBlock::iterator MBBI = BB->end();
  doMultiply(BB, MBBI, DestReg, I.getType(), Op0Reg, Op1Reg);
}


/// visitDivRem - Handle division and remainder instructions... these
/// instruction both require the same instructions to be generated, they just
/// select the result from a different register.  Note that both of these
/// instructions work differently for signed and unsigned operands.
///
void ISel::visitDivRem(BinaryOperator &I) {
  unsigned Class     = getClass(I.getType());
  unsigned Op0Reg    = getReg(I.getOperand(0));
  unsigned Op1Reg    = getReg(I.getOperand(1));
  unsigned ResultReg = getReg(I);

  switch (Class) {
  case cFP:              // Floating point multiply
    if (I.getOpcode() == Instruction::Div)
      BuildMI(BB, X86::FpDIV, 2, ResultReg).addReg(Op0Reg).addReg(Op1Reg);
    else
      BuildMI(BB, X86::FpREM, 2, ResultReg).addReg(Op0Reg).addReg(Op1Reg);
    return;
  default:
  case cLong:
    assert(0 && "div/rem not implemented for this class yet!");
  case cByte:
  case cShort:
  case cInt:          // Small integerals, handled below...
    break;
  }

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

  // Put the first operand into one of the A registers...
  BuildMI(BB, MovOpcode[Class], 1, Reg).addReg(Op0Reg);

  if (isSigned) {
    // Emit a sign extension instruction...
    BuildMI(BB, ExtOpcode[Class], 0);
  } else {
    // If unsigned, emit a zeroing instruction... (reg = xor reg, reg)
    BuildMI(BB, ClrOpcode[Class], 2, ExtReg).addReg(ExtReg).addReg(ExtReg);
  }

  // Emit the appropriate divide or remainder instruction...
  BuildMI(BB, DivOpcode[isSigned][Class], 1).addReg(Op1Reg);

  // Figure out which register we want to pick the result out of...
  unsigned DestReg = (I.getOpcode() == Instruction::Div) ? Reg : ExtReg;
  
  // Put the result into the destination register...
  BuildMI(BB, MovOpcode[Class], 1, ResultReg).addReg(DestReg);
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

  if (OperandClass > cInt)
    visitInstruction(I); // Can't handle longs yet!

  if (ConstantUInt *CUI = dyn_cast<ConstantUInt> (I.getOperand (1)))
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

      BuildMI(BB, OpTab[OperandClass], 1, DestReg).addReg(Op0r);
    }
}


/// visitLoadInst - Implement LLVM load instructions in terms of the x86 'mov'
/// instruction.  The load and store instructions are the only place where we
/// need to worry about the memory layout of the target machine.
///
void ISel::visitLoadInst(LoadInst &I) {
  bool isLittleEndian  = TM.getTargetData().isLittleEndian();
  bool hasLongPointers = TM.getTargetData().getPointerSize() == 8;
  unsigned SrcAddrReg = getReg(I.getOperand(0));
  unsigned DestReg = getReg(I);

  unsigned Class = getClass(I.getType());
  switch (Class) {
  default: visitInstruction(I);   // FIXME: Handle longs...
  case cFP: {
    // FIXME: Handle endian swapping for FP values.
    unsigned Opcode = I.getType() == Type::FloatTy ? X86::FLDr32 : X86::FLDr64;
    addDirectMem(BuildMI(BB, Opcode, 4, DestReg), SrcAddrReg);
    return;
  }
  case cInt:      // Integers of various sizes handled below
  case cShort:
  case cByte: break;
  }

  // We need to adjust the input pointer if we are emulating a big-endian
  // long-pointer target.  On these systems, the pointer that we are interested
  // in is in the upper part of the eight byte memory image of the pointer.  It
  // also happens to be byte-swapped, but this will be handled later.
  //
  if (!isLittleEndian && hasLongPointers && isa<PointerType>(I.getType())) {
    unsigned R = makeAnotherReg(Type::UIntTy);
    BuildMI(BB, X86::ADDri32, 2, R).addReg(SrcAddrReg).addZImm(4);
    SrcAddrReg = R;
  }

  unsigned IReg = DestReg;
  if (!isLittleEndian) {  // If big endian we need an intermediate stage
    IReg = makeAnotherReg(I.getType());
    std::swap(IReg, DestReg);
  }

  static const unsigned Opcode[] = { X86::MOVmr8, X86::MOVmr16, X86::MOVmr32 };
  addDirectMem(BuildMI(BB, Opcode[Class], 4, DestReg), SrcAddrReg);

  if (!isLittleEndian) {
    // Emit the byte swap instruction...
    switch (Class) {
    case cByte:
      // No byteswap neccesary for 8 bit value...
      BuildMI(BB, X86::MOVrr8, 1, IReg).addReg(DestReg);
      break;
    case cInt:
      // Use the 32 bit bswap instruction to do a 32 bit swap...
      BuildMI(BB, X86::BSWAPr32, 1, IReg).addReg(DestReg);
      break;

    case cShort:
      // For 16 bit we have to use an xchg instruction, because there is no
      // 16-bit bswap.  XCHG is neccesarily not in SSA form, so we force things
      // into AX to do the xchg.
      //
      BuildMI(BB, X86::MOVrr16, 1, X86::AX).addReg(DestReg);
      BuildMI(BB, X86::XCHGrr8, 2).addReg(X86::AL, MOTy::UseAndDef)
                                  .addReg(X86::AH, MOTy::UseAndDef);
      BuildMI(BB, X86::MOVrr16, 1, DestReg).addReg(X86::AX);
      break;
    default: assert(0 && "Class not handled yet!");
    }
  }
}


/// visitStoreInst - Implement LLVM store instructions in terms of the x86 'mov'
/// instruction.
///
void ISel::visitStoreInst(StoreInst &I) {
  bool isLittleEndian  = TM.getTargetData().isLittleEndian();
  bool hasLongPointers = TM.getTargetData().getPointerSize() == 8;
  unsigned ValReg = getReg(I.getOperand(0));
  unsigned AddressReg = getReg(I.getOperand(1));

  unsigned Class = getClass(I.getOperand(0)->getType());
  switch (Class) {
  default: visitInstruction(I);   // FIXME: Handle longs...
  case cFP: {
    // FIXME: Handle endian swapping for FP values.
    unsigned Opcode = I.getOperand(0)->getType() == Type::FloatTy ?
                            X86::FSTr32 : X86::FSTr64;
    addDirectMem(BuildMI(BB, Opcode, 1+4), AddressReg).addReg(ValReg);
    return;
  }
  case cInt:      // Integers of various sizes handled below
  case cShort:
  case cByte: break;
  }

  if (!isLittleEndian && hasLongPointers &&
      isa<PointerType>(I.getOperand(0)->getType())) {
    unsigned R = makeAnotherReg(Type::UIntTy);
    BuildMI(BB, X86::ADDri32, 2, R).addReg(AddressReg).addZImm(4);
    AddressReg = R;
  }

  if (!isLittleEndian && Class != cByte) {
    // Emit a byte swap instruction...
    switch (Class) {
    case cInt: {
      unsigned R = makeAnotherReg(I.getOperand(0)->getType());
      BuildMI(BB, X86::BSWAPr32, 1, R).addReg(ValReg);
      ValReg = R;
      break;
    }
    case cShort:
      // For 16 bit we have to use an xchg instruction, because there is no
      // 16-bit bswap.  XCHG is neccesarily not in SSA form, so we force things
      // into AX to do the xchg.
      //
      BuildMI(BB, X86::MOVrr16, 1, X86::AX).addReg(ValReg);
      BuildMI(BB, X86::XCHGrr8, 2).addReg(X86::AL, MOTy::UseAndDef)
                                  .addReg(X86::AH, MOTy::UseAndDef);
      ValReg = X86::AX;
      break;
    default: assert(0 && "Unknown class!");
    }
  }

  static const unsigned Opcode[] = { X86::MOVrm8, X86::MOVrm16, X86::MOVrm32 };
  addDirectMem(BuildMI(BB, Opcode[Class], 1+4), AddressReg).addReg(ValReg);
}


/// visitCastInst - Here we have various kinds of copying with or without
/// sign extension going on.
void
ISel::visitCastInst (CastInst &CI)
{
  const Type *targetType = CI.getType ();
  Value *operand = CI.getOperand (0);
  unsigned operandReg = getReg (operand);
  const Type *sourceType = operand->getType ();
  unsigned destReg = getReg (CI);
  //
  // Currently we handle:
  //
  // 1) cast * to bool
  //
  // 2) cast {sbyte, ubyte} to {sbyte, ubyte}
  //    cast {short, ushort} to {ushort, short}
  //    cast {int, uint, ptr} to {int, uint, ptr}
  //
  // 3) cast {sbyte, ubyte} to {ushort, short}
  //    cast {sbyte, ubyte} to {int, uint, ptr}
  //    cast {short, ushort} to {int, uint, ptr}
  //
  // 4) cast {int, uint, ptr} to {short, ushort}
  //    cast {int, uint, ptr} to {sbyte, ubyte}
  //    cast {short, ushort} to {sbyte, ubyte}

  // 1) Implement casts to bool by using compare on the operand followed
  // by set if not zero on the result.
  if (targetType == Type::BoolTy)
    {
      BuildMI (BB, X86::CMPri8, 2).addReg (operandReg).addZImm (0);
      BuildMI (BB, X86::SETNEr, 1, destReg);
      return;
    }

  // 2) Implement casts between values of the same type class (as determined
  // by getClass) by using a register-to-register move.
  unsigned srcClass = getClassB(sourceType);
  unsigned targClass = getClass(targetType);
  static const unsigned regRegMove[] = {
    X86::MOVrr8, X86::MOVrr16, X86::MOVrr32
  };

  if (srcClass <= cInt && targClass <= cInt && srcClass == targClass) {
    BuildMI(BB, regRegMove[srcClass], 1, destReg).addReg(operandReg);
    return;
  }
  // 3) Handle cast of SMALLER int to LARGER int using a move with sign
  // extension or zero extension, depending on whether the source type
  // was signed.
  if ((srcClass <= cInt) && (targClass <= cInt) && (srcClass < targClass))
    {
      static const unsigned ops[] = {
	X86::MOVSXr16r8, X86::MOVSXr32r8, X86::MOVSXr32r16,
	X86::MOVZXr16r8, X86::MOVZXr32r8, X86::MOVZXr32r16
      };
      unsigned srcSigned = sourceType->isSigned ();
      BuildMI (BB, ops[3 * srcSigned + srcClass + targClass - 1], 1,
	       destReg).addReg (operandReg);
      return;
    }
  // 4) Handle cast of LARGER int to SMALLER int using a move to EAX
  // followed by a move out of AX or AL.
  if ((srcClass <= cInt) && (targClass <= cInt) && (srcClass > targClass))
    {
      static const unsigned AReg[] = { X86::AL, X86::AX, X86::EAX };
      BuildMI (BB, regRegMove[srcClass], 1,
	       AReg[srcClass]).addReg (operandReg);
      BuildMI (BB, regRegMove[targClass], 1, destReg).addReg (AReg[srcClass]);
      return;
    }
  // Anything we haven't handled already, we can't (yet) handle at all.
  //
  // FP to integral casts can be handled with FISTP to store onto the
  // stack while converting to integer, followed by a MOV to load from
  // the stack into the result register. Integral to FP casts can be
  // handled with MOV to store onto the stack, followed by a FILD to
  // load from the stack while converting to FP. For the moment, I
  // can't quite get straight in my head how to borrow myself some
  // stack space and write on it. Otherwise, this would be trivial.
  visitInstruction (CI);
}

// ExactLog2 - This function solves for (Val == 1 << (N-1)) and returns N.  It
// returns zero when the input is not exactly a power of two.
static unsigned ExactLog2(unsigned Val) {
  if (Val == 0) return 0;
  unsigned Count = 0;
  while (Val != 1) {
    if (Val & 1) return 0;
    Val >>= 1;
    ++Count;
  }
  return Count+1;
}

/// visitGetElementPtrInst - I don't know, most programs don't have
/// getelementptr instructions, right? That means we can put off
/// implementing this, right? Right. This method emits machine
/// instructions to perform type-safe pointer arithmetic. I am
/// guessing this could be cleaned up somewhat to use fewer temporary
/// registers.
void
ISel::visitGetElementPtrInst (GetElementPtrInst &I)
{
  unsigned outputReg = getReg (I);
  MachineBasicBlock::iterator MI = BB->end();
  emitGEPOperation(BB, MI, I.getOperand(0),
                   I.op_begin()+1, I.op_end(), outputReg);
}

void ISel::emitGEPOperation(MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator &IP,
                            Value *Src, User::op_iterator IdxBegin,
                            User::op_iterator IdxEnd, unsigned TargetReg) {
  const TargetData &TD = TM.getTargetData();
  const Type *Ty = Src->getType();
  unsigned basePtrReg = getReg(Src, MBB, IP);

  // GEPs have zero or more indices; we must perform a struct access
  // or array access for each one.
  for (GetElementPtrInst::op_iterator oi = IdxBegin,
         oe = IdxEnd; oi != oe; ++oi) {
    Value *idx = *oi;
    unsigned nextBasePtrReg = makeAnotherReg(Type::UIntTy);
    if (const StructType *StTy = dyn_cast<StructType>(Ty)) {
      // It's a struct access.  idx is the index into the structure,
      // which names the field. This index must have ubyte type.
      const ConstantUInt *CUI = cast<ConstantUInt>(idx);
      assert(CUI->getType() == Type::UByteTy
	      && "Funny-looking structure index in GEP");
      // Use the TargetData structure to pick out what the layout of
      // the structure is in memory.  Since the structure index must
      // be constant, we can get its value and use it to find the
      // right byte offset from the StructLayout class's list of
      // structure member offsets.
      unsigned idxValue = CUI->getValue();
      unsigned memberOffset =
	TD.getStructLayout(StTy)->MemberOffsets[idxValue];
      // Emit an ADD to add memberOffset to the basePtr.
      BMI(MBB, IP, X86::ADDri32, 2,
          nextBasePtrReg).addReg(basePtrReg).addZImm(memberOffset);
      // The next type is the member of the structure selected by the
      // index.
      Ty = StTy->getElementTypes()[idxValue];
    } else if (const SequentialType *SqTy = cast<SequentialType>(Ty)) {
      // It's an array or pointer access: [ArraySize x ElementType].

      // idx is the index into the array.  Unlike with structure
      // indices, we may not know its actual value at code-generation
      // time.
      assert(idx->getType() == Type::LongTy && "Bad GEP array index!");

      // We want to add basePtrReg to(idxReg * sizeof ElementType). First, we
      // must find the size of the pointed-to type (Not coincidentally, the next
      // type is the type of the elements in the array).
      Ty = SqTy->getElementType();
      unsigned elementSize = TD.getTypeSize(Ty);

      // If idxReg is a constant, we don't need to perform the multiply!
      if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(idx)) {
        if (CSI->isNullValue()) {
          BMI(MBB, IP, X86::MOVrr32, 1, nextBasePtrReg).addReg(basePtrReg);
        } else {
          unsigned Offset = elementSize*CSI->getValue();

          BMI(MBB, IP, X86::ADDri32, 2,
              nextBasePtrReg).addReg(basePtrReg).addZImm(Offset);
        }
      } else if (elementSize == 1) {
        // If the element size is 1, we don't have to multiply, just add
        unsigned idxReg = getReg(idx, MBB, IP);
        BMI(MBB, IP, X86::ADDrr32, 2,
            nextBasePtrReg).addReg(basePtrReg).addReg(idxReg);
      } else {
        unsigned idxReg = getReg(idx, MBB, IP);
        unsigned OffsetReg = makeAnotherReg(Type::UIntTy);
        if (unsigned Shift = ExactLog2(elementSize)) {
          // If the element size is exactly a power of 2, use a shift to get it.

          BMI(MBB, IP, X86::SHLir32, 2,
              OffsetReg).addReg(idxReg).addZImm(Shift-1);
        } else {
          // Most general case, emit a multiply...
          unsigned elementSizeReg = makeAnotherReg(Type::LongTy);
          BMI(MBB, IP, X86::MOVir32, 1, elementSizeReg).addZImm(elementSize);
        
          // Emit a MUL to multiply the register holding the index by
          // elementSize, putting the result in OffsetReg.
          doMultiply(MBB, IP, OffsetReg, Type::LongTy, idxReg, elementSizeReg);
        }
        // Emit an ADD to add OffsetReg to the basePtr.
        BMI(MBB, IP, X86::ADDrr32, 2,
            nextBasePtrReg).addReg(basePtrReg).addReg(OffsetReg);
      }
    }
    // Now that we are here, further indices refer to subtypes of this
    // one, so we don't need to worry about basePtrReg itself, anymore.
    basePtrReg = nextBasePtrReg;
  }
  // After we have processed all the indices, the result is left in
  // basePtrReg.  Move it to the register where we were expected to
  // put the answer.  A 32-bit move should do it, because we are in
  // ILP32 land.
  BMI(MBB, IP, X86::MOVrr32, 1, TargetReg).addReg(basePtrReg);
}


/// visitAllocaInst - If this is a fixed size alloca, allocate space from the
/// frame manager, otherwise do it the hard way.
///
void ISel::visitAllocaInst(AllocaInst &I) {
  // Find the data size of the alloca inst's getAllocatedType.
  const Type *Ty = I.getAllocatedType();
  unsigned TySize = TM.getTargetData().getTypeSize(Ty);

  // If this is a fixed size alloca in the entry block for the function,
  // statically stack allocate the space.
  //
  if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(I.getArraySize())) {
    if (I.getParent() == I.getParent()->getParent()->begin()) {
      TySize *= CUI->getValue();   // Get total allocated size...
      unsigned Alignment = TM.getTargetData().getTypeAlignment(Ty);
      
      // Create a new stack object using the frame manager...
      int FrameIdx = F->getFrameInfo()->CreateStackObject(TySize, Alignment);
      addFrameReference(BuildMI(BB, X86::LEAr32, 5, getReg(I)), FrameIdx);
      return;
    }
  }
  
  // Create a register to hold the temporary result of multiplying the type size
  // constant by the variable amount.
  unsigned TotalSizeReg = makeAnotherReg(Type::UIntTy);
  unsigned SrcReg1 = getReg(I.getArraySize());
  unsigned SizeReg = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, X86::MOVir32, 1, SizeReg).addZImm(TySize);
  
  // TotalSizeReg = mul <numelements>, <TypeSize>
  MachineBasicBlock::iterator MBBI = BB->end();
  doMultiply(BB, MBBI, TotalSizeReg, Type::UIntTy, SrcReg1, SizeReg);

  // AddedSize = add <TotalSizeReg>, 15
  unsigned AddedSizeReg = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, X86::ADDri32, 2, AddedSizeReg).addReg(TotalSizeReg).addZImm(15);

  // AlignedSize = and <AddedSize>, ~15
  unsigned AlignedSize = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, X86::ANDri32, 2, AlignedSize).addReg(AddedSizeReg).addZImm(~15);
  
  // Subtract size from stack pointer, thereby allocating some space.
  BuildMI(BB, X86::SUBri32, 2, X86::ESP).addReg(X86::ESP).addZImm(AlignedSize);

  // Put a pointer to the space into the result register, by copying
  // the stack pointer.
  BuildMI(BB, X86::MOVrr32, 1, getReg(I)).addReg(X86::ESP);

  // Inform the Frame Information that we have just allocated a variable sized
  // object.
  F->getFrameInfo()->CreateVariableSizedObject();
}
    

/// createSimpleX86InstructionSelector - This pass converts an LLVM function
/// into a machine code representation is a very simple peep-hole fashion.  The
/// generated code sucks but the implementation is nice and simple.
///
Pass *createSimpleX86InstructionSelector(TargetMachine &TM) {
  return new ISel(TM);
}
