//===-- InstSelectSimple.cpp - A simple instruction selector for PowerPC --===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "isel"
#include "PowerPC.h"
#include "PowerPCInstrBuilder.h"
#include "PowerPCInstrInfo.h"
#include "PowerPCTargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include <vector>
using namespace llvm;

namespace {
  Statistic<> GEPConsts("ppc-codegen", "Number of const GEPs");
  Statistic<> GEPSplits("ppc-codegen", "Number of partially const GEPs");

  /// TypeClass - Used by the PowerPC backend to group LLVM types by their basic
  /// PPC Representation.
  ///
  enum TypeClass {
    cByte, cShort, cInt, cFP32, cFP64, cLong
  };

  // This struct is for recording the necessary operations to emit the GEP
  typedef struct CollapsedGepOp {
  public:
    CollapsedGepOp(bool mul, Value *i, ConstantSInt *s) : 
      isMul(mul), index(i), size(s) {}
  
    bool isMul;
    Value *index;
    ConstantSInt *size;
  } CollapsedGepOp;
}

/// getClass - Turn a primitive type into a "class" number which is based on the
/// size of the type, and whether or not it is floating point.
///
static inline TypeClass getClass(const Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::SByteTyID:
  case Type::UByteTyID:   return cByte;      // Byte operands are class #0
  case Type::ShortTyID:
  case Type::UShortTyID:  return cShort;     // Short operands are class #1
  case Type::IntTyID:
  case Type::UIntTyID:
  case Type::PointerTyID: return cInt;       // Ints and pointers are class #2

  case Type::FloatTyID:   return cFP32;      // Single float is #3
  case Type::DoubleTyID:  return cFP64;      // Double Point is #4

  case Type::LongTyID:
  case Type::ULongTyID:   return cLong;      // Longs are class #5
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
    PowerPCTargetMachine &TM;
    MachineFunction *F;                 // The function we are compiling into
    MachineBasicBlock *BB;              // The current MBB we are compiling
    int VarArgsFrameIndex;              // FrameIndex for start of varargs area

    std::map<Value*, unsigned> RegMap;  // Mapping between Values and SSA Regs

    // External functions used in the Module
    Function *fmodfFn, *fmodFn, *__moddi3Fn, *__divdi3Fn, *__umoddi3Fn, 
      *__udivdi3Fn, *__fixsfdiFn, *__fixdfdiFn, *__floatdisfFn, *__floatdidfFn,
      *mallocFn, *freeFn;

    // MBBMap - Mapping between LLVM BB -> Machine BB
    std::map<const BasicBlock*, MachineBasicBlock*> MBBMap;

    // AllocaMap - Mapping from fixed sized alloca instructions to the
    // FrameIndex for the alloca.
    std::map<AllocaInst*, unsigned> AllocaMap;

    ISel(TargetMachine &tm) : TM(reinterpret_cast<PowerPCTargetMachine&>(tm)), 
      F(0), BB(0) {}

    bool doInitialization(Module &M) {
      // Add external functions that we may call
      Type *d = Type::DoubleTy;
      Type *f = Type::FloatTy;
      Type *l = Type::LongTy;
      Type *ul = Type::ULongTy;
      Type *voidPtr = PointerType::get(Type::SByteTy);
      // float fmodf(float, float);
      fmodfFn = M.getOrInsertFunction("fmodf", f, f, f, 0);
      // double fmod(double, double);
      fmodFn = M.getOrInsertFunction("fmod", d, d, d, 0);
      // long __moddi3(long, long);
      __moddi3Fn = M.getOrInsertFunction("__moddi3", l, l, l, 0);
      // long __divdi3(long, long);
      __divdi3Fn = M.getOrInsertFunction("__divdi3", l, l, l, 0);
      // unsigned long __umoddi3(unsigned long, unsigned long);
      __umoddi3Fn = M.getOrInsertFunction("__umoddi3", ul, ul, ul, 0);
      // unsigned long __udivdi3(unsigned long, unsigned long);
      __udivdi3Fn = M.getOrInsertFunction("__udivdi3", ul, ul, ul, 0);
      // long __fixsfdi(float)
      __fixdfdiFn = M.getOrInsertFunction("__fixsfdi", l, f, 0);
      // long __fixdfdi(double)
      __fixdfdiFn = M.getOrInsertFunction("__fixdfdi", l, d, 0);
      // float __floatdisf(long)
      __floatdisfFn = M.getOrInsertFunction("__floatdisf", f, l, 0);
      // double __floatdidf(long)
      __floatdidfFn = M.getOrInsertFunction("__floatdidf", d, l, 0);
      // void* malloc(size_t)
      mallocFn = M.getOrInsertFunction("malloc", voidPtr, Type::UIntTy, 0);
      // void free(void*)
      freeFn = M.getOrInsertFunction("free", Type::VoidTy, voidPtr, 0);
      return false;
    }

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
                const std::vector<ValueRecord> &Args, bool isVarArg);
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

    void doMultiply(MachineBasicBlock *MBB,
                    MachineBasicBlock::iterator IP,
                    unsigned DestReg, Value *Op0, Value *Op1);
  
    /// doMultiplyConst - This method will multiply the value in Op0Reg by the
    /// value of the ContantInt *CI
    void doMultiplyConst(MachineBasicBlock *MBB, 
                         MachineBasicBlock::iterator IP,
                         unsigned DestReg, Value *Op0, ConstantInt *CI);

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
    /// high 32 bits of the long value, and the regNum+1 is the low 32 bits.
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
    
    /// canUseAsImmediateForOpcode - This method returns whether a ConstantInt
    /// is okay to use as an immediate argument to a certain binary operation
    bool canUseAsImmediateForOpcode(ConstantInt *CI, unsigned Opcode);

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
  if (Constant *C = dyn_cast<Constant>(V)) {
    unsigned Reg = makeAnotherReg(V->getType());
    copyConstantToRegister(MBB, IPt, C, Reg);
    return Reg;
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

/// canUseAsImmediateForOpcode - This method returns whether a ConstantInt
/// is okay to use as an immediate argument to a certain binary operator.
///
/// Operator is one of: 0 for Add, 1 for Sub, 2 for And, 3 for Or, 4 for Xor.
bool ISel::canUseAsImmediateForOpcode(ConstantInt *CI, unsigned Operator)
{
  ConstantSInt *Op1Cs;
  ConstantUInt *Op1Cu;
      
  // ADDI, Compare, and non-indexed Load take SIMM
  bool cond1 = (Operator == 0) 
    && (Op1Cs = dyn_cast<ConstantSInt>(CI))
    && (Op1Cs->getValue() <= 32767)
    && (Op1Cs->getValue() >= -32768);

  // SUBI takes -SIMM since it is a mnemonic for ADDI
  bool cond2 = (Operator == 1)
    && (Op1Cs = dyn_cast<ConstantSInt>(CI)) 
    && (Op1Cs->getValue() <= 32768)
    && (Op1Cs->getValue() >= -32767);
      
  // ANDIo, ORI, and XORI take unsigned values
  bool cond3 = (Operator >= 2)
    && (Op1Cs = dyn_cast<ConstantSInt>(CI))
    && (Op1Cs->getValue() >= 0)
    && (Op1Cs->getValue() <= 32767);

  // ADDI and SUBI take SIMMs, so we have to make sure the UInt would fit
  bool cond4 = (Operator < 2)
    && (Op1Cu = dyn_cast<ConstantUInt>(CI)) 
    && (Op1Cu->getValue() <= 32767);

  // ANDIo, ORI, and XORI take UIMMs, so they can be larger
  bool cond5 = (Operator >= 2)
    && (Op1Cu = dyn_cast<ConstantUInt>(CI))
    && (Op1Cu->getValue() <= 65535);

  if (cond1 || cond2 || cond3 || cond4 || cond5)
    return true;

  return false;
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
  if (C->getType()->isIntegral()) {
    unsigned Class = getClassB(C->getType());

    if (Class == cLong) {
      // Copy the value into the register pair.
      uint64_t Val = cast<ConstantInt>(C)->getRawValue();
      
      if (Val < (1ULL << 16)) {
        BuildMI(*MBB, IP, PPC32::LI, 1, R).addSImm(0);
        BuildMI(*MBB, IP, PPC32::LI, 1, R+1).addSImm(Val & 0xFFFF);
      } else if (Val < (1ULL << 32)) {
        unsigned Temp = makeAnotherReg(Type::IntTy);
        BuildMI(*MBB, IP, PPC32::LI, 1, R).addSImm(0);
        BuildMI(*MBB, IP, PPC32::LIS, 1, Temp).addSImm((Val >> 16) & 0xFFFF);
        BuildMI(*MBB, IP, PPC32::ORI, 2, R+1).addReg(Temp).addImm(Val & 0xFFFF);
      } else if (Val < (1ULL << 48)) {
        unsigned Temp = makeAnotherReg(Type::IntTy);
        BuildMI(*MBB, IP, PPC32::LI, 1, R).addSImm((Val >> 32) & 0xFFFF);
        BuildMI(*MBB, IP, PPC32::LIS, 1, Temp).addSImm((Val >> 16) & 0xFFFF);
        BuildMI(*MBB, IP, PPC32::ORI, 2, R+1).addReg(Temp).addImm(Val & 0xFFFF);
      } else {
        unsigned TempLo = makeAnotherReg(Type::IntTy);
        unsigned TempHi = makeAnotherReg(Type::IntTy);
        BuildMI(*MBB, IP, PPC32::LIS, 1, TempHi).addSImm((Val >> 48) & 0xFFFF);
        BuildMI(*MBB, IP, PPC32::ORI, 2, R).addReg(TempHi)
          .addImm((Val >> 32) & 0xFFFF);
        BuildMI(*MBB, IP, PPC32::LIS, 1, TempLo).addSImm((Val >> 16) & 0xFFFF);
        BuildMI(*MBB, IP, PPC32::ORI, 2, R+1).addReg(TempLo)
          .addImm(Val & 0xFFFF);
      }
      return;
    }

    assert(Class <= cInt && "Type not handled yet!");

    if (C->getType() == Type::BoolTy) {
      BuildMI(*MBB, IP, PPC32::LI, 1, R).addSImm(C == ConstantBool::True);
    } else if (Class == cByte || Class == cShort) {
      ConstantInt *CI = cast<ConstantInt>(C);
      BuildMI(*MBB, IP, PPC32::LI, 1, R).addSImm(CI->getRawValue());
    } else {
      ConstantInt *CI = cast<ConstantInt>(C);
      int TheVal = CI->getRawValue() & 0xFFFFFFFF;
      if (TheVal < 32768 && TheVal >= -32768) {
        BuildMI(*MBB, IP, PPC32::LI, 1, R).addSImm(CI->getRawValue());
      } else {
        unsigned TmpReg = makeAnotherReg(Type::IntTy);
        BuildMI(*MBB, IP, PPC32::LIS, 1, TmpReg)
          .addSImm(CI->getRawValue() >> 16);
        BuildMI(*MBB, IP, PPC32::ORI, 2, R).addReg(TmpReg)
          .addImm(CI->getRawValue() & 0xFFFF);
      }
    }
  } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
    // We need to spill the constant to memory...
    MachineConstantPool *CP = F->getConstantPool();
    unsigned CPI = CP->getConstantPoolIndex(CFP);
    const Type *Ty = CFP->getType();

    assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");

    // Load addr of constant to reg; constant is located at PC + distance
    unsigned CurPC = makeAnotherReg(Type::IntTy);
    unsigned Reg1 = makeAnotherReg(Type::IntTy);
    unsigned Reg2 = makeAnotherReg(Type::IntTy);
    // Move PC to destination reg
    BuildMI(*MBB, IP, PPC32::MovePCtoLR, 0, CurPC);
    // Move value at PC + distance into return reg
    BuildMI(*MBB, IP, PPC32::LOADHiAddr, 2, Reg1).addReg(CurPC)
      .addConstantPoolIndex(CPI);
    BuildMI(*MBB, IP, PPC32::LOADLoDirect, 2, Reg2).addReg(Reg1)
      .addConstantPoolIndex(CPI);

    unsigned LoadOpcode = (Ty == Type::FloatTy) ? PPC32::LFS : PPC32::LFD;
    BuildMI(*MBB, IP, LoadOpcode, 2, R).addSImm(0).addReg(Reg2);
  } else if (isa<ConstantPointerNull>(C)) {
    // Copy zero (null pointer) to the register.
    BuildMI(*MBB, IP, PPC32::LI, 1, R).addSImm(0);
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(C)) {
    // GV is located at PC + distance
    unsigned CurPC = makeAnotherReg(Type::IntTy);
    unsigned TmpReg = makeAnotherReg(GV->getType());
    unsigned Opcode = (GV->hasWeakLinkage() || GV->isExternal()) ? 
      PPC32::LOADLoIndirect : PPC32::LOADLoDirect;
      
    // Move PC to destination reg
    BuildMI(*MBB, IP, PPC32::MovePCtoLR, 0, CurPC);
    // Move value at PC + distance into return reg
    BuildMI(*MBB, IP, PPC32::LOADHiAddr, 2, TmpReg).addReg(CurPC)
      .addGlobalAddress(GV);
    BuildMI(*MBB, IP, Opcode, 2, R).addReg(TmpReg).addGlobalAddress(GV);
  
    // Add the GV to the list of things whose addresses have been taken.
    TM.AddressTaken.insert(GV);
  } else {
    std::cerr << "Offending constant: " << *C << "\n";
    assert(0 && "Type not handled yet!");
  }
}

/// LoadArgumentsToVirtualRegs - Load all of the arguments to this function from
/// the stack into virtual registers.
///
/// FIXME: When we can calculate which args are coming in via registers
/// source them from there instead.
void ISel::LoadArgumentsToVirtualRegs(Function &Fn) {
  unsigned ArgOffset = 20;  // FIXME why is this not 24?
  unsigned GPR_remaining = 8;
  unsigned FPR_remaining = 13;
  unsigned GPR_idx = 0, FPR_idx = 0;
  static const unsigned GPR[] = { 
    PPC32::R3, PPC32::R4, PPC32::R5, PPC32::R6,
    PPC32::R7, PPC32::R8, PPC32::R9, PPC32::R10,
  };
  static const unsigned FPR[] = {
    PPC32::F1, PPC32::F2, PPC32::F3, PPC32::F4, PPC32::F5, PPC32::F6, PPC32::F7,
    PPC32::F8, PPC32::F9, PPC32::F10, PPC32::F11, PPC32::F12, PPC32::F13
  };
    
  MachineFrameInfo *MFI = F->getFrameInfo();
 
  for (Function::aiterator I = Fn.abegin(), E = Fn.aend(); I != E; ++I) {
    bool ArgLive = !I->use_empty();
    unsigned Reg = ArgLive ? getReg(*I) : 0;
    int FI;          // Frame object index

    switch (getClassB(I->getType())) {
    case cByte:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(4, ArgOffset);
        if (GPR_remaining > 0) {
          BuildMI(BB, PPC32::IMPLICIT_DEF, 0, GPR[GPR_idx]);
          BuildMI(BB, PPC32::OR, 2, Reg).addReg(GPR[GPR_idx])
            .addReg(GPR[GPR_idx]);
        } else {
          addFrameReference(BuildMI(BB, PPC32::LBZ, 2, Reg), FI);
        }
      }
      break;
    case cShort:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(4, ArgOffset);
        if (GPR_remaining > 0) {
          BuildMI(BB, PPC32::IMPLICIT_DEF, 0, GPR[GPR_idx]);
          BuildMI(BB, PPC32::OR, 2, Reg).addReg(GPR[GPR_idx])
            .addReg(GPR[GPR_idx]);
        } else {
          addFrameReference(BuildMI(BB, PPC32::LHZ, 2, Reg), FI);
        }
      }
      break;
    case cInt:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(4, ArgOffset);
        if (GPR_remaining > 0) {
          BuildMI(BB, PPC32::IMPLICIT_DEF, 0, GPR[GPR_idx]);
          BuildMI(BB, PPC32::OR, 2, Reg).addReg(GPR[GPR_idx])
            .addReg(GPR[GPR_idx]);
        } else {
          addFrameReference(BuildMI(BB, PPC32::LWZ, 2, Reg), FI);
        }
      }
      break;
    case cLong:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(8, ArgOffset);
        if (GPR_remaining > 1) {
          BuildMI(BB, PPC32::IMPLICIT_DEF, 0, GPR[GPR_idx]);
          BuildMI(BB, PPC32::IMPLICIT_DEF, 0, GPR[GPR_idx+1]);
          BuildMI(BB, PPC32::OR, 2, Reg).addReg(GPR[GPR_idx])
            .addReg(GPR[GPR_idx]);
          BuildMI(BB, PPC32::OR, 2, Reg+1).addReg(GPR[GPR_idx+1])
            .addReg(GPR[GPR_idx+1]);
        } else {
          addFrameReference(BuildMI(BB, PPC32::LWZ, 2, Reg), FI);
          addFrameReference(BuildMI(BB, PPC32::LWZ, 2, Reg+1), FI, 4);
        }
      }
      // longs require 4 additional bytes and use 2 GPRs
      ArgOffset += 4;
      if (GPR_remaining > 1) {
        GPR_remaining--;
        GPR_idx++;
      }
      break;
    case cFP32:
     if (ArgLive) {
        FI = MFI->CreateFixedObject(4, ArgOffset);

        if (FPR_remaining > 0) {
          BuildMI(BB, PPC32::IMPLICIT_DEF, 0, FPR[FPR_idx]);
          BuildMI(BB, PPC32::FMR, 1, Reg).addReg(FPR[FPR_idx]);
          FPR_remaining--;
          FPR_idx++;
        } else {
          addFrameReference(BuildMI(BB, PPC32::LFS, 2, Reg), FI);
        }
      }
      break;
    case cFP64:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(8, ArgOffset);

        if (FPR_remaining > 0) {
          BuildMI(BB, PPC32::IMPLICIT_DEF, 0, FPR[FPR_idx]);
          BuildMI(BB, PPC32::FMR, 1, Reg).addReg(FPR[FPR_idx]);
          FPR_remaining--;
          FPR_idx++;
        } else {
          addFrameReference(BuildMI(BB, PPC32::LFD, 2, Reg), FI);
        }
      }

      // doubles require 4 additional bytes and use 2 GPRs of param space
      ArgOffset += 4;   
      if (GPR_remaining > 0) {
        GPR_remaining--;
        GPR_idx++;
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
        MachineBasicBlock *PredMBB = 0;
        for (MachineBasicBlock::pred_iterator PI = MBB.pred_begin (),
             PE = MBB.pred_end (); PI != PE; ++PI)
          if (PN->getIncomingBlock(i) == (*PI)->getBasicBlock()) {
            PredMBB = *PI;
            break;
          }
        assert (PredMBB && "Couldn't find incoming machine-cfg edge for phi");

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
// conditional branch.
//
static SetCondInst *canFoldSetCCIntoBranchOrSelect(Value *V) {
  if (SetCondInst *SCI = dyn_cast<SetCondInst>(V))
    if (SCI->hasOneUse()) {
      Instruction *User = cast<Instruction>(SCI->use_back());
      if ((isa<BranchInst>(User) || isa<SelectInst>(User)) &&
          SCI->getParent() == User->getParent())
        return SCI;
    }
  return 0;
}

// Return a fixed numbering for setcc instructions which does not depend on the
// order of the opcodes.
//
static unsigned getSetCCNumber(unsigned Opcode) {
  switch (Opcode) {
  default: assert(0 && "Unknown setcc instruction!");
  case Instruction::SetEQ: return 0;
  case Instruction::SetNE: return 1;
  case Instruction::SetLT: return 2;
  case Instruction::SetGE: return 3;
  case Instruction::SetGT: return 4;
  case Instruction::SetLE: return 5;
  }
}

static unsigned getPPCOpcodeForSetCCNumber(unsigned Opcode) {
  switch (Opcode) {
  default: assert(0 && "Unknown setcc instruction!");
  case Instruction::SetEQ: return PPC32::BEQ;
  case Instruction::SetNE: return PPC32::BNE;
  case Instruction::SetLT: return PPC32::BLT;
  case Instruction::SetGE: return PPC32::BGE;
  case Instruction::SetGT: return PPC32::BGT;
  case Instruction::SetLE: return PPC32::BLE;
  }
}

static unsigned invertPPCBranchOpcode(unsigned Opcode) {
  switch (Opcode) {
  default: assert(0 && "Unknown PPC32 branch opcode!");
  case PPC32::BEQ: return PPC32::BNE;
  case PPC32::BNE: return PPC32::BEQ;
  case PPC32::BLT: return PPC32::BGE;
  case PPC32::BGE: return PPC32::BLT;
  case PPC32::BGT: return PPC32::BLE;
  case PPC32::BLE: return PPC32::BGT;
  }
}

/// emitUCOM - emits an unordered FP compare.
void ISel::emitUCOM(MachineBasicBlock *MBB, MachineBasicBlock::iterator IP,
                     unsigned LHS, unsigned RHS) {
    BuildMI(*MBB, IP, PPC32::FCMPU, 2, PPC32::CR0).addReg(LHS).addReg(RHS);
}

/// EmitComparison - emits a comparison of the two operands, returning the
/// extended setcc code to use.  The result is in CR0.
///
unsigned ISel::EmitComparison(unsigned OpNum, Value *Op0, Value *Op1,
                              MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator IP) {
  // The arguments are already supposed to be of the same type.
  const Type *CompTy = Op0->getType();
  unsigned Class = getClassB(CompTy);
  unsigned Op0r = getReg(Op0, MBB, IP);
  
  // Use crand for lt, gt and crandc for le, ge
  unsigned CROpcode = (OpNum == 2 || OpNum == 4) ? PPC32::CRAND : PPC32::CRANDC;
  // ? cr1[lt] : cr1[gt]
  unsigned CR1field = (OpNum == 2 || OpNum == 3) ? 4 : 5;
  // ? cr0[lt] : cr0[gt]
  unsigned CR0field = (OpNum == 2 || OpNum == 5) ? 0 : 1;
  unsigned Opcode = CompTy->isSigned() ? PPC32::CMPW : PPC32::CMPLW;
  unsigned OpcodeImm = CompTy->isSigned() ? PPC32::CMPWI : PPC32::CMPLWI;

  // Special case handling of: cmp R, i
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    if (Class == cByte || Class == cShort || Class == cInt) {
      unsigned Op1v = CI->getRawValue() & 0xFFFF;

      // Treat compare like ADDI for the purposes of immediate suitability
      if (canUseAsImmediateForOpcode(CI, 0)) {
        BuildMI(*MBB, IP, OpcodeImm, 2, PPC32::CR0).addReg(Op0r).addSImm(Op1v);
      } else {
        unsigned Op1r = getReg(Op1, MBB, IP);
        BuildMI(*MBB, IP, Opcode, 2, PPC32::CR0).addReg(Op0r).addReg(Op1r);
      }
      return OpNum;
    } else {
      assert(Class == cLong && "Unknown integer class!");
      unsigned LowCst = CI->getRawValue();
      unsigned HiCst = CI->getRawValue() >> 32;
      if (OpNum < 2) {    // seteq, setne
        unsigned LoLow = makeAnotherReg(Type::IntTy);
        unsigned LoTmp = makeAnotherReg(Type::IntTy);
        unsigned HiLow = makeAnotherReg(Type::IntTy);
        unsigned HiTmp = makeAnotherReg(Type::IntTy);
        unsigned FinalTmp = makeAnotherReg(Type::IntTy);
        
        BuildMI(*MBB, IP, PPC32::XORI, 2, LoLow).addReg(Op0r+1)
          .addImm(LowCst & 0xFFFF);
        BuildMI(*MBB, IP, PPC32::XORIS, 2, LoTmp).addReg(LoLow)
          .addImm(LowCst >> 16);
        BuildMI(*MBB, IP, PPC32::XORI, 2, HiLow).addReg(Op0r)
          .addImm(HiCst & 0xFFFF);
        BuildMI(*MBB, IP, PPC32::XORIS, 2, HiTmp).addReg(HiLow)
          .addImm(HiCst >> 16);
        BuildMI(*MBB, IP, PPC32::ORo, 2, FinalTmp).addReg(LoTmp).addReg(HiTmp);
        return OpNum;
      } else {
        unsigned ConstReg = makeAnotherReg(CompTy);
        copyConstantToRegister(MBB, IP, CI, ConstReg);
        
        // cr0 = r3 ccOpcode r5 or (r3 == r5 AND r4 ccOpcode r6)
        BuildMI(*MBB, IP, Opcode, 2, PPC32::CR0).addReg(Op0r)
          .addReg(ConstReg);
        BuildMI(*MBB, IP, Opcode, 2, PPC32::CR1).addReg(Op0r+1)
          .addReg(ConstReg+1);
        BuildMI(*MBB, IP, PPC32::CRAND, 3).addImm(2).addImm(2).addImm(CR1field);
        BuildMI(*MBB, IP, PPC32::CROR, 3).addImm(CR0field).addImm(CR0field)
          .addImm(2);
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
    BuildMI(*MBB, IP, Opcode, 2, PPC32::CR0).addReg(Op0r).addReg(Op1r);
    break;

  case cFP32:
  case cFP64:
    emitUCOM(MBB, IP, Op0r, Op1r);
    break;

  case cLong:
    if (OpNum < 2) {    // seteq, setne
      unsigned LoTmp = makeAnotherReg(Type::IntTy);
      unsigned HiTmp = makeAnotherReg(Type::IntTy);
      unsigned FinalTmp = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, PPC32::XOR, 2, HiTmp).addReg(Op0r).addReg(Op1r);
      BuildMI(*MBB, IP, PPC32::XOR, 2, LoTmp).addReg(Op0r+1).addReg(Op1r+1);
      BuildMI(*MBB, IP, PPC32::ORo,  2, FinalTmp).addReg(LoTmp).addReg(HiTmp);
      break;  // Allow the sete or setne to be generated from flags set by OR
    } else {
      unsigned TmpReg1 = makeAnotherReg(Type::IntTy);
      unsigned TmpReg2 = makeAnotherReg(Type::IntTy);

      // cr0 = r3 ccOpcode r5 or (r3 == r5 AND r4 ccOpcode r6)
      BuildMI(*MBB, IP, Opcode, 2, PPC32::CR0).addReg(Op0r).addReg(Op1r);
      BuildMI(*MBB, IP, Opcode, 2, PPC32::CR1).addReg(Op0r+1).addReg(Op1r+1);
      BuildMI(*MBB, IP, PPC32::CRAND, 3).addImm(2).addImm(2).addImm(CR1field);
      BuildMI(*MBB, IP, PPC32::CROR, 3).addImm(CR0field).addImm(CR0field)
        .addImm(2);
      return OpNum;
    }
  }
  return OpNum;
}

/// visitSetCondInst - emit code to calculate the condition via
/// EmitComparison(), and possibly store a 0 or 1 to a register as a result
///
void ISel::visitSetCondInst(SetCondInst &I) {
  if (canFoldSetCCIntoBranchOrSelect(&I))
    return;

  unsigned DestReg = getReg(I);
  unsigned OpNum = I.getOpcode();
  const Type *Ty = I.getOperand (0)->getType();
                   
  EmitComparison(OpNum, I.getOperand(0), I.getOperand(1), BB, BB->end());
 
  unsigned Opcode = getPPCOpcodeForSetCCNumber(OpNum);
  MachineBasicBlock *thisMBB = BB;
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  ilist<MachineBasicBlock>::iterator It = BB;
  ++It;
  
  //  thisMBB:
  //  ...
  //   cmpTY cr0, r1, r2
  //   bCC copy1MBB
  //   b copy0MBB

  // FIXME: we wouldn't need copy0MBB (we could fold it into thisMBB)
  // if we could insert other, non-terminator instructions after the
  // bCC. But MBB->getFirstTerminator() can't understand this.
  MachineBasicBlock *copy1MBB = new MachineBasicBlock(LLVM_BB);
  F->getBasicBlockList().insert(It, copy1MBB);
  BuildMI(BB, Opcode, 2).addReg(PPC32::CR0).addMBB(copy1MBB);
  MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
  F->getBasicBlockList().insert(It, copy0MBB);
  BuildMI(BB, PPC32::B, 1).addMBB(copy0MBB);
  MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
  F->getBasicBlockList().insert(It, sinkMBB);
  // Update machine-CFG edges
  BB->addSuccessor(copy1MBB);
  BB->addSuccessor(copy0MBB);

  //  copy1MBB:
  //   %TrueValue = li 1
  //   b sinkMBB
  BB = copy1MBB;
  unsigned TrueValue = makeAnotherReg(I.getType());
  BuildMI(BB, PPC32::LI, 1, TrueValue).addSImm(1);
  BuildMI(BB, PPC32::B, 1).addMBB(sinkMBB);
  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  copy0MBB:
  //   %FalseValue = li 0
  //   fallthrough
  BB = copy0MBB;
  unsigned FalseValue = makeAnotherReg(I.getType());
  BuildMI(BB, PPC32::LI, 1, FalseValue).addSImm(0);
  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, copy1MBB ]
  //  ...
  BB = sinkMBB;
  BuildMI(BB, PPC32::PHI, 4, DestReg).addReg(FalseValue)
    .addMBB(copy0MBB).addReg(TrueValue).addMBB(copy1MBB);
}

void ISel::visitSelectInst(SelectInst &SI) {
  unsigned DestReg = getReg(SI);
  MachineBasicBlock::iterator MII = BB->end();
  emitSelectOperation(BB, MII, SI.getCondition(), SI.getTrueValue(),
                      SI.getFalseValue(), DestReg);
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
  unsigned Opcode;

  // See if we can fold the setcc into the select instruction, or if we have
  // to get the register of the Cond value
  if (SetCondInst *SCI = canFoldSetCCIntoBranchOrSelect(Cond)) {
    // We successfully folded the setcc into the select instruction.
    
    unsigned OpNum = getSetCCNumber(SCI->getOpcode());
    OpNum = EmitComparison(OpNum, SCI->getOperand(0), SCI->getOperand(1), MBB,
                           IP);
    Opcode = getPPCOpcodeForSetCCNumber(SCI->getOpcode());
  } else {
    unsigned CondReg = getReg(Cond, MBB, IP);

    BuildMI(*MBB, IP, PPC32::CMPI, 2, PPC32::CR0).addReg(CondReg).addSImm(0);
    Opcode = getPPCOpcodeForSetCCNumber(Instruction::SetNE);
  }

  //  thisMBB:
  //  ...
  //   cmpTY cr0, r1, r2
  //   bCC copy1MBB
  //   b copy0MBB

  MachineBasicBlock *thisMBB = BB;
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  ilist<MachineBasicBlock>::iterator It = BB;
  ++It;

  // FIXME: we wouldn't need copy0MBB (we could fold it into thisMBB)
  // if we could insert other, non-terminator instructions after the
  // bCC. But MBB->getFirstTerminator() can't understand this.
  MachineBasicBlock *copy1MBB = new MachineBasicBlock(LLVM_BB);
  F->getBasicBlockList().insert(It, copy1MBB);
  BuildMI(BB, Opcode, 2).addReg(PPC32::CR0).addMBB(copy1MBB);
  MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
  F->getBasicBlockList().insert(It, copy0MBB);
  BuildMI(BB, PPC32::B, 1).addMBB(copy0MBB);
  MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
  F->getBasicBlockList().insert(It, sinkMBB);
  // Update machine-CFG edges
  BB->addSuccessor(copy1MBB);
  BB->addSuccessor(copy0MBB);

  //  copy1MBB:
  //   %TrueValue = ...
  //   b sinkMBB
  BB = copy1MBB;
  unsigned TrueValue = getReg(TrueVal, BB, BB->begin());
  BuildMI(BB, PPC32::B, 1).addMBB(sinkMBB);
  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  copy0MBB:
  //   %FalseValue = ...
  //   fallthrough
  BB = copy0MBB;
  unsigned FalseValue = getReg(FalseVal, BB, BB->begin());
  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, copy1MBB ]
  //  ...
  BB = sinkMBB;
  BuildMI(BB, PPC32::PHI, 4, DestReg).addReg(FalseValue)
    .addMBB(copy0MBB).addReg(TrueValue).addMBB(copy1MBB);
  // For a register pair representing a long value, define the second reg
  if (getClass(TrueVal->getType()) == cLong)
    BuildMI(BB, PPC32::LI, 1, DestReg+1).addImm(0);
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

    // If this is a simple constant, just emit a load directly to avoid the copy
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
      int TheVal = CI->getRawValue() & 0xFFFFFFFF;

      if (TheVal < 32768 && TheVal >= -32768) {
        BuildMI(BB, PPC32::LI, 1, targetReg).addSImm(TheVal);
      } else {
        unsigned TmpReg = makeAnotherReg(Type::IntTy);
        BuildMI(BB, PPC32::LIS, 1, TmpReg).addSImm(TheVal >> 16);
        BuildMI(BB, PPC32::ORI, 2, targetReg).addReg(TmpReg)
          .addImm(TheVal & 0xFFFF);
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
      BuildMI(BB, PPC32::RLWINM, 4, targetReg).addReg(Reg).addZImm(0)
        .addZImm(24).addZImm(31);
    else
      BuildMI(BB, PPC32::EXTSB, 1, targetReg).addReg(Reg);
    break;
  case cShort:
    // Extend value into target register (16->32)
    if (isUnsigned)
      BuildMI(BB, PPC32::RLWINM, 4, targetReg).addReg(Reg).addZImm(0)
        .addZImm(16).addZImm(31);
    else
      BuildMI(BB, PPC32::EXTSH, 1, targetReg).addReg(Reg);
    break;
  case cInt:
    // Move value into target register (32->32)
    BuildMI(BB, PPC32::OR, 2, targetReg).addReg(Reg).addReg(Reg);
    break;
  default:
    assert(0 && "Unpromotable operand class in promote32");
  }
}

/// visitReturnInst - implemented with BLR
///
void ISel::visitReturnInst(ReturnInst &I) {
  // Only do the processing if this is a non-void return
  if (I.getNumOperands() > 0) {
    Value *RetVal = I.getOperand(0);
    switch (getClassB(RetVal->getType())) {
    case cByte:   // integral return values: extend or move into r3 and return
    case cShort:
    case cInt:
      promote32(PPC32::R3, ValueRecord(RetVal));
      break;
    case cFP32:
    case cFP64: {   // Floats & Doubles: Return in f1
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
  BB->addSuccessor(MBBMap[BI.getSuccessor(0)]);
  if (BI.isConditional())
    BB->addSuccessor(MBBMap[BI.getSuccessor(1)]);
  
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
    BuildMI(BB, PPC32::CMPLI, 3, PPC32::CR1).addImm(0).addReg(condReg)
      .addImm(0);
    if (BI.getSuccessor(1) == NextBB) {
      if (BI.getSuccessor(0) != NextBB)
        BuildMI(BB, PPC32::BNE, 2).addReg(PPC32::CR1)
          .addMBB(MBBMap[BI.getSuccessor(0)]);
    } else {
      BuildMI(BB, PPC32::BEQ, 2).addReg(PPC32::CR1)
        .addMBB(MBBMap[BI.getSuccessor(1)]);
      
      if (BI.getSuccessor(0) != NextBB)
        BuildMI(BB, PPC32::B, 1).addMBB(MBBMap[BI.getSuccessor(0)]);
    }
    return;
  }

  unsigned OpNum = getSetCCNumber(SCI->getOpcode());
  unsigned Opcode = getPPCOpcodeForSetCCNumber(SCI->getOpcode());
  MachineBasicBlock::iterator MII = BB->end();
  OpNum = EmitComparison(OpNum, SCI->getOperand(0), SCI->getOperand(1), BB,MII);
  
  if (BI.getSuccessor(0) != NextBB) {
    BuildMI(BB, Opcode, 2).addReg(PPC32::CR0)
      .addMBB(MBBMap[BI.getSuccessor(0)]);
    if (BI.getSuccessor(1) != NextBB)
      BuildMI(BB, PPC32::B, 1).addMBB(MBBMap[BI.getSuccessor(1)]);
  } else {
    // Change to the inverse condition...
    if (BI.getSuccessor(1) != NextBB) {
      Opcode = invertPPCBranchOpcode(Opcode);
      BuildMI(BB, Opcode, 2).addReg(PPC32::CR0)
        .addMBB(MBBMap[BI.getSuccessor(1)]);
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
                  const std::vector<ValueRecord> &Args, bool isVarArg) {
  // Count how many bytes are to be pushed on the stack...
  unsigned NumBytes = 0;

  if (!Args.empty()) {
    for (unsigned i = 0, e = Args.size(); i != e; ++i)
      switch (getClassB(Args[i].Ty)) {
      case cByte: case cShort: case cInt:
        NumBytes += 4; break;
      case cLong:
        NumBytes += 8; break;
      case cFP32:
        NumBytes += 4; break;
      case cFP64:
        NumBytes += 8; break;
        break;
      default: assert(0 && "Unknown class!");
      }

    // Adjust the stack pointer for the new arguments...
    BuildMI(BB, PPC32::ADJCALLSTACKDOWN, 1).addSImm(NumBytes);

    // Arguments go on the stack in reverse order, as specified by the ABI.
    // Offset to the paramater area on the stack is 24.
    unsigned ArgOffset = 24;
    int GPR_remaining = 8, FPR_remaining = 13;
    unsigned GPR_idx = 0, FPR_idx = 0;
    static const unsigned GPR[] = { 
      PPC32::R3, PPC32::R4, PPC32::R5, PPC32::R6,
      PPC32::R7, PPC32::R8, PPC32::R9, PPC32::R10,
    };
    static const unsigned FPR[] = {
      PPC32::F1, PPC32::F2, PPC32::F3, PPC32::F4, PPC32::F5, PPC32::F6, 
      PPC32::F7, PPC32::F8, PPC32::F9, PPC32::F10, PPC32::F11, PPC32::F12, 
      PPC32::F13
    };
    
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
          BuildMI(BB, PPC32::OR, 2, GPR[GPR_idx]).addReg(ArgReg)
            .addReg(ArgReg);
          CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
        } else {
          BuildMI(BB, PPC32::STW, 3).addReg(ArgReg).addSImm(ArgOffset)
            .addReg(PPC32::R1);
        }
        break;
      case cInt:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;

        // Reg or stack?
        if (GPR_remaining > 0) {
          BuildMI(BB, PPC32::OR, 2, GPR[GPR_idx]).addReg(ArgReg)
            .addReg(ArgReg);
          CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
        } else {
          BuildMI(BB, PPC32::STW, 3).addReg(ArgReg).addSImm(ArgOffset)
            .addReg(PPC32::R1);
        }
        break;
      case cLong:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;

        // Reg or stack?  Note that PPC calling conventions state that long args
        // are passed rN = hi, rN+1 = lo, opposite of LLVM.
        if (GPR_remaining > 1) {
          BuildMI(BB, PPC32::OR, 2, GPR[GPR_idx]).addReg(ArgReg)
            .addReg(ArgReg);
          BuildMI(BB, PPC32::OR, 2, GPR[GPR_idx+1]).addReg(ArgReg+1)
            .addReg(ArgReg+1);
          CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
          CallMI->addRegOperand(GPR[GPR_idx+1], MachineOperand::Use);
        } else {
          BuildMI(BB, PPC32::STW, 3).addReg(ArgReg).addSImm(ArgOffset)
            .addReg(PPC32::R1);
          BuildMI(BB, PPC32::STW, 3).addReg(ArgReg+1).addSImm(ArgOffset+4)
            .addReg(PPC32::R1);
        }

        ArgOffset += 4;        // 8 byte entry, not 4.
        GPR_remaining -= 1;    // uses up 2 GPRs
        GPR_idx += 1;
        break;
      case cFP32:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
        // Reg or stack?
        if (FPR_remaining > 0) {
          BuildMI(BB, PPC32::FMR, 1, FPR[FPR_idx]).addReg(ArgReg);
          CallMI->addRegOperand(FPR[FPR_idx], MachineOperand::Use);
          FPR_remaining--;
          FPR_idx++;
          
          // If this is a vararg function, and there are GPRs left, also
          // pass the float in an int.  Otherwise, put it on the stack.
          if (isVarArg) {
            BuildMI(BB, PPC32::STFS, 3).addReg(ArgReg).addSImm(ArgOffset)
            .addReg(PPC32::R1);
            if (GPR_remaining > 0) {
              BuildMI(BB, PPC32::LWZ, 2, GPR[GPR_idx])
              .addSImm(ArgOffset).addReg(ArgReg);
              CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
            }
          }
        } else {
          BuildMI(BB, PPC32::STFS, 3).addReg(ArgReg).addSImm(ArgOffset)
          .addReg(PPC32::R1);
        }
        break;
      case cFP64:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
        // Reg or stack?
        if (FPR_remaining > 0) {
          BuildMI(BB, PPC32::FMR, 1, FPR[FPR_idx]).addReg(ArgReg);
          CallMI->addRegOperand(FPR[FPR_idx], MachineOperand::Use);
          FPR_remaining--;
          FPR_idx++;
          // For vararg functions, must pass doubles via int regs as well
          if (isVarArg) {
            BuildMI(BB, PPC32::STFD, 3).addReg(ArgReg).addSImm(ArgOffset)
            .addReg(PPC32::R1);
            
            // Doubles can be split across reg + stack for varargs
            if (GPR_remaining > 0) {
              BuildMI(BB, PPC32::LWZ, 2, GPR[GPR_idx]).addSImm(ArgOffset)
              .addReg(PPC32::R1);
              CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
            }
            if (GPR_remaining > 1) {
              BuildMI(BB, PPC32::LWZ, 2, GPR[GPR_idx+1])
                .addSImm(ArgOffset+4).addReg(PPC32::R1);
              CallMI->addRegOperand(GPR[GPR_idx+1], MachineOperand::Use);
            }
          }
        } else {
          BuildMI(BB, PPC32::STFD, 3).addReg(ArgReg).addSImm(ArgOffset)
          .addReg(PPC32::R1);
        }
        // Doubles use 8 bytes, and 2 GPRs worth of param space
        ArgOffset += 4;
        GPR_remaining--;
        GPR_idx++;
        break;
        
      default: assert(0 && "Unknown class!");
      }
      ArgOffset += 4;
      GPR_remaining--;
      GPR_idx++;
    }
  } else {
    BuildMI(BB, PPC32::ADJCALLSTACKDOWN, 1).addSImm(0);
  }

  BB->push_back(CallMI);
  BuildMI(BB, PPC32::ADJCALLSTACKUP, 1).addSImm(NumBytes);

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
      break;
    case cFP32:     // Floating-point return values live in f1
    case cFP64:
      BuildMI(BB, PPC32::FMR, 1, Ret.Reg).addReg(PPC32::F1);
      break;
    case cLong:   // Long values are in r3 hi:r4 lo
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
  Function *F = CI.getCalledFunction();
  if (F) {
    // Is it an intrinsic function call?
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID()) {
      visitIntrinsicCall(ID, CI);   // Special intrinsics are not handled here
      return;
    }
    // Emit a CALL instruction with PC-relative displacement.
    TheCall = BuildMI(PPC32::CALLpcrel, 1).addGlobalAddress(F, true);
    // Add it to the set of functions called to be used by the Printer
    TM.CalledFunctions.insert(F);
  } else {  // Emit an indirect call through the CTR
    unsigned Reg = getReg(CI.getCalledValue());
    BuildMI(BB, PPC32::MTCTR, 1).addReg(Reg);
    TheCall = BuildMI(PPC32::CALLindirect, 2).addZImm(20).addZImm(0);
  }

  std::vector<ValueRecord> Args;
  for (unsigned i = 1, e = CI.getNumOperands(); i != e; ++i)
    Args.push_back(ValueRecord(CI.getOperand(i)));

  unsigned DestReg = CI.getType() != Type::VoidTy ? getReg(CI) : 0;
  bool isVarArg = F ? F->getFunctionType()->isVarArg() : true;
  doCall(ValueRecord(DestReg, CI.getType()), TheCall, Args, isVarArg);
}         


/// dyncastIsNan - Return the operand of an isnan operation if this is an isnan.
///
static Value *dyncastIsNan(Value *V) {
  if (CallInst *CI = dyn_cast<CallInst>(V))
    if (Function *F = CI->getCalledFunction())
      if (F->getIntrinsicID() == Intrinsic::isunordered)
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
            // FIXME: should lower this ourselves
            // case Intrinsic::isunordered:
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
            StoreInst *SI = new StoreInst(CI->getOperand(1),
                                          CI->getOperand(2), true, CI);
            CI->replaceAllUsesWith(SI);
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
    addFrameReference(BuildMI(BB, PPC32::ADDI, 2, TmpReg1), VarArgsFrameIndex, 
                      0, false);
    return;

  case Intrinsic::vacopy:
    TmpReg1 = getReg(CI);
    TmpReg2 = getReg(CI.getOperand(1));
    BuildMI(BB, PPC32::OR, 2, TmpReg1).addReg(TmpReg2).addReg(TmpReg2);
    return;
  case Intrinsic::vaend: return;

  case Intrinsic::returnaddress:
    TmpReg1 = getReg(CI);
    if (cast<Constant>(CI.getOperand(1))->isNullValue()) {
      MachineFrameInfo *MFI = F->getFrameInfo();
      unsigned NumBytes = MFI->getStackSize();
      
      BuildMI(BB, PPC32::LWZ, 2, TmpReg1).addSImm(NumBytes+8)
        .addReg(PPC32::R1);
    } else {
      // Values other than zero are not implemented yet.
      BuildMI(BB, PPC32::LI, 1, TmpReg1).addSImm(0);
    }
    return;

  case Intrinsic::frameaddress:
    TmpReg1 = getReg(CI);
    if (cast<Constant>(CI.getOperand(1))->isNullValue()) {
      BuildMI(BB, PPC32::OR, 2, TmpReg1).addReg(PPC32::R1).addReg(PPC32::R1);
    } else {
      // Values other than zero are not implemented yet.
      BuildMI(BB, PPC32::LI, 1, TmpReg1).addSImm(0);
    }
    return;

#if 0
    // This may be useful for supporting isunordered
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
#endif
    
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
    assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");

    static const unsigned OpcodeTab[][4] = {
      { PPC32::FADDS, PPC32::FSUBS, PPC32::FMULS, PPC32::FDIVS },  // Float
      { PPC32::FADD,  PPC32::FSUB,  PPC32::FMUL,  PPC32::FDIV },   // Double
    };

    unsigned Opcode = OpcodeTab[Ty != Type::FloatTy][OperatorClass];
    unsigned Op1Reg = getReg(Op1C, BB, IP);
    unsigned Op0r = getReg(Op0, BB, IP);
    BuildMI(*BB, IP, Opcode, 2, DestReg).addReg(Op0r).addReg(Op1Reg);
    return;
  }
  
  // Special case: R1 = op <const fp>, R2
  if (ConstantFP *Op0C = dyn_cast<ConstantFP>(Op0))
    if (Op0C->isExactlyValue(-0.0) && OperatorClass == 1) {
      // -0.0 - X === -X
      unsigned op1Reg = getReg(Op1, BB, IP);
      BuildMI(*BB, IP, PPC32::FNEG, 1, DestReg).addReg(op1Reg);
      return;
    } else {
      // R1 = op CST, R2  -->  R1 = opr R2, CST

      // Create a constant pool entry for this constant.
      MachineConstantPool *CP = F->getConstantPool();
      unsigned CPI = CP->getConstantPoolIndex(Op0C);
      const Type *Ty = Op0C->getType();
      assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");

      static const unsigned OpcodeTab[][4] = {
        { PPC32::FADDS, PPC32::FSUBS, PPC32::FMULS, PPC32::FDIVS },  // Float
        { PPC32::FADD,  PPC32::FSUB,  PPC32::FMUL,  PPC32::FDIV },   // Double
      };

      unsigned Opcode = OpcodeTab[Ty != Type::FloatTy][OperatorClass];
      unsigned Op0Reg = getReg(Op0C, BB, IP);
      unsigned Op1Reg = getReg(Op1, BB, IP);
      BuildMI(*BB, IP, Opcode, 2, DestReg).addReg(Op0Reg).addReg(Op1Reg);
      return;
    }

  // General case.
  static const unsigned OpcodeTab[] = {
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
  static const unsigned OpcodeTab[] = {
    PPC32::ADD, PPC32::SUB, PPC32::AND, PPC32::OR, PPC32::XOR
  };
  static const unsigned ImmOpcodeTab[] = {
    PPC32::ADDI, PPC32::SUBI, PPC32::ANDIo, PPC32::ORI, PPC32::XORI
  };
  static const unsigned RImmOpcodeTab[] = {
    PPC32::ADDI, PPC32::SUBFIC, PPC32::ANDIo, PPC32::ORI, PPC32::XORI
  };

  // Otherwise, code generate the full operation with a constant.
  static const unsigned BottomTab[] = {
    PPC32::ADDC, PPC32::SUBC, PPC32::AND, PPC32::OR, PPC32::XOR
  };
  static const unsigned TopTab[] = {
    PPC32::ADDE, PPC32::SUBFE, PPC32::AND, PPC32::OR, PPC32::XOR
  };
  
  if (Class == cFP32 || Class == cFP64) {
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
          BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(TmpReg).addImm(4)
            .addImm(31).addImm(31);
          return;
        }
  }

  // Special case: op <const int>, Reg
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op0)) {
    // sub 0, X -> subfic
    if (OperatorClass == 1 && canUseAsImmediateForOpcode(CI, 0)) {
      unsigned Op1r = getReg(Op1, MBB, IP);
      int imm = CI->getRawValue() & 0xFFFF;

      if (Class == cLong) {
        BuildMI(*MBB, IP, PPC32::SUBFIC, 2, DestReg+1).addReg(Op1r+1)
          .addSImm(imm);
        BuildMI(*MBB, IP, PPC32::SUBFZE, 1, DestReg).addReg(Op1r);
      } else {
        BuildMI(*MBB, IP, PPC32::SUBFIC, 2, DestReg).addReg(Op1r).addSImm(imm);
      }
      return;
    }
    
    // If it is easy to do, swap the operands and emit an immediate op
    if (Class != cLong && OperatorClass != 1 && 
        canUseAsImmediateForOpcode(CI, OperatorClass)) {
      unsigned Op1r = getReg(Op1, MBB, IP);
      int imm = CI->getRawValue() & 0xFFFF;
    
      if (OperatorClass < 2)
        BuildMI(*MBB, IP, RImmOpcodeTab[OperatorClass], 2, DestReg).addReg(Op1r)
          .addSImm(imm);
      else
        BuildMI(*MBB, IP, RImmOpcodeTab[OperatorClass], 2, DestReg).addReg(Op1r)
          .addZImm(imm);
      return;
    }
  }

  // Special case: op Reg, <const int>
  if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
    unsigned Op0r = getReg(Op0, MBB, IP);

    // xor X, -1 -> not X
    if (OperatorClass == 4 && Op1C->isAllOnesValue()) {
      BuildMI(*MBB, IP, PPC32::NOR, 2, DestReg).addReg(Op0r).addReg(Op0r);
      if (Class == cLong)  // Invert the low part too
        BuildMI(*MBB, IP, PPC32::NOR, 2, DestReg+1).addReg(Op0r+1)
          .addReg(Op0r+1);
      return;
    }
    
    if (Class != cLong) {
      if (canUseAsImmediateForOpcode(Op1C, OperatorClass)) {
        int immediate = Op1C->getRawValue() & 0xFFFF;
        
        if (OperatorClass < 2)
          BuildMI(*MBB, IP, ImmOpcodeTab[OperatorClass], 2,DestReg).addReg(Op0r)
            .addSImm(immediate);
        else
          BuildMI(*MBB, IP, ImmOpcodeTab[OperatorClass], 2,DestReg).addReg(Op0r)
            .addZImm(immediate);
      } else {
        unsigned Op1r = getReg(Op1, MBB, IP);
        BuildMI(*MBB, IP, OpcodeTab[OperatorClass], 2, DestReg).addReg(Op0r)
          .addReg(Op1r);
      }
      return;
    }

    unsigned Op1r = getReg(Op1, MBB, IP);

    BuildMI(*MBB, IP, BottomTab[OperatorClass], 2, DestReg+1).addReg(Op0r+1)
      .addReg(Op1r+1);
    BuildMI(*MBB, IP, TopTab[OperatorClass], 2, DestReg).addReg(Op0r)
      .addReg(Op1r);
    return;
  }
  
  // We couldn't generate an immediate variant of the op, load both halves into
  // registers and emit the appropriate opcode.
  unsigned Op0r = getReg(Op0, MBB, IP);
  unsigned Op1r = getReg(Op1, MBB, IP);

  if (Class != cLong) {
    unsigned Opcode = OpcodeTab[OperatorClass];
    BuildMI(*MBB, IP, Opcode, 2, DestReg).addReg(Op0r).addReg(Op1r);
  } else {
    BuildMI(*MBB, IP, BottomTab[OperatorClass], 2, DestReg+1).addReg(Op0r+1)
      .addReg(Op1r+1);
    BuildMI(*MBB, IP, TopTab[OperatorClass], 2, DestReg).addReg(Op0r)
      .addReg(Op1r);
  }
  return;
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
  return Count;
}

/// doMultiply - Emit appropriate instructions to multiply together the
/// Values Op0 and Op1, and put the result in DestReg.
///
void ISel::doMultiply(MachineBasicBlock *MBB,
                      MachineBasicBlock::iterator IP,
                      unsigned DestReg, Value *Op0, Value *Op1) {
  unsigned Class0 = getClass(Op0->getType());
  unsigned Class1 = getClass(Op1->getType());
  
  unsigned Op0r = getReg(Op0, MBB, IP);
  unsigned Op1r = getReg(Op1, MBB, IP);
  
  // 64 x 64 -> 64
  if (Class0 == cLong && Class1 == cLong) {
    unsigned Tmp1 = makeAnotherReg(Type::IntTy);
    unsigned Tmp2 = makeAnotherReg(Type::IntTy);
    unsigned Tmp3 = makeAnotherReg(Type::IntTy);
    unsigned Tmp4 = makeAnotherReg(Type::IntTy);
    BuildMI(*MBB, IP, PPC32::MULHWU, 2, Tmp1).addReg(Op0r+1).addReg(Op1r+1);
    BuildMI(*MBB, IP, PPC32::MULLW, 2, DestReg+1).addReg(Op0r+1).addReg(Op1r+1);
    BuildMI(*MBB, IP, PPC32::MULLW, 2, Tmp2).addReg(Op0r+1).addReg(Op1r);
    BuildMI(*MBB, IP, PPC32::ADD, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);
    BuildMI(*MBB, IP, PPC32::MULLW, 2, Tmp4).addReg(Op0r).addReg(Op1r+1);
    BuildMI(*MBB, IP, PPC32::ADD, 2, DestReg).addReg(Tmp3).addReg(Tmp4);
    return;
  }
  
  // 64 x 32 or less, promote 32 to 64 and do a 64 x 64
  if (Class0 == cLong && Class1 <= cInt) {
    unsigned Tmp0 = makeAnotherReg(Type::IntTy);
    unsigned Tmp1 = makeAnotherReg(Type::IntTy);
    unsigned Tmp2 = makeAnotherReg(Type::IntTy);
    unsigned Tmp3 = makeAnotherReg(Type::IntTy);
    unsigned Tmp4 = makeAnotherReg(Type::IntTy);
    if (Op1->getType()->isSigned())
      BuildMI(*MBB, IP, PPC32::SRAWI, 2, Tmp0).addReg(Op1r).addImm(31);
    else
      BuildMI(*MBB, IP, PPC32::LI, 2, Tmp0).addSImm(0);
    BuildMI(*MBB, IP, PPC32::MULHWU, 2, Tmp1).addReg(Op0r+1).addReg(Op1r);
    BuildMI(*MBB, IP, PPC32::MULLW, 2, DestReg+1).addReg(Op0r+1).addReg(Op1r);
    BuildMI(*MBB, IP, PPC32::MULLW, 2, Tmp2).addReg(Op0r+1).addReg(Tmp0);
    BuildMI(*MBB, IP, PPC32::ADD, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);
    BuildMI(*MBB, IP, PPC32::MULLW, 2, Tmp4).addReg(Op0r).addReg(Op1r);
    BuildMI(*MBB, IP, PPC32::ADD, 2, DestReg).addReg(Tmp3).addReg(Tmp4);
    return;
  }
  
  // 32 x 32 -> 32
  if (Class0 <= cInt && Class1 <= cInt) {
    BuildMI(*MBB, IP, PPC32::MULLW, 2, DestReg).addReg(Op0r).addReg(Op1r);
    return;
  }
  
  assert(0 && "doMultiply cannot operate on unknown type!");
}

/// doMultiplyConst - This method will multiply the value in Op0 by the
/// value of the ContantInt *CI
void ISel::doMultiplyConst(MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator IP,
                           unsigned DestReg, Value *Op0, ConstantInt *CI) {
  unsigned Class = getClass(Op0->getType());

  // Mul op0, 0 ==> 0
  if (CI->isNullValue()) {
    BuildMI(*MBB, IP, PPC32::LI, 1, DestReg).addSImm(0);
    if (Class == cLong)
      BuildMI(*MBB, IP, PPC32::LI, 1, DestReg+1).addSImm(0);
    return;
  }
  
  // Mul op0, 1 ==> op0
  if (CI->equalsInt(1)) {
    unsigned Op0r = getReg(Op0, MBB, IP);
    BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(Op0r).addReg(Op0r);
    if (Class == cLong)
      BuildMI(*MBB, IP, PPC32::OR, 2, DestReg+1).addReg(Op0r+1).addReg(Op0r+1);
    return;
  }

  // If the element size is exactly a power of 2, use a shift to get it.
  if (unsigned Shift = ExactLog2(CI->getRawValue())) {
    ConstantUInt *ShiftCI = ConstantUInt::get(Type::UByteTy, Shift);
    emitShiftOperation(MBB, IP, Op0, ShiftCI, true, Op0->getType(), DestReg);
    return;
  }
  
  // If 32 bits or less and immediate is in right range, emit mul by immediate
  if (Class == cByte || Class == cShort || Class == cInt) {
    if (canUseAsImmediateForOpcode(CI, 0)) {
      unsigned Op0r = getReg(Op0, MBB, IP);
      unsigned imm = CI->getRawValue() & 0xFFFF;
      BuildMI(*MBB, IP, PPC32::MULLI, 2, DestReg).addReg(Op0r).addSImm(imm);
      return;
    }
  }
  
  doMultiply(MBB, IP, DestReg, Op0, CI);
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
  TypeClass Class = getClass(Op0->getType());

  switch (Class) {
  case cByte:
  case cShort:
  case cInt:
  case cLong:
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
      doMultiplyConst(MBB, IP, DestReg, Op0, CI);
    } else {
      doMultiply(MBB, IP, DestReg, Op0, Op1);
    }
    return;
  case cFP32:
  case cFP64:
    emitBinaryFPOperation(MBB, IP, Op0, Op1, 2, DestReg);
    return;
    break;
  }
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
  emitDivRemOperation(BB, IP, Op0, Op1, I.getOpcode() == Instruction::Div,
                      ResultReg);
}

void ISel::emitDivRemOperation(MachineBasicBlock *BB,
                               MachineBasicBlock::iterator IP,
                               Value *Op0, Value *Op1, bool isDiv,
                               unsigned ResultReg) {
  const Type *Ty = Op0->getType();
  unsigned Class = getClass(Ty);
  switch (Class) {
  case cFP32:
    if (isDiv) {
      // Floating point divide...
      emitBinaryFPOperation(BB, IP, Op0, Op1, 3, ResultReg);
      return;
    } else {
      // Floating point remainder via fmodf(float x, float y);
      unsigned Op0Reg = getReg(Op0, BB, IP);
      unsigned Op1Reg = getReg(Op1, BB, IP);
      MachineInstr *TheCall =
        BuildMI(PPC32::CALLpcrel, 1).addGlobalAddress(fmodfFn, true);
      std::vector<ValueRecord> Args;
      Args.push_back(ValueRecord(Op0Reg, Type::FloatTy));
      Args.push_back(ValueRecord(Op1Reg, Type::FloatTy));
      doCall(ValueRecord(ResultReg, Type::FloatTy), TheCall, Args, false);
      TM.CalledFunctions.insert(fmodfFn);
    }
    return;
  case cFP64:
    if (isDiv) {
      // Floating point divide...
      emitBinaryFPOperation(BB, IP, Op0, Op1, 3, ResultReg);
      return;
    } else {               
      // Floating point remainder via fmod(double x, double y);
      unsigned Op0Reg = getReg(Op0, BB, IP);
      unsigned Op1Reg = getReg(Op1, BB, IP);
      MachineInstr *TheCall =
        BuildMI(PPC32::CALLpcrel, 1).addGlobalAddress(fmodFn, true);
      std::vector<ValueRecord> Args;
      Args.push_back(ValueRecord(Op0Reg, Type::DoubleTy));
      Args.push_back(ValueRecord(Op1Reg, Type::DoubleTy));
      doCall(ValueRecord(ResultReg, Type::DoubleTy), TheCall, Args, false);
      TM.CalledFunctions.insert(fmodFn);
    }
    return;
  case cLong: {
    static Function* const Funcs[] =
      { __moddi3Fn, __divdi3Fn, __umoddi3Fn, __udivdi3Fn };
    unsigned Op0Reg = getReg(Op0, BB, IP);
    unsigned Op1Reg = getReg(Op1, BB, IP);
    unsigned NameIdx = Ty->isUnsigned()*2 + isDiv;
    MachineInstr *TheCall =
      BuildMI(PPC32::CALLpcrel, 1).addGlobalAddress(Funcs[NameIdx], true);

    std::vector<ValueRecord> Args;
    Args.push_back(ValueRecord(Op0Reg, Type::LongTy));
    Args.push_back(ValueRecord(Op1Reg, Type::LongTy));
    doCall(ValueRecord(ResultReg, Type::LongTy), TheCall, Args, false);
    TM.CalledFunctions.insert(Funcs[NameIdx]);
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

      unsigned log2V = ExactLog2(V);
      if (log2V != 0 && Ty->isSigned()) {
        unsigned Op0Reg = getReg(Op0, BB, IP);
        unsigned TmpReg = makeAnotherReg(Op0->getType());
        
        BuildMI(*BB, IP, PPC32::SRAWI, 2, TmpReg).addReg(Op0Reg).addImm(log2V);
        BuildMI(*BB, IP, PPC32::ADDZE, 1, ResultReg).addReg(TmpReg);
        return;
      }
    }

  unsigned Op0Reg = getReg(Op0, BB, IP);
  unsigned Op1Reg = getReg(Op1, BB, IP);
  unsigned Opcode = Ty->isSigned() ? PPC32::DIVW : PPC32::DIVWU;
  
  if (isDiv) {
    BuildMI(*BB, IP, Opcode, 2, ResultReg).addReg(Op0Reg).addReg(Op1Reg);
  } else { // Remainder
    unsigned TmpReg1 = makeAnotherReg(Op0->getType());
    unsigned TmpReg2 = makeAnotherReg(Op0->getType());
    
    BuildMI(*BB, IP, Opcode, 2, TmpReg1).addReg(Op0Reg).addReg(Op1Reg);
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
  MachineBasicBlock::iterator IP = BB->end();
  emitShiftOperation(BB, IP, I.getOperand(0), I.getOperand(1),
                     I.getOpcode() == Instruction::Shl, I.getType(),
                     getReg(I));
}

/// emitShiftOperation - Common code shared between visitShiftInst and
/// constant expression support.
///
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
          BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg)
            .addImm(Amount).addImm(0).addImm(31-Amount);
          BuildMI(*MBB, IP, PPC32::RLWIMI, 5).addReg(DestReg).addReg(SrcReg+1)
            .addImm(Amount).addImm(32-Amount).addImm(31);
          BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg+1).addReg(SrcReg+1)
            .addImm(Amount).addImm(0).addImm(31-Amount);
        } else {
          // FIXME: RLWIMI is a use-and-def of DestReg, but that violates SSA
          BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg+1).addReg(SrcReg+1)
            .addImm(32-Amount).addImm(Amount).addImm(31);
          BuildMI(*MBB, IP, PPC32::RLWIMI, 5).addReg(DestReg+1).addReg(SrcReg)
            .addImm(32-Amount).addImm(0).addImm(Amount-1);
          BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg)
            .addImm(32-Amount).addImm(Amount).addImm(31);
        }
      } else {                 // Shifting more than 32 bits
        Amount -= 32;
        if (isLeftShift) {
          if (Amount != 0) {
            BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg+1)
              .addImm(Amount).addImm(0).addImm(31-Amount);
          } else {
            BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(SrcReg+1)
              .addReg(SrcReg+1);
          }
          BuildMI(*MBB, IP, PPC32::LI, 1, DestReg+1).addSImm(0);
        } else {
          if (Amount != 0) {
            if (isSigned)
              BuildMI(*MBB, IP, PPC32::SRAWI, 2, DestReg+1).addReg(SrcReg)
                .addImm(Amount);
            else
              BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg+1).addReg(SrcReg)
                .addImm(32-Amount).addImm(Amount).addImm(31);
          } else {
            BuildMI(*MBB, IP, PPC32::OR, 2, DestReg+1).addReg(SrcReg)
              .addReg(SrcReg);
          }
          BuildMI(*MBB, IP,PPC32::LI, 1, DestReg).addSImm(0);
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
        BuildMI(*MBB, IP, PPC32::SUBFIC, 2, TmpReg1).addReg(ShiftAmountReg)
          .addSImm(32);
        BuildMI(*MBB, IP, PPC32::SLW, 2, TmpReg2).addReg(SrcReg)
          .addReg(ShiftAmountReg);
        BuildMI(*MBB, IP, PPC32::SRW, 2, TmpReg3).addReg(SrcReg+1)
          .addReg(TmpReg1);
        BuildMI(*MBB, IP, PPC32::OR, 2,TmpReg4).addReg(TmpReg2).addReg(TmpReg3);
        BuildMI(*MBB, IP, PPC32::ADDI, 2, TmpReg5).addReg(ShiftAmountReg)
          .addSImm(-32);
        BuildMI(*MBB, IP, PPC32::SLW, 2, TmpReg6).addReg(SrcReg+1)
          .addReg(TmpReg5);
        BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(TmpReg4)
          .addReg(TmpReg6);
        BuildMI(*MBB, IP, PPC32::SLW, 2, DestReg+1).addReg(SrcReg+1)
          .addReg(ShiftAmountReg);
      } else {
        if (isSigned) {
          // FIXME: Unimplemented
          // Page C-3 of the PowerPC 32bit Programming Environments Manual
          std::cerr << "ERROR: Unimplemented: signed right shift\n";
          abort();
        } else {
          BuildMI(*MBB, IP, PPC32::SUBFIC, 2, TmpReg1).addReg(ShiftAmountReg)
            .addSImm(32);
          BuildMI(*MBB, IP, PPC32::SRW, 2, TmpReg2).addReg(SrcReg+1)
            .addReg(ShiftAmountReg);
          BuildMI(*MBB, IP, PPC32::SLW, 2, TmpReg3).addReg(SrcReg)
            .addReg(TmpReg1);
          BuildMI(*MBB, IP, PPC32::OR, 2, TmpReg4).addReg(TmpReg2)
            .addReg(TmpReg3);
          BuildMI(*MBB, IP, PPC32::ADDI, 2, TmpReg5).addReg(ShiftAmountReg)
            .addSImm(-32);
          BuildMI(*MBB, IP, PPC32::SRW, 2, TmpReg6).addReg(SrcReg)
            .addReg(TmpReg5);
          BuildMI(*MBB, IP, PPC32::OR, 2, DestReg+1).addReg(TmpReg4)
            .addReg(TmpReg6);
          BuildMI(*MBB, IP, PPC32::SRW, 2, DestReg).addReg(SrcReg)
            .addReg(ShiftAmountReg);
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
      BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg)
        .addImm(Amount).addImm(0).addImm(31-Amount);
    } else {
      if (isSigned) {
        BuildMI(*MBB, IP, PPC32::SRAWI,2,DestReg).addReg(SrcReg).addImm(Amount);
      } else {
        BuildMI(*MBB, IP, PPC32::RLWINM, 4, DestReg).addReg(SrcReg)
          .addImm(32-Amount).addImm(Amount).addImm(31);
      }
    }
  } else {                  // The shift amount is non-constant.
    unsigned ShiftAmountReg = getReg (ShiftAmount, MBB, IP);

    if (isLeftShift) {
      BuildMI(*MBB, IP, PPC32::SLW, 2, DestReg).addReg(SrcReg)
        .addReg(ShiftAmountReg);
    } else {
      BuildMI(*MBB, IP, isSigned ? PPC32::SRAW : PPC32::SRW, 2, DestReg)
        .addReg(SrcReg).addReg(ShiftAmountReg);
    }
  }
}


/// visitLoadInst - Implement LLVM load instructions
///
void ISel::visitLoadInst(LoadInst &I) {
  static const unsigned Opcodes[] = { 
    PPC32::LBZ, PPC32::LHZ, PPC32::LWZ, PPC32::LFS 
  };

  unsigned Class = getClassB(I.getType());
  unsigned Opcode = Opcodes[Class];
  if (I.getType() == Type::DoubleTy) Opcode = PPC32::LFD;
  if (Class == cShort && I.getType()->isSigned()) Opcode = PPC32::LHA;
  unsigned DestReg = getReg(I);

  if (AllocaInst *AI = dyn_castFixedAlloca(I.getOperand(0))) {
    unsigned FI = getFixedSizedAllocaFI(AI);
    if (Class == cLong) {
      addFrameReference(BuildMI(BB, PPC32::LWZ, 2, DestReg), FI);
      addFrameReference(BuildMI(BB, PPC32::LWZ, 2, DestReg+1), FI, 4);
    } else if (Class == cByte && I.getType()->isSigned()) {
      unsigned TmpReg = makeAnotherReg(I.getType());
      addFrameReference(BuildMI(BB, Opcode, 2, TmpReg), FI);
      BuildMI(BB, PPC32::EXTSB, 1, DestReg).addReg(TmpReg);
    } else {
      addFrameReference(BuildMI(BB, Opcode, 2, DestReg), FI);
    }
  } else {
    unsigned SrcAddrReg = getReg(I.getOperand(0));
    
    if (Class == cLong) {
      BuildMI(BB, PPC32::LWZ, 2, DestReg).addSImm(0).addReg(SrcAddrReg);
      BuildMI(BB, PPC32::LWZ, 2, DestReg+1).addSImm(4).addReg(SrcAddrReg);
    } else if (Class == cByte && I.getType()->isSigned()) {
      unsigned TmpReg = makeAnotherReg(I.getType());
      BuildMI(BB, Opcode, 2, TmpReg).addSImm(0).addReg(SrcAddrReg);
      BuildMI(BB, PPC32::EXTSB, 1, DestReg).addReg(TmpReg);
    } else {
      BuildMI(BB, Opcode, 2, DestReg).addSImm(0).addReg(SrcAddrReg);
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
    BuildMI(BB, PPC32::STW, 3).addReg(ValReg).addSImm(0).addReg(AddressReg);
    BuildMI(BB, PPC32::STW, 3).addReg(ValReg+1).addSImm(4).addReg(AddressReg);
    return;
  }

  static const unsigned Opcodes[] = {
    PPC32::STB, PPC32::STH, PPC32::STW, PPC32::STFS
  };
  unsigned Opcode = Opcodes[Class];
  if (ValTy == Type::DoubleTy) Opcode = PPC32::STFD;
  BuildMI(BB, Opcode, 3).addReg(ValReg).addSImm(0).addReg(AddressReg);
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
void ISel::emitCastOperation(MachineBasicBlock *MBB,
                             MachineBasicBlock::iterator IP,
                             Value *Src, const Type *DestTy,
                             unsigned DestReg) {
  const Type *SrcTy = Src->getType();
  unsigned SrcClass = getClassB(SrcTy);
  unsigned DestClass = getClassB(DestTy);
  unsigned SrcReg = getReg(Src, MBB, IP);

  // Implement casts to bool by using compare on the operand followed by set if
  // not zero on the result.
  if (DestTy == Type::BoolTy) {
    switch (SrcClass) {
    case cByte:
    case cShort:
    case cInt: {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, PPC32::ADDIC, 2, TmpReg).addReg(SrcReg).addSImm(-1);
      BuildMI(*MBB, IP, PPC32::SUBFE, 2, DestReg).addReg(TmpReg).addReg(SrcReg);
      break;
    }
    case cLong: {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      unsigned SrcReg2 = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, PPC32::OR, 2, SrcReg2).addReg(SrcReg).addReg(SrcReg+1);
      BuildMI(*MBB, IP, PPC32::ADDIC, 2, TmpReg).addReg(SrcReg2).addSImm(-1);
      BuildMI(*MBB, IP, PPC32::SUBFE, 2, DestReg).addReg(TmpReg)
        .addReg(SrcReg2);
      break;
    }
    case cFP32:
    case cFP64:
      // FSEL perhaps?
      std::cerr << "ERROR: Cast fp-to-bool not implemented!\n";
      abort();
    }
    return;
  }

  // Implement casts between values of the same type class (as determined by
  // getClass) by using a register-to-register move.
  if (SrcClass == DestClass) {
    if (SrcClass <= cInt) {
      BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
    } else if (SrcClass == cFP32 || SrcClass == cFP64) {
      BuildMI(*MBB, IP, PPC32::FMR, 1, DestReg).addReg(SrcReg);
    } else if (SrcClass == cLong) {
      BuildMI(*MBB, IP, PPC32::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
      BuildMI(*MBB, IP, PPC32::OR, 2, DestReg+1).addReg(SrcReg+1)
        .addReg(SrcReg+1);
    } else {
      assert(0 && "Cannot handle this type of cast instruction!");
      abort();
    }
    return;
  }
  
  // Handle cast of Float -> Double
  if (SrcClass == cFP32 && DestClass == cFP64) {
    BuildMI(*MBB, IP, PPC32::FMR, 1, DestReg).addReg(SrcReg);
    return;
  }
  
  // Handle cast of Double -> Float
  if (SrcClass == cFP64 && DestClass == cFP32) {
    BuildMI(*MBB, IP, PPC32::FRSP, 1, DestReg).addReg(SrcReg);
    return;
  }
  
  // Handle cast of SMALLER int to LARGER int using a move with sign extension
  // or zero extension, depending on whether the source type was signed.
  if (SrcClass <= cInt && (DestClass <= cInt || DestClass == cLong) &&
      SrcClass < DestClass) {
    bool isLong = DestClass == cLong;
    if (isLong) {
      DestClass = cInt;
      ++DestReg;
    }
    
    bool isUnsigned = DestTy->isUnsigned() || DestTy == Type::BoolTy;
    BuildMI(*BB, IP, PPC32::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);

    if (isLong) {  // Handle upper 32 bits as appropriate...
      --DestReg;
      if (isUnsigned)     // Zero out top bits...
        BuildMI(*BB, IP, PPC32::LI, 1, DestReg).addSImm(0);
      else                // Sign extend bottom half...
        BuildMI(*BB, IP, PPC32::SRAWI, 2, DestReg).addReg(SrcReg).addImm(31);
    }
    return;
  }

  // Special case long -> int ...
  if (SrcClass == cLong && DestClass == cInt) {
    BuildMI(*BB, IP, PPC32::OR, 2, DestReg).addReg(SrcReg+1).addReg(SrcReg+1);
    return;
  }
  
  // Handle cast of LARGER int to SMALLER int with a clear or sign extend
  if ((SrcClass <= cInt || SrcClass == cLong) && DestClass <= cInt && 
      SrcClass > DestClass) {
    bool isUnsigned = DestTy->isUnsigned() || DestTy == Type::BoolTy;
    unsigned source = (SrcClass == cLong) ? SrcReg+1 : SrcReg;
    
    if (isUnsigned) {
      unsigned shift = (DestClass == cByte) ? 24 : 16;
      BuildMI(*BB, IP, PPC32::RLWINM, 4, DestReg).addReg(source).addZImm(0)
        .addImm(shift).addImm(31);
    } else {
      BuildMI(*BB, IP, (DestClass == cByte) ? PPC32::EXTSB : PPC32::EXTSH, 1, 
              DestReg).addReg(source);
    }
    return;
  }

  // Handle casts from integer to floating point now...
  if (DestClass == cFP32 || DestClass == cFP64) {

    // Emit a library call for long to float conversion
    if (SrcClass == cLong) {
      std::vector<ValueRecord> Args;
      Args.push_back(ValueRecord(SrcReg, SrcTy));
      Function *floatFn = (DestClass == cFP32) ? __floatdisfFn : __floatdidfFn;
      MachineInstr *TheCall =
        BuildMI(PPC32::CALLpcrel, 1).addGlobalAddress(floatFn, true);
      doCall(ValueRecord(DestReg, DestTy), TheCall, Args, false);
      TM.CalledFunctions.insert(floatFn);
      return;
    }
    
    // Make sure we're dealing with a full 32 bits
    unsigned TmpReg = makeAnotherReg(Type::IntTy);
    promote32(TmpReg, ValueRecord(SrcReg, SrcTy));

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
      BuildMI(*BB, IP, PPC32::LIS, 1, constantHi).addSImm(0x4330);
      BuildMI(*BB, IP, PPC32::LI, 1, constantLo).addSImm(0);
      addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantHi), 
                        ConstantFrameIndex);
      addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantLo), 
                        ConstantFrameIndex, 4);
      addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantHi), 
                        ValueFrameIdx);
      addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(SrcReg), 
                        ValueFrameIdx, 4);
      addFrameReference(BuildMI(*BB, IP, PPC32::LFD, 2, ConstF), 
                        ConstantFrameIndex);
      addFrameReference(BuildMI(*BB, IP, PPC32::LFD, 2, TempF), ValueFrameIdx);
      BuildMI(*BB, IP, PPC32::FSUB, 2, DestReg).addReg(TempF).addReg(ConstF);
    } else {
      unsigned TempLo = makeAnotherReg(Type::IntTy);
      BuildMI(*BB, IP, PPC32::LIS, 1, constantHi).addSImm(0x4330);
      BuildMI(*BB, IP, PPC32::LIS, 1, constantLo).addSImm(0x8000);
      addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantHi), 
                        ConstantFrameIndex);
      addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantLo), 
                        ConstantFrameIndex, 4);
      addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(constantHi), 
                        ValueFrameIdx);
      BuildMI(*BB, IP, PPC32::XORIS, 2, TempLo).addReg(SrcReg).addImm(0x8000);
      addFrameReference(BuildMI(*BB, IP, PPC32::STW, 3).addReg(TempLo), 
                        ValueFrameIdx, 4);
      addFrameReference(BuildMI(*BB, IP, PPC32::LFD, 2, ConstF), 
                        ConstantFrameIndex);
      addFrameReference(BuildMI(*BB, IP, PPC32::LFD, 2, TempF), ValueFrameIdx);
      BuildMI(*BB, IP, PPC32::FSUB, 2, DestReg).addReg(TempF ).addReg(ConstF);
    }
    return;
  }

  // Handle casts from floating point to integer now...
  if (SrcClass == cFP32 || SrcClass == cFP64) {
    // emit library call
    if (DestClass == cLong) {
      std::vector<ValueRecord> Args;
      Args.push_back(ValueRecord(SrcReg, SrcTy));
      Function *floatFn = (DestClass == cFP32) ? __fixsfdiFn : __fixdfdiFn;
      MachineInstr *TheCall =
        BuildMI(PPC32::CALLpcrel, 1).addGlobalAddress(floatFn, true);
      doCall(ValueRecord(DestReg, DestTy), TheCall, Args, false);
      TM.CalledFunctions.insert(floatFn);
      return;
    }

    int ValueFrameIdx =
      F->getFrameInfo()->CreateStackObject(SrcTy, TM.getTargetData());

    if (DestTy->isSigned()) {
      unsigned LoadOp = (DestClass == cShort) ? PPC32::LHA : PPC32::LWZ;
      unsigned TempReg = makeAnotherReg(Type::DoubleTy);
      
      // Convert to integer in the FP reg and store it to a stack slot
      BuildMI(*BB, IP, PPC32::FCTIWZ, 1, TempReg).addReg(SrcReg);
      addFrameReference(BuildMI(*BB, IP, PPC32::STFD, 3)
                          .addReg(TempReg), ValueFrameIdx);
      
      // There is no load signed byte opcode, so we must emit a sign extend
      if (DestClass == cByte) {
        unsigned TempReg2 = makeAnotherReg(DestTy);
        addFrameReference(BuildMI(*BB, IP, LoadOp, 2, TempReg2), 
                          ValueFrameIdx, 4);
        BuildMI(*MBB, IP, PPC32::EXTSB, DestReg).addReg(TempReg2);
      } else {
        addFrameReference(BuildMI(*BB, IP, LoadOp, 2, DestReg), 
                          ValueFrameIdx, 4);
      }
    } else {
      std::cerr << "ERROR: Cast fp-to-unsigned not implemented!\n";
      abort();
    }
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
  switch (I.getArgType()->getTypeID()) {
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
  BuildMI(BB, PPC32::ADDI, 2, DestReg).addReg(VAList).addSImm(Size);
}

void ISel::visitVAArgInst(VAArgInst &I) {
  unsigned VAList = getReg(I.getOperand(0));
  unsigned DestReg = getReg(I);

  switch (I.getType()->getTypeID()) {
  default:
    std::cerr << I;
    assert(0 && "Error: bad type for va_next instruction!");
    return;
  case Type::PointerTyID:
  case Type::UIntTyID:
  case Type::IntTyID:
    BuildMI(BB, PPC32::LWZ, 2, DestReg).addSImm(0).addReg(VAList);
    break;
  case Type::ULongTyID:
  case Type::LongTyID:
    BuildMI(BB, PPC32::LWZ, 2, DestReg).addSImm(0).addReg(VAList);
    BuildMI(BB, PPC32::LWZ, 2, DestReg+1).addSImm(4).addReg(VAList);
    break;
  case Type::DoubleTyID:
    BuildMI(BB, PPC32::LFD, 2, DestReg).addSImm(0).addReg(VAList);
    break;
  }
}

/// visitGetElementPtrInst - instruction-select GEP instructions
///
void ISel::visitGetElementPtrInst(GetElementPtrInst &I) {
  unsigned outputReg = getReg(I);
  emitGEPOperation(BB, BB->end(), I.getOperand(0), I.op_begin()+1, I.op_end(), 
                   outputReg);
}

/// emitGEPOperation - Common code shared between visitGetElementPtrInst and
/// constant expression GEP support.
///
void ISel::emitGEPOperation(MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator IP,
                            Value *Src, User::op_iterator IdxBegin,
                            User::op_iterator IdxEnd, unsigned TargetReg) {
  const TargetData &TD = TM.getTargetData();
  const Type *Ty = Src->getType();
  unsigned basePtrReg = getReg(Src, MBB, IP);
  int64_t constValue = 0;
  bool anyCombined = false;
  
  // Record the operations to emit the GEP in a vector so that we can emit them
  // after having analyzed the entire instruction.
  std::vector<CollapsedGepOp*> ops;
  
  // GEPs have zero or more indices; we must perform a struct access
  // or array access for each one.
  for (GetElementPtrInst::op_iterator oi = IdxBegin, oe = IdxEnd; oi != oe;
       ++oi) {
    Value *idx = *oi;
    if (const StructType *StTy = dyn_cast<StructType>(Ty)) {
      // It's a struct access.  idx is the index into the structure,
      // which names the field. Use the TargetData structure to
      // pick out what the layout of the structure is in memory.
      // Use the (constant) structure index's value to find the
      // right byte offset from the StructLayout class's list of
      // structure member offsets.
      unsigned fieldIndex = cast<ConstantUInt>(idx)->getValue();
      unsigned memberOffset =
        TD.getStructLayout(StTy)->MemberOffsets[fieldIndex];
      if (constValue != 0) anyCombined = true;

      // StructType member offsets are always constant values.  Add it to the
      // running total.
      constValue += memberOffset;

      // The next type is the member of the structure selected by the
      // index.
      Ty = StTy->getElementType (fieldIndex);
    } else if (const SequentialType *SqTy = dyn_cast<SequentialType> (Ty)) {
      // Many GEP instructions use a [cast (int/uint) to LongTy] as their
      // operand.  Handle this case directly now...
      if (CastInst *CI = dyn_cast<CastInst>(idx))
        if (CI->getOperand(0)->getType() == Type::IntTy ||
            CI->getOperand(0)->getType() == Type::UIntTy)
          idx = CI->getOperand(0);

      // It's an array or pointer access: [ArraySize x ElementType].
      // We want to add basePtrReg to (idxReg * sizeof ElementType). First, we
      // must find the size of the pointed-to type (Not coincidentally, the next
      // type is the type of the elements in the array).
      Ty = SqTy->getElementType();
      unsigned elementSize = TD.getTypeSize(Ty);
      
      if (ConstantInt *C = dyn_cast<ConstantInt>(idx)) {
        if (constValue != 0) anyCombined = true;

        if (ConstantSInt *CS = dyn_cast<ConstantSInt>(C))
          constValue += CS->getValue() * elementSize;
        else if (ConstantUInt *CU = dyn_cast<ConstantUInt>(C))
          constValue += CU->getValue() * elementSize;
        else
          assert(0 && "Invalid ConstantInt GEP index type!");
      } else {
        // Push current gep state to this point as an add
        CollapsedGepOp *addition = 
          new CollapsedGepOp(false, 0, ConstantSInt::get(Type::IntTy,
                constValue));
        ops.push_back(addition);
        
        // Push multiply gep op and reset constant value
        CollapsedGepOp *multiply = 
          new CollapsedGepOp(true, idx, ConstantSInt::get(Type::IntTy, 
                elementSize));
        ops.push_back(multiply);
        
        constValue = 0;
      }
    }
  }
  // Do some statistical accounting
  if (ops.empty())
    ++GEPConsts;

  if (anyCombined)
    ++GEPSplits;
    
  // Emit instructions for all the collapsed ops
  for(std::vector<CollapsedGepOp *>::iterator cgo_i = ops.begin(),
      cgo_e = ops.end(); cgo_i != cgo_e; ++cgo_i) {
    CollapsedGepOp *cgo = *cgo_i;
    unsigned nextBasePtrReg = makeAnotherReg (Type::IntTy);

    if (cgo->isMul) {
      // We know the elementSize is a constant, so we can emit a constant mul
      // and then add it to the current base reg
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      doMultiplyConst(MBB, IP, TmpReg, cgo->index, cgo->size);
      BuildMI(*MBB, IP, PPC32::ADD, 2, nextBasePtrReg).addReg(basePtrReg)
        .addReg(TmpReg);
    } else {
      // Try and generate an immediate addition if possible
      if (cgo->size->isNullValue()) {
        BuildMI(*MBB, IP, PPC32::OR, 2, nextBasePtrReg).addReg(basePtrReg)
          .addReg(basePtrReg);
      } else if (canUseAsImmediateForOpcode(cgo->size, 0)) {
        BuildMI(*MBB, IP, PPC32::ADDI, 2, nextBasePtrReg).addReg(basePtrReg)
          .addSImm(cgo->size->getValue());
      } else {
        unsigned Op1r = getReg(cgo->size, MBB, IP);
        BuildMI(*MBB, IP, PPC32::ADD, 2, nextBasePtrReg).addReg(basePtrReg)
          .addReg(Op1r);
      }
    }

    basePtrReg = nextBasePtrReg;
  }
  // Add the current base register plus any accumulated constant value
  ConstantSInt *remainder = ConstantSInt::get(Type::IntTy, constValue);
  
  // After we have processed all the indices, the result is left in
  // basePtrReg.  Move it to the register where we were expected to
  // put the answer.
  if (remainder->isNullValue()) {
    BuildMI (BB, PPC32::OR, 2, TargetReg).addReg(basePtrReg).addReg(basePtrReg);
  } else if (canUseAsImmediateForOpcode(remainder, 0)) {
    BuildMI(*MBB, IP, PPC32::ADDI, 2, TargetReg).addReg(basePtrReg)
      .addSImm(remainder->getValue());
  } else {
    unsigned Op1r = getReg(remainder, MBB, IP);
    BuildMI(*MBB, IP, PPC32::ADD, 2, TargetReg).addReg(basePtrReg).addReg(Op1r);
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
  
  // TotalSizeReg = mul <numelements>, <TypeSize>
  MachineBasicBlock::iterator MBBI = BB->end();
  ConstantUInt *CUI = ConstantUInt::get(Type::UIntTy, TySize);
  doMultiplyConst(BB, MBBI, TotalSizeReg, I.getArraySize(), CUI);

  // AddedSize = add <TotalSizeReg>, 15
  unsigned AddedSizeReg = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, PPC32::ADDI, 2, AddedSizeReg).addReg(TotalSizeReg).addSImm(15);

  // AlignedSize = and <AddedSize>, ~15
  unsigned AlignedSize = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, PPC32::RLWINM, 4, AlignedSize).addReg(AddedSizeReg).addImm(0)
    .addImm(0).addImm(27);
  
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
    MachineBasicBlock::iterator MBBI = BB->end();
    ConstantUInt *CUI = ConstantUInt::get(Type::UIntTy, AllocSize);
    doMultiplyConst(BB, MBBI, Arg, I.getOperand(0), CUI);
  }

  std::vector<ValueRecord> Args;
  Args.push_back(ValueRecord(Arg, Type::UIntTy));
  MachineInstr *TheCall = 
    BuildMI(PPC32::CALLpcrel, 1).addGlobalAddress(mallocFn, true);
  doCall(ValueRecord(getReg(I), I.getType()), TheCall, Args, false);
  TM.CalledFunctions.insert(mallocFn);
}


/// visitFreeInst - Free instructions are code gen'd to call the free libc
/// function.
///
void ISel::visitFreeInst(FreeInst &I) {
  std::vector<ValueRecord> Args;
  Args.push_back(ValueRecord(I.getOperand(0)));
  MachineInstr *TheCall = 
    BuildMI(PPC32::CALLpcrel, 1).addGlobalAddress(freeFn, true);
  doCall(ValueRecord(0, Type::VoidTy), TheCall, Args, false);
  TM.CalledFunctions.insert(freeFn);
}
   
/// createPPC32SimpleInstructionSelector - This pass converts an LLVM function
/// into a machine code representation is a very simple peep-hole fashion.  The
/// generated code sucks but the implementation is nice and simple.
///
FunctionPass *llvm::createPPCSimpleInstructionSelector(TargetMachine &TM) {
  return new ISel(TM);
}
