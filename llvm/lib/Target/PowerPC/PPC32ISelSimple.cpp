//===-- PPC32ISelSimple.cpp - A simple instruction selector PowerPC32 -----===//
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
#include "PPC32TargetMachine.h"
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
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <vector>
using namespace llvm;

namespace {
  /// TypeClass - Used by the PowerPC backend to group LLVM types by their basic
  /// PPC Representation.
  ///
  enum TypeClass {
    cByte, cShort, cInt, cFP32, cFP64, cLong
  };
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
  if (Ty == Type::BoolTy) return cByte;
  return getClass(Ty);
}

namespace {
  struct PPC32ISel : public FunctionPass, InstVisitor<PPC32ISel> {
    PPC32TargetMachine &TM;
    MachineFunction *F;                 // The function we are compiling into
    MachineBasicBlock *BB;              // The current MBB we are compiling
    int VarArgsFrameIndex;              // FrameIndex for start of varargs area
    
    /// CollapsedGepOp - This struct is for recording the intermediate results 
    /// used to calculate the base, index, and offset of a GEP instruction.
    struct CollapsedGepOp {
      ConstantSInt *offset; // the current offset into the struct/array
      Value *index;         // the index of the array element
      ConstantUInt *size;   // the size of each array element
      CollapsedGepOp(ConstantSInt *o, Value *i, ConstantUInt *s) :
        offset(o), index(i), size(s) {}
    };

    /// FoldedGEP - This struct is for recording the necessary information to 
    /// emit the GEP in a load or store instruction, used by emitGEPOperation.
    struct FoldedGEP {
      unsigned base;
      unsigned index;
      ConstantSInt *offset;
      FoldedGEP() : base(0), index(0), offset(0) {}
      FoldedGEP(unsigned b, unsigned i, ConstantSInt *o) : 
        base(b), index(i), offset(o) {}
    };
    
    /// RlwimiRec - This struct is for recording the arguments to a PowerPC 
    /// rlwimi instruction to be output for a particular Instruction::Or when
    /// we recognize the pattern for rlwimi, starting with a shift or and.
    struct RlwimiRec { 
      Value *Target, *Insert;
      unsigned Shift, MB, ME;
      RlwimiRec() : Target(0), Insert(0), Shift(0), MB(0), ME(0) {}
      RlwimiRec(Value *tgt, Value *ins, unsigned s, unsigned b, unsigned e) :
        Target(tgt), Insert(ins), Shift(s), MB(b), ME(e) {}
    };
    
    // External functions we may use in compiling the Module
    Function *fmodfFn, *fmodFn, *__cmpdi2Fn, *__moddi3Fn, *__divdi3Fn, 
      *__umoddi3Fn,  *__udivdi3Fn, *__fixsfdiFn, *__fixdfdiFn, *__fixunssfdiFn,
      *__fixunsdfdiFn, *__floatdisfFn, *__floatdidfFn, *mallocFn, *freeFn;

    // Mapping between Values and SSA Regs
    std::map<Value*, unsigned> RegMap;

    // MBBMap - Mapping between LLVM BB -> Machine BB
    std::map<const BasicBlock*, MachineBasicBlock*> MBBMap;

    // AllocaMap - Mapping from fixed sized alloca instructions to the
    // FrameIndex for the alloca.
    std::map<AllocaInst*, unsigned> AllocaMap;

    // GEPMap - Mapping between basic blocks and GEP definitions
    std::map<GetElementPtrInst*, FoldedGEP> GEPMap;
    
    // RlwimiMap  - Mapping between BinaryOperand (Or) instructions and info
    // needed to properly emit a rlwimi instruction in its place.
    std::map<Instruction *, RlwimiRec> InsertMap;

    // A rlwimi instruction is the combination of at least three instructions.
    // Keep a vector of instructions to skip around so that we do not try to
    // emit instructions that were folded into a rlwimi.
    std::vector<Instruction *> SkipList;

    // A Reg to hold the base address used for global loads and stores, and a
    // flag to set whether or not we need to emit it for this function.
    unsigned GlobalBaseReg;
    bool GlobalBaseInitialized;
    
    PPC32ISel(TargetMachine &tm):TM(reinterpret_cast<PPC32TargetMachine&>(tm)),
      F(0), BB(0) {}

    bool doInitialization(Module &M) {
      // Add external functions that we may call
      Type *i = Type::IntTy;
      Type *d = Type::DoubleTy;
      Type *f = Type::FloatTy;
      Type *l = Type::LongTy;
      Type *ul = Type::ULongTy;
      Type *voidPtr = PointerType::get(Type::SByteTy);
      // float fmodf(float, float);
      fmodfFn = M.getOrInsertFunction("fmodf", f, f, f, 0);
      // double fmod(double, double);
      fmodFn = M.getOrInsertFunction("fmod", d, d, d, 0);
      // int __cmpdi2(long, long);
      __cmpdi2Fn = M.getOrInsertFunction("__cmpdi2", i, l, l, 0);
      // long __moddi3(long, long);
      __moddi3Fn = M.getOrInsertFunction("__moddi3", l, l, l, 0);
      // long __divdi3(long, long);
      __divdi3Fn = M.getOrInsertFunction("__divdi3", l, l, l, 0);
      // unsigned long __umoddi3(unsigned long, unsigned long);
      __umoddi3Fn = M.getOrInsertFunction("__umoddi3", ul, ul, ul, 0);
      // unsigned long __udivdi3(unsigned long, unsigned long);
      __udivdi3Fn = M.getOrInsertFunction("__udivdi3", ul, ul, ul, 0);
      // long __fixsfdi(float)
      __fixsfdiFn = M.getOrInsertFunction("__fixsfdi", l, f, 0);
      // long __fixdfdi(double)
      __fixdfdiFn = M.getOrInsertFunction("__fixdfdi", l, d, 0);
      // unsigned long __fixunssfdi(float)
      __fixunssfdiFn = M.getOrInsertFunction("__fixunssfdi", ul, f, 0);
      // unsigned long __fixunsdfdi(double)
      __fixunsdfdiFn = M.getOrInsertFunction("__fixunsdfdi", ul, d, 0);
      // float __floatdisf(long)
      __floatdisfFn = M.getOrInsertFunction("__floatdisf", f, l, 0);
      // double __floatdidf(long)
      __floatdidfFn = M.getOrInsertFunction("__floatdidf", d, l, 0);
      // void* malloc(size_t)
      mallocFn = M.getOrInsertFunction("malloc", voidPtr, Type::UIntTy, 0);
      // void free(void*)
      freeFn = M.getOrInsertFunction("free", Type::VoidTy, voidPtr, 0);
      return true;
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

      // Make sure we re-emit a set of the global base reg if necessary
      GlobalBaseInitialized = false;

      // Copy incoming arguments off of the stack...
      LoadArgumentsToVirtualRegs(Fn);

      // Instruction select everything except PHI nodes
      visit(Fn);

      // Select the PHI nodes
      SelectPHINodes();

      GEPMap.clear();
      RegMap.clear();
      MBBMap.clear();
      InsertMap.clear();
      AllocaMap.clear();
      SkipList.clear();
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

    // Control flow operators.
    void visitReturnInst(ReturnInst &RI);
    void visitBranchInst(BranchInst &BI);
    void visitUnreachableInst(UnreachableInst &UI) {}

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

    unsigned ExtendOrClear(MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator IP,
                           Value *Op0);

    /// promote32 - Make a value 32-bits wide, and put it somewhere.
    ///
    void promote32(unsigned targetReg, const ValueRecord &VR);

    /// emitGEPOperation - Common code shared between visitGetElementPtrInst and
    /// constant expression GEP support.
    ///
    void emitGEPOperation(MachineBasicBlock *BB, MachineBasicBlock::iterator IP,
                          GetElementPtrInst *GEPI, bool foldGEP);

    /// emitCastOperation - Common code shared between visitCastInst and
    /// constant expression cast support.
    ///
    void emitCastOperation(MachineBasicBlock *BB,MachineBasicBlock::iterator IP,
                           Value *Src, const Type *DestTy, unsigned TargetReg);


    /// emitBitfieldInsert - return true if we were able to fold the sequence of
    /// instructions into a bitfield insert (rlwimi).
    bool emitBitfieldInsert(User *OpUser, unsigned DestReg);
                                  
    /// emitBitfieldExtract - return true if we were able to fold the sequence
    /// of instructions into a bitfield extract (rlwinm).
    bool emitBitfieldExtract(MachineBasicBlock *MBB, 
                             MachineBasicBlock::iterator IP,
                             User *OpUser, unsigned DestReg);

    /// emitBinaryConstOperation - Used by several functions to emit simple
    /// arithmetic and logical operations with constants on a register rather
    /// than a Value.
    ///
    void emitBinaryConstOperation(MachineBasicBlock *MBB, 
                                  MachineBasicBlock::iterator IP,
                                  unsigned Op0Reg, ConstantInt *Op1, 
                                  unsigned Opcode, unsigned DestReg);

    /// emitSimpleBinaryOperation - Implement simple binary operators for 
    /// integral types.  OperatorClass is one of: 0 for Add, 1 for Sub, 
    /// 2 for And, 3 for Or, 4 for Xor.
    ///
    void emitSimpleBinaryOperation(MachineBasicBlock *BB,
                                   MachineBasicBlock::iterator IP,
                                   BinaryOperator *BO, Value *Op0, Value *Op1,
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
                            const Type *ResultTy, ShiftInst *SI, 
                            unsigned DestReg);
      
    /// emitSelectOperation - Common code shared between visitSelectInst and the
    /// constant expression support.
    ///
    void emitSelectOperation(MachineBasicBlock *MBB,
                             MachineBasicBlock::iterator IP,
                             Value *Cond, Value *TrueVal, Value *FalseVal,
                             unsigned DestReg);

    /// getGlobalBaseReg - Output the instructions required to put the
    /// base address to use for accessing globals into a register.  Returns the
    /// register containing the base address.
    ///
    unsigned getGlobalBaseReg();

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
      assert(dynamic_cast<const PPC32RegisterInfo*>(TM.getRegisterInfo()) &&
             "Current target doesn't have PPC reg info??");
      const PPC32RegisterInfo *PPCRI =
        static_cast<const PPC32RegisterInfo*>(TM.getRegisterInfo());
      if (Ty == Type::LongTy || Ty == Type::ULongTy) {
        const TargetRegisterClass *RC = PPCRI->getRegClassForType(Type::IntTy);
        // Create the upper part
        F->getSSARegMap()->createVirtualRegister(RC);
        // Create the lower part.
        return F->getSSARegMap()->createVirtualRegister(RC)-1;
      }

      // Add the mapping of regnumber => reg class to MachineFunction
      const TargetRegisterClass *RC = PPCRI->getRegClassForType(Ty);
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
    bool canUseAsImmediateForOpcode(ConstantInt *CI, unsigned Opcode,
                                    bool Shifted);

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
unsigned PPC32ISel::getReg(Value *V, MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator IPt) {
  if (Constant *C = dyn_cast<Constant>(V)) {
    unsigned Reg = makeAnotherReg(V->getType());
    copyConstantToRegister(MBB, IPt, C, Reg);
    return Reg;
  } else if (CastInst *CI = dyn_cast<CastInst>(V)) {
    // Do not emit noop casts at all, unless it's a double -> float cast.
    if (getClassB(CI->getType()) == getClassB(CI->getOperand(0)->getType()))
      return getReg(CI->getOperand(0), MBB, IPt);
  } else if (AllocaInst *AI = dyn_castFixedAlloca(V)) {
    unsigned Reg = makeAnotherReg(V->getType());
    unsigned FI = getFixedSizedAllocaFI(AI);
    addFrameReference(BuildMI(*MBB, IPt, PPC::ADDI, 2, Reg), FI, 0, false);
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
/// The shifted argument determines if the immediate is suitable to be used with
/// the PowerPC instructions such as addis which concatenate 16 bits of the
/// immediate with 16 bits of zeroes.
///
bool PPC32ISel::canUseAsImmediateForOpcode(ConstantInt *CI, unsigned Opcode,
                                           bool Shifted) {
  ConstantSInt *Op1Cs;
  ConstantUInt *Op1Cu;

  // For shifted immediates, any value with the low halfword cleared may be used
  if (Shifted) {
    if (((int32_t)CI->getRawValue() & 0x0000FFFF) == 0)
      return true;
    else
      return false;
  }

  // Treat subfic like addi for the purposes of constant validation
  if (Opcode == 5) Opcode = 0;
      
  // addi, subfic, compare, and non-indexed load take SIMM
  bool cond1 = (Opcode < 2)
    && ((int32_t)CI->getRawValue() <= 32767)
    && ((int32_t)CI->getRawValue() >= -32768);

  // ANDIo, ORI, and XORI take unsigned values
  bool cond2 = (Opcode >= 2)
    && (Op1Cs = dyn_cast<ConstantSInt>(CI))
    && (Op1Cs->getValue() >= 0)
    && (Op1Cs->getValue() <= 65535);

  // ANDIo, ORI, and XORI take UIMMs, so they can be larger
  bool cond3 = (Opcode >= 2)
    && (Op1Cu = dyn_cast<ConstantUInt>(CI))
    && (Op1Cu->getValue() <= 65535);

  if (cond1 || cond2 || cond3)
    return true;

  return false;
}

/// getFixedSizedAllocaFI - Return the frame index for a fixed sized alloca
/// that is to be statically allocated with the initial stack frame
/// adjustment.
unsigned PPC32ISel::getFixedSizedAllocaFI(AllocaInst *AI) {
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


/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
unsigned PPC32ISel::getGlobalBaseReg() {
  if (!GlobalBaseInitialized) {
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineBasicBlock &FirstMBB = F->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();
    GlobalBaseReg = makeAnotherReg(Type::IntTy);
    BuildMI(FirstMBB, MBBI, PPC::MovePCtoLR, 0, PPC::LR);
    BuildMI(FirstMBB, MBBI, PPC::MFLR, 1, GlobalBaseReg).addReg(PPC::LR);
    GlobalBaseInitialized = true;
  }
  return GlobalBaseReg;
}

/// copyConstantToRegister - Output the instructions required to put the
/// specified constant into the specified register.
///
void PPC32ISel::copyConstantToRegister(MachineBasicBlock *MBB,
                                       MachineBasicBlock::iterator IP,
                                       Constant *C, unsigned R) {
  if (isa<UndefValue>(C)) {
    BuildMI(*MBB, IP, PPC::IMPLICIT_DEF, 0, R);
    if (getClassB(C->getType()) == cLong)
      BuildMI(*MBB, IP, PPC::IMPLICIT_DEF, 0, R+1);
    return;
  }
  if (C->getType()->isIntegral()) {
    unsigned Class = getClassB(C->getType());

    if (Class == cLong) {
      if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(C)) {
        uint64_t uval = CUI->getValue();
        unsigned hiUVal = uval >> 32;
        unsigned loUVal = uval;
        ConstantUInt *CUHi = ConstantUInt::get(Type::UIntTy, hiUVal);
        ConstantUInt *CULo = ConstantUInt::get(Type::UIntTy, loUVal);
        copyConstantToRegister(MBB, IP, CUHi, R);
        copyConstantToRegister(MBB, IP, CULo, R+1);
        return;
      } else if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(C)) {
        int64_t sval = CSI->getValue();
        int hiSVal = sval >> 32;
        int loSVal = sval;
        ConstantSInt *CSHi = ConstantSInt::get(Type::IntTy, hiSVal);
        ConstantSInt *CSLo = ConstantSInt::get(Type::IntTy, loSVal);
        copyConstantToRegister(MBB, IP, CSHi, R);
        copyConstantToRegister(MBB, IP, CSLo, R+1);
        return;
      } else {
        std::cerr << "Unhandled long constant type!\n";
        abort();
      }
    }
    
    assert(Class <= cInt && "Type not handled yet!");

    // Handle bool
    if (C->getType() == Type::BoolTy) {
      BuildMI(*MBB, IP, PPC::LI, 1, R).addSImm(C == ConstantBool::True);
      return;
    }
    
    // Handle int
    if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(C)) {
      unsigned uval = CUI->getValue();
      if (uval < 32768) {
        BuildMI(*MBB, IP, PPC::LI, 1, R).addSImm(uval);
      } else {
        unsigned Temp = makeAnotherReg(Type::IntTy);
        BuildMI(*MBB, IP, PPC::LIS, 1, Temp).addSImm(uval >> 16);
        BuildMI(*MBB, IP, PPC::ORI, 2, R).addReg(Temp).addImm(uval & 0xFFFF);
      }
      return;
    } else if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(C)) {
      int sval = CSI->getValue();
      if (sval < 32768 && sval >= -32768) {
        BuildMI(*MBB, IP, PPC::LI, 1, R).addSImm(sval);
      } else {
        unsigned Temp = makeAnotherReg(Type::IntTy);
        BuildMI(*MBB, IP, PPC::LIS, 1, Temp).addSImm(sval >> 16);
        BuildMI(*MBB, IP, PPC::ORI, 2, R).addReg(Temp).addImm(sval & 0xFFFF);
      }
      return;
    }
    std::cerr << "Unhandled integer constant!\n";
    abort();
  } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
    // We need to spill the constant to memory...
    MachineConstantPool *CP = F->getConstantPool();
    unsigned CPI = CP->getConstantPoolIndex(CFP);
    const Type *Ty = CFP->getType();

    assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");

    // Load addr of constant to reg; constant is located at base + distance
    unsigned GlobalBase = makeAnotherReg(Type::IntTy);
    unsigned Reg1 = makeAnotherReg(Type::IntTy);
    unsigned Opcode = (Ty == Type::FloatTy) ? PPC::LFS : PPC::LFD;
    // Move value at base + distance into return reg
    BuildMI(*MBB, IP, PPC::LOADHiAddr, 2, Reg1)
      .addReg(getGlobalBaseReg()).addConstantPoolIndex(CPI);
    BuildMI(*MBB, IP, Opcode, 2, R).addConstantPoolIndex(CPI).addReg(Reg1);
  } else if (isa<ConstantPointerNull>(C)) {
    // Copy zero (null pointer) to the register.
    BuildMI(*MBB, IP, PPC::LI, 1, R).addSImm(0);
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(C)) {
    // GV is located at base + distance
    
    unsigned GlobalBase = makeAnotherReg(Type::IntTy);
    unsigned TmpReg = makeAnotherReg(GV->getType());
    
    // Move value at base + distance into return reg
    BuildMI(*MBB, IP, PPC::LOADHiAddr, 2, TmpReg)
      .addReg(getGlobalBaseReg()).addGlobalAddress(GV);

    if (GV->hasWeakLinkage() || GV->isExternal()) {
      BuildMI(*MBB, IP, PPC::LWZ, 2, R).addGlobalAddress(GV).addReg(TmpReg);
    } else {
      BuildMI(*MBB, IP, PPC::LA, 2, R).addReg(TmpReg).addGlobalAddress(GV);
    }
  } else {
    std::cerr << "Offending constant: " << *C << "\n";
    assert(0 && "Type not handled yet!");
  }
}

/// LoadArgumentsToVirtualRegs - Load all of the arguments to this function from
/// the stack into virtual registers.
void PPC32ISel::LoadArgumentsToVirtualRegs(Function &Fn) {
  unsigned ArgOffset = 24;
  unsigned GPR_remaining = 8;
  unsigned FPR_remaining = 13;
  unsigned GPR_idx = 0, FPR_idx = 0;
  static const unsigned GPR[] = { 
    PPC::R3, PPC::R4, PPC::R5, PPC::R6,
    PPC::R7, PPC::R8, PPC::R9, PPC::R10,
  };
  static const unsigned FPR[] = {
    PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
    PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, PPC::F13
  };
    
  MachineFrameInfo *MFI = F->getFrameInfo();
 
  for (Function::arg_iterator I = Fn.arg_begin(), E = Fn.arg_end(); I != E; ++I) {
    bool ArgLive = !I->use_empty();
    unsigned Reg = ArgLive ? getReg(*I) : 0;
    int FI;          // Frame object index

    switch (getClassB(I->getType())) {
    case cByte:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(4, ArgOffset);
        if (GPR_remaining > 0) {
          BuildMI(BB, PPC::IMPLICIT_DEF, 0, GPR[GPR_idx]);
          BuildMI(BB, PPC::OR, 2, Reg).addReg(GPR[GPR_idx])
            .addReg(GPR[GPR_idx]);
        } else {
          addFrameReference(BuildMI(BB, PPC::LBZ, 2, Reg), FI);
        }
      }
      break;
    case cShort:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(4, ArgOffset);
        if (GPR_remaining > 0) {
          BuildMI(BB, PPC::IMPLICIT_DEF, 0, GPR[GPR_idx]);
          BuildMI(BB, PPC::OR, 2, Reg).addReg(GPR[GPR_idx])
            .addReg(GPR[GPR_idx]);
        } else {
          addFrameReference(BuildMI(BB, PPC::LHZ, 2, Reg), FI);
        }
      }
      break;
    case cInt:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(4, ArgOffset);
        if (GPR_remaining > 0) {
          BuildMI(BB, PPC::IMPLICIT_DEF, 0, GPR[GPR_idx]);
          BuildMI(BB, PPC::OR, 2, Reg).addReg(GPR[GPR_idx])
            .addReg(GPR[GPR_idx]);
        } else {
          addFrameReference(BuildMI(BB, PPC::LWZ, 2, Reg), FI);
        }
      }
      break;
    case cLong:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(8, ArgOffset);
        if (GPR_remaining > 1) {
          BuildMI(BB, PPC::IMPLICIT_DEF, 0, GPR[GPR_idx]);
          BuildMI(BB, PPC::IMPLICIT_DEF, 0, GPR[GPR_idx+1]);
          BuildMI(BB, PPC::OR, 2, Reg).addReg(GPR[GPR_idx])
            .addReg(GPR[GPR_idx]);
          BuildMI(BB, PPC::OR, 2, Reg+1).addReg(GPR[GPR_idx+1])
            .addReg(GPR[GPR_idx+1]);
        } else {
          addFrameReference(BuildMI(BB, PPC::LWZ, 2, Reg), FI);
          addFrameReference(BuildMI(BB, PPC::LWZ, 2, Reg+1), FI, 4);
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
          BuildMI(BB, PPC::IMPLICIT_DEF, 0, FPR[FPR_idx]);
          BuildMI(BB, PPC::FMR, 1, Reg).addReg(FPR[FPR_idx]);
          FPR_remaining--;
          FPR_idx++;
        } else {
          addFrameReference(BuildMI(BB, PPC::LFS, 2, Reg), FI);
        }
      }
      break;
    case cFP64:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(8, ArgOffset);

        if (FPR_remaining > 0) {
          BuildMI(BB, PPC::IMPLICIT_DEF, 0, FPR[FPR_idx]);
          BuildMI(BB, PPC::FMR, 1, Reg).addReg(FPR[FPR_idx]);
          FPR_remaining--;
          FPR_idx++;
        } else {
          addFrameReference(BuildMI(BB, PPC::LFD, 2, Reg), FI);
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
    VarArgsFrameIndex = MFI->CreateFixedObject(4, ArgOffset);
}


/// SelectPHINodes - Insert machine code to generate phis.  This is tricky
/// because we have to generate our sources into the source basic blocks, not
/// the current one.
///
void PPC32ISel::SelectPHINodes() {
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
                                    PPC::PHI, PN->getNumOperands(), PHIReg);

      MachineInstr *LongPhiMI = 0;
      if (PN->getType() == Type::LongTy || PN->getType() == Type::ULongTy)
        LongPhiMI = BuildMI(MBB, PHIInsertPoint,
                            PPC::PHI, PN->getNumOperands(), PHIReg+1);

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
            while (PI != PredMBB->end() && PI->getOpcode() == PPC::PHI)
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
      if ((isa<BranchInst>(User) ||
           (isa<SelectInst>(User) && User->getOperand(0) == V)) &&
          SCI->getParent() == User->getParent())
        return SCI;
    }
  return 0;
}

// canFoldGEPIntoLoadOrStore - Return the GEP instruction if we can fold it into
// the load or store instruction that is the only user of the GEP.
//
static GetElementPtrInst *canFoldGEPIntoLoadOrStore(Value *V) {
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(V)) {
    bool AllUsesAreMem = true;
    for (Value::use_iterator I = GEPI->use_begin(), E = GEPI->use_end(); 
         I != E; ++I) {
      Instruction *User = cast<Instruction>(*I);

      // If the GEP is the target of a store, but not the source, then we are ok
      // to fold it.
      if (isa<StoreInst>(User) &&
          GEPI->getParent() == User->getParent() &&
          User->getOperand(0) != GEPI &&
          User->getOperand(1) == GEPI)
        continue;

      // If the GEP is the source of a load, then we're always ok to fold it
      if (isa<LoadInst>(User) &&
          GEPI->getParent() == User->getParent() &&
          User->getOperand(0) == GEPI)
        continue;

      // if we got to this point, than the instruction was not a load or store
      // that we are capable of folding the GEP into.
      AllUsesAreMem = false;
      break;
    }
    if (AllUsesAreMem)
      return GEPI;
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
  case Instruction::SetEQ: return PPC::BEQ;
  case Instruction::SetNE: return PPC::BNE;
  case Instruction::SetLT: return PPC::BLT;
  case Instruction::SetGE: return PPC::BGE;
  case Instruction::SetGT: return PPC::BGT;
  case Instruction::SetLE: return PPC::BLE;
  }
}

/// emitUCOM - emits an unordered FP compare.
void PPC32ISel::emitUCOM(MachineBasicBlock *MBB, MachineBasicBlock::iterator IP,
                         unsigned LHS, unsigned RHS) {
    BuildMI(*MBB, IP, PPC::FCMPU, 2, PPC::CR0).addReg(LHS).addReg(RHS);
}

unsigned PPC32ISel::ExtendOrClear(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator IP,
                                  Value *Op0) {
  const Type *CompTy = Op0->getType();
  unsigned Reg = getReg(Op0, MBB, IP);
  unsigned Class = getClassB(CompTy);

  // Since we know that boolean values will be either zero or one, we don't
  // have to extend or clear them.
  if (CompTy == Type::BoolTy)
    return Reg;

  // Before we do a comparison or SetCC, we have to make sure that we truncate
  // the source registers appropriately.
  if (Class == cByte) {
    unsigned TmpReg = makeAnotherReg(CompTy);
    if (CompTy->isSigned())
      BuildMI(*MBB, IP, PPC::EXTSB, 1, TmpReg).addReg(Reg);
    else
      BuildMI(*MBB, IP, PPC::RLWINM, 4, TmpReg).addReg(Reg).addImm(0)
        .addImm(24).addImm(31);
    Reg = TmpReg;
  } else if (Class == cShort) {
    unsigned TmpReg = makeAnotherReg(CompTy);
    if (CompTy->isSigned())
      BuildMI(*MBB, IP, PPC::EXTSH, 1, TmpReg).addReg(Reg);
    else
      BuildMI(*MBB, IP, PPC::RLWINM, 4, TmpReg).addReg(Reg).addImm(0)
        .addImm(16).addImm(31);
    Reg = TmpReg;
  }
  return Reg;
}

/// EmitComparison - emits a comparison of the two operands, returning the
/// extended setcc code to use.  The result is in CR0.
///
unsigned PPC32ISel::EmitComparison(unsigned OpNum, Value *Op0, Value *Op1,
                                   MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator IP) {
  // The arguments are already supposed to be of the same type.
  const Type *CompTy = Op0->getType();
  unsigned Class = getClassB(CompTy);
  unsigned Op0r = ExtendOrClear(MBB, IP, Op0);
  
  // Use crand for lt, gt and crandc for le, ge
  unsigned CROpcode = (OpNum == 2 || OpNum == 4) ? PPC::CRAND : PPC::CRANDC;
  // ? cr1[lt] : cr1[gt]
  unsigned CR1field = (OpNum == 2 || OpNum == 3) ? 4 : 5;
  // ? cr0[lt] : cr0[gt]
  unsigned CR0field = (OpNum == 2 || OpNum == 5) ? 0 : 1;
  unsigned Opcode = CompTy->isSigned() ? PPC::CMPW : PPC::CMPLW;
  unsigned OpcodeImm = CompTy->isSigned() ? PPC::CMPWI : PPC::CMPLWI;

  // Special case handling of: cmp R, i
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    if (Class == cByte || Class == cShort || Class == cInt) {
      unsigned Op1v = CI->getRawValue() & 0xFFFF;
      unsigned OpClass = (CompTy->isSigned()) ? 0 : 2;
      
      // Treat compare like ADDI for the purposes of immediate suitability
      if (canUseAsImmediateForOpcode(CI, OpClass, false)) {
        BuildMI(*MBB, IP, OpcodeImm, 2, PPC::CR0).addReg(Op0r).addSImm(Op1v);
      } else {
        unsigned Op1r = getReg(Op1, MBB, IP);
        BuildMI(*MBB, IP, Opcode, 2, PPC::CR0).addReg(Op0r).addReg(Op1r);
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

        BuildMI(*MBB, IP, PPC::XORI, 2, LoLow).addReg(Op0r+1)
          .addImm(LowCst & 0xFFFF);
        BuildMI(*MBB, IP, PPC::XORIS, 2, LoTmp).addReg(LoLow)
          .addImm(LowCst >> 16);
        BuildMI(*MBB, IP, PPC::XORI, 2, HiLow).addReg(Op0r)
          .addImm(HiCst & 0xFFFF);
        BuildMI(*MBB, IP, PPC::XORIS, 2, HiTmp).addReg(HiLow)
          .addImm(HiCst >> 16);
        BuildMI(*MBB, IP, PPC::ORo, 2, FinalTmp).addReg(LoTmp).addReg(HiTmp);
        return OpNum;
      } else {
        unsigned ConstReg = makeAnotherReg(CompTy);
        copyConstantToRegister(MBB, IP, CI, ConstReg);

        // cr0 = r3 ccOpcode r5 or (r3 == r5 AND r4 ccOpcode r6)
        BuildMI(*MBB, IP, Opcode, 2, PPC::CR0).addReg(Op0r)
          .addReg(ConstReg);
        BuildMI(*MBB, IP, Opcode, 2, PPC::CR1).addReg(Op0r+1)
          .addReg(ConstReg+1);
        BuildMI(*MBB, IP, PPC::CRAND, 3).addImm(2).addImm(2).addImm(CR1field);
        BuildMI(*MBB, IP, PPC::CROR, 3).addImm(CR0field).addImm(CR0field)
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
    BuildMI(*MBB, IP, Opcode, 2, PPC::CR0).addReg(Op0r).addReg(Op1r);
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
      BuildMI(*MBB, IP, PPC::XOR, 2, HiTmp).addReg(Op0r).addReg(Op1r);
      BuildMI(*MBB, IP, PPC::XOR, 2, LoTmp).addReg(Op0r+1).addReg(Op1r+1);
      BuildMI(*MBB, IP, PPC::ORo,  2, FinalTmp).addReg(LoTmp).addReg(HiTmp);
      break;  // Allow the sete or setne to be generated from flags set by OR
    } else {
      unsigned TmpReg1 = makeAnotherReg(Type::IntTy);
      unsigned TmpReg2 = makeAnotherReg(Type::IntTy);

      // cr0 = r3 ccOpcode r5 or (r3 == r5 AND r4 ccOpcode r6)
      BuildMI(*MBB, IP, Opcode, 2, PPC::CR0).addReg(Op0r).addReg(Op1r);
      BuildMI(*MBB, IP, Opcode, 2, PPC::CR1).addReg(Op0r+1).addReg(Op1r+1);
      BuildMI(*MBB, IP, PPC::CRAND, 3).addImm(2).addImm(2).addImm(CR1field);
      BuildMI(*MBB, IP, PPC::CROR, 3).addImm(CR0field).addImm(CR0field)
        .addImm(2);
      return OpNum;
    }
  }
  return OpNum;
}

/// visitSetCondInst - emit code to calculate the condition via
/// EmitComparison(), and possibly store a 0 or 1 to a register as a result
///
void PPC32ISel::visitSetCondInst(SetCondInst &I) {
  if (canFoldSetCCIntoBranchOrSelect(&I))
    return;

  MachineBasicBlock::iterator MI = BB->end();
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  const Type *Ty = Op0->getType();
  unsigned Class = getClassB(Ty);
  unsigned Opcode = I.getOpcode();
  unsigned OpNum = getSetCCNumber(Opcode);      
  unsigned DestReg = getReg(I);

  // If the comparison type is byte, short, or int, then we can emit a
  // branchless version of the SetCC that puts 0 (false) or 1 (true) in the
  // destination register.
  if (Class <= cInt) {
    ConstantInt *CI = dyn_cast<ConstantInt>(Op1);

    if (CI && CI->getRawValue() == 0) {
      unsigned Op0Reg = ExtendOrClear(BB, MI, Op0);
    
      // comparisons against constant zero and negative one often have shorter
      // and/or faster sequences than the set-and-branch general case, handled
      // below.
      switch(OpNum) {
      case 0: { // eq0
        unsigned TempReg = makeAnotherReg(Type::IntTy);
        BuildMI(*BB, MI, PPC::CNTLZW, 1, TempReg).addReg(Op0Reg);
        BuildMI(*BB, MI, PPC::RLWINM, 4, DestReg).addReg(TempReg).addImm(27)
          .addImm(5).addImm(31);
        break;
        } 
      case 1: { // ne0
        unsigned TempReg = makeAnotherReg(Type::IntTy);
        BuildMI(*BB, MI, PPC::ADDIC, 2, TempReg).addReg(Op0Reg).addSImm(-1);
        BuildMI(*BB, MI, PPC::SUBFE, 2, DestReg).addReg(TempReg).addReg(Op0Reg);
        break;
        } 
      case 2: { // lt0, always false if unsigned
        if (Ty->isSigned())
          BuildMI(*BB, MI, PPC::RLWINM, 4, DestReg).addReg(Op0Reg).addImm(1)
            .addImm(31).addImm(31);
        else
          BuildMI(*BB, MI, PPC::LI, 1, DestReg).addSImm(0);
        break;
        }
      case 3: { // ge0, always true if unsigned
        if (Ty->isSigned()) { 
          unsigned TempReg = makeAnotherReg(Type::IntTy);
          BuildMI(*BB, MI, PPC::RLWINM, 4, TempReg).addReg(Op0Reg).addImm(1)
            .addImm(31).addImm(31);
          BuildMI(*BB, MI, PPC::XORI, 2, DestReg).addReg(TempReg).addImm(1);
        } else {
          BuildMI(*BB, MI, PPC::LI, 1, DestReg).addSImm(1);
        }
        break;
        }
      case 4: { // gt0, equivalent to ne0 if unsigned
        unsigned Temp1 = makeAnotherReg(Type::IntTy);
        unsigned Temp2 = makeAnotherReg(Type::IntTy);
        if (Ty->isSigned()) { 
          BuildMI(*BB, MI, PPC::NEG, 2, Temp1).addReg(Op0Reg);
          BuildMI(*BB, MI, PPC::ANDC, 2, Temp2).addReg(Temp1).addReg(Op0Reg);
          BuildMI(*BB, MI, PPC::RLWINM, 4, DestReg).addReg(Temp2).addImm(1)
            .addImm(31).addImm(31);
        } else {
          BuildMI(*BB, MI, PPC::ADDIC, 2, Temp1).addReg(Op0Reg).addSImm(-1);
          BuildMI(*BB, MI, PPC::SUBFE, 2, DestReg).addReg(Temp1).addReg(Op0Reg);
        }
        break;
        }
      case 5: { // le0, equivalent to eq0 if unsigned
        unsigned Temp1 = makeAnotherReg(Type::IntTy);
        unsigned Temp2 = makeAnotherReg(Type::IntTy);
        if (Ty->isSigned()) { 
          BuildMI(*BB, MI, PPC::NEG, 2, Temp1).addReg(Op0Reg);
          BuildMI(*BB, MI, PPC::ORC, 2, Temp2).addReg(Op0Reg).addReg(Temp1);
          BuildMI(*BB, MI, PPC::RLWINM, 4, DestReg).addReg(Temp2).addImm(1)
            .addImm(31).addImm(31);
        } else {
          BuildMI(*BB, MI, PPC::CNTLZW, 1, Temp1).addReg(Op0Reg);
          BuildMI(*BB, MI, PPC::RLWINM, 4, DestReg).addReg(Temp1).addImm(27)
            .addImm(5).addImm(31);
        }
        break;
        }
      } // switch
      return;
  	}
  }
  unsigned PPCOpcode = getPPCOpcodeForSetCCNumber(Opcode);

  // Create an iterator with which to insert the MBB for copying the false value
  // and the MBB to hold the PHI instruction for this SetCC.
  MachineBasicBlock *thisMBB = BB;
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  ilist<MachineBasicBlock>::iterator It = BB;
  ++It;
  
  //  thisMBB:
  //  ...
  //   cmpTY cr0, r1, r2
  //   %TrueValue = li 1
  //   bCC sinkMBB
  EmitComparison(Opcode, Op0, Op1, BB, BB->end());
  unsigned TrueValue = makeAnotherReg(I.getType());
  BuildMI(BB, PPC::LI, 1, TrueValue).addSImm(1);
  MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
  BuildMI(BB, PPCOpcode, 2).addReg(PPC::CR0).addMBB(sinkMBB);
  F->getBasicBlockList().insert(It, copy0MBB);
  F->getBasicBlockList().insert(It, sinkMBB);
  // Update machine-CFG edges
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);

  //  copy0MBB:
  //   %FalseValue = li 0
  //   fallthrough
  BB = copy0MBB;
  unsigned FalseValue = makeAnotherReg(I.getType());
  BuildMI(BB, PPC::LI, 1, FalseValue).addSImm(0);
  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
  //  ...
  BB = sinkMBB;
  BuildMI(BB, PPC::PHI, 4, DestReg).addReg(FalseValue)
    .addMBB(copy0MBB).addReg(TrueValue).addMBB(thisMBB);
}

void PPC32ISel::visitSelectInst(SelectInst &SI) {
  unsigned DestReg = getReg(SI);
  MachineBasicBlock::iterator MII = BB->end();
  emitSelectOperation(BB, MII, SI.getCondition(), SI.getTrueValue(),
                      SI.getFalseValue(), DestReg);
}
 
/// emitSelect - Common code shared between visitSelectInst and the constant
/// expression support.
void PPC32ISel::emitSelectOperation(MachineBasicBlock *MBB,
                                    MachineBasicBlock::iterator IP,
                                    Value *Cond, Value *TrueVal, 
                                    Value *FalseVal, unsigned DestReg) {
  unsigned SelectClass = getClassB(TrueVal->getType());
  unsigned Opcode;

  // See if we can fold the setcc into the select instruction, or if we have
  // to get the register of the Cond value
  if (SetCondInst *SCI = canFoldSetCCIntoBranchOrSelect(Cond)) {
    // We successfully folded the setcc into the select instruction.
    unsigned OpNum = getSetCCNumber(SCI->getOpcode());
    if (OpNum >= 2 && OpNum <= 5) {
      unsigned SetCondClass = getClassB(SCI->getOperand(0)->getType());
      if ((SetCondClass == cFP32 || SetCondClass == cFP64) &&
          (SelectClass == cFP32 || SelectClass == cFP64)) {
        unsigned CondReg = getReg(SCI->getOperand(0), MBB, IP);
        unsigned TrueReg = getReg(TrueVal, MBB, IP);
        unsigned FalseReg = getReg(FalseVal, MBB, IP);
        // if the comparison of the floating point value used to for the select
        // is against 0, then we can emit an fsel without subtraction.
        ConstantFP *Op1C = dyn_cast<ConstantFP>(SCI->getOperand(1));
        if (Op1C && (Op1C->isExactlyValue(-0.0) || Op1C->isExactlyValue(0.0))) {
          switch(OpNum) {
          case 2:   // LT
            BuildMI(*MBB, IP, PPC::FSEL, 3, DestReg).addReg(CondReg)
              .addReg(FalseReg).addReg(TrueReg);
            break;
          case 3:   // GE == !LT
            BuildMI(*MBB, IP, PPC::FSEL, 3, DestReg).addReg(CondReg)
              .addReg(TrueReg).addReg(FalseReg);
            break;
          case 4: {  // GT
            unsigned NegatedReg = makeAnotherReg(SCI->getOperand(0)->getType());
            BuildMI(*MBB, IP, PPC::FNEG, 1, NegatedReg).addReg(CondReg);
            BuildMI(*MBB, IP, PPC::FSEL, 3, DestReg).addReg(NegatedReg)
              .addReg(FalseReg).addReg(TrueReg);
            }
            break;
          case 5: {  // LE == !GT
            unsigned NegatedReg = makeAnotherReg(SCI->getOperand(0)->getType());
            BuildMI(*MBB, IP, PPC::FNEG, 1, NegatedReg).addReg(CondReg);
            BuildMI(*MBB, IP, PPC::FSEL, 3, DestReg).addReg(NegatedReg)
              .addReg(TrueReg).addReg(FalseReg);
            }
            break;
          default:
            assert(0 && "Invalid SetCC opcode to fsel");
            abort();
            break;
          }
        } else {
          unsigned OtherCondReg = getReg(SCI->getOperand(1), MBB, IP);
          unsigned SelectReg = makeAnotherReg(SCI->getOperand(0)->getType());
          switch(OpNum) {
          case 2:   // LT
            BuildMI(*MBB, IP, PPC::FSUB, 2, SelectReg).addReg(CondReg)
              .addReg(OtherCondReg);
            BuildMI(*MBB, IP, PPC::FSEL, 3, DestReg).addReg(SelectReg)
              .addReg(FalseReg).addReg(TrueReg);
            break;
          case 3:   // GE == !LT
            BuildMI(*MBB, IP, PPC::FSUB, 2, SelectReg).addReg(CondReg)
              .addReg(OtherCondReg);
            BuildMI(*MBB, IP, PPC::FSEL, 3, DestReg).addReg(SelectReg)
              .addReg(TrueReg).addReg(FalseReg);
            break;
          case 4:   // GT
            BuildMI(*MBB, IP, PPC::FSUB, 2, SelectReg).addReg(OtherCondReg)
              .addReg(CondReg);
            BuildMI(*MBB, IP, PPC::FSEL, 3, DestReg).addReg(SelectReg)
              .addReg(FalseReg).addReg(TrueReg);
            break;
          case 5:   // LE == !GT
            BuildMI(*MBB, IP, PPC::FSUB, 2, SelectReg).addReg(OtherCondReg)
              .addReg(CondReg);
            BuildMI(*MBB, IP, PPC::FSEL, 3, DestReg).addReg(SelectReg)
              .addReg(TrueReg).addReg(FalseReg);
            break;
          default:
            assert(0 && "Invalid SetCC opcode to fsel");
            abort();
            break;
          }
        }
        return;
      }
    }
    OpNum = EmitComparison(OpNum, SCI->getOperand(0),SCI->getOperand(1),MBB,IP);
    Opcode = getPPCOpcodeForSetCCNumber(SCI->getOpcode());
  } else {
    unsigned CondReg = getReg(Cond, MBB, IP);
    BuildMI(*MBB, IP, PPC::CMPWI, 2, PPC::CR0).addReg(CondReg).addSImm(0);
    Opcode = getPPCOpcodeForSetCCNumber(Instruction::SetNE);
  }

  MachineBasicBlock *thisMBB = BB;
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  ilist<MachineBasicBlock>::iterator It = BB;
  ++It;

  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   cmpTY cr0, r1, r2
  //   bCC copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
  unsigned TrueValue = getReg(TrueVal);
  BuildMI(BB, Opcode, 2).addReg(PPC::CR0).addMBB(sinkMBB);
  F->getBasicBlockList().insert(It, copy0MBB);
  F->getBasicBlockList().insert(It, sinkMBB);
  // Update machine-CFG edges
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);

  //  copy0MBB:
  //   %FalseValue = ...
  //   # fallthrough to sinkMBB
  BB = copy0MBB;
  unsigned FalseValue = getReg(FalseVal);
  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
  //  ...
  BB = sinkMBB;
  BuildMI(BB, PPC::PHI, 4, DestReg).addReg(FalseValue)
    .addMBB(copy0MBB).addReg(TrueValue).addMBB(thisMBB);
    
  // For a register pair representing a long value, define the top part.
  if (getClassB(TrueVal->getType()) == cLong)
    BuildMI(BB, PPC::PHI, 4, DestReg+1).addReg(FalseValue+1)
      .addMBB(copy0MBB).addReg(TrueValue+1).addMBB(thisMBB);
}



/// promote32 - Emit instructions to turn a narrow operand into a 32-bit-wide
/// operand, in the specified target register.
///
void PPC32ISel::promote32(unsigned targetReg, const ValueRecord &VR) {
  bool isUnsigned = VR.Ty->isUnsigned() || VR.Ty == Type::BoolTy;

  Value *Val = VR.Val;
  const Type *Ty = VR.Ty;
  if (Val) {
    if (Constant *C = dyn_cast<Constant>(Val)) {
      Val = ConstantExpr::getCast(C, Type::IntTy);
      if (isa<ConstantExpr>(Val))   // Could not fold
        Val = C;
      else
        Ty = Type::IntTy;           // Folded!
    }

    // If this is a simple constant, just emit a load directly to avoid the copy
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
      copyConstantToRegister(BB, BB->end(), CI, targetReg);
      return;
    }
  }

  // Make sure we have the register number for this value...
  unsigned Reg = Val ? getReg(Val) : VR.Reg;
  switch (getClassB(Ty)) {
  case cByte:
    // Extend value into target register (8->32)
    if (Ty == Type::BoolTy)
      BuildMI(BB, PPC::OR, 2, targetReg).addReg(Reg).addReg(Reg);
    else if (isUnsigned)
      BuildMI(BB, PPC::RLWINM, 4, targetReg).addReg(Reg).addZImm(0)
        .addZImm(24).addZImm(31);
    else
      BuildMI(BB, PPC::EXTSB, 1, targetReg).addReg(Reg);
    break;
  case cShort:
    // Extend value into target register (16->32)
    if (isUnsigned)
      BuildMI(BB, PPC::RLWINM, 4, targetReg).addReg(Reg).addZImm(0)
        .addZImm(16).addZImm(31);
    else
      BuildMI(BB, PPC::EXTSH, 1, targetReg).addReg(Reg);
    break;
  case cInt:
    // Move value into target register (32->32)
    BuildMI(BB, PPC::OR, 2, targetReg).addReg(Reg).addReg(Reg);
    break;
  default:
    assert(0 && "Unpromotable operand class in promote32");
  }
}

/// visitReturnInst - implemented with BLR
///
void PPC32ISel::visitReturnInst(ReturnInst &I) {
  // Only do the processing if this is a non-void return
  if (I.getNumOperands() > 0) {
    Value *RetVal = I.getOperand(0);
    switch (getClassB(RetVal->getType())) {
    case cByte:   // integral return values: extend or move into r3 and return
    case cShort:
    case cInt:
      promote32(PPC::R3, ValueRecord(RetVal));
      break;
    case cFP32:
    case cFP64: {   // Floats & Doubles: Return in f1
      unsigned RetReg = getReg(RetVal);
      BuildMI(BB, PPC::FMR, 1, PPC::F1).addReg(RetReg);
      break;
    }
    case cLong: {
      unsigned RetReg = getReg(RetVal);
      BuildMI(BB, PPC::OR, 2, PPC::R3).addReg(RetReg).addReg(RetReg);
      BuildMI(BB, PPC::OR, 2, PPC::R4).addReg(RetReg+1).addReg(RetReg+1);
      break;
    }
    default:
      visitInstruction(I);
    }
  }
  BuildMI(BB, PPC::BLR, 1).addImm(0);
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
void PPC32ISel::visitBranchInst(BranchInst &BI) {
  // Update machine-CFG edges
  BB->addSuccessor(MBBMap[BI.getSuccessor(0)]);
  if (BI.isConditional())
    BB->addSuccessor(MBBMap[BI.getSuccessor(1)]);
  
  BasicBlock *NextBB = getBlockAfter(BI.getParent());  // BB after current one

  if (!BI.isConditional()) {  // Unconditional branch?
    if (BI.getSuccessor(0) != NextBB) 
      BuildMI(BB, PPC::B, 1).addMBB(MBBMap[BI.getSuccessor(0)]);
    return;
  }
  
  // See if we can fold the setcc into the branch itself...
  SetCondInst *SCI = canFoldSetCCIntoBranchOrSelect(BI.getCondition());
  if (SCI == 0) {
    // Nope, cannot fold setcc into this branch.  Emit a branch on a condition
    // computed some other way...
    unsigned condReg = getReg(BI.getCondition());
    BuildMI(BB, PPC::CMPLI, 3, PPC::CR0).addImm(0).addReg(condReg)
      .addImm(0);
    if (BI.getSuccessor(1) == NextBB) {
      if (BI.getSuccessor(0) != NextBB)
        BuildMI(BB, PPC::COND_BRANCH, 3).addReg(PPC::CR0).addImm(PPC::BNE)
          .addMBB(MBBMap[BI.getSuccessor(0)])
          .addMBB(MBBMap[BI.getSuccessor(1)]);
    } else {
      BuildMI(BB, PPC::COND_BRANCH, 3).addReg(PPC::CR0).addImm(PPC::BEQ)
        .addMBB(MBBMap[BI.getSuccessor(1)])
        .addMBB(MBBMap[BI.getSuccessor(0)]);
      if (BI.getSuccessor(0) != NextBB)
        BuildMI(BB, PPC::B, 1).addMBB(MBBMap[BI.getSuccessor(0)]);
    }
    return;
  }

  unsigned OpNum = getSetCCNumber(SCI->getOpcode());
  unsigned Opcode = getPPCOpcodeForSetCCNumber(SCI->getOpcode());
  MachineBasicBlock::iterator MII = BB->end();
  OpNum = EmitComparison(OpNum, SCI->getOperand(0), SCI->getOperand(1), BB,MII);
  
  if (BI.getSuccessor(0) != NextBB) {
    BuildMI(BB, PPC::COND_BRANCH, 3).addReg(PPC::CR0).addImm(Opcode)
      .addMBB(MBBMap[BI.getSuccessor(0)])
      .addMBB(MBBMap[BI.getSuccessor(1)]);
    if (BI.getSuccessor(1) != NextBB)
      BuildMI(BB, PPC::B, 1).addMBB(MBBMap[BI.getSuccessor(1)]);
  } else {
    // Change to the inverse condition...
    if (BI.getSuccessor(1) != NextBB) {
      Opcode = PPC32InstrInfo::invertPPCBranchOpcode(Opcode);
      BuildMI(BB, PPC::COND_BRANCH, 3).addReg(PPC::CR0).addImm(Opcode)
        .addMBB(MBBMap[BI.getSuccessor(1)])
        .addMBB(MBBMap[BI.getSuccessor(0)]);
    }
  }
}

/// doCall - This emits an abstract call instruction, setting up the arguments
/// and the return value as appropriate.  For the actual function call itself,
/// it inserts the specified CallMI instruction into the stream.
///
/// FIXME: See Documentation at the following URL for "correct" behavior
/// <http://developer.apple.com/documentation/DeveloperTools/Conceptual/MachORuntime/2rt_powerpc_abi/chapter_9_section_5.html>
void PPC32ISel::doCall(const ValueRecord &Ret, MachineInstr *CallMI,
                       const std::vector<ValueRecord> &Args, bool isVarArg) {
  // Count how many bytes are to be pushed on the stack, including the linkage
  // area, and parameter passing area.
  unsigned NumBytes = 24;
  unsigned ArgOffset = 24;

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

    // Just to be safe, we'll always reserve the full 24 bytes of linkage area 
    // plus 32 bytes of argument space in case any called code gets funky on us.
    if (NumBytes < 56) NumBytes = 56;

    // Adjust the stack pointer for the new arguments...
    // These functions are automatically eliminated by the prolog/epilog pass
    BuildMI(BB, PPC::ADJCALLSTACKDOWN, 1).addImm(NumBytes);

    // Arguments go on the stack in reverse order, as specified by the ABI.
    // Offset to the paramater area on the stack is 24.
    int GPR_remaining = 8, FPR_remaining = 13;
    unsigned GPR_idx = 0, FPR_idx = 0;
    static const unsigned GPR[] = { 
      PPC::R3, PPC::R4, PPC::R5, PPC::R6,
      PPC::R7, PPC::R8, PPC::R9, PPC::R10,
    };
    static const unsigned FPR[] = {
      PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, 
      PPC::F7, PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, 
      PPC::F13
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
          BuildMI(BB, PPC::OR, 2, GPR[GPR_idx]).addReg(ArgReg)
            .addReg(ArgReg);
          CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
        }
        if (GPR_remaining <= 0 || isVarArg) {
          BuildMI(BB, PPC::STW, 3).addReg(ArgReg).addSImm(ArgOffset)
            .addReg(PPC::R1);
        }
        break;
      case cInt:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;

        // Reg or stack?
        if (GPR_remaining > 0) {
          BuildMI(BB, PPC::OR, 2, GPR[GPR_idx]).addReg(ArgReg)
            .addReg(ArgReg);
          CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
        }
        if (GPR_remaining <= 0 || isVarArg) {
          BuildMI(BB, PPC::STW, 3).addReg(ArgReg).addSImm(ArgOffset)
            .addReg(PPC::R1);
        }
        break;
      case cLong:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;

        // Reg or stack?  Note that PPC calling conventions state that long args
        // are passed rN = hi, rN+1 = lo, opposite of LLVM.
        if (GPR_remaining > 1) {
          BuildMI(BB, PPC::OR, 2, GPR[GPR_idx]).addReg(ArgReg)
            .addReg(ArgReg);
          BuildMI(BB, PPC::OR, 2, GPR[GPR_idx+1]).addReg(ArgReg+1)
            .addReg(ArgReg+1);
          CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
          CallMI->addRegOperand(GPR[GPR_idx+1], MachineOperand::Use);
        }
        if (GPR_remaining <= 1 || isVarArg) {
          BuildMI(BB, PPC::STW, 3).addReg(ArgReg).addSImm(ArgOffset)
            .addReg(PPC::R1);
          BuildMI(BB, PPC::STW, 3).addReg(ArgReg+1).addSImm(ArgOffset+4)
            .addReg(PPC::R1);
        }

        ArgOffset += 4;        // 8 byte entry, not 4.
        GPR_remaining -= 1;    // uses up 2 GPRs
        GPR_idx += 1;
        break;
      case cFP32:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
        // Reg or stack?
        if (FPR_remaining > 0) {
          BuildMI(BB, PPC::FMR, 1, FPR[FPR_idx]).addReg(ArgReg);
          CallMI->addRegOperand(FPR[FPR_idx], MachineOperand::Use);
          FPR_remaining--;
          FPR_idx++;
          
          // If this is a vararg function, and there are GPRs left, also
          // pass the float in an int.  Otherwise, put it on the stack.
          if (isVarArg) {
            BuildMI(BB, PPC::STFS, 3).addReg(ArgReg).addSImm(ArgOffset)
            .addReg(PPC::R1);
            if (GPR_remaining > 0) {
              BuildMI(BB, PPC::LWZ, 2, GPR[GPR_idx])
              .addSImm(ArgOffset).addReg(PPC::R1);
              CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
            }
          }
        } else {
          BuildMI(BB, PPC::STFS, 3).addReg(ArgReg).addSImm(ArgOffset)
          .addReg(PPC::R1);
        }
        break;
      case cFP64:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
        // Reg or stack?
        if (FPR_remaining > 0) {
          BuildMI(BB, PPC::FMR, 1, FPR[FPR_idx]).addReg(ArgReg);
          CallMI->addRegOperand(FPR[FPR_idx], MachineOperand::Use);
          FPR_remaining--;
          FPR_idx++;
          // For vararg functions, must pass doubles via int regs as well
          if (isVarArg) {
            BuildMI(BB, PPC::STFD, 3).addReg(ArgReg).addSImm(ArgOffset)
            .addReg(PPC::R1);
            
            // Doubles can be split across reg + stack for varargs
            if (GPR_remaining > 0) {
              BuildMI(BB, PPC::LWZ, 2, GPR[GPR_idx]).addSImm(ArgOffset)
              .addReg(PPC::R1);
              CallMI->addRegOperand(GPR[GPR_idx], MachineOperand::Use);
            }
            if (GPR_remaining > 1) {
              BuildMI(BB, PPC::LWZ, 2, GPR[GPR_idx+1])
                .addSImm(ArgOffset+4).addReg(PPC::R1);
              CallMI->addRegOperand(GPR[GPR_idx+1], MachineOperand::Use);
            }
          }
        } else {
          BuildMI(BB, PPC::STFD, 3).addReg(ArgReg).addSImm(ArgOffset)
          .addReg(PPC::R1);
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
    BuildMI(BB, PPC::ADJCALLSTACKDOWN, 1).addImm(NumBytes);
  }
  
  BuildMI(BB, PPC::IMPLICIT_DEF, 0, PPC::LR);
  BB->push_back(CallMI);
  
  // These functions are automatically eliminated by the prolog/epilog pass
  BuildMI(BB, PPC::ADJCALLSTACKUP, 1).addImm(NumBytes);

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
      BuildMI(BB, PPC::OR, 2, Ret.Reg).addReg(PPC::R3).addReg(PPC::R3);
      break;
    case cFP32:   // Floating-point return values live in f1
    case cFP64:
      BuildMI(BB, PPC::FMR, 1, Ret.Reg).addReg(PPC::F1);
      break;
    case cLong:   // Long values are in r3:r4
      BuildMI(BB, PPC::OR, 2, Ret.Reg).addReg(PPC::R3).addReg(PPC::R3);
      BuildMI(BB, PPC::OR, 2, Ret.Reg+1).addReg(PPC::R4).addReg(PPC::R4);
      break;
    default: assert(0 && "Unknown class!");
    }
  }
}


/// visitCallInst - Push args on stack and do a procedure call instruction.
void PPC32ISel::visitCallInst(CallInst &CI) {
  MachineInstr *TheCall;
  Function *F = CI.getCalledFunction();
  if (F) {
    // Is it an intrinsic function call?
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID()) {
      visitIntrinsicCall(ID, CI);   // Special intrinsics are not handled here
      return;
    }
    // Emit a CALL instruction with PC-relative displacement.
    TheCall = BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(F, true);
  } else {  // Emit an indirect call through the CTR
    unsigned Reg = getReg(CI.getCalledValue());
    BuildMI(BB, PPC::OR, 2, PPC::R12).addReg(Reg).addReg(Reg);
    BuildMI(BB, PPC::MTCTR, 1).addReg(PPC::R12);
    TheCall = BuildMI(PPC::CALLindirect, 2).addZImm(20).addZImm(0)
      .addReg(PPC::R12);
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
void PPC32ISel::LowerUnknownIntrinsicFunctionCalls(Function &F) {
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
            // FIXME: should lower these ourselves
            // case Intrinsic::isunordered:
            // case Intrinsic::memcpy: -> doCall().  system memcpy almost
            // guaranteed to be faster than anything we generate ourselves
            // We directly implement these intrinsics
            break;
          case Intrinsic::readio: {
            // On PPC, memory operations are in-order.  Lower this intrinsic
            // into a volatile load.
            LoadInst * LI = new LoadInst(CI->getOperand(1), "", true, CI);
            CI->replaceAllUsesWith(LI);
            BB->getInstList().erase(CI);
            break;
          }
          case Intrinsic::writeio: {
            // On PPC, memory operations are in-order.  Lower this intrinsic
            // into a volatile store.
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

void PPC32ISel::visitIntrinsicCall(Intrinsic::ID ID, CallInst &CI) {
  unsigned TmpReg1, TmpReg2, TmpReg3;
  switch (ID) {
  case Intrinsic::vastart:
    // Get the address of the first vararg value...
    TmpReg1 = getReg(CI);
    addFrameReference(BuildMI(BB, PPC::ADDI, 2, TmpReg1), VarArgsFrameIndex, 
                      0, false);
    return;

  case Intrinsic::vacopy:
    TmpReg1 = getReg(CI);
    TmpReg2 = getReg(CI.getOperand(1));
    BuildMI(BB, PPC::OR, 2, TmpReg1).addReg(TmpReg2).addReg(TmpReg2);
    return;
  case Intrinsic::vaend: return;

  case Intrinsic::returnaddress:
    TmpReg1 = getReg(CI);
    if (cast<Constant>(CI.getOperand(1))->isNullValue()) {
      MachineFrameInfo *MFI = F->getFrameInfo();
      unsigned NumBytes = MFI->getStackSize();
      
      BuildMI(BB, PPC::LWZ, 2, TmpReg1).addSImm(NumBytes+8)
        .addReg(PPC::R1);
    } else {
      // Values other than zero are not implemented yet.
      BuildMI(BB, PPC::LI, 1, TmpReg1).addSImm(0);
    }
    return;

  case Intrinsic::frameaddress:
    TmpReg1 = getReg(CI);
    if (cast<Constant>(CI.getOperand(1))->isNullValue()) {
      BuildMI(BB, PPC::OR, 2, TmpReg1).addReg(PPC::R1).addReg(PPC::R1);
    } else {
      // Values other than zero are not implemented yet.
      BuildMI(BB, PPC::LI, 1, TmpReg1).addSImm(0);
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
    BuildMI(BB, PPC::MFCR, TmpReg2);
    TmpReg3 = getReg(CI);
    BuildMI(BB, PPC::RLWINM, 4, TmpReg3).addReg(TmpReg2).addImm(4).addImm(31).addImm(31);
    return;
#endif
    
  default: assert(0 && "Error: unknown intrinsics should have been lowered!");
  }
}

/// visitSimpleBinary - Implement simple binary operators for integral types...
/// OperatorClass is one of: 0 for Add, 1 for Sub, 2 for And, 3 for Or, 4 for
/// Xor.
///
void PPC32ISel::visitSimpleBinary(BinaryOperator &B, unsigned OperatorClass) {
  if (std::find(SkipList.begin(), SkipList.end(), &B) != SkipList.end())
    return;

  unsigned DestReg = getReg(B);
  MachineBasicBlock::iterator MI = BB->end();
  RlwimiRec RR = InsertMap[&B];
  if (RR.Target != 0) {
    unsigned TargetReg = getReg(RR.Target, BB, MI);
    unsigned InsertReg = getReg(RR.Insert, BB, MI);
    BuildMI(*BB, MI, PPC::RLWIMI, 5, DestReg).addReg(TargetReg)
      .addReg(InsertReg).addImm(RR.Shift).addImm(RR.MB).addImm(RR.ME);
    return;
  }
    
  unsigned Class = getClassB(B.getType());
  Value *Op0 = B.getOperand(0), *Op1 = B.getOperand(1);
  emitSimpleBinaryOperation(BB, MI, &B, Op0, Op1, OperatorClass, DestReg);
}

/// emitBinaryFPOperation - This method handles emission of floating point
/// Add (0), Sub (1), Mul (2), and Div (3) operations.
void PPC32ISel::emitBinaryFPOperation(MachineBasicBlock *BB,
                                      MachineBasicBlock::iterator IP,
                                      Value *Op0, Value *Op1,
                                      unsigned OperatorClass, unsigned DestReg){

  static const unsigned OpcodeTab[][4] = {
    { PPC::FADDS, PPC::FSUBS, PPC::FMULS, PPC::FDIVS },  // Float
    { PPC::FADD,  PPC::FSUB,  PPC::FMUL,  PPC::FDIV },   // Double
  };

  // Special case: R1 = op <const fp>, R2
  if (ConstantFP *Op0C = dyn_cast<ConstantFP>(Op0))
    if (Op0C->isExactlyValue(-0.0) && OperatorClass == 1) {
      // -0.0 - X === -X
      unsigned op1Reg = getReg(Op1, BB, IP);
      BuildMI(*BB, IP, PPC::FNEG, 1, DestReg).addReg(op1Reg);
      return;
    }

  unsigned Opcode = OpcodeTab[Op0->getType() == Type::DoubleTy][OperatorClass];
  unsigned Op0r = getReg(Op0, BB, IP);
  unsigned Op1r = getReg(Op1, BB, IP);
  BuildMI(*BB, IP, Opcode, 2, DestReg).addReg(Op0r).addReg(Op1r);
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

// isRunOfOnes - returns true if Val consists of one contiguous run of 1's with
// any number of 0's on either side.  the 1's are allowed to wrap from LSB to
// MSB.  so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is
// not, since all 1's are not contiguous.
static bool isRunOfOnes(unsigned Val, unsigned &MB, unsigned &ME) {
  bool isRun = true;
  MB = 0; 
  ME = 0;

  // look for first set bit
  int i = 0;
  for (; i < 32; i++) {
    if ((Val & (1 << (31 - i))) != 0) {
      MB = i;
      ME = i;
      break;
    }
  }
  
  // look for last set bit
  for (; i < 32; i++) {
    if ((Val & (1 << (31 - i))) == 0)
      break;
    ME = i;
  }

  // look for next set bit
  for (; i < 32; i++) {
    if ((Val & (1 << (31 - i))) != 0)
      break;
  }
  
  // if we exhausted all the bits, we found a match at this point for 0*1*0*
  if (i == 32)
    return true;

  // since we just encountered more 1's, if it doesn't wrap around to the
  // most significant bit of the word, then we did not find a match to 1*0*1* so
  // exit.
  if (MB != 0)
    return false;

  // look for last set bit
  for (MB = i; i < 32; i++) {
    if ((Val & (1 << (31 - i))) == 0)
      break;
  }
  
  // if we exhausted all the bits, then we found a match for 1*0*1*, otherwise,
  // the value is not a run of ones.
  if (i == 32)
    return true;
  return false;
}

/// isInsertAndHalf - Helper function for emitBitfieldInsert.  Returns true if
/// OpUser has one use, is used by an or instruction, and is itself an and whose
/// second operand is a constant int.  Optionally, set OrI to the Or instruction
/// that is the sole user of OpUser, and Op1User to the other operand of the Or
/// instruction.
static bool isInsertAndHalf(User *OpUser, Instruction **Op1User, 
                            Instruction **OrI, unsigned &Mask) {
  // If this instruction doesn't have one use, then return false.
  if (!OpUser->hasOneUse())
    return false;
  
  Mask = 0xFFFFFFFF;
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(OpUser))
    if (BO->getOpcode() == Instruction::And) {
      Value *AndUse = *(OpUser->use_begin());
      if (BinaryOperator *Or = dyn_cast<BinaryOperator>(AndUse)) {
        if (Or->getOpcode() == Instruction::Or) {
          if (ConstantInt *CI = dyn_cast<ConstantInt>(OpUser->getOperand(1))) {
            if (OrI) *OrI = Or;
            if (Op1User) {
              if (Or->getOperand(0) == OpUser)
                *Op1User = dyn_cast<Instruction>(Or->getOperand(1));
              else
                *Op1User = dyn_cast<Instruction>(Or->getOperand(0));
            }
            Mask &= CI->getRawValue();
            return true;
          }
        }
      }
    }
  return false;
}

/// isInsertShiftHalf - Helper function for emitBitfieldInsert.  Returns true if
/// OpUser has one use, is used by an or instruction, and is itself a shift
/// instruction that is either used directly by the or instruction, or is used
/// by an and instruction whose second operand is a constant int, and which is
/// used by the or instruction.
static bool isInsertShiftHalf(User *OpUser, Instruction **Op1User, 
                              Instruction **OrI, Instruction **OptAndI, 
                              unsigned &Shift, unsigned &Mask) {
  // If this instruction doesn't have one use, then return false.
  if (!OpUser->hasOneUse())
    return false;
  
  Mask = 0xFFFFFFFF;
  if (ShiftInst *SI = dyn_cast<ShiftInst>(OpUser)) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(SI->getOperand(1))) {
      Shift = CI->getRawValue();
      if (SI->getOpcode() == Instruction::Shl)
        Mask <<= Shift;
      else if (!SI->getOperand(0)->getType()->isSigned()) {
        Mask >>= Shift;
        Shift = 32 - Shift;
      }

      // Now check to see if the shift instruction is used by an or.
      Value *ShiftUse = *(OpUser->use_begin());
      Value *OptAndICopy = 0;
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(ShiftUse)) {
        if (BO->getOpcode() == Instruction::And && BO->hasOneUse()) {
          if (ConstantInt *ACI = dyn_cast<ConstantInt>(BO->getOperand(1))) {
            if (OptAndI) *OptAndI = BO;
            OptAndICopy = BO;
            Mask &= ACI->getRawValue();
            BO = dyn_cast<BinaryOperator>(*(BO->use_begin()));
          }
        }
        if (BO && BO->getOpcode() == Instruction::Or) {
          if (OrI) *OrI = BO;
          if (Op1User) {
            if (BO->getOperand(0) == OpUser || BO->getOperand(0) == OptAndICopy)
              *Op1User = dyn_cast<Instruction>(BO->getOperand(1));
            else
              *Op1User = dyn_cast<Instruction>(BO->getOperand(0));
          }
          return true;
        }
      }
    }
  }
  return false;
}

/// emitBitfieldInsert - turn a shift used only by an and with immediate into 
/// the rotate left word immediate then mask insert (rlwimi) instruction.
/// Patterns matched:
/// 1. or shl, and   5. or (shl-and), and   9. or and, and
/// 2. or and, shl   6. or and, (shl-and)
/// 3. or shr, and   7. or (shr-and), and
/// 4. or and, shr   8. or and, (shr-and)
bool PPC32ISel::emitBitfieldInsert(User *OpUser, unsigned DestReg) {
  // Instructions to skip if we match any of the patterns
  Instruction *Op0User, *Op1User = 0, *OptAndI = 0, *OrI = 0;
  unsigned TgtMask, InsMask, Amount = 0;
  bool matched = false;

  // We require OpUser to be an instruction to continue
  Op0User = dyn_cast<Instruction>(OpUser);
  if (0 == Op0User)
    return false;

  // Look for cases 2, 4, 6, 8, and 9
  if (isInsertAndHalf(Op0User, &Op1User, &OrI, TgtMask))
    if (Op1User)
      if (isInsertAndHalf(Op1User, 0, 0, InsMask))
        matched = true;
      else if (isInsertShiftHalf(Op1User, 0, 0, &OptAndI, Amount, InsMask))
        matched = true;
  
  // Look for cases 1, 3, 5, and 7.  Force the shift argument to be the one
  // inserted into the target, since rlwimi can only rotate the value inserted,
  // not the value being inserted into.
  if (matched == false)
    if (isInsertShiftHalf(Op0User, &Op1User, &OrI, &OptAndI, Amount, InsMask))
      if (Op1User && isInsertAndHalf(Op1User, 0, 0, TgtMask)) {
        std::swap(Op0User, Op1User);
        matched = true;
      }
  
  // We didn't succeed in matching one of the patterns, so return false
  if (matched == false)
    return false;
  
  // If the masks xor to -1, and the insert mask is a run of ones, then we have
  // succeeded in matching one of the cases for generating rlwimi.  Update the
  // skip lists and users of the Instruction::Or.
  unsigned MB, ME;
  if (((TgtMask ^ InsMask) == 0xFFFFFFFF) && isRunOfOnes(InsMask, MB, ME)) {
    SkipList.push_back(Op0User);
    SkipList.push_back(Op1User);
    SkipList.push_back(OptAndI);
    InsertMap[OrI] = RlwimiRec(Op0User->getOperand(0), Op1User->getOperand(0), 
                               Amount, MB, ME);
    return true;
  }
  return false;
}

/// emitBitfieldExtract - turn a shift used only by an and with immediate into the
/// rotate left word immediate then and with mask (rlwinm) instruction.
bool PPC32ISel::emitBitfieldExtract(MachineBasicBlock *MBB, 
                                    MachineBasicBlock::iterator IP,
                                    User *OpUser, unsigned DestReg) {
  return false;
  /*
  // Instructions to skip if we match any of the patterns
  Instruction *Op0User, *Op1User = 0;
  unsigned ShiftMask, AndMask, Amount = 0;
  bool matched = false;

  // We require OpUser to be an instruction to continue
  Op0User = dyn_cast<Instruction>(OpUser);
  if (0 == Op0User)
    return false;

  if (isExtractShiftHalf)
    if (isExtractAndHalf)
      matched = true;
  
  if (matched == false && isExtractAndHalf)
    if (isExtractShiftHalf)
    matched = true;
  
  if (matched == false)
    return false;

  if (isRunOfOnes(Imm, MB, ME)) {
    unsigned SrcReg = getReg(Op, MBB, IP);
    BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg).addImm(Rotate)
      .addImm(MB).addImm(ME);
    Op1User->replaceAllUsesWith(Op0User);
    SkipList.push_back(BO);
    return true;
  }
  */
}

/// emitBinaryConstOperation - Implement simple binary operators for integral
/// types with a constant operand.  Opcode is one of: 0 for Add, 1 for Sub, 
/// 2 for And, 3 for Or, 4 for Xor, and 5 for Subtract-From.
///
void PPC32ISel::emitBinaryConstOperation(MachineBasicBlock *MBB, 
                                         MachineBasicBlock::iterator IP,
                                         unsigned Op0Reg, ConstantInt *Op1, 
                                         unsigned Opcode, unsigned DestReg) {
  static const unsigned OpTab[] = {
    PPC::ADD, PPC::SUB, PPC::AND, PPC::OR, PPC::XOR, PPC::SUBF
  };
  static const unsigned ImmOpTab[2][6] = {
    {  PPC::ADDI,  PPC::ADDI,  PPC::ANDIo,  PPC::ORI,  PPC::XORI, PPC::SUBFIC },
    { PPC::ADDIS, PPC::ADDIS, PPC::ANDISo, PPC::ORIS, PPC::XORIS, PPC::SUBFIC }
  };

  // Handle subtract now by inverting the constant value: X-4 == X+(-4)
  if (Opcode == 1) {
    Op1 = cast<ConstantInt>(ConstantExpr::getNeg(Op1));
    Opcode = 0;
  }
  
  // xor X, -1 -> not X
  if (Opcode == 4 && Op1->isAllOnesValue()) {
    BuildMI(*MBB, IP, PPC::NOR, 2, DestReg).addReg(Op0Reg).addReg(Op0Reg);
    return;
  }
  
  if (Opcode == 2 && !Op1->isNullValue()) {
    unsigned MB, ME, mask = Op1->getRawValue();
    if (isRunOfOnes(mask, MB, ME)) {
      BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(Op0Reg).addImm(0)
        .addImm(MB).addImm(ME);
      return;
    }
  }

  // PowerPC 16 bit signed immediates are sign extended before use by the
  // instruction.  Therefore, we can only split up an add of a reg with a 32 bit
  // immediate into addis and addi if the sign bit of the low 16 bits is cleared
  // so that for register A, const imm X, we don't end up with
  // A + XXXX0000 + FFFFXXXX.
  bool WontSignExtend = (0 == (Op1->getRawValue() & 0x8000));

  // For Add, Sub, and SubF the instruction takes a signed immediate.  For And,
  // Or, and Xor, the instruction takes an unsigned immediate.  There is no 
  // shifted immediate form of SubF so disallow its opcode for those constants.
  if (canUseAsImmediateForOpcode(Op1, Opcode, false)) {
    if (Opcode < 2 || Opcode == 5)
      BuildMI(*MBB, IP, ImmOpTab[0][Opcode], 2, DestReg).addReg(Op0Reg)
        .addSImm(Op1->getRawValue());
    else
      BuildMI(*MBB, IP, ImmOpTab[0][Opcode], 2, DestReg).addReg(Op0Reg)
        .addZImm(Op1->getRawValue());
  } else if (canUseAsImmediateForOpcode(Op1, Opcode, true) && (Opcode < 5)) {
    if (Opcode < 2)
      BuildMI(*MBB, IP, ImmOpTab[1][Opcode], 2, DestReg).addReg(Op0Reg)
        .addSImm(Op1->getRawValue() >> 16);
    else
      BuildMI(*MBB, IP, ImmOpTab[1][Opcode], 2, DestReg).addReg(Op0Reg)
        .addZImm(Op1->getRawValue() >> 16);
  } else if ((Opcode < 2 && WontSignExtend) || Opcode == 3 || Opcode == 4) {
    unsigned TmpReg = makeAnotherReg(Op1->getType());
    if (Opcode < 2) {
      BuildMI(*MBB, IP, ImmOpTab[1][Opcode], 2, TmpReg).addReg(Op0Reg)
        .addSImm(Op1->getRawValue() >> 16);
      BuildMI(*MBB, IP, ImmOpTab[0][Opcode], 2, DestReg).addReg(TmpReg)
        .addSImm(Op1->getRawValue());
    } else {
      BuildMI(*MBB, IP, ImmOpTab[1][Opcode], 2, TmpReg).addReg(Op0Reg)
        .addZImm(Op1->getRawValue() >> 16);
      BuildMI(*MBB, IP, ImmOpTab[0][Opcode], 2, DestReg).addReg(TmpReg)
        .addZImm(Op1->getRawValue());
    }
  } else {
    unsigned Op1Reg = getReg(Op1, MBB, IP);
    BuildMI(*MBB, IP, OpTab[Opcode], 2, DestReg).addReg(Op0Reg).addReg(Op1Reg);
  }
}

/// emitSimpleBinaryOperation - Implement simple binary operators for integral
/// types...  OperatorClass is one of: 0 for Add, 1 for Sub, 2 for And, 3 for
/// Or, 4 for Xor.
///
void PPC32ISel::emitSimpleBinaryOperation(MachineBasicBlock *MBB,
                                          MachineBasicBlock::iterator IP,
                                          BinaryOperator *BO, 
                                          Value *Op0, Value *Op1,
                                          unsigned OperatorClass, 
                                          unsigned DestReg) {
  // Arithmetic and Bitwise operators
  static const unsigned OpcodeTab[] = {
    PPC::ADD, PPC::SUB, PPC::AND, PPC::OR, PPC::XOR
  };
  static const unsigned LongOpTab[2][5] = {
    { PPC::ADDC,  PPC::SUBC, PPC::AND, PPC::OR, PPC::XOR },
    { PPC::ADDE, PPC::SUBFE, PPC::AND, PPC::OR, PPC::XOR }
  };
  
  unsigned Class = getClassB(Op0->getType());

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
          BuildMI(*MBB, IP, PPC::MFCR, TmpReg);
          BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(TmpReg).addImm(4)
            .addImm(31).addImm(31);
          return;
        }
  }

  // Special case: op <const int>, Reg
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op0))
    if (Class != cLong) {
      unsigned Opcode = (OperatorClass == 1) ? 5 : OperatorClass;
      unsigned Op1r = getReg(Op1, MBB, IP);
      emitBinaryConstOperation(MBB, IP, Op1r, CI, Opcode, DestReg);
      return;
    }
  // Special case: op Reg, <const int>
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1))
    if (Class != cLong) {
      if (emitBitfieldInsert(BO, DestReg))
        return;
      
      unsigned Op0r = getReg(Op0, MBB, IP);
      emitBinaryConstOperation(MBB, IP, Op0r, CI, OperatorClass, DestReg);
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
    BuildMI(*MBB, IP, LongOpTab[0][OperatorClass], 2, DestReg+1).addReg(Op0r+1)
      .addReg(Op1r+1);
    BuildMI(*MBB, IP, LongOpTab[1][OperatorClass], 2, DestReg).addReg(Op0r)
      .addReg(Op1r);
  }
  return;
}

/// doMultiply - Emit appropriate instructions to multiply together the
/// Values Op0 and Op1, and put the result in DestReg.
///
void PPC32ISel::doMultiply(MachineBasicBlock *MBB,
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
    BuildMI(*MBB, IP, PPC::MULHWU, 2, Tmp1).addReg(Op0r+1).addReg(Op1r+1);
    BuildMI(*MBB, IP, PPC::MULLW, 2, DestReg+1).addReg(Op0r+1).addReg(Op1r+1);
    BuildMI(*MBB, IP, PPC::MULLW, 2, Tmp2).addReg(Op0r+1).addReg(Op1r);
    BuildMI(*MBB, IP, PPC::ADD, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);
    BuildMI(*MBB, IP, PPC::MULLW, 2, Tmp4).addReg(Op0r).addReg(Op1r+1);
    BuildMI(*MBB, IP, PPC::ADD, 2, DestReg).addReg(Tmp3).addReg(Tmp4);
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
      BuildMI(*MBB, IP, PPC::SRAWI, 2, Tmp0).addReg(Op1r).addImm(31);
    else
      BuildMI(*MBB, IP, PPC::LI, 2, Tmp0).addSImm(0);
    BuildMI(*MBB, IP, PPC::MULHWU, 2, Tmp1).addReg(Op0r+1).addReg(Op1r);
    BuildMI(*MBB, IP, PPC::MULLW, 2, DestReg+1).addReg(Op0r+1).addReg(Op1r);
    BuildMI(*MBB, IP, PPC::MULLW, 2, Tmp2).addReg(Op0r+1).addReg(Tmp0);
    BuildMI(*MBB, IP, PPC::ADD, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);
    BuildMI(*MBB, IP, PPC::MULLW, 2, Tmp4).addReg(Op0r).addReg(Op1r);
    BuildMI(*MBB, IP, PPC::ADD, 2, DestReg).addReg(Tmp3).addReg(Tmp4);
    return;
  }
  
  // 32 x 32 -> 32
  if (Class0 <= cInt && Class1 <= cInt) {
    BuildMI(*MBB, IP, PPC::MULLW, 2, DestReg).addReg(Op0r).addReg(Op1r);
    return;
  }
  
  assert(0 && "doMultiply cannot operate on unknown type!");
}

/// doMultiplyConst - This method will multiply the value in Op0 by the
/// value of the ContantInt *CI
void PPC32ISel::doMultiplyConst(MachineBasicBlock *MBB,
                                MachineBasicBlock::iterator IP,
                                unsigned DestReg, Value *Op0, ConstantInt *CI) {
  unsigned Class = getClass(Op0->getType());

  // Mul op0, 0 ==> 0
  if (CI->isNullValue()) {
    BuildMI(*MBB, IP, PPC::LI, 1, DestReg).addSImm(0);
    if (Class == cLong)
      BuildMI(*MBB, IP, PPC::LI, 1, DestReg+1).addSImm(0);
    return;
  }
  
  // Mul op0, 1 ==> op0
  if (CI->equalsInt(1)) {
    unsigned Op0r = getReg(Op0, MBB, IP);
    BuildMI(*MBB, IP, PPC::OR, 2, DestReg).addReg(Op0r).addReg(Op0r);
    if (Class == cLong)
      BuildMI(*MBB, IP, PPC::OR, 2, DestReg+1).addReg(Op0r+1).addReg(Op0r+1);
    return;
  }

  // If the element size is exactly a power of 2, use a shift to get it.
  if (unsigned Shift = ExactLog2(CI->getRawValue())) {
    ConstantUInt *ShiftCI = ConstantUInt::get(Type::UByteTy, Shift);
    emitShiftOperation(MBB, IP, Op0, ShiftCI, true, Op0->getType(), 0, DestReg);
    return;
  }
  
  // If 32 bits or less and immediate is in right range, emit mul by immediate
  if (Class == cByte || Class == cShort || Class == cInt) {
    if (canUseAsImmediateForOpcode(CI, 0, false)) {
      unsigned Op0r = getReg(Op0, MBB, IP);
      unsigned imm = CI->getRawValue() & 0xFFFF;
      BuildMI(*MBB, IP, PPC::MULLI, 2, DestReg).addReg(Op0r).addSImm(imm);
      return;
    }
  }
  
  doMultiply(MBB, IP, DestReg, Op0, CI);
}

void PPC32ISel::visitMul(BinaryOperator &I) {
  unsigned ResultReg = getReg(I);

  Value *Op0 = I.getOperand(0);
  Value *Op1 = I.getOperand(1);

  MachineBasicBlock::iterator IP = BB->end();
  emitMultiply(BB, IP, Op0, Op1, ResultReg);
}

void PPC32ISel::emitMultiply(MachineBasicBlock *MBB,
                             MachineBasicBlock::iterator IP,
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
void PPC32ISel::visitDivRem(BinaryOperator &I) {
  unsigned ResultReg = getReg(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  MachineBasicBlock::iterator IP = BB->end();
  emitDivRemOperation(BB, IP, Op0, Op1, I.getOpcode() == Instruction::Div,
                      ResultReg);
}

void PPC32ISel::emitDivRemOperation(MachineBasicBlock *MBB,
                                    MachineBasicBlock::iterator IP,
                                    Value *Op0, Value *Op1, bool isDiv,
                                    unsigned ResultReg) {
  const Type *Ty = Op0->getType();
  unsigned Class = getClass(Ty);
  switch (Class) {
  case cFP32:
    if (isDiv) {
      // Floating point divide...
      emitBinaryFPOperation(MBB, IP, Op0, Op1, 3, ResultReg);
      return;
    } else {
      // Floating point remainder via fmodf(float x, float y);
      unsigned Op0Reg = getReg(Op0, MBB, IP);
      unsigned Op1Reg = getReg(Op1, MBB, IP);
      MachineInstr *TheCall =
        BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(fmodfFn, true);
      std::vector<ValueRecord> Args;
      Args.push_back(ValueRecord(Op0Reg, Type::FloatTy));
      Args.push_back(ValueRecord(Op1Reg, Type::FloatTy));
      doCall(ValueRecord(ResultReg, Type::FloatTy), TheCall, Args, false);
    }
    return;
  case cFP64:
    if (isDiv) {
      // Floating point divide...
      emitBinaryFPOperation(MBB, IP, Op0, Op1, 3, ResultReg);
      return;
    } else {               
      // Floating point remainder via fmod(double x, double y);
      unsigned Op0Reg = getReg(Op0, MBB, IP);
      unsigned Op1Reg = getReg(Op1, MBB, IP);
      MachineInstr *TheCall =
        BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(fmodFn, true);
      std::vector<ValueRecord> Args;
      Args.push_back(ValueRecord(Op0Reg, Type::DoubleTy));
      Args.push_back(ValueRecord(Op1Reg, Type::DoubleTy));
      doCall(ValueRecord(ResultReg, Type::DoubleTy), TheCall, Args, false);
    }
    return;
  case cLong: {
    static Function* const Funcs[] =
      { __moddi3Fn, __divdi3Fn, __umoddi3Fn, __udivdi3Fn };
    unsigned Op0Reg = getReg(Op0, MBB, IP);
    unsigned Op1Reg = getReg(Op1, MBB, IP);
    unsigned NameIdx = Ty->isUnsigned()*2 + isDiv;
    MachineInstr *TheCall =
      BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(Funcs[NameIdx], true);

    std::vector<ValueRecord> Args;
    Args.push_back(ValueRecord(Op0Reg, Type::LongTy));
    Args.push_back(ValueRecord(Op1Reg, Type::LongTy));
    doCall(ValueRecord(ResultReg, Type::LongTy), TheCall, Args, false);
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
        unsigned Op0Reg = getReg(Op0, MBB, IP);
        BuildMI(*MBB, IP, PPC::OR, 2, ResultReg).addReg(Op0Reg).addReg(Op0Reg);
        return;
      }

      if (V == -1) {      // X /s -1 => -X
        unsigned Op0Reg = getReg(Op0, MBB, IP);
        BuildMI(*MBB, IP, PPC::NEG, 1, ResultReg).addReg(Op0Reg);
        return;
      }

      unsigned log2V = ExactLog2(V);
      if (log2V != 0 && Ty->isSigned()) {
        unsigned Op0Reg = getReg(Op0, MBB, IP);
        unsigned TmpReg = makeAnotherReg(Op0->getType());
        
        BuildMI(*MBB, IP, PPC::SRAWI, 2, TmpReg).addReg(Op0Reg).addImm(log2V);
        BuildMI(*MBB, IP, PPC::ADDZE, 1, ResultReg).addReg(TmpReg);
        return;
      }
    }

  unsigned Op0Reg = getReg(Op0, MBB, IP);

  if (isDiv) {
    unsigned Op1Reg = getReg(Op1, MBB, IP);
    unsigned Opcode = Ty->isSigned() ? PPC::DIVW : PPC::DIVWU;
    BuildMI(*MBB, IP, Opcode, 2, ResultReg).addReg(Op0Reg).addReg(Op1Reg);
  } else { // Remainder
    // FIXME: don't load the CI part of a CI divide twice
    ConstantInt *CI = dyn_cast<ConstantInt>(Op1);
    unsigned TmpReg1 = makeAnotherReg(Op0->getType());
    unsigned TmpReg2 = makeAnotherReg(Op0->getType());
    emitDivRemOperation(MBB, IP, Op0, Op1, true, TmpReg1);
    if (CI && canUseAsImmediateForOpcode(CI, 0, false)) {
      BuildMI(*MBB, IP, PPC::MULLI, 2, TmpReg2).addReg(TmpReg1)
        .addSImm(CI->getRawValue());
    } else {
      unsigned Op1Reg = getReg(Op1, MBB, IP);
      BuildMI(*MBB, IP, PPC::MULLW, 2, TmpReg2).addReg(TmpReg1).addReg(Op1Reg);
    }
    BuildMI(*MBB, IP, PPC::SUBF, 2, ResultReg).addReg(TmpReg2).addReg(Op0Reg);
  }
}


/// Shift instructions: 'shl', 'sar', 'shr' - Some special cases here
/// for constant immediate shift values, and for constant immediate
/// shift values equal to 1. Even the general case is sort of special,
/// because the shift amount has to be in CL, not just any old register.
///
void PPC32ISel::visitShiftInst(ShiftInst &I) {
  if (std::find(SkipList.begin(), SkipList.end(), &I) != SkipList.end())
    return;

  MachineBasicBlock::iterator IP = BB->end();
  emitShiftOperation(BB, IP, I.getOperand(0), I.getOperand(1),
                     I.getOpcode() == Instruction::Shl, I.getType(),
                     &I, getReg(I));
}

/// emitShiftOperation - Common code shared between visitShiftInst and
/// constant expression support.
///
void PPC32ISel::emitShiftOperation(MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator IP,
                                   Value *Op, Value *ShiftAmount, 
                                   bool isLeftShift, const Type *ResultTy,
                                   ShiftInst *SI, unsigned DestReg) {
  bool isSigned = ResultTy->isSigned ();
  unsigned Class = getClass (ResultTy);
  
  // Longs, as usual, are handled specially...
  if (Class == cLong) {
    unsigned SrcReg = getReg (Op, MBB, IP);
    // If we have a constant shift, we can generate much more efficient code
    // than for a variable shift by using the rlwimi instruction.
    if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(ShiftAmount)) {
      unsigned Amount = CUI->getValue();
      if (Amount == 0) {
        BuildMI(*MBB, IP, PPC::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
        BuildMI(*MBB, IP, PPC::OR, 2, DestReg+1)
          .addReg(SrcReg+1).addReg(SrcReg+1);

      } else if (Amount < 32) {
        unsigned TempReg = makeAnotherReg(ResultTy);
        if (isLeftShift) {
          BuildMI(*MBB, IP, PPC::RLWINM, 4, TempReg).addReg(SrcReg)
            .addImm(Amount).addImm(0).addImm(31-Amount);
          BuildMI(*MBB, IP, PPC::RLWIMI, 5, DestReg).addReg(TempReg)
            .addReg(SrcReg+1).addImm(Amount).addImm(32-Amount).addImm(31);
          BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg+1).addReg(SrcReg+1)
            .addImm(Amount).addImm(0).addImm(31-Amount);
        } else {
          BuildMI(*MBB, IP, PPC::RLWINM, 4, TempReg).addReg(SrcReg+1)
            .addImm(32-Amount).addImm(Amount).addImm(31);
          BuildMI(*MBB, IP, PPC::RLWIMI, 5, DestReg+1).addReg(TempReg)
            .addReg(SrcReg).addImm(32-Amount).addImm(0).addImm(Amount-1);
          BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg)
            .addImm(32-Amount).addImm(Amount).addImm(31);
        }
      } else {                 // Shifting more than 32 bits
        Amount -= 32;
        if (isLeftShift) {
          if (Amount != 0) {
            BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg+1)
              .addImm(Amount).addImm(0).addImm(31-Amount);
          } else {
            BuildMI(*MBB, IP, PPC::OR, 2, DestReg).addReg(SrcReg+1)
              .addReg(SrcReg+1);
          }
          BuildMI(*MBB, IP, PPC::LI, 1, DestReg+1).addSImm(0);
        } else {
          if (Amount != 0) {
            if (isSigned)
              BuildMI(*MBB, IP, PPC::SRAWI, 2, DestReg+1).addReg(SrcReg)
                .addImm(Amount);
            else
              BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg+1).addReg(SrcReg)
                .addImm(32-Amount).addImm(Amount).addImm(31);
          } else {
            BuildMI(*MBB, IP, PPC::OR, 2, DestReg+1).addReg(SrcReg)
              .addReg(SrcReg);
          }
          BuildMI(*MBB, IP,PPC::LI, 1, DestReg).addSImm(0);
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
        BuildMI(*MBB, IP, PPC::SUBFIC, 2, TmpReg1).addReg(ShiftAmountReg)
          .addSImm(32);
        BuildMI(*MBB, IP, PPC::SLW, 2, TmpReg2).addReg(SrcReg)
          .addReg(ShiftAmountReg);
        BuildMI(*MBB, IP, PPC::SRW, 2, TmpReg3).addReg(SrcReg+1)
          .addReg(TmpReg1);
        BuildMI(*MBB, IP, PPC::OR, 2,TmpReg4).addReg(TmpReg2).addReg(TmpReg3);
        BuildMI(*MBB, IP, PPC::ADDI, 2, TmpReg5).addReg(ShiftAmountReg)
          .addSImm(-32);
        BuildMI(*MBB, IP, PPC::SLW, 2, TmpReg6).addReg(SrcReg+1)
          .addReg(TmpReg5);
        BuildMI(*MBB, IP, PPC::OR, 2, DestReg).addReg(TmpReg4)
          .addReg(TmpReg6);
        BuildMI(*MBB, IP, PPC::SLW, 2, DestReg+1).addReg(SrcReg+1)
          .addReg(ShiftAmountReg);
      } else {
        if (isSigned) { // shift right algebraic 
          MachineBasicBlock *TmpMBB =new MachineBasicBlock(BB->getBasicBlock());
          MachineBasicBlock *PhiMBB =new MachineBasicBlock(BB->getBasicBlock());
          MachineBasicBlock *OldMBB = BB;
          ilist<MachineBasicBlock>::iterator It = BB; ++It;
          F->getBasicBlockList().insert(It, TmpMBB);
          F->getBasicBlockList().insert(It, PhiMBB);
          BB->addSuccessor(TmpMBB);
          BB->addSuccessor(PhiMBB);

          BuildMI(*MBB, IP, PPC::SUBFIC, 2, TmpReg1).addReg(ShiftAmountReg)
            .addSImm(32);
          BuildMI(*MBB, IP, PPC::SRW, 2, TmpReg2).addReg(SrcReg+1)
            .addReg(ShiftAmountReg);
          BuildMI(*MBB, IP, PPC::SLW, 2, TmpReg3).addReg(SrcReg)
            .addReg(TmpReg1);
          BuildMI(*MBB, IP, PPC::OR, 2, TmpReg4).addReg(TmpReg2)
            .addReg(TmpReg3);
          BuildMI(*MBB, IP, PPC::ADDICo, 2, TmpReg5).addReg(ShiftAmountReg)
            .addSImm(-32);
          BuildMI(*MBB, IP, PPC::SRAW, 2, TmpReg6).addReg(SrcReg)
            .addReg(TmpReg5);
          BuildMI(*MBB, IP, PPC::SRAW, 2, DestReg).addReg(SrcReg)
            .addReg(ShiftAmountReg);
          BuildMI(*MBB, IP, PPC::BLE, 2).addReg(PPC::CR0).addMBB(PhiMBB);
 
          // OrMBB:
          //   Select correct least significant half if the shift amount > 32
          BB = TmpMBB;
          unsigned OrReg = makeAnotherReg(Type::IntTy);
          BuildMI(BB, PPC::OR, 2, OrReg).addReg(TmpReg6).addReg(TmpReg6);
          TmpMBB->addSuccessor(PhiMBB);
          
          BB = PhiMBB;
          BuildMI(BB, PPC::PHI, 4, DestReg+1).addReg(TmpReg4).addMBB(OldMBB)
            .addReg(OrReg).addMBB(TmpMBB);
        } else { // shift right logical
          BuildMI(*MBB, IP, PPC::SUBFIC, 2, TmpReg1).addReg(ShiftAmountReg)
            .addSImm(32);
          BuildMI(*MBB, IP, PPC::SRW, 2, TmpReg2).addReg(SrcReg+1)
            .addReg(ShiftAmountReg);
          BuildMI(*MBB, IP, PPC::SLW, 2, TmpReg3).addReg(SrcReg)
            .addReg(TmpReg1);
          BuildMI(*MBB, IP, PPC::OR, 2, TmpReg4).addReg(TmpReg2)
            .addReg(TmpReg3);
          BuildMI(*MBB, IP, PPC::ADDI, 2, TmpReg5).addReg(ShiftAmountReg)
            .addSImm(-32);
          BuildMI(*MBB, IP, PPC::SRW, 2, TmpReg6).addReg(SrcReg)
            .addReg(TmpReg5);
          BuildMI(*MBB, IP, PPC::OR, 2, DestReg+1).addReg(TmpReg4)
            .addReg(TmpReg6);
          BuildMI(*MBB, IP, PPC::SRW, 2, DestReg).addReg(SrcReg)
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
    
    // If this is a shift with one use, and that use is an And instruction,
    // then attempt to emit a bitfield operation.
    if (SI && emitBitfieldInsert(SI, DestReg))
      return;
    
    unsigned SrcReg = getReg (Op, MBB, IP);
    if (Amount == 0) {
      BuildMI(*MBB, IP, PPC::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
    } else if (isLeftShift) {
      BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg)
        .addImm(Amount).addImm(0).addImm(31-Amount);
    } else {
      if (isSigned) {
        BuildMI(*MBB, IP, PPC::SRAWI,2,DestReg).addReg(SrcReg).addImm(Amount);
      } else {
        BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg)
          .addImm(32-Amount).addImm(Amount).addImm(31);
      }
    }
  } else {                  // The shift amount is non-constant.
    unsigned SrcReg = getReg (Op, MBB, IP);
    unsigned ShiftAmountReg = getReg (ShiftAmount, MBB, IP);

    if (isLeftShift) {
      BuildMI(*MBB, IP, PPC::SLW, 2, DestReg).addReg(SrcReg)
        .addReg(ShiftAmountReg);
    } else {
      BuildMI(*MBB, IP, isSigned ? PPC::SRAW : PPC::SRW, 2, DestReg)
        .addReg(SrcReg).addReg(ShiftAmountReg);
    }
  }
}

/// LoadNeedsSignExtend - On PowerPC, there is no load byte with sign extend.
/// Therefore, if this is a byte load and the destination type is signed, we
/// would normally need to also emit a sign extend instruction after the load.
/// However, store instructions don't care whether a signed type was sign
/// extended across a whole register.  Also, a SetCC instruction will emit its
/// own sign extension to force the value into the appropriate range, so we
/// need not emit it here.  Ideally, this kind of thing wouldn't be necessary
/// once LLVM's type system is improved.
static bool LoadNeedsSignExtend(LoadInst &LI) {
  if (cByte == getClassB(LI.getType()) && LI.getType()->isSigned()) {
    bool AllUsesAreStoresOrSetCC = true;
    for (Value::use_iterator I = LI.use_begin(), E = LI.use_end(); I != E; ++I){
      if (isa<SetCondInst>(*I))
        continue;
      if (StoreInst *SI = dyn_cast<StoreInst>(*I))
        if (cByte == getClassB(SI->getOperand(0)->getType()))
        continue;
      AllUsesAreStoresOrSetCC = false;
      break;
    }
    if (!AllUsesAreStoresOrSetCC)
      return true;
  }
  return false;
}

/// visitLoadInst - Implement LLVM load instructions.  Pretty straightforward
/// mapping of LLVM classes to PPC load instructions, with the exception of
/// signed byte loads, which need a sign extension following them.
///
void PPC32ISel::visitLoadInst(LoadInst &I) {
  // Immediate opcodes, for reg+imm addressing
  static const unsigned ImmOpcodes[] = { 
    PPC::LBZ, PPC::LHZ, PPC::LWZ, 
    PPC::LFS, PPC::LFD, PPC::LWZ
  };
  // Indexed opcodes, for reg+reg addressing
  static const unsigned IdxOpcodes[] = {
    PPC::LBZX, PPC::LHZX, PPC::LWZX,
    PPC::LFSX, PPC::LFDX, PPC::LWZX
  };

  unsigned Class     = getClassB(I.getType());
  unsigned ImmOpcode = ImmOpcodes[Class];
  unsigned IdxOpcode = IdxOpcodes[Class];
  unsigned DestReg   = getReg(I);
  Value *SourceAddr  = I.getOperand(0);
  
  if (Class == cShort && I.getType()->isSigned()) ImmOpcode = PPC::LHA;
  if (Class == cShort && I.getType()->isSigned()) IdxOpcode = PPC::LHAX;

  // If this is a fixed size alloca, emit a load directly from the stack slot
  // corresponding to it.
  if (AllocaInst *AI = dyn_castFixedAlloca(SourceAddr)) {
    unsigned FI = getFixedSizedAllocaFI(AI);
    if (Class == cLong) {
      addFrameReference(BuildMI(BB, ImmOpcode, 2, DestReg), FI);
      addFrameReference(BuildMI(BB, ImmOpcode, 2, DestReg+1), FI, 4);
    } else if (LoadNeedsSignExtend(I)) {
      unsigned TmpReg = makeAnotherReg(I.getType());
      addFrameReference(BuildMI(BB, ImmOpcode, 2, TmpReg), FI);
      BuildMI(BB, PPC::EXTSB, 1, DestReg).addReg(TmpReg);
    } else {
      addFrameReference(BuildMI(BB, ImmOpcode, 2, DestReg), FI);
    }
    return;
  }
  
  // If the offset fits in 16 bits, we can emit a reg+imm load, otherwise, we
  // use the index from the FoldedGEP struct and use reg+reg addressing.
  if (GetElementPtrInst *GEPI = canFoldGEPIntoLoadOrStore(SourceAddr)) {

    // Generate the code for the GEP and get the components of the folded GEP
    emitGEPOperation(BB, BB->end(), GEPI, true);
    unsigned baseReg = GEPMap[GEPI].base;
    unsigned indexReg = GEPMap[GEPI].index;
    ConstantSInt *offset = GEPMap[GEPI].offset;

    if (Class != cLong) {
      unsigned TmpReg = LoadNeedsSignExtend(I) ? makeAnotherReg(I.getType())
                                               : DestReg;
      if (indexReg == 0)
        BuildMI(BB, ImmOpcode, 2, TmpReg).addSImm(offset->getValue())
          .addReg(baseReg);
      else
        BuildMI(BB, IdxOpcode, 2, TmpReg).addReg(indexReg).addReg(baseReg);
      if (LoadNeedsSignExtend(I))
        BuildMI(BB, PPC::EXTSB, 1, DestReg).addReg(TmpReg);
    } else {
      indexReg = (indexReg != 0) ? indexReg : getReg(offset);
      unsigned indexPlus4 = makeAnotherReg(Type::IntTy);
      BuildMI(BB, PPC::ADDI, 2, indexPlus4).addReg(indexReg).addSImm(4);
      BuildMI(BB, IdxOpcode, 2, DestReg).addReg(indexReg).addReg(baseReg);
      BuildMI(BB, IdxOpcode, 2, DestReg+1).addReg(indexPlus4).addReg(baseReg);
    }
    return;
  }
  
  // The fallback case, where the load was from a source that could not be
  // folded into the load instruction. 
  unsigned SrcAddrReg = getReg(SourceAddr);
    
  if (Class == cLong) {
    BuildMI(BB, ImmOpcode, 2, DestReg).addSImm(0).addReg(SrcAddrReg);
    BuildMI(BB, ImmOpcode, 2, DestReg+1).addSImm(4).addReg(SrcAddrReg);
  } else if (LoadNeedsSignExtend(I)) {
    unsigned TmpReg = makeAnotherReg(I.getType());
    BuildMI(BB, ImmOpcode, 2, TmpReg).addSImm(0).addReg(SrcAddrReg);
    BuildMI(BB, PPC::EXTSB, 1, DestReg).addReg(TmpReg);
  } else {
    BuildMI(BB, ImmOpcode, 2, DestReg).addSImm(0).addReg(SrcAddrReg);
  }
}

/// visitStoreInst - Implement LLVM store instructions
///
void PPC32ISel::visitStoreInst(StoreInst &I) {
  // Immediate opcodes, for reg+imm addressing
  static const unsigned ImmOpcodes[] = {
    PPC::STB, PPC::STH, PPC::STW, 
    PPC::STFS, PPC::STFD, PPC::STW
  };
  // Indexed opcodes, for reg+reg addressing
  static const unsigned IdxOpcodes[] = {
    PPC::STBX, PPC::STHX, PPC::STWX, 
    PPC::STFSX, PPC::STFDX, PPC::STWX
  };
  
  Value *SourceAddr  = I.getOperand(1);
  const Type *ValTy  = I.getOperand(0)->getType();
  unsigned Class     = getClassB(ValTy);
  unsigned ImmOpcode = ImmOpcodes[Class];
  unsigned IdxOpcode = IdxOpcodes[Class];
  unsigned ValReg    = getReg(I.getOperand(0));

  // If this is a fixed size alloca, emit a store directly to the stack slot
  // corresponding to it.
  if (AllocaInst *AI = dyn_castFixedAlloca(SourceAddr)) {
    unsigned FI = getFixedSizedAllocaFI(AI);
    addFrameReference(BuildMI(BB, ImmOpcode, 3).addReg(ValReg), FI);
    if (Class == cLong)
      addFrameReference(BuildMI(BB, ImmOpcode, 3).addReg(ValReg+1), FI, 4);
    return;
  }
  
  // If the offset fits in 16 bits, we can emit a reg+imm store, otherwise, we
  // use the index from the FoldedGEP struct and use reg+reg addressing.
  if (GetElementPtrInst *GEPI = canFoldGEPIntoLoadOrStore(SourceAddr)) {
    // Generate the code for the GEP and get the components of the folded GEP
    emitGEPOperation(BB, BB->end(), GEPI, true);
    unsigned baseReg = GEPMap[GEPI].base;
    unsigned indexReg = GEPMap[GEPI].index;
    ConstantSInt *offset = GEPMap[GEPI].offset;
    
    if (Class != cLong) {
      if (indexReg == 0)
        BuildMI(BB, ImmOpcode, 3).addReg(ValReg).addSImm(offset->getValue())
          .addReg(baseReg);
      else
        BuildMI(BB, IdxOpcode, 3).addReg(ValReg).addReg(indexReg)
          .addReg(baseReg);
    } else {
      indexReg = (indexReg != 0) ? indexReg : getReg(offset);
      unsigned indexPlus4 = makeAnotherReg(Type::IntTy);
      BuildMI(BB, PPC::ADDI, 2, indexPlus4).addReg(indexReg).addSImm(4);
      BuildMI(BB, IdxOpcode, 3).addReg(ValReg).addReg(indexReg).addReg(baseReg);
      BuildMI(BB, IdxOpcode, 3).addReg(ValReg+1).addReg(indexPlus4)
        .addReg(baseReg);
    }
    return;
  }
  
  // If the store address wasn't the only use of a GEP, we fall back to the
  // standard path: store the ValReg at the value in AddressReg.
  unsigned AddressReg  = getReg(I.getOperand(1));
  if (Class == cLong) {
    BuildMI(BB, ImmOpcode, 3).addReg(ValReg).addSImm(0).addReg(AddressReg);
    BuildMI(BB, ImmOpcode, 3).addReg(ValReg+1).addSImm(4).addReg(AddressReg);
    return;
  }
  BuildMI(BB, ImmOpcode, 3).addReg(ValReg).addSImm(0).addReg(AddressReg);
}


/// visitCastInst - Here we have various kinds of copying with or without sign
/// extension going on.
///
void PPC32ISel::visitCastInst(CastInst &CI) {
  Value *Op = CI.getOperand(0);

  unsigned SrcClass = getClassB(Op->getType());
  unsigned DestClass = getClassB(CI.getType());

  // Noop casts are not emitted: getReg will return the source operand as the
  // register to use for any uses of the noop cast.
  if (DestClass == SrcClass) return;

  // If this is a cast from a 32-bit integer to a Long type, and the only uses
  // of the cast are GEP instructions, then the cast does not need to be
  // generated explicitly, it will be folded into the GEP.
  if (DestClass == cLong && SrcClass == cInt) {
    bool AllUsesAreGEPs = true;
    for (Value::use_iterator I = CI.use_begin(), E = CI.use_end(); I != E; ++I)
      if (!isa<GetElementPtrInst>(*I)) {
        AllUsesAreGEPs = false;
        break;
      }        
    if (AllUsesAreGEPs) return;
  }
  
  unsigned DestReg = getReg(CI);
  MachineBasicBlock::iterator MI = BB->end();

  // If this is a cast from an integer type to a ubyte, with one use where the
  // use is the shift amount argument of a shift instruction, just emit a move
  // instead (since the shift instruction will only look at the low 5 bits
  // regardless of how it is sign extended)
  if (CI.getType() == Type::UByteTy && SrcClass <= cInt && CI.hasOneUse()) {
    ShiftInst *SI = dyn_cast<ShiftInst>(*(CI.use_begin()));
    if (SI && (SI->getOperand(1) == &CI)) {
      unsigned SrcReg = getReg(Op, BB, MI);
      BuildMI(*BB, MI, PPC::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
      return; 
    }
  }

  // If this is a cast from an byte, short, or int to an integer type of equal
  // or lesser width, and all uses of the cast are store instructions then dont
  // emit them, as the store instruction will implicitly not store the zero or
  // sign extended bytes.
  if (SrcClass <= cInt && SrcClass >= DestClass) {
    bool AllUsesAreStores = true;
    for (Value::use_iterator I = CI.use_begin(), E = CI.use_end(); I != E; ++I)
      if (!isa<StoreInst>(*I)) {
        AllUsesAreStores = false;
        break;
      }        
    // Turn this cast directly into a move instruction, which the register
    // allocator will deal with.
    if (AllUsesAreStores) { 
      unsigned SrcReg = getReg(Op, BB, MI);
      BuildMI(*BB, MI, PPC::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
      return; 
    }
  }
  emitCastOperation(BB, MI, Op, CI.getType(), DestReg);
}

/// emitCastOperation - Common code shared between visitCastInst and constant
/// expression cast support.
///
void PPC32ISel::emitCastOperation(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator IP,
                                  Value *Src, const Type *DestTy,
                                  unsigned DestReg) {
  const Type *SrcTy = Src->getType();
  unsigned SrcClass = getClassB(SrcTy);
  unsigned DestClass = getClassB(DestTy);
  unsigned SrcReg = getReg(Src, MBB, IP);

  // Implement casts from bool to integer types as a move operation
  if (SrcTy == Type::BoolTy) {
    switch (DestClass) {
    case cByte:
    case cShort:
    case cInt:
      BuildMI(*MBB, IP, PPC::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
      return;
    case cLong:
      BuildMI(*MBB, IP, PPC::LI, 1, DestReg).addImm(0);
      BuildMI(*MBB, IP, PPC::OR, 2, DestReg+1).addReg(SrcReg).addReg(SrcReg);
      return;
    default:
      break;
    }
  }

  // Implement casts to bool by using compare on the operand followed by set if
  // not zero on the result.
  if (DestTy == Type::BoolTy) {
    switch (SrcClass) {
    case cByte:
    case cShort:
    case cInt: {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, PPC::ADDIC, 2, TmpReg).addReg(SrcReg).addSImm(-1);
      BuildMI(*MBB, IP, PPC::SUBFE, 2, DestReg).addReg(TmpReg).addReg(SrcReg);
      break;
    }
    case cLong: {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      unsigned SrcReg2 = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, PPC::OR, 2, SrcReg2).addReg(SrcReg).addReg(SrcReg+1);
      BuildMI(*MBB, IP, PPC::ADDIC, 2, TmpReg).addReg(SrcReg2).addSImm(-1);
      BuildMI(*MBB, IP, PPC::SUBFE, 2, DestReg).addReg(TmpReg)
        .addReg(SrcReg2);
      break;
    }
    case cFP32:
    case cFP64:
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      unsigned ConstZero = getReg(ConstantFP::get(Type::DoubleTy, 0.0), BB, IP);
      BuildMI(*MBB, IP, PPC::FCMPU, PPC::CR7).addReg(SrcReg).addReg(ConstZero);
      BuildMI(*MBB, IP, PPC::MFCR, TmpReg);
      BuildMI(*MBB, IP, PPC::RLWINM, DestReg).addReg(TmpReg).addImm(31)
        .addImm(31).addImm(31);
    }
    return;
  }

  // Handle cast of Float -> Double
  if (SrcClass == cFP32 && DestClass == cFP64) {
    BuildMI(*MBB, IP, PPC::FMR, 1, DestReg).addReg(SrcReg);
    return;
  }
  
  // Handle cast of Double -> Float
  if (SrcClass == cFP64 && DestClass == cFP32) {
    BuildMI(*MBB, IP, PPC::FRSP, 1, DestReg).addReg(SrcReg);
    return;
  }
  
  // Handle casts from integer to floating point now...
  if (DestClass == cFP32 || DestClass == cFP64) {

    // Emit a library call for long to float conversion
    if (SrcClass == cLong) {
      Function *floatFn = (DestClass == cFP32) ? __floatdisfFn : __floatdidfFn;
      if (SrcTy->isSigned()) {
        std::vector<ValueRecord> Args;
        Args.push_back(ValueRecord(SrcReg, SrcTy));
        MachineInstr *TheCall =
          BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(floatFn, true);
        doCall(ValueRecord(DestReg, DestTy), TheCall, Args, false);
      } else {
        std::vector<ValueRecord> CmpArgs, ClrArgs, SetArgs;
        unsigned ZeroLong = getReg(ConstantUInt::get(SrcTy, 0));
        unsigned CondReg = makeAnotherReg(Type::IntTy);

        // Update machine-CFG edges
        MachineBasicBlock *ClrMBB = new MachineBasicBlock(BB->getBasicBlock());
        MachineBasicBlock *SetMBB = new MachineBasicBlock(BB->getBasicBlock());
        MachineBasicBlock *PhiMBB = new MachineBasicBlock(BB->getBasicBlock());
        MachineBasicBlock *OldMBB = BB;
        ilist<MachineBasicBlock>::iterator It = BB; ++It;
        F->getBasicBlockList().insert(It, ClrMBB);
        F->getBasicBlockList().insert(It, SetMBB);
        F->getBasicBlockList().insert(It, PhiMBB);
        BB->addSuccessor(ClrMBB);
        BB->addSuccessor(SetMBB);

        CmpArgs.push_back(ValueRecord(SrcReg, SrcTy));
        CmpArgs.push_back(ValueRecord(ZeroLong, SrcTy));
        MachineInstr *TheCall =
          BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(__cmpdi2Fn, true);
        doCall(ValueRecord(CondReg, Type::IntTy), TheCall, CmpArgs, false);
        BuildMI(*MBB, IP, PPC::CMPWI, 2, PPC::CR0).addReg(CondReg).addSImm(0);
        BuildMI(*MBB, IP, PPC::BLE, 2).addReg(PPC::CR0).addMBB(SetMBB);

        // ClrMBB
        BB = ClrMBB;
        unsigned ClrReg = makeAnotherReg(DestTy);
        ClrArgs.push_back(ValueRecord(SrcReg, SrcTy));
        TheCall = BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(floatFn, true);
        doCall(ValueRecord(ClrReg, DestTy), TheCall, ClrArgs, false);
        BuildMI(BB, PPC::B, 1).addMBB(PhiMBB);
        BB->addSuccessor(PhiMBB);
        
        // SetMBB
        BB = SetMBB;
        unsigned SetReg = makeAnotherReg(DestTy);
        unsigned CallReg = makeAnotherReg(DestTy);
        unsigned ShiftedReg = makeAnotherReg(SrcTy);
        ConstantSInt *Const1 = ConstantSInt::get(Type::IntTy, 1);
        emitShiftOperation(BB, BB->end(), Src, Const1, false, SrcTy, 0, 
                           ShiftedReg);
        SetArgs.push_back(ValueRecord(ShiftedReg, SrcTy));
        TheCall = BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(floatFn, true);
        doCall(ValueRecord(CallReg, DestTy), TheCall, SetArgs, false);
        unsigned SetOpcode = (DestClass == cFP32) ? PPC::FADDS : PPC::FADD;
        BuildMI(BB, SetOpcode, 2, SetReg).addReg(CallReg).addReg(CallReg);
        BB->addSuccessor(PhiMBB);
        
        // PhiMBB
        BB = PhiMBB;
        BuildMI(BB, PPC::PHI, 4, DestReg).addReg(ClrReg).addMBB(ClrMBB)
          .addReg(SetReg).addMBB(SetMBB);
      }
      return;
    }
    
    // Make sure we're dealing with a full 32 bits
    if (SrcClass < cInt) {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      promote32(TmpReg, ValueRecord(SrcReg, SrcTy));
      SrcReg = TmpReg;
    }
    
    // Spill the integer to memory and reload it from there.
    // Also spill room for a special conversion constant
    int ValueFrameIdx =
      F->getFrameInfo()->CreateStackObject(Type::DoubleTy, TM.getTargetData());

    MachineConstantPool *CP = F->getConstantPool();
    unsigned constantHi = makeAnotherReg(Type::IntTy);
    unsigned TempF = makeAnotherReg(Type::DoubleTy);
    
    if (!SrcTy->isSigned()) {
      ConstantFP *CFP = ConstantFP::get(Type::DoubleTy, 0x1.000000p52);
      unsigned ConstF = getReg(CFP, BB, IP);
      BuildMI(*MBB, IP, PPC::LIS, 1, constantHi).addSImm(0x4330);
      addFrameReference(BuildMI(*MBB, IP, PPC::STW, 3).addReg(constantHi), 
                        ValueFrameIdx);
      addFrameReference(BuildMI(*MBB, IP, PPC::STW, 3).addReg(SrcReg), 
                        ValueFrameIdx, 4);
      addFrameReference(BuildMI(*MBB, IP, PPC::LFD, 2, TempF), ValueFrameIdx);
      BuildMI(*MBB, IP, PPC::FSUB, 2, DestReg).addReg(TempF).addReg(ConstF);
    } else {
      ConstantFP *CFP = ConstantFP::get(Type::DoubleTy, 0x1.000008p52);
      unsigned ConstF = getReg(CFP, BB, IP);
      unsigned TempLo = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, PPC::LIS, 1, constantHi).addSImm(0x4330);
      addFrameReference(BuildMI(*MBB, IP, PPC::STW, 3).addReg(constantHi), 
                        ValueFrameIdx);
      BuildMI(*MBB, IP, PPC::XORIS, 2, TempLo).addReg(SrcReg).addImm(0x8000);
      addFrameReference(BuildMI(*MBB, IP, PPC::STW, 3).addReg(TempLo), 
                        ValueFrameIdx, 4);
      addFrameReference(BuildMI(*MBB, IP, PPC::LFD, 2, TempF), ValueFrameIdx);
      BuildMI(*MBB, IP, PPC::FSUB, 2, DestReg).addReg(TempF).addReg(ConstF);
    }
    return;
  }

  // Handle casts from floating point to integer now...
  if (SrcClass == cFP32 || SrcClass == cFP64) {
    static Function* const Funcs[] =
      { __fixsfdiFn, __fixdfdiFn, __fixunssfdiFn, __fixunsdfdiFn };
    // emit library call
    if (DestClass == cLong) {
      bool isDouble = SrcClass == cFP64;
      unsigned nameIndex = 2 * DestTy->isSigned() + isDouble;
      std::vector<ValueRecord> Args;
      Args.push_back(ValueRecord(SrcReg, SrcTy));
      Function *floatFn = Funcs[nameIndex];
      MachineInstr *TheCall =
        BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(floatFn, true);
      doCall(ValueRecord(DestReg, DestTy), TheCall, Args, false);
      return;
    }

    int ValueFrameIdx =
      F->getFrameInfo()->CreateStackObject(Type::DoubleTy, TM.getTargetData());

    if (DestTy->isSigned()) {
      unsigned TempReg = makeAnotherReg(Type::DoubleTy);
      
      // Convert to integer in the FP reg and store it to a stack slot
      BuildMI(*MBB, IP, PPC::FCTIWZ, 1, TempReg).addReg(SrcReg);
      addFrameReference(BuildMI(*MBB, IP, PPC::STFD, 3)
                          .addReg(TempReg), ValueFrameIdx);

      // There is no load signed byte opcode, so we must emit a sign extend for
      // that particular size.  Make sure to source the new integer from the 
      // correct offset.
      if (DestClass == cByte) {
        unsigned TempReg2 = makeAnotherReg(DestTy);
        addFrameReference(BuildMI(*MBB, IP, PPC::LBZ, 2, TempReg2), 
                          ValueFrameIdx, 7);
        BuildMI(*MBB, IP, PPC::EXTSB, 1, DestReg).addReg(TempReg2);
      } else {
        int offset = (DestClass == cShort) ? 6 : 4;
        unsigned LoadOp = (DestClass == cShort) ? PPC::LHA : PPC::LWZ;
        addFrameReference(BuildMI(*MBB, IP, LoadOp, 2, DestReg), 
                          ValueFrameIdx, offset);
      }
    } else {
      unsigned Zero = getReg(ConstantFP::get(Type::DoubleTy, 0.0f));
      double maxInt = (1LL << 32) - 1;
      unsigned MaxInt = getReg(ConstantFP::get(Type::DoubleTy, maxInt));
      double border = 1LL << 31;
      unsigned Border = getReg(ConstantFP::get(Type::DoubleTy, border));
      unsigned UseZero = makeAnotherReg(Type::DoubleTy);
      unsigned UseMaxInt = makeAnotherReg(Type::DoubleTy);
      unsigned UseChoice = makeAnotherReg(Type::DoubleTy);
      unsigned TmpReg = makeAnotherReg(Type::DoubleTy);
      unsigned TmpReg2 = makeAnotherReg(Type::DoubleTy);
      unsigned ConvReg = makeAnotherReg(Type::DoubleTy);
      unsigned IntTmp = makeAnotherReg(Type::IntTy);
      unsigned XorReg = makeAnotherReg(Type::IntTy);
      int FrameIdx = 
        F->getFrameInfo()->CreateStackObject(SrcTy, TM.getTargetData());
      // Update machine-CFG edges
      MachineBasicBlock *XorMBB = new MachineBasicBlock(BB->getBasicBlock());
      MachineBasicBlock *PhiMBB = new MachineBasicBlock(BB->getBasicBlock());
      MachineBasicBlock *OldMBB = BB;
      ilist<MachineBasicBlock>::iterator It = BB; ++It;
      F->getBasicBlockList().insert(It, XorMBB);
      F->getBasicBlockList().insert(It, PhiMBB);
      BB->addSuccessor(XorMBB);
      BB->addSuccessor(PhiMBB);

      // Convert from floating point to unsigned 32-bit value
      // Use 0 if incoming value is < 0.0
      BuildMI(*MBB, IP, PPC::FSEL, 3, UseZero).addReg(SrcReg).addReg(SrcReg)
        .addReg(Zero);
      // Use 2**32 - 1 if incoming value is >= 2**32
      BuildMI(*MBB, IP, PPC::FSUB, 2, UseMaxInt).addReg(MaxInt).addReg(SrcReg);
      BuildMI(*MBB, IP, PPC::FSEL, 3, UseChoice).addReg(UseMaxInt)
        .addReg(UseZero).addReg(MaxInt);
      // Subtract 2**31
      BuildMI(*MBB, IP, PPC::FSUB, 2, TmpReg).addReg(UseChoice).addReg(Border);
      // Use difference if >= 2**31
      BuildMI(*MBB, IP, PPC::FCMPU, 2, PPC::CR0).addReg(UseChoice)
        .addReg(Border);
      BuildMI(*MBB, IP, PPC::FSEL, 3, TmpReg2).addReg(TmpReg).addReg(TmpReg)
        .addReg(UseChoice);
      // Convert to integer
      BuildMI(*MBB, IP, PPC::FCTIWZ, 1, ConvReg).addReg(TmpReg2);
      addFrameReference(BuildMI(*MBB, IP, PPC::STFD, 3).addReg(ConvReg),
                        FrameIdx);
      if (DestClass == cByte) {
        addFrameReference(BuildMI(*MBB, IP, PPC::LBZ, 2, DestReg),
                          FrameIdx, 7);
      } else if (DestClass == cShort) {
        addFrameReference(BuildMI(*MBB, IP, PPC::LHZ, 2, DestReg),
                          FrameIdx, 6);
      } if (DestClass == cInt) {
        addFrameReference(BuildMI(*MBB, IP, PPC::LWZ, 2, IntTmp),
                          FrameIdx, 4);
        BuildMI(*MBB, IP, PPC::BLT, 2).addReg(PPC::CR0).addMBB(PhiMBB);
        BuildMI(*MBB, IP, PPC::B, 1).addMBB(XorMBB);

        // XorMBB:
        //   add 2**31 if input was >= 2**31
        BB = XorMBB;
        BuildMI(BB, PPC::XORIS, 2, XorReg).addReg(IntTmp).addImm(0x8000);
        XorMBB->addSuccessor(PhiMBB);

        // PhiMBB:
        //   DestReg = phi [ IntTmp, OldMBB ], [ XorReg, XorMBB ]
        BB = PhiMBB;
        BuildMI(BB, PPC::PHI, 4, DestReg).addReg(IntTmp).addMBB(OldMBB)
          .addReg(XorReg).addMBB(XorMBB);
      }
    }
    return;
  }

  // Check our invariants
  assert((SrcClass <= cInt || SrcClass == cLong) && 
         "Unhandled source class for cast operation!");
  assert((DestClass <= cInt || DestClass == cLong) && 
         "Unhandled destination class for cast operation!");

  bool sourceUnsigned = SrcTy->isUnsigned() || SrcTy == Type::BoolTy;
  bool destUnsigned = DestTy->isUnsigned();

  // Unsigned -> Unsigned, clear if larger, 
  if (sourceUnsigned && destUnsigned) {
    // handle long dest class now to keep switch clean
    if (DestClass == cLong) {
      BuildMI(*MBB, IP, PPC::LI, 1, DestReg).addSImm(0);
      BuildMI(*MBB, IP, PPC::OR, 2, DestReg+1).addReg(SrcReg)
        .addReg(SrcReg);
      return;
    }

    // handle u{ byte, short, int } x u{ byte, short, int }
    unsigned clearBits = (SrcClass == cByte || DestClass == cByte) ? 24 : 16;
    switch (SrcClass) {
    case cByte:
    case cShort:
      BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg)
        .addImm(0).addImm(clearBits).addImm(31);
      break;
    case cLong:
      ++SrcReg;
      // Fall through
    case cInt:
      BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg)
        .addImm(0).addImm(clearBits).addImm(31);
      break;
    }
    return;
  }

  // Signed -> Signed
  if (!sourceUnsigned && !destUnsigned) {
    // handle long dest class now to keep switch clean
    if (DestClass == cLong) {
      BuildMI(*MBB, IP, PPC::SRAWI, 2, DestReg).addReg(SrcReg).addImm(31);
      BuildMI(*MBB, IP, PPC::OR, 2, DestReg+1).addReg(SrcReg)
        .addReg(SrcReg);
      return;
    }

    // handle { byte, short, int } x { byte, short, int }
    switch (SrcClass) {
    case cByte:
      BuildMI(*MBB, IP, PPC::EXTSB, 1, DestReg).addReg(SrcReg);
      break;
    case cShort:
      if (DestClass == cByte)
        BuildMI(*MBB, IP, PPC::EXTSB, 1, DestReg).addReg(SrcReg);
      else
        BuildMI(*MBB, IP, PPC::EXTSH, 1, DestReg).addReg(SrcReg);
      break;
    case cLong:
      ++SrcReg;
      // Fall through
    case cInt:
      if (DestClass == cByte)
        BuildMI(*MBB, IP, PPC::EXTSB, 1, DestReg).addReg(SrcReg);
      else if (DestClass == cShort)
        BuildMI(*MBB, IP, PPC::EXTSH, 1, DestReg).addReg(SrcReg);
      else
        BuildMI(*MBB, IP, PPC::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
      break;
    }
    return;
  }

  // Unsigned -> Signed
  if (sourceUnsigned && !destUnsigned) {
    // handle long dest class now to keep switch clean
    if (DestClass == cLong) {
      BuildMI(*MBB, IP, PPC::LI, 1, DestReg).addSImm(0);
      BuildMI(*MBB, IP, PPC::OR, 2, DestReg+1).addReg(SrcReg)
        .addReg(SrcReg);
      return;
    }

    // handle u{ byte, short, int } -> { byte, short, int }
    switch (SrcClass) {
    case cByte:
      // uByte 255 -> signed short/int == 255
      BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg).addImm(0)
        .addImm(24).addImm(31);
      break;
    case cShort:
      if (DestClass == cByte)
        BuildMI(*MBB, IP, PPC::EXTSB, 1, DestReg).addReg(SrcReg);
      else
        BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg).addImm(0)
          .addImm(16).addImm(31);
      break;
    case cLong:
      ++SrcReg;
      // Fall through
    case cInt:
      if (DestClass == cByte)
        BuildMI(*MBB, IP, PPC::EXTSB, 1, DestReg).addReg(SrcReg);
      else if (DestClass == cShort)
        BuildMI(*MBB, IP, PPC::EXTSH, 1, DestReg).addReg(SrcReg);
      else
        BuildMI(*MBB, IP, PPC::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
      break;
    }
    return;
  }

  // Signed -> Unsigned
  if (!sourceUnsigned && destUnsigned) {
    // handle long dest class now to keep switch clean
    if (DestClass == cLong) {
      BuildMI(*MBB, IP, PPC::SRAWI, 2, DestReg).addReg(SrcReg).addImm(31);
      BuildMI(*MBB, IP, PPC::OR, 2, DestReg+1).addReg(SrcReg)
        .addReg(SrcReg);
      return;
    }

    // handle { byte, short, int } -> u{ byte, short, int }
    unsigned clearBits = (DestClass == cByte) ? 24 : 16;
    switch (SrcClass) {
    case cByte:
       BuildMI(*MBB, IP, PPC::EXTSB, 1, DestReg).addReg(SrcReg);
       break;
    case cShort:
      if (DestClass == cByte)
        BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg)
          .addImm(0).addImm(clearBits).addImm(31);
      else
        BuildMI(*MBB, IP, PPC::EXTSH, 1, DestReg).addReg(SrcReg);
      break;
    case cLong:
      ++SrcReg;
      // Fall through
    case cInt:
      if (DestClass == cInt)
        BuildMI(*MBB, IP, PPC::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
      else
        BuildMI(*MBB, IP, PPC::RLWINM, 4, DestReg).addReg(SrcReg)
          .addImm(0).addImm(clearBits).addImm(31);
      break;
    }
    return;
  }

  // Anything we haven't handled already, we can't (yet) handle at all.
  std::cerr << "Unhandled cast from " << SrcTy->getDescription()
            << "to " << DestTy->getDescription() << '\n';
  abort();
}

/// visitVANextInst - Implement the va_next instruction...
///
void PPC32ISel::visitVANextInst(VANextInst &I) {
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
  BuildMI(BB, PPC::ADDI, 2, DestReg).addReg(VAList).addSImm(Size);
}

void PPC32ISel::visitVAArgInst(VAArgInst &I) {
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
    BuildMI(BB, PPC::LWZ, 2, DestReg).addSImm(0).addReg(VAList);
    break;
  case Type::ULongTyID:
  case Type::LongTyID:
    BuildMI(BB, PPC::LWZ, 2, DestReg).addSImm(0).addReg(VAList);
    BuildMI(BB, PPC::LWZ, 2, DestReg+1).addSImm(4).addReg(VAList);
    break;
  case Type::FloatTyID:
    BuildMI(BB, PPC::LFS, 2, DestReg).addSImm(0).addReg(VAList);
    break;
  case Type::DoubleTyID:
    BuildMI(BB, PPC::LFD, 2, DestReg).addSImm(0).addReg(VAList);
    break;
  }
}

/// visitGetElementPtrInst - instruction-select GEP instructions
///
void PPC32ISel::visitGetElementPtrInst(GetElementPtrInst &I) {
  if (canFoldGEPIntoLoadOrStore(&I))
    return;

  emitGEPOperation(BB, BB->end(), &I, false);
}

/// emitGEPOperation - Common code shared between visitGetElementPtrInst and
/// constant expression GEP support.
///
void PPC32ISel::emitGEPOperation(MachineBasicBlock *MBB,
                                 MachineBasicBlock::iterator IP,
                                 GetElementPtrInst *GEPI, bool GEPIsFolded) {
  // If we've already emitted this particular GEP, just return to avoid
  // multiple definitions of the base register.
  if (GEPIsFolded && (GEPMap[GEPI].base != 0))
    return;
  
  Value *Src = GEPI->getOperand(0);
  User::op_iterator IdxBegin = GEPI->op_begin()+1;
  User::op_iterator IdxEnd = GEPI->op_end();
  const TargetData &TD = TM.getTargetData();
  const Type *Ty = Src->getType();
  int64_t constValue = 0;
  
  // Record the operations to emit the GEP in a vector so that we can emit them
  // after having analyzed the entire instruction.
  std::vector<CollapsedGepOp> ops;
  
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

      // StructType member offsets are always constant values.  Add it to the
      // running total.
      constValue += TD.getStructLayout(StTy)->MemberOffsets[fieldIndex];

      // The next type is the member of the structure selected by the index.
      Ty = StTy->getElementType (fieldIndex);
    } else if (const SequentialType *SqTy = dyn_cast<SequentialType>(Ty)) {
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
        if (ConstantSInt *CS = dyn_cast<ConstantSInt>(C))
          constValue += CS->getValue() * elementSize;
        else if (ConstantUInt *CU = dyn_cast<ConstantUInt>(C))
          constValue += CU->getValue() * elementSize;
        else
          assert(0 && "Invalid ConstantInt GEP index type!");
      } else {
        // Push current gep state to this point as an add and multiply
        ops.push_back(CollapsedGepOp(
          ConstantSInt::get(Type::IntTy, constValue),
          idx, ConstantUInt::get(Type::UIntTy, elementSize)));

        constValue = 0;
      }
    }
  }
  // Emit instructions for all the collapsed ops
  unsigned indexReg = 0;
  for(std::vector<CollapsedGepOp>::iterator cgo_i = ops.begin(),
      cgo_e = ops.end(); cgo_i != cgo_e; ++cgo_i) {
    CollapsedGepOp& cgo = *cgo_i;

    // Avoid emitting known move instructions here for the register allocator
    // to deal with later.  val * 1 == val.  val + 0 == val.
    unsigned TmpReg1;
    if (cgo.size->getValue() == 1) {
      TmpReg1 = getReg(cgo.index, MBB, IP);
    } else {
      TmpReg1 = makeAnotherReg(Type::IntTy);
      doMultiplyConst(MBB, IP, TmpReg1, cgo.index, cgo.size);
    }
    
    unsigned TmpReg2;
    if (cgo.offset->isNullValue()) { 
      TmpReg2 = TmpReg1;
    } else {
      TmpReg2 = makeAnotherReg(Type::IntTy);
      emitBinaryConstOperation(MBB, IP, TmpReg1, cgo.offset, 0, TmpReg2);
    }
    
    if (indexReg == 0)
      indexReg = TmpReg2;
    else {
      unsigned TmpReg3 = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, PPC::ADD, 2, TmpReg3).addReg(indexReg).addReg(TmpReg2);
      indexReg = TmpReg3;
    }
  }
  
  // We now have a base register, an index register, and possibly a constant
  // remainder.  If the GEP is going to be folded, we try to generate the
  // optimal addressing mode.
  ConstantSInt *remainder = ConstantSInt::get(Type::IntTy, constValue);
  
  // If we are emitting this during a fold, copy the current base register to
  // the target, and save the current constant offset so the folding load or
  // store can try and use it as an immediate.
  if (GEPIsFolded) {
    if (indexReg == 0) {
      if (!canUseAsImmediateForOpcode(remainder, 0, false)) {
        indexReg = getReg(remainder, MBB, IP);
        remainder = 0;
      }
    } else if (!remainder->isNullValue()) {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      emitBinaryConstOperation(MBB, IP, indexReg, remainder, 0, TmpReg);
      indexReg = TmpReg;
      remainder = 0;
    }
    unsigned basePtrReg = getReg(Src, MBB, IP);
    GEPMap[GEPI] = FoldedGEP(basePtrReg, indexReg, remainder);
    return;
  }

  // We're not folding, so collapse the base, index, and any remainder into the
  // destination register.
  unsigned TargetReg = getReg(GEPI, MBB, IP);
  unsigned basePtrReg = getReg(Src, MBB, IP);

  if ((indexReg == 0) && remainder->isNullValue()) {
    BuildMI(*MBB, IP, PPC::OR, 2, TargetReg).addReg(basePtrReg)
      .addReg(basePtrReg);
    return;
  }
  if (!remainder->isNullValue()) {
    unsigned TmpReg = (indexReg == 0) ? TargetReg : makeAnotherReg(Type::IntTy);
    emitBinaryConstOperation(MBB, IP, basePtrReg, remainder, 0, TmpReg);
    basePtrReg = TmpReg;
  }
  if (indexReg != 0)
    BuildMI(*MBB, IP, PPC::ADD, 2, TargetReg).addReg(indexReg)
      .addReg(basePtrReg);
}

/// visitAllocaInst - If this is a fixed size alloca, allocate space from the
/// frame manager, otherwise do it the hard way.
///
void PPC32ISel::visitAllocaInst(AllocaInst &I) {
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
  BuildMI(BB, PPC::ADDI, 2, AddedSizeReg).addReg(TotalSizeReg).addSImm(15);

  // AlignedSize = and <AddedSize>, ~15
  unsigned AlignedSize = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, PPC::RLWINM, 4, AlignedSize).addReg(AddedSizeReg).addImm(0)
    .addImm(0).addImm(27);
  
  // Subtract size from stack pointer, thereby allocating some space.
  BuildMI(BB, PPC::SUB, 2, PPC::R1).addReg(PPC::R1).addReg(AlignedSize);

  // Put a pointer to the space into the result register, by copying
  // the stack pointer.
  BuildMI(BB, PPC::OR, 2, getReg(I)).addReg(PPC::R1).addReg(PPC::R1);

  // Inform the Frame Information that we have just allocated a variable-sized
  // object.
  F->getFrameInfo()->CreateVariableSizedObject();
}

/// visitMallocInst - Malloc instructions are code generated into direct calls
/// to the library malloc.
///
void PPC32ISel::visitMallocInst(MallocInst &I) {
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
    BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(mallocFn, true);
  doCall(ValueRecord(getReg(I), I.getType()), TheCall, Args, false);
}

/// visitFreeInst - Free instructions are code gen'd to call the free libc
/// function.
///
void PPC32ISel::visitFreeInst(FreeInst &I) {
  std::vector<ValueRecord> Args;
  Args.push_back(ValueRecord(I.getOperand(0)));
  MachineInstr *TheCall = 
    BuildMI(PPC::CALLpcrel, 1).addGlobalAddress(freeFn, true);
  doCall(ValueRecord(0, Type::VoidTy), TheCall, Args, false);
}
   
/// createPPC32ISelSimple - This pass converts an LLVM function into a machine
/// code representation is a very simple peep-hole fashion.
///
FunctionPass *llvm::createPPC32ISelSimple(TargetMachine &TM) {
  return new PPC32ISel(TM);
}
