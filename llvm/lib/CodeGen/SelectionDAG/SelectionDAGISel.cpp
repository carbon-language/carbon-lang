//===-- SelectionDAGISel.cpp - Implement the SelectionDAGISel class -------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAGISel class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "isel"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <iostream>
using namespace llvm;

#ifndef _NDEBUG
static cl::opt<bool>
ViewDAGs("view-isel-dags", cl::Hidden,
         cl::desc("Pop up a window to show isel dags as they are selected"));
#else
static const bool ViewDAGS = 0;
#endif

namespace llvm {
  //===--------------------------------------------------------------------===//
  /// FunctionLoweringInfo - This contains information that is global to a
  /// function that is used when lowering a region of the function.
  class FunctionLoweringInfo {
  public:
    TargetLowering &TLI;
    Function &Fn;
    MachineFunction &MF;
    SSARegMap *RegMap;

    FunctionLoweringInfo(TargetLowering &TLI, Function &Fn,MachineFunction &MF);

    /// MBBMap - A mapping from LLVM basic blocks to their machine code entry.
    std::map<const BasicBlock*, MachineBasicBlock *> MBBMap;

    /// ValueMap - Since we emit code for the function a basic block at a time,
    /// we must remember which virtual registers hold the values for
    /// cross-basic-block values.
    std::map<const Value*, unsigned> ValueMap;

    /// StaticAllocaMap - Keep track of frame indices for fixed sized allocas in
    /// the entry block.  This allows the allocas to be efficiently referenced
    /// anywhere in the function.
    std::map<const AllocaInst*, int> StaticAllocaMap;

    /// BlockLocalArguments - If any arguments are only used in a single basic
    /// block, and if the target can access the arguments without side-effects,
    /// avoid emitting CopyToReg nodes for those arguments.  This map keeps
    /// track of which arguments are local to each BB.
    std::multimap<BasicBlock*, std::pair<Argument*,
                                         unsigned> > BlockLocalArguments;


    unsigned MakeReg(MVT::ValueType VT) {
      return RegMap->createVirtualRegister(TLI.getRegClassFor(VT));
    }
  
    unsigned CreateRegForValue(const Value *V) {
      MVT::ValueType VT = TLI.getValueType(V->getType());
      // The common case is that we will only create one register for this
      // value.  If we have that case, create and return the virtual register.
      unsigned NV = TLI.getNumElements(VT);
      if (NV == 1) {
        // If we are promoting this value, pick the next largest supported type.
        return MakeReg(TLI.getTypeToTransformTo(VT));
      }
    
      // If this value is represented with multiple target registers, make sure
      // to create enough consequtive registers of the right (smaller) type.
      unsigned NT = VT-1;  // Find the type to use.
      while (TLI.getNumElements((MVT::ValueType)NT) != 1)
        --NT;
    
      unsigned R = MakeReg((MVT::ValueType)NT);
      for (unsigned i = 1; i != NV; ++i)
        MakeReg((MVT::ValueType)NT);
      return R;
    }
  
    unsigned InitializeRegForValue(const Value *V) {
      unsigned &R = ValueMap[V];
      assert(R == 0 && "Already initialized this value register!");
      return R = CreateRegForValue(V);
    }
  };
}

/// isUsedOutsideOfDefiningBlock - Return true if this instruction is used by
/// PHI nodes or outside of the basic block that defines it.
static bool isUsedOutsideOfDefiningBlock(Instruction *I) {
  if (isa<PHINode>(I)) return true;
  BasicBlock *BB = I->getParent();
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E; ++UI)
    if (cast<Instruction>(*UI)->getParent() != BB || isa<PHINode>(*UI))
      return true;
  return false;
}

FunctionLoweringInfo::FunctionLoweringInfo(TargetLowering &tli,
                                           Function &fn, MachineFunction &mf) 
    : TLI(tli), Fn(fn), MF(mf), RegMap(MF.getSSARegMap()) {

  // Initialize the mapping of values to registers.  This is only set up for
  // instruction values that are used outside of the block that defines
  // them.
  for (Function::aiterator AI = Fn.abegin(), E = Fn.aend(); AI != E; ++AI)
    InitializeRegForValue(AI);

  Function::iterator BB = Fn.begin(), E = Fn.end();
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(AI->getArraySize())) {
        const Type *Ty = AI->getAllocatedType();
        uint64_t TySize = TLI.getTargetData().getTypeSize(Ty);
        unsigned Align = TLI.getTargetData().getTypeAlignment(Ty);
        TySize *= CUI->getValue();   // Get total allocated size.
        StaticAllocaMap[AI] =
          MF.getFrameInfo()->CreateStackObject((unsigned)TySize, Align);
      }

  for (; BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), e = BB->end(); I != e; ++I)
      if (!I->use_empty() && isUsedOutsideOfDefiningBlock(I))
        if (!isa<AllocaInst>(I) ||
            !StaticAllocaMap.count(cast<AllocaInst>(I)))
          InitializeRegForValue(I);

  // Create an initial MachineBasicBlock for each LLVM BasicBlock in F.  This
  // also creates the initial PHI MachineInstrs, though none of the input
  // operands are populated.
  for (Function::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    MachineBasicBlock *MBB = new MachineBasicBlock(BB);
    MBBMap[BB] = MBB;
    MF.getBasicBlockList().push_back(MBB);

    // Create Machine PHI nodes for LLVM PHI nodes, lowering them as
    // appropriate.
    PHINode *PN;
    for (BasicBlock::iterator I = BB->begin();
         (PN = dyn_cast<PHINode>(I)); ++I)
      if (!PN->use_empty()) {
        unsigned NumElements =
          TLI.getNumElements(TLI.getValueType(PN->getType()));
        unsigned PHIReg = ValueMap[PN];
        assert(PHIReg &&"PHI node does not have an assigned virtual register!");
        for (unsigned i = 0; i != NumElements; ++i)
          BuildMI(MBB, TargetInstrInfo::PHI, PN->getNumOperands(), PHIReg+i);
      }
  }
}



//===----------------------------------------------------------------------===//
/// SelectionDAGLowering - This is the common target-independent lowering
/// implementation that is parameterized by a TargetLowering object.
/// Also, targets can overload any lowering method.
///
namespace llvm {
class SelectionDAGLowering {
  MachineBasicBlock *CurMBB;

  std::map<const Value*, SDOperand> NodeMap;

  /// PendingLoads - Loads are not emitted to the program immediately.  We bunch
  /// them up and then emit token factor nodes when possible.  This allows us to
  /// get simple disambiguation between loads without worrying about alias
  /// analysis.
  std::vector<SDOperand> PendingLoads;

public:
  // TLI - This is information that describes the available target features we
  // need for lowering.  This indicates when operations are unavailable,
  // implemented with a libcall, etc.
  TargetLowering &TLI;
  SelectionDAG &DAG;
  const TargetData &TD;

  /// FuncInfo - Information about the function as a whole.
  ///
  FunctionLoweringInfo &FuncInfo;

  SelectionDAGLowering(SelectionDAG &dag, TargetLowering &tli,
                       FunctionLoweringInfo &funcinfo) 
    : TLI(tli), DAG(dag), TD(DAG.getTarget().getTargetData()),
      FuncInfo(funcinfo) {
  }

  /// getRoot - Return the current virtual root of the Selection DAG.
  ///
  SDOperand getRoot() {
    if (PendingLoads.empty())
      return DAG.getRoot();
    
    if (PendingLoads.size() == 1) {
      SDOperand Root = PendingLoads[0];
      DAG.setRoot(Root);
      PendingLoads.clear();
      return Root;
    }

    // Otherwise, we have to make a token factor node.
    SDOperand Root = DAG.getNode(ISD::TokenFactor, MVT::Other, PendingLoads);
    PendingLoads.clear();
    DAG.setRoot(Root);
    return Root;
  }

  void visit(Instruction &I) { visit(I.getOpcode(), I); }

  void visit(unsigned Opcode, User &I) {
    switch (Opcode) {
    default: assert(0 && "Unknown instruction type encountered!");
             abort();
      // Build the switch statement using the Instruction.def file.
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    case Instruction::OPCODE:return visit##OPCODE((CLASS&)I);
#include "llvm/Instruction.def"
    }
  }

  void setCurrentBasicBlock(MachineBasicBlock *MBB) { CurMBB = MBB; }


  SDOperand getIntPtrConstant(uint64_t Val) {
    return DAG.getConstant(Val, TLI.getPointerTy());
  }

  SDOperand getValue(const Value *V) {
    SDOperand &N = NodeMap[V];
    if (N.Val) return N;

    MVT::ValueType VT = TLI.getValueType(V->getType());
    if (Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V)))
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
        visit(CE->getOpcode(), *CE);
        assert(N.Val && "visit didn't populate the ValueMap!");
        return N;
      } else if (GlobalValue *GV = dyn_cast<GlobalValue>(C)) {
        return N = DAG.getGlobalAddress(GV, VT);
      } else if (isa<ConstantPointerNull>(C)) {
        return N = DAG.getConstant(0, TLI.getPointerTy());
      } else if (isa<UndefValue>(C)) {
	/// FIXME: Implement UNDEFVALUE better.
        if (MVT::isInteger(VT))
          return N = DAG.getConstant(0, VT);
        else if (MVT::isFloatingPoint(VT))
          return N = DAG.getConstantFP(0, VT);
        else
          assert(0 && "Unknown value type!");

      } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
        return N = DAG.getConstantFP(CFP->getValue(), VT);
      } else {
        // Canonicalize all constant ints to be unsigned.
        return N = DAG.getConstant(cast<ConstantIntegral>(C)->getRawValue(),VT);
      }

    if (const AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
      std::map<const AllocaInst*, int>::iterator SI =
        FuncInfo.StaticAllocaMap.find(AI);
      if (SI != FuncInfo.StaticAllocaMap.end())
        return DAG.getFrameIndex(SI->second, TLI.getPointerTy());
    }

    std::map<const Value*, unsigned>::const_iterator VMI =
      FuncInfo.ValueMap.find(V);
    assert(VMI != FuncInfo.ValueMap.end() && "Value not in map!");

    return N = DAG.getCopyFromReg(VMI->second, VT, DAG.getEntryNode());
  }

  const SDOperand &setValue(const Value *V, SDOperand NewN) {
    SDOperand &N = NodeMap[V];
    assert(N.Val == 0 && "Already set a value for this node!");
    return N = NewN;
  }

  // Terminator instructions.
  void visitRet(ReturnInst &I);
  void visitBr(BranchInst &I);
  void visitUnreachable(UnreachableInst &I) { /* noop */ }

  // These all get lowered before this pass.
  void visitSwitch(SwitchInst &I) { assert(0 && "TODO"); }
  void visitInvoke(InvokeInst &I) { assert(0 && "TODO"); }
  void visitUnwind(UnwindInst &I) { assert(0 && "TODO"); }

  //
  void visitBinary(User &I, unsigned Opcode);
  void visitAdd(User &I) { visitBinary(I, ISD::ADD); }
  void visitSub(User &I) { visitBinary(I, ISD::SUB); }
  void visitMul(User &I) { visitBinary(I, ISD::MUL); }
  void visitDiv(User &I) {
    visitBinary(I, I.getType()->isUnsigned() ? ISD::UDIV : ISD::SDIV);
  }
  void visitRem(User &I) {
    visitBinary(I, I.getType()->isUnsigned() ? ISD::UREM : ISD::SREM);
  }
  void visitAnd(User &I) { visitBinary(I, ISD::AND); }
  void visitOr (User &I) { visitBinary(I, ISD::OR); }
  void visitXor(User &I) { visitBinary(I, ISD::XOR); }
  void visitShl(User &I) { visitBinary(I, ISD::SHL); }
  void visitShr(User &I) {
    visitBinary(I, I.getType()->isUnsigned() ? ISD::SRL : ISD::SRA);
  }

  void visitSetCC(User &I, ISD::CondCode SignedOpc, ISD::CondCode UnsignedOpc);
  void visitSetEQ(User &I) { visitSetCC(I, ISD::SETEQ, ISD::SETEQ); }
  void visitSetNE(User &I) { visitSetCC(I, ISD::SETNE, ISD::SETNE); }
  void visitSetLE(User &I) { visitSetCC(I, ISD::SETLE, ISD::SETULE); }
  void visitSetGE(User &I) { visitSetCC(I, ISD::SETGE, ISD::SETUGE); }
  void visitSetLT(User &I) { visitSetCC(I, ISD::SETLT, ISD::SETULT); }
  void visitSetGT(User &I) { visitSetCC(I, ISD::SETGT, ISD::SETUGT); }

  void visitGetElementPtr(User &I);
  void visitCast(User &I);
  void visitSelect(User &I);
  //

  void visitMalloc(MallocInst &I);
  void visitFree(FreeInst &I);
  void visitAlloca(AllocaInst &I);
  void visitLoad(LoadInst &I);
  void visitStore(StoreInst &I);
  void visitPHI(PHINode &I) { } // PHI nodes are handled specially.
  void visitCall(CallInst &I);

  void visitVAStart(CallInst &I);
  void visitVANext(VANextInst &I);
  void visitVAArg(VAArgInst &I);
  void visitVAEnd(CallInst &I);
  void visitVACopy(CallInst &I);
  void visitFrameReturnAddress(CallInst &I, bool isFrameAddress);

  void visitMemIntrinsic(CallInst &I, unsigned Op);

  void visitUserOp1(Instruction &I) {
    assert(0 && "UserOp1 should not exist at instruction selection time!");
    abort();
  }
  void visitUserOp2(Instruction &I) {
    assert(0 && "UserOp2 should not exist at instruction selection time!");
    abort();
  }
};
} // end namespace llvm

void SelectionDAGLowering::visitRet(ReturnInst &I) {
  if (I.getNumOperands() == 0) {
    DAG.setRoot(DAG.getNode(ISD::RET, MVT::Other, getRoot()));
    return;
  }

  SDOperand Op1 = getValue(I.getOperand(0));
  switch (Op1.getValueType()) {
  default: assert(0 && "Unknown value type!");
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
    // Extend integer types to 32-bits.
    if (I.getOperand(0)->getType()->isSigned())
      Op1 = DAG.getNode(ISD::SIGN_EXTEND, MVT::i32, Op1);
    else
      Op1 = DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, Op1);
    break;
  case MVT::f32:
    // Extend float to double.
    Op1 = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Op1);
    break;
  case MVT::i32:
  case MVT::i64:
  case MVT::f64:
    break; // No extension needed!
  }

  DAG.setRoot(DAG.getNode(ISD::RET, MVT::Other, getRoot(), Op1));
}

void SelectionDAGLowering::visitBr(BranchInst &I) {
  // Update machine-CFG edges.
  MachineBasicBlock *Succ0MBB = FuncInfo.MBBMap[I.getSuccessor(0)];
  CurMBB->addSuccessor(Succ0MBB);

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  if (I.isUnconditional()) {
    // If this is not a fall-through branch, emit the branch.
    if (Succ0MBB != NextBlock)
      DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, getRoot(),
			      DAG.getBasicBlock(Succ0MBB)));
  } else {
    MachineBasicBlock *Succ1MBB = FuncInfo.MBBMap[I.getSuccessor(1)];
    CurMBB->addSuccessor(Succ1MBB);

    SDOperand Cond = getValue(I.getCondition());

    if (Succ1MBB == NextBlock) {
      // If the condition is false, fall through.  This means we should branch
      // if the condition is true to Succ #0.
      DAG.setRoot(DAG.getNode(ISD::BRCOND, MVT::Other, getRoot(),
			      Cond, DAG.getBasicBlock(Succ0MBB)));
    } else if (Succ0MBB == NextBlock) {
      // If the condition is true, fall through.  This means we should branch if
      // the condition is false to Succ #1.  Invert the condition first.
      SDOperand True = DAG.getConstant(1, Cond.getValueType());
      Cond = DAG.getNode(ISD::XOR, Cond.getValueType(), Cond, True);
      DAG.setRoot(DAG.getNode(ISD::BRCOND, MVT::Other, getRoot(),
			      Cond, DAG.getBasicBlock(Succ1MBB)));
    } else {
      // Neither edge is a fall through.  If the comparison is true, jump to
      // Succ#0, otherwise branch unconditionally to succ #1.
      DAG.setRoot(DAG.getNode(ISD::BRCOND, MVT::Other, getRoot(),
			      Cond, DAG.getBasicBlock(Succ0MBB)));
      DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, getRoot(),
			      DAG.getBasicBlock(Succ1MBB)));
    }
  }
}

void SelectionDAGLowering::visitBinary(User &I, unsigned Opcode) {
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));

  if (isa<ShiftInst>(I))
    Op2 = DAG.getNode(ISD::ZERO_EXTEND, TLI.getShiftAmountTy(), Op2);

  setValue(&I, DAG.getNode(Opcode, Op1.getValueType(), Op1, Op2));
}

void SelectionDAGLowering::visitSetCC(User &I,ISD::CondCode SignedOpcode,
                                      ISD::CondCode UnsignedOpcode) {
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  ISD::CondCode Opcode = SignedOpcode;
  if (I.getOperand(0)->getType()->isUnsigned())
    Opcode = UnsignedOpcode;
  setValue(&I, DAG.getSetCC(Opcode, MVT::i1, Op1, Op2));
}

void SelectionDAGLowering::visitSelect(User &I) {
  SDOperand Cond     = getValue(I.getOperand(0));
  SDOperand TrueVal  = getValue(I.getOperand(1));
  SDOperand FalseVal = getValue(I.getOperand(2));
  setValue(&I, DAG.getNode(ISD::SELECT, TrueVal.getValueType(), Cond,
                           TrueVal, FalseVal));
}

void SelectionDAGLowering::visitCast(User &I) {
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType SrcTy = TLI.getValueType(I.getOperand(0)->getType());
  MVT::ValueType DestTy = TLI.getValueType(I.getType());

  if (N.getValueType() == DestTy) {
    setValue(&I, N);  // noop cast.
  } else if (isInteger(SrcTy)) {
    if (isInteger(DestTy)) {        // Int -> Int cast
      if (DestTy < SrcTy)   // Truncating cast?
        setValue(&I, DAG.getNode(ISD::TRUNCATE, DestTy, N));
      else if (I.getOperand(0)->getType()->isSigned())
        setValue(&I, DAG.getNode(ISD::SIGN_EXTEND, DestTy, N));
      else
        setValue(&I, DAG.getNode(ISD::ZERO_EXTEND, DestTy, N));
    } else {                        // Int -> FP cast
      if (I.getOperand(0)->getType()->isSigned())
        setValue(&I, DAG.getNode(ISD::SINT_TO_FP, DestTy, N));
      else
        setValue(&I, DAG.getNode(ISD::UINT_TO_FP, DestTy, N));
    }
  } else {
    assert(isFloatingPoint(SrcTy) && "Unknown value type!");
    if (isFloatingPoint(DestTy)) {  // FP -> FP cast
      if (DestTy < SrcTy)   // Rounding cast?
        setValue(&I, DAG.getNode(ISD::FP_ROUND, DestTy, N));
      else
        setValue(&I, DAG.getNode(ISD::FP_EXTEND, DestTy, N));
    } else {                        // FP -> Int cast.
      if (I.getType()->isSigned())
        setValue(&I, DAG.getNode(ISD::FP_TO_SINT, DestTy, N));
      else
        setValue(&I, DAG.getNode(ISD::FP_TO_UINT, DestTy, N));
    }
  }
}

void SelectionDAGLowering::visitGetElementPtr(User &I) {
  SDOperand N = getValue(I.getOperand(0));
  const Type *Ty = I.getOperand(0)->getType();
  const Type *UIntPtrTy = TD.getIntPtrType();

  for (GetElementPtrInst::op_iterator OI = I.op_begin()+1, E = I.op_end();
       OI != E; ++OI) {
    Value *Idx = *OI;
    if (const StructType *StTy = dyn_cast<StructType> (Ty)) {
      unsigned Field = cast<ConstantUInt>(Idx)->getValue();
      if (Field) {
        // N = N + Offset
        uint64_t Offset = TD.getStructLayout(StTy)->MemberOffsets[Field];
        N = DAG.getNode(ISD::ADD, N.getValueType(), N,
			getIntPtrConstant(Offset));
      }
      Ty = StTy->getElementType(Field);
    } else {
      Ty = cast<SequentialType>(Ty)->getElementType();
      if (!isa<Constant>(Idx) || !cast<Constant>(Idx)->isNullValue()) {
        // N = N + Idx * ElementSize;
        uint64_t ElementSize = TD.getTypeSize(Ty);
        SDOperand IdxN = getValue(Idx), Scale = getIntPtrConstant(ElementSize);

        // If the index is smaller or larger than intptr_t, truncate or extend
        // it.
        if (IdxN.getValueType() < Scale.getValueType()) {
          if (Idx->getType()->isSigned())
            IdxN = DAG.getNode(ISD::SIGN_EXTEND, Scale.getValueType(), IdxN);
          else
            IdxN = DAG.getNode(ISD::ZERO_EXTEND, Scale.getValueType(), IdxN);
        } else if (IdxN.getValueType() > Scale.getValueType())
          IdxN = DAG.getNode(ISD::TRUNCATE, Scale.getValueType(), IdxN);

        IdxN = DAG.getNode(ISD::MUL, N.getValueType(), IdxN, Scale);
			   
        N = DAG.getNode(ISD::ADD, N.getValueType(), N, IdxN);
      }
    }
  }
  setValue(&I, N);
}

void SelectionDAGLowering::visitAlloca(AllocaInst &I) {
  // If this is a fixed sized alloca in the entry block of the function,
  // allocate it statically on the stack.
  if (FuncInfo.StaticAllocaMap.count(&I))
    return;   // getValue will auto-populate this.

  const Type *Ty = I.getAllocatedType();
  uint64_t TySize = TLI.getTargetData().getTypeSize(Ty);
  unsigned Align = TLI.getTargetData().getTypeAlignment(Ty);

  SDOperand AllocSize = getValue(I.getArraySize());
  MVT::ValueType IntPtr = TLI.getPointerTy();
  if (IntPtr < AllocSize.getValueType())
    AllocSize = DAG.getNode(ISD::TRUNCATE, IntPtr, AllocSize);
  else if (IntPtr > AllocSize.getValueType())
    AllocSize = DAG.getNode(ISD::ZERO_EXTEND, IntPtr, AllocSize);

  AllocSize = DAG.getNode(ISD::MUL, IntPtr, AllocSize,
                          getIntPtrConstant(TySize));

  // Handle alignment.  If the requested alignment is less than or equal to the
  // stack alignment, ignore it and round the size of the allocation up to the
  // stack alignment size.  If the size is greater than the stack alignment, we
  // note this in the DYNAMIC_STACKALLOC node.
  unsigned StackAlign =
    TLI.getTargetMachine().getFrameInfo()->getStackAlignment();
  if (Align <= StackAlign) {
    Align = 0;
    // Add SA-1 to the size.
    AllocSize = DAG.getNode(ISD::ADD, AllocSize.getValueType(), AllocSize,
                            getIntPtrConstant(StackAlign-1));
    // Mask out the low bits for alignment purposes.
    AllocSize = DAG.getNode(ISD::AND, AllocSize.getValueType(), AllocSize,
                            getIntPtrConstant(~(uint64_t)(StackAlign-1)));
  }

  SDOperand DSA = DAG.getNode(ISD::DYNAMIC_STACKALLOC, AllocSize.getValueType(),
                              getRoot(), AllocSize,
                              getIntPtrConstant(Align));
  DAG.setRoot(setValue(&I, DSA).getValue(1));

  // Inform the Frame Information that we have just allocated a variable-sized
  // object.
  CurMBB->getParent()->getFrameInfo()->CreateVariableSizedObject();
}


void SelectionDAGLowering::visitLoad(LoadInst &I) {
  SDOperand Ptr = getValue(I.getOperand(0));
  
  SDOperand Root;
  if (I.isVolatile())
    Root = getRoot();
  else {
    // Do not serialize non-volatile loads against each other.
    Root = DAG.getRoot();
  }

  SDOperand L = DAG.getLoad(TLI.getValueType(I.getType()), Root, Ptr);
  setValue(&I, L);

  if (I.isVolatile())
    DAG.setRoot(L.getValue(1));
  else
    PendingLoads.push_back(L.getValue(1));
}


void SelectionDAGLowering::visitStore(StoreInst &I) {
  Value *SrcV = I.getOperand(0);
  SDOperand Src = getValue(SrcV);
  SDOperand Ptr = getValue(I.getOperand(1));
  DAG.setRoot(DAG.getNode(ISD::STORE, MVT::Other, getRoot(), Src, Ptr));
}

void SelectionDAGLowering::visitCall(CallInst &I) {
  const char *RenameFn = 0;
  if (Function *F = I.getCalledFunction())
    switch (F->getIntrinsicID()) {
    case 0: break;  // Not an intrinsic.
    case Intrinsic::vastart:  visitVAStart(I); return;
    case Intrinsic::vaend:    visitVAEnd(I); return;
    case Intrinsic::vacopy:   visitVACopy(I); return;
    case Intrinsic::returnaddress: visitFrameReturnAddress(I, false); return;
    case Intrinsic::frameaddress:  visitFrameReturnAddress(I, true); return;
    default:
      // FIXME: IMPLEMENT THESE.
      // readport, writeport, readio, writeio
      assert(0 && "This intrinsic is not implemented yet!");
      return;
    case Intrinsic::setjmp:  RenameFn = "setjmp"; break;
    case Intrinsic::longjmp: RenameFn = "longjmp"; break;
    case Intrinsic::memcpy:  visitMemIntrinsic(I, ISD::MEMCPY); return;
    case Intrinsic::memset:  visitMemIntrinsic(I, ISD::MEMSET); return;
    case Intrinsic::memmove: visitMemIntrinsic(I, ISD::MEMMOVE); return;
      
    case Intrinsic::isunordered:
      setValue(&I, DAG.getSetCC(ISD::SETUO, MVT::i1, getValue(I.getOperand(1)),
                                getValue(I.getOperand(2))));
      return;
    }
  
  SDOperand Callee;
  if (!RenameFn)
    Callee = getValue(I.getOperand(0));
  else
    Callee = DAG.getExternalSymbol(RenameFn, TLI.getPointerTy());
  std::vector<std::pair<SDOperand, const Type*> > Args;
  
  for (unsigned i = 1, e = I.getNumOperands(); i != e; ++i) {
    Value *Arg = I.getOperand(i);
    SDOperand ArgNode = getValue(Arg);
    Args.push_back(std::make_pair(ArgNode, Arg->getType()));
  }
  
  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerCallTo(getRoot(), I.getType(), Callee, Args, DAG);
  if (I.getType() != Type::VoidTy)
    setValue(&I, Result.first);
  DAG.setRoot(Result.second);
}

void SelectionDAGLowering::visitMalloc(MallocInst &I) {
  SDOperand Src = getValue(I.getOperand(0));

  MVT::ValueType IntPtr = TLI.getPointerTy();

  if (IntPtr < Src.getValueType())
    Src = DAG.getNode(ISD::TRUNCATE, IntPtr, Src);
  else if (IntPtr > Src.getValueType())
    Src = DAG.getNode(ISD::ZERO_EXTEND, IntPtr, Src);

  // Scale the source by the type size.
  uint64_t ElementSize = TD.getTypeSize(I.getType()->getElementType());
  Src = DAG.getNode(ISD::MUL, Src.getValueType(),
                    Src, getIntPtrConstant(ElementSize));

  std::vector<std::pair<SDOperand, const Type*> > Args;
  Args.push_back(std::make_pair(Src, TLI.getTargetData().getIntPtrType()));

  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerCallTo(getRoot(), I.getType(),
                    DAG.getExternalSymbol("malloc", IntPtr),
                    Args, DAG);
  setValue(&I, Result.first);  // Pointers always fit in registers
  DAG.setRoot(Result.second);
}

void SelectionDAGLowering::visitFree(FreeInst &I) {
  std::vector<std::pair<SDOperand, const Type*> > Args;
  Args.push_back(std::make_pair(getValue(I.getOperand(0)),
                                TLI.getTargetData().getIntPtrType()));
  MVT::ValueType IntPtr = TLI.getPointerTy();
  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerCallTo(getRoot(), Type::VoidTy,
                    DAG.getExternalSymbol("free", IntPtr), Args, DAG);
  DAG.setRoot(Result.second);
}

std::pair<SDOperand, SDOperand>
TargetLowering::LowerVAStart(SDOperand Chain, SelectionDAG &DAG) {
  // We have no sane default behavior, just emit a useful error message and bail
  // out.
  std::cerr << "Variable arguments handling not implemented on this target!\n";
  abort();
}

SDOperand TargetLowering::LowerVAEnd(SDOperand Chain, SDOperand L,
                                     SelectionDAG &DAG) {
  // Default to a noop.
  return Chain;
}

std::pair<SDOperand,SDOperand>
TargetLowering::LowerVACopy(SDOperand Chain, SDOperand L, SelectionDAG &DAG) {
  // Default to returning the input list.
  return std::make_pair(L, Chain);
}

std::pair<SDOperand,SDOperand>
TargetLowering::LowerVAArgNext(bool isVANext, SDOperand Chain, SDOperand VAList,
                               const Type *ArgTy, SelectionDAG &DAG) {
  // We have no sane default behavior, just emit a useful error message and bail
  // out.
  std::cerr << "Variable arguments handling not implemented on this target!\n";
  abort();
}


void SelectionDAGLowering::visitVAStart(CallInst &I) {
  std::pair<SDOperand,SDOperand> Result = TLI.LowerVAStart(getRoot(), DAG);
  setValue(&I, Result.first);
  DAG.setRoot(Result.second);
}

void SelectionDAGLowering::visitVAArg(VAArgInst &I) {
  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerVAArgNext(false, getRoot(), getValue(I.getOperand(0)), 
                       I.getType(), DAG);
  setValue(&I, Result.first);
  DAG.setRoot(Result.second);
}

void SelectionDAGLowering::visitVANext(VANextInst &I) {
  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerVAArgNext(true, getRoot(), getValue(I.getOperand(0)), 
                       I.getArgType(), DAG);
  setValue(&I, Result.first);
  DAG.setRoot(Result.second);
}

void SelectionDAGLowering::visitVAEnd(CallInst &I) {
  DAG.setRoot(TLI.LowerVAEnd(getRoot(), getValue(I.getOperand(1)), DAG));
}

void SelectionDAGLowering::visitVACopy(CallInst &I) {
  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerVACopy(getRoot(), getValue(I.getOperand(1)), DAG);
  setValue(&I, Result.first);
  DAG.setRoot(Result.second);
}


// It is always conservatively correct for llvm.returnaddress and
// llvm.frameaddress to return 0.
std::pair<SDOperand, SDOperand>
TargetLowering::LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain,
                                        unsigned Depth, SelectionDAG &DAG) {
  return std::make_pair(DAG.getConstant(0, getPointerTy()), Chain);
}

SDOperand TargetLowering::LowerOperation(SDOperand Op) {
  assert(0 && "LowerOperation not implemented for this target!");
  abort();
}

void SelectionDAGLowering::visitFrameReturnAddress(CallInst &I, bool isFrame) {
  unsigned Depth = (unsigned)cast<ConstantUInt>(I.getOperand(1))->getValue();
  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerFrameReturnAddress(isFrame, getRoot(), Depth, DAG);
  setValue(&I, Result.first);
  DAG.setRoot(Result.second);
}

void SelectionDAGLowering::visitMemIntrinsic(CallInst &I, unsigned Op) {
  std::vector<SDOperand> Ops;
  Ops.push_back(getRoot());
  Ops.push_back(getValue(I.getOperand(1)));
  Ops.push_back(getValue(I.getOperand(2)));
  Ops.push_back(getValue(I.getOperand(3)));
  Ops.push_back(getValue(I.getOperand(4)));
  DAG.setRoot(DAG.getNode(Op, MVT::Other, Ops));
}

//===----------------------------------------------------------------------===//
// SelectionDAGISel code
//===----------------------------------------------------------------------===//

unsigned SelectionDAGISel::MakeReg(MVT::ValueType VT) {
  return RegMap->createVirtualRegister(TLI.getRegClassFor(VT));
}



bool SelectionDAGISel::runOnFunction(Function &Fn) {
  MachineFunction &MF = MachineFunction::construct(&Fn, TLI.getTargetMachine());
  RegMap = MF.getSSARegMap();
  DEBUG(std::cerr << "\n\n\n=== " << Fn.getName() << "\n");

  FunctionLoweringInfo FuncInfo(TLI, Fn, MF);

  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    SelectBasicBlock(I, MF, FuncInfo);
  
  return true;
}


SDOperand SelectionDAGISel::
CopyValueToVirtualRegister(SelectionDAGLowering &SDL, Value *V, unsigned Reg) {
  SelectionDAG &DAG = SDL.DAG;
  SDOperand Op = SDL.getValue(V);
  assert((Op.getOpcode() != ISD::CopyFromReg ||
          cast<RegSDNode>(Op)->getReg() != Reg) &&
         "Copy from a reg to the same reg!");
  return DAG.getCopyToReg(SDL.getRoot(), Op, Reg);
}

/// IsOnlyUsedInOneBasicBlock - If the specified argument is only used in a
/// single basic block, return that block.  Otherwise, return a null pointer.
static BasicBlock *IsOnlyUsedInOneBasicBlock(Argument *A) {
  if (A->use_empty()) return 0;
  BasicBlock *BB = cast<Instruction>(A->use_back())->getParent();
  for (Argument::use_iterator UI = A->use_begin(), E = A->use_end(); UI != E;
       ++UI)
    if (isa<PHINode>(*UI) || cast<Instruction>(*UI)->getParent() != BB)
      return 0;  // Disagreement among the users?

  // Okay, there is a single BB user.  Only permit this optimization if this is
  // the entry block, otherwise, we might sink argument loads into loops and
  // stuff.  Later, when we have global instruction selection, this won't be an
  // issue clearly.
  if (BB == BB->getParent()->begin())
    return BB;
  return 0;
}

void SelectionDAGISel::
LowerArguments(BasicBlock *BB, SelectionDAGLowering &SDL,
               std::vector<SDOperand> &UnorderedChains) {
  // If this is the entry block, emit arguments.
  Function &F = *BB->getParent();
  FunctionLoweringInfo &FuncInfo = SDL.FuncInfo;

  if (BB == &F.front()) {
    SDOperand OldRoot = SDL.DAG.getRoot();

    std::vector<SDOperand> Args = TLI.LowerArguments(F, SDL.DAG);

    // If there were side effects accessing the argument list, do not do
    // anything special.
    if (OldRoot != SDL.DAG.getRoot()) {
      unsigned a = 0;
      for (Function::aiterator AI = F.abegin(), E = F.aend(); AI != E; ++AI,++a)
        if (!AI->use_empty()) {
          SDL.setValue(AI, Args[a]);
          SDOperand Copy = 
            CopyValueToVirtualRegister(SDL, AI, FuncInfo.ValueMap[AI]);
          UnorderedChains.push_back(Copy);
        }
    } else {
      // Otherwise, if any argument is only accessed in a single basic block,
      // emit that argument only to that basic block.
      unsigned a = 0;
      for (Function::aiterator AI = F.abegin(), E = F.aend(); AI != E; ++AI,++a)
        if (!AI->use_empty()) {
          if (BasicBlock *BBU = IsOnlyUsedInOneBasicBlock(AI)) {
            FuncInfo.BlockLocalArguments.insert(std::make_pair(BBU,
                                                      std::make_pair(AI, a)));
          } else {
            SDL.setValue(AI, Args[a]);
            SDOperand Copy = 
              CopyValueToVirtualRegister(SDL, AI, FuncInfo.ValueMap[AI]);
            UnorderedChains.push_back(Copy);
          }
        }
    }
  }

  // See if there are any block-local arguments that need to be emitted in this
  // block.

  if (!FuncInfo.BlockLocalArguments.empty()) {
    std::multimap<BasicBlock*, std::pair<Argument*, unsigned> >::iterator BLAI =
      FuncInfo.BlockLocalArguments.lower_bound(BB);
    if (BLAI != FuncInfo.BlockLocalArguments.end() && BLAI->first == BB) {
      // Lower the arguments into this block.
      std::vector<SDOperand> Args = TLI.LowerArguments(F, SDL.DAG);
      
      // Set up the value mapping for the local arguments.
      for (; BLAI != FuncInfo.BlockLocalArguments.end() && BLAI->first == BB;
           ++BLAI)
        SDL.setValue(BLAI->second.first, Args[BLAI->second.second]);
      
      // Any dead arguments will just be ignored here.
    }
  }
}


void SelectionDAGISel::BuildSelectionDAG(SelectionDAG &DAG, BasicBlock *LLVMBB,
       std::vector<std::pair<MachineInstr*, unsigned> > &PHINodesToUpdate,
                                    FunctionLoweringInfo &FuncInfo) {
  SelectionDAGLowering SDL(DAG, TLI, FuncInfo);

  std::vector<SDOperand> UnorderedChains;
  
  // Lower any arguments needed in this block.
  LowerArguments(LLVMBB, SDL, UnorderedChains);

  BB = FuncInfo.MBBMap[LLVMBB];
  SDL.setCurrentBasicBlock(BB);

  // Lower all of the non-terminator instructions.
  for (BasicBlock::iterator I = LLVMBB->begin(), E = --LLVMBB->end();
       I != E; ++I)
    SDL.visit(*I);

  // Ensure that all instructions which are used outside of their defining
  // blocks are available as virtual registers.
  for (BasicBlock::iterator I = LLVMBB->begin(), E = LLVMBB->end(); I != E;++I)
    if (!I->use_empty() && !isa<PHINode>(I)) {
      std::map<const Value*, unsigned>::iterator VMI =FuncInfo.ValueMap.find(I);
      if (VMI != FuncInfo.ValueMap.end())
        UnorderedChains.push_back(
                           CopyValueToVirtualRegister(SDL, I, VMI->second));
    }

  // Handle PHI nodes in successor blocks.  Emit code into the SelectionDAG to
  // ensure constants are generated when needed.  Remember the virtual registers
  // that need to be added to the Machine PHI nodes as input.  We cannot just
  // directly add them, because expansion might result in multiple MBB's for one
  // BB.  As such, the start of the BB might correspond to a different MBB than
  // the end.
  // 

  // Emit constants only once even if used by multiple PHI nodes.
  std::map<Constant*, unsigned> ConstantsOut;

  // Check successor nodes PHI nodes that expect a constant to be available from
  // this block.
  TerminatorInst *TI = LLVMBB->getTerminator();
  for (unsigned succ = 0, e = TI->getNumSuccessors(); succ != e; ++succ) {
    BasicBlock *SuccBB = TI->getSuccessor(succ);
    MachineBasicBlock::iterator MBBI = FuncInfo.MBBMap[SuccBB]->begin();
    PHINode *PN;

    // At this point we know that there is a 1-1 correspondence between LLVM PHI
    // nodes and Machine PHI nodes, but the incoming operands have not been
    // emitted yet.
    for (BasicBlock::iterator I = SuccBB->begin();
         (PN = dyn_cast<PHINode>(I)); ++I)
      if (!PN->use_empty()) {
        unsigned Reg;
        Value *PHIOp = PN->getIncomingValueForBlock(LLVMBB);
        if (Constant *C = dyn_cast<Constant>(PHIOp)) {
          unsigned &RegOut = ConstantsOut[C];
          if (RegOut == 0) {
            RegOut = FuncInfo.CreateRegForValue(C);
            UnorderedChains.push_back(
                             CopyValueToVirtualRegister(SDL, C, RegOut));
          }
          Reg = RegOut;
        } else {
          Reg = FuncInfo.ValueMap[PHIOp];
          if (Reg == 0) {
            assert(isa<AllocaInst>(PHIOp) && 
                   FuncInfo.StaticAllocaMap.count(cast<AllocaInst>(PHIOp)) &&
                   "Didn't codegen value into a register!??");
            Reg = FuncInfo.CreateRegForValue(PHIOp);
            UnorderedChains.push_back(
                             CopyValueToVirtualRegister(SDL, PHIOp, Reg));
          }
        }
        
        // Remember that this register needs to added to the machine PHI node as
        // the input for this MBB.
        unsigned NumElements =
          TLI.getNumElements(TLI.getValueType(PN->getType()));
        for (unsigned i = 0, e = NumElements; i != e; ++i)
          PHINodesToUpdate.push_back(std::make_pair(MBBI++, Reg+i));
      }
  }
  ConstantsOut.clear();

  // Turn all of the unordered chains into one factored node.
  if (!UnorderedChains.empty()) {
    UnorderedChains.push_back(SDL.getRoot());
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, UnorderedChains));
  }

  // Lower the terminator after the copies are emitted.
  SDL.visit(*LLVMBB->getTerminator());

  // Make sure the root of the DAG is up-to-date.
  DAG.setRoot(SDL.getRoot());
}

void SelectionDAGISel::SelectBasicBlock(BasicBlock *LLVMBB, MachineFunction &MF,
                                        FunctionLoweringInfo &FuncInfo) {
  SelectionDAG DAG(TLI, MF);
  CurDAG = &DAG;
  std::vector<std::pair<MachineInstr*, unsigned> > PHINodesToUpdate;

  // First step, lower LLVM code to some DAG.  This DAG may use operations and
  // types that are not supported by the target.
  BuildSelectionDAG(DAG, LLVMBB, PHINodesToUpdate, FuncInfo);

  DEBUG(std::cerr << "Lowered selection DAG:\n");
  DEBUG(DAG.dump());

  // Second step, hack on the DAG until it only uses operations and types that
  // the target supports.
  DAG.Legalize();

  DEBUG(std::cerr << "Legalized selection DAG:\n");
  DEBUG(DAG.dump());

  // Finally, instruction select all of the operations to machine code, adding
  // the code to the MachineBasicBlock.
  InstructionSelectBasicBlock(DAG);

  if (ViewDAGs) DAG.viewGraph();

  DEBUG(std::cerr << "Selected machine code:\n");
  DEBUG(BB->dump());

  // Finally, now that we know what the last MBB the LLVM BB expanded is, update
  // PHI nodes in successors.
  for (unsigned i = 0, e = PHINodesToUpdate.size(); i != e; ++i) {
    MachineInstr *PHI = PHINodesToUpdate[i].first;
    assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
           "This is not a machine PHI node that we are updating!");
    PHI->addRegOperand(PHINodesToUpdate[i].second);
    PHI->addMachineBasicBlockOperand(BB);
  }
}
