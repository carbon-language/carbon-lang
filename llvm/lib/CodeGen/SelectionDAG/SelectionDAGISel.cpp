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
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineDebugInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <set>
#include <iostream>
using namespace llvm;

#ifndef NDEBUG
static cl::opt<bool>
ViewISelDAGs("view-isel-dags", cl::Hidden,
          cl::desc("Pop up a window to show isel dags as they are selected"));
static cl::opt<bool>
ViewSchedDAGs("view-sched-dags", cl::Hidden,
          cl::desc("Pop up a window to show sched dags as they are processed"));
#else
static const bool ViewISelDAGs = 0;
static const bool ViewSchedDAGs = 0;
#endif

namespace {
  cl::opt<SchedHeuristics>
  ISHeuristic(
    "sched",
    cl::desc("Choose scheduling style"),
    cl::init(defaultScheduling),
    cl::values(
      clEnumValN(defaultScheduling, "default",
                 "Target preferred scheduling style"),
      clEnumValN(noScheduling, "none",
                 "No scheduling: breadth first sequencing"),
      clEnumValN(simpleScheduling, "simple",
                 "Simple two pass scheduling: minimize critical path "
                 "and maximize processor utilization"),
      clEnumValN(simpleNoItinScheduling, "simple-noitin",
                 "Simple two pass scheduling: Same as simple "
                 "except using generic latency"),
      clEnumValN(listSchedulingBURR, "list-burr",
                 "Bottom up register reduction list scheduling"),
      clEnumValEnd));
} // namespace


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

/// isOnlyUsedInEntryBlock - If the specified argument is only used in the
/// entry block, return true.
static bool isOnlyUsedInEntryBlock(Argument *A) {
  BasicBlock *Entry = A->getParent()->begin();
  for (Value::use_iterator UI = A->use_begin(), E = A->use_end(); UI != E; ++UI)
    if (cast<Instruction>(*UI)->getParent() != Entry)
      return false;  // Use not in entry block.
  return true;
}

FunctionLoweringInfo::FunctionLoweringInfo(TargetLowering &tli,
                                           Function &fn, MachineFunction &mf)
    : TLI(tli), Fn(fn), MF(mf), RegMap(MF.getSSARegMap()) {

  // Create a vreg for each argument register that is not dead and is used
  // outside of the entry block for the function.
  for (Function::arg_iterator AI = Fn.arg_begin(), E = Fn.arg_end();
       AI != E; ++AI)
    if (!isOnlyUsedInEntryBlock(AI))
      InitializeRegForValue(AI);

  // Initialize the mapping of values to registers.  This is only set up for
  // instruction values that are used outside of the block that defines
  // them.
  Function::iterator BB = Fn.begin(), EB = Fn.end();
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(AI->getArraySize())) {
        const Type *Ty = AI->getAllocatedType();
        uint64_t TySize = TLI.getTargetData().getTypeSize(Ty);
        unsigned Align = 
          std::max((unsigned)TLI.getTargetData().getTypeAlignment(Ty),
                   AI->getAlignment());

        // If the alignment of the value is smaller than the size of the value,
        // and if the size of the value is particularly small (<= 8 bytes),
        // round up to the size of the value for potentially better performance.
        //
        // FIXME: This could be made better with a preferred alignment hook in
        // TargetData.  It serves primarily to 8-byte align doubles for X86.
        if (Align < TySize && TySize <= 8) Align = TySize;
        TySize *= CUI->getValue();   // Get total allocated size.
        if (TySize == 0) TySize = 1; // Don't create zero-sized stack objects.
        StaticAllocaMap[AI] =
          MF.getFrameInfo()->CreateStackObject((unsigned)TySize, Align);
      }

  for (; BB != EB; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (!I->use_empty() && isUsedOutsideOfDefiningBlock(I))
        if (!isa<AllocaInst>(I) ||
            !StaticAllocaMap.count(cast<AllocaInst>(I)))
          InitializeRegForValue(I);

  // Create an initial MachineBasicBlock for each LLVM BasicBlock in F.  This
  // also creates the initial PHI MachineInstrs, though none of the input
  // operands are populated.
  for (BB = Fn.begin(), EB = Fn.end(); BB != EB; ++BB) {
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

    const Type *VTy = V->getType();
    MVT::ValueType VT = TLI.getValueType(VTy);
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
        return N = DAG.getNode(ISD::UNDEF, VT);
      } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
        return N = DAG.getConstantFP(CFP->getValue(), VT);
      } else if (const PackedType *PTy = dyn_cast<PackedType>(VTy)) {
        unsigned NumElements = PTy->getNumElements();
        MVT::ValueType PVT = TLI.getValueType(PTy->getElementType());
        MVT::ValueType TVT = MVT::getVectorType(PVT, NumElements);
        
        // Now that we know the number and type of the elements, push a
        // Constant or ConstantFP node onto the ops list for each element of
        // the packed constant.
        std::vector<SDOperand> Ops;
        if (ConstantPacked *CP = dyn_cast<ConstantPacked>(C)) {
          if (MVT::isFloatingPoint(PVT)) {
            for (unsigned i = 0; i != NumElements; ++i) {
              const ConstantFP *El = cast<ConstantFP>(CP->getOperand(i));
              Ops.push_back(DAG.getConstantFP(El->getValue(), PVT));
            }
          } else {
            for (unsigned i = 0; i != NumElements; ++i) {
              const ConstantIntegral *El = 
                cast<ConstantIntegral>(CP->getOperand(i));
              Ops.push_back(DAG.getConstant(El->getRawValue(), PVT));
            }
          }
        } else {
          assert(isa<ConstantAggregateZero>(C) && "Unknown packed constant!");
          SDOperand Op;
          if (MVT::isFloatingPoint(PVT))
            Op = DAG.getConstantFP(0, PVT);
          else
            Op = DAG.getConstant(0, PVT);
          Ops.assign(NumElements, Op);
        }
        
        // Handle the case where we have a 1-element vector, in which
        // case we want to immediately turn it into a scalar constant.
        if (Ops.size() == 1) {
          return N = Ops[0];
        } else if (TVT != MVT::Other && TLI.isTypeLegal(TVT)) {
          return N = DAG.getNode(ISD::ConstantVec, TVT, Ops);
        } else {
          // If the packed type isn't legal, then create a ConstantVec node with
          // generic Vector type instead.
          return N = DAG.getNode(ISD::ConstantVec, MVT::Vector, Ops);
        }
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

    unsigned InReg = VMI->second;
   
    // If this type is not legal, make it so now.
    MVT::ValueType DestVT = TLI.getTypeToTransformTo(VT);
    
    N = DAG.getCopyFromReg(DAG.getEntryNode(), InReg, DestVT);
    if (DestVT < VT) {
      // Source must be expanded.  This input value is actually coming from the
      // register pair VMI->second and VMI->second+1.
      N = DAG.getNode(ISD::BUILD_PAIR, VT, N,
                      DAG.getCopyFromReg(DAG.getEntryNode(), InReg+1, DestVT));
    } else {
      if (DestVT > VT) { // Promotion case
        if (MVT::isFloatingPoint(VT))
          N = DAG.getNode(ISD::FP_ROUND, VT, N);
        else
          N = DAG.getNode(ISD::TRUNCATE, VT, N);
      }
    }
    
    return N;
  }

  const SDOperand &setValue(const Value *V, SDOperand NewN) {
    SDOperand &N = NodeMap[V];
    assert(N.Val == 0 && "Already set a value for this node!");
    return N = NewN;
  }
  
  unsigned GetAvailableRegister(bool OutReg, bool InReg,
                                const std::vector<unsigned> &RegChoices,
                                std::set<unsigned> &OutputRegs, 
                                std::set<unsigned> &InputRegs);

  // Terminator instructions.
  void visitRet(ReturnInst &I);
  void visitBr(BranchInst &I);
  void visitUnreachable(UnreachableInst &I) { /* noop */ }

  // These all get lowered before this pass.
  void visitExtractElement(ExtractElementInst &I) { assert(0 && "TODO"); }
  void visitInsertElement(InsertElementInst &I) { assert(0 && "TODO"); }
  void visitSwitch(SwitchInst &I) { assert(0 && "TODO"); }
  void visitInvoke(InvokeInst &I) { assert(0 && "TODO"); }
  void visitUnwind(UnwindInst &I) { assert(0 && "TODO"); }

  //
  void visitBinary(User &I, unsigned IntOp, unsigned FPOp, unsigned VecOp);
  void visitShift(User &I, unsigned Opcode);
  void visitAdd(User &I) { 
    visitBinary(I, ISD::ADD, ISD::FADD, ISD::VADD); 
  }
  void visitSub(User &I);
  void visitMul(User &I) { 
    visitBinary(I, ISD::MUL, ISD::FMUL, ISD::VMUL); 
  }
  void visitDiv(User &I) {
    const Type *Ty = I.getType();
    visitBinary(I, Ty->isSigned() ? ISD::SDIV : ISD::UDIV, ISD::FDIV, 0);
  }
  void visitRem(User &I) {
    const Type *Ty = I.getType();
    visitBinary(I, Ty->isSigned() ? ISD::SREM : ISD::UREM, ISD::FREM, 0);
  }
  void visitAnd(User &I) { visitBinary(I, ISD::AND, 0, 0); }
  void visitOr (User &I) { visitBinary(I, ISD::OR,  0, 0); }
  void visitXor(User &I) { visitBinary(I, ISD::XOR, 0, 0); }
  void visitShl(User &I) { visitShift(I, ISD::SHL); }
  void visitShr(User &I) { 
    visitShift(I, I.getType()->isUnsigned() ? ISD::SRL : ISD::SRA);
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
  void visitInlineAsm(CallInst &I);
  const char *visitIntrinsicCall(CallInst &I, unsigned Intrinsic);

  void visitVAStart(CallInst &I);
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
  std::vector<SDOperand> NewValues;
  NewValues.push_back(getRoot());
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    SDOperand RetOp = getValue(I.getOperand(i));
    
    // If this is an integer return value, we need to promote it ourselves to
    // the full width of a register, since LegalizeOp will use ANY_EXTEND rather
    // than sign/zero.
    if (MVT::isInteger(RetOp.getValueType()) && 
        RetOp.getValueType() < MVT::i64) {
      MVT::ValueType TmpVT;
      if (TLI.getTypeAction(MVT::i32) == TargetLowering::Promote)
        TmpVT = TLI.getTypeToTransformTo(MVT::i32);
      else
        TmpVT = MVT::i32;

      if (I.getOperand(i)->getType()->isSigned())
        RetOp = DAG.getNode(ISD::SIGN_EXTEND, TmpVT, RetOp);
      else
        RetOp = DAG.getNode(ISD::ZERO_EXTEND, TmpVT, RetOp);
    }
    NewValues.push_back(RetOp);
  }
  DAG.setRoot(DAG.getNode(ISD::RET, MVT::Other, NewValues));
}

void SelectionDAGLowering::visitBr(BranchInst &I) {
  // Update machine-CFG edges.
  MachineBasicBlock *Succ0MBB = FuncInfo.MBBMap[I.getSuccessor(0)];

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
      std::vector<SDOperand> Ops;
      Ops.push_back(getRoot());
      Ops.push_back(Cond);
      Ops.push_back(DAG.getBasicBlock(Succ0MBB));
      Ops.push_back(DAG.getBasicBlock(Succ1MBB));
      DAG.setRoot(DAG.getNode(ISD::BRCONDTWOWAY, MVT::Other, Ops));
    }
  }
}

void SelectionDAGLowering::visitSub(User &I) {
  // -0.0 - X --> fneg
  if (I.getType()->isFloatingPoint()) {
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(I.getOperand(0)))
      if (CFP->isExactlyValue(-0.0)) {
        SDOperand Op2 = getValue(I.getOperand(1));
        setValue(&I, DAG.getNode(ISD::FNEG, Op2.getValueType(), Op2));
        return;
      }
  }
  visitBinary(I, ISD::SUB, ISD::FSUB, ISD::VSUB);
}

void SelectionDAGLowering::visitBinary(User &I, unsigned IntOp, unsigned FPOp, 
                                       unsigned VecOp) {
  const Type *Ty = I.getType();
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));

  if (Ty->isIntegral()) {
    setValue(&I, DAG.getNode(IntOp, Op1.getValueType(), Op1, Op2));
  } else if (Ty->isFloatingPoint()) {
    setValue(&I, DAG.getNode(FPOp, Op1.getValueType(), Op1, Op2));
  } else {
    const PackedType *PTy = cast<PackedType>(Ty);
    unsigned NumElements = PTy->getNumElements();
    MVT::ValueType PVT = TLI.getValueType(PTy->getElementType());
    MVT::ValueType TVT = MVT::getVectorType(PVT, NumElements);
    
    // Immediately scalarize packed types containing only one element, so that
    // the Legalize pass does not have to deal with them.  Similarly, if the
    // abstract vector is going to turn into one that the target natively
    // supports, generate that type now so that Legalize doesn't have to deal
    // with that either.  These steps ensure that Legalize only has to handle
    // vector types in its Expand case.
    unsigned Opc = MVT::isFloatingPoint(PVT) ? FPOp : IntOp;
    if (NumElements == 1) {
      setValue(&I, DAG.getNode(Opc, PVT, Op1, Op2));
    } else if (TVT != MVT::Other && TLI.isTypeLegal(TVT)) {
      setValue(&I, DAG.getNode(Opc, TVT, Op1, Op2));
    } else {
      SDOperand Num = DAG.getConstant(NumElements, MVT::i32);
      SDOperand Typ = DAG.getValueType(PVT);
      setValue(&I, DAG.getNode(VecOp, MVT::Vector, Op1, Op2, Num, Typ));
    }
  }
}

void SelectionDAGLowering::visitShift(User &I, unsigned Opcode) {
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  
  Op2 = DAG.getNode(ISD::ANY_EXTEND, TLI.getShiftAmountTy(), Op2);
  
  setValue(&I, DAG.getNode(Opcode, Op1.getValueType(), Op1, Op2));
}

void SelectionDAGLowering::visitSetCC(User &I,ISD::CondCode SignedOpcode,
                                      ISD::CondCode UnsignedOpcode) {
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  ISD::CondCode Opcode = SignedOpcode;
  if (I.getOperand(0)->getType()->isUnsigned())
    Opcode = UnsignedOpcode;
  setValue(&I, DAG.getSetCC(MVT::i1, Op1, Op2, Opcode));
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
  } else if (DestTy == MVT::i1) {
    // Cast to bool is a comparison against zero, not truncation to zero.
    SDOperand Zero = isInteger(SrcTy) ? DAG.getConstant(0, N.getValueType()) :
                                       DAG.getConstantFP(0.0, N.getValueType());
    setValue(&I, DAG.getSetCC(MVT::i1, N, Zero, ISD::SETNE));
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
    if (const StructType *StTy = dyn_cast<StructType>(Ty)) {
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

      // If this is a constant subscript, handle it quickly.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Idx)) {
        if (CI->getRawValue() == 0) continue;

        uint64_t Offs;
        if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(CI))
          Offs = (int64_t)TD.getTypeSize(Ty)*CSI->getValue();
        else
          Offs = TD.getTypeSize(Ty)*cast<ConstantUInt>(CI)->getValue();
        N = DAG.getNode(ISD::ADD, N.getValueType(), N, getIntPtrConstant(Offs));
        continue;
      }
      
      // N = N + Idx * ElementSize;
      uint64_t ElementSize = TD.getTypeSize(Ty);
      SDOperand IdxN = getValue(Idx);

      // If the index is smaller or larger than intptr_t, truncate or extend
      // it.
      if (IdxN.getValueType() < N.getValueType()) {
        if (Idx->getType()->isSigned())
          IdxN = DAG.getNode(ISD::SIGN_EXTEND, N.getValueType(), IdxN);
        else
          IdxN = DAG.getNode(ISD::ZERO_EXTEND, N.getValueType(), IdxN);
      } else if (IdxN.getValueType() > N.getValueType())
        IdxN = DAG.getNode(ISD::TRUNCATE, N.getValueType(), IdxN);

      // If this is a multiply by a power of two, turn it into a shl
      // immediately.  This is a very common case.
      if (isPowerOf2_64(ElementSize)) {
        unsigned Amt = Log2_64(ElementSize);
        IdxN = DAG.getNode(ISD::SHL, N.getValueType(), IdxN,
                           DAG.getConstant(Amt, TLI.getShiftAmountTy()));
        N = DAG.getNode(ISD::ADD, N.getValueType(), N, IdxN);
        continue;
      }
      
      SDOperand Scale = getIntPtrConstant(ElementSize);
      IdxN = DAG.getNode(ISD::MUL, N.getValueType(), IdxN, Scale);
      N = DAG.getNode(ISD::ADD, N.getValueType(), N, IdxN);
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
  unsigned Align = std::max((unsigned)TLI.getTargetData().getTypeAlignment(Ty),
                            I.getAlignment());

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

  std::vector<MVT::ValueType> VTs;
  VTs.push_back(AllocSize.getValueType());
  VTs.push_back(MVT::Other);
  std::vector<SDOperand> Ops;
  Ops.push_back(getRoot());
  Ops.push_back(AllocSize);
  Ops.push_back(getIntPtrConstant(Align));
  SDOperand DSA = DAG.getNode(ISD::DYNAMIC_STACKALLOC, VTs, Ops);
  DAG.setRoot(setValue(&I, DSA).getValue(1));

  // Inform the Frame Information that we have just allocated a variable-sized
  // object.
  CurMBB->getParent()->getFrameInfo()->CreateVariableSizedObject();
}

/// getStringValue - Turn an LLVM constant pointer that eventually points to a
/// global into a string value.  Return an empty string if we can't do it.
///
static std::string getStringValue(GlobalVariable *GV, unsigned Offset = 0) {
  if (GV->hasInitializer() && isa<ConstantArray>(GV->getInitializer())) {
    ConstantArray *Init = cast<ConstantArray>(GV->getInitializer());
    if (Init->isString()) {
      std::string Result = Init->getAsString();
      if (Offset < Result.size()) {
        // If we are pointing INTO The string, erase the beginning...
        Result.erase(Result.begin(), Result.begin()+Offset);
        return Result;
      }
    }
  }
  return "";
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
  
  const Type *Ty = I.getType();
  SDOperand L;
  
  if (const PackedType *PTy = dyn_cast<PackedType>(Ty)) {
    unsigned NumElements = PTy->getNumElements();
    MVT::ValueType PVT = TLI.getValueType(PTy->getElementType());
    MVT::ValueType TVT = MVT::getVectorType(PVT, NumElements);
    
    // Immediately scalarize packed types containing only one element, so that
    // the Legalize pass does not have to deal with them.
    if (NumElements == 1) {
      L = DAG.getLoad(PVT, Root, Ptr, DAG.getSrcValue(I.getOperand(0)));
    } else if (TVT != MVT::Other && TLI.isTypeLegal(TVT)) {
      L = DAG.getLoad(TVT, Root, Ptr, DAG.getSrcValue(I.getOperand(0)));
    } else {
      L = DAG.getVecLoad(NumElements, PVT, Root, Ptr, 
                         DAG.getSrcValue(I.getOperand(0)));
    }
  } else {
    L = DAG.getLoad(TLI.getValueType(Ty), Root, Ptr, 
                    DAG.getSrcValue(I.getOperand(0)));
  }
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
  DAG.setRoot(DAG.getNode(ISD::STORE, MVT::Other, getRoot(), Src, Ptr,
                          DAG.getSrcValue(I.getOperand(1))));
}

/// visitIntrinsicCall - Lower the call to the specified intrinsic function.  If
/// we want to emit this as a call to a named external function, return the name
/// otherwise lower it and return null.
const char *
SelectionDAGLowering::visitIntrinsicCall(CallInst &I, unsigned Intrinsic) {
  switch (Intrinsic) {
  case Intrinsic::vastart:  visitVAStart(I); return 0;
  case Intrinsic::vaend:    visitVAEnd(I); return 0;
  case Intrinsic::vacopy:   visitVACopy(I); return 0;
  case Intrinsic::returnaddress: visitFrameReturnAddress(I, false); return 0;
  case Intrinsic::frameaddress:  visitFrameReturnAddress(I, true); return 0;
  case Intrinsic::setjmp:
    return "_setjmp"+!TLI.usesUnderscoreSetJmpLongJmp();
    break;
  case Intrinsic::longjmp:
    return "_longjmp"+!TLI.usesUnderscoreSetJmpLongJmp();
    break;
  case Intrinsic::memcpy:  visitMemIntrinsic(I, ISD::MEMCPY); return 0;
  case Intrinsic::memset:  visitMemIntrinsic(I, ISD::MEMSET); return 0;
  case Intrinsic::memmove: visitMemIntrinsic(I, ISD::MEMMOVE); return 0;
    
  case Intrinsic::readport:
  case Intrinsic::readio: {
    std::vector<MVT::ValueType> VTs;
    VTs.push_back(TLI.getValueType(I.getType()));
    VTs.push_back(MVT::Other);
    std::vector<SDOperand> Ops;
    Ops.push_back(getRoot());
    Ops.push_back(getValue(I.getOperand(1)));
    SDOperand Tmp = DAG.getNode(Intrinsic == Intrinsic::readport ?
                                ISD::READPORT : ISD::READIO, VTs, Ops);
    
    setValue(&I, Tmp);
    DAG.setRoot(Tmp.getValue(1));
    return 0;
  }
  case Intrinsic::writeport:
  case Intrinsic::writeio:
    DAG.setRoot(DAG.getNode(Intrinsic == Intrinsic::writeport ?
                            ISD::WRITEPORT : ISD::WRITEIO, MVT::Other,
                            getRoot(), getValue(I.getOperand(1)),
                            getValue(I.getOperand(2))));
    return 0;
    
  case Intrinsic::dbg_stoppoint: {
    if (TLI.getTargetMachine().getIntrinsicLowering().EmitDebugFunctions())
      return "llvm_debugger_stop";
    
    MachineDebugInfo *DebugInfo = DAG.getMachineDebugInfo();
    if (DebugInfo &&  DebugInfo->Verify(I.getOperand(4))) {
      std::vector<SDOperand> Ops;

      // Input Chain
      Ops.push_back(getRoot());
      
      // line number
      Ops.push_back(getValue(I.getOperand(2)));
     
      // column
      Ops.push_back(getValue(I.getOperand(3)));

      DebugInfoDesc *DD = DebugInfo->getDescFor(I.getOperand(4));
      assert(DD && "Not a debug information descriptor");
      CompileUnitDesc *CompileUnit = dyn_cast<CompileUnitDesc>(DD);
      assert(CompileUnit && "Not a compile unit");
      Ops.push_back(DAG.getString(CompileUnit->getFileName()));
      Ops.push_back(DAG.getString(CompileUnit->getDirectory()));
      
      if (Ops.size() == 5)  // Found filename/workingdir.
        DAG.setRoot(DAG.getNode(ISD::LOCATION, MVT::Other, Ops));
    }
    
    setValue(&I, DAG.getNode(ISD::UNDEF, TLI.getValueType(I.getType())));
    return 0;
  }
  case Intrinsic::dbg_region_start:
    if (TLI.getTargetMachine().getIntrinsicLowering().EmitDebugFunctions())
      return "llvm_dbg_region_start";
    if (I.getType() != Type::VoidTy)
      setValue(&I, DAG.getNode(ISD::UNDEF, TLI.getValueType(I.getType())));
    return 0;
  case Intrinsic::dbg_region_end:
    if (TLI.getTargetMachine().getIntrinsicLowering().EmitDebugFunctions())
      return "llvm_dbg_region_end";
    if (I.getType() != Type::VoidTy)
      setValue(&I, DAG.getNode(ISD::UNDEF, TLI.getValueType(I.getType())));
    return 0;
  case Intrinsic::dbg_func_start:
    if (TLI.getTargetMachine().getIntrinsicLowering().EmitDebugFunctions())
      return "llvm_dbg_subprogram";
    if (I.getType() != Type::VoidTy)
      setValue(&I, DAG.getNode(ISD::UNDEF, TLI.getValueType(I.getType())));
    return 0;
  case Intrinsic::dbg_declare:
    if (I.getType() != Type::VoidTy)
      setValue(&I, DAG.getNode(ISD::UNDEF, TLI.getValueType(I.getType())));
    return 0;
    
  case Intrinsic::isunordered_f32:
  case Intrinsic::isunordered_f64:
    setValue(&I, DAG.getSetCC(MVT::i1,getValue(I.getOperand(1)),
                              getValue(I.getOperand(2)), ISD::SETUO));
    return 0;
    
  case Intrinsic::sqrt_f32:
  case Intrinsic::sqrt_f64:
    setValue(&I, DAG.getNode(ISD::FSQRT,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::pcmarker: {
    SDOperand Tmp = getValue(I.getOperand(1));
    DAG.setRoot(DAG.getNode(ISD::PCMARKER, MVT::Other, getRoot(), Tmp));
    return 0;
  }
  case Intrinsic::readcyclecounter: {
    std::vector<MVT::ValueType> VTs;
    VTs.push_back(MVT::i64);
    VTs.push_back(MVT::Other);
    std::vector<SDOperand> Ops;
    Ops.push_back(getRoot());
    SDOperand Tmp = DAG.getNode(ISD::READCYCLECOUNTER, VTs, Ops);
    setValue(&I, Tmp);
    DAG.setRoot(Tmp.getValue(1));
    return 0;
  }
  case Intrinsic::bswap_i16:
  case Intrinsic::bswap_i32:
  case Intrinsic::bswap_i64:
    setValue(&I, DAG.getNode(ISD::BSWAP,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::cttz_i8:
  case Intrinsic::cttz_i16:
  case Intrinsic::cttz_i32:
  case Intrinsic::cttz_i64:
    setValue(&I, DAG.getNode(ISD::CTTZ,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::ctlz_i8:
  case Intrinsic::ctlz_i16:
  case Intrinsic::ctlz_i32:
  case Intrinsic::ctlz_i64:
    setValue(&I, DAG.getNode(ISD::CTLZ,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::ctpop_i8:
  case Intrinsic::ctpop_i16:
  case Intrinsic::ctpop_i32:
  case Intrinsic::ctpop_i64:
    setValue(&I, DAG.getNode(ISD::CTPOP,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::stacksave: {
    std::vector<MVT::ValueType> VTs;
    VTs.push_back(TLI.getPointerTy());
    VTs.push_back(MVT::Other);
    std::vector<SDOperand> Ops;
    Ops.push_back(getRoot());
    SDOperand Tmp = DAG.getNode(ISD::STACKSAVE, VTs, Ops);
    setValue(&I, Tmp);
    DAG.setRoot(Tmp.getValue(1));
    return 0;
  }
  case Intrinsic::stackrestore: {
    SDOperand Tmp = getValue(I.getOperand(1));
    DAG.setRoot(DAG.getNode(ISD::STACKRESTORE, MVT::Other, getRoot(), Tmp));
    return 0;
  }
  case Intrinsic::prefetch:
    // FIXME: Currently discarding prefetches.
    return 0;
  default:
    std::cerr << I;
    assert(0 && "This intrinsic is not implemented yet!");
    return 0;
  }
}


void SelectionDAGLowering::visitCall(CallInst &I) {
  const char *RenameFn = 0;
  if (Function *F = I.getCalledFunction()) {
    if (F->isExternal())
      if (unsigned IID = F->getIntrinsicID()) {
        RenameFn = visitIntrinsicCall(I, IID);
        if (!RenameFn)
          return;
      } else {    // Not an LLVM intrinsic.
        const std::string &Name = F->getName();
        if (Name[0] == 'f' && (Name == "fabs" || Name == "fabsf")) {
          if (I.getNumOperands() == 2 &&   // Basic sanity checks.
              I.getOperand(1)->getType()->isFloatingPoint() &&
              I.getType() == I.getOperand(1)->getType()) {
            SDOperand Tmp = getValue(I.getOperand(1));
            setValue(&I, DAG.getNode(ISD::FABS, Tmp.getValueType(), Tmp));
            return;
          }
        } else if (Name[0] == 's' && (Name == "sin" || Name == "sinf")) {
          if (I.getNumOperands() == 2 &&   // Basic sanity checks.
              I.getOperand(1)->getType()->isFloatingPoint() &&
              I.getType() == I.getOperand(1)->getType()) {
            SDOperand Tmp = getValue(I.getOperand(1));
            setValue(&I, DAG.getNode(ISD::FSIN, Tmp.getValueType(), Tmp));
            return;
          }
        } else if (Name[0] == 'c' && (Name == "cos" || Name == "cosf")) {
          if (I.getNumOperands() == 2 &&   // Basic sanity checks.
              I.getOperand(1)->getType()->isFloatingPoint() &&
              I.getType() == I.getOperand(1)->getType()) {
            SDOperand Tmp = getValue(I.getOperand(1));
            setValue(&I, DAG.getNode(ISD::FCOS, Tmp.getValueType(), Tmp));
            return;
          }
        }
      }
  } else if (isa<InlineAsm>(I.getOperand(0))) {
    visitInlineAsm(I);
    return;
  }

  SDOperand Callee;
  if (!RenameFn)
    Callee = getValue(I.getOperand(0));
  else
    Callee = DAG.getExternalSymbol(RenameFn, TLI.getPointerTy());
  std::vector<std::pair<SDOperand, const Type*> > Args;
  Args.reserve(I.getNumOperands());
  for (unsigned i = 1, e = I.getNumOperands(); i != e; ++i) {
    Value *Arg = I.getOperand(i);
    SDOperand ArgNode = getValue(Arg);
    Args.push_back(std::make_pair(ArgNode, Arg->getType()));
  }

  const PointerType *PT = cast<PointerType>(I.getCalledValue()->getType());
  const FunctionType *FTy = cast<FunctionType>(PT->getElementType());

  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerCallTo(getRoot(), I.getType(), FTy->isVarArg(), I.getCallingConv(),
                    I.isTailCall(), Callee, Args, DAG);
  if (I.getType() != Type::VoidTy)
    setValue(&I, Result.first);
  DAG.setRoot(Result.second);
}

/// GetAvailableRegister - Pick a register from RegChoices that is available
/// for input and/or output as specified by isOutReg/isInReg.  If an allocatable
/// register is found, it is returned and added to the specified set of used
/// registers.  If not, zero is returned.
unsigned SelectionDAGLowering::
GetAvailableRegister(bool isOutReg, bool isInReg,
                     const std::vector<unsigned> &RegChoices,
                     std::set<unsigned> &OutputRegs,
                     std::set<unsigned> &InputRegs) {
  const MRegisterInfo *MRI = DAG.getTarget().getRegisterInfo();
  MachineFunction &MF = *CurMBB->getParent();
  for (unsigned i = 0, e = RegChoices.size(); i != e; ++i) {
    unsigned Reg = RegChoices[i];
    // See if this register is available.
    if (isOutReg && OutputRegs.count(Reg)) continue;  // Already used.
    if (isInReg  && InputRegs.count(Reg)) continue;  // Already used.

    // Check to see if this register is allocatable (i.e. don't give out the
    // stack pointer).
    bool Found = false;
    for (MRegisterInfo::regclass_iterator RC = MRI->regclass_begin(),
         E = MRI->regclass_end(); !Found && RC != E; ++RC) {
      // NOTE: This isn't ideal.  In particular, this might allocate the
      // frame pointer in functions that need it (due to them not being taken
      // out of allocation, because a variable sized allocation hasn't been seen
      // yet).  This is a slight code pessimization, but should still work.
      for (TargetRegisterClass::iterator I = (*RC)->allocation_order_begin(MF),
           E = (*RC)->allocation_order_end(MF); I != E; ++I)
        if (*I == Reg) {
          Found = true;
          break;
        }
    }
    if (!Found) continue;
    
    // Okay, this register is good, return it.
    if (isOutReg) OutputRegs.insert(Reg);  // Mark used.
    if (isInReg)  InputRegs.insert(Reg);   // Mark used.
    return Reg;
  }
  return 0;
}

/// visitInlineAsm - Handle a call to an InlineAsm object.
///
void SelectionDAGLowering::visitInlineAsm(CallInst &I) {
  InlineAsm *IA = cast<InlineAsm>(I.getOperand(0));
  
  SDOperand AsmStr = DAG.getTargetExternalSymbol(IA->getAsmString().c_str(),
                                                 MVT::Other);

  // Note, we treat inline asms both with and without side-effects as the same.
  // If an inline asm doesn't have side effects and doesn't access memory, we
  // could not choose to not chain it.
  bool hasSideEffects = IA->hasSideEffects();

  std::vector<InlineAsm::ConstraintInfo> Constraints = IA->ParseConstraints();
  
  /// AsmNodeOperands - A list of pairs.  The first element is a register, the
  /// second is a bitfield where bit #0 is set if it is a use and bit #1 is set
  /// if it is a def of that register.
  std::vector<SDOperand> AsmNodeOperands;
  AsmNodeOperands.push_back(SDOperand());  // reserve space for input chain
  AsmNodeOperands.push_back(AsmStr);
  
  SDOperand Chain = getRoot();
  SDOperand Flag;
  
  // Loop over all of the inputs, copying the operand values into the
  // appropriate registers and processing the output regs.
  unsigned RetValReg = 0;
  std::vector<std::pair<unsigned, Value*> > IndirectStoresToEmit;
  unsigned OpNum = 1;
  bool FoundOutputConstraint = false;
  
  // We fully assign registers here at isel time.  This is not optimal, but
  // should work.  For register classes that correspond to LLVM classes, we
  // could let the LLVM RA do its thing, but we currently don't.  Do a prepass
  // over the constraints, collecting fixed registers that we know we can't use.
  std::set<unsigned> OutputRegs, InputRegs;
  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    assert(Constraints[i].Codes.size() == 1 && "Only handles one code so far!");
    std::string &ConstraintCode = Constraints[i].Codes[0];
    
    std::vector<unsigned> Regs =
      TLI.getRegForInlineAsmConstraint(ConstraintCode);
    if (Regs.size() != 1) continue;  // Not assigned a fixed reg.
    unsigned TheReg = Regs[0];
    
    switch (Constraints[i].Type) {
    case InlineAsm::isOutput:
      // We can't assign any other output to this register.
      OutputRegs.insert(TheReg);
      // If this is an early-clobber output, it cannot be assigned to the same
      // value as the input reg.
      if (Constraints[i].isEarlyClobber || Constraints[i].hasMatchingInput)
        InputRegs.insert(TheReg);
      break;
    case InlineAsm::isClobber:
      // Clobbered regs cannot be used as inputs or outputs.
      InputRegs.insert(TheReg);
      OutputRegs.insert(TheReg);
      break;
    case InlineAsm::isInput:
      // We can't assign any other input to this register.
      InputRegs.insert(TheReg);
      break;
    }
  }      
  
  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    assert(Constraints[i].Codes.size() == 1 && "Only handles one code so far!");
    std::string &ConstraintCode = Constraints[i].Codes[0];
    switch (Constraints[i].Type) {
    case InlineAsm::isOutput: {
      // Copy the output from the appropriate register.
      std::vector<unsigned> Regs =
        TLI.getRegForInlineAsmConstraint(ConstraintCode);

      // Find a regsister that we can use.
      unsigned DestReg;
      if (Regs.size() == 1)
        DestReg = Regs[0];
      else {
        bool UsesInputRegister = false;
        // If this is an early-clobber output, or if there is an input
        // constraint that matches this, we need to reserve the input register
        // so no other inputs allocate to it.
        if (Constraints[i].isEarlyClobber || Constraints[i].hasMatchingInput)
          UsesInputRegister = true;
        DestReg = GetAvailableRegister(true, UsesInputRegister, 
                                       Regs, OutputRegs, InputRegs);
      }
      
      assert(DestReg && "Couldn't allocate output reg!");

      const Type *OpTy;
      if (!Constraints[i].isIndirectOutput) {
        assert(!FoundOutputConstraint &&
               "Cannot have multiple output constraints yet!");
        FoundOutputConstraint = true;
        assert(I.getType() != Type::VoidTy && "Bad inline asm!");
        
        RetValReg = DestReg;
        OpTy = I.getType();
      } else {
        IndirectStoresToEmit.push_back(std::make_pair(DestReg,
                                                      I.getOperand(OpNum)));
        OpTy = I.getOperand(OpNum)->getType();
        OpTy = cast<PointerType>(OpTy)->getElementType();
        OpNum++;  // Consumes a call operand.
      }
      
      // Add information to the INLINEASM node to know that this register is
      // set.
      AsmNodeOperands.push_back(DAG.getRegister(DestReg,
                                                TLI.getValueType(OpTy)));
      AsmNodeOperands.push_back(DAG.getConstant(2, MVT::i32)); // ISDEF
      
      break;
    }
    case InlineAsm::isInput: {
      Value *Operand = I.getOperand(OpNum);
      const Type *OpTy = Operand->getType();
      OpNum++;  // Consumes a call operand.

      unsigned SrcReg;
      SDOperand ResOp;
      unsigned ResOpType;
      SDOperand InOperandVal = getValue(Operand);
      
      if (isdigit(ConstraintCode[0])) {    // Matching constraint?
        // If this is required to match an output register we have already set,
        // just use its register.
        unsigned OperandNo = atoi(ConstraintCode.c_str());
        SrcReg = cast<RegisterSDNode>(AsmNodeOperands[OperandNo*2+2])->getReg();
        ResOp = DAG.getRegister(SrcReg, TLI.getValueType(OpTy));
        ResOpType = 1;
        
        Chain = DAG.getCopyToReg(Chain, SrcReg, InOperandVal, Flag);
        Flag = Chain.getValue(1);
      } else {
        TargetLowering::ConstraintType CTy = TargetLowering::C_RegisterClass;
        if (ConstraintCode.size() == 1)   // not a physreg name.
          CTy = TLI.getConstraintType(ConstraintCode[0]);
        
        switch (CTy) {
        default: assert(0 && "Unknown constraint type! FAIL!");
        case TargetLowering::C_RegisterClass: {
          // Copy the input into the appropriate register.
          std::vector<unsigned> Regs =
            TLI.getRegForInlineAsmConstraint(ConstraintCode);
          if (Regs.size() == 1)
            SrcReg = Regs[0];
          else
            SrcReg = GetAvailableRegister(false, true, Regs, 
                                          OutputRegs, InputRegs);
          // FIXME: should be match fail.
          assert(SrcReg && "Wasn't able to allocate register!");
          Chain = DAG.getCopyToReg(Chain, SrcReg, InOperandVal, Flag);
          Flag = Chain.getValue(1);
          
          ResOp = DAG.getRegister(SrcReg, TLI.getValueType(OpTy));
          ResOpType = 1;
          break;
        }
        case TargetLowering::C_Other:
          if (!TLI.isOperandValidForConstraint(InOperandVal, ConstraintCode[0]))
            assert(0 && "MATCH FAIL!");
          ResOp = InOperandVal;
          ResOpType = 3;
          break;
        }
      }
      
      // Add information to the INLINEASM node to know about this input.
      AsmNodeOperands.push_back(ResOp);
      AsmNodeOperands.push_back(DAG.getConstant(ResOpType, MVT::i32));
      break;
    }
    case InlineAsm::isClobber:
      // Nothing to do.
      break;
    }
  }
  
  // Finish up input operands.
  AsmNodeOperands[0] = Chain;
  if (Flag.Val) AsmNodeOperands.push_back(Flag);
  
  std::vector<MVT::ValueType> VTs;
  VTs.push_back(MVT::Other);
  VTs.push_back(MVT::Flag);
  Chain = DAG.getNode(ISD::INLINEASM, VTs, AsmNodeOperands);
  Flag = Chain.getValue(1);

  // If this asm returns a register value, copy the result from that register
  // and set it as the value of the call.
  if (RetValReg) {
    SDOperand Val = DAG.getCopyFromReg(Chain, RetValReg,
                                       TLI.getValueType(I.getType()), Flag);
    Chain = Val.getValue(1);
    Flag  = Val.getValue(2);
    setValue(&I, Val);
  }
  
  std::vector<std::pair<SDOperand, Value*> > StoresToEmit;
  
  // Process indirect outputs, first output all of the flagged copies out of
  // physregs.
  for (unsigned i = 0, e = IndirectStoresToEmit.size(); i != e; ++i) {
    Value *Ptr = IndirectStoresToEmit[i].second;
    const Type *Ty = cast<PointerType>(Ptr->getType())->getElementType();
    SDOperand Val = DAG.getCopyFromReg(Chain, IndirectStoresToEmit[i].first, 
                                       TLI.getValueType(Ty), Flag);
    Chain = Val.getValue(1);
    Flag  = Val.getValue(2);
    StoresToEmit.push_back(std::make_pair(Val, Ptr));
  }
  
  // Emit the non-flagged stores from the physregs.
  std::vector<SDOperand> OutChains;
  for (unsigned i = 0, e = StoresToEmit.size(); i != e; ++i)
    OutChains.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain, 
                                    StoresToEmit[i].first,
                                    getValue(StoresToEmit[i].second),
                                    DAG.getSrcValue(StoresToEmit[i].second)));
  if (!OutChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, OutChains);
  DAG.setRoot(Chain);
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
    TLI.LowerCallTo(getRoot(), I.getType(), false, CallingConv::C, true,
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
    TLI.LowerCallTo(getRoot(), Type::VoidTy, false, CallingConv::C, true,
                    DAG.getExternalSymbol("free", IntPtr), Args, DAG);
  DAG.setRoot(Result.second);
}

// InsertAtEndOfBasicBlock - This method should be implemented by targets that
// mark instructions with the 'usesCustomDAGSchedInserter' flag.  These
// instructions are special in various ways, which require special support to
// insert.  The specified MachineInstr is created but not inserted into any
// basic blocks, and the scheduler passes ownership of it to this method.
MachineBasicBlock *TargetLowering::InsertAtEndOfBasicBlock(MachineInstr *MI,
                                                       MachineBasicBlock *MBB) {
  std::cerr << "If a target marks an instruction with "
               "'usesCustomDAGSchedInserter', it must implement "
               "TargetLowering::InsertAtEndOfBasicBlock!\n";
  abort();
  return 0;  
}

void SelectionDAGLowering::visitVAStart(CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VASTART, MVT::Other, getRoot(), 
                          getValue(I.getOperand(1)), 
                          DAG.getSrcValue(I.getOperand(1))));
}

void SelectionDAGLowering::visitVAArg(VAArgInst &I) {
  SDOperand V = DAG.getVAArg(TLI.getValueType(I.getType()), getRoot(),
                             getValue(I.getOperand(0)),
                             DAG.getSrcValue(I.getOperand(0)));
  setValue(&I, V);
  DAG.setRoot(V.getValue(1));
}

void SelectionDAGLowering::visitVAEnd(CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VAEND, MVT::Other, getRoot(),
                          getValue(I.getOperand(1)), 
                          DAG.getSrcValue(I.getOperand(1))));
}

void SelectionDAGLowering::visitVACopy(CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VACOPY, MVT::Other, getRoot(), 
                          getValue(I.getOperand(1)), 
                          getValue(I.getOperand(2)),
                          DAG.getSrcValue(I.getOperand(1)),
                          DAG.getSrcValue(I.getOperand(2))));
}

// It is always conservatively correct for llvm.returnaddress and
// llvm.frameaddress to return 0.
std::pair<SDOperand, SDOperand>
TargetLowering::LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain,
                                        unsigned Depth, SelectionDAG &DAG) {
  return std::make_pair(DAG.getConstant(0, getPointerTy()), Chain);
}

SDOperand TargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  assert(0 && "LowerOperation not implemented for this target!");
  abort();
  return SDOperand();
}

SDOperand TargetLowering::CustomPromoteOperation(SDOperand Op,
                                                 SelectionDAG &DAG) {
  assert(0 && "CustomPromoteOperation not implemented for this target!");
  abort();
  return SDOperand();
}

void SelectionDAGLowering::visitFrameReturnAddress(CallInst &I, bool isFrame) {
  unsigned Depth = (unsigned)cast<ConstantUInt>(I.getOperand(1))->getValue();
  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerFrameReturnAddress(isFrame, getRoot(), Depth, DAG);
  setValue(&I, Result.first);
  DAG.setRoot(Result.second);
}

/// getMemsetValue - Vectorized representation of the memset value
/// operand.
static SDOperand getMemsetValue(SDOperand Value, MVT::ValueType VT,
                                SelectionDAG &DAG, TargetLowering &TLI) {
  MVT::ValueType CurVT = VT;
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Value)) {
    uint64_t Val   = C->getValue() & 255;
    unsigned Shift = 8;
    while (CurVT != MVT::i8) {
      Val = (Val << Shift) | Val;
      Shift <<= 1;
      CurVT = (MVT::ValueType)((unsigned)CurVT - 1);
    }
    return DAG.getConstant(Val, VT);
  } else {
    Value = DAG.getNode(ISD::ZERO_EXTEND, VT, Value);
    unsigned Shift = 8;
    while (CurVT != MVT::i8) {
      Value =
        DAG.getNode(ISD::OR, VT,
                    DAG.getNode(ISD::SHL, VT, Value,
                                DAG.getConstant(Shift, MVT::i8)), Value);
      Shift <<= 1;
      CurVT = (MVT::ValueType)((unsigned)CurVT - 1);
    }

    return Value;
  }
}

/// getMemsetStringVal - Similar to getMemsetValue. Except this is only
/// used when a memcpy is turned into a memset when the source is a constant
/// string ptr.
static SDOperand getMemsetStringVal(MVT::ValueType VT,
                                    SelectionDAG &DAG, TargetLowering &TLI,
                                    std::string &Str, unsigned Offset) {
  MVT::ValueType CurVT = VT;
  uint64_t Val = 0;
  unsigned MSB = getSizeInBits(VT) / 8;
  if (TLI.isLittleEndian())
    Offset = Offset + MSB - 1;
  for (unsigned i = 0; i != MSB; ++i) {
    Val = (Val << 8) | Str[Offset];
    Offset += TLI.isLittleEndian() ? -1 : 1;
  }
  return DAG.getConstant(Val, VT);
}

/// getMemBasePlusOffset - Returns base and offset node for the 
static SDOperand getMemBasePlusOffset(SDOperand Base, unsigned Offset,
                                      SelectionDAG &DAG, TargetLowering &TLI) {
  MVT::ValueType VT = Base.getValueType();
  return DAG.getNode(ISD::ADD, VT, Base, DAG.getConstant(Offset, VT));
}

/// MeetsMaxMemopRequirement - Determines if the number of memory ops required
/// to replace the memset / memcpy is below the threshold. It also returns the
/// types of the sequence of  memory ops to perform memset / memcpy.
static bool MeetsMaxMemopRequirement(std::vector<MVT::ValueType> &MemOps,
                                     unsigned Limit, uint64_t Size,
                                     unsigned Align, TargetLowering &TLI) {
  MVT::ValueType VT;

  if (TLI.allowsUnalignedMemoryAccesses()) {
    VT = MVT::i64;
  } else {
    switch (Align & 7) {
    case 0:
      VT = MVT::i64;
      break;
    case 4:
      VT = MVT::i32;
      break;
    case 2:
      VT = MVT::i16;
      break;
    default:
      VT = MVT::i8;
      break;
    }
  }

  MVT::ValueType LVT = MVT::i64;
  while (!TLI.isTypeLegal(LVT))
    LVT = (MVT::ValueType)((unsigned)LVT - 1);
  assert(MVT::isInteger(LVT));

  if (VT > LVT)
    VT = LVT;

  unsigned NumMemOps = 0;
  while (Size != 0) {
    unsigned VTSize = getSizeInBits(VT) / 8;
    while (VTSize > Size) {
      VT = (MVT::ValueType)((unsigned)VT - 1);
      VTSize >>= 1;
    }
    assert(MVT::isInteger(VT));

    if (++NumMemOps > Limit)
      return false;
    MemOps.push_back(VT);
    Size -= VTSize;
  }

  return true;
}

void SelectionDAGLowering::visitMemIntrinsic(CallInst &I, unsigned Op) {
  SDOperand Op1 = getValue(I.getOperand(1));
  SDOperand Op2 = getValue(I.getOperand(2));
  SDOperand Op3 = getValue(I.getOperand(3));
  SDOperand Op4 = getValue(I.getOperand(4));
  unsigned Align = (unsigned)cast<ConstantSDNode>(Op4)->getValue();
  if (Align == 0) Align = 1;

  if (ConstantSDNode *Size = dyn_cast<ConstantSDNode>(Op3)) {
    std::vector<MVT::ValueType> MemOps;

    // Expand memset / memcpy to a series of load / store ops
    // if the size operand falls below a certain threshold.
    std::vector<SDOperand> OutChains;
    switch (Op) {
    default: break;  // Do nothing for now.
    case ISD::MEMSET: {
      if (MeetsMaxMemopRequirement(MemOps, TLI.getMaxStoresPerMemset(),
                                   Size->getValue(), Align, TLI)) {
        unsigned NumMemOps = MemOps.size();
        unsigned Offset = 0;
        for (unsigned i = 0; i < NumMemOps; i++) {
          MVT::ValueType VT = MemOps[i];
          unsigned VTSize = getSizeInBits(VT) / 8;
          SDOperand Value = getMemsetValue(Op2, VT, DAG, TLI);
          SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, getRoot(),
                                        Value,
                                        getMemBasePlusOffset(Op1, Offset, DAG, TLI),
                                        DAG.getSrcValue(I.getOperand(1), Offset));
          OutChains.push_back(Store);
          Offset += VTSize;
        }
      }
      break;
    }
    case ISD::MEMCPY: {
      if (MeetsMaxMemopRequirement(MemOps, TLI.getMaxStoresPerMemcpy(),
                                   Size->getValue(), Align, TLI)) {
        unsigned NumMemOps = MemOps.size();
        unsigned SrcOff = 0, DstOff = 0;
        GlobalAddressSDNode *G = NULL;
        std::string Str;

        if (Op2.getOpcode() == ISD::GlobalAddress)
          G = cast<GlobalAddressSDNode>(Op2);
        else if (Op2.getOpcode() == ISD::ADD &&
                 Op2.getOperand(0).getOpcode() == ISD::GlobalAddress &&
                 Op2.getOperand(1).getOpcode() == ISD::Constant) {
          G = cast<GlobalAddressSDNode>(Op2.getOperand(0));
          SrcOff += cast<ConstantSDNode>(Op2.getOperand(1))->getValue();
        }
        if (G) {
          GlobalVariable *GV = dyn_cast<GlobalVariable>(G->getGlobal());
          if (GV)
            Str = getStringValue(GV);
        }

        for (unsigned i = 0; i < NumMemOps; i++) {
          MVT::ValueType VT = MemOps[i];
          unsigned VTSize = getSizeInBits(VT) / 8;
          SDOperand Value, Chain, Store;

          if (!Str.empty()) {
            Value = getMemsetStringVal(VT, DAG, TLI, Str, SrcOff);
            Chain = getRoot();
            Store =
              DAG.getNode(ISD::STORE, MVT::Other, Chain, Value,
                          getMemBasePlusOffset(Op1, DstOff, DAG, TLI),
                          DAG.getSrcValue(I.getOperand(1), DstOff));
          } else {
            Value = DAG.getLoad(VT, getRoot(),
                        getMemBasePlusOffset(Op2, SrcOff, DAG, TLI),
                        DAG.getSrcValue(I.getOperand(2), SrcOff));
            Chain = Value.getValue(1);
            Store =
              DAG.getNode(ISD::STORE, MVT::Other, Chain, Value,
                          getMemBasePlusOffset(Op1, DstOff, DAG, TLI),
                          DAG.getSrcValue(I.getOperand(1), DstOff));
          }
          OutChains.push_back(Store);
          SrcOff += VTSize;
          DstOff += VTSize;
        }
      }
      break;
    }
    }

    if (!OutChains.empty()) {
      DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, OutChains));
      return;
    }
  }

  std::vector<SDOperand> Ops;
  Ops.push_back(getRoot());
  Ops.push_back(Op1);
  Ops.push_back(Op2);
  Ops.push_back(Op3);
  Ops.push_back(Op4);
  DAG.setRoot(DAG.getNode(Op, MVT::Other, Ops));
}

//===----------------------------------------------------------------------===//
// SelectionDAGISel code
//===----------------------------------------------------------------------===//

unsigned SelectionDAGISel::MakeReg(MVT::ValueType VT) {
  return RegMap->createVirtualRegister(TLI.getRegClassFor(VT));
}

void SelectionDAGISel::getAnalysisUsage(AnalysisUsage &AU) const {
  // FIXME: we only modify the CFG to split critical edges.  This
  // updates dom and loop info.
}


/// InsertGEPComputeCode - Insert code into BB to compute Ptr+PtrOffset,
/// casting to the type of GEPI.
static Value *InsertGEPComputeCode(Value *&V, BasicBlock *BB, Instruction *GEPI,
                                   Value *Ptr, Value *PtrOffset) {
  if (V) return V;   // Already computed.
  
  BasicBlock::iterator InsertPt;
  if (BB == GEPI->getParent()) {
    // If insert into the GEP's block, insert right after the GEP.
    InsertPt = GEPI;
    ++InsertPt;
  } else {
    // Otherwise, insert at the top of BB, after any PHI nodes
    InsertPt = BB->begin();
    while (isa<PHINode>(InsertPt)) ++InsertPt;
  }
  
  // If Ptr is itself a cast, but in some other BB, emit a copy of the cast into
  // BB so that there is only one value live across basic blocks (the cast 
  // operand).
  if (CastInst *CI = dyn_cast<CastInst>(Ptr))
    if (CI->getParent() != BB && isa<PointerType>(CI->getOperand(0)->getType()))
      Ptr = new CastInst(CI->getOperand(0), CI->getType(), "", InsertPt);
  
  // Add the offset, cast it to the right type.
  Ptr = BinaryOperator::createAdd(Ptr, PtrOffset, "", InsertPt);
  Ptr = new CastInst(Ptr, GEPI->getType(), "", InsertPt);
  return V = Ptr;
}


/// OptimizeGEPExpression - Since we are doing basic-block-at-a-time instruction
/// selection, we want to be a bit careful about some things.  In particular, if
/// we have a GEP instruction that is used in a different block than it is
/// defined, the addressing expression of the GEP cannot be folded into loads or
/// stores that use it.  In this case, decompose the GEP and move constant
/// indices into blocks that use it.
static void OptimizeGEPExpression(GetElementPtrInst *GEPI,
                                  const TargetData &TD) {
  // If this GEP is only used inside the block it is defined in, there is no
  // need to rewrite it.
  bool isUsedOutsideDefBB = false;
  BasicBlock *DefBB = GEPI->getParent();
  for (Value::use_iterator UI = GEPI->use_begin(), E = GEPI->use_end(); 
       UI != E; ++UI) {
    if (cast<Instruction>(*UI)->getParent() != DefBB) {
      isUsedOutsideDefBB = true;
      break;
    }
  }
  if (!isUsedOutsideDefBB) return;

  // If this GEP has no non-zero constant indices, there is nothing we can do,
  // ignore it.
  bool hasConstantIndex = false;
  for (GetElementPtrInst::op_iterator OI = GEPI->op_begin()+1,
       E = GEPI->op_end(); OI != E; ++OI) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(*OI))
      if (CI->getRawValue()) {
        hasConstantIndex = true;
        break;
      }
  }
  // If this is a GEP &Alloca, 0, 0, forward subst the frame index into uses.
  if (!hasConstantIndex && !isa<AllocaInst>(GEPI->getOperand(0))) return;
  
  // Otherwise, decompose the GEP instruction into multiplies and adds.  Sum the
  // constant offset (which we now know is non-zero) and deal with it later.
  uint64_t ConstantOffset = 0;
  const Type *UIntPtrTy = TD.getIntPtrType();
  Value *Ptr = new CastInst(GEPI->getOperand(0), UIntPtrTy, "", GEPI);
  const Type *Ty = GEPI->getOperand(0)->getType();

  for (GetElementPtrInst::op_iterator OI = GEPI->op_begin()+1,
       E = GEPI->op_end(); OI != E; ++OI) {
    Value *Idx = *OI;
    if (const StructType *StTy = dyn_cast<StructType>(Ty)) {
      unsigned Field = cast<ConstantUInt>(Idx)->getValue();
      if (Field)
        ConstantOffset += TD.getStructLayout(StTy)->MemberOffsets[Field];
      Ty = StTy->getElementType(Field);
    } else {
      Ty = cast<SequentialType>(Ty)->getElementType();

      // Handle constant subscripts.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Idx)) {
        if (CI->getRawValue() == 0) continue;
        
        if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(CI))
          ConstantOffset += (int64_t)TD.getTypeSize(Ty)*CSI->getValue();
        else
          ConstantOffset+=TD.getTypeSize(Ty)*cast<ConstantUInt>(CI)->getValue();
        continue;
      }
      
      // Ptr = Ptr + Idx * ElementSize;
      
      // Cast Idx to UIntPtrTy if needed.
      Idx = new CastInst(Idx, UIntPtrTy, "", GEPI);
      
      uint64_t ElementSize = TD.getTypeSize(Ty);
      // Mask off bits that should not be set.
      ElementSize &= ~0ULL >> (64-UIntPtrTy->getPrimitiveSizeInBits());
      Constant *SizeCst = ConstantUInt::get(UIntPtrTy, ElementSize);

      // Multiply by the element size and add to the base.
      Idx = BinaryOperator::createMul(Idx, SizeCst, "", GEPI);
      Ptr = BinaryOperator::createAdd(Ptr, Idx, "", GEPI);
    }
  }
  
  // Make sure that the offset fits in uintptr_t.
  ConstantOffset &= ~0ULL >> (64-UIntPtrTy->getPrimitiveSizeInBits());
  Constant *PtrOffset = ConstantUInt::get(UIntPtrTy, ConstantOffset);
  
  // Okay, we have now emitted all of the variable index parts to the BB that
  // the GEP is defined in.  Loop over all of the using instructions, inserting
  // an "add Ptr, ConstantOffset" into each block that uses it and update the
  // instruction to use the newly computed value, making GEPI dead.  When the
  // user is a load or store instruction address, we emit the add into the user
  // block, otherwise we use a canonical version right next to the gep (these 
  // won't be foldable as addresses, so we might as well share the computation).
  
  std::map<BasicBlock*,Value*> InsertedExprs;
  while (!GEPI->use_empty()) {
    Instruction *User = cast<Instruction>(GEPI->use_back());

    // If this use is not foldable into the addressing mode, use a version 
    // emitted in the GEP block.
    Value *NewVal;
    if (!isa<LoadInst>(User) &&
        (!isa<StoreInst>(User) || User->getOperand(0) == GEPI)) {
      NewVal = InsertGEPComputeCode(InsertedExprs[DefBB], DefBB, GEPI, 
                                    Ptr, PtrOffset);
    } else {
      // Otherwise, insert the code in the User's block so it can be folded into
      // any users in that block.
      NewVal = InsertGEPComputeCode(InsertedExprs[User->getParent()], 
                                    User->getParent(), GEPI, 
                                    Ptr, PtrOffset);
    }
    User->replaceUsesOfWith(GEPI, NewVal);
  }
  
  // Finally, the GEP is dead, remove it.
  GEPI->eraseFromParent();
}

bool SelectionDAGISel::runOnFunction(Function &Fn) {
  MachineFunction &MF = MachineFunction::construct(&Fn, TLI.getTargetMachine());
  RegMap = MF.getSSARegMap();
  DEBUG(std::cerr << "\n\n\n=== " << Fn.getName() << "\n");

  // First, split all critical edges for PHI nodes with incoming values that are
  // constants, this way the load of the constant into a vreg will not be placed
  // into MBBs that are used some other way.
  //
  // In this pass we also look for GEP instructions that are used across basic
  // blocks and rewrites them to improve basic-block-at-a-time selection.
  // 
  for (Function::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    PHINode *PN;
    BasicBlock::iterator BBI;
    for (BBI = BB->begin(); (PN = dyn_cast<PHINode>(BBI)); ++BBI)
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (isa<Constant>(PN->getIncomingValue(i)))
          SplitCriticalEdge(PN->getIncomingBlock(i), BB);
    
    for (BasicBlock::iterator E = BB->end(); BBI != E; )
      if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(BBI++))
        OptimizeGEPExpression(GEPI, TLI.getTargetData());
  }
  
  FunctionLoweringInfo FuncInfo(TLI, Fn, MF);

  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    SelectBasicBlock(I, MF, FuncInfo);

  return true;
}


SDOperand SelectionDAGISel::
CopyValueToVirtualRegister(SelectionDAGLowering &SDL, Value *V, unsigned Reg) {
  SDOperand Op = SDL.getValue(V);
  assert((Op.getOpcode() != ISD::CopyFromReg ||
          cast<RegisterSDNode>(Op.getOperand(1))->getReg() != Reg) &&
         "Copy from a reg to the same reg!");
  
  // If this type is not legal, we must make sure to not create an invalid
  // register use.
  MVT::ValueType SrcVT = Op.getValueType();
  MVT::ValueType DestVT = TLI.getTypeToTransformTo(SrcVT);
  SelectionDAG &DAG = SDL.DAG;
  if (SrcVT == DestVT) {
    return DAG.getCopyToReg(SDL.getRoot(), Reg, Op);
  } else if (SrcVT < DestVT) {
    // The src value is promoted to the register.
    if (MVT::isFloatingPoint(SrcVT))
      Op = DAG.getNode(ISD::FP_EXTEND, DestVT, Op);
    else
      Op = DAG.getNode(ISD::ANY_EXTEND, DestVT, Op);
    return DAG.getCopyToReg(SDL.getRoot(), Reg, Op);
  } else  {
    // The src value is expanded into multiple registers.
    SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DestVT,
                               Op, DAG.getConstant(0, MVT::i32));
    SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DestVT,
                               Op, DAG.getConstant(1, MVT::i32));
    Op = DAG.getCopyToReg(SDL.getRoot(), Reg, Lo);
    return DAG.getCopyToReg(Op, Reg+1, Hi);
  }
}

void SelectionDAGISel::
LowerArguments(BasicBlock *BB, SelectionDAGLowering &SDL,
               std::vector<SDOperand> &UnorderedChains) {
  // If this is the entry block, emit arguments.
  Function &F = *BB->getParent();
  FunctionLoweringInfo &FuncInfo = SDL.FuncInfo;
  SDOperand OldRoot = SDL.DAG.getRoot();
  std::vector<SDOperand> Args = TLI.LowerArguments(F, SDL.DAG);

  unsigned a = 0;
  for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end();
       AI != E; ++AI, ++a)
    if (!AI->use_empty()) {
      SDL.setValue(AI, Args[a]);
      
      // If this argument is live outside of the entry block, insert a copy from
      // whereever we got it to the vreg that other BB's will reference it as.
      if (FuncInfo.ValueMap.count(AI)) {
        SDOperand Copy =
          CopyValueToVirtualRegister(SDL, AI, FuncInfo.ValueMap[AI]);
        UnorderedChains.push_back(Copy);
      }
    }

  // Next, if the function has live ins that need to be copied into vregs,
  // emit the copies now, into the top of the block.
  MachineFunction &MF = SDL.DAG.getMachineFunction();
  if (MF.livein_begin() != MF.livein_end()) {
    SSARegMap *RegMap = MF.getSSARegMap();
    const MRegisterInfo &MRI = *MF.getTarget().getRegisterInfo();
    for (MachineFunction::livein_iterator LI = MF.livein_begin(),
         E = MF.livein_end(); LI != E; ++LI)
      if (LI->second)
        MRI.copyRegToReg(*MF.begin(), MF.begin()->end(), LI->second,
                         LI->first, RegMap->getRegClass(LI->second));
  }
    
  // Finally, if the target has anything special to do, allow it to do so.
  EmitFunctionEntryCode(F, SDL.DAG.getMachineFunction());
}


void SelectionDAGISel::BuildSelectionDAG(SelectionDAG &DAG, BasicBlock *LLVMBB,
       std::vector<std::pair<MachineInstr*, unsigned> > &PHINodesToUpdate,
                                    FunctionLoweringInfo &FuncInfo) {
  SelectionDAGLowering SDL(DAG, TLI, FuncInfo);

  std::vector<SDOperand> UnorderedChains;

  // Lower any arguments needed in this block if this is the entry block.
  if (LLVMBB == &LLVMBB->getParent()->front())
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
    SDOperand Root = SDL.getRoot();
    if (Root.getOpcode() != ISD::EntryToken) {
      unsigned i = 0, e = UnorderedChains.size();
      for (; i != e; ++i) {
        assert(UnorderedChains[i].Val->getNumOperands() > 1);
        if (UnorderedChains[i].Val->getOperand(0) == Root)
          break;  // Don't add the root if we already indirectly depend on it.
      }
        
      if (i == e)
        UnorderedChains.push_back(Root);
    }
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, UnorderedChains));
  }

  // Lower the terminator after the copies are emitted.
  SDL.visit(*LLVMBB->getTerminator());

  // Make sure the root of the DAG is up-to-date.
  DAG.setRoot(SDL.getRoot());
}

void SelectionDAGISel::SelectBasicBlock(BasicBlock *LLVMBB, MachineFunction &MF,
                                        FunctionLoweringInfo &FuncInfo) {
  SelectionDAG DAG(TLI, MF, getAnalysisToUpdate<MachineDebugInfo>());
  CurDAG = &DAG;
  std::vector<std::pair<MachineInstr*, unsigned> > PHINodesToUpdate;

  // First step, lower LLVM code to some DAG.  This DAG may use operations and
  // types that are not supported by the target.
  BuildSelectionDAG(DAG, LLVMBB, PHINodesToUpdate, FuncInfo);

  // Run the DAG combiner in pre-legalize mode.
  DAG.Combine(false);
  
  DEBUG(std::cerr << "Lowered selection DAG:\n");
  DEBUG(DAG.dump());

  // Second step, hack on the DAG until it only uses operations and types that
  // the target supports.
  DAG.Legalize();

  DEBUG(std::cerr << "Legalized selection DAG:\n");
  DEBUG(DAG.dump());

  // Run the DAG combiner in post-legalize mode.
  DAG.Combine(true);
  
  if (ViewISelDAGs) DAG.viewGraph();
  
  // Third, instruction select all of the operations to machine code, adding the
  // code to the MachineBasicBlock.
  InstructionSelectBasicBlock(DAG);

  DEBUG(std::cerr << "Selected machine code:\n");
  DEBUG(BB->dump());

  // Next, now that we know what the last MBB the LLVM BB expanded is, update
  // PHI nodes in successors.
  for (unsigned i = 0, e = PHINodesToUpdate.size(); i != e; ++i) {
    MachineInstr *PHI = PHINodesToUpdate[i].first;
    assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
           "This is not a machine PHI node that we are updating!");
    PHI->addRegOperand(PHINodesToUpdate[i].second);
    PHI->addMachineBasicBlockOperand(BB);
  }

  // Finally, add the CFG edges from the last selected MBB to the successor
  // MBBs.
  TerminatorInst *TI = LLVMBB->getTerminator();
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
    MachineBasicBlock *Succ0MBB = FuncInfo.MBBMap[TI->getSuccessor(i)];
    BB->addSuccessor(Succ0MBB);
  }
}

//===----------------------------------------------------------------------===//
/// ScheduleAndEmitDAG - Pick a safe ordering and emit instructions for each
/// target node in the graph.
void SelectionDAGISel::ScheduleAndEmitDAG(SelectionDAG &DAG) {
  if (ViewSchedDAGs) DAG.viewGraph();
  ScheduleDAG *SL = NULL;

  switch (ISHeuristic) {
  default: assert(0 && "Unrecognized scheduling heuristic");
  case defaultScheduling:
    if (TLI.getSchedulingPreference() == TargetLowering::SchedulingForLatency)
      SL = createSimpleDAGScheduler(noScheduling, DAG, BB);
    else /* TargetLowering::SchedulingForRegPressure */
      SL = createBURRListDAGScheduler(DAG, BB);
    break;
  case noScheduling:
  case simpleScheduling:
  case simpleNoItinScheduling:
    SL = createSimpleDAGScheduler(ISHeuristic, DAG, BB);
    break;
  case listSchedulingBURR:
    SL = createBURRListDAGScheduler(DAG, BB);
  }
  BB = SL->Run();
  delete SL;
}
