//===- CostModel.cpp ------ Cost Model Analysis ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the cost model analysis. It provides a very basic cost
// estimation for LLVM-IR. This analysis uses the services of the codegen
// to approximate the cost of any IR instruction when lowered to machine
// instructions. The cost results are unit-less and the cost number represents
// the throughput of the machine assuming that all loads hit the cache, all
// branches are predicted, etc. The cost numbers can be added in order to
// compare two or more transformation alternatives.
//
//===----------------------------------------------------------------------===//

#define CM_NAME "cost-model"
#define DEBUG_TYPE CM_NAME
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  class CostModelAnalysis : public FunctionPass {

  public:
    static char ID; // Class identification, replacement for typeinfo
    CostModelAnalysis() : FunctionPass(ID), F(0), TTI(0) {
      initializeCostModelAnalysisPass(
        *PassRegistry::getPassRegistry());
    }

    /// Returns the expected cost of the instruction.
    /// Returns -1 if the cost is unknown.
    /// Note, this method does not cache the cost calculation and it
    /// can be expensive in some cases.
    unsigned getInstructionCost(const Instruction *I) const;

  private:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool runOnFunction(Function &F);
    virtual void print(raw_ostream &OS, const Module*) const;

    /// The function that we analyze.
    Function *F;
    /// Target information.
    const TargetTransformInfo *TTI;
  };
}  // End of anonymous namespace

// Register this pass.
char CostModelAnalysis::ID = 0;
static const char cm_name[] = "Cost Model Analysis";
INITIALIZE_PASS_BEGIN(CostModelAnalysis, CM_NAME, cm_name, false, true)
INITIALIZE_PASS_END  (CostModelAnalysis, CM_NAME, cm_name, false, true)

FunctionPass *llvm::createCostModelAnalysisPass() {
  return new CostModelAnalysis();
}

void
CostModelAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool
CostModelAnalysis::runOnFunction(Function &F) {
 this->F = &F;
 TTI = getAnalysisIfAvailable<TargetTransformInfo>();

 return false;
}

static bool isReverseVectorMask(SmallVector<int, 16> &Mask) {
  for (unsigned i = 0, MaskSize = Mask.size(); i < MaskSize; ++i)
    if (Mask[i] > 0 && Mask[i] != (int)(MaskSize - 1 - i))
      return false;
  return true;
}

static TargetTransformInfo::OperandValueKind getOperandInfo(Value *V) {
  TargetTransformInfo::OperandValueKind OpInfo =
    TargetTransformInfo::OK_AnyValue;

  // Check for a splat of a constant.
  ConstantDataVector *CDV = 0;
  if ((CDV = dyn_cast<ConstantDataVector>(V)))
    if (CDV->getSplatValue() != NULL)
      OpInfo = TargetTransformInfo::OK_UniformConstantValue;
  ConstantVector *CV = 0;
  if ((CV = dyn_cast<ConstantVector>(V)))
    if (CV->getSplatValue() != NULL)
      OpInfo = TargetTransformInfo::OK_UniformConstantValue;

  return OpInfo;
}

unsigned CostModelAnalysis::getInstructionCost(const Instruction *I) const {
  if (!TTI)
    return -1;

  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:{
    Type *ValTy = I->getOperand(0)->getType()->getPointerElementType();
    return TTI->getAddressComputationCost(ValTy);
  }

  case Instruction::Ret:
  case Instruction::PHI:
  case Instruction::Br: {
    return TTI->getCFInstrCost(I->getOpcode());
  }
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    TargetTransformInfo::OperandValueKind Op1VK =
      getOperandInfo(I->getOperand(0));
    TargetTransformInfo::OperandValueKind Op2VK =
      getOperandInfo(I->getOperand(1));
    return TTI->getArithmeticInstrCost(I->getOpcode(), I->getType(), Op1VK,
                                       Op2VK);
  }
  case Instruction::Select: {
    const SelectInst *SI = cast<SelectInst>(I);
    Type *CondTy = SI->getCondition()->getType();
    return TTI->getCmpSelInstrCost(I->getOpcode(), I->getType(), CondTy);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ValTy = I->getOperand(0)->getType();
    return TTI->getCmpSelInstrCost(I->getOpcode(), ValTy);
  }
  case Instruction::Store: {
    const StoreInst *SI = cast<StoreInst>(I);
    Type *ValTy = SI->getValueOperand()->getType();
    return TTI->getMemoryOpCost(I->getOpcode(), ValTy,
                                 SI->getAlignment(),
                                 SI->getPointerAddressSpace());
  }
  case Instruction::Load: {
    const LoadInst *LI = cast<LoadInst>(I);
    return TTI->getMemoryOpCost(I->getOpcode(), I->getType(),
                                 LI->getAlignment(),
                                 LI->getPointerAddressSpace());
  }
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::FPExt:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::BitCast: {
    Type *SrcTy = I->getOperand(0)->getType();
    return TTI->getCastInstrCost(I->getOpcode(), I->getType(), SrcTy);
  }
  case Instruction::ExtractElement: {
    const ExtractElementInst * EEI = cast<ExtractElementInst>(I);
    ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1));
    unsigned Idx = -1;
    if (CI)
      Idx = CI->getZExtValue();
    return TTI->getVectorInstrCost(I->getOpcode(),
                                   EEI->getOperand(0)->getType(), Idx);
  }
  case Instruction::InsertElement: {
      const InsertElementInst * IE = cast<InsertElementInst>(I);
      ConstantInt *CI = dyn_cast<ConstantInt>(IE->getOperand(2));
      unsigned Idx = -1;
      if (CI)
        Idx = CI->getZExtValue();
      return TTI->getVectorInstrCost(I->getOpcode(),
                                     IE->getType(), Idx);
    }
  case Instruction::ShuffleVector: {
    const ShuffleVectorInst *Shuffle = cast<ShuffleVectorInst>(I);
    Type *VecTypOp0 = Shuffle->getOperand(0)->getType();
    unsigned NumVecElems = VecTypOp0->getVectorNumElements();
    SmallVector<int, 16> Mask = Shuffle->getShuffleMask();

    if (NumVecElems == Mask.size() && isReverseVectorMask(Mask))
      return TTI->getShuffleCost(TargetTransformInfo::SK_Reverse, VecTypOp0, 0,
                                 0);
    return -1;
  }
  case Instruction::Call:
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      SmallVector<Type*, 4> Tys;
      for (unsigned J = 0, JE = II->getNumArgOperands(); J != JE; ++J)
        Tys.push_back(II->getArgOperand(J)->getType());

      return TTI->getIntrinsicInstrCost(II->getIntrinsicID(), II->getType(),
                                        Tys);
    }
    return -1;
  default:
    // We don't have any information on this instruction.
    return -1;
  }
}

void CostModelAnalysis::print(raw_ostream &OS, const Module*) const {
  if (!F)
    return;

  for (Function::iterator B = F->begin(), BE = F->end(); B != BE; ++B) {
    for (BasicBlock::iterator it = B->begin(), e = B->end(); it != e; ++it) {
      Instruction *Inst = it;
      unsigned Cost = getInstructionCost(Inst);
      if (Cost != (unsigned)-1)
        OS << "Cost Model: Found an estimated cost of " << Cost;
      else
        OS << "Cost Model: Unknown cost";

      OS << " for instruction: "<< *Inst << "\n";
    }
  }
}
