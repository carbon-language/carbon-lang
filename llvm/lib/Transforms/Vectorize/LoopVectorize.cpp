//===- LoopVectorize.cpp - A Loop Vectorizer ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "LoopVectorize.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Constants.h"
#include "llvm/DataLayout.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetTransformInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Type.h"
#include "llvm/Value.h"

static cl::opt<unsigned>
VectorizationFactor("force-vector-width", cl::init(0), cl::Hidden,
                    cl::desc("Sets the SIMD width. Zero is autoselect."));

static cl::opt<bool>
EnableIfConversion("enable-if-conversion", cl::init(false), cl::Hidden,
                   cl::desc("Enable if-conversion during vectorization."));

namespace {

/// The LoopVectorize Pass.
struct LoopVectorize : public LoopPass {
  /// Pass identification, replacement for typeid
  static char ID;

  explicit LoopVectorize() : LoopPass(ID) {
    initializeLoopVectorizePass(*PassRegistry::getPassRegistry());
  }

  ScalarEvolution *SE;
  DataLayout *DL;
  LoopInfo *LI;
  TargetTransformInfo *TTI;
  DominatorTree *DT;

  virtual bool runOnLoop(Loop *L, LPPassManager &LPM) {
    // We only vectorize innermost loops.
    if (!L->empty())
      return false;

    SE = &getAnalysis<ScalarEvolution>();
    DL = getAnalysisIfAvailable<DataLayout>();
    LI = &getAnalysis<LoopInfo>();
    TTI = getAnalysisIfAvailable<TargetTransformInfo>();
    DT = &getAnalysis<DominatorTree>();

    DEBUG(dbgs() << "LV: Checking a loop in \"" <<
          L->getHeader()->getParent()->getName() << "\"\n");

    // Check if it is legal to vectorize the loop.
    LoopVectorizationLegality LVL(L, SE, DL, DT);
    if (!LVL.canVectorize()) {
      DEBUG(dbgs() << "LV: Not vectorizing.\n");
      return false;
    }

    // Select the preffered vectorization factor.
    const VectorTargetTransformInfo *VTTI = 0;
    if (TTI)
      VTTI = TTI->getVectorTargetTransformInfo();
    // Use the cost model.
    LoopVectorizationCostModel CM(L, SE, &LVL, VTTI);

    // Check the function attribues to find out if this function should be
    // optimized for size.
    Function *F = L->getHeader()->getParent();
    Attribute::AttrVal SzAttr= Attribute::OptimizeForSize;
    bool OptForSize = F->getFnAttributes().hasAttribute(SzAttr);

    unsigned VF = CM.selectVectorizationFactor(OptForSize, VectorizationFactor);

    if (VF == 1) {
      DEBUG(dbgs() << "LV: Vectorization is possible but not beneficial.\n");
      return false;
    }

    DEBUG(dbgs() << "LV: Found a vectorizable loop ("<< VF << ") in "<<
          F->getParent()->getModuleIdentifier()<<"\n");

    // If we decided that it is *legal* to vectorizer the loop then do it.
    InnerLoopVectorizer LB(L, SE, LI, DT, DL, VF);
    LB.vectorize(&LVL);

    DEBUG(verifyFunction(*L->getHeader()->getParent()));
    return true;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    LoopPass::getAnalysisUsage(AU);
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addRequired<LoopInfo>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<DominatorTree>();
    AU.addPreserved<LoopInfo>();
    AU.addPreserved<DominatorTree>();
  }

};

}// namespace

//===----------------------------------------------------------------------===//
// Implementation of LoopVectorizationLegality, InnerLoopVectorizer and
// LoopVectorizationCostModel.
//===----------------------------------------------------------------------===//

void
LoopVectorizationLegality::RuntimePointerCheck::insert(ScalarEvolution *SE,
                                                       Loop *Lp, Value *Ptr) {
  const SCEV *Sc = SE->getSCEV(Ptr);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Sc);
  assert(AR && "Invalid addrec expression");
  const SCEV *Ex = SE->getExitCount(Lp, Lp->getLoopLatch());
  const SCEV *ScEnd = AR->evaluateAtIteration(Ex, *SE);
  Pointers.push_back(Ptr);
  Starts.push_back(AR->getStart());
  Ends.push_back(ScEnd);
}

Value *InnerLoopVectorizer::getBroadcastInstrs(Value *V) {
  // Create the types.
  LLVMContext &C = V->getContext();
  Type *VTy = VectorType::get(V->getType(), VF);
  Type *I32 = IntegerType::getInt32Ty(C);

  // Save the current insertion location.
  Instruction *Loc = Builder.GetInsertPoint();

  // We need to place the broadcast of invariant variables outside the loop.
  Instruction *Instr = dyn_cast<Instruction>(V);
  bool NewInstr = (Instr && Instr->getParent() == LoopVectorBody);
  bool Invariant = OrigLoop->isLoopInvariant(V) && !NewInstr;

  // Place the code for broadcasting invariant variables in the new preheader.
  if (Invariant)
    Builder.SetInsertPoint(LoopVectorPreHeader->getTerminator());

  Constant *Zero = ConstantInt::get(I32, 0);
  Value *Zeros = ConstantAggregateZero::get(VectorType::get(I32, VF));
  Value *UndefVal = UndefValue::get(VTy);
  // Insert the value into a new vector.
  Value *SingleElem = Builder.CreateInsertElement(UndefVal, V, Zero);
  // Broadcast the scalar into all locations in the vector.
  Value *Shuf = Builder.CreateShuffleVector(SingleElem, UndefVal, Zeros,
                                            "broadcast");

  // Restore the builder insertion point.
  if (Invariant)
    Builder.SetInsertPoint(Loc);

  return Shuf;
}

Value *InnerLoopVectorizer::getConsecutiveVector(Value* Val, bool Negate) {
  assert(Val->getType()->isVectorTy() && "Must be a vector");
  assert(Val->getType()->getScalarType()->isIntegerTy() &&
         "Elem must be an integer");
  // Create the types.
  Type *ITy = Val->getType()->getScalarType();
  VectorType *Ty = cast<VectorType>(Val->getType());
  int VLen = Ty->getNumElements();
  SmallVector<Constant*, 8> Indices;

  // Create a vector of consecutive numbers from zero to VF.
  for (int i = 0; i < VLen; ++i)
    Indices.push_back(ConstantInt::get(ITy, Negate ? (-i): i ));

  // Add the consecutive indices to the vector value.
  Constant *Cv = ConstantVector::get(Indices);
  assert(Cv->getType() == Val->getType() && "Invalid consecutive vec");
  return Builder.CreateAdd(Val, Cv, "induction");
}

bool LoopVectorizationLegality::isConsecutivePtr(Value *Ptr) {
  assert(Ptr->getType()->isPointerTy() && "Unexpected non ptr");

  // If this value is a pointer induction variable we know it is consecutive.
  PHINode *Phi = dyn_cast_or_null<PHINode>(Ptr);
  if (Phi && Inductions.count(Phi)) {
    InductionInfo II = Inductions[Phi];
    if (PtrInduction == II.IK)
      return true;
  }

  GetElementPtrInst *Gep = dyn_cast_or_null<GetElementPtrInst>(Ptr);
  if (!Gep)
    return false;

  unsigned NumOperands = Gep->getNumOperands();
  Value *LastIndex = Gep->getOperand(NumOperands - 1);

  // Check that all of the gep indices are uniform except for the last.
  for (unsigned i = 0; i < NumOperands - 1; ++i)
    if (!SE->isLoopInvariant(SE->getSCEV(Gep->getOperand(i)), TheLoop))
      return false;

  // We can emit wide load/stores only if the last index is the induction
  // variable.
  const SCEV *Last = SE->getSCEV(LastIndex);
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Last)) {
    const SCEV *Step = AR->getStepRecurrence(*SE);

    // The memory is consecutive because the last index is consecutive
    // and all other indices are loop invariant.
    if (Step->isOne())
      return true;
  }

  return false;
}

bool LoopVectorizationLegality::isUniform(Value *V) {
  return (SE->isLoopInvariant(SE->getSCEV(V), TheLoop));
}

Value *InnerLoopVectorizer::getVectorValue(Value *V) {
  assert(V != Induction && "The new induction variable should not be used.");
  assert(!V->getType()->isVectorTy() && "Can't widen a vector");
  // If we saved a vectorized copy of V, use it.
  Value *&MapEntry = WidenMap[V];
  if (MapEntry)
    return MapEntry;

  // Broadcast V and save the value for future uses.
  Value *B = getBroadcastInstrs(V);
  MapEntry = B;
  return B;
}

Constant*
InnerLoopVectorizer::getUniformVector(unsigned Val, Type* ScalarTy) {
  return ConstantVector::getSplat(VF, ConstantInt::get(ScalarTy, Val, true));
}

void InnerLoopVectorizer::scalarizeInstruction(Instruction *Instr) {
  assert(!Instr->getType()->isAggregateType() && "Can't handle vectors");
  // Holds vector parameters or scalars, in case of uniform vals.
  SmallVector<Value*, 8> Params;

  // Find all of the vectorized parameters.
  for (unsigned op = 0, e = Instr->getNumOperands(); op != e; ++op) {
    Value *SrcOp = Instr->getOperand(op);

    // If we are accessing the old induction variable, use the new one.
    if (SrcOp == OldInduction) {
      Params.push_back(getVectorValue(SrcOp));
      continue;
    }

    // Try using previously calculated values.
    Instruction *SrcInst = dyn_cast<Instruction>(SrcOp);

    // If the src is an instruction that appeared earlier in the basic block
    // then it should already be vectorized.
    if (SrcInst && SrcInst->getParent() == Instr->getParent()) {
      assert(WidenMap.count(SrcInst) && "Source operand is unavailable");
      // The parameter is a vector value from earlier.
      Params.push_back(WidenMap[SrcInst]);
    } else {
      // The parameter is a scalar from outside the loop. Maybe even a constant.
      Params.push_back(SrcOp);
    }
  }

  assert(Params.size() == Instr->getNumOperands() &&
         "Invalid number of operands");

  // Does this instruction return a value ?
  bool IsVoidRetTy = Instr->getType()->isVoidTy();
  Value *VecResults = 0;

  // If we have a return value, create an empty vector. We place the scalarized
  // instructions in this vector.
  if (!IsVoidRetTy)
    VecResults = UndefValue::get(VectorType::get(Instr->getType(), VF));

  // For each scalar that we create:
  for (unsigned i = 0; i < VF; ++i) {
    Instruction *Cloned = Instr->clone();
    if (!IsVoidRetTy)
      Cloned->setName(Instr->getName() + ".cloned");
    // Replace the operands of the cloned instrucions with extracted scalars.
    for (unsigned op = 0, e = Instr->getNumOperands(); op != e; ++op) {
      Value *Op = Params[op];
      // Param is a vector. Need to extract the right lane.
      if (Op->getType()->isVectorTy())
        Op = Builder.CreateExtractElement(Op, Builder.getInt32(i));
      Cloned->setOperand(op, Op);
    }

    // Place the cloned scalar in the new loop.
    Builder.Insert(Cloned);

    // If the original scalar returns a value we need to place it in a vector
    // so that future users will be able to use it.
    if (!IsVoidRetTy)
      VecResults = Builder.CreateInsertElement(VecResults, Cloned,
                                               Builder.getInt32(i));
  }

  if (!IsVoidRetTy)
    WidenMap[Instr] = VecResults;
}

Value*
InnerLoopVectorizer::addRuntimeCheck(LoopVectorizationLegality *Legal,
                                     Instruction *Loc) {
  LoopVectorizationLegality::RuntimePointerCheck *PtrRtCheck =
  Legal->getRuntimePointerCheck();

  if (!PtrRtCheck->Need)
    return NULL;

  Value *MemoryRuntimeCheck = 0;
  unsigned NumPointers = PtrRtCheck->Pointers.size();
  SmallVector<Value* , 2> Starts;
  SmallVector<Value* , 2> Ends;

  SCEVExpander Exp(*SE, "induction");

  // Use this type for pointer arithmetic.
  Type* PtrArithTy = Type::getInt8PtrTy(Loc->getContext(), 0);

  for (unsigned i = 0; i < NumPointers; ++i) {
    Value *Ptr = PtrRtCheck->Pointers[i];
    const SCEV *Sc = SE->getSCEV(Ptr);

    if (SE->isLoopInvariant(Sc, OrigLoop)) {
      DEBUG(dbgs() << "LV: Adding RT check for a loop invariant ptr:" <<
            *Ptr <<"\n");
      Starts.push_back(Ptr);
      Ends.push_back(Ptr);
    } else {
      DEBUG(dbgs() << "LV: Adding RT check for range:" << *Ptr <<"\n");

      Value *Start = Exp.expandCodeFor(PtrRtCheck->Starts[i], PtrArithTy, Loc);
      Value *End = Exp.expandCodeFor(PtrRtCheck->Ends[i], PtrArithTy, Loc);
      Starts.push_back(Start);
      Ends.push_back(End);
    }
  }

  for (unsigned i = 0; i < NumPointers; ++i) {
    for (unsigned j = i+1; j < NumPointers; ++j) {
      Instruction::CastOps Op = Instruction::BitCast;
      Value *Start0 = CastInst::Create(Op, Starts[i], PtrArithTy, "bc", Loc);
      Value *Start1 = CastInst::Create(Op, Starts[j], PtrArithTy, "bc", Loc);
      Value *End0 =   CastInst::Create(Op, Ends[i],   PtrArithTy, "bc", Loc);
      Value *End1 =   CastInst::Create(Op, Ends[j],   PtrArithTy, "bc", Loc);

      Value *Cmp0 = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_ULE,
                                    Start0, End1, "bound0", Loc);
      Value *Cmp1 = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_ULE,
                                    Start1, End0, "bound1", Loc);
      Value *IsConflict = BinaryOperator::Create(Instruction::And, Cmp0, Cmp1,
                                                 "found.conflict", Loc);
      if (MemoryRuntimeCheck)
        MemoryRuntimeCheck = BinaryOperator::Create(Instruction::Or,
                                                    MemoryRuntimeCheck,
                                                    IsConflict,
                                                    "conflict.rdx", Loc);
      else
        MemoryRuntimeCheck = IsConflict;

    }
  }

  return MemoryRuntimeCheck;
}

void
InnerLoopVectorizer::createEmptyLoop(LoopVectorizationLegality *Legal) {
  /*
   In this function we generate a new loop. The new loop will contain
   the vectorized instructions while the old loop will continue to run the
   scalar remainder.

       [ ] <-- vector loop bypass.
     /  |
    /   v
   |   [ ]     <-- vector pre header.
   |    |
   |    v
   |   [  ] \
   |   [  ]_|   <-- vector loop.
   |    |
    \   v
      >[ ]   <--- middle-block.
     /  |
    /   v
   |   [ ]     <--- new preheader.
   |    |
   |    v
   |   [ ] \
   |   [ ]_|   <-- old scalar loop to handle remainder.
    \   |
     \  v
      >[ ]     <-- exit block.
   ...
   */

  BasicBlock *OldBasicBlock = OrigLoop->getHeader();
  BasicBlock *BypassBlock = OrigLoop->getLoopPreheader();
  BasicBlock *ExitBlock = OrigLoop->getExitBlock();
  assert(ExitBlock && "Must have an exit block");

  // Some loops have a single integer induction variable, while other loops
  // don't. One example is c++ iterators that often have multiple pointer
  // induction variables. In the code below we also support a case where we
  // don't have a single induction variable.
  OldInduction = Legal->getInduction();
  Type *IdxTy = OldInduction ? OldInduction->getType() :
  DL->getIntPtrType(SE->getContext());

  // Find the loop boundaries.
  const SCEV *ExitCount = SE->getExitCount(OrigLoop, OrigLoop->getLoopLatch());
  assert(ExitCount != SE->getCouldNotCompute() && "Invalid loop count");

  // Get the total trip count from the count by adding 1.
  ExitCount = SE->getAddExpr(ExitCount,
                             SE->getConstant(ExitCount->getType(), 1));

  // Expand the trip count and place the new instructions in the preheader.
  // Notice that the pre-header does not change, only the loop body.
  SCEVExpander Exp(*SE, "induction");

  // Count holds the overall loop count (N).
  Value *Count = Exp.expandCodeFor(ExitCount, ExitCount->getType(),
                                   BypassBlock->getTerminator());

  // The loop index does not have to start at Zero. Find the original start
  // value from the induction PHI node. If we don't have an induction variable
  // then we know that it starts at zero.
  Value *StartIdx = OldInduction ?
  OldInduction->getIncomingValueForBlock(BypassBlock):
  ConstantInt::get(IdxTy, 0);

  assert(BypassBlock && "Invalid loop structure");

  // Generate the code that checks in runtime if arrays overlap.
  Value *MemoryRuntimeCheck = addRuntimeCheck(Legal,
                                              BypassBlock->getTerminator());

  // Split the single block loop into the two loop structure described above.
  BasicBlock *VectorPH =
  BypassBlock->splitBasicBlock(BypassBlock->getTerminator(), "vector.ph");
  BasicBlock *VecBody =
  VectorPH->splitBasicBlock(VectorPH->getTerminator(), "vector.body");
  BasicBlock *MiddleBlock =
  VecBody->splitBasicBlock(VecBody->getTerminator(), "middle.block");
  BasicBlock *ScalarPH =
  MiddleBlock->splitBasicBlock(MiddleBlock->getTerminator(), "scalar.ph");

  // This is the location in which we add all of the logic for bypassing
  // the new vector loop.
  Instruction *Loc = BypassBlock->getTerminator();

  // Use this IR builder to create the loop instructions (Phi, Br, Cmp)
  // inside the loop.
  Builder.SetInsertPoint(VecBody->getFirstInsertionPt());

  // Generate the induction variable.
  Induction = Builder.CreatePHI(IdxTy, 2, "index");
  Constant *Step = ConstantInt::get(IdxTy, VF);

  // We may need to extend the index in case there is a type mismatch.
  // We know that the count starts at zero and does not overflow.
  if (Count->getType() != IdxTy) {
    // The exit count can be of pointer type. Convert it to the correct
    // integer type.
    if (ExitCount->getType()->isPointerTy())
      Count = CastInst::CreatePointerCast(Count, IdxTy, "ptrcnt.to.int", Loc);
    else
      Count = CastInst::CreateZExtOrBitCast(Count, IdxTy, "zext.cnt", Loc);
  }

  // Add the start index to the loop count to get the new end index.
  Value *IdxEnd = BinaryOperator::CreateAdd(Count, StartIdx, "end.idx", Loc);

  // Now we need to generate the expression for N - (N % VF), which is
  // the part that the vectorized body will execute.
  Constant *CIVF = ConstantInt::get(IdxTy, VF);
  Value *R = BinaryOperator::CreateURem(Count, CIVF, "n.mod.vf", Loc);
  Value *CountRoundDown = BinaryOperator::CreateSub(Count, R, "n.vec", Loc);
  Value *IdxEndRoundDown = BinaryOperator::CreateAdd(CountRoundDown, StartIdx,
                                                     "end.idx.rnd.down", Loc);

  // Now, compare the new count to zero. If it is zero skip the vector loop and
  // jump to the scalar loop.
  Value *Cmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_EQ,
                               IdxEndRoundDown,
                               StartIdx,
                               "cmp.zero", Loc);

  // If we are using memory runtime checks, include them in.
  if (MemoryRuntimeCheck)
    Cmp = BinaryOperator::Create(Instruction::Or, Cmp, MemoryRuntimeCheck,
                                 "CntOrMem", Loc);

  BranchInst::Create(MiddleBlock, VectorPH, Cmp, Loc);
  // Remove the old terminator.
  Loc->eraseFromParent();

  // We are going to resume the execution of the scalar loop.
  // Go over all of the induction variables that we found and fix the
  // PHIs that are left in the scalar version of the loop.
  // The starting values of PHI nodes depend on the counter of the last
  // iteration in the vectorized loop.
  // If we come from a bypass edge then we need to start from the original
  // start value.

  // This variable saves the new starting index for the scalar loop.
  PHINode *ResumeIndex = 0;
  LoopVectorizationLegality::InductionList::iterator I, E;
  LoopVectorizationLegality::InductionList *List = Legal->getInductionVars();
  for (I = List->begin(), E = List->end(); I != E; ++I) {
    PHINode *OrigPhi = I->first;
    LoopVectorizationLegality::InductionInfo II = I->second;
    PHINode *ResumeVal = PHINode::Create(OrigPhi->getType(), 2, "resume.val",
                                         MiddleBlock->getTerminator());
    Value *EndValue = 0;
    switch (II.IK) {
    case LoopVectorizationLegality::NoInduction:
      llvm_unreachable("Unknown induction");
    case LoopVectorizationLegality::IntInduction: {
      // Handle the integer induction counter:
      assert(OrigPhi->getType()->isIntegerTy() && "Invalid type");
      assert(OrigPhi == OldInduction && "Unknown integer PHI");
      // We know what the end value is.
      EndValue = IdxEndRoundDown;
      // We also know which PHI node holds it.
      ResumeIndex = ResumeVal;
      break;
    }
    case LoopVectorizationLegality::ReverseIntInduction: {
      // Convert the CountRoundDown variable to the PHI size.
      unsigned CRDSize = CountRoundDown->getType()->getScalarSizeInBits();
      unsigned IISize = II.StartValue->getType()->getScalarSizeInBits();
      Value *CRD = CountRoundDown;
      if (CRDSize > IISize)
        CRD = CastInst::Create(Instruction::Trunc, CountRoundDown,
                               II.StartValue->getType(),
                               "tr.crd", BypassBlock->getTerminator());
      else if (CRDSize < IISize)
        CRD = CastInst::Create(Instruction::SExt, CountRoundDown,
                               II.StartValue->getType(),
                               "sext.crd", BypassBlock->getTerminator());
      // Handle reverse integer induction counter:
      EndValue = BinaryOperator::CreateSub(II.StartValue, CRD, "rev.ind.end",
                                           BypassBlock->getTerminator());
      break;
    }
    case LoopVectorizationLegality::PtrInduction: {
      // For pointer induction variables, calculate the offset using
      // the end index.
      EndValue = GetElementPtrInst::Create(II.StartValue, CountRoundDown,
                                           "ptr.ind.end",
                                           BypassBlock->getTerminator());
      break;
    }
    }// end of case

    // The new PHI merges the original incoming value, in case of a bypass,
    // or the value at the end of the vectorized loop.
    ResumeVal->addIncoming(II.StartValue, BypassBlock);
    ResumeVal->addIncoming(EndValue, VecBody);

    // Fix the scalar body counter (PHI node).
    unsigned BlockIdx = OrigPhi->getBasicBlockIndex(ScalarPH);
    OrigPhi->setIncomingValue(BlockIdx, ResumeVal);
  }

  // If we are generating a new induction variable then we also need to
  // generate the code that calculates the exit value. This value is not
  // simply the end of the counter because we may skip the vectorized body
  // in case of a runtime check.
  if (!OldInduction){
    assert(!ResumeIndex && "Unexpected resume value found");
    ResumeIndex = PHINode::Create(IdxTy, 2, "new.indc.resume.val",
                                  MiddleBlock->getTerminator());
    ResumeIndex->addIncoming(StartIdx, BypassBlock);
    ResumeIndex->addIncoming(IdxEndRoundDown, VecBody);
  }

  // Make sure that we found the index where scalar loop needs to continue.
  assert(ResumeIndex && ResumeIndex->getType()->isIntegerTy() &&
         "Invalid resume Index");

  // Add a check in the middle block to see if we have completed
  // all of the iterations in the first vector loop.
  // If (N - N%VF) == N, then we *don't* need to run the remainder.
  Value *CmpN = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_EQ, IdxEnd,
                                ResumeIndex, "cmp.n",
                                MiddleBlock->getTerminator());

  BranchInst::Create(ExitBlock, ScalarPH, CmpN, MiddleBlock->getTerminator());
  // Remove the old terminator.
  MiddleBlock->getTerminator()->eraseFromParent();

  // Create i+1 and fill the PHINode.
  Value *NextIdx = Builder.CreateAdd(Induction, Step, "index.next");
  Induction->addIncoming(StartIdx, VectorPH);
  Induction->addIncoming(NextIdx, VecBody);
  // Create the compare.
  Value *ICmp = Builder.CreateICmpEQ(NextIdx, IdxEndRoundDown);
  Builder.CreateCondBr(ICmp, MiddleBlock, VecBody);

  // Now we have two terminators. Remove the old one from the block.
  VecBody->getTerminator()->eraseFromParent();

  // Get ready to start creating new instructions into the vectorized body.
  Builder.SetInsertPoint(VecBody->getFirstInsertionPt());

  // Create and register the new vector loop.
  Loop* Lp = new Loop();
  Loop *ParentLoop = OrigLoop->getParentLoop();

  // Insert the new loop into the loop nest and register the new basic blocks.
  if (ParentLoop) {
    ParentLoop->addChildLoop(Lp);
    ParentLoop->addBasicBlockToLoop(ScalarPH, LI->getBase());
    ParentLoop->addBasicBlockToLoop(VectorPH, LI->getBase());
    ParentLoop->addBasicBlockToLoop(MiddleBlock, LI->getBase());
  } else {
    LI->addTopLevelLoop(Lp);
  }

  Lp->addBasicBlockToLoop(VecBody, LI->getBase());

  // Save the state.
  LoopVectorPreHeader = VectorPH;
  LoopScalarPreHeader = ScalarPH;
  LoopMiddleBlock = MiddleBlock;
  LoopExitBlock = ExitBlock;
  LoopVectorBody = VecBody;
  LoopScalarBody = OldBasicBlock;
  LoopBypassBlock = BypassBlock;
}

/// This function returns the identity element (or neutral element) for
/// the operation K.
static unsigned
getReductionIdentity(LoopVectorizationLegality::ReductionKind K) {
  switch (K) {
  case LoopVectorizationLegality::IntegerXor:
  case LoopVectorizationLegality::IntegerAdd:
  case LoopVectorizationLegality::IntegerOr:
    // Adding, Xoring, Oring zero to a number does not change it.
    return 0;
  case LoopVectorizationLegality::IntegerMult:
    // Multiplying a number by 1 does not change it.
    return 1;
  case LoopVectorizationLegality::IntegerAnd:
    // AND-ing a number with an all-1 value does not change it.
    return -1;
  default:
    llvm_unreachable("Unknown reduction kind");
  }
}

static bool
isTriviallyVectorizableIntrinsic(Instruction *Inst) {
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst);
  if (!II)
    return false;
  switch (II->getIntrinsicID()) {
  case Intrinsic::sqrt:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::log2:
  case Intrinsic::fabs:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::pow:
  case Intrinsic::fma:
    return true;
  default:
    return false;
  }
  return false;
}

void
InnerLoopVectorizer::vectorizeLoop(LoopVectorizationLegality *Legal) {
  //===------------------------------------------------===//
  //
  // Notice: any optimization or new instruction that go
  // into the code below should be also be implemented in
  // the cost-model.
  //
  //===------------------------------------------------===//
  BasicBlock &BB = *OrigLoop->getHeader();
  Constant *Zero =
  ConstantInt::get(IntegerType::getInt32Ty(BB.getContext()), 0);

  // In order to support reduction variables we need to be able to vectorize
  // Phi nodes. Phi nodes have cycles, so we need to vectorize them in two
  // stages. First, we create a new vector PHI node with no incoming edges.
  // We use this value when we vectorize all of the instructions that use the
  // PHI. Next, after all of the instructions in the block are complete we
  // add the new incoming edges to the PHI. At this point all of the
  // instructions in the basic block are vectorized, so we can use them to
  // construct the PHI.
  PhiVector RdxPHIsToFix;

  // Scan the loop in a topological order to ensure that defs are vectorized
  // before users.
  LoopBlocksDFS DFS(OrigLoop);
  DFS.perform(LI);

  // Vectorize all of the blocks in the original loop.
  for (LoopBlocksDFS::RPOIterator bb = DFS.beginRPO(),
       be = DFS.endRPO(); bb != be; ++bb)
    vectorizeBlockInLoop(Legal, *bb, &RdxPHIsToFix);

  // At this point every instruction in the original loop is widened to
  // a vector form. We are almost done. Now, we need to fix the PHI nodes
  // that we vectorized. The PHI nodes are currently empty because we did
  // not want to introduce cycles. Notice that the remaining PHI nodes
  // that we need to fix are reduction variables.

  // Create the 'reduced' values for each of the induction vars.
  // The reduced values are the vector values that we scalarize and combine
  // after the loop is finished.
  for (PhiVector::iterator it = RdxPHIsToFix.begin(), e = RdxPHIsToFix.end();
       it != e; ++it) {
    PHINode *RdxPhi = *it;
    PHINode *VecRdxPhi = dyn_cast<PHINode>(WidenMap[RdxPhi]);
    assert(RdxPhi && "Unable to recover vectorized PHI");

    // Find the reduction variable descriptor.
    assert(Legal->getReductionVars()->count(RdxPhi) &&
           "Unable to find the reduction variable");
    LoopVectorizationLegality::ReductionDescriptor RdxDesc =
    (*Legal->getReductionVars())[RdxPhi];

    // We need to generate a reduction vector from the incoming scalar.
    // To do so, we need to generate the 'identity' vector and overide
    // one of the elements with the incoming scalar reduction. We need
    // to do it in the vector-loop preheader.
    Builder.SetInsertPoint(LoopBypassBlock->getTerminator());

    // This is the vector-clone of the value that leaves the loop.
    Value *VectorExit = getVectorValue(RdxDesc.LoopExitInstr);
    Type *VecTy = VectorExit->getType();

    // Find the reduction identity variable. Zero for addition, or, xor,
    // one for multiplication, -1 for And.
    Constant *Identity = getUniformVector(getReductionIdentity(RdxDesc.Kind),
                                          VecTy->getScalarType());

    // This vector is the Identity vector where the first element is the
    // incoming scalar reduction.
    Value *VectorStart = Builder.CreateInsertElement(Identity,
                                                     RdxDesc.StartValue, Zero);

    // Fix the vector-loop phi.
    // We created the induction variable so we know that the
    // preheader is the first entry.
    BasicBlock *VecPreheader = Induction->getIncomingBlock(0);

    // Reductions do not have to start at zero. They can start with
    // any loop invariant values.
    VecRdxPhi->addIncoming(VectorStart, VecPreheader);
    Value *Val =
    getVectorValue(RdxPhi->getIncomingValueForBlock(OrigLoop->getLoopLatch()));
    VecRdxPhi->addIncoming(Val, LoopVectorBody);

    // Before each round, move the insertion point right between
    // the PHIs and the values we are going to write.
    // This allows us to write both PHINodes and the extractelement
    // instructions.
    Builder.SetInsertPoint(LoopMiddleBlock->getFirstInsertionPt());

    // This PHINode contains the vectorized reduction variable, or
    // the initial value vector, if we bypass the vector loop.
    PHINode *NewPhi = Builder.CreatePHI(VecTy, 2, "rdx.vec.exit.phi");
    NewPhi->addIncoming(VectorStart, LoopBypassBlock);
    NewPhi->addIncoming(getVectorValue(RdxDesc.LoopExitInstr), LoopVectorBody);

    // VF is a power of 2 so we can emit the reduction using log2(VF) shuffles
    // and vector ops, reducing the set of values being computed by half each
    // round.
    assert(isPowerOf2_32(VF) &&
           "Reduction emission only supported for pow2 vectors!");
    Value *TmpVec = NewPhi;
    SmallVector<Constant*, 32> ShuffleMask(VF, 0);
    for (unsigned i = VF; i != 1; i >>= 1) {
      // Move the upper half of the vector to the lower half.
      for (unsigned j = 0; j != i/2; ++j)
        ShuffleMask[j] = Builder.getInt32(i/2 + j);

      // Fill the rest of the mask with undef.
      std::fill(&ShuffleMask[i/2], ShuffleMask.end(),
                UndefValue::get(Builder.getInt32Ty()));

      Value *Shuf =
        Builder.CreateShuffleVector(TmpVec,
                                    UndefValue::get(TmpVec->getType()),
                                    ConstantVector::get(ShuffleMask),
                                    "rdx.shuf");

      // Emit the operation on the shuffled value.
      switch (RdxDesc.Kind) {
      case LoopVectorizationLegality::IntegerAdd:
        TmpVec = Builder.CreateAdd(TmpVec, Shuf, "add.rdx");
        break;
      case LoopVectorizationLegality::IntegerMult:
        TmpVec = Builder.CreateMul(TmpVec, Shuf, "mul.rdx");
        break;
      case LoopVectorizationLegality::IntegerOr:
        TmpVec = Builder.CreateOr(TmpVec, Shuf, "or.rdx");
        break;
      case LoopVectorizationLegality::IntegerAnd:
        TmpVec = Builder.CreateAnd(TmpVec, Shuf, "and.rdx");
        break;
      case LoopVectorizationLegality::IntegerXor:
        TmpVec = Builder.CreateXor(TmpVec, Shuf, "xor.rdx");
        break;
      default:
        llvm_unreachable("Unknown reduction operation");
      }
    }

    // The result is in the first element of the vector.
    Value *Scalar0 = Builder.CreateExtractElement(TmpVec, Builder.getInt32(0));

    // Now, we need to fix the users of the reduction variable
    // inside and outside of the scalar remainder loop.
    // We know that the loop is in LCSSA form. We need to update the
    // PHI nodes in the exit blocks.
    for (BasicBlock::iterator LEI = LoopExitBlock->begin(),
         LEE = LoopExitBlock->end(); LEI != LEE; ++LEI) {
      PHINode *LCSSAPhi = dyn_cast<PHINode>(LEI);
      if (!LCSSAPhi) continue;

      // All PHINodes need to have a single entry edge, or two if
      // we already fixed them.
      assert(LCSSAPhi->getNumIncomingValues() < 3 && "Invalid LCSSA PHI");

      // We found our reduction value exit-PHI. Update it with the
      // incoming bypass edge.
      if (LCSSAPhi->getIncomingValue(0) == RdxDesc.LoopExitInstr) {
        // Add an edge coming from the bypass.
        LCSSAPhi->addIncoming(Scalar0, LoopMiddleBlock);
        break;
      }
    }// end of the LCSSA phi scan.

    // Fix the scalar loop reduction variable with the incoming reduction sum
    // from the vector body and from the backedge value.
    int IncomingEdgeBlockIdx =
    (RdxPhi)->getBasicBlockIndex(OrigLoop->getLoopLatch());
    assert(IncomingEdgeBlockIdx >= 0 && "Invalid block index");
    // Pick the other block.
    int SelfEdgeBlockIdx = (IncomingEdgeBlockIdx ? 0 : 1);
    (RdxPhi)->setIncomingValue(SelfEdgeBlockIdx, Scalar0);
    (RdxPhi)->setIncomingValue(IncomingEdgeBlockIdx, RdxDesc.LoopExitInstr);
  }// end of for each redux variable.
}

Value *InnerLoopVectorizer::createEdgeMask(BasicBlock *Src, BasicBlock *Dst) {
  assert(std::find(pred_begin(Dst), pred_end(Dst), Src) != pred_end(Dst) &&
         "Invalid edge");

  Value *SrcMask = createBlockInMask(Src);

  // The terminator has to be a branch inst!
  BranchInst *BI = dyn_cast<BranchInst>(Src->getTerminator());
  assert(BI && "Unexpected terminator found");

  Value *EdgeMask = SrcMask;
  if (BI->isConditional()) {
    EdgeMask = getVectorValue(BI->getCondition());
    if (BI->getSuccessor(0) != Dst)
      EdgeMask = Builder.CreateNot(EdgeMask);
  }

  return Builder.CreateAnd(EdgeMask, SrcMask);
}

Value *InnerLoopVectorizer::createBlockInMask(BasicBlock *BB) {
  assert(OrigLoop->contains(BB) && "Block is not a part of a loop");

  // Loop incoming mask is all-one.
  if (OrigLoop->getHeader() == BB) {
    Value *C = ConstantInt::get(IntegerType::getInt1Ty(BB->getContext()), 1);
    return getVectorValue(C);
  }

  // This is the block mask. We OR all incoming edges, and with zero.
  Value *Zero = ConstantInt::get(IntegerType::getInt1Ty(BB->getContext()), 0);
  Value *BlockMask = getVectorValue(Zero);

  // For each pred:
  for (pred_iterator it = pred_begin(BB), e = pred_end(BB); it != e; ++it)
    BlockMask = Builder.CreateOr(BlockMask, createEdgeMask(*it, BB));

  return BlockMask;
}

void
InnerLoopVectorizer::vectorizeBlockInLoop(LoopVectorizationLegality *Legal,
                                          BasicBlock *BB, PhiVector *PV) {
  Constant *Zero =
  ConstantInt::get(IntegerType::getInt32Ty(BB->getContext()), 0);

  // For each instruction in the old loop.
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    switch (it->getOpcode()) {
    case Instruction::Br:
      // Nothing to do for PHIs and BR, since we already took care of the
      // loop control flow instructions.
      continue;
    case Instruction::PHI:{
      PHINode* P = cast<PHINode>(it);
      // Handle reduction variables:
      if (Legal->getReductionVars()->count(P)) {
        // This is phase one of vectorizing PHIs.
        Type *VecTy = VectorType::get(it->getType(), VF);
        WidenMap[it] =
          PHINode::Create(VecTy, 2, "vec.phi",
                          LoopVectorBody->getFirstInsertionPt());
        PV->push_back(P);
        continue;
      }

      // Check for PHI nodes that are lowered to vector selects.
      if (P->getParent() != OrigLoop->getHeader()) {
        // We know that all PHIs in non header blocks are converted into
        // selects, so we don't have to worry about the insertion order and we
        // can just use the builder.

        // At this point we generate the predication tree. There may be
        // duplications since this is a simple recursive scan, but future
        // optimizations will clean it up.
        Value *Cond = createEdgeMask(P->getIncomingBlock(0), P->getParent());
        WidenMap[P] =
          Builder.CreateSelect(Cond,
                               getVectorValue(P->getIncomingValue(0)),
                               getVectorValue(P->getIncomingValue(1)),
                               "predphi");
        continue;
      }

      // This PHINode must be an induction variable.
      // Make sure that we know about it.
      assert(Legal->getInductionVars()->count(P) &&
             "Not an induction variable");

      LoopVectorizationLegality::InductionInfo II =
        Legal->getInductionVars()->lookup(P);

      switch (II.IK) {
      case LoopVectorizationLegality::NoInduction:
        llvm_unreachable("Unknown induction");
      case LoopVectorizationLegality::IntInduction: {
        assert(P == OldInduction && "Unexpected PHI");
        Value *Broadcasted = getBroadcastInstrs(Induction);
        // After broadcasting the induction variable we need to make the
        // vector consecutive by adding 0, 1, 2 ...
        Value *ConsecutiveInduction = getConsecutiveVector(Broadcasted);
        WidenMap[OldInduction] = ConsecutiveInduction;
        continue;
      }
      case LoopVectorizationLegality::ReverseIntInduction:
      case LoopVectorizationLegality::PtrInduction:
        // Handle reverse integer and pointer inductions.
        Value *StartIdx = 0;
        // If we have a single integer induction variable then use it.
        // Otherwise, start counting at zero.
        if (OldInduction) {
          LoopVectorizationLegality::InductionInfo OldII =
            Legal->getInductionVars()->lookup(OldInduction);
          StartIdx = OldII.StartValue;
        } else {
          StartIdx = ConstantInt::get(Induction->getType(), 0);
        }
        // This is the normalized GEP that starts counting at zero.
        Value *NormalizedIdx = Builder.CreateSub(Induction, StartIdx,
                                                 "normalized.idx");

        // Handle the reverse integer induction variable case.
        if (LoopVectorizationLegality::ReverseIntInduction == II.IK) {
          IntegerType *DstTy = cast<IntegerType>(II.StartValue->getType());
          Value *CNI = Builder.CreateSExtOrTrunc(NormalizedIdx, DstTy,
                                                 "resize.norm.idx");
          Value *ReverseInd  = Builder.CreateSub(II.StartValue, CNI,
                                                 "reverse.idx");

          // This is a new value so do not hoist it out.
          Value *Broadcasted = getBroadcastInstrs(ReverseInd);
          // After broadcasting the induction variable we need to make the
          // vector consecutive by adding  ... -3, -2, -1, 0.
          Value *ConsecutiveInduction = getConsecutiveVector(Broadcasted,
                                                             true);
          WidenMap[it] = ConsecutiveInduction;
          continue;
        }

        // Handle the pointer induction variable case.
        assert(P->getType()->isPointerTy() && "Unexpected type.");

        // This is the vector of results. Notice that we don't generate
        // vector geps because scalar geps result in better code.
        Value *VecVal = UndefValue::get(VectorType::get(P->getType(), VF));
        for (unsigned int i = 0; i < VF; ++i) {
          Constant *Idx = ConstantInt::get(Induction->getType(), i);
          Value *GlobalIdx = Builder.CreateAdd(NormalizedIdx, Idx,
                                               "gep.idx");
          Value *SclrGep = Builder.CreateGEP(II.StartValue, GlobalIdx,
                                             "next.gep");
          VecVal = Builder.CreateInsertElement(VecVal, SclrGep,
                                               Builder.getInt32(i),
                                               "insert.gep");
        }

        WidenMap[it] = VecVal;
        continue;
      }

    }// End of PHI.

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
      // Just widen binops.
      BinaryOperator *BinOp = dyn_cast<BinaryOperator>(it);
      Value *A = getVectorValue(it->getOperand(0));
      Value *B = getVectorValue(it->getOperand(1));

      // Use this vector value for all users of the original instruction.
      Value *V = Builder.CreateBinOp(BinOp->getOpcode(), A, B);
      WidenMap[it] = V;

      // Update the NSW, NUW and Exact flags.
      BinaryOperator *VecOp = cast<BinaryOperator>(V);
      if (isa<OverflowingBinaryOperator>(BinOp)) {
        VecOp->setHasNoSignedWrap(BinOp->hasNoSignedWrap());
        VecOp->setHasNoUnsignedWrap(BinOp->hasNoUnsignedWrap());
      }
      if (isa<PossiblyExactOperator>(VecOp))
        VecOp->setIsExact(BinOp->isExact());
      break;
    }
    case Instruction::Select: {
      // Widen selects.
      // If the selector is loop invariant we can create a select
      // instruction with a scalar condition. Otherwise, use vector-select.
      Value *Cond = it->getOperand(0);
      bool InvariantCond = SE->isLoopInvariant(SE->getSCEV(Cond), OrigLoop);

      // The condition can be loop invariant  but still defined inside the
      // loop. This means that we can't just use the original 'cond' value.
      // We have to take the 'vectorized' value and pick the first lane.
      // Instcombine will make this a no-op.
      Cond = getVectorValue(Cond);
      if (InvariantCond)
        Cond = Builder.CreateExtractElement(Cond, Builder.getInt32(0));

      Value *Op0 = getVectorValue(it->getOperand(1));
      Value *Op1 = getVectorValue(it->getOperand(2));
      WidenMap[it] = Builder.CreateSelect(Cond, Op0, Op1);
      break;
    }

    case Instruction::ICmp:
    case Instruction::FCmp: {
      // Widen compares. Generate vector compares.
      bool FCmp = (it->getOpcode() == Instruction::FCmp);
      CmpInst *Cmp = dyn_cast<CmpInst>(it);
      Value *A = getVectorValue(it->getOperand(0));
      Value *B = getVectorValue(it->getOperand(1));
      if (FCmp)
        WidenMap[it] = Builder.CreateFCmp(Cmp->getPredicate(), A, B);
      else
        WidenMap[it] = Builder.CreateICmp(Cmp->getPredicate(), A, B);
      break;
    }

    case Instruction::Store: {
      // Attempt to issue a wide store.
      StoreInst *SI = dyn_cast<StoreInst>(it);
      Type *StTy = VectorType::get(SI->getValueOperand()->getType(), VF);
      Value *Ptr = SI->getPointerOperand();
      unsigned Alignment = SI->getAlignment();

      assert(!Legal->isUniform(Ptr) &&
             "We do not allow storing to uniform addresses");

      GetElementPtrInst *Gep = dyn_cast<GetElementPtrInst>(Ptr);

      // This store does not use GEPs.
      if (!Legal->isConsecutivePtr(Ptr)) {
        scalarizeInstruction(it);
        break;
      }

      if (Gep) {
        // The last index does not have to be the induction. It can be
        // consecutive and be a function of the index. For example A[I+1];
        unsigned NumOperands = Gep->getNumOperands();
        Value *LastIndex = getVectorValue(Gep->getOperand(NumOperands - 1));
        LastIndex = Builder.CreateExtractElement(LastIndex, Zero);

        // Create the new GEP with the new induction variable.
        GetElementPtrInst *Gep2 = cast<GetElementPtrInst>(Gep->clone());
        Gep2->setOperand(NumOperands - 1, LastIndex);
        Ptr = Builder.Insert(Gep2);
      } else {
        // Use the induction element ptr.
        assert(isa<PHINode>(Ptr) && "Invalid induction ptr");
        Ptr = Builder.CreateExtractElement(getVectorValue(Ptr), Zero);
      }
      Ptr = Builder.CreateBitCast(Ptr, StTy->getPointerTo());
      Value *Val = getVectorValue(SI->getValueOperand());
      Builder.CreateStore(Val, Ptr)->setAlignment(Alignment);
      break;
    }
    case Instruction::Load: {
      // Attempt to issue a wide load.
      LoadInst *LI = dyn_cast<LoadInst>(it);
      Type *RetTy = VectorType::get(LI->getType(), VF);
      Value *Ptr = LI->getPointerOperand();
      unsigned Alignment = LI->getAlignment();
      GetElementPtrInst *Gep = dyn_cast<GetElementPtrInst>(Ptr);

      // If the pointer is loop invariant or if it is non consecutive,
      // scalarize the load.
      bool Con = Legal->isConsecutivePtr(Ptr);
      if (Legal->isUniform(Ptr) || !Con) {
        scalarizeInstruction(it);
        break;
      }

      if (Gep) {
        // The last index does not have to be the induction. It can be
        // consecutive and be a function of the index. For example A[I+1];
        unsigned NumOperands = Gep->getNumOperands();
        Value *LastIndex = getVectorValue(Gep->getOperand(NumOperands -1));
        LastIndex = Builder.CreateExtractElement(LastIndex, Zero);

        // Create the new GEP with the new induction variable.
        GetElementPtrInst *Gep2 = cast<GetElementPtrInst>(Gep->clone());
        Gep2->setOperand(NumOperands - 1, LastIndex);
        Ptr = Builder.Insert(Gep2);
      } else {
        // Use the induction element ptr.
        assert(isa<PHINode>(Ptr) && "Invalid induction ptr");
        Ptr = Builder.CreateExtractElement(getVectorValue(Ptr), Zero);
      }

      Ptr = Builder.CreateBitCast(Ptr, RetTy->getPointerTo());
      LI = Builder.CreateLoad(Ptr);
      LI->setAlignment(Alignment);
      // Use this vector value for all users of the load.
      WidenMap[it] = LI;
      break;
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
      CastInst *CI = dyn_cast<CastInst>(it);
      /// Optimize the special case where the source is the induction
      /// variable. Notice that we can only optimize the 'trunc' case
      /// because: a. FP conversions lose precision, b. sext/zext may wrap,
      /// c. other casts depend on pointer size.
      if (CI->getOperand(0) == OldInduction &&
          it->getOpcode() == Instruction::Trunc) {
        Value *ScalarCast = Builder.CreateCast(CI->getOpcode(), Induction,
                                               CI->getType());
        Value *Broadcasted = getBroadcastInstrs(ScalarCast);
        WidenMap[it] = getConsecutiveVector(Broadcasted);
        break;
      }
      /// Vectorize casts.
      Value *A = getVectorValue(it->getOperand(0));
      Type *DestTy = VectorType::get(CI->getType()->getScalarType(), VF);
      WidenMap[it] = Builder.CreateCast(CI->getOpcode(), A, DestTy);
      break;
    }

    case Instruction::Call: {
      assert(isTriviallyVectorizableIntrinsic(it));
      Module *M = BB->getParent()->getParent();
      IntrinsicInst *II = cast<IntrinsicInst>(it);
      Intrinsic::ID ID = II->getIntrinsicID();
      SmallVector<Value*, 4> Args;
      for (unsigned i = 0, ie = II->getNumArgOperands(); i != ie; ++i)
        Args.push_back(getVectorValue(II->getArgOperand(i)));
      Type *Tys[] = { VectorType::get(II->getType()->getScalarType(), VF) };
      Function *F = Intrinsic::getDeclaration(M, ID, Tys);
      WidenMap[it] = Builder.CreateCall(F, Args);
      break;
    }

    default:
      // All other instructions are unsupported. Scalarize them.
      scalarizeInstruction(it);
      break;
    }// end of switch.
  }// end of for_each instr.
}

void InnerLoopVectorizer::updateAnalysis() {
  // Forget the original basic block.
  SE->forgetLoop(OrigLoop);

  // Update the dominator tree information.
  assert(DT->properlyDominates(LoopBypassBlock, LoopExitBlock) &&
         "Entry does not dominate exit.");

  DT->addNewBlock(LoopVectorPreHeader, LoopBypassBlock);
  DT->addNewBlock(LoopVectorBody, LoopVectorPreHeader);
  DT->addNewBlock(LoopMiddleBlock, LoopBypassBlock);
  DT->addNewBlock(LoopScalarPreHeader, LoopMiddleBlock);
  DT->changeImmediateDominator(LoopScalarBody, LoopScalarPreHeader);
  DT->changeImmediateDominator(LoopExitBlock, LoopMiddleBlock);

  DEBUG(DT->verifyAnalysis());
}

bool LoopVectorizationLegality::canVectorizeWithIfConvert() {
  if (!EnableIfConversion)
    return false;

  assert(TheLoop->getNumBlocks() > 1 && "Single block loops are vectorizable");
  std::vector<BasicBlock*> &LoopBlocks = TheLoop->getBlocksVector();

  // Collect the blocks that need predication.
  for (unsigned i = 0, e = LoopBlocks.size(); i < e; ++i) {
    BasicBlock *BB = LoopBlocks[i];

    // We don't support switch statements inside loops.
    if (!isa<BranchInst>(BB->getTerminator()))
      return false;

    // We must have at most two predecessors because we need to convert
    // all PHIs to selects.
    unsigned Preds = std::distance(pred_begin(BB), pred_end(BB));
    if (Preds > 2)
      return false;

    // We must be able to predicate all blocks that need to be predicated.
    if (blockNeedsPredication(BB) && !blockCanBePredicated(BB))
      return false;
  }

  // We can if-convert this loop.
  return true;
}

bool LoopVectorizationLegality::canVectorize() {
  assert(TheLoop->getLoopPreheader() && "No preheader!!");

  // We can only vectorize innermost loops.
  if (TheLoop->getSubLoopsVector().size())
    return false;

  // We must have a single backedge.
  if (TheLoop->getNumBackEdges() != 1)
    return false;

  // We must have a single exiting block.
  if (!TheLoop->getExitingBlock())
    return false;

  unsigned NumBlocks = TheLoop->getNumBlocks();

  // Check if we can if-convert non single-bb loops.
  if (NumBlocks != 1 && !canVectorizeWithIfConvert()) {
    DEBUG(dbgs() << "LV: Can't if-convert the loop.\n");
    return false;
  }

  // We need to have a loop header.
  BasicBlock *Latch = TheLoop->getLoopLatch();
  DEBUG(dbgs() << "LV: Found a loop: " <<
        TheLoop->getHeader()->getName() << "\n");

  // ScalarEvolution needs to be able to find the exit count.
  const SCEV *ExitCount = SE->getExitCount(TheLoop, Latch);
  if (ExitCount == SE->getCouldNotCompute()) {
    DEBUG(dbgs() << "LV: SCEV could not compute the loop exit count.\n");
    return false;
  }

  // Do not loop-vectorize loops with a tiny trip count.
  unsigned TC = SE->getSmallConstantTripCount(TheLoop, Latch);
  if (TC > 0u && TC < TinyTripCountThreshold) {
    DEBUG(dbgs() << "LV: Found a loop with a very small trip count. " <<
          "This loop is not worth vectorizing.\n");
    return false;
  }

  // Check if we can vectorize the instructions and CFG in this loop.
  if (!canVectorizeInstrs()) {
    DEBUG(dbgs() << "LV: Can't vectorize the instructions or CFG\n");
    return false;
  }

  // Go over each instruction and look at memory deps.
  if (!canVectorizeMemory()) {
    DEBUG(dbgs() << "LV: Can't vectorize due to memory conflicts\n");
    return false;
  }

  // Collect all of the variables that remain uniform after vectorization.
  collectLoopUniforms();

  DEBUG(dbgs() << "LV: We can vectorize this loop" <<
        (PtrRtCheck.Need ? " (with a runtime bound check)" : "")
        <<"!\n");

  // Okay! We can vectorize. At this point we don't have any other mem analysis
  // which may limit our maximum vectorization factor, so just return true with
  // no restrictions.
  return true;
}

bool LoopVectorizationLegality::canVectorizeInstrs() {
  BasicBlock *PreHeader = TheLoop->getLoopPreheader();
  BasicBlock *Header = TheLoop->getHeader();

  // For each block in the loop.
  for (Loop::block_iterator bb = TheLoop->block_begin(),
       be = TheLoop->block_end(); bb != be; ++bb) {

    // Scan the instructions in the block and look for hazards.
    for (BasicBlock::iterator it = (*bb)->begin(), e = (*bb)->end(); it != e;
         ++it) {

      if (PHINode *Phi = dyn_cast<PHINode>(it)) {
        // This should not happen because the loop should be normalized.
        if (Phi->getNumIncomingValues() != 2) {
          DEBUG(dbgs() << "LV: Found an invalid PHI.\n");
          return false;
        }

        // Check that this PHI type is allowed.
        if (!Phi->getType()->isIntegerTy() &&
            !Phi->getType()->isPointerTy()) {
          DEBUG(dbgs() << "LV: Found an non-int non-pointer PHI.\n");
          return false;
        }

        // If this PHINode is not in the header block, then we know that we
        // can convert it to select during if-conversion. No need to check if
        // the PHIs in this block are induction or reduction variables.
        if (*bb != Header)
          continue;

        // This is the value coming from the preheader.
        Value *StartValue = Phi->getIncomingValueForBlock(PreHeader);
        // Check if this is an induction variable.
        InductionKind IK = isInductionVariable(Phi);

        if (NoInduction != IK) {
          // Int inductions are special because we only allow one IV.
          if (IK == IntInduction) {
            if (Induction) {
              DEBUG(dbgs() << "LV: Found too many inductions."<< *Phi <<"\n");
              return false;
            }
            Induction = Phi;
          }

          DEBUG(dbgs() << "LV: Found an induction variable.\n");
          Inductions[Phi] = InductionInfo(StartValue, IK);
          continue;
        }

        if (AddReductionVar(Phi, IntegerAdd)) {
          DEBUG(dbgs() << "LV: Found an ADD reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, IntegerMult)) {
          DEBUG(dbgs() << "LV: Found a MUL reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, IntegerOr)) {
          DEBUG(dbgs() << "LV: Found an OR reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, IntegerAnd)) {
          DEBUG(dbgs() << "LV: Found an AND reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, IntegerXor)) {
          DEBUG(dbgs() << "LV: Found a XOR reduction PHI."<< *Phi <<"\n");
          continue;
        }

        DEBUG(dbgs() << "LV: Found an unidentified PHI."<< *Phi <<"\n");
        return false;
      }// end of PHI handling

      // We still don't handle functions.
      CallInst *CI = dyn_cast<CallInst>(it);
      if (CI && !isTriviallyVectorizableIntrinsic(it)) {
        DEBUG(dbgs() << "LV: Found a call site.\n");
        return false;
      }

      // We do not re-vectorize vectors.
      if (!VectorType::isValidElementType(it->getType()) &&
          !it->getType()->isVoidTy()) {
        DEBUG(dbgs() << "LV: Found unvectorizable type." << "\n");
        return false;
      }

      // Reduction instructions are allowed to have exit users.
      // All other instructions must not have external users.
      if (!AllowedExit.count(it))
        //Check that all of the users of the loop are inside the BB.
        for (Value::use_iterator I = it->use_begin(), E = it->use_end();
             I != E; ++I) {
          Instruction *U = cast<Instruction>(*I);
          // This user may be a reduction exit value.
          if (!TheLoop->contains(U)) {
            DEBUG(dbgs() << "LV: Found an outside user for : "<< *U << "\n");
            return false;
          }
        }
    } // next instr.

  }

  if (!Induction) {
    DEBUG(dbgs() << "LV: Did not find one integer induction var.\n");
    assert(getInductionVars()->size() && "No induction variables");
  }

  return true;
}

void LoopVectorizationLegality::collectLoopUniforms() {
  // We now know that the loop is vectorizable!
  // Collect variables that will remain uniform after vectorization.
  std::vector<Value*> Worklist;
  BasicBlock *Latch = TheLoop->getLoopLatch();

  // Start with the conditional branch and walk up the block.
  Worklist.push_back(Latch->getTerminator()->getOperand(0));

  while (Worklist.size()) {
    Instruction *I = dyn_cast<Instruction>(Worklist.back());
    Worklist.pop_back();

    // Look at instructions inside this loop.
    // Stop when reaching PHI nodes.
    // TODO: we need to follow values all over the loop, not only in this block.
    if (!I || !TheLoop->contains(I) || isa<PHINode>(I))
      continue;

    // This is a known uniform.
    Uniforms.insert(I);

    // Insert all operands.
    for (int i = 0, Op = I->getNumOperands(); i < Op; ++i) {
      Worklist.push_back(I->getOperand(i));
    }
  }
}

bool LoopVectorizationLegality::canVectorizeMemory() {
  typedef SmallVector<Value*, 16> ValueVector;
  typedef SmallPtrSet<Value*, 16> ValueSet;
  // Holds the Load and Store *instructions*.
  ValueVector Loads;
  ValueVector Stores;
  PtrRtCheck.Pointers.clear();
  PtrRtCheck.Need = false;

  // For each block.
  for (Loop::block_iterator bb = TheLoop->block_begin(),
       be = TheLoop->block_end(); bb != be; ++bb) {

    // Scan the BB and collect legal loads and stores.
    for (BasicBlock::iterator it = (*bb)->begin(), e = (*bb)->end(); it != e;
         ++it) {

      // If this is a load, save it. If this instruction can read from memory
      // but is not a load, then we quit. Notice that we don't handle function
      // calls that read or write.
      if (it->mayReadFromMemory()) {
        LoadInst *Ld = dyn_cast<LoadInst>(it);
        if (!Ld) return false;
        if (!Ld->isSimple()) {
          DEBUG(dbgs() << "LV: Found a non-simple load.\n");
          return false;
        }
        Loads.push_back(Ld);
        continue;
      }

      // Save 'store' instructions. Abort if other instructions write to memory.
      if (it->mayWriteToMemory()) {
        StoreInst *St = dyn_cast<StoreInst>(it);
        if (!St) return false;
        if (!St->isSimple()) {
          DEBUG(dbgs() << "LV: Found a non-simple store.\n");
          return false;
        }
        Stores.push_back(St);
      }
    } // next instr.
  } // next block.

  // Now we have two lists that hold the loads and the stores.
  // Next, we find the pointers that they use.

  // Check if we see any stores. If there are no stores, then we don't
  // care if the pointers are *restrict*.
  if (!Stores.size()) {
    DEBUG(dbgs() << "LV: Found a read-only loop!\n");
    return true;
  }

  // Holds the read and read-write *pointers* that we find.
  ValueVector Reads;
  ValueVector ReadWrites;

  // Holds the analyzed pointers. We don't want to call GetUnderlyingObjects
  // multiple times on the same object. If the ptr is accessed twice, once
  // for read and once for write, it will only appear once (on the write
  // list). This is okay, since we are going to check for conflicts between
  // writes and between reads and writes, but not between reads and reads.
  ValueSet Seen;

  ValueVector::iterator I, IE;
  for (I = Stores.begin(), IE = Stores.end(); I != IE; ++I) {
    StoreInst *ST = dyn_cast<StoreInst>(*I);
    assert(ST && "Bad StoreInst");
    Value* Ptr = ST->getPointerOperand();

    if (isUniform(Ptr)) {
      DEBUG(dbgs() << "LV: We don't allow storing to uniform addresses\n");
      return false;
    }

    // If we did *not* see this pointer before, insert it to
    // the read-write list. At this phase it is only a 'write' list.
    if (Seen.insert(Ptr))
      ReadWrites.push_back(Ptr);
  }

  for (I = Loads.begin(), IE = Loads.end(); I != IE; ++I) {
    LoadInst *LD = dyn_cast<LoadInst>(*I);
    assert(LD && "Bad LoadInst");
    Value* Ptr = LD->getPointerOperand();
    // If we did *not* see this pointer before, insert it to the
    // read list. If we *did* see it before, then it is already in
    // the read-write list. This allows us to vectorize expressions
    // such as A[i] += x;  Because the address of A[i] is a read-write
    // pointer. This only works if the index of A[i] is consecutive.
    // If the address of i is unknown (for example A[B[i]]) then we may
    // read a few words, modify, and write a few words, and some of the
    // words may be written to the same address.
    if (Seen.insert(Ptr) || !isConsecutivePtr(Ptr))
      Reads.push_back(Ptr);
  }

  // If we write (or read-write) to a single destination and there are no
  // other reads in this loop then is it safe to vectorize.
  if (ReadWrites.size() == 1 && Reads.size() == 0) {
    DEBUG(dbgs() << "LV: Found a write-only loop!\n");
    return true;
  }

  // Find pointers with computable bounds. We are going to use this information
  // to place a runtime bound check.
  bool RT = true;
  for (I = ReadWrites.begin(), IE = ReadWrites.end(); I != IE; ++I)
    if (hasComputableBounds(*I)) {
      PtrRtCheck.insert(SE, TheLoop, *I);
      DEBUG(dbgs() << "LV: Found a runtime check ptr:" << **I <<"\n");
    } else {
      RT = false;
      break;
    }
  for (I = Reads.begin(), IE = Reads.end(); I != IE; ++I)
    if (hasComputableBounds(*I)) {
      PtrRtCheck.insert(SE, TheLoop, *I);
      DEBUG(dbgs() << "LV: Found a runtime check ptr:" << **I <<"\n");
    } else {
      RT = false;
      break;
    }

  // Check that we did not collect too many pointers or found a
  // unsizeable pointer.
  if (!RT || PtrRtCheck.Pointers.size() > RuntimeMemoryCheckThreshold) {
    PtrRtCheck.reset();
    RT = false;
  }

  PtrRtCheck.Need = RT;

  if (RT) {
    DEBUG(dbgs() << "LV: We can perform a memory runtime check if needed.\n");
  }

  // Now that the pointers are in two lists (Reads and ReadWrites), we
  // can check that there are no conflicts between each of the writes and
  // between the writes to the reads.
  ValueSet WriteObjects;
  ValueVector TempObjects;

  // Check that the read-writes do not conflict with other read-write
  // pointers.
  for (I = ReadWrites.begin(), IE = ReadWrites.end(); I != IE; ++I) {
    GetUnderlyingObjects(*I, TempObjects, DL);
    for (ValueVector::iterator it=TempObjects.begin(), e=TempObjects.end();
         it != e; ++it) {
      if (!isIdentifiedObject(*it)) {
        DEBUG(dbgs() << "LV: Found an unidentified write ptr:"<< **it <<"\n");
        return RT;
      }
      if (!WriteObjects.insert(*it)) {
        DEBUG(dbgs() << "LV: Found a possible write-write reorder:"
              << **it <<"\n");
        return RT;
      }
    }
    TempObjects.clear();
  }

  /// Check that the reads don't conflict with the read-writes.
  for (I = Reads.begin(), IE = Reads.end(); I != IE; ++I) {
    GetUnderlyingObjects(*I, TempObjects, DL);
    for (ValueVector::iterator it=TempObjects.begin(), e=TempObjects.end();
         it != e; ++it) {
      if (!isIdentifiedObject(*it)) {
        DEBUG(dbgs() << "LV: Found an unidentified read ptr:"<< **it <<"\n");
        return RT;
      }
      if (WriteObjects.count(*it)) {
        DEBUG(dbgs() << "LV: Found a possible read/write reorder:"
              << **it <<"\n");
        return RT;
      }
    }
    TempObjects.clear();
  }

  // It is safe to vectorize and we don't need any runtime checks.
  DEBUG(dbgs() << "LV: We don't need a runtime memory check.\n");
  PtrRtCheck.reset();
  return true;
}

bool LoopVectorizationLegality::AddReductionVar(PHINode *Phi,
                                                ReductionKind Kind) {
  if (Phi->getNumIncomingValues() != 2)
    return false;

  // Reduction variables are only found in the loop header block.
  if (Phi->getParent() != TheLoop->getHeader())
    return false;

  // Obtain the reduction start value from the value that comes from the loop
  // preheader.
  Value *RdxStart = Phi->getIncomingValueForBlock(TheLoop->getLoopPreheader());

  // ExitInstruction is the single value which is used outside the loop.
  // We only allow for a single reduction value to be used outside the loop.
  // This includes users of the reduction, variables (which form a cycle
  // which ends in the phi node).
  Instruction *ExitInstruction = 0;

  // Iter is our iterator. We start with the PHI node and scan for all of the
  // users of this instruction. All users must be instructions which can be
  // used as reduction variables (such as ADD). We may have a single
  // out-of-block user. They cycle must end with the original PHI.
  // Also, we can't have multiple block-local users.
  Instruction *Iter = Phi;
  while (true) {
    // If the instruction has no users then this is a broken
    // chain and can't be a reduction variable.
    if (Iter->use_empty())
      return false;

    // Any reduction instr must be of one of the allowed kinds.
    if (!isReductionInstr(Iter, Kind))
      return false;

    // Did we find a user inside this block ?
    bool FoundInBlockUser = false;
    // Did we reach the initial PHI node ?
    bool FoundStartPHI = false;

    // For each of the *users* of iter.
    for (Value::use_iterator it = Iter->use_begin(), e = Iter->use_end();
         it != e; ++it) {
      Instruction *U = cast<Instruction>(*it);
      // We already know that the PHI is a user.
      if (U == Phi) {
        FoundStartPHI = true;
        continue;
      }

      // Check if we found the exit user.
      BasicBlock *Parent = U->getParent();
      if (!TheLoop->contains(Parent)) {
        // Exit if you find multiple outside users.
        if (ExitInstruction != 0)
          return false;
        ExitInstruction = Iter;
      }

      // We allow in-loop PHINodes which are not the original reduction PHI
      // node. If this PHI is the only user of Iter (happens in IF w/ no ELSE
      // structure) then don't skip this PHI.
      if (isa<PHINode>(U) && U->getParent() != TheLoop->getHeader() &&
          TheLoop->contains(U) && Iter->getNumUses() > 1)
        continue;

      // We can't have multiple inside users.
      if (FoundInBlockUser)
        return false;
      FoundInBlockUser = true;
      Iter = U;
    }

    // We found a reduction var if we have reached the original
    // phi node and we only have a single instruction with out-of-loop
    // users.
    if (FoundStartPHI && ExitInstruction) {
      // This instruction is allowed to have out-of-loop users.
      AllowedExit.insert(ExitInstruction);

      // Save the description of this reduction variable.
      ReductionDescriptor RD(RdxStart, ExitInstruction, Kind);
      Reductions[Phi] = RD;
      return true;
    }

    // If we've reached the start PHI but did not find an outside user then
    // this is dead code. Abort.
    if (FoundStartPHI)
      return false;
  }
}

bool
LoopVectorizationLegality::isReductionInstr(Instruction *I,
                                            ReductionKind Kind) {
  switch (I->getOpcode()) {
  default:
    return false;
  case Instruction::PHI:
    // possibly.
    return true;
  case Instruction::Add:
  case Instruction::Sub:
    return Kind == IntegerAdd;
  case Instruction::Mul:
    return Kind == IntegerMult;
  case Instruction::And:
    return Kind == IntegerAnd;
  case Instruction::Or:
    return Kind == IntegerOr;
  case Instruction::Xor:
    return Kind == IntegerXor;
  }
}

LoopVectorizationLegality::InductionKind
LoopVectorizationLegality::isInductionVariable(PHINode *Phi) {
  Type *PhiTy = Phi->getType();
  // We only handle integer and pointer inductions variables.
  if (!PhiTy->isIntegerTy() && !PhiTy->isPointerTy())
    return NoInduction;

  // Check that the PHI is consecutive and starts at zero.
  const SCEV *PhiScev = SE->getSCEV(Phi);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PhiScev);
  if (!AR) {
    DEBUG(dbgs() << "LV: PHI is not a poly recurrence.\n");
    return NoInduction;
  }
  const SCEV *Step = AR->getStepRecurrence(*SE);

  // Integer inductions need to have a stride of one.
  if (PhiTy->isIntegerTy()) {
    if (Step->isOne())
      return IntInduction;
    if (Step->isAllOnesValue())
      return ReverseIntInduction;
    return NoInduction;
  }

  // Calculate the pointer stride and check if it is consecutive.
  const SCEVConstant *C = dyn_cast<SCEVConstant>(Step);
  if (!C)
    return NoInduction;

  assert(PhiTy->isPointerTy() && "The PHI must be a pointer");
  uint64_t Size = DL->getTypeAllocSize(PhiTy->getPointerElementType());
  if (C->getValue()->equalsInt(Size))
    return PtrInduction;

  return NoInduction;
}

bool LoopVectorizationLegality::isInductionVariable(const Value *V) {
  Value *In0 = const_cast<Value*>(V);
  PHINode *PN = dyn_cast_or_null<PHINode>(In0);
  if (!PN)
    return false;

  return Inductions.count(PN);
}

bool LoopVectorizationLegality::blockNeedsPredication(BasicBlock *BB)  {
  assert(TheLoop->contains(BB) && "Unknown block used");

  // Blocks that do not dominate the latch need predication.
  BasicBlock* Latch = TheLoop->getLoopLatch();
  return !DT->dominates(BB, Latch);
}

bool LoopVectorizationLegality::blockCanBePredicated(BasicBlock *BB) {
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    // We don't predicate loads/stores at the moment.
    if (it->mayReadFromMemory() || it->mayWriteToMemory() || it->mayThrow())
      return false;

    // The instructions below can trap.
    switch (it->getOpcode()) {
    default: continue;
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
             return false;
    }
  }

  return true;
}

bool LoopVectorizationLegality::hasComputableBounds(Value *Ptr) {
  const SCEV *PhiScev = SE->getSCEV(Ptr);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PhiScev);
  if (!AR)
    return false;

  return AR->isAffine();
}

unsigned
LoopVectorizationCostModel::selectVectorizationFactor(bool OptForSize,
                                                        unsigned UserVF) {
  if (OptForSize && Legal->getRuntimePointerCheck()->Need) {
    DEBUG(dbgs() << "LV: Aborting. Runtime ptr check is required in Os.\n");
    return 1;
  }

  // Find the trip count.
  unsigned TC = SE->getSmallConstantTripCount(TheLoop, TheLoop->getLoopLatch());
  DEBUG(dbgs() << "LV: Found trip count:"<<TC<<"\n");

  unsigned VF = MaxVectorSize;

  // If we optimize the program for size, avoid creating the tail loop.
  if (OptForSize) {
    // If we are unable to calculate the trip count then don't try to vectorize.
    if (TC < 2) {
      DEBUG(dbgs() << "LV: Aborting. A tail loop is required in Os.\n");
      return 1;
    }

    // Find the maximum SIMD width that can fit within the trip count.
    VF = TC % MaxVectorSize;

    if (VF == 0)
      VF = MaxVectorSize;

    // If the trip count that we found modulo the vectorization factor is not
    // zero then we require a tail.
    if (VF < 2) {
      DEBUG(dbgs() << "LV: Aborting. A tail loop is required in Os.\n");
      return 1;
    }
  }

  if (UserVF != 0) {
    assert(isPowerOf2_32(UserVF) && "VF needs to be a power of two");
    DEBUG(dbgs() << "LV: Using user VF "<<UserVF<<".\n");

    return UserVF;
  }

  if (!VTTI) {
    DEBUG(dbgs() << "LV: No vector target information. Not vectorizing. \n");
    return 1;
  }

  float Cost = expectedCost(1);
  unsigned Width = 1;
  DEBUG(dbgs() << "LV: Scalar loop costs: "<< (int)Cost << ".\n");
  for (unsigned i=2; i <= VF; i*=2) {
    // Notice that the vector loop needs to be executed less times, so
    // we need to divide the cost of the vector loops by the width of
    // the vector elements.
    float VectorCost = expectedCost(i) / (float)i;
    DEBUG(dbgs() << "LV: Vector loop of width "<< i << " costs: " <<
          (int)VectorCost << ".\n");
    if (VectorCost < Cost) {
      Cost = VectorCost;
      Width = i;
    }
  }

  DEBUG(dbgs() << "LV: Selecting VF = : "<< Width << ".\n");
  return Width;
}

unsigned LoopVectorizationCostModel::expectedCost(unsigned VF) {
  unsigned Cost = 0;

  // For each block.
  for (Loop::block_iterator bb = TheLoop->block_begin(),
       be = TheLoop->block_end(); bb != be; ++bb) {
    unsigned BlockCost = 0;
    BasicBlock *BB = *bb;

    // For each instruction in the old loop.
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
      unsigned C = getInstructionCost(it, VF);
      Cost += C;
      DEBUG(dbgs() << "LV: Found an estimated cost of "<< C <<" for VF " <<
            VF << " For instruction: "<< *it << "\n");
    }

    // We assume that if-converted blocks have a 50% chance of being executed.
    // When the code is scalar then some of the blocks are avoided due to CF.
    // When the code is vectorized we execute all code paths.
    if (Legal->blockNeedsPredication(*bb) && VF == 1)
      BlockCost /= 2;

    Cost += BlockCost;
  }

  return Cost;
}

unsigned
LoopVectorizationCostModel::getInstructionCost(Instruction *I, unsigned VF) {
  assert(VTTI && "Invalid vector target transformation info");

  // If we know that this instruction will remain uniform, check the cost of
  // the scalar version.
  if (Legal->isUniformAfterVectorization(I))
    VF = 1;

  Type *RetTy = I->getType();
  Type *VectorTy = ToVectorTy(RetTy, VF);

  // TODO: We need to estimate the cost of intrinsic calls.
  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:
    // We mark this instruction as zero-cost because scalar GEPs are usually
    // lowered to the intruction addressing mode. At the moment we don't
    // generate vector geps.
    return 0;
  case Instruction::Br: {
    return VTTI->getCFInstrCost(I->getOpcode());
  }
  case Instruction::PHI:
    //TODO: IF-converted IFs become selects.
    return 0;
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
  case Instruction::Xor:
    return VTTI->getArithmeticInstrCost(I->getOpcode(), VectorTy);
  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(I);
    const SCEV *CondSCEV = SE->getSCEV(SI->getCondition());
    bool ScalarCond = (SE->isLoopInvariant(CondSCEV, TheLoop));
    Type *CondTy = SI->getCondition()->getType();
    if (ScalarCond)
      CondTy = VectorType::get(CondTy, VF);

    return VTTI->getCmpSelInstrCost(I->getOpcode(), VectorTy, CondTy);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ValTy = I->getOperand(0)->getType();
    VectorTy = ToVectorTy(ValTy, VF);
    return VTTI->getCmpSelInstrCost(I->getOpcode(), VectorTy);
  }
  case Instruction::Store: {
    StoreInst *SI = cast<StoreInst>(I);
    Type *ValTy = SI->getValueOperand()->getType();
    VectorTy = ToVectorTy(ValTy, VF);

    if (VF == 1)
      return VTTI->getMemoryOpCost(I->getOpcode(), ValTy,
                                   SI->getAlignment(),
                                   SI->getPointerAddressSpace());

    // Scalarized stores.
    if (!Legal->isConsecutivePtr(SI->getPointerOperand())) {
      unsigned Cost = 0;
      unsigned ExtCost = VTTI->getInstrCost(Instruction::ExtractElement,
                                            ValTy);
      // The cost of extracting from the value vector.
      Cost += VF * (ExtCost);
      // The cost of the scalar stores.
      Cost += VF * VTTI->getMemoryOpCost(I->getOpcode(),
                                         ValTy->getScalarType(),
                                         SI->getAlignment(),
                                         SI->getPointerAddressSpace());
      return Cost;
    }

    // Wide stores.
    return VTTI->getMemoryOpCost(I->getOpcode(), VectorTy, SI->getAlignment(),
                                 SI->getPointerAddressSpace());
  }
  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(I);

    if (VF == 1)
      return VTTI->getMemoryOpCost(I->getOpcode(), RetTy,
                                   LI->getAlignment(),
                                   LI->getPointerAddressSpace());

    // Scalarized loads.
    if (!Legal->isConsecutivePtr(LI->getPointerOperand())) {
      unsigned Cost = 0;
      unsigned InCost = VTTI->getInstrCost(Instruction::InsertElement, RetTy);
      // The cost of inserting the loaded value into the result vector.
      Cost += VF * (InCost);
      // The cost of the scalar stores.
      Cost += VF * VTTI->getMemoryOpCost(I->getOpcode(),
                                         RetTy->getScalarType(),
                                         LI->getAlignment(),
                                         LI->getPointerAddressSpace());
      return Cost;
    }

    // Wide loads.
    return VTTI->getMemoryOpCost(I->getOpcode(), VectorTy, LI->getAlignment(),
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
    // We optimize the truncation of induction variable.
    // The cost of these is the same as the scalar operation.
    if (I->getOpcode() == Instruction::Trunc &&
        Legal->isInductionVariable(I->getOperand(0)))
         return VTTI->getCastInstrCost(I->getOpcode(), I->getType(),
                                       I->getOperand(0)->getType());

    Type *SrcVecTy = ToVectorTy(I->getOperand(0)->getType(), VF);
    return VTTI->getCastInstrCost(I->getOpcode(), VectorTy, SrcVecTy);
  }
  case Instruction::Call: {
    assert(isTriviallyVectorizableIntrinsic(I));
    IntrinsicInst *II = cast<IntrinsicInst>(I);
    Type *RetTy = ToVectorTy(II->getType(), VF);
    SmallVector<Type*, 4> Tys;
    for (unsigned i = 0, ie = II->getNumArgOperands(); i != ie; ++i)
      Tys.push_back(ToVectorTy(II->getArgOperand(i)->getType(), VF));
    return VTTI->getIntrinsicInstrCost(II->getIntrinsicID(), RetTy, Tys);
  }
  default: {
    // We are scalarizing the instruction. Return the cost of the scalar
    // instruction, plus the cost of insert and extract into vector
    // elements, times the vector width.
    unsigned Cost = 0;

    bool IsVoid = RetTy->isVoidTy();

    unsigned InsCost = (IsVoid ? 0 :
                        VTTI->getInstrCost(Instruction::InsertElement,
                                           VectorTy));

    unsigned ExtCost = VTTI->getInstrCost(Instruction::ExtractElement,
                                          VectorTy);

    // The cost of inserting the results plus extracting each one of the
    // operands.
    Cost += VF * (InsCost + ExtCost * I->getNumOperands());

    // The cost of executing VF copies of the scalar instruction.
    Cost += VF * VTTI->getInstrCost(I->getOpcode(), RetTy);
    return Cost;
  }
  }// end of switch.
}

Type* LoopVectorizationCostModel::ToVectorTy(Type *Scalar, unsigned VF) {
  if (Scalar->isVoidTy() || VF == 1)
    return Scalar;
  return VectorType::get(Scalar, VF);
}

char LoopVectorize::ID = 0;
static const char lv_name[] = "Loop Vectorization";
INITIALIZE_PASS_BEGIN(LoopVectorize, LV_NAME, lv_name, false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(LoopVectorize, LV_NAME, lv_name, false, false)

namespace llvm {
  Pass *createLoopVectorizePass() {
    return new LoopVectorize();
  }
}


