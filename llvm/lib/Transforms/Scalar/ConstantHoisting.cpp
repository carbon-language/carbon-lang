//===- ConstantHoisting.cpp - Prepare code for expensive constants --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass identifies expensive constants to hoist and coalesces them to
// better prepare it for SelectionDAG-based code generation. This works around
// the limitations of the basic-block-at-a-time approach.
//
// First it scans all instructions for integer constants and calculates its
// cost. If the constant can be folded into the instruction (the cost is
// TCC_Free) or the cost is just a simple operation (TCC_BASIC), then we don't
// consider it expensive and leave it alone. This is the default behavior and
// the default implementation of getIntImmCost will always return TCC_Free.
//
// If the cost is more than TCC_BASIC, then the integer constant can't be folded
// into the instruction and it might be beneficial to hoist the constant.
// Similar constants are coalesced to reduce register pressure and
// materialization code.
//
// When a constant is hoisted, it is also hidden behind a bitcast to force it to
// be live-out of the basic block. Otherwise the constant would be just
// duplicated and each basic block would have its own copy in the SelectionDAG.
// The SelectionDAG recognizes such constants as opaque and doesn't perform
// certain transformations on them, which would create a new expensive constant.
//
// This optimization is only applied to integer constants in instructions and
// simple (this means not nested) constant cast experessions. For example:
// %0 = load i64* inttoptr (i64 big_constant to i64*)
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "consthoist"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

STATISTIC(NumConstantsHoisted, "Number of constants hoisted");
STATISTIC(NumConstantsRebased, "Number of constants rebased");


namespace {
typedef SmallVector<User *, 4> ConstantUseListType;
struct ConstantCandidate {
  unsigned CumulativeCost;
  ConstantUseListType Uses;
};

struct ConstantInfo {
  ConstantInt *BaseConstant;
  struct RebasedConstantInfo {
    ConstantInt *OriginalConstant;
    Constant *Offset;
    ConstantUseListType Uses;
  };
  typedef SmallVector<RebasedConstantInfo, 4> RebasedConstantListType;
  RebasedConstantListType RebasedConstants;
};

class ConstantHoisting : public FunctionPass {
  const TargetTransformInfo *TTI;
  DominatorTree *DT;

  /// Keeps track of expensive constants found in the function.
  typedef MapVector<ConstantInt *, ConstantCandidate> ConstantMapType;
  ConstantMapType ConstantMap;

  /// These are the final constants we decided to hoist.
  SmallVector<ConstantInfo, 4> Constants;
public:
  static char ID; // Pass identification, replacement for typeid
  ConstantHoisting() : FunctionPass(ID), TTI(0) {
    initializeConstantHoistingPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F);

  const char *getPassName() const { return "Constant Hoisting"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfo>();
  }

private:
  void CollectConstant(User *U, unsigned Opcode, Intrinsic::ID IID,
                        ConstantInt *C);
  void CollectConstants(Instruction *I);
  void CollectConstants(Function &F);
  void FindAndMakeBaseConstant(ConstantMapType::iterator S,
                               ConstantMapType::iterator E);
  void FindBaseConstants();
  Instruction *FindConstantInsertionPoint(Function &F,
                                          const ConstantInfo &CI) const;
  void EmitBaseConstants(Function &F, User *U, Instruction *Base,
                         Constant *Offset, ConstantInt *OriginalConstant);
  bool EmitBaseConstants(Function &F);
  bool OptimizeConstants(Function &F);
};
}

char ConstantHoisting::ID = 0;
INITIALIZE_PASS_BEGIN(ConstantHoisting, "consthoist", "Constant Hoisting",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_END(ConstantHoisting, "consthoist", "Constant Hoisting",
                    false, false)

FunctionPass *llvm::createConstantHoistingPass() {
  return new ConstantHoisting();
}

/// \brief Perform the constant hoisting optimization for the given function.
bool ConstantHoisting::runOnFunction(Function &F) {
  DEBUG(dbgs() << "********** Constant Hoisting **********\n");
  DEBUG(dbgs() << "********** Function: " << F.getName() << '\n');

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  TTI = &getAnalysis<TargetTransformInfo>();

  return OptimizeConstants(F);
}

void ConstantHoisting::CollectConstant(User * U, unsigned Opcode,
                                       Intrinsic::ID IID, ConstantInt *C) {
  unsigned Cost;
  if (Opcode)
    Cost = TTI->getIntImmCost(Opcode, C->getValue(), C->getType());
  else
    Cost = TTI->getIntImmCost(IID, C->getValue(), C->getType());

  if (Cost > TargetTransformInfo::TCC_Basic) {
    ConstantCandidate &CC = ConstantMap[C];
    CC.CumulativeCost += Cost;
    CC.Uses.push_back(U);
  }
}

/// \brief Scan the instruction or constant expression for expensive integer
/// constants and record them in the constant map.
void ConstantHoisting::CollectConstants(Instruction *I) {
  unsigned Opcode = 0;
  Intrinsic::ID IID = Intrinsic::not_intrinsic;
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
    IID = II->getIntrinsicID();
  else
    Opcode = I->getOpcode();

  // Scan all operands.
  for (User::op_iterator O = I->op_begin(), E = I->op_end(); O != E; ++O) {
    if (ConstantInt *C = dyn_cast<ConstantInt>(O)) {
      CollectConstant(I, Opcode, IID, C);
      continue;
    }
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(O)) {
      // We only handle constant cast expressions.
      if (!CE->isCast())
        continue;

      if (ConstantInt *C = dyn_cast<ConstantInt>(CE->getOperand(0))) {
        // Ignore the cast expression and use the opcode of the instruction.
        CollectConstant(CE, Opcode, IID, C);
        continue;
      }
    }
  }
}

/// \brief Collect all integer constants in the function that cannot be folded
/// into an instruction itself.
void ConstantHoisting::CollectConstants(Function &F) {
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      CollectConstants(I);
}

/// \brief Compare function for sorting integer constants by type and by value
/// within a type in ConstantMaps.
static bool
ConstantMapLessThan(const std::pair<ConstantInt *, ConstantCandidate> &LHS,
                    const std::pair<ConstantInt *, ConstantCandidate> &RHS) {
  if (LHS.first->getType() == RHS.first->getType())
    return LHS.first->getValue().ult(RHS.first->getValue());
  else
    return LHS.first->getType()->getBitWidth() <
           RHS.first->getType()->getBitWidth();
}

/// \brief Find the base constant within the given range and rebase all other
/// constants with respect to the base constant.
void ConstantHoisting::FindAndMakeBaseConstant(ConstantMapType::iterator S,
                                               ConstantMapType::iterator E) {
  ConstantMapType::iterator MaxCostItr = S;
  unsigned NumUses = 0;
  // Use the constant that has the maximum cost as base constant.
  for (ConstantMapType::iterator I = S; I != E; ++I) {
    NumUses += I->second.Uses.size();
    if (I->second.CumulativeCost > MaxCostItr->second.CumulativeCost)
      MaxCostItr = I;
  }

  // Don't hoist constants that have only one use.
  if (NumUses <= 1)
    return;

  ConstantInfo CI;
  CI.BaseConstant = MaxCostItr->first;
  Type *Ty = CI.BaseConstant->getType();
  // Rebase the constants with respect to the base constant.
  for (ConstantMapType::iterator I = S; I != E; ++I) {
    APInt Diff = I->first->getValue() - CI.BaseConstant->getValue();
    ConstantInfo::RebasedConstantInfo RCI;
    RCI.OriginalConstant = I->first;
    RCI.Offset = ConstantInt::get(Ty, Diff);
    RCI.Uses = llvm_move(I->second.Uses);
    CI.RebasedConstants.push_back(RCI);
  }
  Constants.push_back(CI);
}

/// \brief Finds and combines constants that can be easily rematerialized with
/// an add from a common base constant.
void ConstantHoisting::FindBaseConstants() {
  // Sort the constants by value and type. This invalidates the mapping.
  std::sort(ConstantMap.begin(), ConstantMap.end(), ConstantMapLessThan);

  // Simple linear scan through the sorted constant map for viable merge
  // candidates.
  ConstantMapType::iterator MinValItr = ConstantMap.begin();
  for (ConstantMapType::iterator I = llvm::next(ConstantMap.begin()),
       E = ConstantMap.end(); I != E; ++I) {
    if (MinValItr->first->getType() == I->first->getType()) {
      // Check if the constant is in range of an add with immediate.
      APInt Diff = I->first->getValue() - MinValItr->first->getValue();
      if ((Diff.getBitWidth() <= 64) &&
          TTI->isLegalAddImmediate(Diff.getSExtValue()))
        continue;
    }
    // We either have now a different constant type or the constant is not in
    // range of an add with immediate anymore.
    FindAndMakeBaseConstant(MinValItr, I);
    // Start a new base constant search.
    MinValItr = I;
  }
  // Finalize the last base constant search.
  FindAndMakeBaseConstant(MinValItr, ConstantMap.end());
}

/// \brief Records the basic block of the instruction or all basic blocks of the
/// users of the constant expression.
static void CollectBasicBlocks(SmallPtrSet<BasicBlock *, 4> &BBs, Function &F,
                               User *U) {
  if (Instruction *I = dyn_cast<Instruction>(U))
    BBs.insert(I->getParent());
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U))
    // Find all users of this constant expression.
    for (Value::use_iterator UU = CE->use_begin(), E = CE->use_end();
         UU != E; ++UU)
      // Only record users that are instructions. We don't want to go down a
      // nested constant expression chain. Also check if the instruction is even
      // in the current function.
      if (Instruction *I = dyn_cast<Instruction>(*UU))
        if(I->getParent()->getParent() == &F)
          BBs.insert(I->getParent());
}

/// \brief Find an insertion point that dominates all uses.
Instruction *ConstantHoisting::
FindConstantInsertionPoint(Function &F, const ConstantInfo &CI) const {
  BasicBlock *Entry = &F.getEntryBlock();

  // Collect all basic blocks.
  SmallPtrSet<BasicBlock *, 4> BBs;
  ConstantInfo::RebasedConstantListType::const_iterator RCI, RCE;
  for (RCI = CI.RebasedConstants.begin(), RCE = CI.RebasedConstants.end();
       RCI != RCE; ++RCI)
    for (SmallVectorImpl<User *>::const_iterator U = RCI->Uses.begin(),
         E = RCI->Uses.end(); U != E; ++U)
        CollectBasicBlocks(BBs, F, *U);

  if (BBs.count(Entry))
    return Entry->getFirstInsertionPt();

  while (BBs.size() >= 2) {
    BasicBlock *BB, *BB1, *BB2;
    BB1 = *BBs.begin();
    BB2 = *llvm::next(BBs.begin());
    BB = DT->findNearestCommonDominator(BB1, BB2);
    if (BB == Entry)
      return Entry->getFirstInsertionPt();
    BBs.erase(BB1);
    BBs.erase(BB2);
    BBs.insert(BB);
  }
  assert((BBs.size() == 1) && "Expected only one element.");
  return (*BBs.begin())->getFirstInsertionPt();
}

/// \brief Find the instruction we should insert the constant materialization
/// before.
static Instruction *getMatInsertPt(Instruction *I, const DominatorTree *DT) {
  if (!isa<PHINode>(I) && !isa<LandingPadInst>(I)) // Simple case.
    return I;

  // We can't insert directly before a phi node or landing pad. Insert before
  // the terminator of the dominating block.
  assert(&I->getParent()->getParent()->getEntryBlock() != I->getParent() &&
         "PHI or landing pad in entry block!");
  BasicBlock *IDom = DT->getNode(I->getParent())->getIDom()->getBlock();
  return IDom->getTerminator();
}

/// \brief Emit materialization code for all rebased constants and update their
/// users.
void ConstantHoisting::EmitBaseConstants(Function &F, User *U,
                                         Instruction *Base, Constant *Offset,
                                         ConstantInt *OriginalConstant) {
  if (Instruction *I = dyn_cast<Instruction>(U)) {
    Instruction *Mat = Base;
    if (!Offset->isNullValue()) {
      Mat = BinaryOperator::Create(Instruction::Add, Base, Offset,
                                   "const_mat", getMatInsertPt(I, DT));

      // Use the same debug location as the instruction we are about to update.
      Mat->setDebugLoc(I->getDebugLoc());

      DEBUG(dbgs() << "Materialize constant (" << *Base->getOperand(0)
                   << " + " << *Offset << ") in BB "
                   << I->getParent()->getName() << '\n' << *Mat << '\n');
    }
    DEBUG(dbgs() << "Update: " << *I << '\n');
    I->replaceUsesOfWith(OriginalConstant, Mat);
    DEBUG(dbgs() << "To: " << *I << '\n');
    return;
  }
  assert(isa<ConstantExpr>(U) && "Expected a ConstantExpr.");
  ConstantExpr *CE = cast<ConstantExpr>(U);
  SmallVector<std::pair<Instruction *, Instruction *>, 8> WorkList;
  DEBUG(dbgs() << "Visit ConstantExpr " << *CE << '\n');
  for (Value::use_iterator UU = CE->use_begin(), E = CE->use_end();
       UU != E; ++UU) {
    DEBUG(dbgs() << "Check user "; UU->print(dbgs()); dbgs() << '\n');
    // We only handel instructions here and won't walk down a ConstantExpr chain
    // to replace all ConstExpr with instructions.
    if (Instruction *I = dyn_cast<Instruction>(*UU)) {
      // Only update constant expressions in the current function.
      if (I->getParent()->getParent() != &F) {
        DEBUG(dbgs() << "Not in the same function - skip.\n");
        continue;
      }

      Instruction *Mat = Base;
      Instruction *InsertBefore = getMatInsertPt(I, DT);
      if (!Offset->isNullValue()) {
        Mat = BinaryOperator::Create(Instruction::Add, Base, Offset,
                                     "const_mat", InsertBefore);

        // Use the same debug location as the instruction we are about to
        // update.
        Mat->setDebugLoc(I->getDebugLoc());

        DEBUG(dbgs() << "Materialize constant (" << *Base->getOperand(0)
                     << " + " << *Offset << ") in BB "
                     << I->getParent()->getName() << '\n' << *Mat << '\n');
      }
      Instruction *ICE = CE->getAsInstruction();
      ICE->replaceUsesOfWith(OriginalConstant, Mat);
      ICE->insertBefore(InsertBefore);

      // Use the same debug location as the instruction we are about to update.
      ICE->setDebugLoc(I->getDebugLoc());

      WorkList.push_back(std::make_pair(I, ICE));
    } else {
      DEBUG(dbgs() << "Not an instruction - skip.\n");
    }
  }
  SmallVectorImpl<std::pair<Instruction *, Instruction *> >::iterator I, E;
  for (I = WorkList.begin(), E = WorkList.end(); I != E; ++I) {
    DEBUG(dbgs() << "Create instruction: " << *I->second << '\n');
    DEBUG(dbgs() << "Update: " << *I->first << '\n');
    I->first->replaceUsesOfWith(CE, I->second);
    DEBUG(dbgs() << "To: " << *I->first << '\n');
  }
}

/// \brief Hoist and hide the base constant behind a bitcast and emit
/// materialization code for derived constants.
bool ConstantHoisting::EmitBaseConstants(Function &F) {
  bool MadeChange = false;
  SmallVectorImpl<ConstantInfo>::iterator CI, CE;
  for (CI = Constants.begin(), CE = Constants.end(); CI != CE; ++CI) {
    // Hoist and hide the base constant behind a bitcast.
    Instruction *IP = FindConstantInsertionPoint(F, *CI);
    IntegerType *Ty = CI->BaseConstant->getType();
    Instruction *Base = new BitCastInst(CI->BaseConstant, Ty, "const", IP);
    DEBUG(dbgs() << "Hoist constant (" << *CI->BaseConstant << ") to BB "
                 << IP->getParent()->getName() << '\n');
    NumConstantsHoisted++;

    // Emit materialization code for all rebased constants.
    ConstantInfo::RebasedConstantListType::iterator RCI, RCE;
    for (RCI = CI->RebasedConstants.begin(), RCE = CI->RebasedConstants.end();
         RCI != RCE; ++RCI) {
      NumConstantsRebased++;
      for (SmallVectorImpl<User *>::iterator U = RCI->Uses.begin(),
           E = RCI->Uses.end(); U != E; ++U)
        EmitBaseConstants(F, *U, Base, RCI->Offset, RCI->OriginalConstant);
    }

    // Use the same debug location as the last user of the constant.
    assert(!Base->use_empty() && "The use list is empty!?");
    assert(isa<Instruction>(Base->use_back()) &&
           "All uses should be instructions.");
    Base->setDebugLoc(cast<Instruction>(Base->use_back())->getDebugLoc());

    // Correct for base constant, which we counted above too.
    NumConstantsRebased--;
    MadeChange = true;
  }
  return MadeChange;
}

/// \brief Optimize expensive integer constants in the given function.
bool ConstantHoisting::OptimizeConstants(Function &F) {
  bool MadeChange = false;

  // Collect all constant candidates.
  CollectConstants(F);

  // There are no constants to worry about.
  if (ConstantMap.empty())
    return MadeChange;

  // Combine constants that can be easily materialized with an add from a common
  // base constant.
  FindBaseConstants();

  // Finaly hoist the base constant and emit materializating code for dependent
  // constants.
  MadeChange |= EmitBaseConstants(F);

  ConstantMap.clear();
  Constants.clear();

  return MadeChange;
}
