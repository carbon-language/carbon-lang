//===- IVUsers.cpp - Induction Variable Users -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements bookkeeping for "interesting" users of expressions
// computed from induction variables.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "iv-users"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Assembly/AsmAnnotationWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

char IVUsers::ID = 0;
INITIALIZE_PASS(IVUsers, "iv-users", "Induction Variable Users", false, true);

Pass *llvm::createIVUsersPass() {
  return new IVUsers();
}

/// findInterestingAddRec - Test whether the given expression is interesting.
/// Return the addrec with the current loop which makes it interesting, or
/// null if it is not interesting.
const SCEVAddRecExpr *IVUsers::findInterestingAddRec(const SCEV *S) const {
  // An addrec is interesting if it's affine or if it has an interesting start.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    // Keep things simple. Don't touch loop-variant strides.
    if (AR->getLoop() == L)
      return AR;
    // We don't yet know how to do effective SCEV expansions for addrecs
    // with interesting steps.
    if (findInterestingAddRec(AR->getStepRecurrence(*SE)))
      return 0;
    // Otherwise recurse to see if the start value is interesting.
    return findInterestingAddRec(AR->getStart());
  }

  // An add is interesting if exactly one of its operands is interesting.
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    for (SCEVAddExpr::op_iterator OI = Add->op_begin(), OE = Add->op_end();
         OI != OE; ++OI)
      if (const SCEVAddRecExpr *AR = findInterestingAddRec(*OI))
        return AR;
    return 0;
  }

  // Nothing else is interesting here.
  return 0;
}

bool IVUsers::isInterestingUser(const Instruction *User) const {
  // Void and FP expressions cannot be reduced.
  if (!SE->isSCEVable(User->getType()))
    return false;

  // LSR is not APInt clean, do not touch integers bigger than 64-bits.
  if (SE->getTypeSizeInBits(User->getType()) > 64)
    return false;

  // Don't descend into PHI nodes outside the current loop.
  if (LI->getLoopFor(User->getParent()) != L &&
      isa<PHINode>(User))
    return false;

  // Otherwise, it may be interesting.
  return true;
}

/// AddUsersIfInteresting - Inspect the specified instruction.  If it is a
/// reducible SCEV, recursively add its users to the IVUsesByStride set and
/// return true.  Otherwise, return false.
void IVUsers::AddUsersIfInteresting(Instruction *I) {
  // Stop if we've seen this before.
  if (!Processed.insert(I))
    return;

  // If this PHI node is not SCEVable, ignore it.
  if (!SE->isSCEVable(I->getType()))
    return;

  // If this PHI node is not an addrec for this loop, ignore it.
  const SCEVAddRecExpr *Expr = findInterestingAddRec(SE->getSCEV(I));
  if (!Expr)
    return;

  // Walk the def-use graph.
  SmallVector<std::pair<Instruction *, const SCEVAddRecExpr *>, 16> Worklist;
  Worklist.push_back(std::make_pair(I, Expr));
  do {
    std::pair<Instruction *, const SCEVAddRecExpr *> P =
      Worklist.pop_back_val();
    Instruction *Op = P.first;
    const SCEVAddRecExpr *OpAR = P.second;

    // Visit Op's users.
    SmallPtrSet<Instruction *, 8> VisitedUsers;
    for (Value::use_iterator UI = Op->use_begin(), E = Op->use_end();
         UI != E; ++UI) {
      // Don't visit any individual user more than once.
      Instruction *User = cast<Instruction>(*UI);
      if (!VisitedUsers.insert(User))
        continue;

      // If it's an affine addrec (which we can pretty safely re-expand) inside
      // the loop, or a potentially non-affine addrec outside the loop (which
      // we can evaluate outside of the loop), follow it.
      if (OpAR->isAffine() || !L->contains(User)) {
        if (isInterestingUser(User)) {
          const SCEV *UserExpr = SE->getSCEV(User);

          if (const SCEVAddRecExpr *AR = findInterestingAddRec(UserExpr)) {
            // Interesting. Keep searching.
            if (Processed.insert(User))
              Worklist.push_back(std::make_pair(User, AR));
            continue;
          }
        }
      }

      // Otherwise, this is the point where the def-use chain
      // becomes uninteresting. Call it an IV User.
      AddUser(User, Op);
    }
  } while (!Worklist.empty());
}

IVStrideUse &IVUsers::AddUser(Instruction *User, Value *Operand) {
  IVUses.push_back(new IVStrideUse(this, User, Operand));
  IVStrideUse &NewUse = IVUses.back();

  // Auto-detect and remember post-inc loops for this expression.
  const SCEV *S = SE->getSCEV(Operand);
  (void)TransformForPostIncUse(NormalizeAutodetect,
                               S, User, Operand,
                               NewUse.PostIncLoops,
                               *SE, *DT);
  return NewUse;
}

IVUsers::IVUsers()
 : LoopPass(ID) {
}

void IVUsers::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfo>();
  AU.addRequired<DominatorTree>();
  AU.addRequired<ScalarEvolution>();
  AU.setPreservesAll();
}

bool IVUsers::runOnLoop(Loop *l, LPPassManager &LPM) {

  L = l;
  LI = &getAnalysis<LoopInfo>();
  DT = &getAnalysis<DominatorTree>();
  SE = &getAnalysis<ScalarEvolution>();

  // Find all uses of induction variables in this loop, and categorize
  // them by stride.  Start by finding all of the PHI nodes in the header for
  // this loop.  If they are induction variables, inspect their uses.
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I)
    AddUsersIfInteresting(I);

  return false;
}

void IVUsers::print(raw_ostream &OS, const Module *M) const {
  OS << "IV Users for loop ";
  WriteAsOperand(OS, L->getHeader(), false);
  if (SE->hasLoopInvariantBackedgeTakenCount(L)) {
    OS << " with backedge-taken count "
       << *SE->getBackedgeTakenCount(L);
  }
  OS << ":\n";

  // Use a default AssemblyAnnotationWriter to suppress the default info
  // comments, which aren't relevant here.
  AssemblyAnnotationWriter Annotator;
  for (ilist<IVStrideUse>::const_iterator UI = IVUses.begin(),
       E = IVUses.end(); UI != E; ++UI) {
    OS << "  ";
    WriteAsOperand(OS, UI->getOperandValToReplace(), false);
    OS << " = " << *getReplacementExpr(*UI);
    for (PostIncLoopSet::const_iterator
         I = UI->PostIncLoops.begin(),
         E = UI->PostIncLoops.end(); I != E; ++I) {
      OS << " (post-inc with loop ";
      WriteAsOperand(OS, (*I)->getHeader(), false);
      OS << ")";
    }
    OS << " in  ";
    UI->getUser()->print(OS, &Annotator);
    OS << '\n';
  }
}

void IVUsers::dump() const {
  print(dbgs());
}

void IVUsers::releaseMemory() {
  Processed.clear();
  IVUses.clear();
}

/// getReplacementExpr - Return a SCEV expression which computes the
/// value of the OperandValToReplace.
const SCEV *IVUsers::getReplacementExpr(const IVStrideUse &IU) const {
  return SE->getSCEV(IU.getOperandValToReplace());
}

/// getExpr - Return the expression for the use.
const SCEV *IVUsers::getExpr(const IVStrideUse &IU) const {
  return
    TransformForPostIncUse(Normalize, getReplacementExpr(IU),
                           IU.getUser(), IU.getOperandValToReplace(),
                           const_cast<PostIncLoopSet &>(IU.getPostIncLoops()),
                           *SE, *DT);
}

static const SCEVAddRecExpr *findAddRecForLoop(const SCEV *S, const Loop *L) {
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    if (AR->getLoop() == L)
      return AR;
    return findAddRecForLoop(AR->getStart(), L);
  }

  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    for (SCEVAddExpr::op_iterator I = Add->op_begin(), E = Add->op_end();
         I != E; ++I)
      if (const SCEVAddRecExpr *AR = findAddRecForLoop(*I, L))
        return AR;
    return 0;
  }

  return 0;
}

const SCEV *IVUsers::getStride(const IVStrideUse &IU, const Loop *L) const {
  if (const SCEVAddRecExpr *AR = findAddRecForLoop(getExpr(IU), L))
    return AR->getStepRecurrence(*SE);
  return 0;
}

void IVStrideUse::transformToPostInc(const Loop *L) {
  PostIncLoops.insert(L);
}

void IVStrideUse::deleted() {
  // Remove this user from the list.
  Parent->IVUses.erase(this);
  // this now dangles!
}
