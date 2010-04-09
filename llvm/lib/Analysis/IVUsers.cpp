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
static RegisterPass<IVUsers>
X("iv-users", "Induction Variable Users", false, true);

Pass *llvm::createIVUsersPass() {
  return new IVUsers();
}

/// CollectSubexprs - Split S into subexpressions which can be pulled out into
/// separate registers.
static void CollectSubexprs(const SCEV *S,
                            SmallVectorImpl<const SCEV *> &Ops,
                            ScalarEvolution &SE) {
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    // Break out add operands.
    for (SCEVAddExpr::op_iterator I = Add->op_begin(), E = Add->op_end();
         I != E; ++I)
      CollectSubexprs(*I, Ops, SE);
    return;
  } else if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    // Split a non-zero base out of an addrec.
    if (!AR->getStart()->isZero()) {
      CollectSubexprs(AR->getStart(), Ops, SE);
      CollectSubexprs(SE.getAddRecExpr(SE.getIntegerSCEV(0, AR->getType()),
                                       AR->getStepRecurrence(SE),
                                       AR->getLoop()), Ops, SE);
      return;
    }
  }

  // Otherwise use the value itself.
  Ops.push_back(S);
}

/// isInteresting - Test whether the given expression is "interesting" when
/// used by the given expression, within the context of analyzing the
/// given loop.
static bool isInteresting(const SCEV *S, const Instruction *I, const Loop *L) {
  // Anything loop-invariant is interesting.
  if (!isa<SCEVUnknown>(S) && S->isLoopInvariant(L))
    return true;

  // An addrec is interesting if it's affine or if it has an interesting start.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    // Keep things simple. Don't touch loop-variant strides.
    if (AR->getLoop() == L)
      return AR->isAffine() || !L->contains(I);
    // Otherwise recurse to see if the start value is interesting.
    return isInteresting(AR->getStart(), I, L);
  }

  // An add is interesting if any of its operands is.
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    for (SCEVAddExpr::op_iterator OI = Add->op_begin(), OE = Add->op_end();
         OI != OE; ++OI)
      if (isInteresting(*OI, I, L))
        return true;
    return false;
  }

  // Nothing else is interesting here.
  return false;
}

/// AddUsersIfInteresting - Inspect the specified instruction.  If it is a
/// reducible SCEV, recursively add its users to the IVUsesByStride set and
/// return true.  Otherwise, return false.
bool IVUsers::AddUsersIfInteresting(Instruction *I) {
  if (!SE->isSCEVable(I->getType()))
    return false;   // Void and FP expressions cannot be reduced.

  // LSR is not APInt clean, do not touch integers bigger than 64-bits.
  if (SE->getTypeSizeInBits(I->getType()) > 64)
    return false;

  if (!Processed.insert(I))
    return true;    // Instruction already handled.

  // Get the symbolic expression for this instruction.
  const SCEV *ISE = SE->getSCEV(I);
  if (isa<SCEVCouldNotCompute>(ISE)) return false;

  // If we've come to an uninteresting expression, stop the traversal and
  // call this a user.
  if (!isInteresting(ISE, I, L))
    return false;

  SmallPtrSet<Instruction *, 4> UniqueUsers;
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    if (!UniqueUsers.insert(User))
      continue;

    // Do not infinitely recurse on PHI nodes.
    if (isa<PHINode>(User) && Processed.count(User))
      continue;

    // Descend recursively, but not into PHI nodes outside the current loop.
    // It's important to see the entire expression outside the loop to get
    // choices that depend on addressing mode use right, although we won't
    // consider references outside the loop in all cases.
    // If User is already in Processed, we don't want to recurse into it again,
    // but do want to record a second reference in the same instruction.
    bool AddUserToIVUsers = false;
    if (LI->getLoopFor(User->getParent()) != L) {
      if (isa<PHINode>(User) || Processed.count(User) ||
          !AddUsersIfInteresting(User)) {
        DEBUG(dbgs() << "FOUND USER in other loop: " << *User << '\n'
                     << "   OF SCEV: " << *ISE << '\n');
        AddUserToIVUsers = true;
      }
    } else if (Processed.count(User) ||
               !AddUsersIfInteresting(User)) {
      DEBUG(dbgs() << "FOUND USER: " << *User << '\n'
                   << "   OF SCEV: " << *ISE << '\n');
      AddUserToIVUsers = true;
    }

    if (AddUserToIVUsers) {
      // Okay, we found a user that we cannot reduce.
      IVUses.push_back(new IVStrideUse(this, ISE, User, I));
      IVStrideUse &NewUse = IVUses.back();
      // Transform the expression into a normalized form.
      NewUse.Expr =
        TransformForPostIncUse(NormalizeAutodetect, NewUse.Expr,
                               User, I,
                               NewUse.PostIncLoops,
                               *SE, *DT);
      DEBUG(dbgs() << "   NORMALIZED TO: " << *NewUse.Expr << '\n');
    }
  }
  return true;
}

IVStrideUse &IVUsers::AddUser(const SCEV *Expr,
                              Instruction *User, Value *Operand) {
  IVUses.push_back(new IVStrideUse(this, Expr, User, Operand));
  return IVUses.back();
}

IVUsers::IVUsers()
 : LoopPass(&ID) {
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

/// getReplacementExpr - Return a SCEV expression which computes the
/// value of the OperandValToReplace of the given IVStrideUse.
const SCEV *IVUsers::getReplacementExpr(const IVStrideUse &U) const {
  PostIncLoopSet &Loops = const_cast<PostIncLoopSet &>(U.PostIncLoops);
  return TransformForPostIncUse(Denormalize, U.getExpr(),
                                U.getUser(), U.getOperandValToReplace(),
                                Loops, *SE, *DT);
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
    OS << " = "
       << *getReplacementExpr(*UI);
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

const SCEV *IVStrideUse::getStride(const Loop *L) const {
  if (const SCEVAddRecExpr *AR = findAddRecForLoop(getExpr(), L))
    return AR->getStepRecurrence(*Parent->SE);
  return 0;
}

void IVStrideUse::transformToPostInc(const Loop *L) {
  PostIncLoopSet Loops;
  Loops.insert(L);
  Expr = TransformForPostIncUse(Normalize, Expr,
                                getUser(), getOperandValToReplace(),
                                Loops, *Parent->SE, *Parent->DT);
  PostIncLoops.insert(L);
}

void IVStrideUse::deleted() {
  // Remove this user from the list.
  Parent->IVUses.erase(this);
  // this now dangles!
}
