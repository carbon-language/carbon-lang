//===-- AffineSCEVIterator.h - Iterate the SCEV in an affine way -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The iterator can be used to iterate over the affine component of the SCEV
// expression.
//
//===----------------------------------------------------------------------===//

#ifndef AFFINE_SCEV_ITERATOR_H
#define AFFINE_SCEV_ITERATOR_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include <map>

#include "llvm/ADT/SmallVector.h"

using namespace llvm;

namespace polly {

/// @brief The itertor transform the scalar expressions to the form of sum of
/// (constant * variable)s, and return the variable/constant pairs one by one
/// on the fly.
///
/// For example, we can write SCEV:
///      {{%x,+,sizeof(i32)}<%bb2.preheader>,+,(4 * sizeof(i32))}<%bb1>
/// in affine form:
///      (4 * sizeof(i32)) * %indvar + sizeof(i32) * %0 + 1 * %x + 0 * 1
/// so we can get the follow pair from the iterator:
///      {%indvar, (4 * sizeof(i32))}, {%0, sizeof(i32)}, {%x, 1} and {1, 0}
/// where %indvar is the induction variable of loop %bb1 and %0 is the induction
/// variable of loop %bb2.preheader.
///
/// In the returned pair,
/// The "first" field is the variable part, the "second" field constant part.
/// And the translation part of the expression will always return last.
///
class AffineSCEVIterator : public std::iterator<std::forward_iterator_tag,
                            std::pair<const SCEV*, const SCEV*>, ptrdiff_t>,
                            SCEVVisitor<AffineSCEVIterator,
                                        std::pair<const SCEV*, const SCEV*> >
                            {
  typedef std::iterator<std::forward_iterator_tag,
                       std::pair<const SCEV*, const SCEV*>, ptrdiff_t> super;

  friend struct llvm::SCEVVisitor<AffineSCEVIterator,
                            std::pair<const SCEV*, const SCEV*> >;

  ScalarEvolution *SE;
public:
  typedef super::value_type value_type;
  typedef super::pointer    ptr_type;
  typedef AffineSCEVIterator Self;
private:
  typedef SCEVNAryExpr::op_iterator scev_op_it;

  // The stack help us remember the SCEVs that not visit yet.
  SmallVector<const SCEV*, 8> visitStack;

  // The current value of this iterator.
  value_type val;

  const SCEVConstant* getSCEVOne(const SCEV* S) const {
    return cast<SCEVConstant>(SE->getConstant(S->getType(), 1));
  }

  //===-------------------------------------------------------------------===//
  /// Functions for SCEVVisitor.
  ///
  /// These function compute the constant part and variable part of the SCEV,
  /// and return them in a std::pair, where the first field is the variable,
  /// and the second field is the constant.
  ///
  value_type visitConstant(const SCEVConstant *S) {
    return std::make_pair(getSCEVOne(S), S);
  }

  value_type visitUnknown(const SCEVUnknown* S) {
    Type *AllocTy;
    Constant *FieldNo;
    // We treat these as constant.
    if (S->isSizeOf  (AllocTy) ||
        S->isAlignOf (AllocTy) ||
        S->isOffsetOf(AllocTy, FieldNo))
      return std::make_pair(getSCEVOne(S), S);

    return std::make_pair(S, getSCEVOne(S));
  }

  value_type visitMulExpr(const SCEVMulExpr* S) {
    SmallVector<const SCEV*, 4> Coeffs, Variables;

    // Do not worry about the Constant * Variable * (Variable + Variable)
    // MulExpr, we will never get a affine expression from it, so we just
    // leave it there.
    for (scev_op_it I = S->op_begin(), E = S->op_end(); I != E; ++I) {
      // Get the constant part and the variable part of each operand.
      value_type res = visit(*I);

      Coeffs.push_back(res.second);
      Variables.push_back(res.first);
    }

    // Get the constant part and variable part of this MulExpr by
    // multiply them together.
    const SCEV *Coeff = SE->getMulExpr(Coeffs);
    // There maybe "sizeof" and others.
    // TODO: Assert the allowed coeff type.
    // assert(Coeff && "Expect Coeff to be a const!");

    const SCEV *Var = SE->getMulExpr(Variables);

    return std::make_pair(Var, Coeff);
  }

  value_type visitCastExpr(const SCEVCastExpr *S) {
    return std::make_pair(S, getSCEVOne(S));
  }

  value_type visitTruncateExpr(const SCEVTruncateExpr *S) {
    return visitCastExpr(S);
  }

  value_type visitZeroExtendExpr(const SCEVZeroExtendExpr *S) {
    return visitCastExpr(S);
  }

  value_type visitSignExtendExpr(const SCEVSignExtendExpr *S) {
    return visitCastExpr(S);
  }

  value_type visitAddExpr(const SCEVAddExpr *S) {
    // AddExpr will handled out in visit Next;
    return std::make_pair(S, getSCEVOne(S));
  }

  value_type visitAddRecExpr(const SCEVAddRecExpr *S) {
    // AddRecExpr will handled out in visit Next;
    return std::make_pair(S, getSCEVOne(S));
  }

  value_type visitUDivExpr(const SCEVUDivExpr *S) {
    return std::make_pair(S, getSCEVOne(S));
  }

  value_type visitSMaxExpr(const SCEVSMaxExpr *S) {
    return std::make_pair(S, getSCEVOne(S));
  }

  value_type visitUMaxExpr(const SCEVUMaxExpr *S) {
    return std::make_pair(S, getSCEVOne(S));
  }

  /// Get the next {variable, constant} pair of the SCEV.
  value_type visitNext() {
    value_type ret(0, 0);

    if (visitStack.empty())
      return ret;
    const SCEV* nextS = visitStack.back();

    if (const SCEVAddRecExpr *ARec = dyn_cast<SCEVAddRecExpr>(nextS)){
      // Visiting the AddRec, check if its Affine;
      PHINode *IV = ARec->getLoop()->getCanonicalInductionVariable();
      // Only decompose the AddRec, if the loop has a canonical induction
      // variable.
      if (ARec->isAffine() && IV != 0) {
        ret = visit(ARec->getStepRecurrence(*SE));
        if (isa<SCEVConstant>(ret.first)) { // If the step is constant.
          const SCEV *Start = ARec->getStart();
          visitStack.back() = Start;

          // The AddRec is expect to be decomposed to
          //
          // | start + step * {1, +, 1}<loop>
          //
          // Now we get the {1, +, 1}<loop> part.
          ret.first = SE->getSCEV(IV);

          // Push CouldNotCompute to take the place.
          visitStack.push_back(SE->getCouldNotCompute());

          return ret;
        }
        // The step is not a constant. Then this AddRec is not Affine or
        // no canonical induction variable found.
        // Fall through.
      }
    }

    // Get the constant part and variable part of the SCEV.
    ret = visit(nextS);

    // If the reach the last constant
    if (isa<SCEVConstant>(ret.first) && (visitStack.size() != 1)) {
      // Else, merge all constant component, we will output it at last.
      visitStack.front() = SE->getAddExpr(visitStack.front(), ret.second);
      //assert(isa<SCEVConstant>(visitStack.front().first));
      // Pop the top constant, because it already merged into the bottom of the Stack
      // and output it last.
      visitStack.pop_back();
      // Try again.
      return visitNext();
    }
    // Not a constant or Stack not empty
    // If ret is in (xxx) * AddExpr form, we will decompose the AddExpr
    else if (const SCEVAddExpr *AddExpr = dyn_cast<SCEVAddExpr>(ret.first)) {
      // Pop the current SCEV, we will decompose it.
      visitStack.pop_back();
      assert(AddExpr->getNumOperands() && "AddExpr without operand?");
      for (scev_op_it I = AddExpr->op_begin(), E = AddExpr->op_end(); I != E; ++I){
        visitStack.push_back(SE->getMulExpr(ret.second, *I));
      }
      // Try again with the new SCEV.
      return visitNext();
    }

    return ret;
  }

public:

  /// @brief Create the iterator from a SCEV and the ScalarEvolution analysis.
  AffineSCEVIterator(const SCEV* S, ScalarEvolution *se ) : SE(se) {
    // Dont iterate CouldNotCompute.
    if (isa<SCEVCouldNotCompute>(S))
      return;

    Type *Ty = S->getType();

    // Init the constant component.
    visitStack.push_back(SE->getConstant(Ty, 0));

    // Get the first affine component.
    visitStack.push_back(S);
    val = visitNext();
  }

  /// @brief Create an end iterator.
  inline AffineSCEVIterator() {}

  inline bool operator==(const Self& x) const {
    return visitStack == x.visitStack;
  }
  inline bool operator!=(const Self& x) const { return !operator==(x); }

  /// @brief Return the current (constant * variable) component of the SCEV.
  ///
  /// @return The "first" field of the pair is the variable part,
  ///         the "second" field of the pair is the constant part.
  inline value_type operator*() const {
    assert(val.first && val.second && "Cant dereference iterator!");
    return val;
  }

  inline const value_type* operator->() const {
    assert(val.first && val.second && "Cant dereference iterator!");
    return &val;
  }

  inline Self& operator++() {   // Preincrement
    assert(!visitStack.empty() && "Cant ++ iterator!");
    // Pop the last SCEV.
    visitStack.pop_back();
    val = visitNext();
    return *this;
  }

  inline Self operator++(int) { // Postincrement
    Self tmp = *this; ++*this; return tmp;
  }
};

inline static AffineSCEVIterator affine_begin(const SCEV* S, ScalarEvolution *SE) {
  return AffineSCEVIterator(S, SE);
}

inline static AffineSCEVIterator affine_end() {
  return AffineSCEVIterator();
}

} // end namespace polly
#endif
