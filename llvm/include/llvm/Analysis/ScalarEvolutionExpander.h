//===---- llvm/Analysis/ScalarEvolutionExpander.h - SCEV Exprs --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the classes used to generate code from scalar expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTION_EXPANDER_H
#define LLVM_ANALYSIS_SCALAREVOLUTION_EXPANDER_H

#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TargetFolder.h"
#include <set>

namespace llvm {
  /// SCEVExpander - This class uses information about analyze scalars to
  /// rewrite expressions in canonical form.
  ///
  /// Clients should create an instance of this class when rewriting is needed,
  /// and destroy it when finished to allow the release of the associated 
  /// memory.
  struct SCEVExpander : public SCEVVisitor<SCEVExpander, Value*> {
    ScalarEvolution &SE;
    std::map<std::pair<const SCEV *, Instruction *>, AssertingVH<Value> >
      InsertedExpressions;
    std::set<Value*> InsertedValues;

    typedef IRBuilder<true, TargetFolder> BuilderType;
    BuilderType Builder;

    friend struct SCEVVisitor<SCEVExpander, Value*>;
  public:
    explicit SCEVExpander(ScalarEvolution &se)
      : SE(se), Builder(se.getContext(),
                        TargetFolder(se.TD, se.getContext())) {}

    /// clear - Erase the contents of the InsertedExpressions map so that users
    /// trying to expand the same expression into multiple BasicBlocks or
    /// different places within the same BasicBlock can do so.
    void clear() { InsertedExpressions.clear(); }

    /// getOrInsertCanonicalInductionVariable - This method returns the
    /// canonical induction variable of the specified type for the specified
    /// loop (inserting one if there is none).  A canonical induction variable
    /// starts at zero and steps by one on each iteration.
    Value *getOrInsertCanonicalInductionVariable(const Loop *L, const Type *Ty);

    /// expandCodeFor - Insert code to directly compute the specified SCEV
    /// expression into the program.  The inserted code is inserted into the
    /// specified block.
    Value *expandCodeFor(const SCEV *SH, const Type *Ty, Instruction *IP) {
      Builder.SetInsertPoint(IP->getParent(), IP);
      return expandCodeFor(SH, Ty);
    }

  private:
    LLVMContext &getContext() const { return SE.getContext(); }
    
    /// InsertBinop - Insert the specified binary operator, doing a small amount
    /// of work to avoid inserting an obviously redundant operation.
    Value *InsertBinop(Instruction::BinaryOps Opcode, Value *LHS, Value *RHS);

    /// InsertNoopCastOfTo - Insert a cast of V to the specified type,
    /// which must be possible with a noop cast, doing what we can to
    /// share the casts.
    Value *InsertNoopCastOfTo(Value *V, const Type *Ty);

    /// expandAddToGEP - Expand a SCEVAddExpr with a pointer type into a GEP
    /// instead of using ptrtoint+arithmetic+inttoptr.
    Value *expandAddToGEP(const SCEV *const *op_begin,
                          const SCEV *const *op_end,
                          const PointerType *PTy, const Type *Ty, Value *V);

    Value *expand(const SCEV *S);

    /// expandCodeFor - Insert code to directly compute the specified SCEV
    /// expression into the program.  The inserted code is inserted into the
    /// SCEVExpander's current insertion point. If a type is specified, the
    /// result will be expanded to have that type, with a cast if necessary.
    Value *expandCodeFor(const SCEV *SH, const Type *Ty = 0);

    /// isInsertedInstruction - Return true if the specified instruction was
    /// inserted by the code rewriter.  If so, the client should not modify the
    /// instruction.
    bool isInsertedInstruction(Instruction *I) const {
      return InsertedValues.count(I);
    }

    Value *visitConstant(const SCEVConstant *S) {
      return S->getValue();
    }

    Value *visitTruncateExpr(const SCEVTruncateExpr *S);

    Value *visitZeroExtendExpr(const SCEVZeroExtendExpr *S);

    Value *visitSignExtendExpr(const SCEVSignExtendExpr *S);

    Value *visitAddExpr(const SCEVAddExpr *S);

    Value *visitMulExpr(const SCEVMulExpr *S);

    Value *visitUDivExpr(const SCEVUDivExpr *S);

    Value *visitAddRecExpr(const SCEVAddRecExpr *S);

    Value *visitSMaxExpr(const SCEVSMaxExpr *S);

    Value *visitUMaxExpr(const SCEVUMaxExpr *S);

    Value *visitFieldOffsetExpr(const SCEVFieldOffsetExpr *S);

    Value *visitAllocSizeExpr(const SCEVAllocSizeExpr *S);

    Value *visitUnknown(const SCEVUnknown *S) {
      return S->getValue();
    }
  };
}

#endif

