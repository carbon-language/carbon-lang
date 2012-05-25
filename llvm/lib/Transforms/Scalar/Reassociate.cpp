//===- Reassociate.cpp - Reassociate binary expressions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass reassociates commutative expressions in an order that is designed
// to promote better constant propagation, GCSE, LICM, PRE, etc.
//
// For example: 4 + (x + 5) -> x + (4 + 5)
//
// In the implementation of this algorithm, constants are assigned rank = 0,
// function arguments are rank = 1, and other values are assigned ranks
// corresponding to the reverse post order traversal of current function
// (starting at 2), which effectively gives values in deep loops higher rank
// than values not in loops.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reassociate"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumChanged, "Number of insts reassociated");
STATISTIC(NumAnnihil, "Number of expr tree annihilated");
STATISTIC(NumFactor , "Number of multiplies factored");

namespace {
  struct ValueEntry {
    unsigned Rank;
    Value *Op;
    ValueEntry(unsigned R, Value *O) : Rank(R), Op(O) {}
  };
  inline bool operator<(const ValueEntry &LHS, const ValueEntry &RHS) {
    return LHS.Rank > RHS.Rank;   // Sort so that highest rank goes to start.
  }
}

#ifndef NDEBUG
/// PrintOps - Print out the expression identified in the Ops list.
///
static void PrintOps(Instruction *I, const SmallVectorImpl<ValueEntry> &Ops) {
  Module *M = I->getParent()->getParent()->getParent();
  dbgs() << Instruction::getOpcodeName(I->getOpcode()) << " "
       << *Ops[0].Op->getType() << '\t';
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    dbgs() << "[ ";
    WriteAsOperand(dbgs(), Ops[i].Op, false, M);
    dbgs() << ", #" << Ops[i].Rank << "] ";
  }
}
#endif

namespace {
  /// \brief Utility class representing a base and exponent pair which form one
  /// factor of some product.
  struct Factor {
    Value *Base;
    unsigned Power;

    Factor(Value *Base, unsigned Power) : Base(Base), Power(Power) {}

    /// \brief Sort factors by their Base.
    struct BaseSorter {
      bool operator()(const Factor &LHS, const Factor &RHS) {
        return LHS.Base < RHS.Base;
      }
    };

    /// \brief Compare factors for equal bases.
    struct BaseEqual {
      bool operator()(const Factor &LHS, const Factor &RHS) {
        return LHS.Base == RHS.Base;
      }
    };

    /// \brief Sort factors in descending order by their power.
    struct PowerDescendingSorter {
      bool operator()(const Factor &LHS, const Factor &RHS) {
        return LHS.Power > RHS.Power;
      }
    };

    /// \brief Compare factors for equal powers.
    struct PowerEqual {
      bool operator()(const Factor &LHS, const Factor &RHS) {
        return LHS.Power == RHS.Power;
      }
    };
  };
}

namespace {
  class Reassociate : public FunctionPass {
    DenseMap<BasicBlock*, unsigned> RankMap;
    DenseMap<AssertingVH<Value>, unsigned> ValueRankMap;
    SmallVector<WeakVH, 8> RedoInsts;
    SmallVector<WeakVH, 8> DeadInsts;
    bool MadeChange;
  public:
    static char ID; // Pass identification, replacement for typeid
    Reassociate() : FunctionPass(ID) {
      initializeReassociatePass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  private:
    void BuildRankMap(Function &F);
    unsigned getRank(Value *V);
    Value *ReassociateExpression(BinaryOperator *I);
    void RewriteExprTree(BinaryOperator *I, SmallVectorImpl<ValueEntry> &Ops);
    Value *OptimizeExpression(BinaryOperator *I,
                              SmallVectorImpl<ValueEntry> &Ops);
    Value *OptimizeAdd(Instruction *I, SmallVectorImpl<ValueEntry> &Ops);
    bool collectMultiplyFactors(SmallVectorImpl<ValueEntry> &Ops,
                                SmallVectorImpl<Factor> &Factors);
    Value *buildMinimalMultiplyDAG(IRBuilder<> &Builder,
                                   SmallVectorImpl<Factor> &Factors);
    Value *OptimizeMul(BinaryOperator *I, SmallVectorImpl<ValueEntry> &Ops);
    void LinearizeExprTree(BinaryOperator *I, SmallVectorImpl<ValueEntry> &Ops);
    Value *RemoveFactorFromExpression(Value *V, Value *Factor);
    void ReassociateInst(BasicBlock::iterator &BBI);

    void RemoveDeadBinaryOp(Value *V);
  };
}

char Reassociate::ID = 0;
INITIALIZE_PASS(Reassociate, "reassociate",
                "Reassociate expressions", false, false)

// Public interface to the Reassociate pass
FunctionPass *llvm::createReassociatePass() { return new Reassociate(); }

/// isReassociableOp - Return true if V is an instruction of the specified
/// opcode and if it only has one use.
static BinaryOperator *isReassociableOp(Value *V, unsigned Opcode) {
  if (V->hasOneUse() && isa<Instruction>(V) &&
      cast<Instruction>(V)->getOpcode() == Opcode)
    return cast<BinaryOperator>(V);
  return 0;
}

void Reassociate::RemoveDeadBinaryOp(Value *V) {
  BinaryOperator *Op = dyn_cast<BinaryOperator>(V);
  if (!Op)
    return;

  ValueRankMap.erase(Op);
  DeadInsts.push_back(Op);

  BinaryOperator *LHS = isReassociableOp(Op->getOperand(0), Op->getOpcode());
  BinaryOperator *RHS = isReassociableOp(Op->getOperand(1), Op->getOpcode());
  Op->setOperand(0, UndefValue::get(Op->getType()));
  Op->setOperand(1, UndefValue::get(Op->getType()));

  if (LHS)
    RemoveDeadBinaryOp(LHS);
  if (RHS)
    RemoveDeadBinaryOp(RHS);
}

static bool isUnmovableInstruction(Instruction *I) {
  if (I->getOpcode() == Instruction::PHI ||
      I->getOpcode() == Instruction::LandingPad ||
      I->getOpcode() == Instruction::Alloca ||
      I->getOpcode() == Instruction::Load ||
      I->getOpcode() == Instruction::Invoke ||
      (I->getOpcode() == Instruction::Call &&
       !isa<DbgInfoIntrinsic>(I)) ||
      I->getOpcode() == Instruction::UDiv ||
      I->getOpcode() == Instruction::SDiv ||
      I->getOpcode() == Instruction::FDiv ||
      I->getOpcode() == Instruction::URem ||
      I->getOpcode() == Instruction::SRem ||
      I->getOpcode() == Instruction::FRem)
    return true;
  return false;
}

void Reassociate::BuildRankMap(Function &F) {
  unsigned i = 2;

  // Assign distinct ranks to function arguments
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
    ValueRankMap[&*I] = ++i;

  ReversePostOrderTraversal<Function*> RPOT(&F);
  for (ReversePostOrderTraversal<Function*>::rpo_iterator I = RPOT.begin(),
         E = RPOT.end(); I != E; ++I) {
    BasicBlock *BB = *I;
    unsigned BBRank = RankMap[BB] = ++i << 16;

    // Walk the basic block, adding precomputed ranks for any instructions that
    // we cannot move.  This ensures that the ranks for these instructions are
    // all different in the block.
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (isUnmovableInstruction(I))
        ValueRankMap[&*I] = ++BBRank;
  }
}

unsigned Reassociate::getRank(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) {
    if (isa<Argument>(V)) return ValueRankMap[V];   // Function argument.
    return 0;  // Otherwise it's a global or constant, rank 0.
  }

  if (unsigned Rank = ValueRankMap[I])
    return Rank;    // Rank already known?

  // If this is an expression, return the 1+MAX(rank(LHS), rank(RHS)) so that
  // we can reassociate expressions for code motion!  Since we do not recurse
  // for PHI nodes, we cannot have infinite recursion here, because there
  // cannot be loops in the value graph that do not go through PHI nodes.
  unsigned Rank = 0, MaxRank = RankMap[I->getParent()];
  for (unsigned i = 0, e = I->getNumOperands();
       i != e && Rank != MaxRank; ++i)
    Rank = std::max(Rank, getRank(I->getOperand(i)));

  // If this is a not or neg instruction, do not count it for rank.  This
  // assures us that X and ~X will have the same rank.
  if (!I->getType()->isIntegerTy() ||
      (!BinaryOperator::isNot(I) && !BinaryOperator::isNeg(I)))
    ++Rank;

  //DEBUG(dbgs() << "Calculated Rank[" << V->getName() << "] = "
  //     << Rank << "\n");

  return ValueRankMap[I] = Rank;
}

/// LowerNegateToMultiply - Replace 0-X with X*-1.
///
static BinaryOperator *LowerNegateToMultiply(Instruction *Neg,
                         DenseMap<AssertingVH<Value>, unsigned> &ValueRankMap) {
  Constant *Cst = Constant::getAllOnesValue(Neg->getType());

  BinaryOperator *Res =
    BinaryOperator::CreateMul(Neg->getOperand(1), Cst, "",Neg);
  ValueRankMap.erase(Neg);
  Res->takeName(Neg);
  Neg->replaceAllUsesWith(Res);
  Res->setDebugLoc(Neg->getDebugLoc());
  Neg->eraseFromParent();
  return Res;
}

/// LinearizeExprTree - Given an associative binary expression, return the leaf
/// nodes in Ops.  The original expression is the same as Ops[0] op ... Ops[N].
/// Note that a node may occur multiple times in Ops, but if so all occurrences
/// are consecutive in the vector.
///
/// A leaf node is either not a binary operation of the same kind as the root
/// node 'I' (i.e. is not a binary operator at all, or is, but with a different
/// opcode), or is the same kind of binary operator but has a use which either
/// does not belong to the expression, or does belong to the expression but is
/// a leaf node.  Every leaf node has at least one use that is a non-leaf node
/// of the expression, while for non-leaf nodes (except for the root 'I') every
/// use is a non-leaf node of the expression.
///
/// For example:
///           expression graph        node names
///
///                     +        |        I
///                    / \       |
///                   +   +      |      A,  B
///                  / \ / \     |
///                 *   +   *    |    C,  D,  E
///                / \ / \ / \   |
///                   +   *      |      F,  G
///
/// The leaf nodes are C, E, F and G.  The Ops array will contain (maybe not in
/// that order) C, E, F, F, G, G.
///
/// The expression is maximal: if some instruction is a binary operator of the
/// same kind as 'I', and all of its uses are non-leaf nodes of the expression,
/// then the instruction also belongs to the expression, is not a leaf node of
/// it, and its operands also belong to the expression (but may be leaf nodes).
///
/// NOTE: This routine will set operands of non-leaf non-root nodes to undef in
/// order to ensure that every non-root node in the expression has *exactly one*
/// use by a non-leaf node of the expression.  This destruction means that the
/// caller MUST use something like RewriteExprTree to put the values back in.
///
/// In the above example either the right operand of A or the left operand of B
/// will be replaced by undef.  If it is B's operand then this gives:
///
///                     +        |        I
///                    / \       |
///                   +   +      |      A,  B - operand of B replaced with undef
///                  / \   \     |
///                 *   +   *    |    C,  D,  E
///                / \ / \ / \   |
///                   +   *      |      F,  G
///
/// Note that if you visit operands recursively starting from a leaf node then
/// you will never encounter such an undef operand unless you get back to 'I',
/// which requires passing through a phi node.
///
/// Note that this routine may also mutate binary operators of the wrong type
/// that have all uses inside the expression (i.e. only used by non-leaf nodes
/// of the expression) if it can turn them into binary operators of the right
/// type and thus make the expression bigger.

void Reassociate::LinearizeExprTree(BinaryOperator *I,
                                    SmallVectorImpl<ValueEntry> &Ops) {
  DEBUG(dbgs() << "LINEARIZE: " << *I << '\n');

  // Visit all operands of the expression, keeping track of their weight (the
  // number of paths from the expression root to the operand, or if you like
  // the number of times that operand occurs in the linearized expression).
  // For example, if I = X + A, where X = A + B, then I, X and B have weight 1
  // while A has weight two.

  // Worklist of non-leaf nodes (their operands are in the expression too) along
  // with their weights, representing a certain number of paths to the operator.
  // If an operator occurs in the worklist multiple times then we found multiple
  // ways to get to it.
  SmallVector<std::pair<BinaryOperator*, unsigned>, 8> Worklist; // (Op, Weight)
  Worklist.push_back(std::make_pair(I, 1));
  unsigned Opcode = I->getOpcode();

  // Leaves of the expression are values that either aren't the right kind of
  // operation (eg: a constant, or a multiply in an add tree), or are, but have
  // some uses that are not inside the expression.  For example, in I = X + X,
  // X = A + B, the value X has two uses (by I) that are in the expression.  If
  // X has any other uses, for example in a return instruction, then we consider
  // X to be a leaf, and won't analyze it further.  When we first visit a value,
  // if it has more than one use then at first we conservatively consider it to
  // be a leaf.  Later, as the expression is explored, we may discover some more
  // uses of the value from inside the expression.  If all uses turn out to be
  // from within the expression (and the value is a binary operator of the right
  // kind) then the value is no longer considered to be a leaf, and its operands
  // are explored.

  // Leaves - Keeps track of the set of putative leaves as well as the number of
  // paths to each leaf seen so far.
  typedef SmallMap<Value*, unsigned, 8> LeafMap;
  LeafMap Leaves; // Leaf -> Total weight so far.
  SmallVector<Value*, 8> LeafOrder; // Ensure deterministic leaf output order.

#ifndef NDEBUG
  SmallPtrSet<Value*, 8> Visited; // For sanity checking the iteration scheme.
#endif
  while (!Worklist.empty()) {
    std::pair<BinaryOperator*, unsigned> P = Worklist.pop_back_val();
    I = P.first; // We examine the operands of this binary operator.
    assert(P.second >= 1 && "No paths to here, so how did we get here?!");

    for (unsigned OpIdx = 0; OpIdx < 2; ++OpIdx) { // Visit operands.
      Value *Op = I->getOperand(OpIdx);
      unsigned Weight = P.second; // Number of paths to this operand.
      DEBUG(dbgs() << "OPERAND: " << *Op << " (" << Weight << ")\n");
      assert(!Op->use_empty() && "No uses, so how did we get to it?!");

      // If this is a binary operation of the right kind with only one use then
      // add its operands to the expression.
      if (BinaryOperator *BO = isReassociableOp(Op, Opcode)) {
        assert(Visited.insert(Op) && "Not first visit!");
        DEBUG(dbgs() << "DIRECT ADD: " << *Op << " (" << Weight << ")\n");
        Worklist.push_back(std::make_pair(BO, Weight));
        continue;
      }

      // Appears to be a leaf.  Is the operand already in the set of leaves?
      LeafMap::iterator It = Leaves.find(Op);
      if (It == Leaves.end()) {
        // Not in the leaf map.  Must be the first time we saw this operand.
        assert(Visited.insert(Op) && "Not first visit!");
        if (!Op->hasOneUse()) {
          // This value has uses not accounted for by the expression, so it is
          // not safe to modify.  Mark it as being a leaf.
          DEBUG(dbgs() << "ADD USES LEAF: " << *Op << " (" << Weight << ")\n");
          LeafOrder.push_back(Op);
          Leaves[Op] = Weight;
          continue;
        }
        // No uses outside the expression, try morphing it.
      } else if (It != Leaves.end()) {
        // Already in the leaf map.
        assert(Visited.count(Op) && "In leaf map but not visited!");

        // Update the number of paths to the leaf.
        It->second += Weight;

        // The leaf already has one use from inside the expression.  As we want
        // exactly one such use, drop this new use of the leaf.
        assert(!Op->hasOneUse() && "Only one use, but we got here twice!");
        I->setOperand(OpIdx, UndefValue::get(I->getType()));
        MadeChange = true;

        // If the leaf is a binary operation of the right kind and we now see
        // that its multiple original uses were in fact all by nodes belonging
        // to the expression, then no longer consider it to be a leaf and add
        // its operands to the expression.
        if (BinaryOperator *BO = isReassociableOp(Op, Opcode)) {
          DEBUG(dbgs() << "UNLEAF: " << *Op << " (" << It->second << ")\n");
          Worklist.push_back(std::make_pair(BO, It->second));
          Leaves.erase(It);
          continue;
        }

        // If we still have uses that are not accounted for by the expression
        // then it is not safe to modify the value.
        if (!Op->hasOneUse())
          continue;

        // No uses outside the expression, try morphing it.
        Weight = It->second;
        Leaves.erase(It); // Since the value may be morphed below.
      }

      // At this point we have a value which, first of all, is not a binary
      // expression of the right kind, and secondly, is only used inside the
      // expression.  This means that it can safely be modified.  See if we
      // can usefully morph it into an expression of the right kind.
      assert((!isa<Instruction>(Op) ||
              cast<Instruction>(Op)->getOpcode() != Opcode) &&
             "Should have been handled above!");
      assert(Op->hasOneUse() && "Has uses outside the expression tree!");

      // If this is a multiply expression, turn any internal negations into
      // multiplies by -1 so they can be reassociated.
      BinaryOperator *BO = dyn_cast<BinaryOperator>(Op);
      if (Opcode == Instruction::Mul && BO && BinaryOperator::isNeg(BO)) {
        DEBUG(dbgs() << "MORPH LEAF: " << *Op << " (" << Weight << ") TO ");
        BO = LowerNegateToMultiply(BO, ValueRankMap);
        DEBUG(dbgs() << *BO << 'n');
        Worklist.push_back(std::make_pair(BO, Weight));
        MadeChange = true;
        continue;
      }

      // Failed to morph into an expression of the right type.  This really is
      // a leaf.
      DEBUG(dbgs() << "ADD LEAF: " << *Op << " (" << Weight << ")\n");
      assert(!isReassociableOp(Op, Opcode) && "Value was morphed?");
      LeafOrder.push_back(Op);
      Leaves[Op] = Weight;
    }
  }

  // The leaves, repeated according to their weights, represent the linearized
  // form of the expression.
  for (unsigned i = 0, e = LeafOrder.size(); i != e; ++i) {
    Value *V = LeafOrder[i];
    LeafMap::iterator It = Leaves.find(V);
    if (It == Leaves.end())
      // Leaf already output, or node initially thought to be a leaf wasn't.
      continue;
    assert(!isReassociableOp(V, Opcode) && "Shouldn't be a leaf!");
    unsigned Weight = It->second;
    assert(Weight > 0 && "No paths to this value!");
    // FIXME: Rather than repeating values Weight times, use a vector of
    // (ValueEntry, multiplicity) pairs.
    Ops.append(Weight, ValueEntry(getRank(V), V));
    // Ensure the leaf is only output once.
    Leaves.erase(It);
  }
}

// RewriteExprTree - Now that the operands for this expression tree are
// linearized and optimized, emit them in-order.
void Reassociate::RewriteExprTree(BinaryOperator *I,
                                  SmallVectorImpl<ValueEntry> &Ops) {
  assert(Ops.size() > 1 && "Single values should be used directly!");

  // Since our optimizations never increase the number of operations, the new
  // expression can always be written by reusing the existing binary operators
  // from the original expression tree, without creating any new instructions,
  // though the rewritten expression may have a completely different topology.
  // We take care to not change anything if the new expression will be the same
  // as the original.  If more than trivial changes (like commuting operands)
  // were made then we are obliged to clear out any optional subclass data like
  // nsw flags.

  /// NodesToRewrite - Nodes from the original expression available for writing
  /// the new expression into.
  SmallVector<BinaryOperator*, 8> NodesToRewrite;
  unsigned Opcode = I->getOpcode();
  NodesToRewrite.push_back(I);

  // ExpressionChanged - Whether the rewritten expression differs non-trivially
  // from the original, requiring the clearing of all optional flags.
  bool ExpressionChanged = false;
  BinaryOperator *Previous;
  BinaryOperator *Op = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    assert(!NodesToRewrite.empty() &&
           "Optimized expressions has more nodes than original!");
    Previous = Op; Op = NodesToRewrite.pop_back_val();
    // Compactify the tree instructions together with each other to guarantee
    // that the expression tree is dominated by all of Ops.
    if (Previous)
      Op->moveBefore(Previous);

    // The last operation (which comes earliest in the IR) is special as both
    // operands will come from Ops, rather than just one with the other being
    // a subexpression.
    if (i+2 == Ops.size()) {
      Value *NewLHS = Ops[i].Op;
      Value *NewRHS = Ops[i+1].Op;
      Value *OldLHS = Op->getOperand(0);
      Value *OldRHS = Op->getOperand(1);

      if (NewLHS == OldLHS && NewRHS == OldRHS)
        // Nothing changed, leave it alone.
        break;

      if (NewLHS == OldRHS && NewRHS == OldLHS) {
        // The order of the operands was reversed.  Swap them.
        DEBUG(dbgs() << "RA: " << *Op << '\n');
        Op->swapOperands();
        DEBUG(dbgs() << "TO: " << *Op << '\n');
        MadeChange = true;
        ++NumChanged;
        break;
      }

      // The new operation differs non-trivially from the original. Overwrite
      // the old operands with the new ones.
      DEBUG(dbgs() << "RA: " << *Op << '\n');
      if (NewLHS != OldLHS) {
        if (BinaryOperator *BO = isReassociableOp(OldLHS, Opcode))
          NodesToRewrite.push_back(BO);
        Op->setOperand(0, NewLHS);
      }
      if (NewRHS != OldRHS) {
        if (BinaryOperator *BO = isReassociableOp(OldRHS, Opcode))
          NodesToRewrite.push_back(BO);
        Op->setOperand(1, NewRHS);
      }
      DEBUG(dbgs() << "TO: " << *Op << '\n');

      ExpressionChanged = true;
      MadeChange = true;
      ++NumChanged;

      break;
    }

    // Not the last operation.  The left-hand side will be a sub-expression
    // while the right-hand side will be the current element of Ops.
    Value *NewRHS = Ops[i].Op;
    if (NewRHS != Op->getOperand(1)) {
      DEBUG(dbgs() << "RA: " << *Op << '\n');
      if (NewRHS == Op->getOperand(0)) {
        // The new right-hand side was already present as the left operand.  If
        // we are lucky then swapping the operands will sort out both of them.
        Op->swapOperands();
      } else {
        // Overwrite with the new right-hand side.
        if (BinaryOperator *BO = isReassociableOp(Op->getOperand(1), Opcode))
          NodesToRewrite.push_back(BO);
        Op->setOperand(1, NewRHS);
        ExpressionChanged = true;
      }
      DEBUG(dbgs() << "TO: " << *Op << '\n');
      MadeChange = true;
      ++NumChanged;
    }

    // Now deal with the left-hand side.  If this is already an operation node
    // from the original expression then just rewrite the rest of the expression
    // into it.
    if (BinaryOperator *BO = isReassociableOp(Op->getOperand(0), Opcode)) {
      NodesToRewrite.push_back(BO);
      continue;
    }

    // Otherwise, grab a spare node from the original expression and use that as
    // the left-hand side.
    assert(!NodesToRewrite.empty() &&
           "Optimized expressions has more nodes than original!");
    DEBUG(dbgs() << "RA: " << *Op << '\n');
    Op->setOperand(0, NodesToRewrite.back());
    DEBUG(dbgs() << "TO: " << *Op << '\n');
    ExpressionChanged = true;
    MadeChange = true;
    ++NumChanged;
  }

  // If the expression changed non-trivially then clear out all subclass data in
  // the entire rewritten expression.
  if (ExpressionChanged) {
    do {
      Op->clearSubclassOptionalData();
      if (Op == I)
        break;
      Op = cast<BinaryOperator>(*Op->use_begin());
    } while (1);
  }

  // Throw away any left over nodes from the original expression.
  for (unsigned i = 0, e = NodesToRewrite.size(); i != e; ++i)
    RemoveDeadBinaryOp(NodesToRewrite[i]);
}

/// NegateValue - Insert instructions before the instruction pointed to by BI,
/// that computes the negative version of the value specified.  The negative
/// version of the value is returned, and BI is left pointing at the instruction
/// that should be processed next by the reassociation pass.
static Value *NegateValue(Value *V, Instruction *BI) {
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getNeg(C);

  // We are trying to expose opportunity for reassociation.  One of the things
  // that we want to do to achieve this is to push a negation as deep into an
  // expression chain as possible, to expose the add instructions.  In practice,
  // this means that we turn this:
  //   X = -(A+12+C+D)   into    X = -A + -12 + -C + -D = -12 + -A + -C + -D
  // so that later, a: Y = 12+X could get reassociated with the -12 to eliminate
  // the constants.  We assume that instcombine will clean up the mess later if
  // we introduce tons of unnecessary negation instructions.
  //
  if (BinaryOperator *I = isReassociableOp(V, Instruction::Add)) {
    // Push the negates through the add.
    I->setOperand(0, NegateValue(I->getOperand(0), BI));
    I->setOperand(1, NegateValue(I->getOperand(1), BI));

    // We must move the add instruction here, because the neg instructions do
    // not dominate the old add instruction in general.  By moving it, we are
    // assured that the neg instructions we just inserted dominate the
    // instruction we are about to insert after them.
    //
    I->moveBefore(BI);
    I->setName(I->getName()+".neg");
    return I;
  }

  // Okay, we need to materialize a negated version of V with an instruction.
  // Scan the use lists of V to see if we have one already.
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;++UI){
    User *U = *UI;
    if (!BinaryOperator::isNeg(U)) continue;

    // We found one!  Now we have to make sure that the definition dominates
    // this use.  We do this by moving it to the entry block (if it is a
    // non-instruction value) or right after the definition.  These negates will
    // be zapped by reassociate later, so we don't need much finesse here.
    BinaryOperator *TheNeg = cast<BinaryOperator>(U);

    // Verify that the negate is in this function, V might be a constant expr.
    if (TheNeg->getParent()->getParent() != BI->getParent()->getParent())
      continue;

    BasicBlock::iterator InsertPt;
    if (Instruction *InstInput = dyn_cast<Instruction>(V)) {
      if (InvokeInst *II = dyn_cast<InvokeInst>(InstInput)) {
        InsertPt = II->getNormalDest()->begin();
      } else {
        InsertPt = InstInput;
        ++InsertPt;
      }
      while (isa<PHINode>(InsertPt)) ++InsertPt;
    } else {
      InsertPt = TheNeg->getParent()->getParent()->getEntryBlock().begin();
    }
    TheNeg->moveBefore(InsertPt);
    return TheNeg;
  }

  // Insert a 'neg' instruction that subtracts the value from zero to get the
  // negation.
  return BinaryOperator::CreateNeg(V, V->getName() + ".neg", BI);
}

/// ShouldBreakUpSubtract - Return true if we should break up this subtract of
/// X-Y into (X + -Y).
static bool ShouldBreakUpSubtract(Instruction *Sub) {
  // If this is a negation, we can't split it up!
  if (BinaryOperator::isNeg(Sub))
    return false;

  // Don't bother to break this up unless either the LHS is an associable add or
  // subtract or if this is only used by one.
  if (isReassociableOp(Sub->getOperand(0), Instruction::Add) ||
      isReassociableOp(Sub->getOperand(0), Instruction::Sub))
    return true;
  if (isReassociableOp(Sub->getOperand(1), Instruction::Add) ||
      isReassociableOp(Sub->getOperand(1), Instruction::Sub))
    return true;
  if (Sub->hasOneUse() &&
      (isReassociableOp(Sub->use_back(), Instruction::Add) ||
       isReassociableOp(Sub->use_back(), Instruction::Sub)))
    return true;

  return false;
}

/// BreakUpSubtract - If we have (X-Y), and if either X is an add, or if this is
/// only used by an add, transform this into (X+(0-Y)) to promote better
/// reassociation.
static Instruction *BreakUpSubtract(Instruction *Sub,
                         DenseMap<AssertingVH<Value>, unsigned> &ValueRankMap) {
  // Convert a subtract into an add and a neg instruction. This allows sub
  // instructions to be commuted with other add instructions.
  //
  // Calculate the negative value of Operand 1 of the sub instruction,
  // and set it as the RHS of the add instruction we just made.
  //
  Value *NegVal = NegateValue(Sub->getOperand(1), Sub);
  Instruction *New =
    BinaryOperator::CreateAdd(Sub->getOperand(0), NegVal, "", Sub);
  New->takeName(Sub);

  // Everyone now refers to the add instruction.
  ValueRankMap.erase(Sub);
  Sub->replaceAllUsesWith(New);
  New->setDebugLoc(Sub->getDebugLoc());
  Sub->eraseFromParent();

  DEBUG(dbgs() << "Negated: " << *New << '\n');
  return New;
}

/// ConvertShiftToMul - If this is a shift of a reassociable multiply or is used
/// by one, change this into a multiply by a constant to assist with further
/// reassociation.
static Instruction *ConvertShiftToMul(Instruction *Shl,
                         DenseMap<AssertingVH<Value>, unsigned> &ValueRankMap) {
  // If an operand of this shift is a reassociable multiply, or if the shift
  // is used by a reassociable multiply or add, turn into a multiply.
  if (isReassociableOp(Shl->getOperand(0), Instruction::Mul) ||
      (Shl->hasOneUse() &&
       (isReassociableOp(Shl->use_back(), Instruction::Mul) ||
        isReassociableOp(Shl->use_back(), Instruction::Add)))) {
    Constant *MulCst = ConstantInt::get(Shl->getType(), 1);
    MulCst = ConstantExpr::getShl(MulCst, cast<Constant>(Shl->getOperand(1)));

    Instruction *Mul =
      BinaryOperator::CreateMul(Shl->getOperand(0), MulCst, "", Shl);
    ValueRankMap.erase(Shl);
    Mul->takeName(Shl);
    Shl->replaceAllUsesWith(Mul);
    Mul->setDebugLoc(Shl->getDebugLoc());
    Shl->eraseFromParent();
    return Mul;
  }
  return 0;
}

/// FindInOperandList - Scan backwards and forwards among values with the same
/// rank as element i to see if X exists.  If X does not exist, return i.  This
/// is useful when scanning for 'x' when we see '-x' because they both get the
/// same rank.
static unsigned FindInOperandList(SmallVectorImpl<ValueEntry> &Ops, unsigned i,
                                  Value *X) {
  unsigned XRank = Ops[i].Rank;
  unsigned e = Ops.size();
  for (unsigned j = i+1; j != e && Ops[j].Rank == XRank; ++j)
    if (Ops[j].Op == X)
      return j;
  // Scan backwards.
  for (unsigned j = i-1; j != ~0U && Ops[j].Rank == XRank; --j)
    if (Ops[j].Op == X)
      return j;
  return i;
}

/// EmitAddTreeOfValues - Emit a tree of add instructions, summing Ops together
/// and returning the result.  Insert the tree before I.
static Value *EmitAddTreeOfValues(Instruction *I,
                                  SmallVectorImpl<WeakVH> &Ops){
  if (Ops.size() == 1) return Ops.back();

  Value *V1 = Ops.back();
  Ops.pop_back();
  Value *V2 = EmitAddTreeOfValues(I, Ops);
  return BinaryOperator::CreateAdd(V2, V1, "tmp", I);
}

/// RemoveFactorFromExpression - If V is an expression tree that is a
/// multiplication sequence, and if this sequence contains a multiply by Factor,
/// remove Factor from the tree and return the new tree.
Value *Reassociate::RemoveFactorFromExpression(Value *V, Value *Factor) {
  BinaryOperator *BO = isReassociableOp(V, Instruction::Mul);
  if (!BO) return 0;

  SmallVector<ValueEntry, 8> Factors;
  LinearizeExprTree(BO, Factors);

  bool FoundFactor = false;
  bool NeedsNegate = false;
  for (unsigned i = 0, e = Factors.size(); i != e; ++i) {
    if (Factors[i].Op == Factor) {
      FoundFactor = true;
      Factors.erase(Factors.begin()+i);
      break;
    }

    // If this is a negative version of this factor, remove it.
    if (ConstantInt *FC1 = dyn_cast<ConstantInt>(Factor))
      if (ConstantInt *FC2 = dyn_cast<ConstantInt>(Factors[i].Op))
        if (FC1->getValue() == -FC2->getValue()) {
          FoundFactor = NeedsNegate = true;
          Factors.erase(Factors.begin()+i);
          break;
        }
  }

  if (!FoundFactor) {
    // Make sure to restore the operands to the expression tree.
    RewriteExprTree(BO, Factors);
    return 0;
  }

  BasicBlock::iterator InsertPt = BO; ++InsertPt;

  // If this was just a single multiply, remove the multiply and return the only
  // remaining operand.
  if (Factors.size() == 1) {
    RemoveDeadBinaryOp(BO);
    V = Factors[0].Op;
  } else {
    RewriteExprTree(BO, Factors);
    V = BO;
  }

  if (NeedsNegate)
    V = BinaryOperator::CreateNeg(V, "neg", InsertPt);

  return V;
}

/// FindSingleUseMultiplyFactors - If V is a single-use multiply, recursively
/// add its operands as factors, otherwise add V to the list of factors.
///
/// Ops is the top-level list of add operands we're trying to factor.
static void FindSingleUseMultiplyFactors(Value *V,
                                         SmallVectorImpl<Value*> &Factors,
                                       const SmallVectorImpl<ValueEntry> &Ops) {
  BinaryOperator *BO = isReassociableOp(V, Instruction::Mul);
  if (!BO) {
    Factors.push_back(V);
    return;
  }

  // Otherwise, add the LHS and RHS to the list of factors.
  FindSingleUseMultiplyFactors(BO->getOperand(1), Factors, Ops);
  FindSingleUseMultiplyFactors(BO->getOperand(0), Factors, Ops);
}

/// OptimizeAndOrXor - Optimize a series of operands to an 'and', 'or', or 'xor'
/// instruction.  This optimizes based on identities.  If it can be reduced to
/// a single Value, it is returned, otherwise the Ops list is mutated as
/// necessary.
static Value *OptimizeAndOrXor(unsigned Opcode,
                               SmallVectorImpl<ValueEntry> &Ops) {
  // Scan the operand lists looking for X and ~X pairs, along with X,X pairs.
  // If we find any, we can simplify the expression. X&~X == 0, X|~X == -1.
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    // First, check for X and ~X in the operand list.
    assert(i < Ops.size());
    if (BinaryOperator::isNot(Ops[i].Op)) {    // Cannot occur for ^.
      Value *X = BinaryOperator::getNotArgument(Ops[i].Op);
      unsigned FoundX = FindInOperandList(Ops, i, X);
      if (FoundX != i) {
        if (Opcode == Instruction::And)   // ...&X&~X = 0
          return Constant::getNullValue(X->getType());

        if (Opcode == Instruction::Or)    // ...|X|~X = -1
          return Constant::getAllOnesValue(X->getType());
      }
    }

    // Next, check for duplicate pairs of values, which we assume are next to
    // each other, due to our sorting criteria.
    assert(i < Ops.size());
    if (i+1 != Ops.size() && Ops[i+1].Op == Ops[i].Op) {
      if (Opcode == Instruction::And || Opcode == Instruction::Or) {
        // Drop duplicate values for And and Or.
        Ops.erase(Ops.begin()+i);
        --i; --e;
        ++NumAnnihil;
        continue;
      }

      // Drop pairs of values for Xor.
      assert(Opcode == Instruction::Xor);
      if (e == 2)
        return Constant::getNullValue(Ops[0].Op->getType());

      // Y ^ X^X -> Y
      Ops.erase(Ops.begin()+i, Ops.begin()+i+2);
      i -= 1; e -= 2;
      ++NumAnnihil;
    }
  }
  return 0;
}

/// OptimizeAdd - Optimize a series of operands to an 'add' instruction.  This
/// optimizes based on identities.  If it can be reduced to a single Value, it
/// is returned, otherwise the Ops list is mutated as necessary.
Value *Reassociate::OptimizeAdd(Instruction *I,
                                SmallVectorImpl<ValueEntry> &Ops) {
  // Scan the operand lists looking for X and -X pairs.  If we find any, we
  // can simplify the expression. X+-X == 0.  While we're at it, scan for any
  // duplicates.  We want to canonicalize Y+Y+Y+Z -> 3*Y+Z.
  //
  // TODO: We could handle "X + ~X" -> "-1" if we wanted, since "-X = ~X+1".
  //
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    Value *TheOp = Ops[i].Op;
    // Check to see if we've seen this operand before.  If so, we factor all
    // instances of the operand together.  Due to our sorting criteria, we know
    // that these need to be next to each other in the vector.
    if (i+1 != Ops.size() && Ops[i+1].Op == TheOp) {
      // Rescan the list, remove all instances of this operand from the expr.
      unsigned NumFound = 0;
      do {
        Ops.erase(Ops.begin()+i);
        ++NumFound;
      } while (i != Ops.size() && Ops[i].Op == TheOp);

      DEBUG(errs() << "\nFACTORING [" << NumFound << "]: " << *TheOp << '\n');
      ++NumFactor;

      // Insert a new multiply.
      Value *Mul = ConstantInt::get(cast<IntegerType>(I->getType()), NumFound);
      Mul = BinaryOperator::CreateMul(TheOp, Mul, "factor", I);

      // Now that we have inserted a multiply, optimize it. This allows us to
      // handle cases that require multiple factoring steps, such as this:
      // (X*2) + (X*2) + (X*2) -> (X*2)*3 -> X*6
      RedoInsts.push_back(Mul);

      // If every add operand was a duplicate, return the multiply.
      if (Ops.empty())
        return Mul;

      // Otherwise, we had some input that didn't have the dupe, such as
      // "A + A + B" -> "A*2 + B".  Add the new multiply to the list of
      // things being added by this operation.
      Ops.insert(Ops.begin(), ValueEntry(getRank(Mul), Mul));

      --i;
      e = Ops.size();
      continue;
    }

    // Check for X and -X in the operand list.
    if (!BinaryOperator::isNeg(TheOp))
      continue;

    Value *X = BinaryOperator::getNegArgument(TheOp);
    unsigned FoundX = FindInOperandList(Ops, i, X);
    if (FoundX == i)
      continue;

    // Remove X and -X from the operand list.
    if (Ops.size() == 2)
      return Constant::getNullValue(X->getType());

    Ops.erase(Ops.begin()+i);
    if (i < FoundX)
      --FoundX;
    else
      --i;   // Need to back up an extra one.
    Ops.erase(Ops.begin()+FoundX);
    ++NumAnnihil;
    --i;     // Revisit element.
    e -= 2;  // Removed two elements.
  }

  // Scan the operand list, checking to see if there are any common factors
  // between operands.  Consider something like A*A+A*B*C+D.  We would like to
  // reassociate this to A*(A+B*C)+D, which reduces the number of multiplies.
  // To efficiently find this, we count the number of times a factor occurs
  // for any ADD operands that are MULs.
  DenseMap<Value*, unsigned> FactorOccurrences;

  // Keep track of each multiply we see, to avoid triggering on (X*4)+(X*4)
  // where they are actually the same multiply.
  unsigned MaxOcc = 0;
  Value *MaxOccVal = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    BinaryOperator *BOp = isReassociableOp(Ops[i].Op, Instruction::Mul);
    if (!BOp)
      continue;

    // Compute all of the factors of this added value.
    SmallVector<Value*, 8> Factors;
    FindSingleUseMultiplyFactors(BOp, Factors, Ops);
    assert(Factors.size() > 1 && "Bad linearize!");

    // Add one to FactorOccurrences for each unique factor in this op.
    SmallPtrSet<Value*, 8> Duplicates;
    for (unsigned i = 0, e = Factors.size(); i != e; ++i) {
      Value *Factor = Factors[i];
      if (!Duplicates.insert(Factor)) continue;

      unsigned Occ = ++FactorOccurrences[Factor];
      if (Occ > MaxOcc) { MaxOcc = Occ; MaxOccVal = Factor; }

      // If Factor is a negative constant, add the negated value as a factor
      // because we can percolate the negate out.  Watch for minint, which
      // cannot be positivified.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Factor))
        if (CI->isNegative() && !CI->isMinValue(true)) {
          Factor = ConstantInt::get(CI->getContext(), -CI->getValue());
          assert(!Duplicates.count(Factor) &&
                 "Shouldn't have two constant factors, missed a canonicalize");

          unsigned Occ = ++FactorOccurrences[Factor];
          if (Occ > MaxOcc) { MaxOcc = Occ; MaxOccVal = Factor; }
        }
    }
  }

  // If any factor occurred more than one time, we can pull it out.
  if (MaxOcc > 1) {
    DEBUG(errs() << "\nFACTORING [" << MaxOcc << "]: " << *MaxOccVal << '\n');
    ++NumFactor;

    // Create a new instruction that uses the MaxOccVal twice.  If we don't do
    // this, we could otherwise run into situations where removing a factor
    // from an expression will drop a use of maxocc, and this can cause
    // RemoveFactorFromExpression on successive values to behave differently.
    Instruction *DummyInst = BinaryOperator::CreateAdd(MaxOccVal, MaxOccVal);
    SmallVector<WeakVH, 4> NewMulOps;
    for (unsigned i = 0; i != Ops.size(); ++i) {
      // Only try to remove factors from expressions we're allowed to.
      BinaryOperator *BOp = isReassociableOp(Ops[i].Op, Instruction::Mul);
      if (!BOp)
        continue;

      if (Value *V = RemoveFactorFromExpression(Ops[i].Op, MaxOccVal)) {
        // The factorized operand may occur several times.  Convert them all in
        // one fell swoop.
        for (unsigned j = Ops.size(); j != i;) {
          --j;
          if (Ops[j].Op == Ops[i].Op) {
            NewMulOps.push_back(V);
            Ops.erase(Ops.begin()+j);
          }
        }
        --i;
      }
    }

    // No need for extra uses anymore.
    delete DummyInst;

    unsigned NumAddedValues = NewMulOps.size();
    Value *V = EmitAddTreeOfValues(I, NewMulOps);

    // Now that we have inserted the add tree, optimize it. This allows us to
    // handle cases that require multiple factoring steps, such as this:
    // A*A*B + A*A*C   -->   A*(A*B+A*C)   -->   A*(A*(B+C))
    assert(NumAddedValues > 1 && "Each occurrence should contribute a value");
    (void)NumAddedValues;
    RedoInsts.push_back(V);

    // Create the multiply.
    Value *V2 = BinaryOperator::CreateMul(V, MaxOccVal, "tmp", I);

    // Rerun associate on the multiply in case the inner expression turned into
    // a multiply.  We want to make sure that we keep things in canonical form.
    RedoInsts.push_back(V2);

    // If every add operand included the factor (e.g. "A*B + A*C"), then the
    // entire result expression is just the multiply "A*(B+C)".
    if (Ops.empty())
      return V2;

    // Otherwise, we had some input that didn't have the factor, such as
    // "A*B + A*C + D" -> "A*(B+C) + D".  Add the new multiply to the list of
    // things being added by this operation.
    Ops.insert(Ops.begin(), ValueEntry(getRank(V2), V2));
  }

  return 0;
}

namespace {
  /// \brief Predicate tests whether a ValueEntry's op is in a map.
  struct IsValueInMap {
    const DenseMap<Value *, unsigned> &Map;

    IsValueInMap(const DenseMap<Value *, unsigned> &Map) : Map(Map) {}

    bool operator()(const ValueEntry &Entry) {
      return Map.find(Entry.Op) != Map.end();
    }
  };
}

/// \brief Build up a vector of value/power pairs factoring a product.
///
/// Given a series of multiplication operands, build a vector of factors and
/// the powers each is raised to when forming the final product. Sort them in
/// the order of descending power.
///
///      (x*x)          -> [(x, 2)]
///     ((x*x)*x)       -> [(x, 3)]
///   ((((x*y)*x)*y)*x) -> [(x, 3), (y, 2)]
///
/// \returns Whether any factors have a power greater than one.
bool Reassociate::collectMultiplyFactors(SmallVectorImpl<ValueEntry> &Ops,
                                         SmallVectorImpl<Factor> &Factors) {
  // FIXME: Have Ops be (ValueEntry, Multiplicity) pairs, simplifying this.
  // Compute the sum of powers of simplifiable factors.
  unsigned FactorPowerSum = 0;
  for (unsigned Idx = 1, Size = Ops.size(); Idx < Size; ++Idx) {
    Value *Op = Ops[Idx-1].Op;

    // Count the number of occurrences of this value.
    unsigned Count = 1;
    for (; Idx < Size && Ops[Idx].Op == Op; ++Idx)
      ++Count;
    // Track for simplification all factors which occur 2 or more times.
    if (Count > 1)
      FactorPowerSum += Count;
  }

  // We can only simplify factors if the sum of the powers of our simplifiable
  // factors is 4 or higher. When that is the case, we will *always* have
  // a simplification. This is an important invariant to prevent cyclicly
  // trying to simplify already minimal formations.
  if (FactorPowerSum < 4)
    return false;

  // Now gather the simplifiable factors, removing them from Ops.
  FactorPowerSum = 0;
  for (unsigned Idx = 1; Idx < Ops.size(); ++Idx) {
    Value *Op = Ops[Idx-1].Op;

    // Count the number of occurrences of this value.
    unsigned Count = 1;
    for (; Idx < Ops.size() && Ops[Idx].Op == Op; ++Idx)
      ++Count;
    if (Count == 1)
      continue;
    // Move an even number of occurences to Factors.
    Count &= ~1U;
    Idx -= Count;
    FactorPowerSum += Count;
    Factors.push_back(Factor(Op, Count));
    Ops.erase(Ops.begin()+Idx, Ops.begin()+Idx+Count);
  }

  // None of the adjustments above should have reduced the sum of factor powers
  // below our mininum of '4'.
  assert(FactorPowerSum >= 4);

  std::sort(Factors.begin(), Factors.end(), Factor::PowerDescendingSorter());
  return true;
}

/// \brief Build a tree of multiplies, computing the product of Ops.
static Value *buildMultiplyTree(IRBuilder<> &Builder,
                                SmallVectorImpl<Value*> &Ops) {
  if (Ops.size() == 1)
    return Ops.back();

  Value *LHS = Ops.pop_back_val();
  do {
    LHS = Builder.CreateMul(LHS, Ops.pop_back_val());
  } while (!Ops.empty());

  return LHS;
}

/// \brief Build a minimal multiplication DAG for (a^x)*(b^y)*(c^z)*...
///
/// Given a vector of values raised to various powers, where no two values are
/// equal and the powers are sorted in decreasing order, compute the minimal
/// DAG of multiplies to compute the final product, and return that product
/// value.
Value *Reassociate::buildMinimalMultiplyDAG(IRBuilder<> &Builder,
                                            SmallVectorImpl<Factor> &Factors) {
  assert(Factors[0].Power);
  SmallVector<Value *, 4> OuterProduct;
  for (unsigned LastIdx = 0, Idx = 1, Size = Factors.size();
       Idx < Size && Factors[Idx].Power > 0; ++Idx) {
    if (Factors[Idx].Power != Factors[LastIdx].Power) {
      LastIdx = Idx;
      continue;
    }

    // We want to multiply across all the factors with the same power so that
    // we can raise them to that power as a single entity. Build a mini tree
    // for that.
    SmallVector<Value *, 4> InnerProduct;
    InnerProduct.push_back(Factors[LastIdx].Base);
    do {
      InnerProduct.push_back(Factors[Idx].Base);
      ++Idx;
    } while (Idx < Size && Factors[Idx].Power == Factors[LastIdx].Power);

    // Reset the base value of the first factor to the new expression tree.
    // We'll remove all the factors with the same power in a second pass.
    Factors[LastIdx].Base = buildMultiplyTree(Builder, InnerProduct);
    RedoInsts.push_back(Factors[LastIdx].Base);

    LastIdx = Idx;
  }
  // Unique factors with equal powers -- we've folded them into the first one's
  // base.
  Factors.erase(std::unique(Factors.begin(), Factors.end(),
                            Factor::PowerEqual()),
                Factors.end());

  // Iteratively collect the base of each factor with an add power into the
  // outer product, and halve each power in preparation for squaring the
  // expression.
  for (unsigned Idx = 0, Size = Factors.size(); Idx != Size; ++Idx) {
    if (Factors[Idx].Power & 1)
      OuterProduct.push_back(Factors[Idx].Base);
    Factors[Idx].Power >>= 1;
  }
  if (Factors[0].Power) {
    Value *SquareRoot = buildMinimalMultiplyDAG(Builder, Factors);
    OuterProduct.push_back(SquareRoot);
    OuterProduct.push_back(SquareRoot);
  }
  if (OuterProduct.size() == 1)
    return OuterProduct.front();

  Value *V = buildMultiplyTree(Builder, OuterProduct);
  return V;
}

Value *Reassociate::OptimizeMul(BinaryOperator *I,
                                SmallVectorImpl<ValueEntry> &Ops) {
  // We can only optimize the multiplies when there is a chain of more than
  // three, such that a balanced tree might require fewer total multiplies.
  if (Ops.size() < 4)
    return 0;

  // Try to turn linear trees of multiplies without other uses of the
  // intermediate stages into minimal multiply DAGs with perfect sub-expression
  // re-use.
  SmallVector<Factor, 4> Factors;
  if (!collectMultiplyFactors(Ops, Factors))
    return 0; // All distinct factors, so nothing left for us to do.

  IRBuilder<> Builder(I);
  Value *V = buildMinimalMultiplyDAG(Builder, Factors);
  if (Ops.empty())
    return V;

  ValueEntry NewEntry = ValueEntry(getRank(V), V);
  Ops.insert(std::lower_bound(Ops.begin(), Ops.end(), NewEntry), NewEntry);
  return 0;
}

Value *Reassociate::OptimizeExpression(BinaryOperator *I,
                                       SmallVectorImpl<ValueEntry> &Ops) {
  // Now that we have the linearized expression tree, try to optimize it.
  // Start by folding any constants that we found.
  bool IterateOptimization = false;
  if (Ops.size() == 1) return Ops[0].Op;

  unsigned Opcode = I->getOpcode();

  if (Constant *V1 = dyn_cast<Constant>(Ops[Ops.size()-2].Op))
    if (Constant *V2 = dyn_cast<Constant>(Ops.back().Op)) {
      Ops.pop_back();
      Ops.back().Op = ConstantExpr::get(Opcode, V1, V2);
      return OptimizeExpression(I, Ops);
    }

  // Check for destructive annihilation due to a constant being used.
  if (ConstantInt *CstVal = dyn_cast<ConstantInt>(Ops.back().Op))
    switch (Opcode) {
    default: break;
    case Instruction::And:
      if (CstVal->isZero())                  // X & 0 -> 0
        return CstVal;
      if (CstVal->isAllOnesValue())          // X & -1 -> X
        Ops.pop_back();
      break;
    case Instruction::Mul:
      if (CstVal->isZero()) {                // X * 0 -> 0
        ++NumAnnihil;
        return CstVal;
      }

      if (cast<ConstantInt>(CstVal)->isOne())
        Ops.pop_back();                      // X * 1 -> X
      break;
    case Instruction::Or:
      if (CstVal->isAllOnesValue())          // X | -1 -> -1
        return CstVal;
      // FALLTHROUGH!
    case Instruction::Add:
    case Instruction::Xor:
      if (CstVal->isZero())                  // X [|^+] 0 -> X
        Ops.pop_back();
      break;
    }
  if (Ops.size() == 1) return Ops[0].Op;

  // Handle destructive annihilation due to identities between elements in the
  // argument list here.
  unsigned NumOps = Ops.size();
  switch (Opcode) {
  default: break;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    if (Value *Result = OptimizeAndOrXor(Opcode, Ops))
      return Result;
    break;

  case Instruction::Add:
    if (Value *Result = OptimizeAdd(I, Ops))
      return Result;
    break;

  case Instruction::Mul:
    if (Value *Result = OptimizeMul(I, Ops))
      return Result;
    break;
  }

  if (IterateOptimization || Ops.size() != NumOps)
    return OptimizeExpression(I, Ops);
  return 0;
}

/// ReassociateInst - Inspect and reassociate the instruction at the
/// given position, post-incrementing the position.
void Reassociate::ReassociateInst(BasicBlock::iterator &BBI) {
  Instruction *BI = BBI++;
  if (BI->getOpcode() == Instruction::Shl &&
      isa<ConstantInt>(BI->getOperand(1)))
    if (Instruction *NI = ConvertShiftToMul(BI, ValueRankMap)) {
      MadeChange = true;
      BI = NI;
    }

  // Floating point binary operators are not associative, but we can still
  // commute (some) of them, to canonicalize the order of their operands.
  // This can potentially expose more CSE opportunities, and makes writing
  // other transformations simpler.
  if (isa<BinaryOperator>(BI) &&
      (BI->getType()->isFloatingPointTy() || BI->getType()->isVectorTy())) {
    // FAdd and FMul can be commuted.
    if (BI->getOpcode() != Instruction::FMul &&
        BI->getOpcode() != Instruction::FAdd)
      return;

    Value *LHS = BI->getOperand(0);
    Value *RHS = BI->getOperand(1);
    unsigned LHSRank = getRank(LHS);
    unsigned RHSRank = getRank(RHS);

    // Sort the operands by rank.
    if (RHSRank < LHSRank) {
      BI->setOperand(0, RHS);
      BI->setOperand(1, LHS);
    }

    return;
  }

  // Do not reassociate operations that we do not understand.
  if (!isa<BinaryOperator>(BI))
    return;

  // Do not reassociate boolean (i1) expressions.  We want to preserve the
  // original order of evaluation for short-circuited comparisons that
  // SimplifyCFG has folded to AND/OR expressions.  If the expression
  // is not further optimized, it is likely to be transformed back to a
  // short-circuited form for code gen, and the source order may have been
  // optimized for the most likely conditions.
  if (BI->getType()->isIntegerTy(1))
    return;

  // If this is a subtract instruction which is not already in negate form,
  // see if we can convert it to X+-Y.
  if (BI->getOpcode() == Instruction::Sub) {
    if (ShouldBreakUpSubtract(BI)) {
      BI = BreakUpSubtract(BI, ValueRankMap);
      // Reset the BBI iterator in case BreakUpSubtract changed the
      // instruction it points to.
      BBI = BI;
      ++BBI;
      MadeChange = true;
    } else if (BinaryOperator::isNeg(BI)) {
      // Otherwise, this is a negation.  See if the operand is a multiply tree
      // and if this is not an inner node of a multiply tree.
      if (isReassociableOp(BI->getOperand(1), Instruction::Mul) &&
          (!BI->hasOneUse() ||
           !isReassociableOp(BI->use_back(), Instruction::Mul))) {
        BI = LowerNegateToMultiply(BI, ValueRankMap);
        MadeChange = true;
      }
    }
  }

  // If this instruction is a commutative binary operator, process it.
  if (!BI->isAssociative()) return;
  BinaryOperator *I = cast<BinaryOperator>(BI);

  // If this is an interior node of a reassociable tree, ignore it until we
  // get to the root of the tree, to avoid N^2 analysis.
  if (I->hasOneUse() && isReassociableOp(I->use_back(), I->getOpcode()))
    return;

  // If this is an add tree that is used by a sub instruction, ignore it
  // until we process the subtract.
  if (I->hasOneUse() && I->getOpcode() == Instruction::Add &&
      cast<Instruction>(I->use_back())->getOpcode() == Instruction::Sub)
    return;

  ReassociateExpression(I);
}

Value *Reassociate::ReassociateExpression(BinaryOperator *I) {

  // First, walk the expression tree, linearizing the tree, collecting the
  // operand information.
  SmallVector<ValueEntry, 8> Ops;
  LinearizeExprTree(I, Ops);

  // Now that we have linearized the tree to a list and have gathered all of
  // the operands and their ranks, sort the operands by their rank.  Use a
  // stable_sort so that values with equal ranks will have their relative
  // positions maintained (and so the compiler is deterministic).  Note that
  // this sorts so that the highest ranking values end up at the beginning of
  // the vector.
  std::stable_sort(Ops.begin(), Ops.end());

  DEBUG(dbgs() << "RAIn:\t"; PrintOps(I, Ops); dbgs() << '\n');

  // OptimizeExpression - Now that we have the expression tree in a convenient
  // sorted form, optimize it globally if possible.
  if (Value *V = OptimizeExpression(I, Ops)) {
    // This expression tree simplified to something that isn't a tree,
    // eliminate it.
    DEBUG(dbgs() << "Reassoc to scalar: " << *V << '\n');
    I->replaceAllUsesWith(V);
    if (Instruction *VI = dyn_cast<Instruction>(V))
      VI->setDebugLoc(I->getDebugLoc());
    RemoveDeadBinaryOp(I);
    ++NumAnnihil;
    return V;
  }

  // We want to sink immediates as deeply as possible except in the case where
  // this is a multiply tree used only by an add, and the immediate is a -1.
  // In this case we reassociate to put the negation on the outside so that we
  // can fold the negation into the add: (-X)*Y + Z -> Z-X*Y
  if (I->getOpcode() == Instruction::Mul && I->hasOneUse() &&
      cast<Instruction>(I->use_back())->getOpcode() == Instruction::Add &&
      isa<ConstantInt>(Ops.back().Op) &&
      cast<ConstantInt>(Ops.back().Op)->isAllOnesValue()) {
    ValueEntry Tmp = Ops.pop_back_val();
    Ops.insert(Ops.begin(), Tmp);
  }

  DEBUG(dbgs() << "RAOut:\t"; PrintOps(I, Ops); dbgs() << '\n');

  if (Ops.size() == 1) {
    // This expression tree simplified to something that isn't a tree,
    // eliminate it.
    I->replaceAllUsesWith(Ops[0].Op);
    if (Instruction *OI = dyn_cast<Instruction>(Ops[0].Op))
      OI->setDebugLoc(I->getDebugLoc());
    RemoveDeadBinaryOp(I);
    return Ops[0].Op;
  }

  // Now that we ordered and optimized the expressions, splat them back into
  // the expression tree, removing any unneeded nodes.
  RewriteExprTree(I, Ops);
  return I;
}

bool Reassociate::runOnFunction(Function &F) {
  // Recalculate the rank map for F
  BuildRankMap(F);

  MadeChange = false;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    for (BasicBlock::iterator BBI = FI->begin(); BBI != FI->end(); )
      ReassociateInst(BBI);

  // Now that we're done, revisit any instructions which are likely to
  // have secondary reassociation opportunities.
  while (!RedoInsts.empty())
    if (Value *V = RedoInsts.pop_back_val()) {
      BasicBlock::iterator BBI = cast<Instruction>(V);
      ReassociateInst(BBI);
    }

  // We are done with the rank map.
  RankMap.clear();
  ValueRankMap.clear();

  // Now that we're done, delete any instructions which are no longer used.
  while (!DeadInsts.empty())
    if (Value *V = DeadInsts.pop_back_val())
      RecursivelyDeleteTriviallyDeadInstructions(V);

  return MadeChange;
}
