//===- Reassociate.cpp - Reassociate binary expressions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass reassociates commutative expressions in an order that is designed
// to promote better constant propagation, GCSE, LICM, PRE...
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
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>
using namespace llvm;

namespace {
  Statistic<> NumLinear ("reassociate","Number of insts linearized");
  Statistic<> NumChanged("reassociate","Number of insts reassociated");
  Statistic<> NumSwapped("reassociate","Number of insts with operands swapped");

  struct ValueEntry {
    unsigned Rank;
    Value *Op;
    ValueEntry(unsigned R, Value *O) : Rank(R), Op(O) {}
  };
  inline bool operator<(const ValueEntry &LHS, const ValueEntry &RHS) {
    return LHS.Rank > RHS.Rank;   // Sort so that highest rank goes to start.
  }

  class Reassociate : public FunctionPass {
    std::map<BasicBlock*, unsigned> RankMap;
    std::map<Value*, unsigned> ValueRankMap;
    bool MadeChange;
  public:
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  private:
    void BuildRankMap(Function &F);
    unsigned getRank(Value *V);
    void RewriteExprTree(BinaryOperator *I, unsigned Idx,
                         std::vector<ValueEntry> &Ops);
    void OptimizeExpression(unsigned Opcode, std::vector<ValueEntry> &Ops);
    void LinearizeExprTree(BinaryOperator *I, std::vector<ValueEntry> &Ops);
    void LinearizeExpr(BinaryOperator *I);
    void ReassociateBB(BasicBlock *BB);
  };

  RegisterOpt<Reassociate> X("reassociate", "Reassociate expressions");
}

// Public interface to the Reassociate pass
FunctionPass *llvm::createReassociatePass() { return new Reassociate(); }

void Reassociate::BuildRankMap(Function &F) {
  unsigned i = 2;

  // Assign distinct ranks to function arguments
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
    ValueRankMap[I] = ++i;

  ReversePostOrderTraversal<Function*> RPOT(&F);
  for (ReversePostOrderTraversal<Function*>::rpo_iterator I = RPOT.begin(),
         E = RPOT.end(); I != E; ++I)
    RankMap[*I] = ++i << 16;
}

unsigned Reassociate::getRank(Value *V) {
  if (isa<Argument>(V)) return ValueRankMap[V];   // Function argument...

  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return 0;  // Otherwise it's a global or constant, rank 0.

  unsigned &CachedRank = ValueRankMap[I];
  if (CachedRank) return CachedRank;    // Rank already known?
  
  // If this is an expression, return the 1+MAX(rank(LHS), rank(RHS)) so that
  // we can reassociate expressions for code motion!  Since we do not recurse
  // for PHI nodes, we cannot have infinite recursion here, because there
  // cannot be loops in the value graph that do not go through PHI nodes.
  //
  if (I->getOpcode() == Instruction::PHI ||
      I->getOpcode() == Instruction::Alloca ||
      I->getOpcode() == Instruction::Malloc || isa<TerminatorInst>(I) ||
      I->mayWriteToMemory())  // Cannot move inst if it writes to memory!
    return RankMap[I->getParent()];
  
  // If not, compute it!
  unsigned Rank = 0, MaxRank = RankMap[I->getParent()];
  for (unsigned i = 0, e = I->getNumOperands();
       i != e && Rank != MaxRank; ++i)
    Rank = std::max(Rank, getRank(I->getOperand(i)));
  
  // If this is a not or neg instruction, do not count it for rank.  This
  // assures us that X and ~X will have the same rank.
  if (!I->getType()->isIntegral() ||
      (!BinaryOperator::isNot(I) && !BinaryOperator::isNeg(I)))
    ++Rank;

  DEBUG(std::cerr << "Calculated Rank[" << V->getName() << "] = "
        << Rank << "\n");
  
  return CachedRank = Rank;
}

/// isReassociableOp - Return true if V is an instruction of the specified
/// opcode and if it only has one use.
static BinaryOperator *isReassociableOp(Value *V, unsigned Opcode) {
  if (V->hasOneUse() && isa<Instruction>(V) &&
      cast<Instruction>(V)->getOpcode() == Opcode)
    return cast<BinaryOperator>(V);
  return 0;
}

// Given an expression of the form '(A+B)+(D+C)', turn it into '(((A+B)+C)+D)'.
// Note that if D is also part of the expression tree that we recurse to
// linearize it as well.  Besides that case, this does not recurse into A,B, or
// C.
void Reassociate::LinearizeExpr(BinaryOperator *I) {
  BinaryOperator *LHS = cast<BinaryOperator>(I->getOperand(0));
  BinaryOperator *RHS = cast<BinaryOperator>(I->getOperand(1));
  assert(isReassociableOp(LHS, I->getOpcode()) && 
         isReassociableOp(RHS, I->getOpcode()) &&
         "Not an expression that needs linearization?");

  DEBUG(std::cerr << "Linear" << *LHS << *RHS << *I);

  // Move the RHS instruction to live immediately before I, avoiding breaking
  // dominator properties.
  I->getParent()->getInstList().splice(I, RHS->getParent()->getInstList(), RHS);

  // Move operands around to do the linearization.
  I->setOperand(1, RHS->getOperand(0));
  RHS->setOperand(0, LHS);
  I->setOperand(0, RHS);
  
  ++NumLinear;
  MadeChange = true;
  DEBUG(std::cerr << "Linearized: " << *I);

  // If D is part of this expression tree, tail recurse.
  if (isReassociableOp(I->getOperand(1), I->getOpcode()))
    LinearizeExpr(I);
}


/// LinearizeExprTree - Given an associative binary expression tree, traverse
/// all of the uses putting it into canonical form.  This forces a left-linear
/// form of the the expression (((a+b)+c)+d), and collects information about the
/// rank of the non-tree operands.
///
/// This returns the rank of the RHS operand, which is known to be the highest
/// rank value in the expression tree.
///
void Reassociate::LinearizeExprTree(BinaryOperator *I,
                                    std::vector<ValueEntry> &Ops) {
  Value *LHS = I->getOperand(0), *RHS = I->getOperand(1);
  unsigned Opcode = I->getOpcode();

  // First step, linearize the expression if it is in ((A+B)+(C+D)) form.
  BinaryOperator *LHSBO = isReassociableOp(LHS, Opcode);
  BinaryOperator *RHSBO = isReassociableOp(RHS, Opcode);

  if (!LHSBO) {
    if (!RHSBO) {
      // Neither the LHS or RHS as part of the tree, thus this is a leaf.  As
      // such, just remember these operands and their rank.
      Ops.push_back(ValueEntry(getRank(LHS), LHS));
      Ops.push_back(ValueEntry(getRank(RHS), RHS));
      return;
    } else {
      // Turn X+(Y+Z) -> (Y+Z)+X
      std::swap(LHSBO, RHSBO);
      std::swap(LHS, RHS);
      bool Success = !I->swapOperands();
      assert(Success && "swapOperands failed");
      MadeChange = true;
    }
  } else if (RHSBO) {
    // Turn (A+B)+(C+D) -> (((A+B)+C)+D).  This guarantees the the RHS is not
    // part of the expression tree.
    LinearizeExpr(I);
    LHS = LHSBO = cast<BinaryOperator>(I->getOperand(0));
    RHS = I->getOperand(1);
    RHSBO = 0;
  }

  // Okay, now we know that the LHS is a nested expression and that the RHS is
  // not.  Perform reassociation.
  assert(!isReassociableOp(RHS, Opcode) && "LinearizeExpr failed!");

  // Move LHS right before I to make sure that the tree expression dominates all
  // values.
  I->getParent()->getInstList().splice(I,
                                      LHSBO->getParent()->getInstList(), LHSBO);

  // Linearize the expression tree on the LHS.
  LinearizeExprTree(LHSBO, Ops);

  // Remember the RHS operand and its rank.
  Ops.push_back(ValueEntry(getRank(RHS), RHS));
}

// RewriteExprTree - Now that the operands for this expression tree are
// linearized and optimized, emit them in-order.  This function is written to be
// tail recursive.
void Reassociate::RewriteExprTree(BinaryOperator *I, unsigned i,
                                  std::vector<ValueEntry> &Ops) {
  if (i+2 == Ops.size()) {
    if (I->getOperand(0) != Ops[i].Op ||
        I->getOperand(1) != Ops[i+1].Op) {
      DEBUG(std::cerr << "RA: " << *I);
      I->setOperand(0, Ops[i].Op);
      I->setOperand(1, Ops[i+1].Op);
      DEBUG(std::cerr << "TO: " << *I);
      MadeChange = true;
      ++NumChanged;
    }
    return;
  }
  assert(i+2 < Ops.size() && "Ops index out of range!");

  if (I->getOperand(1) != Ops[i].Op) {
    DEBUG(std::cerr << "RA: " << *I);
    I->setOperand(1, Ops[i].Op);
    DEBUG(std::cerr << "TO: " << *I);
    MadeChange = true;
    ++NumChanged;
  }
  RewriteExprTree(cast<BinaryOperator>(I->getOperand(0)), i+1, Ops);
}



// NegateValue - Insert instructions before the instruction pointed to by BI,
// that computes the negative version of the value specified.  The negative
// version of the value is returned, and BI is left pointing at the instruction
// that should be processed next by the reassociation pass.
//
static Value *NegateValue(Value *V, Instruction *BI) {
  // We are trying to expose opportunity for reassociation.  One of the things
  // that we want to do to achieve this is to push a negation as deep into an
  // expression chain as possible, to expose the add instructions.  In practice,
  // this means that we turn this:
  //   X = -(A+12+C+D)   into    X = -A + -12 + -C + -D = -12 + -A + -C + -D
  // so that later, a: Y = 12+X could get reassociated with the -12 to eliminate
  // the constants.  We assume that instcombine will clean up the mess later if
  // we introduce tons of unnecessary negation instructions...
  //
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (I->getOpcode() == Instruction::Add && I->hasOneUse()) {
      Value *RHS = NegateValue(I->getOperand(1), BI);
      Value *LHS = NegateValue(I->getOperand(0), BI);

      // We must actually insert a new add instruction here, because the neg
      // instructions do not dominate the old add instruction in general.  By
      // adding it now, we are assured that the neg instructions we just
      // inserted dominate the instruction we are about to insert after them.
      //
      return BinaryOperator::create(Instruction::Add, LHS, RHS,
                                    I->getName()+".neg", BI);
    }

  // Insert a 'neg' instruction that subtracts the value from zero to get the
  // negation.
  //
  return BinaryOperator::createNeg(V, V->getName() + ".neg", BI);
}

/// BreakUpSubtract - If we have (X-Y), and if either X is an add, or if this is
/// only used by an add, transform this into (X+(0-Y)) to promote better
/// reassociation.
static Instruction *BreakUpSubtract(Instruction *Sub) {
  // Reject cases where it is pointless to do this.
  if (Sub->getType()->isFloatingPoint())
    return 0;  // Floating point adds are not associative.

  // Don't bother to break this up unless either the LHS is an associable add or
  // if this is only used by one.
  if (!isReassociableOp(Sub->getOperand(0), Instruction::Add) &&
      !isReassociableOp(Sub->getOperand(1), Instruction::Add) &&
      !(Sub->hasOneUse() &&isReassociableOp(Sub->use_back(), Instruction::Add)))
    return 0;

  // Convert a subtract into an add and a neg instruction... so that sub
  // instructions can be commuted with other add instructions...
  //
  // Calculate the negative value of Operand 1 of the sub instruction...
  // and set it as the RHS of the add instruction we just made...
  //
  std::string Name = Sub->getName();
  Sub->setName("");
  Value *NegVal = NegateValue(Sub->getOperand(1), Sub);
  Instruction *New =
    BinaryOperator::createAdd(Sub->getOperand(0), NegVal, Name, Sub);

  // Everyone now refers to the add instruction.
  Sub->replaceAllUsesWith(New);
  Sub->eraseFromParent();
  
  DEBUG(std::cerr << "Negated: " << *New);
  return New;
}

/// ConvertShiftToMul - If this is a shift of a reassociable multiply or is used
/// by one, change this into a multiply by a constant to assist with further
/// reassociation.
static Instruction *ConvertShiftToMul(Instruction *Shl) {
  if (!isReassociableOp(Shl->getOperand(0), Instruction::Mul) &&
      !(Shl->hasOneUse() && isReassociableOp(Shl->use_back(),Instruction::Mul)))
    return 0;

  Constant *MulCst = ConstantInt::get(Shl->getType(), 1);
  MulCst = ConstantExpr::getShl(MulCst, cast<Constant>(Shl->getOperand(1)));

  std::string Name = Shl->getName();  Shl->setName("");
  Instruction *Mul = BinaryOperator::createMul(Shl->getOperand(0), MulCst,
                                               Name, Shl);
  Shl->replaceAllUsesWith(Mul);
  Shl->eraseFromParent();
  return Mul;
}

void Reassociate::OptimizeExpression(unsigned Opcode,
                                     std::vector<ValueEntry> &Ops) {
  // Now that we have the linearized expression tree, try to optimize it.
  // Start by folding any constants that we found.
 FoldConstants:
  if (Ops.size() == 1) return;

  if (Constant *V1 = dyn_cast<Constant>(Ops[Ops.size()-2].Op))
    if (Constant *V2 = dyn_cast<Constant>(Ops.back().Op)) {
      Ops.pop_back();
      Ops.back().Op = ConstantExpr::get(Opcode, V1, V2);
      goto FoldConstants;
    }

  // Check for destructive annihilation due to a constant being used.
  if (ConstantIntegral *CstVal = dyn_cast<ConstantIntegral>(Ops.back().Op))
    switch (Opcode) {
    default: break;
    case Instruction::And:
      if (CstVal->isNullValue()) {           // ... & 0 -> 0
        Ops[0].Op = CstVal;
        Ops.erase(Ops.begin()+1, Ops.end());
      } else if (CstVal->isAllOnesValue()) { // ... & -1 -> ...
        Ops.pop_back();
      }
      break;
    case Instruction::Mul:
      if (CstVal->isNullValue()) {           // ... * 0 -> 0
        Ops[0].Op = CstVal;
        Ops.erase(Ops.begin()+1, Ops.end());
      } else if (cast<ConstantInt>(CstVal)->getRawValue() == 1) {
        Ops.pop_back();                      // ... * 1 -> ...
      }
      break;
    case Instruction::Or:
      if (CstVal->isAllOnesValue()) {        // ... | -1 -> -1
        Ops[0].Op = CstVal;
        Ops.erase(Ops.begin()+1, Ops.end());
      }
      // FALLTHROUGH!
    case Instruction::Add:
    case Instruction::Xor:
      if (CstVal->isNullValue())             // ... [|^+] 0 -> ...
        Ops.pop_back();
      break;
    }

  // Handle destructive annihilation do to identities between elements in the
  // argument list here.
}


/// ReassociateBB - Inspect all of the instructions in this basic block,
/// reassociating them as we go.
void Reassociate::ReassociateBB(BasicBlock *BB) {
  for (BasicBlock::iterator BI = BB->begin(); BI != BB->end(); ++BI) {
    // If this is a subtract instruction which is not already in negate form,
    // see if we can convert it to X+-Y.
    if (BI->getOpcode() == Instruction::Sub && !BinaryOperator::isNeg(BI))
      if (Instruction *NI = BreakUpSubtract(BI)) {
        MadeChange = true;
        BI = NI;
      }
    if (BI->getOpcode() == Instruction::Shl &&
        isa<ConstantInt>(BI->getOperand(1)))
      if (Instruction *NI = ConvertShiftToMul(BI)) {
        MadeChange = true;
        BI = NI;
      }

    // If this instruction is a commutative binary operator, process it.
    if (!BI->isAssociative()) continue;
    BinaryOperator *I = cast<BinaryOperator>(BI);
    
    // If this is an interior node of a reassociable tree, ignore it until we
    // get to the root of the tree, to avoid N^2 analysis.
    if (I->hasOneUse() && isReassociableOp(I->use_back(), I->getOpcode()))
      continue;

    // First, walk the expression tree, linearizing the tree, collecting 
    std::vector<ValueEntry> Ops;
    LinearizeExprTree(I, Ops);

    // Now that we have linearized the tree to a list and have gathered all of
    // the operands and their ranks, sort the operands by their rank.  Use a
    // stable_sort so that values with equal ranks will have their relative
    // positions maintained (and so the compiler is deterministic).  Note that
    // this sorts so that the highest ranking values end up at the beginning of
    // the vector.
    std::stable_sort(Ops.begin(), Ops.end());

    // OptimizeExpression - Now that we have the expression tree in a convenient
    // sorted form, optimize it globally if possible.
    OptimizeExpression(I->getOpcode(), Ops);

    if (Ops.size() == 1) {
      // This expression tree simplified to something that isn't a tree,
      // eliminate it.
      I->replaceAllUsesWith(Ops[0].Op);
    } else {
      // Now that we ordered and optimized the expressions, splat them back into
      // the expression tree, removing any unneeded nodes.
      RewriteExprTree(I, 0, Ops);
    }
  }
}


bool Reassociate::runOnFunction(Function &F) {
  // Recalculate the rank map for F
  BuildRankMap(F);

  MadeChange = false;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    ReassociateBB(FI);

  // We are done with the rank map...
  RankMap.clear();
  ValueRankMap.clear();
  return MadeChange;
}

