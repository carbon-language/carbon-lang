//===- Reassociate.cpp - Reassociate binary expressions -------------------===//
//
// This pass reassociates commutative expressions in an order that is designed
// to promote better constant propogation, GCSE, LICM, PRE...
//
// For example: 4 + (x + 5) -> x + (4 + 5)
//
// Note that this pass works best if left shifts have been promoted to explicit
// multiplies before this pass executes.
//
// In the implementation of this algorithm, constants are assigned rank = 0,
// function arguments are rank = 1, and other values are assigned ranks
// corresponding to the reverse post order traversal of current function
// (starting at 2), which effectively gives values in deep loops higher rank
// than values not in loops.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/iOperators.h"
#include "llvm/Type.h"
#include "llvm/Pass.h"
#include "llvm/Constant.h"
#include "llvm/Support/CFG.h"
#include "Support/PostOrderIterator.h"

namespace {
  class Reassociate : public FunctionPass {
    map<BasicBlock*, unsigned> RankMap;
  public:
    const char *getPassName() const {
      return "Expression Reassociation";
    }

    bool runOnFunction(Function *F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.preservesCFG();
    }
  private:
    void BuildRankMap(Function *F);
    unsigned getRank(Value *V);
    bool ReassociateExpr(BinaryOperator *I);
    bool ReassociateBB(BasicBlock *BB);
  };
}

Pass *createReassociatePass() { return new Reassociate(); }

void Reassociate::BuildRankMap(Function *F) {
  unsigned i = 1;
  ReversePostOrderTraversal<Function*> RPOT(F);
  for (ReversePostOrderTraversal<Function*>::rpo_iterator I = RPOT.begin(),
         E = RPOT.end(); I != E; ++I)
    RankMap[*I] = ++i;
}

unsigned Reassociate::getRank(Value *V) {
  if (isa<Argument>(V)) return 1;   // Function argument...
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If this is an expression, return the MAX(rank(LHS), rank(RHS)) so that we
    // can reassociate expressions for code motion!  Since we do not recurse for
    // PHI nodes, we cannot have infinite recursion here, because there cannot
    // be loops in the value graph (except for PHI nodes).
    //
    if (I->getOpcode() == Instruction::PHINode ||
        I->getOpcode() == Instruction::Alloca ||
        I->getOpcode() == Instruction::Malloc || isa<TerminatorInst>(I) ||
        I->hasSideEffects())
      return RankMap[I->getParent()];

    unsigned Rank = 0;
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
      Rank = std::max(Rank, getRank(I->getOperand(i)));

    return Rank;
  }

  // Otherwise it's a global or constant, rank 0.
  return 0;
}


// isCommutativeOperator - Return true if the specified instruction is
// commutative and associative.  If the instruction is not commutative and
// associative, we can not reorder its operands!
//
static inline BinaryOperator *isCommutativeOperator(Instruction *I) {
  // Floating point operations do not commute!
  if (I->getType()->isFloatingPoint()) return 0;

  if (I->getOpcode() == Instruction::Add || 
      I->getOpcode() == Instruction::Mul ||
      I->getOpcode() == Instruction::And || 
      I->getOpcode() == Instruction::Or  ||
      I->getOpcode() == Instruction::Xor)
    return cast<BinaryOperator>(I);
  return 0;    
}


bool Reassociate::ReassociateExpr(BinaryOperator *I) {
  Value *LHS = I->getOperand(0);
  Value *RHS = I->getOperand(1);
  unsigned LHSRank = getRank(LHS);
  unsigned RHSRank = getRank(RHS);
  
  bool Changed = false;

  // Make sure the LHS of the operand always has the greater rank...
  if (LHSRank < RHSRank) {
    I->swapOperands();
    std::swap(LHS, RHS);
    std::swap(LHSRank, RHSRank);
    Changed = true;
    //cerr << "Transposed: " << I << " Result BB: " << I->getParent();
  }
  
  // If the LHS is the same operator as the current one is, and if we are the
  // only expression using it...
  //
  if (BinaryOperator *LHSI = dyn_cast<BinaryOperator>(LHS))
    if (LHSI->getOpcode() == I->getOpcode() && LHSI->use_size() == 1) {
      // If the rank of our current RHS is less than the rank of the LHS's LHS,
      // then we reassociate the two instructions...
      if (RHSRank < getRank(LHSI->getOperand(0))) {
        unsigned TakeOp = 0;
        if (BinaryOperator *IOp = dyn_cast<BinaryOperator>(LHSI->getOperand(0)))
          if (IOp->getOpcode() == LHSI->getOpcode())
            TakeOp = 1;   // Hoist out non-tree portion

        // Convert ((a + 12) + 10) into (a + (12 + 10))
        I->setOperand(0, LHSI->getOperand(TakeOp));
        LHSI->setOperand(TakeOp, RHS);
        I->setOperand(1, LHSI);

        //cerr << "Reassociated: " << I << " Result BB: " << I->getParent();

        // Since we modified the RHS instruction, make sure that we recheck it.
        ReassociateExpr(LHSI);
        return true;
      }
    }

  return Changed;
}


bool Reassociate::ReassociateBB(BasicBlock *BB) {
  bool Changed = false;
  for (BasicBlock::iterator BI = BB->begin(); BI != BB->end(); ++BI) {
    Instruction *Inst = *BI;

    // If this instruction is a commutative binary operator, and the ranks of
    // the two operands are sorted incorrectly, fix it now.
    //
    if (BinaryOperator *I = isCommutativeOperator(Inst)) {
      // Make sure that this expression is correctly reassociated with respect
      // to it's used values...
      //
      Changed |= ReassociateExpr(I);

    } else if (Inst->getOpcode() == Instruction::Sub &&
               Inst->getOperand(0) != Constant::getNullValue(Inst->getType())) {
      // Convert a subtract into an add and a neg instruction... so that sub
      // instructions can be commuted with other add instructions...
      //
      Instruction *New = BinaryOperator::create(Instruction::Add,
                                                Inst->getOperand(0), Inst,
                                                Inst->getName());
      // Everyone now refers to the add instruction...
      Inst->replaceAllUsesWith(New);
      Inst->setName(Inst->getOperand(1)->getName()+".neg");
      New->setOperand(1, Inst);        // Except for the add inst itself!

      BI = BB->getInstList().insert(BI+1, New)-1;  // Add to the basic block...
      Inst->setOperand(0, Constant::getNullValue(Inst->getType()));
      Changed = true;
    }
  }

  return Changed;
}


bool Reassociate::runOnFunction(Function *F) {
  // Recalculate the rank map for F
  BuildRankMap(F);

  bool Changed = false;
  for (Function::iterator FI = F->begin(), FE = F->end(); FI != FE; ++FI)
    Changed |= ReassociateBB(*FI);

  // We are done with the rank map...
  RankMap.clear();
  return Changed;
}
