//===- Reassociate.cpp - Reassociate binary expressions -------------------===//
//
// This pass reassociates commutative expressions in an order that is designed
// to promote better constant propagation, GCSE, LICM, PRE...
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
// This code was originally written by Chris Lattner, and was then cleaned up
// and perfected by Casey Carter.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/iOperators.h"
#include "llvm/Type.h"
#include "llvm/Pass.h"
#include "llvm/Constant.h"
#include "llvm/Support/CFG.h"
#include "Support/Debug.h"
#include "Support/PostOrderIterator.h"
#include "Support/Statistic.h"

namespace {
  Statistic<> NumLinear ("reassociate","Number of insts linearized");
  Statistic<> NumChanged("reassociate","Number of insts reassociated");
  Statistic<> NumSwapped("reassociate","Number of insts with operands swapped");

  class Reassociate : public FunctionPass {
    std::map<BasicBlock*, unsigned> RankMap;
    std::map<Value*, unsigned> ValueRankMap;
  public:
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  private:
    void BuildRankMap(Function &F);
    unsigned getRank(Value *V);
    bool ReassociateExpr(BinaryOperator *I);
    bool ReassociateBB(BasicBlock *BB);
  };

  RegisterOpt<Reassociate> X("reassociate", "Reassociate expressions");
}

Pass *createReassociatePass() { return new Reassociate(); }

void Reassociate::BuildRankMap(Function &F) {
  unsigned i = 2;

  // Assign distinct ranks to function arguments
  for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
    ValueRankMap[I] = ++i;

  ReversePostOrderTraversal<Function*> RPOT(&F);
  for (ReversePostOrderTraversal<Function*>::rpo_iterator I = RPOT.begin(),
         E = RPOT.end(); I != E; ++I)
    RankMap[*I] = ++i << 16;
}

unsigned Reassociate::getRank(Value *V) {
  if (isa<Argument>(V)) return ValueRankMap[V];   // Function argument...

  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If this is an expression, return the 1+MAX(rank(LHS), rank(RHS)) so that
    // we can reassociate expressions for code motion!  Since we do not recurse
    // for PHI nodes, we cannot have infinite recursion here, because there
    // cannot be loops in the value graph that do not go through PHI nodes.
    //
    if (I->getOpcode() == Instruction::PHINode ||
        I->getOpcode() == Instruction::Alloca ||
        I->getOpcode() == Instruction::Malloc || isa<TerminatorInst>(I) ||
        I->mayWriteToMemory())  // Cannot move inst if it writes to memory!
      return RankMap[I->getParent()];

    unsigned &CachedRank = ValueRankMap[I];
    if (CachedRank) return CachedRank;    // Rank already known?

    // If not, compute it!
    unsigned Rank = 0, MaxRank = RankMap[I->getParent()];
    for (unsigned i = 0, e = I->getNumOperands();
         i != e && Rank != MaxRank; ++i)
      Rank = std::max(Rank, getRank(I->getOperand(i)));

    DEBUG(std::cerr << "Calculated Rank[" << V->getName() << "] = "
                    << Rank+1 << "\n");

    return CachedRank = Rank+1;
  }

  // Otherwise it's a global or constant, rank 0.
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
    bool Success = !I->swapOperands();
    assert(Success && "swapOperands failed");

    std::swap(LHS, RHS);
    std::swap(LHSRank, RHSRank);
    Changed = true;
    ++NumSwapped;
    DEBUG(std::cerr << "Transposed: " << I
          /* << " Result BB: " << I->getParent()*/);
  }
  
  // If the LHS is the same operator as the current one is, and if we are the
  // only expression using it...
  //
  if (BinaryOperator *LHSI = dyn_cast<BinaryOperator>(LHS))
    if (LHSI->getOpcode() == I->getOpcode() && LHSI->use_size() == 1) {
      // If the rank of our current RHS is less than the rank of the LHS's LHS,
      // then we reassociate the two instructions...

      unsigned TakeOp = 0;
      if (BinaryOperator *IOp = dyn_cast<BinaryOperator>(LHSI->getOperand(0)))
        if (IOp->getOpcode() == LHSI->getOpcode())
          TakeOp = 1;   // Hoist out non-tree portion

      if (RHSRank < getRank(LHSI->getOperand(TakeOp))) {
        // Convert ((a + 12) + 10) into (a + (12 + 10))
        I->setOperand(0, LHSI->getOperand(TakeOp));
        LHSI->setOperand(TakeOp, RHS);
        I->setOperand(1, LHSI);

        // Move the LHS expression forward, to ensure that it is dominated by
        // its operands.
        LHSI->getParent()->getInstList().remove(LHSI);
        I->getParent()->getInstList().insert(I, LHSI);

        ++NumChanged;
        DEBUG(std::cerr << "Reassociated: " << I/* << " Result BB: "
                                                   << I->getParent()*/);

        // Since we modified the RHS instruction, make sure that we recheck it.
        ReassociateExpr(LHSI);
        ReassociateExpr(I);
        return true;
      }
    }

  return Changed;
}


// NegateValue - Insert instructions before the instruction pointed to by BI,
// that computes the negative version of the value specified.  The negative
// version of the value is returned, and BI is left pointing at the instruction
// that should be processed next by the reassociation pass.
//
static Value *NegateValue(Value *V, BasicBlock::iterator &BI) {
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
    if (I->getOpcode() == Instruction::Add && I->use_size() == 1) {
      Value *RHS = NegateValue(I->getOperand(1), BI);
      Value *LHS = NegateValue(I->getOperand(0), BI);

      // We must actually insert a new add instruction here, because the neg
      // instructions do not dominate the old add instruction in general.  By
      // adding it now, we are assured that the neg instructions we just
      // inserted dominate the instruction we are about to insert after them.
      //
      return BinaryOperator::create(Instruction::Add, LHS, RHS,
                                    I->getName()+".neg",
                                    cast<Instruction>(RHS)->getNext());
    }

  // Insert a 'neg' instruction that subtracts the value from zero to get the
  // negation.
  //
  return BI = BinaryOperator::createNeg(V, V->getName() + ".neg", BI);
}


bool Reassociate::ReassociateBB(BasicBlock *BB) {
  bool Changed = false;
  for (BasicBlock::iterator BI = BB->begin(); BI != BB->end(); ++BI) {

    DEBUG(std::cerr << "Processing: " << *BI);
    if (BI->getOpcode() == Instruction::Sub && !BinaryOperator::isNeg(BI)) {
      // Convert a subtract into an add and a neg instruction... so that sub
      // instructions can be commuted with other add instructions...
      //
      // Calculate the negative value of Operand 1 of the sub instruction...
      // and set it as the RHS of the add instruction we just made...
      //
      std::string Name = BI->getName();
      BI->setName("");
      Instruction *New =
        BinaryOperator::create(Instruction::Add, BI->getOperand(0),
                               BI->getOperand(1), Name, BI);

      // Everyone now refers to the add instruction...
      BI->replaceAllUsesWith(New);

      // Put the new add in the place of the subtract... deleting the subtract
      BB->getInstList().erase(BI);

      BI = New;
      New->setOperand(1, NegateValue(New->getOperand(1), BI));
      
      Changed = true;
      DEBUG(std::cerr << "Negated: " << New /*<< " Result BB: " << BB*/);
    }

    // If this instruction is a commutative binary operator, and the ranks of
    // the two operands are sorted incorrectly, fix it now.
    //
    if (BI->isAssociative()) {
      BinaryOperator *I = cast<BinaryOperator>(BI);
      if (!I->use_empty()) {
        // Make sure that we don't have a tree-shaped computation.  If we do,
        // linearize it.  Convert (A+B)+(C+D) into ((A+B)+C)+D
        //
        Instruction *LHSI = dyn_cast<Instruction>(I->getOperand(0));
        Instruction *RHSI = dyn_cast<Instruction>(I->getOperand(1));
        if (LHSI && (int)LHSI->getOpcode() == I->getOpcode() &&
            RHSI && (int)RHSI->getOpcode() == I->getOpcode() &&
            RHSI->use_size() == 1) {
          // Insert a new temporary instruction... (A+B)+C
          BinaryOperator *Tmp = BinaryOperator::create(I->getOpcode(), LHSI,
                                                       RHSI->getOperand(0),
                                                       RHSI->getName()+".ra",
                                                       BI);
          BI = Tmp;
          I->setOperand(0, Tmp);
          I->setOperand(1, RHSI->getOperand(1));

          // Process the temporary instruction for reassociation now.
          I = Tmp;
          ++NumLinear;
          Changed = true;
          DEBUG(std::cerr << "Linearized: " << I/* << " Result BB: " << BB*/);
        }

        // Make sure that this expression is correctly reassociated with respect
        // to it's used values...
        //
        Changed |= ReassociateExpr(I);
      }
    }
  }

  return Changed;
}


bool Reassociate::runOnFunction(Function &F) {
  // Recalculate the rank map for F
  BuildRankMap(F);

  bool Changed = false;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    Changed |= ReassociateBB(FI);

  // We are done with the rank map...
  RankMap.clear();
  ValueRankMap.clear();
  return Changed;
}
