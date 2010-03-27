//===------- ABCD.cpp - Removes redundant conditional branches ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass removes redundant branch instructions. This algorithm was
// described by Rastislav Bodik, Rajiv Gupta and Vivek Sarkar in their paper
// "ABCD: Eliminating Array Bounds Checks on Demand (2000)". The original
// Algorithm was created to remove array bound checks for strongly typed
// languages. This implementation expands the idea and removes any conditional
// branches that can be proved redundant, not only those used in array bound
// checks. With the SSI representation, each variable has a
// constraint. By analyzing these constraints we can prove that a branch is
// redundant. When a branch is proved redundant it means that
// one direction will always be taken; thus, we can change this branch into an
// unconditional jump.
// It is advisable to run SimplifyCFG and Aggressive Dead Code Elimination
// after ABCD to clean up the code.
// This implementation was created based on the implementation of the ABCD
// algorithm implemented for the compiler Jitrino.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "abcd"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/SSI.h"

using namespace llvm;

STATISTIC(NumBranchTested, "Number of conditional branches analyzed");
STATISTIC(NumBranchRemoved, "Number of conditional branches removed");

namespace {

class ABCD : public FunctionPass {
 public:
  static char ID;  // Pass identification, replacement for typeid.
  ABCD() : FunctionPass(&ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<SSI>();
  }

  bool runOnFunction(Function &F);

 private:
  /// Keep track of whether we've modified the program yet.
  bool modified;

  enum ProveResult {
    False = 0,
    Reduced = 1,
    True = 2
  };

  typedef ProveResult (*meet_function)(ProveResult, ProveResult);
  static ProveResult max(ProveResult res1, ProveResult res2) {
    return (ProveResult) std::max(res1, res2);
  }
  static ProveResult min(ProveResult res1, ProveResult res2) {
    return (ProveResult) std::min(res1, res2);
  }

  class Bound {
   public:
    Bound(APInt v, bool upper) : value(v), upper_bound(upper) {}
    Bound(const Bound &b, int cnst)
      : value(b.value - cnst), upper_bound(b.upper_bound) {}
    Bound(const Bound &b, const APInt &cnst)
      : value(b.value - cnst), upper_bound(b.upper_bound) {}

    /// Test if Bound is an upper bound
    bool isUpperBound() const { return upper_bound; }

    /// Get the bitwidth of this bound
    int32_t getBitWidth() const { return value.getBitWidth(); }

    /// Creates a Bound incrementing the one received
    static Bound createIncrement(const Bound &b) {
      return Bound(b.isUpperBound() ? b.value+1 : b.value-1,
                   b.upper_bound);
    }

    /// Creates a Bound decrementing the one received
    static Bound createDecrement(const Bound &b) {
      return Bound(b.isUpperBound() ? b.value-1 : b.value+1,
                   b.upper_bound);
    }

    /// Test if two bounds are equal
    static bool eq(const Bound *a, const Bound *b) {
      if (!a || !b) return false;

      assert(a->isUpperBound() == b->isUpperBound());
      return a->value == b->value;
    }

    /// Test if val is less than or equal to Bound b
    static bool leq(APInt val, const Bound &b) {
      return b.isUpperBound() ? val.sle(b.value) : val.sge(b.value);
    }

    /// Test if Bound a is less then or equal to Bound
    static bool leq(const Bound &a, const Bound &b) {
      assert(a.isUpperBound() == b.isUpperBound());
      return a.isUpperBound() ? a.value.sle(b.value) :
                                a.value.sge(b.value);
    }

    /// Test if Bound a is less then Bound b
    static bool lt(const Bound &a, const Bound &b) {
      assert(a.isUpperBound() == b.isUpperBound());
      return a.isUpperBound() ? a.value.slt(b.value) :
                                a.value.sgt(b.value);
    }

    /// Test if Bound b is greater then or equal val
    static bool geq(const Bound &b, APInt val) {
      return leq(val, b);
    }

    /// Test if Bound a is greater then or equal Bound b
    static bool geq(const Bound &a, const Bound &b) {
      return leq(b, a);
    }

   private:
    APInt value;
    bool upper_bound;
  };

  /// This class is used to store results some parts of the graph,
  /// so information does not need to be recalculated. The maximum false,
  /// minimum true and minimum reduced results are stored
  class MemoizedResultChart {
   public:
     MemoizedResultChart() {}
     MemoizedResultChart(const MemoizedResultChart &other) {
       if (other.max_false)
         max_false.reset(new Bound(*other.max_false));
       if (other.min_true)
         min_true.reset(new Bound(*other.min_true));
       if (other.min_reduced)
         min_reduced.reset(new Bound(*other.min_reduced));
     }

    /// Returns the max false
    const Bound *getFalse() const { return max_false.get(); }

    /// Returns the min true
    const Bound *getTrue() const { return min_true.get(); }

    /// Returns the min reduced
    const Bound *getReduced() const { return min_reduced.get(); }

    /// Return the stored result for this bound
    ProveResult getResult(const Bound &bound) const;

    /// Stores a false found
    void addFalse(const Bound &bound);

    /// Stores a true found
    void addTrue(const Bound &bound);

    /// Stores a Reduced found
    void addReduced(const Bound &bound);

    /// Clears redundant reduced
    /// If a min_true is smaller than a min_reduced then the min_reduced
    /// is unnecessary and then removed. It also works for min_reduced
    /// begin smaller than max_false.
    void clearRedundantReduced();

    void clear() {
      max_false.reset();
      min_true.reset();
      min_reduced.reset();
    }

  private:
    OwningPtr<Bound> max_false, min_true, min_reduced;
  };

  /// This class stores the result found for a node of the graph,
  /// so these results do not need to be recalculated, only searched for.
  class MemoizedResult {
  public:
    /// Test if there is true result stored from b to a
    /// that is less then the bound
    bool hasTrue(Value *b, const Bound &bound) const {
      const Bound *trueBound = map.lookup(b).getTrue();
      return trueBound && Bound::leq(*trueBound, bound);
    }

    /// Test if there is false result stored from b to a
    /// that is less then the bound
    bool hasFalse(Value *b, const Bound &bound) const {
      const Bound *falseBound = map.lookup(b).getFalse();
      return falseBound && Bound::leq(*falseBound, bound);
    }

    /// Test if there is reduced result stored from b to a
    /// that is less then the bound
    bool hasReduced(Value *b, const Bound &bound) const {
      const Bound *reducedBound = map.lookup(b).getReduced();
      return reducedBound && Bound::leq(*reducedBound, bound);
    }

    /// Returns the stored bound for b
    ProveResult getBoundResult(Value *b, const Bound &bound) {
      return map[b].getResult(bound);
    }

    /// Clears the map
    void clear() {
      DenseMapIterator<Value*, MemoizedResultChart> begin = map.begin();
      DenseMapIterator<Value*, MemoizedResultChart> end = map.end();
      for (; begin != end; ++begin) {
	begin->second.clear();
      }
      map.clear();
    }

    /// Stores the bound found
    void updateBound(Value *b, const Bound &bound, const ProveResult res);

  private:
    // Maps a nod in the graph with its results found.
    DenseMap<Value*, MemoizedResultChart> map;
  };

  /// This class represents an edge in the inequality graph used by the
  /// ABCD algorithm. An edge connects node v to node u with a value c if
  /// we could infer a constraint v <= u + c in the source program.
  class Edge {
  public:
    Edge(Value *V, APInt val, bool upper)
      : vertex(V), value(val), upper_bound(upper) {}

    Value *getVertex() const { return vertex; }
    const APInt &getValue() const { return value; }
    bool isUpperBound() const { return upper_bound; }

  private:
    Value *vertex;
    APInt value;
    bool upper_bound;
  };

  /// Weighted and Directed graph to represent constraints.
  /// There is one type of constraint, a <= b + X, which will generate an
  /// edge from b to a with weight X.
  class InequalityGraph {
  public:

    /// Adds an edge from V_from to V_to with weight value
    void addEdge(Value *V_from, Value *V_to, APInt value, bool upper);

    /// Test if there is a node V
    bool hasNode(Value *V) const { return graph.count(V); }

    /// Test if there is any edge from V in the upper direction
    bool hasEdge(Value *V, bool upper) const;

    /// Returns all edges pointed by vertex V
    SmallVector<Edge, 16> getEdges(Value *V) const {
      return graph.lookup(V);
    }

    /// Prints the graph in dot format.
    /// Blue edges represent upper bound and Red lower bound.
    void printGraph(raw_ostream &OS, Function &F) const {
      printHeader(OS, F);
      printBody(OS);
      printFooter(OS);
    }

    /// Clear the graph
    void clear() {
      graph.clear();
    }

  private:
    DenseMap<Value *, SmallVector<Edge, 16> > graph;

    /// Prints the header of the dot file
    void printHeader(raw_ostream &OS, Function &F) const;

    /// Prints the footer of the dot file
    void printFooter(raw_ostream &OS) const {
      OS << "}\n";
    }

    /// Prints the body of the dot file
    void printBody(raw_ostream &OS) const;

    /// Prints vertex source to the dot file
    void printVertex(raw_ostream &OS, Value *source) const;

    /// Prints the edge to the dot file
    void printEdge(raw_ostream &OS, Value *source, const Edge &edge) const;

    void printName(raw_ostream &OS, Value *info) const;
  };

  /// Iterates through all BasicBlocks, if the Terminator Instruction
  /// uses an Comparator Instruction, all operands of this comparator
  /// are sent to be transformed to SSI. Only Instruction operands are
  /// transformed.
  void createSSI(Function &F);

  /// Creates the graphs for this function.
  /// It will look for all comparators used in branches, and create them.
  /// These comparators will create constraints for any instruction as an
  /// operand.
  void executeABCD(Function &F);

  /// Seeks redundancies in the comparator instruction CI.
  /// If the ABCD algorithm can prove that the comparator CI always
  /// takes one way, then the Terminator Instruction TI is substituted from
  /// a conditional branch to a unconditional one.
  /// This code basically receives a comparator, and verifies which kind of
  /// instruction it is. Depending on the kind of instruction, we use different
  /// strategies to prove its redundancy.
  void seekRedundancy(ICmpInst *ICI, TerminatorInst *TI);

  /// Substitutes Terminator Instruction TI, that is a conditional branch,
  /// with one unconditional branch. Succ_edge determines if the new
  /// unconditional edge will be the first or second edge of the former TI
  /// instruction.
  void removeRedundancy(TerminatorInst *TI, bool Succ_edge);

  /// When an conditional branch is removed, the BasicBlock that is no longer
  /// reachable will have problems in phi functions. This method fixes these
  /// phis removing the former BasicBlock from the list of incoming BasicBlocks
  /// of all phis. In case the phi remains with no predecessor it will be
  /// marked to be removed later.
  void fixPhi(BasicBlock *BB, BasicBlock *Succ);

  /// Removes phis that have no predecessor
  void removePhis();

  /// Creates constraints for Instructions.
  /// If the constraint for this instruction has already been created
  /// nothing is done.
  void createConstraintInstruction(Instruction *I);

  /// Creates constraints for Binary Operators.
  /// It will create constraints only for addition and subtraction,
  /// the other binary operations are not treated by ABCD.
  /// For additions in the form a = b + X and a = X + b, where X is a constant,
  /// the constraint a <= b + X can be obtained. For this constraint, an edge
  /// a->b with weight X is added to the lower bound graph, and an edge
  /// b->a with weight -X is added to the upper bound graph.
  /// Only subtractions in the format a = b - X is used by ABCD.
  /// Edges are created using the same semantic as addition.
  void createConstraintBinaryOperator(BinaryOperator *BO);

  /// Creates constraints for Comparator Instructions.
  /// Only comparators that have any of the following operators
  /// are used to create constraints: >=, >, <=, <. And only if
  /// at least one operand is an Instruction. In a Comparator Instruction
  /// a op b, there will be 4 sigma functions a_t, a_f, b_t and b_f. Where
  /// t and f represent sigma for operands in true and false branches. The
  /// following constraints can be obtained. a_t <= a, a_f <= a, b_t <= b and
  /// b_f <= b. There are two more constraints that depend on the operator.
  /// For the operator <= : a_t <= b_t   and b_f <= a_f-1
  /// For the operator <  : a_t <= b_t-1 and b_f <= a_f
  /// For the operator >= : b_t <= a_t   and a_f <= b_f-1
  /// For the operator >  : b_t <= a_t-1 and a_f <= b_f
  void createConstraintCmpInst(ICmpInst *ICI, TerminatorInst *TI);

  /// Creates constraints for PHI nodes.
  /// In a PHI node a = phi(b,c) we can create the constraint
  /// a<= max(b,c). With this constraint there will be the edges,
  /// b->a and c->a with weight 0 in the lower bound graph, and the edges
  /// a->b and a->c with weight 0 in the upper bound graph.
  void createConstraintPHINode(PHINode *PN);

  /// Given a binary operator, we are only interest in the case
  /// that one operand is an Instruction and the other is a ConstantInt. In
  /// this case the method returns true, otherwise false. It also obtains the
  /// Instruction and ConstantInt from the BinaryOperator and returns it.
  bool createBinaryOperatorInfo(BinaryOperator *BO, Instruction **I1,
				Instruction **I2, ConstantInt **C1,
				ConstantInt **C2);

  /// This method creates a constraint between a Sigma and an Instruction.
  /// These constraints are created as soon as we find a comparator that uses a
  /// SSI variable.
  void createConstraintSigInst(Instruction *I_op, BasicBlock *BB_succ_t,
                               BasicBlock *BB_succ_f, PHINode **SIG_op_t,
                               PHINode **SIG_op_f);

  /// If PN_op1 and PN_o2 are different from NULL, create a constraint
  /// PN_op2 -> PN_op1 with value. In case any of them is NULL, replace
  /// with the respective V_op#, if V_op# is a ConstantInt.
  void createConstraintSigSig(PHINode *SIG_op1, PHINode *SIG_op2, 
                              ConstantInt *V_op1, ConstantInt *V_op2,
                              APInt value);

  /// Returns the sigma representing the Instruction I in BasicBlock BB.
  /// Returns NULL in case there is no sigma for this Instruction in this
  /// Basic Block. This methods assume that sigmas are the first instructions
  /// in a block, and that there can be only two sigmas in a block. So it will
  /// only look on the first two instructions of BasicBlock BB.
  PHINode *findSigma(BasicBlock *BB, Instruction *I);

  /// Original ABCD algorithm to prove redundant checks.
  /// This implementation works on any kind of inequality branch.
  bool demandProve(Value *a, Value *b, int c, bool upper_bound);

  /// Prove that distance between b and a is <= bound
  ProveResult prove(Value *a, Value *b, const Bound &bound, unsigned level);

  /// Updates the distance value for a and b
  void updateMemDistance(Value *a, Value *b, const Bound &bound, unsigned level,
                         meet_function meet);

  InequalityGraph inequality_graph;
  MemoizedResult mem_result;
  DenseMap<Value*, const Bound*> active;
  SmallPtrSet<Value*, 16> created;
  SmallVector<PHINode *, 16> phis_to_remove;
};

}  // end anonymous namespace.

char ABCD::ID = 0;
static RegisterPass<ABCD> X("abcd", "ABCD: Eliminating Array Bounds Checks on Demand");


bool ABCD::runOnFunction(Function &F) {
  modified = false;
  createSSI(F);
  executeABCD(F);
  DEBUG(inequality_graph.printGraph(dbgs(), F));
  removePhis();

  inequality_graph.clear();
  mem_result.clear();
  active.clear();
  created.clear();
  phis_to_remove.clear();
  return modified;
}

/// Iterates through all BasicBlocks, if the Terminator Instruction
/// uses an Comparator Instruction, all operands of this comparator
/// are sent to be transformed to SSI. Only Instruction operands are
/// transformed.
void ABCD::createSSI(Function &F) {
  SSI *ssi = &getAnalysis<SSI>();

  SmallVector<Instruction *, 16> Insts;

  for (Function::iterator begin = F.begin(), end = F.end();
       begin != end; ++begin) {
    BasicBlock *BB = begin;
    TerminatorInst *TI = BB->getTerminator();
    if (TI->getNumOperands() == 0)
      continue;

    if (ICmpInst *ICI = dyn_cast<ICmpInst>(TI->getOperand(0))) {
      if (Instruction *I = dyn_cast<Instruction>(ICI->getOperand(0))) {
        modified = true;  // XXX: but yet createSSI might do nothing
        Insts.push_back(I);
      }
      if (Instruction *I = dyn_cast<Instruction>(ICI->getOperand(1))) {
        modified = true;
        Insts.push_back(I);
      }
    }
  }
  ssi->createSSI(Insts);
}

/// Creates the graphs for this function.
/// It will look for all comparators used in branches, and create them.
/// These comparators will create constraints for any instruction as an
/// operand.
void ABCD::executeABCD(Function &F) {
  for (Function::iterator begin = F.begin(), end = F.end();
       begin != end; ++begin) {
    BasicBlock *BB = begin;
    TerminatorInst *TI = BB->getTerminator();
    if (TI->getNumOperands() == 0)
      continue;

    ICmpInst *ICI = dyn_cast<ICmpInst>(TI->getOperand(0));
    if (!ICI || !ICI->getOperand(0)->getType()->isIntegerTy())
      continue;

    createConstraintCmpInst(ICI, TI);
    seekRedundancy(ICI, TI);
  }
}

/// Seeks redundancies in the comparator instruction CI.
/// If the ABCD algorithm can prove that the comparator CI always
/// takes one way, then the Terminator Instruction TI is substituted from
/// a conditional branch to a unconditional one.
/// This code basically receives a comparator, and verifies which kind of
/// instruction it is. Depending on the kind of instruction, we use different
/// strategies to prove its redundancy.
void ABCD::seekRedundancy(ICmpInst *ICI, TerminatorInst *TI) {
  CmpInst::Predicate Pred = ICI->getPredicate();

  Value *source, *dest;
  int distance1, distance2;
  bool upper;

  switch(Pred) {
    case CmpInst::ICMP_SGT: // signed greater than
      upper = false;
      distance1 = 1;
      distance2 = 0;
      break;

    case CmpInst::ICMP_SGE: // signed greater or equal
      upper = false;
      distance1 = 0;
      distance2 = -1;
      break;

    case CmpInst::ICMP_SLT: // signed less than
      upper = true;
      distance1 = -1;
      distance2 = 0;
      break;

    case CmpInst::ICMP_SLE: // signed less or equal
      upper = true;
      distance1 = 0;
      distance2 = 1;
      break;

    default:
      return;
  }

  ++NumBranchTested;
  source = ICI->getOperand(0);
  dest = ICI->getOperand(1);
  if (demandProve(dest, source, distance1, upper)) {
    removeRedundancy(TI, true);
  } else if (demandProve(dest, source, distance2, !upper)) {
    removeRedundancy(TI, false);
  }
}

/// Substitutes Terminator Instruction TI, that is a conditional branch,
/// with one unconditional branch. Succ_edge determines if the new
/// unconditional edge will be the first or second edge of the former TI
/// instruction.
void ABCD::removeRedundancy(TerminatorInst *TI, bool Succ_edge) {
  BasicBlock *Succ;
  if (Succ_edge) {
    Succ = TI->getSuccessor(0);
    fixPhi(TI->getParent(), TI->getSuccessor(1));
  } else {
    Succ = TI->getSuccessor(1);
    fixPhi(TI->getParent(), TI->getSuccessor(0));
  }

  BranchInst::Create(Succ, TI);
  TI->eraseFromParent();  // XXX: invoke
  ++NumBranchRemoved;
  modified = true;
}

/// When an conditional branch is removed, the BasicBlock that is no longer
/// reachable will have problems in phi functions. This method fixes these
/// phis removing the former BasicBlock from the list of incoming BasicBlocks
/// of all phis. In case the phi remains with no predecessor it will be
/// marked to be removed later.
void ABCD::fixPhi(BasicBlock *BB, BasicBlock *Succ) {
  BasicBlock::iterator begin = Succ->begin();
  while (PHINode *PN = dyn_cast<PHINode>(begin++)) {
    PN->removeIncomingValue(BB, false);
    if (PN->getNumIncomingValues() == 0)
      phis_to_remove.push_back(PN);
  }
}

/// Removes phis that have no predecessor
void ABCD::removePhis() {
  for (unsigned i = 0, e = phis_to_remove.size(); i != e; ++i) {
    PHINode *PN = phis_to_remove[i];
    PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
    PN->eraseFromParent();
  }
}

/// Creates constraints for Instructions.
/// If the constraint for this instruction has already been created
/// nothing is done.
void ABCD::createConstraintInstruction(Instruction *I) {
  // Test if this instruction has not been created before
  if (created.insert(I)) {
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
      createConstraintBinaryOperator(BO);
    } else if (PHINode *PN = dyn_cast<PHINode>(I)) {
      createConstraintPHINode(PN);
    }
  }
}

/// Creates constraints for Binary Operators.
/// It will create constraints only for addition and subtraction,
/// the other binary operations are not treated by ABCD.
/// For additions in the form a = b + X and a = X + b, where X is a constant,
/// the constraint a <= b + X can be obtained. For this constraint, an edge
/// a->b with weight X is added to the lower bound graph, and an edge
/// b->a with weight -X is added to the upper bound graph.
/// Only subtractions in the format a = b - X is used by ABCD.
/// Edges are created using the same semantic as addition.
void ABCD::createConstraintBinaryOperator(BinaryOperator *BO) {
  Instruction *I1 = NULL, *I2 = NULL;
  ConstantInt *CI1 = NULL, *CI2 = NULL;

  // Test if an operand is an Instruction and the other is a Constant
  if (!createBinaryOperatorInfo(BO, &I1, &I2, &CI1, &CI2))
    return;

  Instruction *I = 0;
  APInt value;

  switch (BO->getOpcode()) {
    case Instruction::Add:
      if (I1) {
        I = I1;
        value = CI2->getValue();
      } else if (I2) {
        I = I2;
        value = CI1->getValue();
      }
      break;

    case Instruction::Sub:
      // Instructions like a = X-b, where X is a constant are not represented
      // in the graph.
      if (!I1)
        return;

      I = I1;
      value = -CI2->getValue();
      break;

    default:
      return;
  }

  inequality_graph.addEdge(I, BO, value, true);
  inequality_graph.addEdge(BO, I, -value, false);
  createConstraintInstruction(I);
}

/// Given a binary operator, we are only interest in the case
/// that one operand is an Instruction and the other is a ConstantInt. In
/// this case the method returns true, otherwise false. It also obtains the
/// Instruction and ConstantInt from the BinaryOperator and returns it.
bool ABCD::createBinaryOperatorInfo(BinaryOperator *BO, Instruction **I1,
                                    Instruction **I2, ConstantInt **C1,
                                    ConstantInt **C2) {
  Value *op1 = BO->getOperand(0);
  Value *op2 = BO->getOperand(1);

  if ((*I1 = dyn_cast<Instruction>(op1))) {
    if ((*C2 = dyn_cast<ConstantInt>(op2)))
      return true; // First is Instruction and second ConstantInt

    return false; // Both are Instruction
  } else {
    if ((*C1 = dyn_cast<ConstantInt>(op1)) &&
        (*I2 = dyn_cast<Instruction>(op2)))
      return true; // First is ConstantInt and second Instruction

    return false; // Both are not Instruction
  }
}

/// Creates constraints for Comparator Instructions.
/// Only comparators that have any of the following operators
/// are used to create constraints: >=, >, <=, <. And only if
/// at least one operand is an Instruction. In a Comparator Instruction
/// a op b, there will be 4 sigma functions a_t, a_f, b_t and b_f. Where
/// t and f represent sigma for operands in true and false branches. The
/// following constraints can be obtained. a_t <= a, a_f <= a, b_t <= b and
/// b_f <= b. There are two more constraints that depend on the operator.
/// For the operator <= : a_t <= b_t   and b_f <= a_f-1
/// For the operator <  : a_t <= b_t-1 and b_f <= a_f
/// For the operator >= : b_t <= a_t   and a_f <= b_f-1
/// For the operator >  : b_t <= a_t-1 and a_f <= b_f
void ABCD::createConstraintCmpInst(ICmpInst *ICI, TerminatorInst *TI) {
  Value *V_op1 = ICI->getOperand(0);
  Value *V_op2 = ICI->getOperand(1);

  if (!V_op1->getType()->isIntegerTy())
    return;

  Instruction *I_op1 = dyn_cast<Instruction>(V_op1);
  Instruction *I_op2 = dyn_cast<Instruction>(V_op2);

  // Test if at least one operand is an Instruction
  if (!I_op1 && !I_op2)
    return;

  BasicBlock *BB_succ_t = TI->getSuccessor(0);
  BasicBlock *BB_succ_f = TI->getSuccessor(1);

  PHINode *SIG_op1_t = NULL, *SIG_op1_f = NULL,
          *SIG_op2_t = NULL, *SIG_op2_f = NULL;

  createConstraintSigInst(I_op1, BB_succ_t, BB_succ_f, &SIG_op1_t, &SIG_op1_f);
  createConstraintSigInst(I_op2, BB_succ_t, BB_succ_f, &SIG_op2_t, &SIG_op2_f);

  int32_t width = cast<IntegerType>(V_op1->getType())->getBitWidth();
  APInt MinusOne = APInt::getAllOnesValue(width);
  APInt Zero = APInt::getNullValue(width);

  CmpInst::Predicate Pred = ICI->getPredicate();
  ConstantInt *CI1 = dyn_cast<ConstantInt>(V_op1);
  ConstantInt *CI2 = dyn_cast<ConstantInt>(V_op2);
  switch (Pred) {
  case CmpInst::ICMP_SGT:  // signed greater than
    createConstraintSigSig(SIG_op2_t, SIG_op1_t, CI2, CI1, MinusOne);
    createConstraintSigSig(SIG_op1_f, SIG_op2_f, CI1, CI2, Zero);
    break;

  case CmpInst::ICMP_SGE:  // signed greater or equal
    createConstraintSigSig(SIG_op2_t, SIG_op1_t, CI2, CI1, Zero);
    createConstraintSigSig(SIG_op1_f, SIG_op2_f, CI1, CI2, MinusOne);
    break;

  case CmpInst::ICMP_SLT:  // signed less than
    createConstraintSigSig(SIG_op1_t, SIG_op2_t, CI1, CI2, MinusOne);
    createConstraintSigSig(SIG_op2_f, SIG_op1_f, CI2, CI1, Zero);
    break;

  case CmpInst::ICMP_SLE:  // signed less or equal
    createConstraintSigSig(SIG_op1_t, SIG_op2_t, CI1, CI2, Zero);
    createConstraintSigSig(SIG_op2_f, SIG_op1_f, CI2, CI1, MinusOne);
    break;

  default:
    break;
  }

  if (I_op1)
    createConstraintInstruction(I_op1);
  if (I_op2)
    createConstraintInstruction(I_op2);
}

/// Creates constraints for PHI nodes.
/// In a PHI node a = phi(b,c) we can create the constraint
/// a<= max(b,c). With this constraint there will be the edges,
/// b->a and c->a with weight 0 in the lower bound graph, and the edges
/// a->b and a->c with weight 0 in the upper bound graph.
void ABCD::createConstraintPHINode(PHINode *PN) {
  // FIXME: We really want to disallow sigma nodes, but I don't know the best
  // way to detect the other than this.
  if (PN->getNumOperands() == 2) return;
  
  int32_t width = cast<IntegerType>(PN->getType())->getBitWidth();
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    Value *V = PN->getIncomingValue(i);
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      createConstraintInstruction(I);
    }
    inequality_graph.addEdge(V, PN, APInt(width, 0), true);
    inequality_graph.addEdge(V, PN, APInt(width, 0), false);
  }
}

/// This method creates a constraint between a Sigma and an Instruction.
/// These constraints are created as soon as we find a comparator that uses a
/// SSI variable.
void ABCD::createConstraintSigInst(Instruction *I_op, BasicBlock *BB_succ_t,
                                   BasicBlock *BB_succ_f, PHINode **SIG_op_t,
                                   PHINode **SIG_op_f) {
  *SIG_op_t = findSigma(BB_succ_t, I_op);
  *SIG_op_f = findSigma(BB_succ_f, I_op);

  if (*SIG_op_t) {
    int32_t width = cast<IntegerType>((*SIG_op_t)->getType())->getBitWidth();
    inequality_graph.addEdge(I_op, *SIG_op_t, APInt(width, 0), true);
    inequality_graph.addEdge(*SIG_op_t, I_op, APInt(width, 0), false);
  }
  if (*SIG_op_f) {
    int32_t width = cast<IntegerType>((*SIG_op_f)->getType())->getBitWidth();
    inequality_graph.addEdge(I_op, *SIG_op_f, APInt(width, 0), true);
    inequality_graph.addEdge(*SIG_op_f, I_op, APInt(width, 0), false);
  }
}

/// If PN_op1 and PN_o2 are different from NULL, create a constraint
/// PN_op2 -> PN_op1 with value. In case any of them is NULL, replace
/// with the respective V_op#, if V_op# is a ConstantInt.
void ABCD::createConstraintSigSig(PHINode *SIG_op1, PHINode *SIG_op2,
                                  ConstantInt *V_op1, ConstantInt *V_op2,
                                  APInt value) {
  if (SIG_op1 && SIG_op2) {
    inequality_graph.addEdge(SIG_op2, SIG_op1, value, true);
    inequality_graph.addEdge(SIG_op1, SIG_op2, -value, false);
  } else if (SIG_op1 && V_op2) {
    inequality_graph.addEdge(V_op2, SIG_op1, value, true);
    inequality_graph.addEdge(SIG_op1, V_op2, -value, false);
  } else if (SIG_op2 && V_op1) {
    inequality_graph.addEdge(SIG_op2, V_op1, value, true);
    inequality_graph.addEdge(V_op1, SIG_op2, -value, false);
  }
}

/// Returns the sigma representing the Instruction I in BasicBlock BB.
/// Returns NULL in case there is no sigma for this Instruction in this
/// Basic Block. This methods assume that sigmas are the first instructions
/// in a block, and that there can be only two sigmas in a block. So it will
/// only look on the first two instructions of BasicBlock BB.
PHINode *ABCD::findSigma(BasicBlock *BB, Instruction *I) {
  // BB has more than one predecessor, BB cannot have sigmas.
  if (I == NULL || BB->getSinglePredecessor() == NULL)
    return NULL;

  BasicBlock::iterator begin = BB->begin();
  BasicBlock::iterator end = BB->end();

  for (unsigned i = 0; i < 2 && begin != end; ++i, ++begin) {
    Instruction *I_succ = begin;
    if (PHINode *PN = dyn_cast<PHINode>(I_succ))
      if (PN->getIncomingValue(0) == I)
        return PN;
  }

  return NULL;
}

/// Original ABCD algorithm to prove redundant checks.
/// This implementation works on any kind of inequality branch.
bool ABCD::demandProve(Value *a, Value *b, int c, bool upper_bound) {
  int32_t width = cast<IntegerType>(a->getType())->getBitWidth();
  Bound bound(APInt(width, c), upper_bound);

  mem_result.clear();
  active.clear();

  ProveResult res = prove(a, b, bound, 0);
  return res != False;
}

/// Prove that distance between b and a is <= bound
ABCD::ProveResult ABCD::prove(Value *a, Value *b, const Bound &bound,
                              unsigned level) {
  // if (C[b-a<=e] == True for some e <= bound
  // Same or stronger difference was already proven
  if (mem_result.hasTrue(b, bound))
    return True;

  // if (C[b-a<=e] == False for some e >= bound
  // Same or weaker difference was already disproved
  if (mem_result.hasFalse(b, bound))
    return False;

  // if (C[b-a<=e] == Reduced for some e <= bound
  // b is on a cycle that was reduced for same or stronger difference
  if (mem_result.hasReduced(b, bound))
    return Reduced;

  // traversal reached the source vertex
  if (a == b && Bound::geq(bound, APInt(bound.getBitWidth(), 0, true)))
    return True;

  // if b has no predecessor then fail
  if (!inequality_graph.hasEdge(b, bound.isUpperBound()))
    return False;

  // a cycle was encountered
  if (active.count(b)) {
    if (Bound::leq(*active.lookup(b), bound))
      return Reduced; // a "harmless" cycle

    return False; // an amplifying cycle
  }

  active[b] = &bound;
  PHINode *PN = dyn_cast<PHINode>(b);

  // Test if a Value is a Phi. If it is a PHINode with more than 1 incoming
  // value, then it is a phi, if it has 1 incoming value it is a sigma.
  if (PN && PN->getNumIncomingValues() > 1)
    updateMemDistance(a, b, bound, level, min);
  else
    updateMemDistance(a, b, bound, level, max);

  active.erase(b);

  ABCD::ProveResult res = mem_result.getBoundResult(b, bound);
  return res;
}

/// Updates the distance value for a and b
void ABCD::updateMemDistance(Value *a, Value *b, const Bound &bound,
                             unsigned level, meet_function meet) {
  ABCD::ProveResult res = (meet == max) ? False : True;

  SmallVector<Edge, 16> Edges = inequality_graph.getEdges(b);
  SmallVector<Edge, 16>::iterator begin = Edges.begin(), end = Edges.end();

  for (; begin != end ; ++begin) {
    if (((res >= Reduced) && (meet == max)) ||
       ((res == False) && (meet == min))) {
      break;
    }
    const Edge &in = *begin;
    if (in.isUpperBound() == bound.isUpperBound()) {
      Value *succ = in.getVertex();
      res = meet(res, prove(a, succ, Bound(bound, in.getValue()),
                            level+1));
    }
  }

  mem_result.updateBound(b, bound, res);
}

/// Return the stored result for this bound
ABCD::ProveResult ABCD::MemoizedResultChart::getResult(const Bound &bound)const{
  if (max_false && Bound::leq(bound, *max_false))
    return False;
  if (min_true && Bound::leq(*min_true, bound))
    return True;
  if (min_reduced && Bound::leq(*min_reduced, bound))
    return Reduced;
  return False;
}

/// Stores a false found
void ABCD::MemoizedResultChart::addFalse(const Bound &bound) {
  if (!max_false || Bound::leq(*max_false, bound))
    max_false.reset(new Bound(bound));

  if (Bound::eq(max_false.get(), min_reduced.get()))
    min_reduced.reset(new Bound(Bound::createIncrement(*min_reduced)));
  if (Bound::eq(max_false.get(), min_true.get()))
    min_true.reset(new Bound(Bound::createIncrement(*min_true)));
  if (Bound::eq(min_reduced.get(), min_true.get()))
    min_reduced.reset();
  clearRedundantReduced();
}

/// Stores a true found
void ABCD::MemoizedResultChart::addTrue(const Bound &bound) {
  if (!min_true || Bound::leq(bound, *min_true))
    min_true.reset(new Bound(bound));

  if (Bound::eq(min_true.get(), min_reduced.get()))
    min_reduced.reset(new Bound(Bound::createDecrement(*min_reduced)));
  if (Bound::eq(min_true.get(), max_false.get()))
    max_false.reset(new Bound(Bound::createDecrement(*max_false)));
  if (Bound::eq(max_false.get(), min_reduced.get()))
    min_reduced.reset();
  clearRedundantReduced();
}

/// Stores a Reduced found
void ABCD::MemoizedResultChart::addReduced(const Bound &bound) {
  if (!min_reduced || Bound::leq(bound, *min_reduced))
    min_reduced.reset(new Bound(bound));

  if (Bound::eq(min_reduced.get(), min_true.get()))
    min_true.reset(new Bound(Bound::createIncrement(*min_true)));
  if (Bound::eq(min_reduced.get(), max_false.get()))
    max_false.reset(new Bound(Bound::createDecrement(*max_false)));
}

/// Clears redundant reduced
/// If a min_true is smaller than a min_reduced then the min_reduced
/// is unnecessary and then removed. It also works for min_reduced
/// begin smaller than max_false.
void ABCD::MemoizedResultChart::clearRedundantReduced() {
  if (min_true && min_reduced && Bound::lt(*min_true, *min_reduced))
    min_reduced.reset();
  if (max_false && min_reduced && Bound::lt(*min_reduced, *max_false))
    min_reduced.reset();
}

/// Stores the bound found
void ABCD::MemoizedResult::updateBound(Value *b, const Bound &bound,
                                       const ProveResult res) {
  if (res == False) {
    map[b].addFalse(bound);
  } else if (res == True) {
    map[b].addTrue(bound);
  } else {
    map[b].addReduced(bound);
  }
}

/// Adds an edge from V_from to V_to with weight value
void ABCD::InequalityGraph::addEdge(Value *V_to, Value *V_from,
                                    APInt value, bool upper) {
  assert(V_from->getType() == V_to->getType());
  assert(cast<IntegerType>(V_from->getType())->getBitWidth() ==
         value.getBitWidth());

  graph[V_from].push_back(Edge(V_to, value, upper));
}

/// Test if there is any edge from V in the upper direction
bool ABCD::InequalityGraph::hasEdge(Value *V, bool upper) const {
  SmallVector<Edge, 16> it = graph.lookup(V);

  SmallVector<Edge, 16>::iterator begin = it.begin();
  SmallVector<Edge, 16>::iterator end = it.end();
  for (; begin != end; ++begin) {
    if (begin->isUpperBound() == upper) {
      return true;
    }
  }
  return false;
}

/// Prints the header of the dot file
void ABCD::InequalityGraph::printHeader(raw_ostream &OS, Function &F) const {
  OS << "digraph dotgraph {\n";
  OS << "label=\"Inequality Graph for \'";
  OS << F.getNameStr() << "\' function\";\n";
  OS << "node [shape=record,fontname=\"Times-Roman\",fontsize=14];\n";
}

/// Prints the body of the dot file
void ABCD::InequalityGraph::printBody(raw_ostream &OS) const {
  DenseMap<Value *, SmallVector<Edge, 16> >::const_iterator begin =
      graph.begin(), end = graph.end();

  for (; begin != end ; ++begin) {
    SmallVector<Edge, 16>::const_iterator begin_par =
        begin->second.begin(), end_par = begin->second.end();
    Value *source = begin->first;

    printVertex(OS, source);

    for (; begin_par != end_par ; ++begin_par) {
      const Edge &edge = *begin_par;
      printEdge(OS, source, edge);
    }
  }
}

/// Prints vertex source to the dot file
///
void ABCD::InequalityGraph::printVertex(raw_ostream &OS, Value *source) const {
  OS << "\"";
  printName(OS, source);
  OS << "\"";
  OS << " [label=\"{";
  printName(OS, source);
  OS << "}\"];\n";
}

/// Prints the edge to the dot file
void ABCD::InequalityGraph::printEdge(raw_ostream &OS, Value *source,
                                      const Edge &edge) const {
  Value *dest = edge.getVertex();
  APInt value = edge.getValue();
  bool upper = edge.isUpperBound();

  OS << "\"";
  printName(OS, source);
  OS << "\"";
  OS << " -> ";
  OS << "\"";
  printName(OS, dest);
  OS << "\"";
  OS << " [label=\"" << value << "\"";
  if (upper) {
    OS << "color=\"blue\"";
  } else {
    OS << "color=\"red\"";
  }
  OS << "];\n";
}

void ABCD::InequalityGraph::printName(raw_ostream &OS, Value *info) const {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(info)) {
    OS << *CI;
  } else {
    if (!info->hasName()) {
      info->setName("V");
    }
    OS << info->getNameStr();
  }
}

/// createABCDPass - The public interface to this file...
FunctionPass *llvm::createABCDPass() {
  return new ABCD();
}
