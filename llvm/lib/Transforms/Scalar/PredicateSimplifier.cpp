//===-- PredicateSimplifier.cpp - Path Sensitive Simplifier -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nick Lewycky and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===------------------------------------------------------------------===//
//
// Path-sensitive optimizer. In a branch where x == y, replace uses of
// x with y. Permits further optimization, such as the elimination of
// the unreachable call:
//
// void test(int *p, int *q)
// {
//   if (p != q)
//     return;
// 
//   if (*p != *q)
//     foo(); // unreachable
// }
//
//===------------------------------------------------------------------===//
//
// This optimization works by substituting %q for %p when protected by a
// conditional that assures us of that fact. Properties are stored as
// relationships between two values.
//
//===------------------------------------------------------------------===//

#define DEBUG_TYPE "predsimplify"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include <iostream>
using namespace llvm;

typedef DominatorTree::Node DTNodeType;

namespace {
  Statistic<>
  NumVarsReplaced("predsimplify", "Number of argument substitutions");
  Statistic<>
  NumInstruction("predsimplify", "Number of instructions removed");
  Statistic<>
  NumSwitchCases("predsimplify", "Number of switch cases removed");
  Statistic<>
  NumBranches("predsimplify", "Number of branches made unconditional");

  /// Returns true if V1 is a better choice than V2. Note that it is
  /// not a total ordering.
  struct compare {
    bool operator()(Value *V1, Value *V2) const {
      if (isa<Constant>(V1)) {
        if (!isa<Constant>(V2)) {
          return true;
        }
      } else if (isa<Argument>(V1)) {
        if (!isa<Constant>(V2) && !isa<Argument>(V2)) {
          return true;
        }
      }
      if (User *U = dyn_cast<User>(V2)) {
        for (User::const_op_iterator I = U->op_begin(), E = U->op_end();
             I != E; ++I) {
          if (*I == V1) {
            return true;
          }
        }
      }
      return false;
    }
  };

  /// Used for choosing the canonical Value in a synonym set.
  /// Leaves the better choice in V1.
  static void order(Value *&V1, Value *&V2) {
    static compare c;
    if (c(V2, V1))
      std::swap(V1, V2);
  }

  /// Similar to EquivalenceClasses, this stores the set of equivalent
  /// types. Beyond EquivalenceClasses, it allows the user to specify
  /// which element will act as leader through a StrictWeakOrdering
  /// function.
  template<typename ElemTy, typename StrictWeak>
  class VISIBILITY_HIDDEN Synonyms {
    std::map<ElemTy, unsigned> mapping;
    std::vector<ElemTy> leaders;
    StrictWeak swo;

  public:
    typedef unsigned iterator;
    typedef const unsigned const_iterator;

    // Inspection

    bool empty() const {
      return leaders.empty();
    }

    iterator findLeader(ElemTy e) {
      typename std::map<ElemTy, unsigned>::iterator MI = mapping.find(e);
      if (MI == mapping.end()) return 0;

      return MI->second;
    }

    const_iterator findLeader(ElemTy e) const {
      typename std::map<ElemTy, unsigned>::const_iterator MI =
          mapping.find(e);
      if (MI == mapping.end()) return 0;

      return MI->second;
    }

    ElemTy &getLeader(iterator I) {
      assert(I != 0 && "Element zero is out of range.");
      return leaders[I-1];
    }

    const ElemTy &getLeader(const_iterator I) const {
      assert(I != 0 && "Element zero is out of range.");
      return leaders[I-1];
    }

#ifdef DEBUG
    void debug(std::ostream &os) const {
      for (unsigned i = 1, e = leaders.size()+1; i != e; ++i) {
        os << i << ". " << *leaders[i-1] << ": [";
        for (std::map<Value *, unsigned>::const_iterator
             I = mapping.begin(), E = mapping.end(); I != E; ++I) {
          if ((*I).second == i && (*I).first != leaders[i-1]) {
            os << *(*I).first << "  ";
          }
        }
        os << "]\n";
      }
    }
#endif

    // Mutators

    /// Combine two sets referring to the same element, inserting the
    /// elements as needed. Returns a valid iterator iff two already
    /// existing disjoint synonym sets were combined. The iterator
    /// points to the removed element.
    iterator unionSets(ElemTy E1, ElemTy E2) {
      if (swo(E2, E1)) std::swap(E1, E2);

      iterator I1 = findLeader(E1),
               I2 = findLeader(E2);

      if (!I1 && !I2) { // neither entry is in yet
        leaders.push_back(E1);
        I1 = leaders.size();
        mapping[E1] = I1;
        mapping[E2] = I1;
        return 0;
      }

      if (!I1 && I2) {
        mapping[E1] = I2;
        std::swap(getLeader(I2), E1);
        return 0;
      }

      if (I1 && !I2) {
        mapping[E2] = I1;
        return 0;
      }

      if (I1 == I2) return 0;

      // This is the case where we have two sets, [%a1, %a2, %a3] and
      // [%p1, %p2, %p3] and someone says that %a2 == %p3. We need to
      // combine the two synsets.

      if (I1 > I2) --I1;

      for (std::map<Value *, unsigned>::iterator I = mapping.begin(),
           E = mapping.end(); I != E; ++I) {
        if (I->second == I2) I->second = I1;
        else if (I->second > I2) --I->second;
      }

      leaders.erase(leaders.begin() + I2 - 1);

      return I2;
    }

    /// Returns an iterator pointing to the synonym set containing
    /// element e. If none exists, a new one is created and returned.
    iterator findOrInsert(ElemTy e) {
      iterator I = findLeader(e);
      if (I) return I;

      leaders.push_back(e);
      I = leaders.size();
      mapping[e] = I;
      return I;
    }
  };

  /// Represents the set of equivalent Value*s and provides insertion
  /// and fast lookup. Also stores the set of inequality relationships.
  class PropertySet {
    struct Property;
  public:
    class Synonyms<Value *, compare> union_find;

    typedef std::vector<Property>::iterator       PropertyIterator;
    typedef std::vector<Property>::const_iterator ConstPropertyIterator;
    typedef Synonyms<Value *, compare>::iterator  SynonymIterator;

    enum Ops {
      EQ,
      NE
    };

    Value *canonicalize(Value *V) const {
      Value *C = lookup(V);
      return C ? C : V;
    }

    Value *lookup(Value *V) const {
      Synonyms<Value *, compare>::iterator SI = union_find.findLeader(V);
      if (!SI) return NULL;
      return union_find.getLeader(SI);
    }

    bool empty() const {
      return union_find.empty();
    }

    void addEqual(Value *V1, Value *V2) {
      // If %x = 0. and %y = -0., seteq %x, %y is true, but
      // copysign(%x) is not the same as copysign(%y).
      if (V1->getType()->isFloatingPoint()) return;

      order(V1, V2);
      if (isa<Constant>(V2)) return; // refuse to set false == true.

      DEBUG(std::cerr << "equal: " << *V1 << " and " << *V2 << "\n");
      SynonymIterator deleted = union_find.unionSets(V1, V2);
      if (deleted) {
        SynonymIterator replacement = union_find.findLeader(V1);
        // Move Properties
        for (PropertyIterator I = Properties.begin(), E = Properties.end();
             I != E; ++I) {
          if (I->I1 == deleted) I->I1 = replacement;
          else if (I->I1 > deleted) --I->I1;
          if (I->I2 == deleted) I->I2 = replacement;
          else if (I->I2 > deleted) --I->I2;
        }
      }
      addImpliedProperties(EQ, V1, V2);
    }

    void addNotEqual(Value *V1, Value *V2) {
      // If %x = NAN then seteq %x, %x is false.
      if (V1->getType()->isFloatingPoint()) return;

      // For example, %x = setne int 0, 0 causes "0 != 0".
      if (isa<Constant>(V1) && isa<Constant>(V2)) return;

      DEBUG(std::cerr << "not equal: " << *V1 << " and " << *V2 << "\n");
      if (findProperty(NE, V1, V2) != Properties.end())
        return; // found.

      // Add the property.
      SynonymIterator I1 = union_find.findOrInsert(V1),
                      I2 = union_find.findOrInsert(V2);

      // Technically this means that the block is unreachable.
      if (I1 == I2) return;

      Properties.push_back(Property(NE, I1, I2));
      addImpliedProperties(NE, V1, V2);
    }

    PropertyIterator findProperty(Ops Opcode, Value *V1, Value *V2) {
      assert(Opcode != EQ && "Can't findProperty on EQ."
             "Use the lookup method instead.");

      SynonymIterator I1 = union_find.findLeader(V1),
                      I2 = union_find.findLeader(V2);
      if (!I1 || !I2) return Properties.end();

      return
      find(Properties.begin(), Properties.end(), Property(Opcode, I1, I2));
    }

    ConstPropertyIterator
    findProperty(Ops Opcode, Value *V1, Value *V2) const {
      assert(Opcode != EQ && "Can't findProperty on EQ."
             "Use the lookup method instead.");

      SynonymIterator I1 = union_find.findLeader(V1),
                      I2 = union_find.findLeader(V2);
      if (!I1 || !I2) return Properties.end();

      return
      find(Properties.begin(), Properties.end(), Property(Opcode, I1, I2));
    }

  private:
    // Represents Head OP [Tail1, Tail2, ...]
    // For example: %x != %a, %x != %b.
    struct VISIBILITY_HIDDEN Property {
      typedef Synonyms<Value *, compare>::iterator Iter;

      Property(Ops opcode, Iter i1, Iter i2)
        : Opcode(opcode), I1(i1), I2(i2)
      { assert(opcode != EQ && "Equality belongs in the synonym set, "
                               "not a property."); }

      bool operator==(const Property &P) const {
        return (Opcode == P.Opcode) &&
               ((I1 == P.I1 && I2 == P.I2) ||
                (I1 == P.I2 && I2 == P.I1));
      }

      Ops Opcode;
      Iter I1, I2;
    };

    void add(Ops Opcode, Value *V1, Value *V2, bool invert) {
      switch (Opcode) {
        case EQ:
          if (invert) addNotEqual(V1, V2);
          else        addEqual(V1, V2);
          break;
        case NE:
          if (invert) addEqual(V1, V2);
          else        addNotEqual(V1, V2);
          break;
        default:
          assert(0 && "Unknown property opcode.");
      }
    }

    // Finds the properties implied by an equivalence and adds them too.
    // Example: ("seteq %a, %b", true,  EQ) --> (%a, %b, EQ)
    //          ("seteq %a, %b", false, EQ) --> (%a, %b, NE)
    void addImpliedProperties(Ops Opcode, Value *V1, Value *V2) {
      order(V1, V2);

      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(V2)) {
        switch (BO->getOpcode()) {
        case Instruction::SetEQ:
          if (V1 == ConstantBool::True)
            add(Opcode, BO->getOperand(0), BO->getOperand(1), false);
          if (V1 == ConstantBool::False)
            add(Opcode, BO->getOperand(0), BO->getOperand(1), true);
          break;
        case Instruction::SetNE:
          if (V1 == ConstantBool::True)
            add(Opcode, BO->getOperand(0), BO->getOperand(1), true);
          if (V1 == ConstantBool::False)
            add(Opcode, BO->getOperand(0), BO->getOperand(1), false);
          break;
        case Instruction::SetLT:
        case Instruction::SetGT:
          if (V1 == ConstantBool::True)
            add(Opcode, BO->getOperand(0), BO->getOperand(1), true);
          break;
        case Instruction::SetLE:
        case Instruction::SetGE:
          if (V1 == ConstantBool::False)
            add(Opcode, BO->getOperand(0), BO->getOperand(1), true);
          break;
        case Instruction::And:
          if (V1 == ConstantBool::True) {
            add(Opcode, ConstantBool::True, BO->getOperand(0), false);
            add(Opcode, ConstantBool::True, BO->getOperand(1), false);
          }
          break;
        case Instruction::Or:
          if (V1 == ConstantBool::False) {
            add(Opcode, ConstantBool::False, BO->getOperand(0), false);
            add(Opcode, ConstantBool::False, BO->getOperand(1), false);
          }
          break;
        case Instruction::Xor:
          if (V1 == ConstantBool::True) {
            if (BO->getOperand(0) == ConstantBool::True)
              add(Opcode, ConstantBool::False, BO->getOperand(1), false);
            if (BO->getOperand(1) == ConstantBool::True)
              add(Opcode, ConstantBool::False, BO->getOperand(0), false);
          }
          if (V1 == ConstantBool::False) {
            if (BO->getOperand(0) == ConstantBool::True)
              add(Opcode, ConstantBool::True, BO->getOperand(1), false);
            if (BO->getOperand(1) == ConstantBool::True)
              add(Opcode, ConstantBool::True, BO->getOperand(0), false);
          }
          break;
        default:
          break;
        }
      } else if (SelectInst *SI = dyn_cast<SelectInst>(V2)) {
        if (Opcode != EQ && Opcode != NE) return;

        ConstantBool *True  = (Opcode==EQ) ? ConstantBool::True
                                           : ConstantBool::False,
                     *False = (Opcode==EQ) ? ConstantBool::False
                                           : ConstantBool::True;

        if (V1 == SI->getTrueValue())
          addEqual(SI->getCondition(), True);
        else if (V1 == SI->getFalseValue())
          addEqual(SI->getCondition(), False);
        else if (Opcode == EQ)
          assert("Result of select not equal to either value.");
      }
    }

  public:
#ifdef DEBUG
    void debug(std::ostream &os) const {
      static const char *OpcodeTable[] = { "EQ", "NE" };

      union_find.debug(os);
      for (std::vector<Property>::const_iterator I = Properties.begin(),
           E = Properties.end(); I != E; ++I) {
        os << (*I).I1 << " " << OpcodeTable[(*I).Opcode] << " "
           << (*I).I2 << "\n";
      }
      os << "\n";
    }
#endif

    std::vector<Property> Properties;
  };

  /// PredicateSimplifier - This class is a simplifier that replaces
  /// one equivalent variable with another. It also tracks what
  /// can't be equal and will solve setcc instructions when possible.
  class PredicateSimplifier : public FunctionPass {
  public:
    bool runOnFunction(Function &F);
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  private:
    // Try to replace the Use of the instruction with something simpler.
    Value *resolve(SetCondInst *SCI, const PropertySet &);
    Value *resolve(BinaryOperator *BO, const PropertySet &);
    Value *resolve(SelectInst *SI, const PropertySet &);
    Value *resolve(Value *V, const PropertySet &);

    // Used by terminator instructions to proceed from the current basic
    // block to the next. Verifies that "current" dominates "next",
    // then calls visitBasicBlock.
    void proceedToSuccessor(PropertySet &CurrentPS, PropertySet &NextPS,
                            DTNodeType *Current, DTNodeType *Next);
    void proceedToSuccessor(PropertySet &CurrentPS,
                            DTNodeType *Current, DTNodeType *Next);

    // Visits each instruction in the basic block.
    void visitBasicBlock(DTNodeType *DTNode, PropertySet &KnownProperties);

    // Tries to simplify each Instruction and add new properties to
    // the PropertySet. Returns true if it erase the instruction.
    void visitInstruction(Instruction *I, DTNodeType *, PropertySet &);
    // For each instruction, add the properties to KnownProperties.

    void visit(TerminatorInst *TI, DTNodeType *, PropertySet &);
    void visit(BranchInst *BI, DTNodeType *, PropertySet &);
    void visit(SwitchInst *SI, DTNodeType *, PropertySet);
    void visit(LoadInst *LI, DTNodeType *, PropertySet &);
    void visit(StoreInst *SI, DTNodeType *, PropertySet &);
    void visit(BinaryOperator *BO, DTNodeType *, PropertySet &);

    DominatorTree *DT;
    bool modified;
  };

  RegisterPass<PredicateSimplifier> X("predsimplify",
                                      "Predicate Simplifier");
}

FunctionPass *llvm::createPredicateSimplifierPass() {
  return new PredicateSimplifier();
}

bool PredicateSimplifier::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTree>();

  modified = false;
  PropertySet KnownProperties;
  visitBasicBlock(DT->getRootNode(), KnownProperties);
  return modified;
}

void PredicateSimplifier::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTree>();
}

// resolve catches cases addProperty won't because it wasn't used as a
// condition in the branch, and that visit won't, because the instruction
// was defined outside of the scope that the properties apply to.
Value *PredicateSimplifier::resolve(SetCondInst *SCI,
                                    const PropertySet &KP) {
  // Attempt to resolve the SetCondInst to a boolean.

  Value *SCI0 = resolve(SCI->getOperand(0), KP),
        *SCI1 = resolve(SCI->getOperand(1), KP);

  ConstantIntegral *CI1 = dyn_cast<ConstantIntegral>(SCI0),
                   *CI2 = dyn_cast<ConstantIntegral>(SCI1);

  if (!CI1 || !CI2) {
    PropertySet::ConstPropertyIterator NE =
        KP.findProperty(PropertySet::NE, SCI0, SCI1);

    if (NE != KP.Properties.end()) {
      switch (SCI->getOpcode()) {
        case Instruction::SetEQ:
          return ConstantBool::False;
        case Instruction::SetNE:
          return ConstantBool::True;
        case Instruction::SetLE:
        case Instruction::SetGE:
        case Instruction::SetLT:
        case Instruction::SetGT:
          break;
        default:
          assert(0 && "Unknown opcode in SetCondInst.");
          break;
      }
    }
    return SCI;
  }

  switch(SCI->getOpcode()) {
    case Instruction::SetLE:
    case Instruction::SetGE:
    case Instruction::SetEQ:
      if (CI1->getRawValue() == CI2->getRawValue())
        return ConstantBool::True;
      else
        return ConstantBool::False;
    case Instruction::SetLT:
    case Instruction::SetGT:
    case Instruction::SetNE:
      if (CI1->getRawValue() == CI2->getRawValue())
        return ConstantBool::False;
      else
        return ConstantBool::True;
    default:
      assert(0 && "Unknown opcode in SetContInst.");
      break;
  }
}

Value *PredicateSimplifier::resolve(BinaryOperator *BO,
                                    const PropertySet &KP) {
  if (SetCondInst *SCI = dyn_cast<SetCondInst>(BO))
    return resolve(SCI, KP);

  Value *lhs = resolve(BO->getOperand(0), KP),
        *rhs = resolve(BO->getOperand(1), KP);
  ConstantIntegral *CI1 = dyn_cast<ConstantIntegral>(lhs);
  ConstantIntegral *CI2 = dyn_cast<ConstantIntegral>(rhs);

  if (!CI1 || !CI2) return BO;

  Value *V = ConstantExpr::get(BO->getOpcode(), CI1, CI2);
  if (V) return V;
  return BO;
}

Value *PredicateSimplifier::resolve(SelectInst *SI, const PropertySet &KP) {
  Value *Condition = resolve(SI->getCondition(), KP);
  if (Condition == ConstantBool::True)
    return resolve(SI->getTrueValue(), KP);
  else if (Condition == ConstantBool::False)
    return resolve(SI->getFalseValue(), KP);
  return SI;
}

Value *PredicateSimplifier::resolve(Value *V, const PropertySet &KP) {
  if (isa<Constant>(V) || isa<BasicBlock>(V) || KP.empty()) return V;

  V = KP.canonicalize(V);

  DEBUG(std::cerr << "peering into " << *V << "\n");

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(V))
    return resolve(BO, KP);
  else if (SelectInst *SI = dyn_cast<SelectInst>(V))
    return resolve(SI, KP);

  return V;
}

void PredicateSimplifier::visitBasicBlock(DTNodeType *DTNode,
                                          PropertySet &KnownProperties) {
  BasicBlock *BB = DTNode->getBlock();
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
    visitInstruction(I++, DTNode, KnownProperties);
  }
}

void PredicateSimplifier::visitInstruction(Instruction *I,
                                           DTNodeType *DTNode,
                                           PropertySet &KnownProperties) {

  DEBUG(std::cerr << "Considering instruction " << *I << "\n");
  DEBUG(KnownProperties.debug(std::cerr));

  // Try to replace the whole instruction.
  Value *V = resolve(I, KnownProperties);
  assert(V->getType() == I->getType() && "Instruction type mutated!");
  if (V != I) {
    modified = true;
    ++NumInstruction;
    I->replaceAllUsesWith(V);
    I->eraseFromParent();
    return;
  }

  // Try to substitute operands.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    Value *Oper = I->getOperand(i);
    Value *V = resolve(Oper, KnownProperties);
    assert(V->getType() == Oper->getType() && "Operand type mutated!");
    if (V != Oper) {
      modified = true;
      ++NumVarsReplaced;
      DEBUG(std::cerr << "resolving " << *I);
      I->setOperand(i, V);
      DEBUG(std::cerr << "into " << *I);
    }
  }

  if (TerminatorInst *TI = dyn_cast<TerminatorInst>(I))
    visit(TI, DTNode, KnownProperties);
  else if (LoadInst *LI = dyn_cast<LoadInst>(I))
    visit(LI, DTNode, KnownProperties);
  else if (StoreInst *SI = dyn_cast<StoreInst>(I))
    visit(SI, DTNode, KnownProperties);
  else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I))
    visit(BO, DTNode, KnownProperties);
}

void PredicateSimplifier::proceedToSuccessor(PropertySet &CurrentPS,
                                             PropertySet &NextPS,
                                             DTNodeType *Current,
                                             DTNodeType *Next) {
  if (Next->getBlock()->getSinglePredecessor() == Current->getBlock())
    proceedToSuccessor(NextPS, Current, Next);
  else
    proceedToSuccessor(CurrentPS, Current, Next);
}

void PredicateSimplifier::proceedToSuccessor(PropertySet &KP,
                                             DTNodeType *Current,
                                             DTNodeType *Next) {
  if (Current->properlyDominates(Next))
    visitBasicBlock(Next, KP);
}

void PredicateSimplifier::visit(TerminatorInst *TI, DTNodeType *Node,
                                PropertySet &KP) {
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    visit(BI, Node, KP);
    return;
  }
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    visit(SI, Node, KP);
    return;
  }

  for (unsigned i = 0, E = TI->getNumSuccessors(); i != E; ++i) {
    BasicBlock *BB = TI->getSuccessor(i);
    PropertySet KPcopy(KP);
    proceedToSuccessor(KPcopy, Node, DT->getNode(TI->getSuccessor(i)));
  }
}

void PredicateSimplifier::visit(BranchInst *BI, DTNodeType *Node,
                                PropertySet &KP) {
  if (BI->isUnconditional()) {
    proceedToSuccessor(KP, Node, DT->getNode(BI->getSuccessor(0)));
    return;
  }

  Value *Condition = BI->getCondition();

  BasicBlock *TrueDest  = BI->getSuccessor(0),
             *FalseDest = BI->getSuccessor(1);

  if (Condition == ConstantBool::True) {
    FalseDest->removePredecessor(BI->getParent());
    BI->setUnconditionalDest(TrueDest);
    modified = true;
    ++NumBranches;
    proceedToSuccessor(KP, Node, DT->getNode(TrueDest));
    return;
  } else if (Condition == ConstantBool::False) {
    TrueDest->removePredecessor(BI->getParent());
    BI->setUnconditionalDest(FalseDest);
    modified = true;
    ++NumBranches;
    proceedToSuccessor(KP, Node, DT->getNode(FalseDest));
    return;
  }

  PropertySet TrueProperties(KP), FalseProperties(KP);
  DEBUG(std::cerr << "true set:\n");
  TrueProperties.addEqual(ConstantBool::True,   Condition);
  DEBUG(TrueProperties.debug(std::cerr));
  DEBUG(std::cerr << "false set:\n");
  FalseProperties.addEqual(ConstantBool::False, Condition);
  DEBUG(FalseProperties.debug(std::cerr));

  PropertySet KPcopy(KP);
  proceedToSuccessor(KP,     TrueProperties,  Node, DT->getNode(TrueDest));
  proceedToSuccessor(KPcopy, FalseProperties, Node, DT->getNode(FalseDest));
}

void PredicateSimplifier::visit(SwitchInst *SI, DTNodeType *DTNode,
                                PropertySet KP) {
  Value *Condition = SI->getCondition();
  assert(Condition == KP.canonicalize(Condition) &&
         "Instruction wasn't already canonicalized?");

  // If there's an NEProperty covering this SwitchInst, we may be able to
  // eliminate one of the cases.
  for (PropertySet::ConstPropertyIterator I = KP.Properties.begin(),
       E = KP.Properties.end(); I != E; ++I) {
    if (I->Opcode != PropertySet::NE) continue;
    Value *V1 = KP.union_find.getLeader(I->I1),
          *V2 = KP.union_find.getLeader(I->I2);

    // Find a Property with a ConstantInt on one side and our
    // Condition on the other.
    ConstantInt *CI = NULL;
    if (V1 == Condition)
      CI = dyn_cast<ConstantInt>(V2);
    else if (V2 == Condition)
      CI = dyn_cast<ConstantInt>(V1);

    if (!CI) continue;

    unsigned i = SI->findCaseValue(CI);
    if (i != 0) { // zero is reserved for the default case.
      SI->getSuccessor(i)->removePredecessor(SI->getParent());
      SI->removeCase(i);
      modified = true;
      ++NumSwitchCases;
    }
  }

  // Set the EQProperty in each of the cases BBs,
  // and the NEProperties in the default BB.
  PropertySet DefaultProperties(KP);

  DTNodeType *Node        = DT->getNode(SI->getParent()),
             *DefaultNode = DT->getNode(SI->getSuccessor(0));
  if (!Node->dominates(DefaultNode)) DefaultNode = NULL;

  for (unsigned I = 1, E = SI->getNumCases(); I < E; ++I) {
    ConstantInt *CI = SI->getCaseValue(I);

    BasicBlock *SuccBB = SI->getSuccessor(I);
    PropertySet copy(KP);
    if (SuccBB->getSinglePredecessor()) {
      PropertySet NewProperties(KP);
      NewProperties.addEqual(Condition, CI);
      proceedToSuccessor(copy, NewProperties, DTNode, DT->getNode(SuccBB));
    } else
      proceedToSuccessor(copy, DTNode, DT->getNode(SuccBB));

    if (DefaultNode)
      DefaultProperties.addNotEqual(Condition, CI);
  }

  if (DefaultNode)
    proceedToSuccessor(DefaultProperties, DTNode, DefaultNode);
}

void PredicateSimplifier::visit(LoadInst *LI, DTNodeType *,
                                PropertySet &KP) {
  Value *Ptr = LI->getPointerOperand();
  KP.addNotEqual(Constant::getNullValue(Ptr->getType()), Ptr);
}

void PredicateSimplifier::visit(StoreInst *SI, DTNodeType *,
                                PropertySet &KP) {
  Value *Ptr = SI->getPointerOperand();
  KP.addNotEqual(Constant::getNullValue(Ptr->getType()), Ptr);
}

void PredicateSimplifier::visit(BinaryOperator *BO, DTNodeType *,
                                PropertySet &KP) {
  Instruction::BinaryOps ops = BO->getOpcode();

  switch (ops) {
    case Instruction::Div:
    case Instruction::Rem: {
      Value *Divisor = BO->getOperand(1);
      KP.addNotEqual(Constant::getNullValue(Divisor->getType()), Divisor);
      break;
    }
    default:
      break;
  }
}
