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

  class PropertySet;

  /// Similar to EquivalenceClasses, this stores the set of equivalent
  /// types. Beyond EquivalenceClasses, it allows us to specify which
  /// element will act as leader.
  template<typename ElemTy>
  class VISIBILITY_HIDDEN Synonyms {
    std::map<ElemTy, unsigned> mapping;
    std::vector<ElemTy> leaders;
    PropertySet *PS;

  public:
    typedef unsigned iterator;
    typedef const unsigned const_iterator;

    Synonyms(PropertySet *PS) : PS(PS) {}

    // Inspection

    bool empty() const {
      return leaders.empty();
    }

    typename std::vector<ElemTy>::size_type countLeaders() const {
      return leaders.size();
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
      assert(I && I <= leaders.size() && "Illegal leader to get.");
      return leaders[I-1];
    }

    const ElemTy &getLeader(const_iterator I) const {
      assert(I && I <= leaders.size() && "Illegal leaders to get.");
      return leaders[I-1];
    }

#ifdef DEBUG
    void debug(std::ostream &os) const {
      for (unsigned i = 1, e = leaders.size()+1; i != e; ++i) {
        os << i << ". " << *getLeader(i) << ": [";
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
    /// points to the no longer existing element.
    iterator unionSets(ElemTy E1, ElemTy E2);

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
    /// Returns true if V1 is a better choice than V2. Note that it is
    /// not a total ordering.
    bool compare(Value *V1, Value *V2) const {
      if (isa<Constant>(V1)) {
        if (!isa<Constant>(V2)) {
          return true;
        }
      } else if (isa<Argument>(V1)) {
        if (!isa<Constant>(V2) && !isa<Argument>(V2)) {
          return true;
        }
      }
      if (Instruction *I1 = dyn_cast<Instruction>(V1)) {
        if (Instruction *I2 = dyn_cast<Instruction>(V2)) {
          BasicBlock *BB1 = I1->getParent(),
                     *BB2 = I2->getParent();
          if (BB1 == BB2) {
            for (BasicBlock::const_iterator I = BB1->begin(), E = BB1->end();
                 I != E; ++I) {
              if (&*I == I1) return true;
              if (&*I == I2) return false;
            }
            assert(0 && "Instructions not found in parent BasicBlock?");
          } else
            return DT->getNode(BB1)->properlyDominates(DT->getNode(BB2));
        }
      }
      return false;
    }

    struct Property;
  public:
    /// Choose the canonical Value in a synonym set.
    /// Leaves the more canonical choice in V1.
    void order(Value *&V1, Value *&V2) const {
      if (compare(V2, V1)) std::swap(V1, V2);
    }

    PropertySet(DominatorTree *DT) : union_find(this), DT(DT) {}

    class Synonyms<Value *> union_find;

    typedef std::vector<Property>::iterator       PropertyIterator;
    typedef std::vector<Property>::const_iterator ConstPropertyIterator;
    typedef Synonyms<Value *>::iterator  SynonymIterator;

    enum Ops {
      EQ,
      NE
    };

    Value *canonicalize(Value *V) const {
      Value *C = lookup(V);
      return C ? C : V;
    }

    Value *lookup(Value *V) const {
      SynonymIterator SI = union_find.findLeader(V);
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
      typedef SynonymIterator Iter;

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

    DominatorTree *DT;
  public:
#ifdef DEBUG
    void debug(std::ostream &os) const {
      static const char *OpcodeTable[] = { "EQ", "NE" };

      unsigned int size = union_find.countLeaders();

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
    void proceedToSuccessor(TerminatorInst *TI, unsigned edge,
                            PropertySet &CurrentPS, PropertySet &NextPS);
    void proceedToSuccessors(PropertySet &CurrentPS, BasicBlock *Current);

    // Visits each instruction in the basic block.
    void visitBasicBlock(BasicBlock *Block, PropertySet &KnownProperties);

    // Tries to simplify each Instruction and add new properties to
    // the PropertySet. Returns true if it erase the instruction.
    void visitInstruction(Instruction *I, PropertySet &);
    // For each instruction, add the properties to KnownProperties.

    void visit(TerminatorInst *TI, PropertySet &);
    void visit(BranchInst *BI, PropertySet &);
    void visit(SwitchInst *SI, PropertySet);
    void visit(LoadInst *LI, PropertySet &);
    void visit(StoreInst *SI, PropertySet &);
    void visit(BinaryOperator *BO, PropertySet &);

    DominatorTree *DT;
    bool modified;
  };

  RegisterPass<PredicateSimplifier> X("predsimplify",
                                      "Predicate Simplifier");

  template <typename ElemTy>
  typename Synonyms<ElemTy>::iterator
  Synonyms<ElemTy>::unionSets(ElemTy E1, ElemTy E2) {
    PS->order(E1, E2);

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
}

FunctionPass *llvm::createPredicateSimplifierPass() {
  return new PredicateSimplifier();
}

bool PredicateSimplifier::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTree>();

  modified = false;
  PropertySet KnownProperties(DT);
  visitBasicBlock(DT->getRootNode()->getBlock(), KnownProperties);
  return modified;
}

void PredicateSimplifier::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTree>();
  AU.setPreservesCFG();
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

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(V))
    return resolve(BO, KP);
  else if (SelectInst *SI = dyn_cast<SelectInst>(V))
    return resolve(SI, KP);

  return V;
}

void PredicateSimplifier::visitBasicBlock(BasicBlock *BB,
                                          PropertySet &KnownProperties) {
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
    visitInstruction(I++, KnownProperties);
  }
}

void PredicateSimplifier::visitInstruction(Instruction *I,
                                           PropertySet &KnownProperties) {
  // Try to replace the whole instruction.
  Value *V = resolve(I, KnownProperties);
  if (V != I) {
    modified = true;
    ++NumInstruction;
    DEBUG(std::cerr << "Removing " << *I);
    I->replaceAllUsesWith(V);
    I->eraseFromParent();
    return;
  }

  // Try to substitute operands.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    Value *Oper = I->getOperand(i);
    Value *V = resolve(Oper, KnownProperties);
    if (V != Oper) {
      modified = true;
      ++NumVarsReplaced;
      DEBUG(std::cerr << "resolving " << *I);
      I->setOperand(i, V);
      DEBUG(std::cerr << "into " << *I);
    }
  }

  if (TerminatorInst *TI = dyn_cast<TerminatorInst>(I))
    visit(TI, KnownProperties);
  else if (LoadInst *LI = dyn_cast<LoadInst>(I))
    visit(LI, KnownProperties);
  else if (StoreInst *SI = dyn_cast<StoreInst>(I))
    visit(SI, KnownProperties);
  else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I))
    visit(BO, KnownProperties);
}

// The basic block on the target of the specified edge must be known
// to be immediately dominated by the parent of the TerminatorInst.
void PredicateSimplifier::proceedToSuccessor(TerminatorInst *TI,
                                             unsigned edge,
                                             PropertySet &CurrentPS,
                                             PropertySet &NextPS) {
  assert(edge < TI->getNumSuccessors() && "Invalid index for edge.");

  BasicBlock *BB     = TI->getParent(),
             *BBNext = TI->getSuccessor(edge);

  if (BBNext->getSinglePredecessor() == BB)
    visitBasicBlock(BBNext, NextPS);
  else
    visitBasicBlock(BBNext, CurrentPS);
}

void PredicateSimplifier::proceedToSuccessors(PropertySet &KP,
                                              BasicBlock *BBCurrent) {
  DTNodeType *Current = DT->getNode(BBCurrent);
  for (DTNodeType::iterator I = Current->begin(), E = Current->end();
       I != E; ++I) {
    PropertySet Copy(KP);
    visitBasicBlock((*I)->getBlock(), Copy);
  }
}

void PredicateSimplifier::visit(TerminatorInst *TI, PropertySet &KP) {
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    visit(BI, KP);
    return;
  }
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    visit(SI, KP);
    return;
  }

  proceedToSuccessors(KP, TI->getParent());
}

void PredicateSimplifier::visit(BranchInst *BI, PropertySet &KP) {
  BasicBlock *BB = BI->getParent();

  if (BI->isUnconditional()) {
    proceedToSuccessors(KP, BB);
    return;
  }

  Value *Condition = BI->getCondition();

  BasicBlock *TrueDest  = BI->getSuccessor(0),
             *FalseDest = BI->getSuccessor(1);

  if (Condition == ConstantBool::True || TrueDest == FalseDest) {
    proceedToSuccessors(KP, BB);
    return;
  } else if (Condition == ConstantBool::False) {
    proceedToSuccessors(KP, BB);
    return;
  }

  DTNodeType *Node = DT->getNode(BB);
  for (DTNodeType::iterator I = Node->begin(), E = Node->end(); I != E; ++I) {
    if ((*I)->getBlock() == TrueDest) {
      PropertySet TrueProperties(KP);
      TrueProperties.addEqual(ConstantBool::True, Condition);
      proceedToSuccessor(BI, 0, KP, TrueProperties);
      continue;
    }

    if ((*I)->getBlock() == FalseDest) {
      PropertySet FalseProperties(KP);
      FalseProperties.addEqual(ConstantBool::False, Condition);
      proceedToSuccessor(BI, 1, KP, FalseProperties);
      continue;
    }

    visitBasicBlock((*I)->getBlock(), KP);
  }
}

void PredicateSimplifier::visit(SwitchInst *SI, PropertySet KP) {
  Value *Condition = SI->getCondition();

  // Set the EQProperty in each of the cases BBs,
  // and the NEProperties in the default BB.
  PropertySet DefaultProperties(KP);

  DTNodeType *Node = DT->getNode(SI->getParent());
  for (DTNodeType::iterator I = Node->begin(), E = Node->end(); I != E; ++I) {
    BasicBlock *BB = (*I)->getBlock();

    PropertySet Copy(KP);

    if (BB == SI->getDefaultDest()) {
      PropertySet NewProperties(KP);
      for (unsigned i = 1, e = SI->getNumCases(); i < e; ++i)
        NewProperties.addNotEqual(Condition, SI->getCaseValue(i));

      proceedToSuccessor(SI, 0, Copy, NewProperties);
    } else if (ConstantInt *CI = SI->findCaseDest(BB)) {
      PropertySet NewProperties(KP);
      NewProperties.addEqual(Condition, CI);
      proceedToSuccessor(SI, SI->findCaseValue(CI), Copy, NewProperties);
    } else 
      visitBasicBlock(BB, Copy);
  }
}

void PredicateSimplifier::visit(LoadInst *LI, PropertySet &KP) {
  Value *Ptr = LI->getPointerOperand();
  KP.addNotEqual(Constant::getNullValue(Ptr->getType()), Ptr);
}

void PredicateSimplifier::visit(StoreInst *SI, PropertySet &KP) {
  Value *Ptr = SI->getPointerOperand();
  KP.addNotEqual(Constant::getNullValue(Ptr->getType()), Ptr);
}

void PredicateSimplifier::visit(BinaryOperator *BO, PropertySet &KP) {
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
