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
// conditional that assures us of that fact. Equivalent variables are
// called SynSets; sets of synonyms. We maintain a mapping from Value *
// to the SynSet, and the SynSet maintains the best canonical form of the
// Value.
//
// Properties are stored as relationships between two SynSets.
//
//===------------------------------------------------------------------===//

// TODO:
// * Handle SelectInst
// * Switch to EquivalenceClasses ADT
// * Check handling of NAN in floating point types
// * Don't descend into false side of branches with ConstantBool condition.

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

namespace {
  Statistic<>
  NumVarsReplaced("predsimplify", "Number of argument substitutions");
  Statistic<>
  NumResolved("predsimplify", "Number of instruction substitutions");
  Statistic<>
  NumSwitchCases("predsimplify", "Number of switch cases removed");

  /// Used for choosing the canonical Value in a synonym set.
  /// Leaves the better one in V1. Returns whether a swap took place.
  static void order(Value *&V1, Value *&V2) {
    if (isa<Constant>(V2)) {
      if (!isa<Constant>(V1)) {
        std::swap(V1, V2);
        return;
      }
    } else if (isa<Argument>(V2)) {
      if (!isa<Constant>(V1) && !isa<Argument>(V1)) {
        std::swap(V1, V2);
        return;
      }
    }
    if (User *U1 = dyn_cast<User>(V1)) {
      for (User::const_op_iterator I = U1->op_begin(), E = U1->op_end();
           I != E; ++I) {
        if (*I == V2) {
          std::swap(V1, V2);
          return;
        }
      }
    }
    return;
  }

  /// Represents the set of equivalent Value*s and provides insertion
  /// and fast lookup. Also stores the set of inequality relationships.
  class PropertySet {
    struct Property;
  public:
    typedef unsigned SynSet;
    typedef std::map<Value*, unsigned>::iterator       SynonymIterator;
    typedef std::map<Value*, unsigned>::const_iterator ConstSynonymIterator;
    typedef std::vector<Property>::iterator       PropertyIterator;
    typedef std::vector<Property>::const_iterator ConstPropertyIterator;

    enum Ops {
      EQ,
      NE
    };

    Value *canonicalize(Value *V) const {
      Value *C = lookup(V);
      return C ? C : V;
    }

    Value *lookup(Value *V) const {
      ConstSynonymIterator SI = SynonymMap.find(V);
      if (SI == SynonymMap.end()) return NULL;

      return Synonyms[SI->second];
    }

    Value *lookup(SynSet SS) const {
      assert(SS < Synonyms.size());
      return Synonyms[SS];
    }

    // Find a SynSet for a given Value.
    //
    // Given the Value *V sets SS to a valid SynSet. Returns true if it
    // found it.
    bool findSynSet(Value *V, SynSet &SS) const {
      ConstSynonymIterator SI = SynonymMap.find(V);
      if (SI != SynonymMap.end()) {
        SS = SI->second;
        return true;
      }

      std::vector<Value *>::const_iterator I =
          std::find(Synonyms.begin(), Synonyms.end(), V);
      if (I != Synonyms.end()) { 
        SS = I-Synonyms.begin();
        return true;
      }

      return false;
    }

    bool empty() const {
      return Synonyms.empty();
    }

    void addEqual(Value *V1, Value *V2) {
      order(V1, V2);
      if (isa<Constant>(V2)) return; // refuse to set false == true.

      V1 = canonicalize(V1);
      V2 = canonicalize(V2);

      if (V1 == V2) return; // already equivalent.

      SynSet I1, I2;
      bool F1 = findSynSet(V1, I1),
           F2 = findSynSet(V2, I2);

      DEBUG(std::cerr << "V1: " << *V1 << " I1: " << I1
                      << " F1: " << F1 << "\n");
      DEBUG(std::cerr << "V2: " << *V2 << " I2: " << I2
                      << " F2: " << F2 << "\n");

      if (!F1 && !F2) {
        SynSet SS = addSynSet(V1);
        SynonymMap[V1] = SS;
        SynonymMap[V2] = SS;
      }

      else if (!F1 && F2) {
        SynonymMap[V1] = I2;
      }

      else if (F1 && !F2) {
        SynonymMap[V2] = I1;
      }

      else {
        // This is the case where we have two sets, [%a1, %a2, %a3] and
        // [%p1, %p2, %p3] and someone says that %a2 == %p3. We need to
        // combine the two synsets.

        // Collapse synonyms of V2 into V1.
        for (SynonymIterator I = SynonymMap.begin(), E = SynonymMap.end();
             I != E; ++I) {
          if (I->second == I2) I->second = I1;
          else if (I->second > I2) --I->second;
        }

        // Move Properties
        for (PropertyIterator I = Properties.begin(), E = Properties.end();
             I != E; ++I) {
          if (I->S1 == I2) I->S1 = I1;
          else if (I->S1 > I2) --I->S1;
          if (I->S2 == I2) I->S2 = I1;
          else if (I->S2 > I2) --I->S2;
        }

        // Remove the synonym
        Synonyms.erase(Synonyms.begin() + I2);
      }

      addImpliedProperties(EQ, V1, V2);
    }

    void addNotEqual(Value *V1, Value *V2) {
      DEBUG(std::cerr << "not equal: " << *V1 << " and " << *V2 << "\n");
      bool skip_search = false;
      V1 = canonicalize(V1);
      V2 = canonicalize(V2);

      SynSet S1, S2;
      if (!findSynSet(V1, S1)) {
        skip_search = true;
        S1 = addSynSet(V1);
      }
      if (!findSynSet(V2, S2)) {
        skip_search = true;
        S2 = addSynSet(V2);
      }

      if (!skip_search) {
        // Does the property already exist?
        for (PropertyIterator I = Properties.begin(), E = Properties.end();
             I != E; ++I) {
          if (I->Opcode != NE) continue;

          if ((I->S1 == S1 && I->S2 == S2) ||
              (I->S1 == S2 && I->S2 == S1)) {
            return; // Found.
          }
        }
      }

      // Add the property.
      Properties.push_back(Property(NE, S1, S2));
      addImpliedProperties(NE, V1, V2);
    }

    PropertyIterator findProperty(Ops Opcode, Value *V1, Value *V2) {
      assert(Opcode != EQ && "Can't findProperty on EQ."
             "Use the lookup method instead.");

      SynSet S1, S2;
      if (!findSynSet(V1, S1)) return Properties.end();
      if (!findSynSet(V2, S2)) return Properties.end();

      // Does the property already exist?
      for (PropertyIterator I = Properties.begin(), E = Properties.end();
           I != E; ++I) {
        if (I->Opcode != Opcode) continue;

        if ((I->S1 == S1 && I->S2 == S2) ||
            (I->S1 == S2 && I->S2 == S1)) {
          return I; // Found.
        }
      }
      return Properties.end();
    }

    ConstPropertyIterator
    findProperty(Ops Opcode, Value *V1, Value *V2) const {
      assert(Opcode != EQ && "Can't findProperty on EQ."
             "Use the lookup method instead.");

      SynSet S1, S2;
      if (!findSynSet(V1, S1)) return Properties.end();
      if (!findSynSet(V2, S2)) return Properties.end();

      // Does the property already exist?
      for (ConstPropertyIterator I = Properties.begin(),
           E = Properties.end(); I != E; ++I) {
        if (I->Opcode != Opcode) continue;

        if ((I->S1 == S1 && I->S2 == S2) ||
            (I->S1 == S2 && I->S2 == S1)) {
          return I; // Found.
        }
      }
      return Properties.end();
    }

  private:
    // Represents Head OP [Tail1, Tail2, ...]
    // For example: %x != %a, %x != %b.
    struct Property {
      Property(Ops opcode, SynSet s1, SynSet s2)
        : Opcode(opcode), S1(s1), S2(s2)
      { assert(opcode != EQ && "Equality belongs in the synonym set,"
               "not a property."); }

      bool operator<(const Property &rhs) const {
        if (Opcode != rhs.Opcode) return Opcode < rhs.Opcode;
        if (S1 != rhs.S1) return S1 < rhs.S1;
        return S2 < rhs.S2;
      }

      Ops Opcode;
      SynSet S1, S2;
    };

    SynSet addSynSet(Value *V) {
      Synonyms.push_back(V);
      return Synonyms.size()-1;
    }

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

    // Finds the properties implied by a synonym and adds them too.
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
      }
    }

    std::map<Value *, unsigned> SynonymMap;
    std::vector<Value *> Synonyms;

  public:
    void debug(std::ostream &os) const {
      os << Synonyms.size() << " synsets:\n";
      for (unsigned I = 0, E = Synonyms.size(); I != E; ++I) {
        os << I << ". " << *Synonyms[I] << "\n";
      }
      for (ConstSynonymIterator I = SynonymMap.begin(),E = SynonymMap.end();
           I != E; ++I) {
        os << *I->first << "-> #" << I->second << "\n";
      }
      os << Properties.size() << " properties:\n";
      for (unsigned I = 0, E = Properties.size(); I != E; ++I) {
        os << I << ". (" << Properties[I].Opcode << ","
           << Properties[I].S1 << "," << Properties[I].S2 << ")\n";
      }
    }

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
    Value *resolve(Value *V, const PropertySet &);

    // Used by terminator instructions to proceed from the current basic
    // block to the next. Verifies that "current" dominates "next",
    // then calls visitBasicBlock.
    void proceedToSuccessor(PropertySet &CurrentPS, PropertySet &NextPS,
                  DominatorTree::Node *Current, DominatorTree::Node *Next);
    void proceedToSuccessor(PropertySet &CurrentPS,
                  DominatorTree::Node *Current, DominatorTree::Node *Next);

    // Visits each instruction in the basic block.
    void visitBasicBlock(DominatorTree::Node *DTNode,
                         PropertySet &KnownProperties);

    // For each instruction, add the properties to KnownProperties.
    void visit(Instruction *I, DominatorTree::Node *, PropertySet &);
    void visit(TerminatorInst *TI, DominatorTree::Node *, PropertySet &);
    void visit(BranchInst *BI, DominatorTree::Node *, PropertySet &);
    void visit(SwitchInst *SI, DominatorTree::Node *, PropertySet);
    void visit(LoadInst *LI, DominatorTree::Node *, PropertySet &);
    void visit(StoreInst *SI, DominatorTree::Node *, PropertySet &);
    void visit(BinaryOperator *BO, DominatorTree::Node *, PropertySet &);

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
// was defined outside of the range that the properties apply to.
Value *PredicateSimplifier::resolve(SetCondInst *SCI,
                                    const PropertySet &KP) {
  // Attempt to resolve the SetCondInst to a boolean.

  Value *SCI0 = SCI->getOperand(0),
        *SCI1 = SCI->getOperand(1);
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

  SCI0 = KP.canonicalize(SCI0);
  SCI1 = KP.canonicalize(SCI1);

  ConstantIntegral *CI1 = dyn_cast<ConstantIntegral>(SCI0),
                   *CI2 = dyn_cast<ConstantIntegral>(SCI1);

  if (!CI1 || !CI2) return SCI;

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

  DEBUG(std::cerr << "BO->getOperand(1) = " << *BO->getOperand(1) << "\n");

  Value *lhs = resolve(BO->getOperand(0), KP),
        *rhs = resolve(BO->getOperand(1), KP);
  ConstantIntegral *CI1 = dyn_cast<ConstantIntegral>(lhs);
  ConstantIntegral *CI2 = dyn_cast<ConstantIntegral>(rhs);

  DEBUG(std::cerr << "resolveBO: lhs = " << *lhs
                  << ", rhs = " << *rhs << "\n");
  if (CI1) DEBUG(std::cerr << "CI1 = " << *CI1);
  if (CI2) DEBUG(std::cerr << "CI2 = " << *CI2);

  if (!CI1 || !CI2) return BO;

  Value *V = ConstantExpr::get(BO->getOpcode(), CI1, CI2);
  if (V) return V;
  return BO;
}

Value *PredicateSimplifier::resolve(Value *V, const PropertySet &KP) {
  if (isa<Constant>(V) || isa<BasicBlock>(V) || KP.empty()) return V;

  V = KP.canonicalize(V);

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(V))
    return resolve(BO, KP);

  return V;
}

void PredicateSimplifier::visitBasicBlock(DominatorTree::Node *DTNode,
                                          PropertySet &KnownProperties) {
  BasicBlock *BB = DTNode->getBlock();
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    visit(I, DTNode, KnownProperties);
  }
}

void PredicateSimplifier::visit(Instruction *I, DominatorTree::Node *DTNode,
                                PropertySet &KnownProperties) {
  DEBUG(std::cerr << "Considering instruction " << *I << "\n");
  DEBUG(KnownProperties.debug(std::cerr));

  // Substitute values known to be equal.
  for (unsigned i = 0, E = I->getNumOperands(); i != E; ++i) {
    Value *Oper = I->getOperand(i);
    Value *V = resolve(Oper, KnownProperties);
    assert(V && "resolve not supposed to return NULL.");
    if (V != Oper) {
      modified = true;
      ++NumVarsReplaced;
      DEBUG(std::cerr << "resolving " << *I);
      I->setOperand(i, V);
      DEBUG(std::cerr << "into " << *I);
    }
  }

  Value *V = resolve(I, KnownProperties);
  assert(V && "resolve not supposed to return NULL.");
  if (V != I) {
    modified = true;
    ++NumResolved;
    I->replaceAllUsesWith(V);
    I->eraseFromParent();
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
    PropertySet &NextPS, DominatorTree::Node *Current,
    DominatorTree::Node *Next) {
  if (Next->getBlock()->getSinglePredecessor() == Current->getBlock())
    proceedToSuccessor(NextPS, Current, Next);
  else
    proceedToSuccessor(CurrentPS, Current, Next);
}

void PredicateSimplifier::proceedToSuccessor(PropertySet &KP,
    DominatorTree::Node *Current, DominatorTree::Node *Next) {
  if (Current->properlyDominates(Next))
    visitBasicBlock(Next, KP);
}

void PredicateSimplifier::visit(TerminatorInst *TI,
                                DominatorTree::Node *Node, PropertySet &KP){
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

void PredicateSimplifier::visit(BranchInst *BI,
                                DominatorTree::Node *Node, PropertySet &KP){
  if (BI->isUnconditional()) {
    proceedToSuccessor(KP, Node, DT->getNode(BI->getSuccessor(0)));
    return;
  }

  Value *Condition = BI->getCondition();

  PropertySet TrueProperties(KP), FalseProperties(KP);
  DEBUG(std::cerr << "true set:\n");
  TrueProperties.addEqual(ConstantBool::True,   Condition);
  DEBUG(std::cerr << "false set:\n");
  FalseProperties.addEqual(ConstantBool::False, Condition);

  BasicBlock *TrueDest  = BI->getSuccessor(0),
             *FalseDest = BI->getSuccessor(1);

  PropertySet KPcopy(KP);
  proceedToSuccessor(KP,     TrueProperties,  Node, DT->getNode(TrueDest));
  proceedToSuccessor(KPcopy, FalseProperties, Node, DT->getNode(FalseDest));
}

void PredicateSimplifier::visit(SwitchInst *SI,
                             DominatorTree::Node *DTNode, PropertySet KP) {
  Value *Condition = SI->getCondition();

  // If there's an NEProperty covering this SwitchInst, we may be able to
  // eliminate one of the cases.
  PropertySet::SynSet S;

  if (KP.findSynSet(Condition, S)) {
    for (PropertySet::ConstPropertyIterator I = KP.Properties.begin(),
         E = KP.Properties.end(); I != E; ++I) {
      if (I->Opcode != PropertySet::NE) continue;
      if (I->S1 != S && I->S2 != S) continue;

      // Is one side a number?
      ConstantInt *CI = dyn_cast<ConstantInt>(KP.lookup(I->S1));
      if (!CI)     CI = dyn_cast<ConstantInt>(KP.lookup(I->S2));

      if (CI) {
        unsigned i = SI->findCaseValue(CI);
        if (i != 0) {
          SI->getSuccessor(i)->removePredecessor(SI->getParent());
          SI->removeCase(i);
          modified = true;
          ++NumSwitchCases;
        }
      }
    }
  }

  // Set the EQProperty in each of the cases BBs,
  // and the NEProperties in the default BB.
  PropertySet DefaultProperties(KP);

  DominatorTree::Node *Node        = DT->getNode(SI->getParent()),
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

void PredicateSimplifier::visit(LoadInst *LI,
                                DominatorTree::Node *, PropertySet &KP) {
  Value *Ptr = LI->getPointerOperand();
  KP.addNotEqual(Constant::getNullValue(Ptr->getType()), Ptr);
}

void PredicateSimplifier::visit(StoreInst *SI,
                                DominatorTree::Node *, PropertySet &KP) {
  Value *Ptr = SI->getPointerOperand();
  KP.addNotEqual(Constant::getNullValue(Ptr->getType()), Ptr);
}

void PredicateSimplifier::visit(BinaryOperator *BO,
                                DominatorTree::Node *, PropertySet &KP) {
  Instruction::BinaryOps ops = BO->getOpcode();
  if (ops != Instruction::Div && ops != Instruction::Rem) return;

  Value *Divisor = BO->getOperand(1);
  const Type *Ty = cast<Type>(Divisor->getType());
  KP.addNotEqual(Constant::getNullValue(Ty), Divisor);

  // Some other things we could do:
  // In f=x*y, if x != 1 && y != 1 then f != x && f != y.
  // In f=x+y, if x != 0 then f != y and if y != 0 then f != x.
}
