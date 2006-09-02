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
#include "llvm/ADT/EquivalenceClasses.h"
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
  NumInstruction("predsimplify", "Number of instructions removed");
  Statistic<>
  NumSwitchCases("predsimplify", "Number of switch cases removed");
  Statistic<>
  NumBranches("predsimplify", "Number of branches made unconditional");

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
    class EquivalenceClasses<Value *> union_find;
  public:
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
      EquivalenceClasses<Value *>::member_iterator SI =
          union_find.findLeader(V);
      if (SI == union_find.member_end()) return NULL;
      return *SI;
    }

    bool empty() const {
      return union_find.empty();
    }

    void addEqual(Value *V1, Value *V2) {
      // If %x = 0. and %y = -0., seteq %x, %y is true, but
      // copysign(%x) is not the same as copysign(%y).
      if (V2->getType()->isFloatingPoint()) return;

      order(V1, V2);
      if (isa<Constant>(V2)) return; // refuse to set false == true.

      DEBUG(std::cerr << "equal: " << *V1 << " and " << *V2 << "\n");
      union_find.unionSets(V1, V2);
      addImpliedProperties(EQ, V1, V2);
    }

    void addNotEqual(Value *V1, Value *V2) {
      // If %x = NAN then seteq %x, %x is false.
      if (V2->getType()->isFloatingPoint()) return;

      DEBUG(std::cerr << "not equal: " << *V1 << " and " << *V2 << "\n");
      if (findProperty(NE, V1, V2) != Properties.end())
        return; // found.

      // Add the property.
      Properties.push_back(Property(NE, V1, V2));
      addImpliedProperties(NE, V1, V2);
    }

    PropertyIterator findProperty(Ops Opcode, Value *V1, Value *V2) {
      assert(Opcode != EQ && "Can't findProperty on EQ."
             "Use the lookup method instead.");

      V1 = canonicalize(V1);
      V2 = canonicalize(V2);

      // Does the property already exist?
      for (PropertyIterator I = Properties.begin(), E = Properties.end();
           I != E; ++I) {
        if (I->Opcode != Opcode) continue;

        I->V1 = canonicalize(I->V1);
        I->V2 = canonicalize(I->V2);
        if ((I->V1 == V1 && I->V2 == V2) ||
            (I->V1 == V2 && I->V2 == V1)) {
          return I; // Found.
        }
      }
      return Properties.end();
    }

    ConstPropertyIterator
    findProperty(Ops Opcode, Value *V1, Value *V2) const {
      assert(Opcode != EQ && "Can't findProperty on EQ."
             "Use the lookup method instead.");

      V1 = canonicalize(V1);
      V2 = canonicalize(V2);

      // Does the property already exist?
      for (ConstPropertyIterator I = Properties.begin(),
           E = Properties.end(); I != E; ++I) {
        if (I->Opcode != Opcode) continue;

        Value *v1 = canonicalize(I->V1),
              *v2 = canonicalize(I->V2);
        if ((v1 == V1 && v2 == V2) ||
            (v1 == V2 && v2 == V1)) {
          return I; // Found.
        }
      }
      return Properties.end();
    }

  private:
    // Represents Head OP [Tail1, Tail2, ...]
    // For example: %x != %a, %x != %b.
    struct Property {
      Property(Ops opcode, Value *v1, Value *v2)
        : Opcode(opcode), V1(v1), V2(v2)
      { assert(opcode != EQ && "Equality belongs in the synonym set, "
               "not a property."); }

      Ops Opcode;
      Value *V1, *V2;
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

    // Finds the properties implied by a equivalence and adds them too.
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
      for (EquivalenceClasses<Value*>::iterator I = union_find.begin(),
           E = union_find.end(); I != E; ++I) {
        if (!I->isLeader()) continue;
        for (EquivalenceClasses<Value*>::member_iterator MI =
             union_find.member_begin(I); MI != union_find.member_end(); ++MI)
          std::cerr << **MI << " ";
        std::cerr << "\n--\n";
      }
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
// was defined outside of the scope that the properties apply to.
Value *PredicateSimplifier::resolve(SetCondInst *SCI,
                                    const PropertySet &KP) {
  // Attempt to resolve the SetCondInst to a boolean.

  Value *SCI0 = resolve(SCI->getOperand(0), KP),
        *SCI1 = resolve(SCI->getOperand(1), KP);
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

  // Try to replace whole instruction.
  Value *V = resolve(I, KnownProperties);
  assert(V && "resolve not supposed to return NULL.");
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
    assert(V && "resolve not supposed to return NULL.");
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

void PredicateSimplifier::visit(SwitchInst *SI,
                             DominatorTree::Node *DTNode, PropertySet KP) {
  Value *Condition = SI->getCondition();
  DEBUG(assert(Condition == KP.canonicalize(Condition) &&
               "Instruction wasn't already canonicalized?"));

  // If there's an NEProperty covering this SwitchInst, we may be able to
  // eliminate one of the cases.
  for (PropertySet::ConstPropertyIterator I = KP.Properties.begin(),
       E = KP.Properties.end(); I != E; ++I) {
    if (I->Opcode != PropertySet::NE) continue;
    Value *V1 = KP.canonicalize(I->V1),
          *V2 = KP.canonicalize(I->V2);
    if (V1 != Condition && V2 != Condition) continue;

    // Is one side a number?
    ConstantInt *CI = dyn_cast<ConstantInt>(KP.canonicalize(I->V1));
    if (!CI)     CI = dyn_cast<ConstantInt>(KP.canonicalize(I->V2));

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

  // Some other things we could do:
  // In f=x*y, if x != 1 && y != 1 then f != x && f != y.
  // In f=x+y, if x != 0 then f != y and if y != 0 then f != x.
}
