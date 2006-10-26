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
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstVisitor.h"
#include <iostream>
#include <list>
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

    iterator findLeader(ElemTy &e) {
      typename std::map<ElemTy, unsigned>::iterator MI = mapping.find(e);
      if (MI == mapping.end()) return 0;

      return MI->second;
    }

    const_iterator findLeader(ElemTy &e) const {
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

    void remove(ElemTy &e) {
      ElemTy E = e;      // The parameter to erase must not be a reference to
      mapping.erase(E);  // an element contained in the map.
    }

    /// Combine two sets referring to the same element, inserting the
    /// elements as needed. Returns a valid iterator iff two already
    /// existing disjoint synonym sets were combined. The iterator
    /// points to the no longer existing element.
    iterator unionSets(ElemTy E1, ElemTy E2);

    /// Returns an iterator pointing to the synonym set containing
    /// element e. If none exists, a new one is created and returned.
    iterator findOrInsert(ElemTy &e) {
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
    /// Returns true if V1 is a better choice than V2.
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

    Synonyms<Value *> union_find;

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

    void remove(Value *V) {
      SynonymIterator I = union_find.findLeader(V);
      if (!I) return;

      union_find.remove(V);

      for (PropertyIterator PI = Properties.begin(), PE = Properties.end();
           PI != PE;) {
        Property &P = *PI++;
        if (P.I1 == I || P.I2 == I) Properties.erase(PI);
      }
    }

    void addEqual(Value *V1, Value *V2) {
      // If %x = 0. and %y = -0., seteq %x, %y is true, but
      // copysign(%x) is not the same as copysign(%y).
      if (V1->getType()->isFloatingPoint()) return;

      order(V1, V2);
      if (isa<Constant>(V2)) return; // refuse to set false == true.

      if (union_find.findLeader(V1) &&
          union_find.findLeader(V1) == union_find.findLeader(V2))
        return; // no-op

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
        return; // no-op.

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

    void addToResolve(Value *V, std::list<Value *> &WorkList) {
      if (!isa<Constant>(V) && !isa<BasicBlock>(V)) {
        for (Value::use_iterator UI = V->use_begin(), UE = V->use_end();
             UI != UE; ++UI) {
          if (!isa<Constant>(*UI) && !isa<BasicBlock>(*UI)) {
            WorkList.push_back(*UI);
          }
        }
      }
    }

    void resolve(std::list<Value *> &WorkList) {
      if (WorkList.empty()) return;

      Value *V = WorkList.front();
      WorkList.pop_front();

      if (empty()) return;

      Instruction *I = dyn_cast<Instruction>(V);
      if (!I) return;

      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
        Value *lhs = canonicalize(BO->getOperand(0)),
              *rhs = canonicalize(BO->getOperand(1));

        ConstantIntegral *CI1 = dyn_cast<ConstantIntegral>(lhs),
                         *CI2 = dyn_cast<ConstantIntegral>(rhs);

        if (CI1 && CI2) {
          addToResolve(BO, WorkList);
          addEqual(BO, ConstantExpr::get(BO->getOpcode(), CI1, CI2));
        } else if (SetCondInst *SCI = dyn_cast<SetCondInst>(BO)) {
          PropertySet::ConstPropertyIterator NE =
             findProperty(PropertySet::NE, lhs, rhs);

          if (NE != Properties.end()) {
            switch (SCI->getOpcode()) {
            case Instruction::SetEQ:
              addToResolve(SCI, WorkList);
              addEqual(SCI, ConstantBool::getFalse());
              break;
            case Instruction::SetNE:
              addToResolve(SCI, WorkList);
              addEqual(SCI, ConstantBool::getTrue());
              break;
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
        }
      } else if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
        Value *Condition = canonicalize(SI->getCondition());
        if (ConstantBool *CB = dyn_cast<ConstantBool>(Condition)) {
          addToResolve(SI, WorkList);
          addEqual(SI, CB->getValue() ? SI->getTrueValue() : SI->getFalseValue());
        }
      }
      if (!WorkList.empty()) resolve(WorkList);
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

    /// Finds the properties implied by an equivalence and adds them too.
    /// Example: ("seteq %a, %b", true,  EQ) --> (%a, %b, EQ)
    ///          ("seteq %a, %b", false, EQ) --> (%a, %b, NE)
    void addImpliedProperties(Ops Opcode, Value *V1, Value *V2) {
      order(V1, V2);

      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(V2)) {
        switch (BO->getOpcode()) {
        case Instruction::SetEQ:
          // "seteq int %a, %b" EQ true  then %a EQ %b
          // "seteq int %a, %b" EQ false then %a NE %b
          // "seteq int %a, %b" NE true  then %a NE %b
          // "seteq int %a, %b" NE false then %a EQ %b
          if (ConstantBool *V1CB = dyn_cast<ConstantBool>(V1))
            add(Opcode, BO->getOperand(0), BO->getOperand(1),!V1CB->getValue());
          break;
        case Instruction::SetNE:
          // "setne int %a, %b" EQ true  then %a NE %b
          // "setne int %a, %b" EQ false then %a EQ %b
          // "setne int %a, %b" NE true  then %a EQ %b
          // "setne int %a, %b" NE false then %a NE %b
          if (ConstantBool *V1CB = dyn_cast<ConstantBool>(V1))
            add(Opcode, BO->getOperand(0), BO->getOperand(1), V1CB->getValue());
          break;
        case Instruction::SetLT:
        case Instruction::SetGT:
          // "setlt/gt int %a, %b" EQ true  then %a NE %b
          // "setlt/gt int %a, %b" NE false then %a NE %b

          // "setlt int %a, %b" NE true then %a EQ %b

          if (ConstantBool *CB = dyn_cast<ConstantBool>(V1)) {
            if (CB->getValue() ^ Opcode==NE)
              addNotEqual(BO->getOperand(0), BO->getOperand(1));
	  }
          break;
        case Instruction::SetLE:
        case Instruction::SetGE:
          // "setle/ge int %a, %b" EQ false then %a NE %b
          // "setle/ge int %a, %b" NE true  then %a NE %b
          if (ConstantBool *CB = dyn_cast<ConstantBool>(V1)) {
            if (CB->getValue() ^ Opcode==EQ)
              addNotEqual(BO->getOperand(0), BO->getOperand(1));
	  }
          break;
        case Instruction::And: {
          // "and int %a, %b"  EQ 0xff  then %a EQ 0xff and %b EQ 0xff
          // "and bool %a, %b" EQ true  then %a EQ true and %b EQ true
          // "and bool %a, %b" NE false then %a EQ true and %b EQ true
          if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(V1)) {
            if (Opcode == EQ && CI->isAllOnesValue()) {
              addEqual(CI, BO->getOperand(0));
              addEqual(CI, BO->getOperand(1));
            } else if (Opcode == NE && CI == ConstantBool::getFalse()) {
              addEqual(ConstantBool::getTrue(), BO->getOperand(0));
              addEqual(ConstantBool::getTrue(), BO->getOperand(1));
            }
          }
        } break;
        case Instruction::Or: {
          // "or int %a, %b"  EQ 0     then %a EQ 0     and %b EQ 0
          // "or bool %a, %b" EQ false then %a EQ false and %b EQ false
          // "or bool %a, %b" NE true  then %a EQ false and %b EQ false
          if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(V1)) {
            if (Opcode == EQ && CI->isNullValue()) {
              addEqual(CI, BO->getOperand(0));
              addEqual(CI, BO->getOperand(1));
            } else if (Opcode == NE && CI == ConstantBool::getTrue()) {
              addEqual(ConstantBool::getFalse(), BO->getOperand(0));
              addEqual(ConstantBool::getFalse(), BO->getOperand(1));
            }
          }
        } break;
        case Instruction::Xor: {
          // "xor bool true,  %a" EQ true  then %a = false
          // "xor bool true,  %a" EQ false then %a = true
          // "xor bool false, %a" EQ true  then %a = true
          // "xor bool false, %a" EQ false then %a = false
          // 1. Repeat all of the above, with "NE false" in place of
          //    "EQ true" and "NE true" in place of "EQ false".
          // "xor int %c, %a" EQ %c then %a = 0
          // "xor int %c, %a" NE %c then %a != 0
          // 2. Repeat all of the above, with order of operands reversed.

          Value *LHS = BO->getOperand(0), *RHS = BO->getOperand(1);
          if (!isa<Constant>(LHS)) std::swap(LHS, RHS);

          if (ConstantBool *CB = dyn_cast<ConstantBool>(V1)) {
            if (ConstantBool *A = dyn_cast<ConstantBool>(LHS)) {
              addEqual(RHS, ConstantBool::get(A->getValue() ^ CB->getValue()
                                              ^ Opcode==NE));
            }
          }
          else if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(V1)) {
            if (ConstantIntegral *A = dyn_cast<ConstantIntegral>(LHS)) {
              if (A == CI)
                add(Opcode, RHS, Constant::getNullValue(A->getType()), false);
            }
          }
        } break;
        default:
          break;
        }
      } else if (SelectInst *SI = dyn_cast<SelectInst>(V2)) {
        ConstantBool *True  = ConstantBool::get(Opcode==EQ),
                     *False = ConstantBool::get(Opcode!=EQ);

        if (V1 == canonicalize(SI->getTrueValue()))
          addEqual(SI->getCondition(), True);
        else if (V1 == canonicalize(SI->getFalseValue()))
          addEqual(SI->getCondition(), False);
      }

      std::list<Value *> WorkList;
      addToResolve(V1, WorkList);
      addToResolve(V2, WorkList);
      resolve(WorkList);
    }

    DominatorTree *DT;
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
    /// Forwards - Adds new properties into PropertySet and uses them to
    /// simplify instructions. Because new properties sometimes apply to
    /// a transition from one BasicBlock to another, this will use the
    /// PredicateSimplifier::proceedToSuccessor(s) interface to enter the
    /// basic block with the new PropertySet.
    class Forwards : public InstVisitor<Forwards> {
      friend class InstVisitor<Forwards>;
      PredicateSimplifier *PS;
    public:
      PropertySet &KP;

      Forwards(PredicateSimplifier *PS, PropertySet &KP) : PS(PS), KP(KP) {}

      // Tries to simplify each Instruction and add new properties to
      // the PropertySet. Returns true if it erase the instruction.
      //void visitInstruction(Instruction *I);

      void visitTerminatorInst(TerminatorInst &TI);
      void visitBranchInst(BranchInst &BI);
      void visitSwitchInst(SwitchInst &SI);

      void visitAllocaInst(AllocaInst &AI);
      void visitLoadInst(LoadInst &LI);
      void visitStoreInst(StoreInst &SI);
      void visitBinaryOperator(BinaryOperator &BO);
    };

    // Used by terminator instructions to proceed from the current basic
    // block to the next. Verifies that "current" dominates "next",
    // then calls visitBasicBlock.
    void proceedToSuccessors(PropertySet &CurrentPS, BasicBlock *Current);
    void proceedToSuccessor(PropertySet &Properties, BasicBlock *Next);

    // Visits each instruction in the basic block.
    void visitBasicBlock(BasicBlock *Block, PropertySet &KnownProperties);

    // Tries to simplify each Instruction and add new properties to
    // the PropertySet.
    void visitInstruction(Instruction *I, PropertySet &);

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
  AU.addRequiredID(BreakCriticalEdgesID);
  AU.addRequired<DominatorTree>();
  AU.setPreservesCFG();
  AU.addPreservedID(BreakCriticalEdgesID);
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
  Value *V = KnownProperties.canonicalize(I);
  if (V != I) {
    modified = true;
    ++NumInstruction;
    DEBUG(std::cerr << "Removing " << *I);
    KnownProperties.remove(I);
    I->replaceAllUsesWith(V);
    I->eraseFromParent();
    return;
  }

  // Try to substitute operands.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    Value *Oper = I->getOperand(i);
    Value *V = KnownProperties.canonicalize(Oper);
    if (V != Oper) {
      modified = true;
      ++NumVarsReplaced;
      DEBUG(std::cerr << "Resolving " << *I);
      I->setOperand(i, V);
      DEBUG(std::cerr << "into " << *I);
    }
  }

  Forwards visit(this, KnownProperties);
  visit.visit(*I);
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

void PredicateSimplifier::proceedToSuccessor(PropertySet &KP, BasicBlock *BB) {
  visitBasicBlock(BB, KP);
}

void PredicateSimplifier::Forwards::visitTerminatorInst(TerminatorInst &TI) {
  PS->proceedToSuccessors(KP, TI.getParent());
}

void PredicateSimplifier::Forwards::visitBranchInst(BranchInst &BI) {
  BasicBlock *BB = BI.getParent();

  if (BI.isUnconditional()) {
    PS->proceedToSuccessors(KP, BB);
    return;
  }

  Value *Condition = BI.getCondition();

  BasicBlock *TrueDest  = BI.getSuccessor(0),
             *FalseDest = BI.getSuccessor(1);

  if (isa<ConstantBool>(Condition) || TrueDest == FalseDest) {
    PS->proceedToSuccessors(KP, BB);
    return;
  }

  DTNodeType *Node = PS->DT->getNode(BB);
  for (DTNodeType::iterator I = Node->begin(), E = Node->end(); I != E; ++I) {
    BasicBlock *Dest = (*I)->getBlock();
    PropertySet DestProperties(KP);

    if (Dest == TrueDest)
      DestProperties.addEqual(ConstantBool::getTrue(), Condition);
    else if (Dest == FalseDest)
      DestProperties.addEqual(ConstantBool::getFalse(), Condition);

    PS->proceedToSuccessor(DestProperties, Dest);
  }
}

void PredicateSimplifier::Forwards::visitSwitchInst(SwitchInst &SI) {
  Value *Condition = SI.getCondition();

  // Set the EQProperty in each of the cases BBs,
  // and the NEProperties in the default BB.
  PropertySet DefaultProperties(KP);

  DTNodeType *Node = PS->DT->getNode(SI.getParent());
  for (DTNodeType::iterator I = Node->begin(), E = Node->end(); I != E; ++I) {
    BasicBlock *BB = (*I)->getBlock();

    PropertySet BBProperties(KP);
    if (BB == SI.getDefaultDest()) {
      for (unsigned i = 1, e = SI.getNumCases(); i < e; ++i)
        if (SI.getSuccessor(i) != BB)
          BBProperties.addNotEqual(Condition, SI.getCaseValue(i));
    } else if (ConstantInt *CI = SI.findCaseDest(BB)) {
      BBProperties.addEqual(Condition, CI);
    }
    PS->proceedToSuccessor(BBProperties, BB);
  }
}

void PredicateSimplifier::Forwards::visitAllocaInst(AllocaInst &AI) {
  KP.addNotEqual(Constant::getNullValue(AI.getType()), &AI);
}

void PredicateSimplifier::Forwards::visitLoadInst(LoadInst &LI) {
  Value *Ptr = LI.getPointerOperand();
  KP.addNotEqual(Constant::getNullValue(Ptr->getType()), Ptr);
}

void PredicateSimplifier::Forwards::visitStoreInst(StoreInst &SI) {
  Value *Ptr = SI.getPointerOperand();
  KP.addNotEqual(Constant::getNullValue(Ptr->getType()), Ptr);
}

void PredicateSimplifier::Forwards::visitBinaryOperator(BinaryOperator &BO) {
  Instruction::BinaryOps ops = BO.getOpcode();

  switch (ops) {
    case Instruction::Div:
    case Instruction::Rem: {
      Value *Divisor = BO.getOperand(1);
      KP.addNotEqual(Constant::getNullValue(Divisor->getType()), Divisor);
      break;
    }
    default:
      break;
  }
}
