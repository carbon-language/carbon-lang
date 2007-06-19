//===- GVNPRE.cpp - Eliminate redundant values and expressions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs a hybrid of global value numbering and partial redundancy
// elimination, known as GVN-PRE.  It performs partial redundancy elimination on
// values, rather than lexical expressions, allowing a more comprehensive view 
// the optimization.  It replaces redundant values with uses of earlier 
// occurences of the same value.  While this is beneficial in that it eliminates
// unneeded computation, it also increases register pressure by creating large
// live ranges, and should be used with caution on platforms that are very 
// sensitive to register pressure.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "gvnpre"
#include "llvm/Value.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <deque>
#include <map>
#include <vector>
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                         ValueTable Class
//===----------------------------------------------------------------------===//

/// This class holds the mapping between values and value numbers.

namespace {
  class VISIBILITY_HIDDEN ValueTable {
    public:
      struct Expression {
        enum ExpressionOpcode { ADD, SUB, MUL, UDIV, SDIV, FDIV, UREM, SREM, 
                              FREM, SHL, LSHR, ASHR, AND, OR, XOR, ICMPEQ, 
                              ICMPNE, ICMPUGT, ICMPUGE, ICMPULT, ICMPULE, 
                              ICMPSGT, ICMPSGE, ICMPSLT, ICMPSLE, FCMPOEQ, 
                              FCMPOGT, FCMPOGE, FCMPOLT, FCMPOLE, FCMPONE, 
                              FCMPORD, FCMPUNO, FCMPUEQ, FCMPUGT, FCMPUGE, 
                              FCMPULT, FCMPULE, FCMPUNE };
    
        ExpressionOpcode opcode;
        uint32_t leftVN;
        uint32_t rightVN;
      
        bool operator< (const Expression& other) const {
          if (opcode < other.opcode)
            return true;
          else if (opcode > other.opcode)
            return false;
          else if (leftVN < other.leftVN)
            return true;
          else if (leftVN > other.leftVN)
            return false;
          else if (rightVN < other.rightVN)
            return true;
          else if (rightVN > other.rightVN)
            return false;
          else
            return false;
        }
      };
    
    private:
      std::map<Value*, uint32_t> valueNumbering;
      std::map<Expression, uint32_t> expressionNumbering;
  
      std::set<Expression> maximalExpressions;
      std::set<Value*> maximalValues;
  
      uint32_t nextValueNumber;
    
      Expression::ExpressionOpcode getOpcode(BinaryOperator* BO);
      Expression::ExpressionOpcode getOpcode(CmpInst* C);
    public:
      ValueTable() { nextValueNumber = 1; }
      uint32_t lookup_or_add(Value* V);
      uint32_t lookup(Value* V);
      void add(Value* V, uint32_t num);
      void clear();
      std::set<Expression>& getMaximalExpressions() {
        return maximalExpressions;
      
      }
      std::set<Value*>& getMaximalValues() { return maximalValues; }
      Expression create_expression(BinaryOperator* BO);
      Expression create_expression(CmpInst* C);
      void erase(Value* v);
  };
}

ValueTable::Expression::ExpressionOpcode 
                                     ValueTable::getOpcode(BinaryOperator* BO) {
  switch(BO->getOpcode()) {
    case Instruction::Add:
      return Expression::ADD;
    case Instruction::Sub:
      return Expression::SUB;
    case Instruction::Mul:
      return Expression::MUL;
    case Instruction::UDiv:
      return Expression::UDIV;
    case Instruction::SDiv:
      return Expression::SDIV;
    case Instruction::FDiv:
      return Expression::FDIV;
    case Instruction::URem:
      return Expression::UREM;
    case Instruction::SRem:
      return Expression::SREM;
    case Instruction::FRem:
      return Expression::FREM;
    case Instruction::Shl:
      return Expression::SHL;
    case Instruction::LShr:
      return Expression::LSHR;
    case Instruction::AShr:
      return Expression::ASHR;
    case Instruction::And:
      return Expression::AND;
    case Instruction::Or:
      return Expression::OR;
    case Instruction::Xor:
      return Expression::XOR;
    
    // THIS SHOULD NEVER HAPPEN
    default:
      assert(0 && "Binary operator with unknown opcode?");
      return Expression::ADD;
  }
}

ValueTable::Expression::ExpressionOpcode ValueTable::getOpcode(CmpInst* C) {
  if (C->getOpcode() == Instruction::ICmp) {
    switch (C->getPredicate()) {
      case ICmpInst::ICMP_EQ:
        return Expression::ICMPEQ;
      case ICmpInst::ICMP_NE:
        return Expression::ICMPNE;
      case ICmpInst::ICMP_UGT:
        return Expression::ICMPUGT;
      case ICmpInst::ICMP_UGE:
        return Expression::ICMPUGE;
      case ICmpInst::ICMP_ULT:
        return Expression::ICMPULT;
      case ICmpInst::ICMP_ULE:
        return Expression::ICMPULE;
      case ICmpInst::ICMP_SGT:
        return Expression::ICMPSGT;
      case ICmpInst::ICMP_SGE:
        return Expression::ICMPSGE;
      case ICmpInst::ICMP_SLT:
        return Expression::ICMPSLT;
      case ICmpInst::ICMP_SLE:
        return Expression::ICMPSLE;
      
      // THIS SHOULD NEVER HAPPEN
      default:
        assert(0 && "Comparison with unknown predicate?");
        return Expression::ICMPEQ;
    }
  } else {
    switch (C->getPredicate()) {
      case FCmpInst::FCMP_OEQ:
        return Expression::FCMPOEQ;
      case FCmpInst::FCMP_OGT:
        return Expression::FCMPOGT;
      case FCmpInst::FCMP_OGE:
        return Expression::FCMPOGE;
      case FCmpInst::FCMP_OLT:
        return Expression::FCMPOLT;
      case FCmpInst::FCMP_OLE:
        return Expression::FCMPOLE;
      case FCmpInst::FCMP_ONE:
        return Expression::FCMPONE;
      case FCmpInst::FCMP_ORD:
        return Expression::FCMPORD;
      case FCmpInst::FCMP_UNO:
        return Expression::FCMPUNO;
      case FCmpInst::FCMP_UEQ:
        return Expression::FCMPUEQ;
      case FCmpInst::FCMP_UGT:
        return Expression::FCMPUGT;
      case FCmpInst::FCMP_UGE:
        return Expression::FCMPUGE;
      case FCmpInst::FCMP_ULT:
        return Expression::FCMPULT;
      case FCmpInst::FCMP_ULE:
        return Expression::FCMPULE;
      case FCmpInst::FCMP_UNE:
        return Expression::FCMPUNE;
      
      // THIS SHOULD NEVER HAPPEN
      default:
        assert(0 && "Comparison with unknown predicate?");
        return Expression::FCMPOEQ;
    }
  }
}

uint32_t ValueTable::lookup_or_add(Value* V) {
  maximalValues.insert(V);

  std::map<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    return VI->second;
  
  
  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V)) {
    Expression e = create_expression(BO);
    
    std::map<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (CmpInst* C = dyn_cast<CmpInst>(V)) {
    Expression e = create_expression(C);
    
    std::map<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else {
    valueNumbering.insert(std::make_pair(V, nextValueNumber));
    return nextValueNumber++;
  }
}

uint32_t ValueTable::lookup(Value* V) {
  std::map<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    return VI->second;
  else
    assert(0 && "Value not numbered?");
  
  return 0;
}

void ValueTable::add(Value* V, uint32_t num) {
  std::map<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    valueNumbering.erase(VI);
  valueNumbering.insert(std::make_pair(V, num));
}

ValueTable::Expression ValueTable::create_expression(BinaryOperator* BO) {
  Expression e;
    
  e.leftVN = lookup_or_add(BO->getOperand(0));
  e.rightVN = lookup_or_add(BO->getOperand(1));
  e.opcode = getOpcode(BO);
  
  maximalExpressions.insert(e);
  
  return e;
}

ValueTable::Expression ValueTable::create_expression(CmpInst* C) {
  Expression e;
    
  e.leftVN = lookup_or_add(C->getOperand(0));
  e.rightVN = lookup_or_add(C->getOperand(1));
  e.opcode = getOpcode(C);
  
  maximalExpressions.insert(e);
  
  return e;
}

void ValueTable::clear() {
  valueNumbering.clear();
  expressionNumbering.clear();
  maximalExpressions.clear();
  maximalValues.clear();
  nextValueNumber = 1;
}

void ValueTable::erase(Value* V) {
  maximalValues.erase(V);
  valueNumbering.erase(V);
  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V))
    maximalExpressions.erase(create_expression(BO));
  else if (CmpInst* C = dyn_cast<CmpInst>(V))
    maximalExpressions.erase(create_expression(C));
}

namespace {

  class VISIBILITY_HIDDEN GVNPRE : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    GVNPRE() : FunctionPass((intptr_t)&ID) { }

  private:
    ValueTable VN;
    std::vector<Instruction*> createdExpressions;
    
    std::map<BasicBlock*, std::set<Value*> > availableOut;
    std::map<BasicBlock*, std::set<Value*> > anticipatedIn;
    std::map<User*, bool> invokeDep;
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<PostDominatorTree>();
    }
  
    // Helper fuctions
    // FIXME: eliminate or document these better
    void dump(const std::set<Value*>& s) const;
    void dump_unique(const std::set<Value*>& s) const;
    void clean(std::set<Value*>& set);
    Value* find_leader(std::set<Value*>& vals,
                       uint32_t v);
    Value* phi_translate(Value* V, BasicBlock* pred, BasicBlock* succ);
    void phi_translate_set(std::set<Value*>& anticIn, BasicBlock* pred,
                           BasicBlock* succ, std::set<Value*>& out);
    
    void topo_sort(std::set<Value*>& set,
                   std::vector<Value*>& vec);
    
    // For a given block, calculate the generated expressions, temporaries,
    // and the AVAIL_OUT set
    void cleanup();
    void elimination();
    
    void val_insert(std::set<Value*>& s, Value* v);
    void val_replace(std::set<Value*>& s, Value* v);
    bool dependsOnInvoke(Value* V);
  
  };
  
  char GVNPRE::ID = 0;
  
}

FunctionPass *llvm::createGVNPREPass() { return new GVNPRE(); }

RegisterPass<GVNPRE> X("gvnpre",
                       "Global Value Numbering/Partial Redundancy Elimination");


STATISTIC(NumInsertedVals, "Number of values inserted");
STATISTIC(NumInsertedPhis, "Number of PHI nodes inserted");
STATISTIC(NumEliminated, "Number of redundant instructions eliminated");

Value* GVNPRE::find_leader(std::set<Value*>& vals, uint32_t v) {
  for (std::set<Value*>::iterator I = vals.begin(), E = vals.end();
       I != E; ++I)
    if (v == VN.lookup(*I))
      return *I;
  
  return 0;
}

void GVNPRE::val_insert(std::set<Value*>& s, Value* v) {
  uint32_t num = VN.lookup(v);
  Value* leader = find_leader(s, num);
  if (leader == 0)
    s.insert(v);
}

void GVNPRE::val_replace(std::set<Value*>& s, Value* v) {
  uint32_t num = VN.lookup(v);
  Value* leader = find_leader(s, num);
  while (leader != 0) {
    s.erase(leader);
    leader = find_leader(s, num);
  }
  s.insert(v);
}

Value* GVNPRE::phi_translate(Value* V, BasicBlock* pred, BasicBlock* succ) {
  if (V == 0)
    return 0;
  
  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V)) {
    Value* newOp1 = 0;
    if (isa<Instruction>(BO->getOperand(0)))
      newOp1 = phi_translate(find_leader(anticipatedIn[succ],         
                                         VN.lookup(BO->getOperand(0))),
                             pred, succ);
    else
      newOp1 = BO->getOperand(0);
    
    if (newOp1 == 0)
      return 0;
    
    Value* newOp2 = 0;
    if (isa<Instruction>(BO->getOperand(1)))
      newOp2 = phi_translate(find_leader(anticipatedIn[succ],         
                                         VN.lookup(BO->getOperand(1))),
                             pred, succ);
    else
      newOp2 = BO->getOperand(1);
    
    if (newOp2 == 0)
      return 0;
    
    if (newOp1 != BO->getOperand(0) || newOp2 != BO->getOperand(1)) {
      Instruction* newVal = BinaryOperator::create(BO->getOpcode(),
                                             newOp1, newOp2,
                                             BO->getName()+".expr");
      
      uint32_t v = VN.lookup_or_add(newVal);
      
      Value* leader = find_leader(availableOut[pred], v);
      if (leader == 0) {
        createdExpressions.push_back(newVal);
        return newVal;
      } else {
        VN.erase(newVal);
        delete newVal;
        return leader;
      }
    }
  } else if (PHINode* P = dyn_cast<PHINode>(V)) {
    if (P->getParent() == succ)
      return P->getIncomingValueForBlock(pred);
  } else if (CmpInst* C = dyn_cast<CmpInst>(V)) {
    Value* newOp1 = 0;
    if (isa<Instruction>(C->getOperand(0)))
      newOp1 = phi_translate(find_leader(anticipatedIn[succ],         
                                         VN.lookup(C->getOperand(0))),
                             pred, succ);
    else
      newOp1 = C->getOperand(0);
    
    if (newOp1 == 0)
      return 0;
    
    Value* newOp2 = 0;
    if (isa<Instruction>(C->getOperand(1)))
      newOp2 = phi_translate(find_leader(anticipatedIn[succ],         
                                         VN.lookup(C->getOperand(1))),
                             pred, succ);
    else
      newOp2 = C->getOperand(1);
      
    if (newOp2 == 0)
      return 0;
    
    if (newOp1 != C->getOperand(0) || newOp2 != C->getOperand(1)) {
      Instruction* newVal = CmpInst::create(C->getOpcode(),
                                            C->getPredicate(),
                                             newOp1, newOp2,
                                             C->getName()+".expr");
      
      uint32_t v = VN.lookup_or_add(newVal);
        
      Value* leader = find_leader(availableOut[pred], v);
      if (leader == 0) {
        createdExpressions.push_back(newVal);
        return newVal;
      } else {
        VN.erase(newVal);
        delete newVal;
        return leader;
      }
    }
  }
  
  return V;
}

void GVNPRE::phi_translate_set(std::set<Value*>& anticIn,
                              BasicBlock* pred, BasicBlock* succ,
                              std::set<Value*>& out) {
  for (std::set<Value*>::iterator I = anticIn.begin(),
       E = anticIn.end(); I != E; ++I) {
    Value* V = phi_translate(*I, pred, succ);
    if (V != 0)
      out.insert(V);
  }
}

bool GVNPRE::dependsOnInvoke(Value* V) {
  if (PHINode* p = dyn_cast<PHINode>(V)) {
    for (PHINode::op_iterator I = p->op_begin(), E = p->op_end(); I != E; ++I)
      if (isa<InvokeInst>(*I))
        return true;
    return false;
  } else {
    return false;
  }
}

// Remove all expressions whose operands are not themselves in the set
void GVNPRE::clean(std::set<Value*>& set) {
  std::vector<Value*> worklist;
  topo_sort(set, worklist);
  
  for (unsigned i = 0; i < worklist.size(); ++i) {
    Value* v = worklist[i];
    
    if (BinaryOperator* BO = dyn_cast<BinaryOperator>(v)) {   
      bool lhsValid = !isa<Instruction>(BO->getOperand(0));
      if (!lhsValid)
        for (std::set<Value*>::iterator I = set.begin(), E = set.end();
             I != E; ++I)
          if (VN.lookup(*I) == VN.lookup(BO->getOperand(0))) {
            lhsValid = true;
            break;
          }
      if (lhsValid)
        lhsValid = !dependsOnInvoke(BO->getOperand(0));
    
      bool rhsValid = !isa<Instruction>(BO->getOperand(1));
      if (!rhsValid)
        for (std::set<Value*>::iterator I = set.begin(), E = set.end();
             I != E; ++I)
          if (VN.lookup(*I) == VN.lookup(BO->getOperand(1))) {
            rhsValid = true;
            break;
          }
      if (rhsValid)
        rhsValid = !dependsOnInvoke(BO->getOperand(1));
      
      if (!lhsValid || !rhsValid)
        set.erase(BO);
    } else if (CmpInst* C = dyn_cast<CmpInst>(v)) {
      bool lhsValid = !isa<Instruction>(C->getOperand(0));
      if (!lhsValid)
        for (std::set<Value*>::iterator I = set.begin(), E = set.end();
             I != E; ++I)
          if (VN.lookup(*I) == VN.lookup(C->getOperand(0))) {
            lhsValid = true;
            break;
          }
      if (lhsValid)
        lhsValid = !dependsOnInvoke(C->getOperand(0));
      
      bool rhsValid = !isa<Instruction>(C->getOperand(1));
      if (!rhsValid)
      for (std::set<Value*>::iterator I = set.begin(), E = set.end();
           I != E; ++I)
        if (VN.lookup(*I) == VN.lookup(C->getOperand(1))) {
          rhsValid = true;
          break;
        }
      if (rhsValid)
        rhsValid = !dependsOnInvoke(C->getOperand(1));
    
      if (!lhsValid || !rhsValid)
        set.erase(C);
    }
  }
}

void GVNPRE::topo_sort(std::set<Value*>& set,
                       std::vector<Value*>& vec) {
  std::set<Value*> toErase;
  for (std::set<Value*>::iterator I = set.begin(), E = set.end();
       I != E; ++I) {
    if (BinaryOperator* BO = dyn_cast<BinaryOperator>(*I))
      for (std::set<Value*>::iterator SI = set.begin(); SI != E; ++SI) {
        if (VN.lookup(BO->getOperand(0)) == VN.lookup(*SI) ||
            VN.lookup(BO->getOperand(1)) == VN.lookup(*SI)) {
          toErase.insert(*SI);
        }
      }
    else if (CmpInst* C = dyn_cast<CmpInst>(*I))
      for (std::set<Value*>::iterator SI = set.begin(); SI != E; ++SI) {
        if (VN.lookup(C->getOperand(0)) == VN.lookup(*SI) ||
            VN.lookup(C->getOperand(1)) == VN.lookup(*SI)) {
          toErase.insert(*SI);
        }
      }
  }
  
  std::vector<Value*> Q;
  for (std::set<Value*>::iterator I = set.begin(), E = set.end();
       I != E; ++I) {
    if (toErase.find(*I) == toErase.end())
      Q.push_back(*I);
  }
  
  std::set<Value*> visited;
  while (!Q.empty()) {
    Value* e = Q.back();
  
    if (BinaryOperator* BO = dyn_cast<BinaryOperator>(e)) {
      Value* l = find_leader(set, VN.lookup(BO->getOperand(0)));
      Value* r = find_leader(set, VN.lookup(BO->getOperand(1)));
      
      if (l != 0 && isa<Instruction>(l) &&
          visited.find(l) == visited.end())
        Q.push_back(l);
      else if (r != 0 && isa<Instruction>(r) &&
               visited.find(r) == visited.end())
        Q.push_back(r);
      else {
        vec.push_back(e);
        visited.insert(e);
        Q.pop_back();
      }
    } else if (CmpInst* C = dyn_cast<CmpInst>(e)) {
      Value* l = find_leader(set, VN.lookup(C->getOperand(0)));
      Value* r = find_leader(set, VN.lookup(C->getOperand(1)));
      
      if (l != 0 && isa<Instruction>(l) &&
          visited.find(l) == visited.end())
        Q.push_back(l);
      else if (r != 0 && isa<Instruction>(r) &&
               visited.find(r) == visited.end())
        Q.push_back(r);
      else {
        vec.push_back(e);
        visited.insert(e);
        Q.pop_back();
      }
    } else {
      visited.insert(e);
      vec.push_back(e);
      Q.pop_back();
    }
  }
}


void GVNPRE::dump(const std::set<Value*>& s) const {
  DOUT << "{ ";
  for (std::set<Value*>::iterator I = s.begin(), E = s.end();
       I != E; ++I) {
    DEBUG((*I)->dump());
  }
  DOUT << "}\n\n";
}

void GVNPRE::dump_unique(const std::set<Value*>& s) const {
  DOUT << "{ ";
  for (std::set<Value*>::iterator I = s.begin(), E = s.end();
       I != E; ++I) {
    DEBUG((*I)->dump());
  }
  DOUT << "}\n\n";
}

void GVNPRE::elimination() {
  DOUT << "\n\nPhase 3: Elimination\n\n";
  
  std::vector<std::pair<Instruction*, Value*> > replace;
  std::vector<Instruction*> erase;
  
  DominatorTree& DT = getAnalysis<DominatorTree>();
  
  for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    BasicBlock* BB = DI->getBlock();
    
    DOUT << "Block: " << BB->getName() << "\n";
    dump_unique(availableOut[BB]);
    DOUT << "\n\n";
    
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE; ++BI) {

      if (isa<BinaryOperator>(BI) || isa<CmpInst>(BI)) {
         Value *leader = find_leader(availableOut[BB], VN.lookup(BI));
  
        if (leader != 0)
          if (Instruction* Instr = dyn_cast<Instruction>(leader))
            if (Instr->getParent() != 0 && Instr != BI) {
              replace.push_back(std::make_pair(BI, leader));
              erase.push_back(BI);
              ++NumEliminated;
            }
      }
    }
  }
  
  while (!replace.empty()) {
    std::pair<Instruction*, Value*> rep = replace.back();
    replace.pop_back();
    rep.first->replaceAllUsesWith(rep.second);
  }
    
  for (std::vector<Instruction*>::iterator I = erase.begin(), E = erase.end();
       I != E; ++I)
     (*I)->eraseFromParent();
}


void GVNPRE::cleanup() {
  while (!createdExpressions.empty()) {
    Instruction* I = createdExpressions.back();
    createdExpressions.pop_back();
    
    delete I;
  }
}

bool GVNPRE::runOnFunction(Function &F) {
  VN.clear();
  createdExpressions.clear();
  availableOut.clear();
  anticipatedIn.clear();
  invokeDep.clear();

  std::map<BasicBlock*, std::set<Value*> > generatedExpressions;
  std::map<BasicBlock*, std::set<PHINode*> > generatedPhis;
  std::map<BasicBlock*, std::set<Value*> > generatedTemporaries;
  
  
  DominatorTree &DT = getAnalysis<DominatorTree>();   
  
  // Phase 1: BuildSets
  
  // Phase 1, Part 1: calculate AVAIL_OUT
  
  // Top-down walk of the dominator tree
  for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    
    // Get the sets to update for this block
    std::set<Value*>& currExps = generatedExpressions[DI->getBlock()];
    std::set<PHINode*>& currPhis = generatedPhis[DI->getBlock()];
    std::set<Value*>& currTemps = generatedTemporaries[DI->getBlock()];
    std::set<Value*>& currAvail = availableOut[DI->getBlock()];     
    
    BasicBlock* BB = DI->getBlock();
  
    // A block inherits AVAIL_OUT from its dominator
    if (DI->getIDom() != 0)
    currAvail.insert(availableOut[DI->getIDom()->getBlock()].begin(),
                     availableOut[DI->getIDom()->getBlock()].end());
    
    
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE; ++BI) {
       
      // Handle PHI nodes...
      if (PHINode* p = dyn_cast<PHINode>(BI)) {
        VN.lookup_or_add(p);
        currPhis.insert(p);
    
      // Handle binary ops...
      } else if (BinaryOperator* BO = dyn_cast<BinaryOperator>(BI)) {
        Value* leftValue = BO->getOperand(0);
        Value* rightValue = BO->getOperand(1);
      
        VN.lookup_or_add(BO);
      
        if (isa<Instruction>(leftValue))
          val_insert(currExps, leftValue);
        if (isa<Instruction>(rightValue))
          val_insert(currExps, rightValue);
        val_insert(currExps, BO);
      
      // Handle cmp ops...
      } else if (CmpInst* C = dyn_cast<CmpInst>(BI)) {
        Value* leftValue = C->getOperand(0);
        Value* rightValue = C->getOperand(1);
      
        VN.lookup_or_add(C);
      
        if (isa<Instruction>(leftValue))
          val_insert(currExps, leftValue);
        if (isa<Instruction>(rightValue))
          val_insert(currExps, rightValue);
        val_insert(currExps, C);
      
      // Handle unsupported ops
      } else if (!BI->isTerminator()){
        VN.lookup_or_add(BI);
        currTemps.insert(BI);
      }
    
      if (!BI->isTerminator())
        val_insert(currAvail, BI);
    }
  }
  
  DOUT << "Maximal Set: ";
  dump_unique(VN.getMaximalValues());
  DOUT << "\n";
  
  // If function has no exit blocks, only perform GVN
  PostDominatorTree &PDT = getAnalysis<PostDominatorTree>();
  if (PDT[&F.getEntryBlock()] == 0) {
    elimination();
    cleanup();
    
    return true;
  }
  
  
  // Phase 1, Part 2: calculate ANTIC_IN
  
  std::set<BasicBlock*> visited;
  
  bool changed = true;
  unsigned iterations = 0;
  while (changed) {
    changed = false;
    std::set<Value*> anticOut;
    
    // Top-down walk of the postdominator tree
    for (df_iterator<DomTreeNode*> PDI = 
         df_begin(PDT.getRootNode()), E = df_end(PDT.getRootNode());
         PDI != E; ++PDI) {
      BasicBlock* BB = PDI->getBlock();
      if (BB == 0)
        continue;
      
      DOUT << "Block: " << BB->getName() << "\n";
      DOUT << "TMP_GEN: ";
      dump(generatedTemporaries[BB]);
      DOUT << "\n";
    
      DOUT << "EXP_GEN: ";
      dump_unique(generatedExpressions[BB]);
      visited.insert(BB);
      
      std::set<Value*>& anticIn = anticipatedIn[BB];
      std::set<Value*> old (anticIn.begin(), anticIn.end());
      
      if (BB->getTerminator()->getNumSuccessors() == 1) {
         if (visited.find(BB->getTerminator()->getSuccessor(0)) == 
             visited.end())
           phi_translate_set(VN.getMaximalValues(), BB,    
                             BB->getTerminator()->getSuccessor(0),
                             anticOut);
         else
          phi_translate_set(anticipatedIn[BB->getTerminator()->getSuccessor(0)],
                            BB,  BB->getTerminator()->getSuccessor(0), 
                            anticOut);
      } else if (BB->getTerminator()->getNumSuccessors() > 1) {
        BasicBlock* first = BB->getTerminator()->getSuccessor(0);
        anticOut.insert(anticipatedIn[first].begin(),
                        anticipatedIn[first].end());
        for (unsigned i = 1; i < BB->getTerminator()->getNumSuccessors(); ++i) {
          BasicBlock* currSucc = BB->getTerminator()->getSuccessor(i);
          std::set<Value*>& succAnticIn = anticipatedIn[currSucc];
          
          std::set<Value*> temp;
          std::insert_iterator<std::set<Value*> >  temp_ins(temp, 
                                                            temp.begin());
          std::set_intersection(anticOut.begin(), anticOut.end(),
                                succAnticIn.begin(), succAnticIn.end(),
                                temp_ins);
          
          anticOut.clear();
          anticOut.insert(temp.begin(), temp.end());
        }
      }
      
      DOUT << "ANTIC_OUT: ";
      dump_unique(anticOut);
      DOUT << "\n";
      
      std::set<Value*> S;
      std::insert_iterator<std::set<Value*> >  s_ins(S, S.begin());
      std::set_difference(anticOut.begin(), anticOut.end(),
                     generatedTemporaries[BB].begin(),
                     generatedTemporaries[BB].end(),
                     s_ins);
      
      anticIn.clear();
      std::insert_iterator<std::set<Value*> >  ai_ins(anticIn, anticIn.begin());
      std::set_difference(generatedExpressions[BB].begin(),
                     generatedExpressions[BB].end(),
                     generatedTemporaries[BB].begin(),
                     generatedTemporaries[BB].end(),
                     ai_ins);
      
      for (std::set<Value*>::iterator I = S.begin(), E = S.end();
           I != E; ++I) {
        // For non-opaque values, we should already have a value numbering.
        // However, for opaques, such as constants within PHI nodes, it is
        // possible that they have not yet received a number.  Make sure they do
        // so now.
        uint32_t valNum = 0;
        if (isa<BinaryOperator>(*I) || isa<CmpInst>(*I))
          valNum = VN.lookup(*I);
        else
          valNum = VN.lookup_or_add(*I);
        if (find_leader(anticIn, valNum) == 0)
          val_insert(anticIn, *I);
      }
      
      clean(anticIn);
      
      DOUT << "ANTIC_IN: ";
      dump_unique(anticIn);
      DOUT << "\n";
      
      if (old.size() != anticIn.size())
        changed = true;
      
      anticOut.clear();
    }
    
    iterations++;
  }
  
  DOUT << "Iterations: " << iterations << "\n";
  
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    DOUT << "Name: " << I->getName().c_str() << "\n";
    
    DOUT << "TMP_GEN: ";
    dump(generatedTemporaries[I]);
    DOUT << "\n";
    
    DOUT << "EXP_GEN: ";
    dump_unique(generatedExpressions[I]);
    DOUT << "\n";
    
    DOUT << "ANTIC_IN: ";
    dump_unique(anticipatedIn[I]);
    DOUT << "\n";
    
    DOUT << "AVAIL_OUT: ";
    dump_unique(availableOut[I]);
    DOUT << "\n";
  }
  
  // Phase 2: Insert
  DOUT<< "\nPhase 2: Insertion\n";
  
  std::map<BasicBlock*, std::set<Value*> > new_sets;
  unsigned i_iterations = 0;
  bool new_stuff = true;
  while (new_stuff) {
    new_stuff = false;
    DOUT << "Iteration: " << i_iterations << "\n\n";
    for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
      BasicBlock* BB = DI->getBlock();
      
      if (BB == 0)
        continue;
      
      std::set<Value*>& new_set = new_sets[BB];
      std::set<Value*>& availOut = availableOut[BB];
      std::set<Value*>& anticIn = anticipatedIn[BB];
      
      new_set.clear();
      
      // Replace leaders with leaders inherited from dominator
      if (DI->getIDom() != 0) {
        std::set<Value*>& dom_set = new_sets[DI->getIDom()->getBlock()];
        for (std::set<Value*>::iterator I = dom_set.begin(),
             E = dom_set.end(); I != E; ++I) {
          new_set.insert(*I);
          val_replace(availOut, *I);
        }
      }
      
      // If there is more than one predecessor...
      if (pred_begin(BB) != pred_end(BB) && ++pred_begin(BB) != pred_end(BB)) {
        std::vector<Value*> workList;
        topo_sort(anticIn, workList);
        
        DOUT << "Merge Block: " << BB->getName() << "\n";
        DOUT << "ANTIC_IN: ";
        dump_unique(anticIn);
        DOUT << "\n";
        
        for (unsigned i = 0; i < workList.size(); ++i) {
          Value* e = workList[i];
          
          if (isa<BinaryOperator>(e) || isa<CmpInst>(e)) {
            if (find_leader(availableOut[DI->getIDom()->getBlock()], VN.lookup(e)) != 0)
              continue;
            
            std::map<BasicBlock*, Value*> avail;
            bool by_some = false;
            int num_avail = 0;
            
            for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE;
                 ++PI) {
              Value *e2 = phi_translate(e, *PI, BB);
              Value *e3 = find_leader(availableOut[*PI], VN.lookup(e2));
              
              if (e3 == 0) {
                std::map<BasicBlock*, Value*>::iterator av = avail.find(*PI);
                if (av != avail.end())
                  avail.erase(av);
                avail.insert(std::make_pair(*PI, e2));
              } else {
                std::map<BasicBlock*, Value*>::iterator av = avail.find(*PI);
                if (av != avail.end())
                  avail.erase(av);
                avail.insert(std::make_pair(*PI, e3));
                
                by_some = true;
                num_avail++;
              }
            }
            
            if (by_some &&
                num_avail < std::distance(pred_begin(BB), pred_end(BB))) {
              DOUT << "Processing Value: ";
              DEBUG(e->dump());
              DOUT << "\n\n";
            
              for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
                   PI != PE; ++PI) {
                Value* e2 = avail[*PI];
                if (!find_leader(availableOut[*PI], VN.lookup(e2))) {
                  User* U = cast<User>(e2);
                
                  Value* s1 = 0;
                  if (isa<BinaryOperator>(U->getOperand(0)) ||
                      isa<CmpInst>(U->getOperand(0)))
                    s1 = find_leader(availableOut[*PI], VN.lookup(U->getOperand(0)));
                  else
                    s1 = U->getOperand(0);
                  
                  Value* s2 = 0;
                  if (isa<BinaryOperator>(U->getOperand(1)) ||
                      isa<CmpInst>(U->getOperand(1)))
                    s2 = find_leader(availableOut[*PI], VN.lookup(U->getOperand(1)));
                  else
                    s2 = U->getOperand(1);
                  
                  Value* newVal = 0;
                  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(U))
                    newVal = BinaryOperator::create(BO->getOpcode(),
                                             s1, s2,
                                             BO->getName()+".gvnpre",
                                             (*PI)->getTerminator());
                  else if (CmpInst* C = dyn_cast<CmpInst>(U))
                    newVal = CmpInst::create(C->getOpcode(),
                                             C->getPredicate(),
                                             s1, s2,
                                             C->getName()+".gvnpre",
                                             (*PI)->getTerminator());
                  
                  VN.add(newVal, VN.lookup(U));
                  
                  std::set<Value*>& predAvail = availableOut[*PI];
                  val_replace(predAvail, newVal);
                  
                  DOUT << "Creating value: " << std::hex << newVal << std::dec << "\n";
                  
                  std::map<BasicBlock*, Value*>::iterator av = avail.find(*PI);
                  if (av != avail.end())
                    avail.erase(av);
                  avail.insert(std::make_pair(*PI, newVal));
                  
                  ++NumInsertedVals;
                }
              }
              
              PHINode* p = 0;
              
              for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
                   PI != PE; ++PI) {
                if (p == 0)
                  p = new PHINode(avail[*PI]->getType(), "gvnpre-join", 
                                  BB->begin());
                
                p->addIncoming(avail[*PI], *PI);
              }
              
              VN.add(p, VN.lookup(e));
              DOUT << "Creating value: " << std::hex << p << std::dec << "\n";
              
              val_replace(availOut, p);
              availOut.insert(p);
              
              new_stuff = true;
              
              DOUT << "Preds After Processing: ";
              for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
                   PI != PE; ++PI)
                DEBUG((*PI)->dump());
              DOUT << "\n\n";
              
              DOUT << "Merge Block After Processing: ";
              DEBUG(BB->dump());
              DOUT << "\n\n";
              
              new_set.insert(p);
              
              ++NumInsertedPhis;
            }
          }
        }
      }
    }
    i_iterations++;
  }
  
  // Phase 3: Eliminate
  elimination();
  
  // Phase 4: Cleanup
  cleanup();
  
  return true;
}
