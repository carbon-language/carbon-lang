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
// live ranges, and should be used with caution on platforms that a very 
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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <map>
#include <set>
using namespace llvm;

namespace {

  class VISIBILITY_HIDDEN GVNPRE : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    GVNPRE() : FunctionPass((intptr_t)&ID) { nextValueNumber = 0; }

  private:
    uint32_t nextValueNumber;
  
    struct Expression {
      char opcode;
      Value* value;
      uint32_t lhs;
      uint32_t rhs;
      
      bool operator<(const Expression& other) const {
        if (opcode < other.opcode)
          return true;
        else if (other.opcode < opcode)
          return false;
        
        if (opcode == 0) {
          if (value < other.value)
            return true;
          else
            return false;
        } else {
          if (lhs < other.lhs)
            return true;
          else if (other.lhs < lhs)
            return true;
          else if (rhs < other.rhs)
            return true;
          else
            return false;
        }
      }
      
      bool operator==(const Expression& other) const {
        if (opcode != other.opcode)
          return false;
        
        if (value != other.value)
          return false;
        
        if (lhs != other.lhs)
          return false;
          
        if (rhs != other.rhs)
          return false;
        
        return true;
      }
    };
  
    typedef std::map<Expression, uint32_t> ValueTable;
  
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<PostDominatorTree>();
    }
  
    // Helper fuctions
    // FIXME: eliminate or document these better
    void dump(ValueTable& VN, std::set<Expression>& s);
    void clean(ValueTable VN, std::set<Expression>& set);
    Expression add(ValueTable& VN, std::set<Expression>& MS, Instruction* V);
    ValueTable::iterator lookup(ValueTable& VN, Value* V);
    Expression buildExpression(ValueTable& VN, Value* V);
    std::set<Expression>::iterator find_leader(ValueTable VN, 
                                               std::set<Expression>& vals,
                                               uint32_t v);
    void phi_translate(ValueTable& VN, 
                       std::set<Expression>& anticIn, BasicBlock* B,
                       std::set<Expression>& out);
    
    // For a given block, calculate the generated expressions, temporaries,
    // and the AVAIL_OUT set
    void CalculateAvailOut(ValueTable& VN, std::set<Expression>& MS,
                       DominatorTree::Node* DI,
                       std::set<Expression>& currExps,
                       std::set<PHINode*>& currPhis,
                       std::set<Expression>& currTemps,
                       std::set<Expression>& currAvail,
                       std::map<BasicBlock*, std::set<Expression> > availOut);
  
  };
  
  char GVNPRE::ID = 0;
  
}

FunctionPass *llvm::createGVNPREPass() { return new GVNPRE(); }

RegisterPass<GVNPRE> X("gvnpre",
                       "Global Value Numbering/Partial Redundancy Elimination");

// Given a Value, build an Expression to represent it
GVNPRE::Expression GVNPRE::buildExpression(ValueTable& VN, Value* V) {
  if (Instruction* I = dyn_cast<Instruction>(V)) {
    Expression e;
    
    switch (I->getOpcode()) {
    case 7:
      e.opcode = 1; // ADD
      break;
    case 8:
      e.opcode = 2; // SUB
      break;
    case 9:
      e.opcode = 3; // MUL
      break;
    case 10:
      e.opcode = 4; // UDIV
      break;
    case 11:
      e.opcode = 5; // SDIV
      break;
    case 12:
      e.opcode = 6; // FDIV
      break;
    case 13:
      e.opcode = 7; // UREM
      break;
    case 14:
      e.opcode = 8; // SREM
      break;
    case 15:
      e.opcode = 9; // FREM
      break;
    default:
     e.opcode = 0; // OPAQUE
     e.lhs = 0;
     e.rhs = 0;
     e.value = V;
     return e;
    }
    
    e.value = 0;
    
    ValueTable::iterator lhs = lookup(VN, I->getOperand(0));
    if (lhs == VN.end()) {
      Expression lhsExp = buildExpression(VN, I->getOperand(0));
      VN.insert(std::make_pair(lhsExp, nextValueNumber));
      e.lhs = nextValueNumber;
      nextValueNumber++;
    } else
      e.lhs = lhs->second;
    ValueTable::iterator rhs = lookup(VN, I->getOperand(1));
    if (rhs == VN.end()) {
      Expression rhsExp = buildExpression(VN, I->getOperand(1));
      VN.insert(std::make_pair(rhsExp, nextValueNumber));
      e.rhs = nextValueNumber;
      nextValueNumber++;
    } else
      e.rhs = rhs->second;
  
    return e;
  } else {
    Expression e;
    e.opcode = 0;
    e.value = V;
    e.lhs = 0;
    e.rhs = 0;
    
    return e;
  }
}

GVNPRE::Expression GVNPRE::add(ValueTable& VN, std::set<Expression>& MS,
                               Instruction* V) {
  Expression e = buildExpression(VN, V);
  if (VN.insert(std::make_pair(e, nextValueNumber)).second)
    nextValueNumber++;
  if (e.opcode != 0 || (e.opcode == 0 && isa<PHINode>(e.value)))
    MS.insert(e);
  return e;
}

GVNPRE::ValueTable::iterator GVNPRE::lookup(ValueTable& VN, Value* V) {
  Expression e = buildExpression(VN, V);
  return VN.find(e);
}

std::set<GVNPRE::Expression>::iterator GVNPRE::find_leader(GVNPRE::ValueTable VN,
                           std::set<GVNPRE::Expression>& vals,
                           uint32_t v) {
  for (std::set<Expression>::iterator I = vals.begin(), E = vals.end();
       I != E; ++I)
    if (VN[*I] == v)
      return I;
  
  return vals.end();
}

void GVNPRE::phi_translate(GVNPRE::ValueTable& VN, 
                           std::set<GVNPRE::Expression>& anticIn, BasicBlock* B,
                           std::set<GVNPRE::Expression>& out) {
  BasicBlock* succ = B->getTerminator()->getSuccessor(0);
  
  for (std::set<Expression>::iterator I = anticIn.begin(), E = anticIn.end();
       I != E; ++I) {
    if (I->opcode == 0) {
      Value *v = I->value;
      if (PHINode* p = dyn_cast<PHINode>(v))
        if (p->getParent() == succ) {
          out.insert(buildExpression(VN, p->getIncomingValueForBlock(B)));
          continue;
        }
    }
    //out.insert(*I);
  }
}

// Remove all expressions whose operands are not themselves in the set
void GVNPRE::clean(GVNPRE::ValueTable VN, std::set<GVNPRE::Expression>& set) {
  unsigned size = set.size();
  unsigned old = 0;
  
  while (size != old) {
    old = size;
  
    std::vector<Expression> worklist(set.begin(), set.end());
    while (!worklist.empty()) {
      Expression e = worklist.back();
      worklist.pop_back();
    
      if (e.opcode == 0) // OPAQUE
        continue;
      
      bool lhsValid = false;
      for (std::set<Expression>::iterator I = set.begin(), E = set.end();
           I != E; ++I)
        if (VN[*I] == e.lhs);
          lhsValid = true;
          
      bool rhsValid = false;
      for (std::set<Expression>::iterator I = set.begin(), E = set.end();
           I != E; ++I)
        if (VN[*I] == e.rhs);
          rhsValid = true;
      
      if (!lhsValid || !rhsValid)
        set.erase(e);
    }
    
    size = set.size();
  }
}

void GVNPRE::dump(GVNPRE::ValueTable& VN, std::set<GVNPRE::Expression>& s) {
  DOUT << "{ ";
  for (std::set<Expression>::iterator I = s.begin(), E = s.end(); I != E; ++I) {
    DOUT << "( " << I->opcode << ", "
         << (I->value == 0 ? "0" : I->value->getName().c_str())
         << ", value." << I->lhs << ", value." << I->rhs << " ) ";
  }
  DOUT << "}\n\n";
}

void GVNPRE::CalculateAvailOut(GVNPRE::ValueTable& VN, std::set<Expression>& MS,
                       DominatorTree::Node* DI,
                       std::set<Expression>& currExps,
                       std::set<PHINode*>& currPhis,
                       std::set<Expression>& currTemps,
                       std::set<Expression>& currAvail,
                       std::map<BasicBlock*, std::set<Expression> > availOut) {
  
  BasicBlock* BB = DI->getBlock();
  
  // A block inherits AVAIL_OUT from its dominator
  if (DI->getIDom() != 0)
  currAvail.insert(availOut[DI->getIDom()->getBlock()].begin(),
                   availOut[DI->getIDom()->getBlock()].end());
    
    
 for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
      BI != BE; ++BI) {
       
    // Handle PHI nodes...
    if (PHINode* p = dyn_cast<PHINode>(BI)) {
      add(VN, MS, p);
      currPhis.insert(p);
    
    // Handle binary ops...
    } else if (BinaryOperator* BO = dyn_cast<BinaryOperator>(BI)) {
      Expression leftValue = buildExpression(VN, BO->getOperand(0));
      Expression rightValue = buildExpression(VN, BO->getOperand(1));
      
      Expression e = add(VN, MS, BO);
      
      currExps.insert(leftValue);
      currExps.insert(rightValue);
      currExps.insert(e);
      
      currTemps.insert(e);
        
    // Handle unsupported ops
    } else {
      Expression e = add(VN, MS, BI);
      currTemps.insert(e);
    }
      
    currAvail.insert(buildExpression(VN, BI));
  }
}

bool GVNPRE::runOnFunction(Function &F) {
  ValueTable VN;
  std::set<Expression> maximalSet;

  std::map<BasicBlock*, std::set<Expression> > generatedExpressions;
  std::map<BasicBlock*, std::set<PHINode*> > generatedPhis;
  std::map<BasicBlock*, std::set<Expression> > generatedTemporaries;
  std::map<BasicBlock*, std::set<Expression> > availableOut;
  std::map<BasicBlock*, std::set<Expression> > anticipatedIn;
  
  DominatorTree &DT = getAnalysis<DominatorTree>();   
  
  // First Phase of BuildSets - calculate AVAIL_OUT
  
  // Top-down walk of the dominator tree
  for (df_iterator<DominatorTree::Node*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    
    // Get the sets to update for this block
    std::set<Expression>& currExps = generatedExpressions[DI->getBlock()];
    std::set<PHINode*>& currPhis = generatedPhis[DI->getBlock()];
    std::set<Expression>& currTemps = generatedTemporaries[DI->getBlock()];
    std::set<Expression>& currAvail = availableOut[DI->getBlock()];     
    
    CalculateAvailOut(VN, maximalSet, *DI, currExps, currPhis,
                      currTemps, currAvail, availableOut);
  }
  
  PostDominatorTree &PDT = getAnalysis<PostDominatorTree>();
  
  // Second Phase of BuildSets - calculate ANTIC_IN
  
  std::set<BasicBlock*> visited;
  
  bool changed = true;
  unsigned iterations = 0;
  while (changed) {
    changed = false;
    std::set<Expression> anticOut;
    
    // Top-down walk of the postdominator tree
    for (df_iterator<PostDominatorTree::Node*> PDI = 
         df_begin(PDT.getRootNode()), E = df_end(DT.getRootNode());
         PDI != E; ++PDI) {
      BasicBlock* BB = PDI->getBlock();
      
      visited.insert(BB);
      
      std::set<Expression>& anticIn = anticipatedIn[BB];
      std::set<Expression> old (anticIn.begin(), anticIn.end());
      
      if (BB->getTerminator()->getNumSuccessors() == 1) {
         phi_translate(VN, maximalSet, BB, anticOut);
      } else if (BB->getTerminator()->getNumSuccessors() > 1) {
        for (unsigned i = 0; i < BB->getTerminator()->getNumSuccessors(); ++i) {
          BasicBlock* currSucc = BB->getTerminator()->getSuccessor(i);
          std::set<Expression> temp;
          if (visited.find(currSucc) == visited.end())
            temp.insert(maximalSet.begin(), maximalSet.end());
          else
            temp.insert(anticIn.begin(), anticIn.end());
       
          anticIn.clear();
          std::insert_iterator<std::set<Expression> >  ai_ins(anticIn,
                                                       anticIn.begin());
                                                       
          std::set_difference(anticipatedIn[currSucc].begin(),
                              anticipatedIn[currSucc].end(),
                              temp.begin(),
                              temp.end(),
                              ai_ins);
        }
      }
      
      std::set<Expression> S;
      std::insert_iterator<std::set<Expression> >  s_ins(S, S.begin());
      std::set_union(anticOut.begin(), anticOut.end(),
                     generatedExpressions[BB].begin(),
                     generatedExpressions[BB].end(),
                     s_ins);
      
      anticIn.clear();
      std::insert_iterator<std::set<Expression> >  antic_ins(anticIn, 
                                                             anticIn.begin());
      std::set_difference(S.begin(), S.end(),
                          generatedTemporaries[BB].begin(),
                          generatedTemporaries[BB].end(),
                          antic_ins);
      
      clean(VN, anticIn);
      
      
      
      if (old != anticIn)
        changed = true;
      
      anticOut.clear();
    }
    iterations++;
  }
  
  DOUT << "Iterations: " << iterations << "\n";
  
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    DOUT << "Name: " << I->getName().c_str() << "\n";
    
    DOUT << "TMP_GEN: ";
    dump(VN, generatedTemporaries[I]);
    DOUT << "\n";
    
    DOUT << "EXP_GEN: ";
    dump(VN, generatedExpressions[I]);
    DOUT << "\n";
    
    DOUT << "ANTIC_IN: ";
    dump(VN, anticipatedIn[I]);
    DOUT << "\n";
  }
  
  return false;
}
