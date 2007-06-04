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
#include <deque>
#include <map>
#include <vector>
#include <set>
using namespace llvm;

struct ExprLT {
  bool operator()(Value* left, Value* right) {
    if (!isa<BinaryOperator>(left) || !isa<BinaryOperator>(right))
      return left < right;
    
    BinaryOperator* BO1 = cast<BinaryOperator>(left);
    BinaryOperator* BO2 = cast<BinaryOperator>(right);
    
    if ((*this)(BO1->getOperand(0), BO2->getOperand(0)))
      return true;
    else if ((*this)(BO2->getOperand(0), BO1->getOperand(0)))
      return false;
    else
      return (*this)(BO1->getOperand(1), BO2->getOperand(1));
  }
};

namespace {

  class VISIBILITY_HIDDEN GVNPRE : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    GVNPRE() : FunctionPass((intptr_t)&ID) { nextValueNumber = 0; }

  private:
    uint32_t nextValueNumber;
    typedef std::map<Value*, uint32_t, ExprLT> ValueTable;
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<PostDominatorTree>();
    }
  
    // Helper fuctions
    // FIXME: eliminate or document these better
    void dump(ValueTable& VN, std::set<Value*>& s);
    void dump_unique(ValueTable& VN, std::set<Value*, ExprLT>& s);
    void clean(ValueTable VN, std::set<Value*, ExprLT>& set);
    bool add(ValueTable& VN, std::set<Value*, ExprLT>& MS, Value* V);
    Value* find_leader(ValueTable VN, std::set<Value*, ExprLT>& vals, uint32_t v);
    void phi_translate(ValueTable& VN, std::set<Value*, ExprLT>& MS,
                       std::set<Value*, ExprLT>& anticIn, BasicBlock* B,
                       std::set<Value*, ExprLT>& out);
    
    void topo_sort(ValueTable& VN, std::set<Value*, ExprLT>& set,
                   std::vector<Value*>& vec);
    
    // For a given block, calculate the generated expressions, temporaries,
    // and the AVAIL_OUT set
    void CalculateAvailOut(ValueTable& VN, std::set<Value*, ExprLT>& MS,
                       DomTreeNode* DI,
                       std::set<Value*, ExprLT>& currExps,
                       std::set<PHINode*>& currPhis,
                       std::set<Value*>& currTemps,
                       std::set<Value*, ExprLT>& currAvail,
                       std::map<BasicBlock*, std::set<Value*, ExprLT> > availOut);
  
  };
  
  char GVNPRE::ID = 0;
  
}

FunctionPass *llvm::createGVNPREPass() { return new GVNPRE(); }

RegisterPass<GVNPRE> X("gvnpre",
                       "Global Value Numbering/Partial Redundancy Elimination");



bool GVNPRE::add(ValueTable& VN, std::set<Value*, ExprLT>& MS, Value* V) {
  std::pair<ValueTable::iterator, bool> ret = VN.insert(std::make_pair(V, nextValueNumber));
  if (ret.second)
    nextValueNumber++;
  if (isa<BinaryOperator>(V) || isa<PHINode>(V))
    MS.insert(V);
  return ret.second;
}

Value* GVNPRE::find_leader(GVNPRE::ValueTable VN,
                           std::set<Value*, ExprLT>& vals,
                           uint32_t v) {
  for (std::set<Value*, ExprLT>::iterator I = vals.begin(), E = vals.end();
       I != E; ++I)
    if (VN[*I] == v)
      return *I;
  
  return 0;
}

void GVNPRE::phi_translate(GVNPRE::ValueTable& VN,
                           std::set<Value*, ExprLT>& MS,
                           std::set<Value*, ExprLT>& anticIn, BasicBlock* B,
                           std::set<Value*, ExprLT>& out) {
  BasicBlock* succ = B->getTerminator()->getSuccessor(0);
  
  for (std::set<Value*, ExprLT>::iterator I = anticIn.begin(), E = anticIn.end();
       I != E; ++I) {
    if (!isa<BinaryOperator>(*I)) {
      if (PHINode* p = dyn_cast<PHINode>(*I)) {
        if (p->getParent() == succ)
          out.insert(p);
      } else {
        out.insert(*I);
      }
    } else {
      BinaryOperator* BO = cast<BinaryOperator>(*I);
      Value* lhs = find_leader(VN, anticIn, VN[BO->getOperand(0)]);
      if (lhs == 0)
        continue;
      
      if (PHINode* p = dyn_cast<PHINode>(lhs))
        if (p->getParent() == succ) {
          lhs = p->getIncomingValueForBlock(B);
          out.insert(lhs);
        }
      
      Value* rhs = find_leader(VN, anticIn, VN[BO->getOperand(1)]);
      if (rhs == 0)
        continue;
      
      if (PHINode* p = dyn_cast<PHINode>(rhs))
        if (p->getParent() == succ) {
          rhs = p->getIncomingValueForBlock(B);
          out.insert(rhs);
        }
      
      if (lhs != BO->getOperand(0) || rhs != BO->getOperand(1)) {
        BO = BinaryOperator::create(BO->getOpcode(), lhs, rhs, BO->getName()+".gvnpre");
        if (VN.insert(std::make_pair(BO, nextValueNumber)).second)
          nextValueNumber++;
        MS.insert(BO);
      }
      
      out.insert(BO);
      
    }
  }
}

// Remove all expressions whose operands are not themselves in the set
void GVNPRE::clean(GVNPRE::ValueTable VN, std::set<Value*, ExprLT>& set) {
  std::vector<Value*> worklist;
  topo_sort(VN, set, worklist);
  
  while (!worklist.empty()) {
    Value* v = worklist.back();
    worklist.pop_back();
    
    if (BinaryOperator* BO = dyn_cast<BinaryOperator>(v)) {   
      bool lhsValid = false;
      for (std::set<Value*, ExprLT>::iterator I = set.begin(), E = set.end();
           I != E; ++I)
        if (VN[*I] == VN[BO->getOperand(0)]);
          lhsValid = true;
    
      bool rhsValid = false;
      for (std::set<Value*, ExprLT>::iterator I = set.begin(), E = set.end();
           I != E; ++I)
        if (VN[*I] == VN[BO->getOperand(1)]);
          rhsValid = true;
      
      if (!lhsValid || !rhsValid)
        set.erase(BO);
    }
  }
}

void GVNPRE::topo_sort(GVNPRE::ValueTable& VN,
                       std::set<Value*, ExprLT>& set,
                       std::vector<Value*>& vec) {
  std::set<Value*, ExprLT> toErase;               
  for (std::set<Value*, ExprLT>::iterator I = set.begin(), E = set.end();
       I != E; ++I) {
    if (BinaryOperator* BO = dyn_cast<BinaryOperator>(*I))
      for (std::set<Value*, ExprLT>::iterator SI = set.begin(); SI != E; ++SI) {
        if (VN[BO->getOperand(0)] == VN[*SI] || VN[BO->getOperand(1)] == VN[*SI]) {
          toErase.insert(BO);
        }
    }
  }
  
  std::vector<Value*> Q;
  std::insert_iterator<std::vector<Value*> > q_ins(Q, Q.begin());
  std::set_difference(set.begin(), set.end(),
                     toErase.begin(), toErase.end(),
                     q_ins, ExprLT());
  
  std::set<Value*, ExprLT> visited;
  while (!Q.empty()) {
    Value* e = Q.back();
  
    if (BinaryOperator* BO = dyn_cast<BinaryOperator>(e)) {
      Value* l = find_leader(VN, set, VN[BO->getOperand(0)]);
      Value* r = find_leader(VN, set, VN[BO->getOperand(1)]);
      
      if (l != 0 && visited.find(l) == visited.end())
        Q.push_back(l);
      else if (r != 0 && visited.find(r) == visited.end())
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


void GVNPRE::dump(GVNPRE::ValueTable& VN, std::set<Value*>& s) {
  DOUT << "{ ";
  for (std::set<Value*>::iterator I = s.begin(), E = s.end();
       I != E; ++I) {
    DEBUG((*I)->dump());
  }
  DOUT << "}\n\n";
}

void GVNPRE::dump_unique(GVNPRE::ValueTable& VN, std::set<Value*, ExprLT>& s) {
  DOUT << "{ ";
  for (std::set<Value*>::iterator I = s.begin(), E = s.end();
       I != E; ++I) {
    DEBUG((*I)->dump());
  }
  DOUT << "}\n\n";
}

void GVNPRE::CalculateAvailOut(GVNPRE::ValueTable& VN, std::set<Value*, ExprLT>& MS,
                       DomTreeNode* DI,
                       std::set<Value*, ExprLT>& currExps,
                       std::set<PHINode*>& currPhis,
                       std::set<Value*>& currTemps,
                       std::set<Value*, ExprLT>& currAvail,
                       std::map<BasicBlock*, std::set<Value*, ExprLT> > availOut) {
  
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
      Value* leftValue = BO->getOperand(0);
      Value* rightValue = BO->getOperand(1);
      
      add(VN, MS, BO);
      
      currExps.insert(leftValue);
      currExps.insert(rightValue);
      currExps.insert(BO);
      
      currTemps.insert(BO);
      
    // Handle unsupported ops
    } else if (!BI->isTerminator()){
      add(VN, MS, BI);
      currTemps.insert(BI);
    }
    
    if (!BI->isTerminator())
      currAvail.insert(BI);
  }
}

bool GVNPRE::runOnFunction(Function &F) {
  ValueTable VN;
  std::set<Value*, ExprLT> maximalSet;

  std::map<BasicBlock*, std::set<Value*, ExprLT> > generatedExpressions;
  std::map<BasicBlock*, std::set<PHINode*> > generatedPhis;
  std::map<BasicBlock*, std::set<Value*> > generatedTemporaries;
  std::map<BasicBlock*, std::set<Value*, ExprLT> > availableOut;
  std::map<BasicBlock*, std::set<Value*, ExprLT> > anticipatedIn;
  
  DominatorTree &DT = getAnalysis<DominatorTree>();   
  
  // First Phase of BuildSets - calculate AVAIL_OUT
  
  // Top-down walk of the dominator tree
  for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    
    // Get the sets to update for this block
    std::set<Value*, ExprLT>& currExps = generatedExpressions[DI->getBlock()];
    std::set<PHINode*>& currPhis = generatedPhis[DI->getBlock()];
    std::set<Value*>& currTemps = generatedTemporaries[DI->getBlock()];
    std::set<Value*, ExprLT>& currAvail = availableOut[DI->getBlock()];     
    
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
    std::set<Value*, ExprLT> anticOut;
    
    // Top-down walk of the postdominator tree
    for (df_iterator<DomTreeNode*> PDI = 
         df_begin(PDT.getRootNode()), E = df_end(DT.getRootNode());
         PDI != E; ++PDI) {
      BasicBlock* BB = PDI->getBlock();
      
      visited.insert(BB);
      
      std::set<Value*, ExprLT>& anticIn = anticipatedIn[BB];
      std::set<Value*, ExprLT> old (anticIn.begin(), anticIn.end());
      
      if (BB->getTerminator()->getNumSuccessors() == 1) {
         if (visited.find(BB) == visited.end())
           phi_translate(VN, maximalSet, anticIn, BB, anticOut);
         else
           phi_translate(VN, anticIn, anticIn, BB, anticOut);
      } else if (BB->getTerminator()->getNumSuccessors() > 1) {
        for (unsigned i = 0; i < BB->getTerminator()->getNumSuccessors(); ++i) {
          BasicBlock* currSucc = BB->getTerminator()->getSuccessor(i);
          std::set<Value*, ExprLT> temp;
          if (visited.find(currSucc) == visited.end())
            temp.insert(maximalSet.begin(), maximalSet.end());
          else
            temp.insert(anticIn.begin(), anticIn.end());
       
          anticIn.clear();
          std::insert_iterator<std::set<Value*, ExprLT> >  ai_ins(anticIn,
                                                       anticIn.begin());
                                                       
          std::set_difference(anticipatedIn[currSucc].begin(),
                              anticipatedIn[currSucc].end(),
                              temp.begin(),
                              temp.end(),
                              ai_ins,
                              ExprLT());
        }
      }
      
      std::set<Value*, ExprLT> S;
      std::insert_iterator<std::set<Value*, ExprLT> >  s_ins(S, S.begin());
      std::set_union(anticOut.begin(), anticOut.end(),
                     generatedExpressions[BB].begin(),
                     generatedExpressions[BB].end(),
                     s_ins, ExprLT());
      
      anticIn.clear();
      std::insert_iterator<std::set<Value*, ExprLT> >  antic_ins(anticIn, 
                                                             anticIn.begin());
      std::set_difference(S.begin(), S.end(),
                          generatedTemporaries[BB].begin(),
                          generatedTemporaries[BB].end(),
                          antic_ins,
                          ExprLT());
      
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
    dump_unique(VN, generatedExpressions[I]);
    DOUT << "\n";
    
    DOUT << "ANTIC_IN: ";
    dump_unique(VN, anticipatedIn[I]);
    DOUT << "\n";
    
    DOUT << "AVAIL_OUT: ";
    dump_unique(VN, availableOut[I]);
    DOUT << "\n";
  }
  
  return false;
}
