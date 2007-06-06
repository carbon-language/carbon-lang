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
    Value* find_leader(std::set<Value*, ExprLT>& vals, Value* v);
    Value* phi_translate(ValueTable& VN, std::set<Value*, ExprLT>& MS,
                         std::set<Value*, ExprLT>& set,
                         Value* V, BasicBlock* pred);
    void phi_translate_set(ValueTable& VN, std::set<Value*, ExprLT>& MS,
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

Value* GVNPRE::find_leader(std::set<Value*, ExprLT>& vals,
                           Value* v) {
  ExprLT cmp;
  for (std::set<Value*, ExprLT>::iterator I = vals.begin(), E = vals.end();
       I != E; ++I)
    if (!cmp(v, *I) && !cmp(*I, v))
      return *I;
  
  return 0;
}

Value* GVNPRE::phi_translate(ValueTable& VN, std::set<Value*, ExprLT>& MS,
                             std::set<Value*, ExprLT>& set,
                             Value* V, BasicBlock* pred) {
  if (V == 0)
    return 0;
  
  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V)) {
    Value* newOp1 = isa<Instruction>(BO->getOperand(0))
                                ? phi_translate(VN, MS, set,
                                  find_leader(set, BO->getOperand(0)),
                                  pred)
                                : BO->getOperand(0);
    if (newOp1 == 0)
      return 0;
    
    Value* newOp2 = isa<Instruction>(BO->getOperand(1))
                                ? phi_translate(VN, MS, set,
                                  find_leader(set, BO->getOperand(1)),
                                  pred)
                                : BO->getOperand(1);
    if (newOp2 == 0)
      return 0;
    
    if (newOp1 != BO->getOperand(0) || newOp2 != BO->getOperand(1)) {
      Value* newVal = BinaryOperator::create(BO->getOpcode(),
                                             newOp1, newOp2,
                                             BO->getName()+".gvnpre");
      
      if (!find_leader(set, newVal)) {
        add(VN, MS, newVal);
        return newVal;
      } else {
        delete newVal;
        return 0;
      }
    }
  } else if (PHINode* P = dyn_cast<PHINode>(V)) {
    if (P->getParent() == pred->getTerminator()->getSuccessor(0))
      return P->getIncomingValueForBlock(pred);
  }
  
  return V;
}

void GVNPRE::phi_translate_set(GVNPRE::ValueTable& VN,
                           std::set<Value*, ExprLT>& MS,
                           std::set<Value*, ExprLT>& anticIn, BasicBlock* B,
                           std::set<Value*, ExprLT>& out) {
  for (std::set<Value*, ExprLT>::iterator I = anticIn.begin(),
       E = anticIn.end(); I != E; ++I) {
    Value* V = phi_translate(VN, MS, anticIn, *I, B);
    if (V != 0)
      out.insert(V);
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
                     q_ins);
  
  std::set<Value*> visited;
  while (!Q.empty()) {
    Value* e = Q.back();
  
    if (BinaryOperator* BO = dyn_cast<BinaryOperator>(e)) {
      Value* l = find_leader(set, BO->getOperand(0));
      Value* r = find_leader(set, BO->getOperand(1));
      
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
      
      if (isa<Instruction>(leftValue))
        currExps.insert(leftValue);
      if (isa<Instruction>(rightValue))
        currExps.insert(rightValue);
      currExps.insert(BO);
      
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
  
  // Phase 1: BuildSets
  
  // Phase 1, Part 1: calculate AVAIL_OUT
  
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
  
  DOUT << "Maximal Set: ";
  dump_unique(VN, maximalSet);
  DOUT << "\n";
  
  PostDominatorTree &PDT = getAnalysis<PostDominatorTree>();
  
  // Phase 1, Part 2: calculate ANTIC_IN
  
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
      DOUT << "Block: " << BB->getName() << "\n";
      DOUT << "TMP_GEN: ";
      dump(VN, generatedTemporaries[BB]);
      DOUT << "\n";
    
      DOUT << "EXP_GEN: ";
      dump_unique(VN, generatedExpressions[BB]);
      visited.insert(BB);
      
      std::set<Value*, ExprLT>& anticIn = anticipatedIn[BB];
      std::set<Value*, ExprLT> old (anticIn.begin(), anticIn.end());
      
      if (BB->getTerminator()->getNumSuccessors() == 1) {
         if (visited.find(BB->getTerminator()->getSuccessor(0)) == 
             visited.end())
           phi_translate_set(VN, maximalSet, maximalSet, BB, anticOut);
         else
           phi_translate_set(VN, maximalSet, 
             anticipatedIn[BB->getTerminator()->getSuccessor(0)], BB, anticOut);
      } else if (BB->getTerminator()->getNumSuccessors() > 1) {
        BasicBlock* first = BB->getTerminator()->getSuccessor(0);
        anticOut.insert(anticipatedIn[first].begin(),
                        anticipatedIn[first].end());
        for (unsigned i = 1; i < BB->getTerminator()->getNumSuccessors(); ++i) {
          BasicBlock* currSucc = BB->getTerminator()->getSuccessor(i);
          std::set<Value*, ExprLT>& succAnticIn = anticipatedIn[currSucc];
          
          std::set<Value*, ExprLT> temp;
          std::insert_iterator<std::set<Value*, ExprLT> >  temp_ins(temp, 
                                                                  temp.begin());
          std::set_intersection(anticOut.begin(), anticOut.end(),
                                succAnticIn.begin(), succAnticIn.end(),
                                temp_ins, ExprLT());
          
          anticOut.clear();
          anticOut.insert(temp.begin(), temp.end());
        }
      }
      
      DOUT << "ANTIC_OUT: ";
      dump_unique(VN, anticOut);
      DOUT << "\n";
      
      std::set<Value*, ExprLT> S;
      std::insert_iterator<std::set<Value*, ExprLT> >  s_ins(S, S.begin());
      std::set_union(anticOut.begin(), anticOut.end(),
                     generatedExpressions[BB].begin(),
                     generatedExpressions[BB].end(),
                     s_ins, ExprLT());
      
      anticIn.clear();
      
      for (std::set<Value*, ExprLT>::iterator I = S.begin(), E = S.end();
           I != E; ++I) {
        if (generatedTemporaries[BB].find(*I) == generatedTemporaries[BB].end())
          anticIn.insert(*I);
      }
      
      clean(VN, anticIn);
      
      DOUT << "ANTIC_IN: ";
      dump_unique(VN, anticIn);
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
  
  
  // Phase 2: Insert
  // FIXME: Not implemented yet
  
  // Phase 3: Eliminate
  for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    BasicBlock* BB = DI->getBlock();
    
    std::vector<Instruction*> erase;
    
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE; ++BI) {
      Value* leader = find_leader(availableOut[BB], BI);
      if (leader != 0)
        if (Instruction* Instr = dyn_cast<Instruction>(leader))
          if (Instr->getParent() != 0 && Instr != BI) {
            BI->replaceAllUsesWith(leader);
            erase.push_back(BI);
          }
    }
    
    for (std::vector<Instruction*>::iterator I = erase.begin(), E = erase.end();
         I != E; ++I)
      (*I)->eraseFromParent();
  }
  
  return false;
}
