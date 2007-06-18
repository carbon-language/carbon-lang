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

struct ExprLT {
  bool operator()(Value* left, Value* right) {
    if (BinaryOperator* leftBO = dyn_cast<BinaryOperator>(left)) {
      if (BinaryOperator* rightBO = dyn_cast<BinaryOperator>(right))
        return cmpBinaryOperator(leftBO, rightBO);
      else
        if (isa<CmpInst>(right)) {
          return false;
        } else {
          return true;
        }
    } else if (CmpInst* leftCmp = dyn_cast<CmpInst>(left)) {
      if (CmpInst* rightCmp = dyn_cast<CmpInst>(right))
        return cmpComparison(leftCmp, rightCmp);
      else
        return true;
    } else {
      if (isa<BinaryOperator>(right) || isa<CmpInst>(right))
        return false;
      else
        return left < right;
    }
  }
  
  bool cmpBinaryOperator(BinaryOperator* left, BinaryOperator* right) {
    if (left->getOpcode() != right->getOpcode())
      return left->getOpcode() < right->getOpcode();
    else if ((*this)(left->getOperand(0), right->getOperand(0)))
      return true;
    else if ((*this)(right->getOperand(0), left->getOperand(0)))
      return false;
    else
      return (*this)(left->getOperand(1), right->getOperand(1));
  }
  
  bool cmpComparison(CmpInst* left, CmpInst* right) {
    if (left->getOpcode() != right->getOpcode())
      return left->getOpcode() < right->getOpcode();
    else if (left->getPredicate() != right->getPredicate())
      return left->getPredicate() < right->getPredicate();
    else if ((*this)(left->getOperand(0), right->getOperand(0)))
      return true;
    else if ((*this)(right->getOperand(0), left->getOperand(0)))
      return false;
    else
      return (*this)(left->getOperand(1), right->getOperand(1));
  }
};

namespace {

  class VISIBILITY_HIDDEN GVNPRE : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    GVNPRE() : FunctionPass((intptr_t)&ID) { nextValueNumber = 1; }

  private:
    uint32_t nextValueNumber;
    typedef std::map<Value*, uint32_t, ExprLT> ValueTable;
    ValueTable VN;
    std::set<Value*, ExprLT> MS;
    std::vector<Instruction*> createdExpressions;
    
    std::map<BasicBlock*, std::set<Value*, ExprLT> > availableOut;
    std::map<BasicBlock*, std::set<Value*, ExprLT> > anticipatedIn;
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<PostDominatorTree>();
    }
  
    // Helper fuctions
    // FIXME: eliminate or document these better
    void dump(const std::set<Value*>& s) const;
    void dump_unique(const std::set<Value*, ExprLT>& s) const;
    void clean(std::set<Value*, ExprLT>& set);
    bool add(Value* V, uint32_t number);
    Value* find_leader(std::set<Value*, ExprLT>& vals,
                       Value* v);
    Value* phi_translate(Value* V, BasicBlock* pred, BasicBlock* succ);
    void phi_translate_set(std::set<Value*, ExprLT>& anticIn, BasicBlock* pred,
                           BasicBlock* succ, std::set<Value*, ExprLT>& out);
    
    void topo_sort(std::set<Value*, ExprLT>& set,
                   std::vector<Value*>& vec);
    
    // For a given block, calculate the generated expressions, temporaries,
    // and the AVAIL_OUT set
    void CalculateAvailOut(DomTreeNode* DI,
                       std::set<Value*, ExprLT>& currExps,
                       std::set<PHINode*>& currPhis,
                       std::set<Value*>& currTemps,
                       std::set<Value*, ExprLT>& currAvail,
                       std::map<BasicBlock*, std::set<Value*, ExprLT> > availOut);
    void cleanup();
    void elimination();
  
  };
  
  char GVNPRE::ID = 0;
  
}

FunctionPass *llvm::createGVNPREPass() { return new GVNPRE(); }

RegisterPass<GVNPRE> X("gvnpre",
                       "Global Value Numbering/Partial Redundancy Elimination");


STATISTIC(NumInsertedVals, "Number of values inserted");
STATISTIC(NumInsertedPhis, "Number of PHI nodes inserted");
STATISTIC(NumEliminated, "Number of redundant instructions eliminated");


bool GVNPRE::add(Value* V, uint32_t number) {
  std::pair<ValueTable::iterator, bool> ret = VN.insert(std::make_pair(V, number));
  if (isa<BinaryOperator>(V) || isa<PHINode>(V) || isa<CmpInst>(V))
    MS.insert(V);
  return ret.second;
}

Value* GVNPRE::find_leader(std::set<Value*, ExprLT>& vals, Value* v) {
  if (!isa<Instruction>(v))
    return v;
  
  for (std::set<Value*, ExprLT>::iterator I = vals.begin(), E = vals.end();
       I != E; ++I) {
    assert(VN.find(v) != VN.end() && "Value not numbered?");
    assert(VN.find(*I) != VN.end() && "Value not numbered?");
    if (VN[v] == VN[*I])
      return *I;
  }
  
  return 0;
}

Value* GVNPRE::phi_translate(Value* V, BasicBlock* pred, BasicBlock* succ) {
  if (V == 0)
    return 0;
  
  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V)) {
    Value* newOp1 = isa<Instruction>(BO->getOperand(0))
                                ? phi_translate(
                                    find_leader(anticipatedIn[succ], BO->getOperand(0)),
                                    pred, succ)
                                : BO->getOperand(0);
    if (newOp1 == 0)
      return 0;
    
    Value* newOp2 = isa<Instruction>(BO->getOperand(1))
                                ? phi_translate(
                                    find_leader(anticipatedIn[succ], BO->getOperand(1)),
                                    pred, succ)
                                : BO->getOperand(1);
    if (newOp2 == 0)
      return 0;
    
    if (newOp1 != BO->getOperand(0) || newOp2 != BO->getOperand(1)) {
      Instruction* newVal = BinaryOperator::create(BO->getOpcode(),
                                             newOp1, newOp2,
                                             BO->getName()+".gvnpre");
      
      if (add(newVal, nextValueNumber))
        nextValueNumber++;
      
      Value* leader = find_leader(availableOut[pred], newVal);
      if (leader == 0) {
        DOUT << "Creating value: " << std::hex << newVal << std::dec << "\n";
        createdExpressions.push_back(newVal);
        return newVal;
      } else {
        ValueTable::iterator I = VN.find(newVal);
        if (I->first == newVal)
          VN.erase(newVal);
        
        std::set<Value*, ExprLT>::iterator F = MS.find(newVal);
        if (*F == newVal)
          MS.erase(newVal);
        
        delete newVal;
        return leader;
      }
    }
  } else if (PHINode* P = dyn_cast<PHINode>(V)) {
    if (P->getParent() == succ)
      return P->getIncomingValueForBlock(pred);
  } else if (CmpInst* C = dyn_cast<CmpInst>(V)) {
    Value* newOp1 = isa<Instruction>(C->getOperand(0))
                                ? phi_translate(
                                    find_leader(anticipatedIn[succ], C->getOperand(0)),
                                    pred, succ)
                                : C->getOperand(0);
    if (newOp1 == 0)
      return 0;
    
    Value* newOp2 = isa<Instruction>(C->getOperand(1))
                                ? phi_translate(
                                    find_leader(anticipatedIn[succ], C->getOperand(1)),
                                    pred, succ)
                                : C->getOperand(1);
    if (newOp2 == 0)
      return 0;
    
    if (newOp1 != C->getOperand(0) || newOp2 != C->getOperand(1)) {
      Instruction* newVal = CmpInst::create(C->getOpcode(),
                                            C->getPredicate(),
                                             newOp1, newOp2,
                                             C->getName()+".gvnpre");
      
      if (add(newVal, nextValueNumber))
        nextValueNumber++;
        
      Value* leader = find_leader(availableOut[pred], newVal);
      if (leader == 0) {
        DOUT << "Creating value: " << std::hex << newVal << std::dec << "\n";
        createdExpressions.push_back(newVal);
        return newVal;
      } else {
        ValueTable::iterator I = VN.find(newVal);
        if (I->first == newVal)
          VN.erase(newVal);
        
        std::set<Value*, ExprLT>::iterator F = MS.find(newVal);
        if (*F == newVal)
          MS.erase(newVal);
        
        delete newVal;
        return leader;
      }
    }
  }
  
  return V;
}

void GVNPRE::phi_translate_set(std::set<Value*, ExprLT>& anticIn,
                              BasicBlock* pred, BasicBlock* succ,
                              std::set<Value*, ExprLT>& out) {
  for (std::set<Value*, ExprLT>::iterator I = anticIn.begin(),
       E = anticIn.end(); I != E; ++I) {
    Value* V = phi_translate(*I, pred, succ);
    if (V != 0)
      out.insert(V);
  }
}

bool dependsOnInvoke(Value* V) {
  if (!isa<User>(V))
    return false;
    
  User* U = cast<User>(V);
  std::vector<Value*> worklist(U->op_begin(), U->op_end());
  std::set<Value*> visited;
  
  while (!worklist.empty()) {
    Value* current = worklist.back();
    worklist.pop_back();
    visited.insert(current);
    
    if (!isa<User>(current))
      continue;
    else if (isa<InvokeInst>(current))
      return true;
    
    User* curr = cast<User>(current);
    for (unsigned i = 0; i < curr->getNumOperands(); ++i)
      if (visited.find(curr->getOperand(i)) == visited.end())
        worklist.push_back(curr->getOperand(i));
  }
  
  return false;
}

// Remove all expressions whose operands are not themselves in the set
void GVNPRE::clean(std::set<Value*, ExprLT>& set) {
  std::vector<Value*> worklist;
  topo_sort(set, worklist);
  
  for (unsigned i = 0; i < worklist.size(); ++i) {
    Value* v = worklist[i];
    
    if (BinaryOperator* BO = dyn_cast<BinaryOperator>(v)) {   
      bool lhsValid = !isa<Instruction>(BO->getOperand(0));
      if (!lhsValid)
        for (std::set<Value*, ExprLT>::iterator I = set.begin(), E = set.end();
             I != E; ++I)
          if (VN[*I] == VN[BO->getOperand(0)]) {
            lhsValid = true;
            break;
          }
          
      // Check for dependency on invoke insts
      // NOTE: This check is expensive, so don't do it if we
      // don't have to
      if (lhsValid)
        lhsValid = !dependsOnInvoke(BO->getOperand(0));
    
      bool rhsValid = !isa<Instruction>(BO->getOperand(1));
      if (!rhsValid)
      for (std::set<Value*, ExprLT>::iterator I = set.begin(), E = set.end();
           I != E; ++I)
        if (VN[*I] == VN[BO->getOperand(1)]) {
          rhsValid = true;
          break;
        }
      
      // Check for dependency on invoke insts
      // NOTE: This check is expensive, so don't do it if we
      // don't have to
      if (rhsValid)
        rhsValid = !dependsOnInvoke(BO->getOperand(1));
      
      if (!lhsValid || !rhsValid)
        set.erase(BO);
    } else if (CmpInst* C = dyn_cast<CmpInst>(v)) {
      bool lhsValid = !isa<Instruction>(C->getOperand(0));
      if (!lhsValid)
        for (std::set<Value*, ExprLT>::iterator I = set.begin(), E = set.end();
             I != E; ++I)
          if (VN[*I] == VN[C->getOperand(0)]) {
            lhsValid = true;
            break;
          }
      lhsValid &= !dependsOnInvoke(C->getOperand(0));
      
      bool rhsValid = !isa<Instruction>(C->getOperand(1));
      if (!rhsValid)
      for (std::set<Value*, ExprLT>::iterator I = set.begin(), E = set.end();
           I != E; ++I)
        if (VN[*I] == VN[C->getOperand(1)]) {
          rhsValid = true;
          break;
        }
      rhsValid &= !dependsOnInvoke(C->getOperand(1));
    
      if (!lhsValid || !rhsValid)
        set.erase(C);
    }
  }
}

void GVNPRE::topo_sort(std::set<Value*, ExprLT>& set,
                       std::vector<Value*>& vec) {
  std::set<Value*, ExprLT> toErase;
  for (std::set<Value*, ExprLT>::iterator I = set.begin(), E = set.end();
       I != E; ++I) {
    if (BinaryOperator* BO = dyn_cast<BinaryOperator>(*I))
      for (std::set<Value*, ExprLT>::iterator SI = set.begin(); SI != E; ++SI) {
        if (VN[BO->getOperand(0)] == VN[*SI] ||
            VN[BO->getOperand(1)] == VN[*SI]) {
          toErase.insert(*SI);
        }
      }
    else if (CmpInst* C = dyn_cast<CmpInst>(*I))
      for (std::set<Value*, ExprLT>::iterator SI = set.begin(); SI != E; ++SI) {
        if (VN[C->getOperand(0)] == VN[*SI] ||
            VN[C->getOperand(1)] == VN[*SI]) {
          toErase.insert(*SI);
        }
      }
  }
  
  std::vector<Value*> Q;
  for (std::set<Value*, ExprLT>::iterator I = set.begin(), E = set.end();
       I != E; ++I) {
    if (toErase.find(*I) == toErase.end())
      Q.push_back(*I);
  }
  
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
    } else if (CmpInst* C = dyn_cast<CmpInst>(e)) {
      Value* l = find_leader(set, C->getOperand(0));
      Value* r = find_leader(set, C->getOperand(1));
      
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

void GVNPRE::dump_unique(const std::set<Value*, ExprLT>& s) const {
  DOUT << "{ ";
  for (std::set<Value*>::iterator I = s.begin(), E = s.end();
       I != E; ++I) {
    DEBUG((*I)->dump());
  }
  DOUT << "}\n\n";
}

void GVNPRE::CalculateAvailOut(DomTreeNode* DI,
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
      if (add(p, nextValueNumber))
        nextValueNumber++;
      currPhis.insert(p);
    
    // Handle binary ops...
    } else if (BinaryOperator* BO = dyn_cast<BinaryOperator>(BI)) {
      Value* leftValue = BO->getOperand(0);
      Value* rightValue = BO->getOperand(1);
      
      if (add(BO, nextValueNumber))
        nextValueNumber++;
      
      if (isa<Instruction>(leftValue))
        currExps.insert(leftValue);
      if (isa<Instruction>(rightValue))
        currExps.insert(rightValue);
      currExps.insert(BO);
      
    // Handle cmp ops...
    } else if (CmpInst* C = dyn_cast<CmpInst>(BI)) {
      Value* leftValue = C->getOperand(0);
      Value* rightValue = C->getOperand(1);
      
      if (add(C, nextValueNumber))
        nextValueNumber++;
      
      if (isa<Instruction>(leftValue))
        currExps.insert(leftValue);
      if (isa<Instruction>(rightValue))
        currExps.insert(rightValue);
      currExps.insert(C);
      
    // Handle unsupported ops
    } else if (!BI->isTerminator()){
      if (add(BI, nextValueNumber))
        nextValueNumber++;
      currTemps.insert(BI);
    }
    
    if (!BI->isTerminator())
      currAvail.insert(BI);
  }
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
         Value *leader = find_leader(availableOut[BB], BI);
  
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
  MS.clear();
  createdExpressions.clear();
  availableOut.clear();
  anticipatedIn.clear();

  std::map<BasicBlock*, std::set<Value*, ExprLT> > generatedExpressions;
  std::map<BasicBlock*, std::set<PHINode*> > generatedPhis;
  std::map<BasicBlock*, std::set<Value*> > generatedTemporaries;
  
  
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
    
    CalculateAvailOut(*DI, currExps, currPhis,
                      currTemps, currAvail, availableOut);
  }
  
  DOUT << "Maximal Set: ";
  dump_unique(MS);
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
    std::set<Value*, ExprLT> anticOut;
    
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
      
      std::set<Value*, ExprLT>& anticIn = anticipatedIn[BB];
      std::set<Value*, ExprLT> old (anticIn.begin(), anticIn.end());
      
      if (BB->getTerminator()->getNumSuccessors() == 1) {
         if (visited.find(BB->getTerminator()->getSuccessor(0)) == 
             visited.end())
           phi_translate_set(MS, BB, BB->getTerminator()->getSuccessor(0),
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
      dump_unique(anticOut);
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
  
  std::map<BasicBlock*, std::set<Value*, ExprLT> > new_sets;
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
      
      std::set<Value*, ExprLT>& new_set = new_sets[BB];
      std::set<Value*, ExprLT>& availOut = availableOut[BB];
      std::set<Value*, ExprLT>& anticIn = anticipatedIn[BB];
      
      new_set.clear();
      
      // Replace leaders with leaders inherited from dominator
      if (DI->getIDom() != 0) {
        std::set<Value*, ExprLT>& dom_set = new_sets[DI->getIDom()->getBlock()];
        for (std::set<Value*, ExprLT>::iterator I = dom_set.begin(),
             E = dom_set.end(); I != E; ++I) {
          new_set.insert(*I);
          
          Value* val = find_leader(availOut, *I);
          while (val != 0) {
            availOut.erase(val);
            val = find_leader(availOut, *I);
          }
          availOut.insert(*I);
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
            if (find_leader(availableOut[DI->getIDom()->getBlock()], e) != 0)
              continue;
            
            std::map<BasicBlock*, Value*> avail;
            bool by_some = false;
            int num_avail = 0;
            
            for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE;
                 ++PI) {
              Value *e2 = phi_translate(e, *PI, BB);
              Value *e3 = find_leader(availableOut[*PI], e2);
              
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
                if (!find_leader(availableOut[*PI], e2)) {
                  User* U = cast<User>(e2);
                
                  Value* s1 = 0;
                  if (isa<BinaryOperator>(U->getOperand(0)) ||
                      isa<CmpInst>(U->getOperand(0)))
                    s1 = find_leader(availableOut[*PI], U->getOperand(0));
                  else
                    s1 = U->getOperand(0);
                  
                  Value* s2 = 0;
                  if (isa<BinaryOperator>(U->getOperand(1)) ||
                      isa<CmpInst>(U->getOperand(1)))
                    s2 = find_leader(availableOut[*PI], U->getOperand(1));
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
                  
                  add(newVal, VN[U]);
                  
                  std::set<Value*, ExprLT>& predAvail = availableOut[*PI];
                  Value* val = find_leader(predAvail, newVal);
                  while (val != 0) {
                    predAvail.erase(val);
                    val = find_leader(predAvail, newVal);
                  }
                  predAvail.insert(newVal);
                  
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
              
              add(p, VN[e]);
              DOUT << "Creating value: " << std::hex << p << std::dec << "\n";
              
              Value* val = find_leader(availOut, p);
              while (val != 0) {
                availOut.erase(val);
                val = find_leader(availOut, p);
              }
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
