//===-- InstLoops.cpp -----------------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is the first-level instrumentation pass for the Reoptimizer. It
// instrument the back-edges of loops by inserting a basic block
// containing a call to llvm_first_trigger (the first-level trigger function),
// and inserts an initialization call to the main() function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/iOther.h"
#include "llvm/Type.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "Support/Debug.h"
#include "../ProfilingUtils.h"

namespace llvm {

//this is used to color vertices
//during DFS

enum Color{
  WHITE,
  GREY,
  BLACK
};

namespace {
  typedef std::map<BasicBlock *, BasicBlock *> BBMap;
  struct InstLoops : public FunctionPass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorSet>();
    }
  private:
    Function *inCountMth;
    DominatorSet *DS;
    void getBackEdgesVisit(BasicBlock *u,
			   std::map<BasicBlock *, Color > &color,
			   std::map<BasicBlock *, int > &d, 
			   int &time, BBMap &be);
    void removeRedundant(BBMap &be);
    void findAndInstrumentBackEdges(Function &F);
  public:
    bool doInitialization(Module &M);
    bool runOnFunction(Function &F);
  };
  
  RegisterOpt<InstLoops> X("instloops", "Instrument backedges for profiling");
}

//helper function to get back edges: it is called by 
//the "getBackEdges" function below
void InstLoops::getBackEdgesVisit(BasicBlock *u,
                       std::map<BasicBlock *, Color > &color,
                       std::map<BasicBlock *, int > &d, 
                       int &time, BBMap &be) {
  color[u]=GREY;
  time++;
  d[u]=time;

  for(succ_iterator vl = succ_begin(u), ve = succ_end(u); vl != ve; ++vl){
    BasicBlock *BB = *vl;

    if(color[BB]!=GREY && color[BB]!=BLACK){
      getBackEdgesVisit(BB, color, d, time, be);
    }
    
    //now checking for d and f vals
    else if(color[BB]==GREY){
      //so v is ancestor of u if time of u > time of v
      if(d[u] >= d[BB]){
	//u->BB is a backedge
	be[u] = BB;
      }
    }
  }
  color[u]=BLACK;//done with visiting the node and its neighbors
}

//look at all BEs, and remove all BEs that are dominated by other BE's in the
//set
void InstLoops::removeRedundant(BBMap &be) {
  std::vector<BasicBlock *> toDelete;
  for(std::map<BasicBlock *, BasicBlock *>::iterator MI = be.begin(), 
	ME = be.end(); MI != ME; ++MI)
    for(BBMap::iterator MMI = be.begin(), MME = be.end(); MMI != MME; ++MMI)
      if(DS->properlyDominates(MI->first, MMI->first))
	toDelete.push_back(MMI->first);
  // Remove all the back-edges we found from be.
  for(std::vector<BasicBlock *>::iterator VI = toDelete.begin(), 
	VE = toDelete.end(); VI != VE; ++VI)
    be.erase(*VI);
}

//getting the backedges in a graph
//Its a variation of DFS to get the backedges in the graph
//We get back edges by associating a time
//and a color with each vertex.
//The time of a vertex is the time when it was first visited
//The color of a vertex is initially WHITE,
//Changes to GREY when it is first visited,
//and changes to BLACK when ALL its neighbors
//have been visited
//So we have a back edge when we meet a successor of
//a node with smaller time, and GREY color
void InstLoops::findAndInstrumentBackEdges(Function &F){
  std::map<BasicBlock *, Color > color;
  std::map<BasicBlock *, int> d;
  BBMap be;
  int time=0;
  getBackEdgesVisit(F.begin(), color, d, time, be);

  removeRedundant(be);

  for(std::map<BasicBlock *, BasicBlock *>::iterator MI = be.begin(),
	ME = be.end(); MI != ME; ++MI){
    BasicBlock *u = MI->first;
    BasicBlock *BB = MI->second;
    // We have a back-edge from BB --> u.
    DEBUG (std::cerr << "Instrumenting back-edge from " << BB->getName ()
                     << "-->" << u->getName () << "\n");
    // Split the back-edge, inserting a new basic block on it, and modify the
    // source BB's terminator accordingly.
    BasicBlock *newBB = new BasicBlock("backEdgeInst", u->getParent());
    BranchInst *ti = cast<BranchInst>(u->getTerminator());
    unsigned char index = ((ti->getSuccessor(0) == BB) ? 0 : 1);

    assert(ti->getNumSuccessors() > index && "Not enough successors!");
    ti->setSuccessor(index, newBB);
        
    BasicBlock::InstListType &lt = newBB->getInstList();
    lt.push_back(new CallInst(inCountMth));
    new BranchInst(BB, newBB);
      
    // Now, set the sources of Phi nodes corresponding to the back-edge
    // in BB to come from the instrumentation block instead.
    for(BasicBlock::iterator BB2Inst = BB->begin(), BBend = BB->end(); 
        BB2Inst != BBend; ++BB2Inst) {
      if (PHINode *phiInst = dyn_cast<PHINode>(BB2Inst)) {
        int bbIndex = phiInst->getBasicBlockIndex(u);
        if (bbIndex>=0)
          phiInst->setIncomingBlock(bbIndex, newBB);
      }
    }
  }
}

bool InstLoops::doInitialization (Module &M) {
  inCountMth = M.getOrInsertFunction("llvm_first_trigger", Type::VoidTy, 0);
  return true;  // Module was modified.
}

/// runOnFunction - Entry point for FunctionPass that inserts calls to
/// trigger function.
///
bool InstLoops::runOnFunction(Function &F){
  if (F.isExternal ())
    return false;

  DS = &getAnalysis<DominatorSet> ();

  // Add a call to reoptimizerInitialize() to beginning of function named main.
  if (F.getName() == "main")
    InsertProfilingInitCall (&F, "reoptimizerInitialize");

  findAndInstrumentBackEdges(F);
  return true;  // Function was modified.
}

} // End llvm namespace
