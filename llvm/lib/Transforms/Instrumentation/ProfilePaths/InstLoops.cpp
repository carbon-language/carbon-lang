//===-- InstLoops.cpp ---------------------------------------- ---*- C++ -*--=//
// Pass to instrument loops
//
// At every backedge, insert a counter for that backedge and a call function
//===----------------------------------------------------------------------===//

#include "llvm/Reoptimizer/InstLoops.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Constants.h"
#include "llvm/iMemory.h"
#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iOther.h"
#include "llvm/iOperators.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"

//this is used to color vertices
//during DFS

enum Color{
  WHITE,
  GREY,
  BLACK
};

namespace{
  typedef std::map<BasicBlock *, BasicBlock *> BBMap;
  struct InstLoops : public FunctionPass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorSet>();
    }
  private:
    DominatorSet *DS;
    void getBackEdgesVisit(BasicBlock *u,
			   std::map<BasicBlock *, Color > &color,
			   std::map<BasicBlock *, int > &d, 
			   int &time, BBMap &be);
    void removeRedundant(BBMap &be);
    void findAndInstrumentBackEdges(Function &F);
  public:
    bool runOnFunction(Function &F);
  };
  
  RegisterOpt<InstLoops> X("instloops", "Instrument backedges for profiling");
}

// createInstLoopsPass - Create a new pass to add path profiling
//
Pass *createInstLoopsPass() {
  return new InstLoops();
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

  for(BasicBlock::succ_iterator vl = succ_begin(u), 
	ve = succ_end(u); vl != ve; ++vl){
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

  // FIXME: THIS IS HORRIBLY BROKEN.  FunctionPass's cannot do this, except in
  // their initialize function!!
  Function *inCountMth = 
    F.getParent()->getOrInsertFunction("llvm_first_trigger",
                                       Type::VoidTy, 0);

  for(std::map<BasicBlock *, BasicBlock *>::iterator MI = be.begin(),
	ME = be.end(); MI != ME; ++MI){
    BasicBlock *u = MI->first;
    BasicBlock *BB = MI->second;
    //std::cerr<<"Edge from: "<<BB->getName()<<"->"<<u->getName()<<"\n";
    //insert a new basic block: modify terminator accordingly!
    BasicBlock *newBB = new BasicBlock("", u->getParent());
    BranchInst *ti = cast<BranchInst>(u->getTerminator());
    unsigned char index = 1;
    if(ti->getSuccessor(0) == BB){
      index = 0;
    }
    assert(ti->getNumSuccessors() > index && "Not enough successors!");
    ti->setSuccessor(index, newBB);
        
    BasicBlock::InstListType &lt = newBB->getInstList();

    Instruction *call = new CallInst(inCountMth);
    lt.push_back(call);
    lt.push_back(new BranchInst(BB));
      
    //now iterate over *vl, and set its Phi nodes right
    for(BasicBlock::iterator BB2Inst = BB->begin(), BBend = BB->end(); 
	BB2Inst != BBend; ++BB2Inst){
        
      if (PHINode *phiInst = dyn_cast<PHINode>(BB2Inst)){
	int bbIndex = phiInst->getBasicBlockIndex(u);
	if(bbIndex>=0){
	  phiInst->setIncomingBlock(bbIndex, newBB);
	}
      }
    }
  }
}

/// Entry point for FunctionPass that inserts calls to trigger function.
///
bool InstLoops::runOnFunction(Function &F){
  DS  = &getAnalysis<DominatorSet>();
  if(F.isExternal()) {
    return false;
  }
  // Add a call to reoptimizerInitialize() to beginning of function named main.
  if(F.getName() == "main"){
    std::vector<const Type*> argTypes;  // Empty formal parameter list.
    const FunctionType *Fty = FunctionType::get(Type::VoidTy, argTypes, false);
    Function *initialMeth =
      F.getParent()->getOrInsertFunction("reoptimizerInitialize", Fty);
    assert(initialMeth && "Initialize method could not be inserted!");
    new CallInst(initialMeth, "", F.begin()->begin());  // Insert it.
  }
  findAndInstrumentBackEdges(F);
  return true;  // Function was modified.
}
