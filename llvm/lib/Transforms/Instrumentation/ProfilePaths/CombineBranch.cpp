//===-- CombineBranch.cpp -------------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Pass to instrument loops
//
// At every backedge, insert a counter for that backedge and a call function
//
//===----------------------------------------------------------------------===//

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

namespace llvm {

//this is used to color vertices
//during DFS

enum Color{
  WHITE,
  GREY,
  BLACK
};

namespace {
  struct CombineBranches : public FunctionPass {
  private:
    //DominatorSet *DS;
    void getBackEdgesVisit(BasicBlock *u,
			   std::map<BasicBlock *, Color > &color,
			   std::map<BasicBlock *, int > &d, 
			   int &time,
			   std::map<BasicBlock *, BasicBlock *> &be);
    void removeRedundant(std::map<BasicBlock *, BasicBlock *> &be);
    void getBackEdges(Function &F);
  public:
    bool runOnFunction(Function &F);
  };
  
  RegisterOpt<CombineBranches> X("branch-combine", "Multiple backedges going to same target are merged");
}

//helper function to get back edges: it is called by 
//the "getBackEdges" function below
void CombineBranches::getBackEdgesVisit(BasicBlock *u,
                       std::map<BasicBlock *, Color > &color,
                       std::map<BasicBlock *, int > &d, 
                       int &time,
		       std::map<BasicBlock *, BasicBlock *> &be) {
  
  color[u]=GREY;
  time++;
  d[u]=time;

  for (succ_iterator vl = succ_begin(u), ve = succ_end(u); vl != ve; ++vl){
    
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
void CombineBranches::removeRedundant(std::map<BasicBlock *, BasicBlock *> &be){
  std::vector<BasicBlock *> toDelete;
  std::map<BasicBlock *, int> seenBB;
  
  for(std::map<BasicBlock *, BasicBlock *>::iterator MI = be.begin(), 
	ME = be.end(); MI != ME; ++MI){
    
    if(seenBB[MI->second])
      continue;
    
    seenBB[MI->second] = 1;

    std::vector<BasicBlock *> sameTarget;
    sameTarget.clear();
    
    for(std::map<BasicBlock *, BasicBlock *>::iterator MMI = be.begin(), 
	  MME = be.end(); MMI != MME; ++MMI){
      
      if(MMI->first == MI->first)
	continue;
      
      if(MMI->second == MI->second)
	sameTarget.push_back(MMI->first);
      
    }
    
    //so more than one branch to same target
    if(sameTarget.size()){

      sameTarget.push_back(MI->first);

      BasicBlock *newBB = new BasicBlock("newCommon", MI->first->getParent());
      BranchInst *newBranch = new BranchInst(MI->second);

      newBB->getInstList().push_back(newBranch);

      std::map<PHINode *, std::vector<unsigned int> > phiMap;

      
      for(std::vector<BasicBlock *>::iterator VBI = sameTarget.begin(),
	    VBE = sameTarget.end(); VBI != VBE; ++VBI){

	//std::cerr<<(*VBI)->getName()<<"\n";

	BranchInst *ti = cast<BranchInst>((*VBI)->getTerminator());
	unsigned char index = 1;
	if(ti->getSuccessor(0) == MI->second){
	  index = 0;
	}

	ti->setSuccessor(index, newBB);

	for(BasicBlock::iterator BB2Inst = MI->second->begin(), 
	      BBend = MI->second->end(); BB2Inst != BBend; ++BB2Inst){
	  
	  if (PHINode *phiInst = dyn_cast<PHINode>(BB2Inst)){
	    int bbIndex;
	    bbIndex = phiInst->getBasicBlockIndex(*VBI);
	    if(bbIndex>=0){
	      phiMap[phiInst].push_back(bbIndex);
	      //phiInst->setIncomingBlock(bbIndex, newBB); 
	    }
	  }
	}
      }

      for(std::map<PHINode *, std::vector<unsigned int> >::iterator
	    PI = phiMap.begin(), PE = phiMap.end(); PI != PE; ++PI){
	
	PHINode *phiNode = new PHINode(PI->first->getType(), "phi", newBranch);
	for(std::vector<unsigned int>::iterator II = PI->second.begin(),
	      IE = PI->second.end(); II != IE; ++II){
	  phiNode->addIncoming(PI->first->getIncomingValue(*II),
			       PI->first->getIncomingBlock(*II));
	}

	std::vector<BasicBlock *> tempBB;
	for(std::vector<unsigned int>::iterator II = PI->second.begin(),
	      IE = PI->second.end(); II != IE; ++II){
	  tempBB.push_back(PI->first->getIncomingBlock(*II));
	}

	for(std::vector<BasicBlock *>::iterator II = tempBB.begin(),
	      IE = tempBB.end(); II != IE; ++II){
	  PI->first->removeIncomingValue(*II);
	}

	PI->first->addIncoming(phiNode, newBB);
      }
      //std::cerr<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
      //std::cerr<<MI->second;
      //std::cerr<<"-----------------------------------\n";
      //std::cerr<<newBB;
      //std::cerr<<"END%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
      
    }
  }
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
void CombineBranches::getBackEdges(Function &F){
  std::map<BasicBlock *, Color > color;
  std::map<BasicBlock *, int> d;
  std::map<BasicBlock *, BasicBlock *> be;
  int time=0;
  getBackEdgesVisit(F.begin(), color, d, time, be);

  removeRedundant(be);
}

//Per function pass for inserting counters and call function
bool CombineBranches::runOnFunction(Function &F){
  
  if(F.isExternal()) {
    return false;
  }

  //if(F.getName() == "main"){
   // F.setName("llvm_gprof_main");
  //}
  
  //std::cerr<<F;
  //std::cerr<<"///////////////////////////////////////////////\n";
  getBackEdges(F);
  
  return true;
}

} // End llvm namespace
