//===- PromoteMemoryToRegister.cpp - Convert memory refs to regs ----------===//
//
// This pass is used to promote memory references to be register references.  A
// simple example of the transformation performed by this pass is:
//
//        FROM CODE                           TO CODE
//   %X = alloca int, uint 1                 ret int 42
//   store int 42, int *%X
//   %Y = load int* %X
//   ret int %Y
//
// To do this transformation, a simple analysis is done to ensure it is safe.
// Currently this just loops over all alloca instructions, looking for
// instructions that are only used in simple load and stores.
//
// After this, the code is transformed by...something magical :)
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/PromoteMemoryToRegister.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/iMemory.h"
#include "llvm/iPHINode.h"
#include "llvm/iTerminators.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/ConstantVals.h"

using namespace std;

namespace {

//instance of the promoter -- to keep all the local function data.
// gets re-created for each function processed
class PromoteInstance
{
	protected:
	vector<AllocaInst*>            		Allocas;   // the alloca instruction..
	map<Instruction *, int>	   		AllocaLookup; //reverse mapping of above

	vector<vector<BasicBlock *> >  		WriteSets; // index corresponds to Allocas
	vector<vector<BasicBlock *> > 	 	PhiNodes;  // index corresponds to Allocas
	vector<vector<Value *> > 		CurrentValue; //the current value stack

	//list of instructions to remove at end of pass :)
	vector<Instruction *> killlist;

	set<BasicBlock *> 			visited;	//the basic blocks we've already visited
	map<BasicBlock *, vector<PHINode *> > 	new_phinodes;	//the phinodes we're adding


	void traverse(BasicBlock *f, BasicBlock * predecessor);
	bool PromoteFunction(Function *F, DominanceFrontier &DF);
	bool queuePhiNode(BasicBlock *bb, int alloca_index);
	void findSafeAllocas(Function *M);
	bool didchange;
	public:
	// I do this so that I can force the deconstruction of the local variables
	PromoteInstance(Function *F, DominanceFrontier &DF)
	{
		didchange=PromoteFunction(F, DF);
	}
	//This returns whether the pass changes anything
	operator bool () { return didchange; }
};

}  // end of anonymous namespace

// findSafeAllocas - Find allocas that are safe to promote
//
void PromoteInstance::findSafeAllocas(Function *F)  
{
  BasicBlock *BB = F->getEntryNode();  // Get the entry node for the function

  // Look at all instructions in the entry node
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(*I))       // Is it an alloca?
      if (!AI->isArrayAllocation()) {
	bool isSafe = true;
	for (Value::use_iterator UI = AI->use_begin(), UE = AI->use_end();
	     UI != UE; ++UI) {   // Loop over all of the uses of the alloca

	  // Only allow nonindexed memory access instructions...
	  if (MemAccessInst *MAI = dyn_cast<MemAccessInst>(*UI)) {
	    if (MAI->hasIndices()) {  // indexed?
	      // Allow the access if there is only one index and the index is zero.
	      if (*MAI->idx_begin() != ConstantUInt::get(Type::UIntTy, 0) ||
		  MAI->idx_begin()+1 != MAI->idx_end()) {
		isSafe = false; break;
	      }
	    }
	  } else {
	    isSafe = false; break;   // Not a load or store?
	  }
	}
	if (isSafe)              // If all checks pass, add alloca to safe list
	  {
	    AllocaLookup[AI]=Allocas.size();
	    Allocas.push_back(AI);
	  }
      }
}



bool PromoteInstance::PromoteFunction(Function *F, DominanceFrontier & DF) {
	// Calculate the set of safe allocas
	findSafeAllocas(F);

	// Add each alloca to the killlist
	// note: killlist is destroyed MOST recently added to least recently.
	killlist.assign(Allocas.begin(), Allocas.end());

	// Calculate the set of write-locations for each alloca.
	// this is analogous to counting the number of 'redefinitions' of each variable.
	for (unsigned i = 0; i<Allocas.size(); ++i)
	{
		AllocaInst * AI = Allocas[i];
		WriteSets.push_back(std::vector<BasicBlock *>()); //add a new set
		for (Value::use_iterator U = AI->use_begin();U!=AI->use_end();++U)
		{
			if (MemAccessInst *MAI = dyn_cast<StoreInst>(*U)) {
				WriteSets[i].push_back(MAI->getParent()); // jot down the basic-block it came from
			}
		}
	}

	// Compute the locations where PhiNodes need to be inserted
	// look at the dominance frontier of EACH basic-block we have a write in
	PhiNodes.resize(Allocas.size());
	for (unsigned i = 0; i<Allocas.size(); ++i)
	{
		for (unsigned j = 0; j<WriteSets[i].size(); j++)
		{
			//look up the DF for this write, add it to PhiNodes
			DominanceFrontier::const_iterator it = DF.find(WriteSets[i][j]);
			DominanceFrontier::DomSetType     s = (*it).second;
			for (DominanceFrontier::DomSetType::iterator p = s.begin();p!=s.end(); ++p)
			{
				if (queuePhiNode(*p, i))
                                  PhiNodes[i].push_back(*p);
			}
		}
		// perform iterative step
		for (unsigned k = 0; k<PhiNodes[i].size(); k++)
		{
			DominanceFrontier::const_iterator it = DF.find(PhiNodes[i][k]);
			DominanceFrontier::DomSetType     s = it->second;
			for (DominanceFrontier::DomSetType::iterator p = s.begin(); p!=s.end(); ++p)
			{
				if (queuePhiNode(*p,i))
				PhiNodes[i].push_back(*p);
			}
		}
	}

	// Walks all basic blocks in the function
	// performing the SSA rename algorithm
	// and inserting the phi nodes we marked as necessary
	BasicBlock * f = F->front(); //get root basic-block

	CurrentValue.push_back(vector<Value *>(Allocas.size()));

	traverse(f, NULL);  // there is no predecessor of the root node


	// ** REMOVE EVERYTHING IN THE KILL-LIST **
	// we need to kill 'uses' before root values
	// so we should probably run through in reverse
	for (vector<Instruction *>::reverse_iterator i = killlist.rbegin(); i!=killlist.rend(); ++i)
	{
		Instruction * r = *i;
		BasicBlock * o = r->getParent();
		//now go find..

		BasicBlock::InstListType & l = o->getInstList();
		o->getInstList().remove(r);
		delete r;
	}

	return !Allocas.empty();
}



void PromoteInstance::traverse(BasicBlock *f, BasicBlock * predecessor)
{
	vector<Value *> * tos = &CurrentValue.back(); //look at top-

	//if this is a BB needing a phi node, lookup/create the phinode for
	// each variable we need phinodes for.
	map<BasicBlock *, vector<PHINode *> >::iterator nd = new_phinodes.find(f);
	if (nd!=new_phinodes.end())
	{
		for (unsigned k = 0; k!=nd->second.size(); ++k)
		if (nd->second[k])
		{
			//at this point we can assume that the array has phi nodes.. let's
			// add the incoming data
			if ((*tos)[k])
			nd->second[k]->addIncoming((*tos)[k],predecessor);
			//also note that the active variable IS designated by the phi node
			(*tos)[k] = nd->second[k];
		}
	}

	//don't revisit nodes
	if (visited.find(f)!=visited.end())
	return;
	//mark as visited
	visited.insert(f);

	BasicBlock::iterator i = f->begin();
	//keep track of the value of each variable we're watching.. how?
	while(i!=f->end())
	{
		Instruction * inst = *i; //get the instruction
		//is this a write/read?
		if (LoadInst * LI = dyn_cast<LoadInst>(inst))
		{
			// This is a bit weird...
			Value * ptr = LI->getPointerOperand(); //of type value
			if (AllocaInst * srcinstr = dyn_cast<AllocaInst>(ptr))
			{
				map<Instruction *, int>::iterator ai = AllocaLookup.find(srcinstr);
				if (ai!=AllocaLookup.end())
				{
					if (Value *r = (*tos)[ai->second])
					{
						//walk the use list of this load and replace
						// all uses with r
						LI->replaceAllUsesWith(r);
						//now delete the instruction.. somehow..
						killlist.push_back((Instruction *)LI);
					}
				}
			}
		}
		else if (StoreInst * SI = dyn_cast<StoreInst>(inst))
		{
			// delete this instruction and mark the name as the
			// current holder of the value
			Value * ptr =  SI->getPointerOperand(); //of type value
			if (Instruction * srcinstr = dyn_cast<Instruction>(ptr))
			{
				map<Instruction *, int>::iterator ai = AllocaLookup.find(srcinstr);
				if (ai!=AllocaLookup.end())
				{
					//what value were we writing?
					Value * writeval = SI->getOperand(0);
					//write down...
					(*tos)[ai->second] = writeval;
					//now delete it.. somehow?
					killlist.push_back((Instruction *)SI);
				}
			}

		}
		else if (TerminatorInst * TI = dyn_cast<TerminatorInst>(inst))
		{
			// Recurse across our sucessors
			for (unsigned i = 0; i!=TI->getNumSuccessors(); i++)
			{
				CurrentValue.push_back(CurrentValue.back());
				traverse(TI->getSuccessor(i),f); //this node IS the predecessor
				CurrentValue.pop_back();
			}
		}
		i++;
	}
}

// queues a phi-node to be added to a basic-block for a specific Alloca
// returns true  if there wasn't already a phi-node for that variable


bool PromoteInstance::queuePhiNode(BasicBlock *bb, int i /*the alloca*/)
{
	map<BasicBlock *, vector<PHINode *> >::iterator nd;
	//look up the basic-block in question
	nd = new_phinodes.find(bb);
	//if the basic-block has no phi-nodes added, or at least none
	//for the i'th alloca. then add.
	if (nd==new_phinodes.end() || nd->second[i]==NULL)
	{
		//we're not added any phi nodes to this basicblock yet
		// create the phi-node array.
		if (nd==new_phinodes.end())
		{
			new_phinodes[bb] = vector<PHINode *>(Allocas.size());
			nd = new_phinodes.find(bb);
		}

		//find the type the alloca returns
		const PointerType * pt = Allocas[i]->getType();
		//create a phi-node using the DEREFERENCED type
		PHINode * ph = new PHINode(pt->getElementType(), Allocas[i]->getName()+".mem2reg");
		nd->second[i] = ph;
		//add the phi-node to the basic-block
		bb->getInstList().push_front(ph);
		return true;
	}
	return false;
}


namespace {
  struct PromotePass : public FunctionPass {

    // runOnFunction - To run this pass, first we calculate the alloca
    // instructions that are safe for promotion, then we promote each one.
    //
    virtual bool runOnFunction(Function *F) {
      return (bool)PromoteInstance(F, getAnalysis<DominanceFrontier>());
    }
    

    // getAnalysisUsage - We need dominance frontiers
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired(DominanceFrontier::ID);
    }
  };
}
  

// createPromoteMemoryToRegister - Provide an entry point to create this pass.
//
Pass *createPromoteMemoryToRegister() {
	return new PromotePass();
}


