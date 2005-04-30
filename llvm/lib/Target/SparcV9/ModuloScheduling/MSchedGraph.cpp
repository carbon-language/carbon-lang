//===-- MSchedGraph.cpp - Scheduling Graph ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A graph class for dependencies. This graph only contains true, anti, and
// output data dependencies for a given MachineBasicBlock. Dependencies
// across iterations are also computed. Unless data dependence analysis
// is provided, a conservative approach of adding dependencies between all
// loads and stores is taken.
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "ModuloSched"

#include "MSchedGraph.h"
#include "../SparcV9RegisterInfo.h"
#include "../MachineCodeForInstruction.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include <cstdlib>
#include <algorithm>
#include <set>

using namespace llvm;

//MSchedGraphNode constructor
MSchedGraphNode::MSchedGraphNode(const MachineInstr* inst,
				 MSchedGraph *graph, unsigned idx,
				 unsigned late, bool isBranch) 
  : Inst(inst), Parent(graph), index(idx), latency(late), 
    isBranchInstr(isBranch) {

  //Add to the graph
  graph->addNode(inst, this);
}

//MSchedGraphNode copy constructor
MSchedGraphNode::MSchedGraphNode(const MSchedGraphNode &N)
  : Predecessors(N.Predecessors), Successors(N.Successors) {

  Inst = N.Inst;
  Parent = N.Parent;
  index = N.index;
  latency = N.latency;
  isBranchInstr = N.isBranchInstr;

}

//Print the node (instruction and latency)
void MSchedGraphNode::print(std::ostream &os) const {
  os << "MSchedGraphNode: Inst=" << *Inst << ", latency= " << latency << "\n";
}


//Get the edge from a predecessor to this node
MSchedGraphEdge MSchedGraphNode::getInEdge(MSchedGraphNode *pred) {
  //Loop over all the successors of our predecessor
  //return the edge the corresponds to this in edge
  for (MSchedGraphNode::succ_iterator I = pred->succ_begin(),
         E = pred->succ_end(); I != E; ++I) {
    if (*I == this)
      return I.getEdge();
  }
  assert(0 && "Should have found edge between this node and its predecessor!");
  abort();
}

//Get the iteration difference for the edge from this node to its successor
unsigned MSchedGraphNode::getIteDiff(MSchedGraphNode *succ) {
  for(std::vector<MSchedGraphEdge>::iterator I = Successors.begin(), 
	E = Successors.end();
      I != E; ++I) {
    if(I->getDest() == succ)
      return I->getIteDiff();
  }
  return 0;
}

//Get the index into the vector of edges for the edge from pred to this node
unsigned MSchedGraphNode::getInEdgeNum(MSchedGraphNode *pred) {
  //Loop over all the successors of our predecessor
  //return the edge the corresponds to this in edge
  int count = 0;
  for(MSchedGraphNode::succ_iterator I = pred->succ_begin(), 
	E = pred->succ_end();
      I != E; ++I) {
    if(*I == this)
      return count;
    count++;
  }
  assert(0 && "Should have found edge between this node and its predecessor!");
  abort();
}

//Determine if succ is a successor of this node
bool MSchedGraphNode::isSuccessor(MSchedGraphNode *succ) {
  for(succ_iterator I = succ_begin(), E = succ_end(); I != E; ++I)
    if(*I == succ)
      return true;
  return false;
}

//Dtermine if pred is a predecessor of this node
bool MSchedGraphNode::isPredecessor(MSchedGraphNode *pred) {
  if(std::find( Predecessors.begin(),  Predecessors.end(), 
		pred) !=   Predecessors.end())
    return true;
  else
    return false;
}

//Add a node to the graph
void MSchedGraph::addNode(const MachineInstr *MI,
			  MSchedGraphNode *node) {

  //Make sure node does not already exist
  assert(GraphMap.find(MI) == GraphMap.end()
	 && "New MSchedGraphNode already exists for this instruction");

  GraphMap[MI] = node;
}

//Delete a node to the graph
void MSchedGraph::deleteNode(MSchedGraphNode *node) {

  //Delete the edge to this node from all predecessors
  while(node->pred_size() > 0) {
    //DEBUG(std::cerr << "Delete edge from: " << **P << " to " << *node << "\n");
    MSchedGraphNode *pred = *(node->pred_begin());
    pred->deleteSuccessor(node);
  }

  //Remove this node from the graph
  GraphMap.erase(node->getInst());

}


//Create a graph for a machine block. The ignoreInstrs map is so that
//we ignore instructions associated to the index variable since this
//is a special case in Modulo Scheduling.  We only want to deal with
//the body of the loop.
MSchedGraph::MSchedGraph(const MachineBasicBlock *bb, 
			 const TargetMachine &targ, 
			 std::map<const MachineInstr*, unsigned> &ignoreInstrs, 
			 DependenceAnalyzer &DA, 
			 std::map<MachineInstr*, Instruction*> &machineTollvm)
  : Target(targ) {

  //Make sure BB is not null,
  assert(bb != NULL && "Basic Block is null");

  BBs.push_back(bb);
  
  //Create nodes and edges for this BB
  buildNodesAndEdges(ignoreInstrs, DA, machineTollvm);

  //Experimental!
  //addBranchEdges();
}

//Copies the graph and keeps a map from old to new nodes
MSchedGraph::MSchedGraph(const MSchedGraph &G, 
			 std::map<MSchedGraphNode*, MSchedGraphNode*> &newNodes) 
  : Target(G.Target) {

  BBs = G.BBs;

  std::map<MSchedGraphNode*, MSchedGraphNode*> oldToNew;
  //Copy all nodes
  for(MSchedGraph::const_iterator N = G.GraphMap.begin(), 
	NE = G.GraphMap.end(); N != NE; ++N) {

    MSchedGraphNode *newNode = new MSchedGraphNode(*(N->second));
    oldToNew[&*(N->second)] = newNode;
    newNodes[newNode] = &*(N->second);
    GraphMap[&*(N->first)] = newNode;
  }

  //Loop over nodes and update edges to point to new nodes
  for(MSchedGraph::iterator N = GraphMap.begin(), NE = GraphMap.end(); 
      N != NE; ++N) {

    //Get the node we are dealing with
    MSchedGraphNode *node = &*(N->second);

    node->setParent(this);

    //Loop over nodes successors and predecessors and update to the new nodes
    for(unsigned i = 0; i < node->pred_size(); ++i) {
      node->setPredecessor(i, oldToNew[node->getPredecessor(i)]);
    }

    for(unsigned i = 0; i < node->succ_size(); ++i) {
      MSchedGraphEdge *edge = node->getSuccessor(i);
      MSchedGraphNode *oldDest = edge->getDest();
      edge->setDest(oldToNew[oldDest]);
    }
  }
}

//Deconstructor, deletes all nodes in the graph
MSchedGraph::~MSchedGraph () {
  for(MSchedGraph::iterator I = GraphMap.begin(), E = GraphMap.end(); 
      I != E; ++I)
    delete I->second;
}

//Print out graph
void MSchedGraph::print(std::ostream &os) const {
  for(MSchedGraph::const_iterator N = GraphMap.begin(), NE = GraphMap.end(); 
      N != NE; ++N) {
    
    //Get the node we are dealing with
    MSchedGraphNode *node = &*(N->second);

    os << "Node Start\n";
    node->print(os);
    os << "Successors:\n";
    //print successors
    for(unsigned i = 0; i < node->succ_size(); ++i) {
      MSchedGraphEdge *edge = node->getSuccessor(i);
      MSchedGraphNode *oldDest = edge->getDest();
      oldDest->print(os);
    }
    os << "Node End\n";
  }
}

//Calculate total delay
int MSchedGraph::totalDelay() {
  int sum = 0;

  for(MSchedGraph::const_iterator N = GraphMap.begin(), NE = GraphMap.end(); 
      N != NE; ++N) {
    
    //Get the node we are dealing with
    MSchedGraphNode *node = &*(N->second);
    sum += node->getLatency();
  }
  return sum;
}
//Experimental code to add edges from the branch to all nodes dependent upon it.
void hasPath(MSchedGraphNode *node, std::set<MSchedGraphNode*> &visited, 
	   std::set<MSchedGraphNode*> &branches, MSchedGraphNode *startNode,
	   std::set<std::pair<MSchedGraphNode*,MSchedGraphNode*> > &newEdges ) {

  visited.insert(node);
  DEBUG(std::cerr << "Visiting: " << *node << "\n");
  //Loop over successors
  for(unsigned i = 0; i < node->succ_size(); ++i) {
    MSchedGraphEdge *edge = node->getSuccessor(i);
    MSchedGraphNode *dest = edge->getDest();
    if(branches.count(dest))
      newEdges.insert(std::make_pair(dest, startNode));

    //only visit if we have not already
    else if(!visited.count(dest)) {
      if(edge->getIteDiff() == 0)
	hasPath(dest, visited, branches, startNode, newEdges);}

  }

}

//Experimental code to add edges from the branch to all nodes dependent upon it.
void MSchedGraph::addBranchEdges() {
  std::set<MSchedGraphNode*> branches;
  std::set<MSchedGraphNode*> nodes;

  for(MSchedGraph::iterator I = GraphMap.begin(), E = GraphMap.end(); 
      I != E; ++I) {
    if(I->second->isBranch())
      if(I->second->hasPredecessors())
	branches.insert(I->second);
  }

  //See if there is a path first instruction to the branches, if so, add an
  //iteration dependence between that node and the branch
  std::set<std::pair<MSchedGraphNode*, MSchedGraphNode*> > newEdges;
  for(MSchedGraph::iterator I = GraphMap.begin(), E = GraphMap.end(); 
      I != E; ++I) {
    std::set<MSchedGraphNode*> visited;
    hasPath((I->second), visited, branches, (I->second), newEdges);
  }

  //Spit out all edges we are going to add
  unsigned min = GraphMap.size();
  if(newEdges.size() == 1) {
    ((newEdges.begin())->first)->addOutEdge(((newEdges.begin())->second),
			   MSchedGraphEdge::BranchDep,
			   MSchedGraphEdge::NonDataDep, 1);
  }
  else {

    unsigned count = 0;
    MSchedGraphNode *start;
    MSchedGraphNode *end;
    for(std::set<std::pair<MSchedGraphNode*, MSchedGraphNode*> >::iterator I = newEdges.begin(), E = newEdges.end(); I != E; ++I) {

      DEBUG(std::cerr << "Branch Edge from: " << *(I->first) << " to " << *(I->second) << "\n");

      //      if(I->second->getIndex() <= min) {
	start = I->first;
	end = I->second;
	//min = I->second->getIndex();
	//}
	start->addOutEdge(end,
			  MSchedGraphEdge::BranchDep,
			  MSchedGraphEdge::NonDataDep, 1);
    }
  }
}


//Add edges between the nodes
void MSchedGraph::buildNodesAndEdges(std::map<const MachineInstr*, unsigned> &ignoreInstrs,
				     DependenceAnalyzer &DA,
		       std::map<MachineInstr*, Instruction*> &machineTollvm) {
  

  //Get Machine target information for calculating latency
  const TargetInstrInfo *MTI = Target.getInstrInfo();

  std::vector<MSchedGraphNode*> memInstructions;
  std::map<int, std::vector<OpIndexNodePair> > regNumtoNodeMap;
  std::map<const Value*, std::vector<OpIndexNodePair> > valuetoNodeMap;

  //Save PHI instructions to deal with later
  std::vector<const MachineInstr*> phiInstrs;
  unsigned index = 0;

  for(std::vector<const MachineBasicBlock*>::iterator B = BBs.begin(), 
	BE = BBs.end(); B != BE; ++B) {
    
    const MachineBasicBlock *BB = *B;

    //Loop over instructions in MBB and add nodes and edges
    for (MachineBasicBlock::const_iterator MI = BB->begin(), e = BB->end(); 
	 MI != e; ++MI) {
      
      //Ignore indvar instructions
      if(ignoreInstrs.count(MI)) {
	++index;
	continue;
      }
      
      //Get each instruction of machine basic block, get the delay
      //using the op code, create a new node for it, and add to the
      //graph.
      
      MachineOpCode opCode = MI->getOpcode();
      int delay;
      
#if 0  // FIXME: LOOK INTO THIS
      //Check if subsequent instructions can be issued before
      //the result is ready, if so use min delay.
      if(MTI->hasResultInterlock(MIopCode))
	delay = MTI->minLatency(MIopCode);
      else
#endif
	//Get delay
	delay = MTI->maxLatency(opCode);
      
      //Create new node for this machine instruction and add to the graph.
      //Create only if not a nop
      if(MTI->isNop(opCode))
	continue;
      
      //Sparc BE does not use PHI opcode, so assert on this case
      assert(opCode != TargetInstrInfo::PHI && "Did not expect PHI opcode");
      
      bool isBranch = false;
      
      //We want to flag the branch node to treat it special
      if(MTI->isBranch(opCode))
	isBranch = true;
      
      //Node is created and added to the graph automatically
      MSchedGraphNode *node =  new MSchedGraphNode(MI, this, index, delay, 
						   isBranch);
      
      DEBUG(std::cerr << "Created Node: " << *node << "\n");
      
      //Check OpCode to keep track of memory operations to add memory
      //dependencies later.
      if(MTI->isLoad(opCode) || MTI->isStore(opCode))
	memInstructions.push_back(node);
      
      //Loop over all operands, and put them into the register number to
      //graph node map for determining dependencies
      //If an operands is a use/def, we have an anti dependence to itself
      for(unsigned i=0; i < MI->getNumOperands(); ++i) {
	//Get Operand
	const MachineOperand &mOp = MI->getOperand(i);
	
	//Check if it has an allocated register
	if(mOp.hasAllocatedReg()) {
	  int regNum = mOp.getReg();
	  
	  if(regNum != SparcV9::g0) {
	    //Put into our map
	    regNumtoNodeMap[regNum].push_back(std::make_pair(i, node));
	  }
	  continue;
	}
	
	
	//Add virtual registers dependencies
	//Check if any exist in the value map already and create dependencies
	//between them.
	if(mOp.getType() == MachineOperand::MO_VirtualRegister 
	   ||  mOp.getType() == MachineOperand::MO_CCRegister) {
	  
	  //Make sure virtual register value is not null
	  assert((mOp.getVRegValue() != NULL) && "Null value is defined");
	  
	  //Check if this is a read operation in a phi node, if so DO NOT PROCESS
	  if(mOp.isUse() && (opCode == TargetInstrInfo::PHI)) {
	    DEBUG(std::cerr << "Read Operation in a PHI node\n");
	    continue;
	  }
	  
	  if (const Value* srcI = mOp.getVRegValue()) {
	    
	    //Find value in the map
	    std::map<const Value*, std::vector<OpIndexNodePair> >::iterator V
	      = valuetoNodeMap.find(srcI);
	    
	    //If there is something in the map already, add edges from
	    //those instructions
	    //to this one we are processing
	    if(V != valuetoNodeMap.end()) {
	      addValueEdges(V->second, node, mOp.isUse(), mOp.isDef(), phiInstrs);
	      
	      //Add to value map
	      V->second.push_back(std::make_pair(i,node));
	    }
	    //Otherwise put it in the map
	    else
	      //Put into value map
	      valuetoNodeMap[mOp.getVRegValue()].push_back(std::make_pair(i, node));
	  }
	}
      }
      ++index;
    }
    
    //Loop over LLVM BB, examine phi instructions, and add them to our
    //phiInstr list to process
    const BasicBlock *llvm_bb = BB->getBasicBlock();
    for(BasicBlock::const_iterator I = llvm_bb->begin(), E = llvm_bb->end(); 
	I != E; ++I) {
      if(const PHINode *PN = dyn_cast<PHINode>(I)) {
	MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(PN);
	for (unsigned j = 0; j < tempMvec.size(); j++) {
	  if(!ignoreInstrs.count(tempMvec[j])) {
	    DEBUG(std::cerr << "Inserting phi instr into map: " << *tempMvec[j] << "\n");
	    phiInstrs.push_back((MachineInstr*) tempMvec[j]);
	  }
	}
      }
      
    }
    
    addMemEdges(memInstructions, DA, machineTollvm);
    addMachRegEdges(regNumtoNodeMap);
    
    //Finally deal with PHI Nodes and Value*
    for(std::vector<const MachineInstr*>::iterator I = phiInstrs.begin(), 
	  E = phiInstrs.end(); I != E;  ++I) {
      
      //Get Node for this instruction
      std::map<const MachineInstr*, MSchedGraphNode*>::iterator X;
      X = find(*I);
      
      if(X == GraphMap.end())
	continue;
      
      MSchedGraphNode *node = X->second;
      
      DEBUG(std::cerr << "Adding ite diff edges for node: " << *node << "\n");
      
      //Loop over operands for this instruction and add value edges
      for(unsigned i=0; i < (*I)->getNumOperands(); ++i) {
	//Get Operand
	const MachineOperand &mOp = (*I)->getOperand(i);
	if((mOp.getType() == MachineOperand::MO_VirtualRegister 
	    ||  mOp.getType() == MachineOperand::MO_CCRegister) && mOp.isUse()) {
	  
	  //find the value in the map
	  if (const Value* srcI = mOp.getVRegValue()) {
	    
	    //Find value in the map
	    std::map<const Value*, std::vector<OpIndexNodePair> >::iterator V
	      = valuetoNodeMap.find(srcI);
	    
	    //If there is something in the map already, add edges from
	    //those instructions
	    //to this one we are processing
	    if(V != valuetoNodeMap.end()) {
	      addValueEdges(V->second, node, mOp.isUse(), mOp.isDef(), 
			    phiInstrs, 1);
	    }
	  }
	}
      }
    }
  }
}
//Add dependencies for Value*s
void MSchedGraph::addValueEdges(std::vector<OpIndexNodePair> &NodesInMap,
				MSchedGraphNode *destNode, bool nodeIsUse,
				bool nodeIsDef, std::vector<const MachineInstr*> &phiInstrs, int diff) {

  for(std::vector<OpIndexNodePair>::iterator I = NodesInMap.begin(),
	E = NodesInMap.end(); I != E; ++I) {

    //Get node in vectors machine operand that is the same value as node
    MSchedGraphNode *srcNode = I->second;
    MachineOperand mOp = srcNode->getInst()->getOperand(I->first);

    if(diff > 0)
      if(std::find(phiInstrs.begin(), phiInstrs.end(), srcNode->getInst()) == phiInstrs.end())
	continue;

    //Node is a Def, so add output dep.
    if(nodeIsDef) {
      if(mOp.isUse()) {
	DEBUG(std::cerr << "Edge from " << *srcNode << " to " << *destNode << " (itediff=" << diff << ", type=anti)\n");
	srcNode->addOutEdge(destNode, MSchedGraphEdge::ValueDep,
			    MSchedGraphEdge::AntiDep, diff);
      }
      if(mOp.isDef()) {
	DEBUG(std::cerr << "Edge from " << *srcNode << " to " << *destNode << " (itediff=" << diff << ", type=output)\n");
	srcNode->addOutEdge(destNode, MSchedGraphEdge::ValueDep,
			    MSchedGraphEdge::OutputDep, diff);
      }
    }
    if(nodeIsUse) {
      if(mOp.isDef()) {
	DEBUG(std::cerr << "Edge from " << *srcNode << " to " << *destNode << " (itediff=" << diff << ", type=true)\n");
	srcNode->addOutEdge(destNode, MSchedGraphEdge::ValueDep,
			    MSchedGraphEdge::TrueDep, diff);
      }
    }
  }
}

//Add dependencies for machine registers across iterations
void MSchedGraph::addMachRegEdges(std::map<int, std::vector<OpIndexNodePair> >& regNumtoNodeMap) {
  //Loop over all machine registers in the map, and add dependencies
  //between the instructions that use it
  typedef std::map<int, std::vector<OpIndexNodePair> > regNodeMap;
  for(regNodeMap::iterator I = regNumtoNodeMap.begin(); 
      I != regNumtoNodeMap.end(); ++I) {
    //Get the register number
    int regNum = (*I).first;

    //Get Vector of nodes that use this register
    std::vector<OpIndexNodePair> Nodes = (*I).second;

    //Loop over nodes and determine the dependence between the other
    //nodes in the vector
    for(unsigned i =0; i < Nodes.size(); ++i) {

      //Get src node operator index that uses this machine register
      int srcOpIndex = Nodes[i].first;

      //Get the actual src Node
      MSchedGraphNode *srcNode = Nodes[i].second;

      //Get Operand
      const MachineOperand &srcMOp = srcNode->getInst()->getOperand(srcOpIndex);

      bool srcIsUseandDef = srcMOp.isDef() && srcMOp.isUse();
      bool srcIsUse = srcMOp.isUse() && !srcMOp.isDef();


      //Look at all instructions after this in execution order
      for(unsigned j=i+1; j < Nodes.size(); ++j) {
	
	//Sink node is a write
	if(Nodes[j].second->getInst()->getOperand(Nodes[j].first).isDef()) {
	              //Src only uses the register (read)
            if(srcIsUse)
	      srcNode->addOutEdge(Nodes[j].second, 
				  MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::AntiDep);
	
            else if(srcIsUseandDef) {
	      srcNode->addOutEdge(Nodes[j].second, 
				  MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::AntiDep);
	      
	      srcNode->addOutEdge(Nodes[j].second, 
				  MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::OutputDep);
	    }
            else
	      srcNode->addOutEdge(Nodes[j].second, 
				  MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::OutputDep);
	}
	//Dest node is a read
	else {
	  if(!srcIsUse || srcIsUseandDef)
	    srcNode->addOutEdge(Nodes[j].second, 
				MSchedGraphEdge::MachineRegister,
				MSchedGraphEdge::TrueDep);
	}

      }

      //Look at all the instructions before this one since machine registers
      //could live across iterations.
      for(unsigned j = 0; j < i; ++j) {
		//Sink node is a write
	if(Nodes[j].second->getInst()->getOperand(Nodes[j].first).isDef()) {
	              //Src only uses the register (read)
            if(srcIsUse)
	      srcNode->addOutEdge(Nodes[j].second, 
				  MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::AntiDep, 1);
            else if(srcIsUseandDef) {
	      srcNode->addOutEdge(Nodes[j].second, 
				  MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::AntiDep, 1);
	      
	      srcNode->addOutEdge(Nodes[j].second, 
				  MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::OutputDep, 1);
	    }
            else
	      srcNode->addOutEdge(Nodes[j].second, 
				  MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::OutputDep, 1);
	}
	//Dest node is a read
	else {
	  if(!srcIsUse || srcIsUseandDef)
	    srcNode->addOutEdge(Nodes[j].second, 
				MSchedGraphEdge::MachineRegister,
				MSchedGraphEdge::TrueDep,1 );
	}
	

      }

    }

  }

}

//Add edges between all loads and stores
//Can be less strict with alias analysis and data dependence analysis.
void MSchedGraph::addMemEdges(const std::vector<MSchedGraphNode*>& memInst, 
		      DependenceAnalyzer &DA, 
		      std::map<MachineInstr*, Instruction*> &machineTollvm) {

  //Get Target machine instruction info
  const TargetInstrInfo *TMI = Target.getInstrInfo();

  //Loop over all memory instructions in the vector
  //Knowing that they are in execution, add true, anti, and output dependencies
  for (unsigned srcIndex = 0; srcIndex < memInst.size(); ++srcIndex) {

    MachineInstr *srcInst = (MachineInstr*) memInst[srcIndex]->getInst();

    //Get the machine opCode to determine type of memory instruction
    MachineOpCode srcNodeOpCode = srcInst->getOpcode();
    
    //All instructions after this one in execution order have an
    //iteration delay of 0
    for(unsigned destIndex = 0; destIndex < memInst.size(); ++destIndex) {

      //No self loops
      if(destIndex == srcIndex)
	continue;

      MachineInstr *destInst = (MachineInstr*) memInst[destIndex]->getInst();

      DEBUG(std::cerr << "MInst1: " << *srcInst << "\n");
      DEBUG(std::cerr << "MInst2: " << *destInst << "\n");
      
      //Assuming instructions without corresponding llvm instructions
      //are from constant pools.
      if (!machineTollvm.count(srcInst) || !machineTollvm.count(destInst))
	continue;
      
      bool useDepAnalyzer = true;

      //Some machine loads and stores are generated by casts, so be
      //conservative and always add deps
      Instruction *srcLLVM = machineTollvm[srcInst];
      Instruction *destLLVM = machineTollvm[destInst];
      if(!isa<LoadInst>(srcLLVM) 
	 && !isa<StoreInst>(srcLLVM)) {
	if(isa<BinaryOperator>(srcLLVM)) {
	  if(isa<ConstantFP>(srcLLVM->getOperand(0)) || isa<ConstantFP>(srcLLVM->getOperand(1)))
	    continue;
	}
	useDepAnalyzer = false;
      }
      if(!isa<LoadInst>(destLLVM) 
	 && !isa<StoreInst>(destLLVM)) {
	if(isa<BinaryOperator>(destLLVM)) {
	  if(isa<ConstantFP>(destLLVM->getOperand(0)) || isa<ConstantFP>(destLLVM->getOperand(1)))
	    continue;
	}
	useDepAnalyzer = false;
      }

      //Use dep analysis when we have corresponding llvm loads/stores
      if(useDepAnalyzer) {
	bool srcBeforeDest = true;
	if(destIndex < srcIndex)
	  srcBeforeDest = false;

	DependenceResult dr = DA.getDependenceInfo(machineTollvm[srcInst], 
						   machineTollvm[destInst], 
						   srcBeforeDest);
	
	for(std::vector<Dependence>::iterator d = dr.dependences.begin(), 
	      de = dr.dependences.end(); d != de; ++d) {
	  //Add edge from load to store
	  memInst[srcIndex]->addOutEdge(memInst[destIndex], 
					MSchedGraphEdge::MemoryDep, 
					d->getDepType(), d->getIteDiff());
	  
	}
      }
      //Otherwise, we can not do any further analysis and must make a dependence
      else {
		
	//Get the machine opCode to determine type of memory instruction
	MachineOpCode destNodeOpCode = destInst->getOpcode();

	//Get the Value* that we are reading from the load, always the first op
	const MachineOperand &mOp = srcInst->getOperand(0);
	const MachineOperand &mOp2 = destInst->getOperand(0);
	
	if(mOp.hasAllocatedReg())
	  if(mOp.getReg() == SparcV9::g0)
	    continue;
	if(mOp2.hasAllocatedReg())
	  if(mOp2.getReg() == SparcV9::g0)
	    continue;

	DEBUG(std::cerr << "Adding dependence for machine instructions\n");
	//Load-Store deps
	if(TMI->isLoad(srcNodeOpCode)) {

	  if(TMI->isStore(destNodeOpCode))
	    memInst[srcIndex]->addOutEdge(memInst[destIndex], 
					  MSchedGraphEdge::MemoryDep, 
					  MSchedGraphEdge::AntiDep, 0);
	}
	else if(TMI->isStore(srcNodeOpCode)) {
	  if(TMI->isStore(destNodeOpCode))
	    memInst[srcIndex]->addOutEdge(memInst[destIndex], 
					  MSchedGraphEdge::MemoryDep, 
					  MSchedGraphEdge::OutputDep, 0);

	  else
	    memInst[srcIndex]->addOutEdge(memInst[destIndex], 
					  MSchedGraphEdge::MemoryDep, 
					  MSchedGraphEdge::TrueDep, 0);
	}
      }
    }
  }
}
