//===-- MSchedGraph.h - Scheduling Graph ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A graph class for dependencies
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "ModuloSched"

#include "MSchedGraph.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "Support/Debug.h"
#include <iostream>
using namespace llvm;

MSchedGraphNode::MSchedGraphNode(const MachineInstr* inst, 
				 MSchedGraph *graph, 
				 unsigned late) 
  : Inst(inst), Parent(graph), latency(late) {

  //Add to the graph
  graph->addNode(inst, this);
}

void MSchedGraphNode::print(std::ostream &os) const {
  os << "MSchedGraphNode: Inst=" << *Inst << ", latency= " << latency << "\n"; 
}

MSchedGraphEdge MSchedGraphNode::getInEdge(MSchedGraphNode *pred) {
  //Loop over all the successors of our predecessor
  //return the edge the corresponds to this in edge
  for(MSchedGraphNode::succ_iterator I = pred->succ_begin(), E = pred->succ_end();
      I != E; ++I) {
    if(*I == this)
      return I.getEdge();
  }
  assert(0 && "Should have found edge between this node and its predecessor!");
 
}

unsigned MSchedGraphNode::getInEdgeNum(MSchedGraphNode *pred) {
  //Loop over all the successors of our predecessor
  //return the edge the corresponds to this in edge
  int count = 0;
  for(MSchedGraphNode::succ_iterator I = pred->succ_begin(), E = pred->succ_end();
      I != E; ++I) {
    if(*I == this)
      return count;
    count++;
  }
  assert(0 && "Should have found edge between this node and its predecessor!");
  abort();
}
bool MSchedGraphNode::isSuccessor(MSchedGraphNode *succ) {
  for(succ_iterator I = succ_begin(), E = succ_end(); I != E; ++I)
    if(*I == succ)
      return true;
  return false;
}


bool MSchedGraphNode::isPredecessor(MSchedGraphNode *pred) {
  if(find( Predecessors.begin(),  Predecessors.end(), pred) !=   Predecessors.end())
    return true;
  else
    return false;
}


void MSchedGraph::addNode(const MachineInstr *MI,
			  MSchedGraphNode *node) {
  
  //Make sure node does not already exist  
  assert(GraphMap.find(MI) == GraphMap.end() 
	 && "New MSchedGraphNode already exists for this instruction");
  
  GraphMap[MI] = node;
}

MSchedGraph::MSchedGraph(const MachineBasicBlock *bb, const TargetMachine &targ)
  : BB(bb), Target(targ) {
  
  //Make sure BB is not null, 
  assert(BB != NULL && "Basic Block is null");
  
  DEBUG(std::cerr << "Constructing graph for " << bb << "\n");

  //Create nodes and edges for this BB
  buildNodesAndEdges();
}

MSchedGraph::~MSchedGraph () {
  for(MSchedGraph::iterator I = GraphMap.begin(), E = GraphMap.end(); I != E; ++I)
    delete I->second;
}

void MSchedGraph::buildNodesAndEdges() {
  
  //Get Machine target information for calculating latency
  const TargetInstrInfo &MTI = Target.getInstrInfo();

  std::vector<MSchedGraphNode*> memInstructions;
  std::map<int, std::vector<OpIndexNodePair> > regNumtoNodeMap;
  std::map<const Value*, std::vector<OpIndexNodePair> > valuetoNodeMap;

  //Save PHI instructions to deal with later
  std::vector<const MachineInstr*> phiInstrs;

  //Loop over instructions in MBB and add nodes and edges
  for (MachineBasicBlock::const_iterator MI = BB->begin(), e = BB->end(); MI != e; ++MI) {
    //Get each instruction of machine basic block, get the delay
    //using the op code, create a new node for it, and add to the
    //graph.
    
    MachineOpCode MIopCode = MI->getOpcode();
    int delay;

#if 0  // FIXME: LOOK INTO THIS
    //Check if subsequent instructions can be issued before
    //the result is ready, if so use min delay.
    if(MTI.hasResultInterlock(MIopCode))
      delay = MTI.minLatency(MIopCode);
    else
#endif
      /// FIxME: get this from the sched class.
      delay = 7; //MTI.maxLatency(MIopCode);
    
    //Create new node for this machine instruction and add to the graph.
    //Create only if not a nop
    if(MTI.isNop(MIopCode))
      continue;
    
    //Add PHI to phi instruction list to be processed later
    if (MIopCode == TargetInstrInfo::PHI)
      phiInstrs.push_back(MI);

    //Node is created and added to the graph automatically
    MSchedGraphNode *node =  new MSchedGraphNode(MI, this, delay);

    DEBUG(std::cerr << "Created Node: " << *node << "\n"); 
    
    //Check OpCode to keep track of memory operations to add memory dependencies later.
    MachineOpCode opCode = MI->getOpcode();

    if(MTI.isLoad(opCode) || MTI.isStore(opCode))
      memInstructions.push_back(node);

    //Loop over all operands, and put them into the register number to
    //graph node map for determining dependencies
    //If an operands is a use/def, we have an anti dependence to itself
    for(unsigned i=0; i < MI->getNumOperands(); ++i) {
      //Get Operand
      const MachineOperand &mOp = MI->getOperand(i);
      
      //Check if it has an allocated register (Note: this means it
      //is greater then zero because zero is a special register for
      //Sparc that holds the constant zero
      if(mOp.hasAllocatedReg()) {
	int regNum = mOp.getReg();
	
	//Put into our map
	regNumtoNodeMap[regNum].push_back(std::make_pair(i, node));
	continue;
      }
      
      
      //Add virtual registers dependencies
      //Check if any exist in the value map already and create dependencies
      //between them.
      if(mOp.getType() == MachineOperand::MO_VirtualRegister ||  mOp.getType() == MachineOperand::MO_CCRegister) {

	//Make sure virtual register value is not null
	assert((mOp.getVRegValue() != NULL) && "Null value is defined");

	//Check if this is a read operation in a phi node, if so DO NOT PROCESS
	if(mOp.isUse() && (MIopCode == TargetInstrInfo::PHI))
	  continue;

      
	if (const Value* srcI = mOp.getVRegValue()) {
	  
	  //Find value in the map
	  std::map<const Value*, std::vector<OpIndexNodePair> >::iterator V 
	    = valuetoNodeMap.find(srcI);
	  
	  //If there is something in the map already, add edges from
	  //those instructions
	  //to this one we are processing
	  if(V != valuetoNodeMap.end()) {
	    addValueEdges(V->second, node, mOp.isUse(), mOp.isDef());
	    
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
  }
  addMemEdges(memInstructions);
  addMachRegEdges(regNumtoNodeMap);

  //Finally deal with PHI Nodes and Value*
  for(std::vector<const MachineInstr*>::iterator I = phiInstrs.begin(), E = phiInstrs.end(); I != E;  ++I) {
    //Get Node for this instruction
    MSchedGraphNode *node = find(*I)->second;
  
    //Loop over operands for this instruction and add value edges
    for(unsigned i=0; i < (*I)->getNumOperands(); ++i) {
      //Get Operand
      const MachineOperand &mOp = (*I)->getOperand(i);
      if((mOp.getType() == MachineOperand::MO_VirtualRegister ||  mOp.getType() == MachineOperand::MO_CCRegister) && mOp.isUse()) {
	//find the value in the map
	if (const Value* srcI = mOp.getVRegValue()) {
	  
	  //Find value in the map
	  std::map<const Value*, std::vector<OpIndexNodePair> >::iterator V 
	    = valuetoNodeMap.find(srcI);
	  
	  //If there is something in the map already, add edges from
	  //those instructions
	  //to this one we are processing
	  if(V != valuetoNodeMap.end()) {
	    addValueEdges(V->second, node, mOp.isUse(), mOp.isDef(), 1);
	  }
	}
      }
    }
  }
} 

void MSchedGraph::addValueEdges(std::vector<OpIndexNodePair> &NodesInMap,
				MSchedGraphNode *destNode, bool nodeIsUse, 
				bool nodeIsDef, int diff) {

  for(std::vector<OpIndexNodePair>::iterator I = NodesInMap.begin(), 
	E = NodesInMap.end(); I != E; ++I) {
    
    //Get node in vectors machine operand that is the same value as node
    MSchedGraphNode *srcNode = I->second;
    MachineOperand mOp = srcNode->getInst()->getOperand(I->first);

    //Node is a Def, so add output dep.
    if(nodeIsDef) {
      if(mOp.isUse())
	srcNode->addOutEdge(destNode, MSchedGraphEdge::ValueDep, 
			    MSchedGraphEdge::AntiDep, diff);
      if(mOp.isDef())
	srcNode->addOutEdge(destNode, MSchedGraphEdge::ValueDep, 
			    MSchedGraphEdge::OutputDep, diff);
      
    }
    if(nodeIsUse) {
      if(mOp.isDef())
	srcNode->addOutEdge(destNode, MSchedGraphEdge::ValueDep, 
			    MSchedGraphEdge::TrueDep, diff);
    }
  } 
}


void MSchedGraph::addMachRegEdges(std::map<int, std::vector<OpIndexNodePair> >& regNumtoNodeMap) {
  //Loop over all machine registers in the map, and add dependencies
  //between the instructions that use it
  typedef std::map<int, std::vector<OpIndexNodePair> > regNodeMap;
  for(regNodeMap::iterator I = regNumtoNodeMap.begin(); I != regNumtoNodeMap.end(); ++I) {
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
	      srcNode->addOutEdge(Nodes[j].second, MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::AntiDep);
	    
            else if(srcIsUseandDef) {
	      srcNode->addOutEdge(Nodes[j].second, MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::AntiDep);
	      
	      srcNode->addOutEdge(Nodes[j].second, MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::OutputDep);
	    }
            else
	      srcNode->addOutEdge(Nodes[j].second, MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::OutputDep);
	}
	//Dest node is a read
	else {
	  if(!srcIsUse || srcIsUseandDef)
	    srcNode->addOutEdge(Nodes[j].second, MSchedGraphEdge::MachineRegister,
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
	      srcNode->addOutEdge(Nodes[j].second, MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::AntiDep, 1);
	    
            else if(srcIsUseandDef) {
	      srcNode->addOutEdge(Nodes[j].second, MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::AntiDep, 1);
	      
	      srcNode->addOutEdge(Nodes[j].second, MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::OutputDep, 1);
	    }
            else
	      srcNode->addOutEdge(Nodes[j].second, MSchedGraphEdge::MachineRegister,
				  MSchedGraphEdge::OutputDep, 1);
	}
	//Dest node is a read
	else {
	  if(!srcIsUse || srcIsUseandDef)
	    srcNode->addOutEdge(Nodes[j].second, MSchedGraphEdge::MachineRegister,
				MSchedGraphEdge::TrueDep,1 );
	}
	

      }

    }
    
  }
  
}

void MSchedGraph::addMemEdges(const std::vector<MSchedGraphNode*>& memInst) {

  //Get Target machine instruction info
  const TargetInstrInfo& TMI = Target.getInstrInfo();
  
  //Loop over all memory instructions in the vector
  //Knowing that they are in execution, add true, anti, and output dependencies
  for (unsigned srcIndex = 0; srcIndex < memInst.size(); ++srcIndex) {

    //Get the machine opCode to determine type of memory instruction
    MachineOpCode srcNodeOpCode = memInst[srcIndex]->getInst()->getOpcode();
      
    //All instructions after this one in execution order have an iteration delay of 0
    for(unsigned destIndex = srcIndex + 1; destIndex < memInst.size(); ++destIndex) {
       
      //source is a Load, so add anti-dependencies (store after load)
      if(TMI.isLoad(srcNodeOpCode))
	if(TMI.isStore(memInst[destIndex]->getInst()->getOpcode()))
	  memInst[srcIndex]->addOutEdge(memInst[destIndex], 
			      MSchedGraphEdge::MemoryDep, 
			      MSchedGraphEdge::AntiDep);
      
      //If source is a store, add output and true dependencies
      if(TMI.isStore(srcNodeOpCode)) {
	if(TMI.isStore(memInst[destIndex]->getInst()->getOpcode()))
	   memInst[srcIndex]->addOutEdge(memInst[destIndex], 
			      MSchedGraphEdge::MemoryDep, 
			      MSchedGraphEdge::OutputDep);
	else
	  memInst[srcIndex]->addOutEdge(memInst[destIndex], 
			      MSchedGraphEdge::MemoryDep, 
			      MSchedGraphEdge::TrueDep);
      }
    }
    
    //All instructions before the src in execution order have an iteration delay of 1
    for(unsigned destIndex = 0; destIndex < srcIndex; ++destIndex) {
      //source is a Load, so add anti-dependencies (store after load)
      if(TMI.isLoad(srcNodeOpCode))
	if(TMI.isStore(memInst[destIndex]->getInst()->getOpcode()))
	  memInst[srcIndex]->addOutEdge(memInst[destIndex], 
			      MSchedGraphEdge::MemoryDep, 
			      MSchedGraphEdge::AntiDep, 1);
      if(TMI.isStore(srcNodeOpCode)) {
	if(TMI.isStore(memInst[destIndex]->getInst()->getOpcode()))
	  memInst[srcIndex]->addOutEdge(memInst[destIndex], 
			      MSchedGraphEdge::MemoryDep, 
			      MSchedGraphEdge::OutputDep, 1);
	else
	  memInst[srcIndex]->addOutEdge(memInst[destIndex], 
			      MSchedGraphEdge::MemoryDep, 
			      MSchedGraphEdge::TrueDep, 1);
      }
	  
    }
    
  }
}
