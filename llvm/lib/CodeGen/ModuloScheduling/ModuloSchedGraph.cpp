//===- ModuloSchedGraph.cpp - Modulo Scheduling Graph and Set -*- C++ -*---===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// Description here
//===----------------------------------------------------------------------===//

#include "ModuloSchedGraph.h"
#include "llvm/Type.h"

ModuloSchedGraphNode::ModuloSchedGraphNode(unsigned id, int index, 
					   const Instruction *inst, 
					   const TargetMachine &targ) 
  : SchedGraphNodeCommon(id, index), Inst(inst), Target(targ) {
}

void ModuloSchedGraphNode::print(std::ostream &os) const {
  os << "Modulo Scheduling Node\n";
}

ModuloSchedGraph::ModuloSchedGraph(const BasicBlock *bb, const TargetMachine &targ) 
  : SchedGraphCommon(), BB(bb), Target(targ) {

  assert(BB != NULL && "Basic Block is null");

  //Builds nodes from each instruction in the basic block
  buildNodesForBB();

}

void ModuloSchedGraph::buildNodesForBB() {
  int count = 0;
  for (BasicBlock::const_iterator i = BB->begin(), e = BB->end(); i != e; ++i) {
    addNode(i,new ModuloSchedGraphNode(size(), count, i, Target));
    count++;
  }

	    //Get machine instruction(s) for the llvm instruction
	    //MachineCodeForInstruction &MC = MachineCodeForInstruction::get(Node->first);
	    

}

void ModuloSchedGraph::addNode(const Instruction *I,
			       ModuloSchedGraphNode *node) {
  assert(node!= NULL && "New ModuloSchedGraphNode is null");
  GraphMap[I] = node;
}

void ModuloSchedGraph::addDepEdges() {
  
  //Get Machine target information for calculating delay
  const TargetInstrInfo &MTI = Target.getInstrInfo();
  
  //Loop over instruction in BB and gather dependencies
  for(BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    
    //Ignore instructions of the void type
    if(I->getType() != Type::VoidTy) {
      
      //Iterate over def-use chain and add true dependencies
      for (Value::use_const_iterator U = I->use_begin(), e = I->use_end(); U != e; 
	   ++U) {
	if (Instruction *Inst = dyn_cast<Instruction>(*U)) {
	  //Check if a node already exists for this instruction
	  ModuloSchedGraph::iterator Sink = find(Inst);
	  
	  //If the instruction is in our graph, add appropriate edges
	  if(Sink->second != NULL) {
	    //assert if self loop
	    assert(&*I == Sink->first && "Use edge to itself!");
	    
	    //Create edge and set delay equal to node latency
	    //FIXME: Is it safe to do this?
	    ModuloSchedGraph::iterator Src = find(I);
	    SchedGraphEdge *trueDep = new SchedGraphEdge(&*Src->second ,&*Sink->second,
							 &*I, SchedGraphEdge::TrueDep,
							 Src->second->getLatency());
	    //Determine the iteration difference
	    //FIXME: Will this ever happen?
	  }
	}
      }
    }
    
  }
  
  
}

void ModuloSchedGraph::ASAP() {


}

void ModuloSchedGraph::ALAP() {


}

void ModuloSchedGraph::MOB() {

}

void ModuloSchedGraph::ComputeDepth() {

}

void  ModuloSchedGraph::ComputeHeight() {

}

void ModuloSchedGraphSet::addGraph(ModuloSchedGraph *graph) {
  assert(graph!=NULL && "Graph for BasicBlock is null");
  Graphs.push_back(graph);
}


ModuloSchedGraphSet::ModuloSchedGraphSet(const Function *F, 
					 const TargetMachine &targ) 
  : function(F) {

  //Create graph for each BB in this function
  for (Function::const_iterator BI = F->begin(); BI != F->end(); ++BI)
    addGraph(new ModuloSchedGraph(BI, targ));
}

ModuloSchedGraphSet::~ModuloSchedGraphSet(){
  
  //delete all the graphs
}

