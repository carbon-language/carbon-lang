//===-- ModuloScheduling.cpp - ModuloScheduling  ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// 
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ModuloSched"

#include "ModuloScheduling.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CFG.h"
#include "llvm/Target/TargetSchedInfo.h"
#include "Support/Debug.h"
#include "Support/GraphWriter.h"
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace llvm;

/// Create ModuloSchedulingPass
///
FunctionPass *llvm::createModuloSchedulingPass(TargetMachine & targ) {
  DEBUG(std::cerr << "Created ModuloSchedulingPass\n");
  return new ModuloSchedulingPass(targ); 
}

template<typename GraphType>
static void WriteGraphToFile(std::ostream &O, const std::string &GraphName,
                             const GraphType &GT) {
  std::string Filename = GraphName + ".dot";
  O << "Writing '" << Filename << "'...";
  std::ofstream F(Filename.c_str());
  
  if (F.good())
    WriteGraph(F, GT);
  else
    O << "  error opening file for writing!";
  O << "\n";
};

namespace llvm {

  template<>
  struct DOTGraphTraits<MSchedGraph*> : public DefaultDOTGraphTraits {
    static std::string getGraphName(MSchedGraph *F) {
      return "Dependence Graph";
    }
    
    static std::string getNodeLabel(MSchedGraphNode *Node, MSchedGraph *Graph) {
      if (Node->getInst()) {
	std::stringstream ss;
	ss << *(Node->getInst());
	return ss.str(); //((MachineInstr*)Node->getInst());
      }
      else
	return "No Inst";
    }
    static std::string getEdgeSourceLabel(MSchedGraphNode *Node,
					  MSchedGraphNode::succ_iterator I) {
      //Label each edge with the type of dependence
      std::string edgelabel = "";
      switch (I.getEdge().getDepOrderType()) {
	
      case MSchedGraphEdge::TrueDep: 
	edgelabel = "True";
	break;
    
      case MSchedGraphEdge::AntiDep: 
	edgelabel =  "Anti";
	break;
	
      case MSchedGraphEdge::OutputDep: 
	edgelabel = "Output";
	break;
	
      default:
	edgelabel = "Unknown";
	break;
      }
      if(I.getEdge().getIteDiff() > 0)
	edgelabel += I.getEdge().getIteDiff();
      
      return edgelabel;
  }



  };
}

/// ModuloScheduling::runOnFunction - main transformation entry point
bool ModuloSchedulingPass::runOnFunction(Function &F) {
  bool Changed = false;

  DEBUG(std::cerr << "Creating ModuloSchedGraph for each BasicBlock in" + F.getName() + "\n");
  
  //Get MachineFunction
  MachineFunction &MF = MachineFunction::get(&F);

  //Iterate over BasicBlocks and do ModuloScheduling if they are valid
  for (MachineFunction::const_iterator BI = MF.begin(); BI != MF.end(); ++BI) {
    if(MachineBBisValid(BI)) {
      MSchedGraph *MSG = new MSchedGraph(BI, target);
    
      //Write Graph out to file
      DEBUG(WriteGraphToFile(std::cerr, "dependgraph", MSG));

      //Print out BB for debugging
      DEBUG(BI->print(std::cerr));

      //Calculate Resource II
      int ResMII = calculateResMII(BI);
  
      calculateNodeAttributes(MSG, ResMII);
    
    }
  }


  return Changed;
}


bool ModuloSchedulingPass::MachineBBisValid(const MachineBasicBlock *BI) {

  //Valid basic blocks must be loops and can not have if/else statements or calls.
  bool isLoop = false;
  
  //Check first if its a valid loop
  for(succ_const_iterator I = succ_begin(BI->getBasicBlock()), 
	E = succ_end(BI->getBasicBlock()); I != E; ++I) {
    if (*I == BI->getBasicBlock())    // has single block loop
      isLoop = true;
  }
  
  if(!isLoop) {
    DEBUG(std::cerr << "Basic Block is not a loop\n");
    return false;
  }
  else 
    DEBUG(std::cerr << "Basic Block is a loop\n");
  
  //Get Target machine instruction info
  /*const TargetInstrInfo& TMI = targ.getInstrInfo();
    
  //Check each instruction and look for calls or if/else statements
  unsigned count = 0;
  for(MachineBasicBlock::const_iterator I = BI->begin(), E = BI->end(); I != E; ++I) {
  //Get opcode to check instruction type
  MachineOpCode OC = I->getOpcode();
  if(TMI.isControlFlow(OC) && (count+1 < BI->size()))
  return false;
  count++;
  }*/
  return true;

}

//ResMII is calculated by determining the usage count for each resource
//and using the maximum.
//FIXME: In future there should be a way to get alternative resources
//for each instruction
int ModuloSchedulingPass::calculateResMII(const MachineBasicBlock *BI) {
  
  const TargetInstrInfo & mii = target.getInstrInfo();
  const TargetSchedInfo & msi = target.getSchedInfo();

  int ResMII = 0;
  
  //Map to keep track of usage count of each resource
  std::map<unsigned, unsigned> resourceUsageCount;

  for(MachineBasicBlock::const_iterator I = BI->begin(), E = BI->end(); I != E; ++I) {

    //Get resource usage for this instruction
    InstrRUsage rUsage = msi.getInstrRUsage(I->getOpcode());
    std::vector<std::vector<resourceId_t> > resources = rUsage.resourcesByCycle;

    //Loop over resources in each cycle and increments their usage count
    for(unsigned i=0; i < resources.size(); ++i)
      for(unsigned j=0; j < resources[i].size(); ++j) {
	if( resourceUsageCount.find(resources[i][j]) == resourceUsageCount.end()) {
	  resourceUsageCount[resources[i][j]] = 1;
	}
	else {
	  resourceUsageCount[resources[i][j]] =  resourceUsageCount[resources[i][j]] + 1;
	}
      }
  }

  //Find maximum usage count
  
  //Get max number of instructions that can be issued at once.
  int issueSlots = msi.maxNumIssueTotal;

  for(std::map<unsigned,unsigned>::iterator RB = resourceUsageCount.begin(), RE = resourceUsageCount.end(); RB != RE; ++RB) {
    //Get the total number of the resources in our cpu
    //int resourceNum = msi.getCPUResourceNum(RB->first);
    
    //Get total usage count for this resources
    unsigned usageCount = RB->second;
    
    //Divide the usage count by either the max number we can issue or the number of
    //resources (whichever is its upper bound)
    double finalUsageCount;
    //if( resourceNum <= issueSlots)
    //finalUsageCount = ceil(1.0 * usageCount / resourceNum);
    //else
      finalUsageCount = ceil(1.0 * usageCount / issueSlots);
    
    
    DEBUG(std::cerr << "Resource ID: " << RB->first << " (usage=" << usageCount << ", resourceNum=X" << ", issueSlots=" << issueSlots << ", finalUsage=" << finalUsageCount << ")\n");

    //Only keep track of the max
    ResMII = std::max( (int) finalUsageCount, ResMII);

  }

  DEBUG(std::cerr << "Final Resource MII: " << ResMII << "\n");
  return ResMII;

}

void ModuloSchedulingPass::calculateNodeAttributes(MSchedGraph *graph, int MII) {

  //Loop over the nodes and add them to the map
  for(MSchedGraph::iterator I = graph->begin(), E = graph->end(); I != E; ++I) {
    //Assert if its already in the map
    assert(nodeToAttributesMap.find(I->second) == nodeToAttributesMap.end() && "Node attributes are already in the map");
    
    //Put into the map with default attribute values
    nodeToAttributesMap[I->second] = MSNodeAttributes();
  }

  //Create set to deal with reccurrences
  std::set<MSchedGraphNode*> visitedNodes;
  std::vector<MSchedGraphNode*> vNodes;
  //Now Loop over map and calculate the node attributes
  for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    // calculateASAP(I->first, (I->second), MII, visitedNodes);
    findAllReccurrences(I->first, vNodes);
    vNodes.clear();
    visitedNodes.clear();
  }
  
  //Calculate ALAP which depends on ASAP being totally calculated
  /*for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    calculateALAP(I->first, (I->second), MII, MII, visitedNodes);
    visitedNodes.clear();
  }*/

  //Calculate MOB which depends on ASAP being totally calculated, also do depth and height
  /*for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    (I->second).MOB = (I->second).ALAP - (I->second).ASAP;
    DEBUG(std::cerr << "MOB: " << (I->second).MOB << " (" << *(I->first) << ")\n");
    calculateDepth(I->first, (I->second), visitedNodes);
    visitedNodes.clear();
    calculateHeight(I->first, (I->second), visitedNodes);
    visitedNodes.clear();
  }*/


}

void ModuloSchedulingPass::calculateASAP(MSchedGraphNode *node, MSNodeAttributes &attributes, 
					 int MII, std::set<MSchedGraphNode*> &visitedNodes) {
    
  DEBUG(std::cerr << "Calculating ASAP for " << *node << "\n");

  if(attributes.ASAP != -1 || (visitedNodes.find(node) != visitedNodes.end())) {
    visitedNodes.erase(node);
    return;
  }
  if(node->hasPredecessors()) {
    int maxPredValue = 0;
    
    //Iterate over all of the predecessors and fine max
    for(MSchedGraphNode::pred_iterator P = node->pred_begin(), E = node->pred_end(); P != E; ++P) {

      //Get that nodes ASAP
      MSNodeAttributes predAttributes = nodeToAttributesMap.find(*P)->second;
      if(predAttributes.ASAP == -1) {
	//Put into set before you recurse
	visitedNodes.insert(node);
	calculateASAP(*P, predAttributes, MII, visitedNodes);
	predAttributes = nodeToAttributesMap.find(*P)->second;
      }
      int iteDiff = node->getInEdge(*P).getIteDiff();

      int currentPredValue = predAttributes.ASAP + node->getLatency() - iteDiff * MII;
      DEBUG(std::cerr << "Current ASAP pred: " << currentPredValue << "\n");
      maxPredValue = std::max(maxPredValue, currentPredValue);
    }
    visitedNodes.erase(node);
    attributes.ASAP = maxPredValue;
  }
  else {
    visitedNodes.erase(node);
    attributes.ASAP = 0;
  }

  DEBUG(std::cerr << "ASAP: " << attributes.ASAP << " (" << *node << ")\n");
}


void ModuloSchedulingPass::calculateALAP(MSchedGraphNode *node, MSNodeAttributes &attributes, 
					 int MII, int maxASAP, 
					 std::set<MSchedGraphNode*> &visitedNodes) {
  
  DEBUG(std::cerr << "Calculating AlAP for " << *node << "\n");
  
  if(attributes.ALAP != -1|| (visitedNodes.find(node) != visitedNodes.end())) {
   visitedNodes.erase(node);
   return;
  }
  if(node->hasSuccessors()) {
    int minSuccValue = 0;
    
    //Iterate over all of the predecessors and fine max
    for(MSchedGraphNode::succ_iterator P = node->succ_begin(), 
	  E = node->succ_end(); P != E; ++P) {

      MSNodeAttributes succAttributes = nodeToAttributesMap.find(*P)->second;
      if(succAttributes.ASAP == -1) {
	
	//Put into set before recursing
	visitedNodes.insert(node);

	calculateALAP(*P, succAttributes, MII, maxASAP, visitedNodes);
	succAttributes = nodeToAttributesMap.find(*P)->second;
	assert(succAttributes.ASAP == -1 && "Successors ALAP should have been caclulated");
      }
      int iteDiff = P.getEdge().getIteDiff();
      int currentSuccValue = succAttributes.ALAP + node->getLatency() + iteDiff * MII;
      minSuccValue = std::min(minSuccValue, currentSuccValue);
    }
    visitedNodes.erase(node);
    attributes.ALAP = minSuccValue;
  }
  else {
    visitedNodes.erase(node);
    attributes.ALAP = maxASAP;
  }
  DEBUG(std::cerr << "ALAP: " << attributes.ALAP << " (" << *node << ")\n");
}

int ModuloSchedulingPass::findMaxASAP() {
  int maxASAP = 0;

  for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I = nodeToAttributesMap.begin(),
	E = nodeToAttributesMap.end(); I != E; ++I)
    maxASAP = std::max(maxASAP, I->second.ASAP);
  return maxASAP;
}


void ModuloSchedulingPass::calculateHeight(MSchedGraphNode *node, 
					   MSNodeAttributes &attributes,
					   std::set<MSchedGraphNode*> &visitedNodes) {

  if(attributes.depth != -1 || (visitedNodes.find(node) != visitedNodes.end())) {
    //Remove from map before returning
    visitedNodes.erase(node);
    return;
  }

  if(node->hasSuccessors()) {
    int maxHeight = 0;
    
    //Iterate over all of the predecessors and fine max
    for(MSchedGraphNode::succ_iterator P = node->succ_begin(), 
	  E = node->succ_end(); P != E; ++P) {

      MSNodeAttributes succAttributes = nodeToAttributesMap.find(*P)->second;
      if(succAttributes.height == -1) {
	
	//Put into map before recursing
	visitedNodes.insert(node);

	calculateHeight(*P, succAttributes, visitedNodes);
	succAttributes = nodeToAttributesMap.find(*P)->second;
	assert(succAttributes.height == -1 && "Successors Height should have been caclulated");
      }
      int currentHeight = succAttributes.height + node->getLatency();
      maxHeight = std::max(maxHeight, currentHeight);
    }
    visitedNodes.erase(node);
    attributes.height = maxHeight;
  }
  else {
    visitedNodes.erase(node);
    attributes.height = 0;
  }

    DEBUG(std::cerr << "Height: " << attributes.height << " (" << *node << ")\n");
}


void ModuloSchedulingPass::calculateDepth(MSchedGraphNode *node, 
					  MSNodeAttributes &attributes, 
					  std::set<MSchedGraphNode*> &visitedNodes) {
  
  if(attributes.depth != -1 || (visitedNodes.find(node) != visitedNodes.end())) {
    //Remove from map before returning
    visitedNodes.erase(node);
    return;
  }

  if(node->hasPredecessors()) {
    int maxDepth = 0;
    
    //Iterate over all of the predecessors and fine max
    for(MSchedGraphNode::pred_iterator P = node->pred_begin(), E = node->pred_end(); P != E; ++P) {

      //Get that nodes depth
      MSNodeAttributes predAttributes = nodeToAttributesMap.find(*P)->second;
      if(predAttributes.depth == -1) {
	
	//Put into set before recursing
	visitedNodes.insert(node);
	
	calculateDepth(*P, predAttributes, visitedNodes);
	predAttributes = nodeToAttributesMap.find(*P)->second;
	assert(predAttributes.depth == -1 && "Predecessors ASAP should have been caclulated");
      }
      int currentDepth = predAttributes.depth + node->getLatency();
      maxDepth = std::max(maxDepth, currentDepth);
    }

    //Remove from map before returning
    visitedNodes.erase(node);
   
    attributes.height = maxDepth;
  }
  else {
    //Remove from map before returning
    visitedNodes.erase(node);
    attributes.depth = 0;
  }

  DEBUG(std::cerr << "Depth: " << attributes.depth << " (" << *node << "*)\n");

}


void ModuloSchedulingPass::findAllReccurrences(MSchedGraphNode *node, 
					       std::vector<MSchedGraphNode*> &visitedNodes) {
  
  if(find(visitedNodes.begin(), visitedNodes.end(), node) != visitedNodes.end()) {
    //DUMP out recurrence
    DEBUG(std::cerr << "Reccurrence:\n");
    bool first = true;
    for(std::vector<MSchedGraphNode*>::iterator I = visitedNodes.begin(), E = visitedNodes.end();
	I !=E; ++I) {
      if(*I == node)
	first = false;
      if(first)
	continue;
      DEBUG(std::cerr << **I << "\n");
    }
     DEBUG(std::cerr << "End Reccurrence:\n");
    return;
  }

  for(MSchedGraphNode::succ_iterator I = node->succ_begin(), E = node->succ_end(); I != E; ++I) {
    visitedNodes.push_back(node);
    findAllReccurrences(*I, visitedNodes);
    visitedNodes.pop_back();
  }

}









void ModuloSchedulingPass::orderNodes() {
  
  int BOTTOM_UP = 0;
  int TOP_DOWN = 1;

  //FIXME: Group nodes into sets and order all the sets based on RecMII
  typedef std::vector<MSchedGraphNode*> NodeVector;
  typedef std::pair<int, NodeVector> NodeSet; 
  
  std::vector<NodeSet> NodeSetsToOrder;
  
  //Order the resulting sets
  NodeVector FinalNodeOrder;

  //Loop over all the sets and place them in the final node order
  for(unsigned i=0; i < NodeSetsToOrder.size(); ++i) {

    //Set default order
    int order = BOTTOM_UP;

    //Get Nodes in Current set
    NodeVector CurrentSet = NodeSetsToOrder[i].second;

    //Loop through the predecessors for each node in the final order
    //and only keeps nodes both in the pred_set and currentset
    NodeVector IntersectCurrent;

    //Sort CurrentSet so we can use lowerbound
    sort(CurrentSet.begin(), CurrentSet.end());

    for(unsigned j=0; j < FinalNodeOrder.size(); ++j) {
      for(MSchedGraphNode::pred_iterator P = FinalNodeOrder[j]->pred_begin(), 
	    E = FinalNodeOrder[j]->pred_end(); P != E; ++P) {
	if(lower_bound(CurrentSet.begin(), 
		       CurrentSet.end(), *P) != CurrentSet.end())
	  IntersectCurrent.push_back(*P);
      }
    }

    //If the intersection of predecessor and current set is not empty
    //sort nodes bottom up
    if(IntersectCurrent.size() != 0)
      order = BOTTOM_UP;
    
    //If empty, use successors
    else {

      for(unsigned j=0; j < FinalNodeOrder.size(); ++j) {
	for(MSchedGraphNode::succ_iterator P = FinalNodeOrder[j]->succ_begin(), 
	      E = FinalNodeOrder[j]->succ_end(); P != E; ++P) {
	  if(lower_bound(CurrentSet.begin(), 
			 CurrentSet.end(), *P) != CurrentSet.end())
	    IntersectCurrent.push_back(*P);
	}
      }

      //sort top-down
      if(IntersectCurrent.size() != 0)
	order = TOP_DOWN;

      else {
	//Find node with max ASAP in current Set
	MSchedGraphNode *node;
	int maxASAP = 0;
	for(unsigned j=0; j < CurrentSet.size(); ++j) {
	  //Get node attributes
	  MSNodeAttributes nodeAttr= nodeToAttributesMap.find(CurrentSet[j])->second;
	  //assert(nodeAttr != nodeToAttributesMap.end() && "Node not in attributes map!");
      
	  if(maxASAP < nodeAttr.ASAP) {
	    maxASAP = nodeAttr.ASAP;
	    node = CurrentSet[j];
	  }
	}
	order = BOTTOM_UP;
      }
    }
      
    //Repeat until all nodes are put into the final order from current set
    /*while(IntersectCurrent.size() > 0) {
      
      if(order == TOP_DOWN) {
	while(IntersectCurrent.size() > 0) {

	  //FIXME
	  //Get node attributes
	  MSNodeAttributes nodeAttr= nodeToAttributesMap.find(IntersectCurrent[0])->second;
	  assert(nodeAttr != nodeToAttributesMap.end() && "Node not in attributes map!");

	  //Get node with highest height, if a tie, use one with lowest
	  //MOB
	  int MOB = nodeAttr.MBO;
	  int height = nodeAttr.height;
	  ModuloSchedGraphNode *V = IntersectCurrent[0];

	  for(unsigned j=0; j < IntersectCurrent.size(); ++j) {
	    int temp = IntersectCurrent[j]->getHeight();
	    if(height < temp) {
	      V = IntersectCurrent[j];
	      height = temp;
	      MOB = V->getMobility();
	    }
	    else if(height == temp) {
	      if(MOB > IntersectCurrent[j]->getMobility()) {
		V = IntersectCurrent[j];
		height = temp;
		MOB = V->getMobility();
	      }
	    }
	  }
	  
	  //Append V to the NodeOrder
	  NodeOrder.push_back(V);

	  //Remove V from IntersectOrder
	  IntersectCurrent.erase(find(IntersectCurrent.begin(), 
				      IntersectCurrent.end(), V));

	  //Intersect V's successors with CurrentSet
	  for(mod_succ_iterator P = succ_begin(V), 
		E = succ_end(V); P != E; ++P) {
	    if(lower_bound(CurrentSet.begin(), 
			   CurrentSet.end(), *P) != CurrentSet.end()) {
	      //If not already in Intersect, add
	      if(find(IntersectCurrent.begin(), IntersectCurrent.end(), *P) == IntersectCurrent.end())
		IntersectCurrent.push_back(*P);
	    }
	  }
     	} //End while loop over Intersect Size

	//Change direction
	order = BOTTOM_UP;

	//Reset Intersect to reflect changes in OrderNodes
	IntersectCurrent.clear();
	for(unsigned j=0; j < NodeOrder.size(); ++j) {
	  for(mod_pred_iterator P = pred_begin(NodeOrder[j]), 
		E = pred_end(NodeOrder[j]); P != E; ++P) {
	    if(lower_bound(CurrentSet.begin(), 
			   CurrentSet.end(), *P) != CurrentSet.end())
	      IntersectCurrent.push_back(*P);
	  }
	}
      } //End If TOP_DOWN
	
	//Begin if BOTTOM_UP
	else {
	  while(IntersectCurrent.size() > 0) {
	    //Get node with highest depth, if a tie, use one with lowest
	    //MOB
	    int MOB = IntersectCurrent[0]->getMobility();
	    int depth = IntersectCurrent[0]->getDepth();
	    ModuloSchedGraphNode *V = IntersectCurrent[0];
	    
	    for(unsigned j=0; j < IntersectCurrent.size(); ++j) {
	      int temp = IntersectCurrent[j]->getDepth();
	      if(depth < temp) {
		V = IntersectCurrent[j];
		depth = temp;
		MOB = V->getMobility();
	      }
	      else if(depth == temp) {
		if(MOB > IntersectCurrent[j]->getMobility()) {
		  V = IntersectCurrent[j];
		  depth = temp;
		  MOB = V->getMobility();
		}
	      }
	    }
	    
	    //Append V to the NodeOrder
	    NodeOrder.push_back(V);
	    
	    //Remove V from IntersectOrder
	    IntersectCurrent.erase(find(IntersectCurrent.begin(), 
					IntersectCurrent.end(),V));
	    
	    //Intersect V's pred with CurrentSet
	    for(mod_pred_iterator P = pred_begin(V), 
		  E = pred_end(V); P != E; ++P) {
	      if(lower_bound(CurrentSet.begin(), 
			     CurrentSet.end(), *P) != CurrentSet.end()) {
		//If not already in Intersect, add
		if(find(IntersectCurrent.begin(), IntersectCurrent.end(), *P) == IntersectCurrent.end())
		  IntersectCurrent.push_back(*P);
	      }
	    }
	  } //End while loop over Intersect Size
	  
	  //Change order
	  order = TOP_DOWN;
	  
	  //Reset IntersectCurrent to reflect changes in OrderNodes
	  IntersectCurrent.clear();
	  for(unsigned j=0; j < NodeOrder.size(); ++j) {
	    for(mod_succ_iterator P = succ_begin(NodeOrder[j]), 
		  E = succ_end(NodeOrder[j]); P != E; ++P) {
	      if(lower_bound(CurrentSet.begin(), 
			     CurrentSet.end(), *P) != CurrentSet.end())
		IntersectCurrent.push_back(*P);
	    }
	    
	  }
	} //End if BOTTOM_DOWN
	
	}*/
//End Wrapping while loop
      
    }//End for over all sets of nodes
   
    //Return final Order
    //return FinalNodeOrder;
}
