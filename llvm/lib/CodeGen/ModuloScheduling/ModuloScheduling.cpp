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
#include "Support/StringExtras.h"
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

      //FIXME
      int iteDiff = I.getEdge().getIteDiff();
      std::string intStr = "(IteDiff: ";
      intStr += itostr(iteDiff);

      intStr += ")";
      edgelabel += intStr;

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
      DEBUG(WriteGraphToFile(std::cerr, F.getName(), MSG));

      //Print out BB for debugging
      DEBUG(BI->print(std::cerr));

      //Calculate Resource II
      int ResMII = calculateResMII(BI);
  
      //Calculate Recurrence II
      int RecMII = calculateRecMII(MSG, ResMII);

      II = std::max(RecMII, ResMII);

      DEBUG(std::cerr << "II starts out as " << II << "\n");

      //Calculate Node Properties
      calculateNodeAttributes(MSG, ResMII);

      //Dump node properties if in debug mode
      for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I =  nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I !=E; ++I) {
	DEBUG(std::cerr << "Node: " << *(I->first) << " ASAP: " << I->second.ASAP << " ALAP: " << I->second.ALAP << " MOB: " << I->second.MOB << " Depth: " << I->second.depth << " Height: " << I->second.height << "\n");
      }
    
      //Put nodes in order to schedule them
      computePartialOrder();

      //Dump out partial order
      for(std::vector<std::vector<MSchedGraphNode*> >::iterator I = partialOrder.begin(), E = partialOrder.end(); I !=E; ++I) {
	DEBUG(std::cerr << "Start set in PO\n");
	for(std::vector<MSchedGraphNode*>::iterator J = I->begin(), JE = I->end(); J != JE; ++J)
	  DEBUG(std::cerr << "PO:" << **J << "\n");
      }

      orderNodes();

      //Dump out order of nodes
      for(std::vector<MSchedGraphNode*>::iterator I = FinalNodeOrder.begin(), E = FinalNodeOrder.end(); I != E; ++I)
	DEBUG(std::cerr << "FO:" << **I << "\n");


      //Finally schedule nodes
      computeSchedule();


      //Dump out final schedule
      //std::cerr << "FINALSCHEDULE\n";
  //Dump out current schedule
  /*for(std::map<unsigned, std::vector<std::pair<unsigned, MSchedGraphNode*> > >::iterator J = schedule.begin(), 
	JE = schedule.end(); J != JE; ++J) {
    std::cerr << "Cycle " << J->first << ":\n";
    for(std::vector<std::pair<unsigned, MSchedGraphNode*> >::iterator VI = J->second.begin(), VE = J->second.end(); VI != VE; ++VI)
      std::cerr << "Resource ID: " << VI->first << " by node " << *(VI->second) << "\n";
  }
  std::cerr << "END FINAL SCHEDULE\n";

      DEBUG(std::cerr << "II ends up as " << II << "\n");
  */  


      nodeToAttributesMap.clear();
      partialOrder.clear();
      recurrenceList.clear();
      FinalNodeOrder.clear();
      schedule.clear();
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
  
  //Get max number of instructions that can be issued at once. (FIXME)
  int issueSlots = 1; // msi.maxNumIssueTotal;

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

int ModuloSchedulingPass::calculateRecMII(MSchedGraph *graph, int MII) {
  std::vector<MSchedGraphNode*> vNodes;
  //Loop over all nodes in the graph
  for(MSchedGraph::iterator I = graph->begin(), E = graph->end(); I != E; ++I) {
    findAllReccurrences(I->second, vNodes, MII);
    vNodes.clear();
  }

  int RecMII = 0;
  
  for(std::set<std::pair<int, std::vector<MSchedGraphNode*> > >::iterator I = recurrenceList.begin(), E=recurrenceList.end(); I !=E; ++I) {
    std::cerr << "Recurrence: \n";
    for(std::vector<MSchedGraphNode*>::const_iterator N = I->second.begin(), NE = I->second.end(); N != NE; ++N) {
      std::cerr << **N << "\n";
    }
    RecMII = std::max(RecMII, I->first);
    std::cerr << "End Recurrence with RecMII: " << I->first << "\n";
    }
  DEBUG(std::cerr << "RecMII: " << RecMII << "\n");
  
  return MII;
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
  
  //Now Loop over map and calculate the node attributes
  for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    calculateASAP(I->first, MII, (MSchedGraphNode*) 0);
    visitedNodes.clear();
  }
  
  int maxASAP = findMaxASAP();
  //Calculate ALAP which depends on ASAP being totally calculated
  for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    calculateALAP(I->first, MII, maxASAP, (MSchedGraphNode*) 0);
    visitedNodes.clear();
  }

  //Calculate MOB which depends on ASAP being totally calculated, also do depth and height
  for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    (I->second).MOB = std::max(0,(I->second).ALAP - (I->second).ASAP);
   
    DEBUG(std::cerr << "MOB: " << (I->second).MOB << " (" << *(I->first) << ")\n");
    calculateDepth(I->first, (MSchedGraphNode*) 0);
    calculateHeight(I->first, (MSchedGraphNode*) 0);
  }


}

bool ModuloSchedulingPass::ignoreEdge(MSchedGraphNode *srcNode, MSchedGraphNode *destNode) {
  if(destNode == 0 || srcNode ==0)
    return false;

  bool findEdge = edgesToIgnore.count(std::make_pair(srcNode, destNode->getInEdgeNum(srcNode)));
  DEBUG(std::cerr << "Ignore Edge from " << *srcNode << " to " << *destNode << "? " << findEdge << "\n");
  return findEdge;
}

int  ModuloSchedulingPass::calculateASAP(MSchedGraphNode *node, int MII, MSchedGraphNode *destNode) {
    
  DEBUG(std::cerr << "Calculating ASAP for " << *node << "\n");

  //Get current node attributes
  MSNodeAttributes &attributes = nodeToAttributesMap.find(node)->second;

  if(attributes.ASAP != -1)
    return attributes.ASAP;
  
  int maxPredValue = 0;
  
  //Iterate over all of the predecessors and find max
  for(MSchedGraphNode::pred_iterator P = node->pred_begin(), E = node->pred_end(); P != E; ++P) {
    
    //Only process if we are not ignoring the edge
    if(!ignoreEdge(*P, node)) {
      int predASAP = -1;
      predASAP = calculateASAP(*P, MII, node);
    
      assert(predASAP != -1 && "ASAP has not been calculated");
      int iteDiff = node->getInEdge(*P).getIteDiff();
      
      int currentPredValue = predASAP + (*P)->getLatency() - (iteDiff * MII);
      DEBUG(std::cerr << "pred ASAP: " << predASAP << ", iteDiff: " << iteDiff << ", PredLatency: " << (*P)->getLatency() << ", Current ASAP pred: " << currentPredValue << "\n");
      maxPredValue = std::max(maxPredValue, currentPredValue);
    }
  }
  
  attributes.ASAP = maxPredValue;

  DEBUG(std::cerr << "ASAP: " << attributes.ASAP << " (" << *node << ")\n");
  
  return maxPredValue;
}


int ModuloSchedulingPass::calculateALAP(MSchedGraphNode *node, int MII, 
					int maxASAP, MSchedGraphNode *srcNode) {
  
  DEBUG(std::cerr << "Calculating ALAP for " << *node << "\n");
  
  MSNodeAttributes &attributes = nodeToAttributesMap.find(node)->second;
 
  if(attributes.ALAP != -1)
    return attributes.ALAP;
 
  if(node->hasSuccessors()) {
    
    //Trying to deal with the issue where the node has successors, but
    //we are ignoring all of the edges to them. So this is my hack for
    //now.. there is probably a more elegant way of doing this (FIXME)
    bool processedOneEdge = false;

    //FIXME, set to something high to start
    int minSuccValue = 9999999;
    
    //Iterate over all of the predecessors and fine max
    for(MSchedGraphNode::succ_iterator P = node->succ_begin(), 
	  E = node->succ_end(); P != E; ++P) {
      
      //Only process if we are not ignoring the edge
      if(!ignoreEdge(node, *P)) {
	processedOneEdge = true;
	int succALAP = -1;
	succALAP = calculateALAP(*P, MII, maxASAP, node);
	
	assert(succALAP != -1 && "Successors ALAP should have been caclulated");
	
	int iteDiff = P.getEdge().getIteDiff();
	
	int currentSuccValue = succALAP - node->getLatency() + iteDiff * MII;
	
	DEBUG(std::cerr << "succ ALAP: " << succALAP << ", iteDiff: " << iteDiff << ", SuccLatency: " << (*P)->getLatency() << ", Current ALAP succ: " << currentSuccValue << "\n");

	minSuccValue = std::min(minSuccValue, currentSuccValue);
      }
    }
    
    if(processedOneEdge)
    	attributes.ALAP = minSuccValue;
    
    else
      attributes.ALAP = maxASAP;
  }
  else
    attributes.ALAP = maxASAP;

  DEBUG(std::cerr << "ALAP: " << attributes.ALAP << " (" << *node << ")\n");

  if(attributes.ALAP < 0)
    attributes.ALAP = 0;

  return attributes.ALAP;
}

int ModuloSchedulingPass::findMaxASAP() {
  int maxASAP = 0;

  for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I = nodeToAttributesMap.begin(),
	E = nodeToAttributesMap.end(); I != E; ++I)
    maxASAP = std::max(maxASAP, I->second.ASAP);
  return maxASAP;
}


int ModuloSchedulingPass::calculateHeight(MSchedGraphNode *node,MSchedGraphNode *srcNode) {
  
  MSNodeAttributes &attributes = nodeToAttributesMap.find(node)->second;

  if(attributes.height != -1)
    return attributes.height;

  int maxHeight = 0;
    
  //Iterate over all of the predecessors and find max
  for(MSchedGraphNode::succ_iterator P = node->succ_begin(), 
	E = node->succ_end(); P != E; ++P) {
    
    
    if(!ignoreEdge(node, *P)) {
      int succHeight = calculateHeight(*P, node);

      assert(succHeight != -1 && "Successors Height should have been caclulated");

      int currentHeight = succHeight + node->getLatency();
      maxHeight = std::max(maxHeight, currentHeight);
    }
  }
  attributes.height = maxHeight;
  DEBUG(std::cerr << "Height: " << attributes.height << " (" << *node << ")\n");
  return maxHeight;
}


int ModuloSchedulingPass::calculateDepth(MSchedGraphNode *node, 
					  MSchedGraphNode *destNode) {

  MSNodeAttributes &attributes = nodeToAttributesMap.find(node)->second;

  if(attributes.depth != -1)
    return attributes.depth;

  int maxDepth = 0;
      
  //Iterate over all of the predecessors and fine max
  for(MSchedGraphNode::pred_iterator P = node->pred_begin(), E = node->pred_end(); P != E; ++P) {

    if(!ignoreEdge(*P, node)) {
      int predDepth = -1;
      predDepth = calculateDepth(*P, node);
      
      assert(predDepth != -1 && "Predecessors ASAP should have been caclulated");

      int currentDepth = predDepth + (*P)->getLatency();
      maxDepth = std::max(maxDepth, currentDepth);
    }
  }
  attributes.depth = maxDepth;
  
  DEBUG(std::cerr << "Depth: " << attributes.depth << " (" << *node << "*)\n");
  return maxDepth;
}



void ModuloSchedulingPass::addReccurrence(std::vector<MSchedGraphNode*> &recurrence, int II, MSchedGraphNode *srcBENode, MSchedGraphNode *destBENode) {
  //Check to make sure that this recurrence is unique
  bool same = false;


  //Loop over all recurrences already in our list
  for(std::set<std::pair<int, std::vector<MSchedGraphNode*> > >::iterator R = recurrenceList.begin(), RE = recurrenceList.end(); R != RE; ++R) {
    
    bool all_same = true;
     //First compare size
    if(R->second.size() == recurrence.size()) {
      
      for(std::vector<MSchedGraphNode*>::const_iterator node = R->second.begin(), end = R->second.end(); node != end; ++node) {
	if(find(recurrence.begin(), recurrence.end(), *node) == recurrence.end()) {
	  all_same = all_same && false;
	  break;
	}
	else
	  all_same = all_same && true;
      }
      if(all_same) {
	same = true;
	break;
      }
    }
  }
  
  if(!same) {
    //if(srcBENode == 0 || destBENode == 0) {
      srcBENode = recurrence.back();
      destBENode = recurrence.front();
      //}
    DEBUG(std::cerr << "Back Edge to Remove: " << *srcBENode << " to " << *destBENode << "\n");
    edgesToIgnore.insert(std::make_pair(srcBENode, destBENode->getInEdgeNum(srcBENode)));
    recurrenceList.insert(std::make_pair(II, recurrence));
  }
  
}

void ModuloSchedulingPass::findAllReccurrences(MSchedGraphNode *node, 
					       std::vector<MSchedGraphNode*> &visitedNodes,
					       int II) {

  if(find(visitedNodes.begin(), visitedNodes.end(), node) != visitedNodes.end()) {
    std::vector<MSchedGraphNode*> recurrence;
    bool first = true;
    int delay = 0;
    int distance = 0;
    int RecMII = II; //Starting value
    MSchedGraphNode *last = node;
    MSchedGraphNode *srcBackEdge;
    MSchedGraphNode *destBackEdge;
    


    for(std::vector<MSchedGraphNode*>::iterator I = visitedNodes.begin(), E = visitedNodes.end();
	I !=E; ++I) {

      if(*I == node) 
	first = false;
      if(first)
	continue;

      delay = delay + (*I)->getLatency();

      if(*I != node) {
	int diff = (*I)->getInEdge(last).getIteDiff();
	distance += diff;
	if(diff > 0) {
	  srcBackEdge = last;
	  destBackEdge = *I;
	}
      }

      recurrence.push_back(*I);
      last = *I;
    }


      
    //Get final distance calc
    distance += node->getInEdge(last).getIteDiff();
   

    //Adjust II until we get close to the inequality delay - II*distance <= 0
    
    int value = delay-(RecMII * distance);
    int lastII = II;
    while(value <= 0) {
      
      lastII = RecMII;
      RecMII--;
      value = delay-(RecMII * distance);
    }
    
    
    DEBUG(std::cerr << "Final II for this recurrence: " << lastII << "\n");
    addReccurrence(recurrence, lastII, srcBackEdge, destBackEdge);
    assert(distance != 0 && "Recurrence distance should not be zero");
    return;
  }

  for(MSchedGraphNode::succ_iterator I = node->succ_begin(), E = node->succ_end(); I != E; ++I) {
    visitedNodes.push_back(node);
    findAllReccurrences(*I, visitedNodes, II);
    visitedNodes.pop_back();
  }
}





void ModuloSchedulingPass::computePartialOrder() {
  
  
  //Loop over all recurrences and add to our partial order
  //be sure to remove nodes that are already in the partial order in
  //a different recurrence and don't add empty recurrences.
  for(std::set<std::pair<int, std::vector<MSchedGraphNode*> > >::reverse_iterator I = recurrenceList.rbegin(), E=recurrenceList.rend(); I !=E; ++I) {
    
    //Add nodes that connect this recurrence to the previous recurrence
    
    //If this is the first recurrence in the partial order, add all predecessors
    for(std::vector<MSchedGraphNode*>::const_iterator N = I->second.begin(), NE = I->second.end(); N != NE; ++N) {

    }


    std::vector<MSchedGraphNode*> new_recurrence;
    //Loop through recurrence and remove any nodes already in the partial order
    for(std::vector<MSchedGraphNode*>::const_iterator N = I->second.begin(), NE = I->second.end(); N != NE; ++N) {
      bool found = false;
      for(std::vector<std::vector<MSchedGraphNode*> >::iterator PO = partialOrder.begin(), PE = partialOrder.end(); PO != PE; ++PO) {
	if(find(PO->begin(), PO->end(), *N) != PO->end())
	  found = true;
      }
      if(!found) {
	new_recurrence.push_back(*N);
	 
	if(partialOrder.size() == 0)
	  //For each predecessors, add it to this recurrence ONLY if it is not already in it
	  for(MSchedGraphNode::pred_iterator P = (*N)->pred_begin(), 
		PE = (*N)->pred_end(); P != PE; ++P) {
	    
	    //Check if we are supposed to ignore this edge or not
	    if(!ignoreEdge(*P, *N))
	      //Check if already in this recurrence
	      if(find(I->second.begin(), I->second.end(), *P) == I->second.end()) {
		//Also need to check if in partial order
		bool predFound = false;
		for(std::vector<std::vector<MSchedGraphNode*> >::iterator PO = partialOrder.begin(), PEND = partialOrder.end(); PO != PEND; ++PO) {
		  if(find(PO->begin(), PO->end(), *P) != PO->end())
		    predFound = true;
		}
		
		if(!predFound)
		  if(find(new_recurrence.begin(), new_recurrence.end(), *P) == new_recurrence.end())
		     new_recurrence.push_back(*P);
		
	      }
	  }
      }
    }

        
    if(new_recurrence.size() > 0)
      partialOrder.push_back(new_recurrence);
  }
  
  //Add any nodes that are not already in the partial order
  std::vector<MSchedGraphNode*> lastNodes;
  for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    bool found = false;
    //Check if its already in our partial order, if not add it to the final vector
    for(std::vector<std::vector<MSchedGraphNode*> >::iterator PO = partialOrder.begin(), PE = partialOrder.end(); PO != PE; ++PO) {
      if(find(PO->begin(), PO->end(), I->first) != PO->end())
	found = true;
    }
    if(!found)
      lastNodes.push_back(I->first);
  }

  if(lastNodes.size() > 0)
    partialOrder.push_back(lastNodes);
  
}


void ModuloSchedulingPass::predIntersect(std::vector<MSchedGraphNode*> &CurrentSet, std::vector<MSchedGraphNode*> &IntersectResult) {
  
  //Sort CurrentSet so we can use lowerbound
  sort(CurrentSet.begin(), CurrentSet.end());
  
  for(unsigned j=0; j < FinalNodeOrder.size(); ++j) {
    for(MSchedGraphNode::pred_iterator P = FinalNodeOrder[j]->pred_begin(), 
	  E = FinalNodeOrder[j]->pred_end(); P != E; ++P) {
   
      //Check if we are supposed to ignore this edge or not
      if(ignoreEdge(*P,FinalNodeOrder[j]))
	continue;
	 
      if(find(CurrentSet.begin(), 
		     CurrentSet.end(), *P) != CurrentSet.end())
	if(find(FinalNodeOrder.begin(), FinalNodeOrder.end(), *P) == FinalNodeOrder.end())
	  IntersectResult.push_back(*P);
    }
  } 
}

void ModuloSchedulingPass::succIntersect(std::vector<MSchedGraphNode*> &CurrentSet, std::vector<MSchedGraphNode*> &IntersectResult) {

  //Sort CurrentSet so we can use lowerbound
  sort(CurrentSet.begin(), CurrentSet.end());
  
  for(unsigned j=0; j < FinalNodeOrder.size(); ++j) {
    for(MSchedGraphNode::succ_iterator P = FinalNodeOrder[j]->succ_begin(), 
	  E = FinalNodeOrder[j]->succ_end(); P != E; ++P) {

      //Check if we are supposed to ignore this edge or not
      if(ignoreEdge(FinalNodeOrder[j],*P))
	continue;

      if(find(CurrentSet.begin(), 
		     CurrentSet.end(), *P) != CurrentSet.end())
	if(find(FinalNodeOrder.begin(), FinalNodeOrder.end(), *P) == FinalNodeOrder.end())
	  IntersectResult.push_back(*P);
    }
  }
}

void dumpIntersection(std::vector<MSchedGraphNode*> &IntersectCurrent) {
  std::cerr << "Intersection (";
  for(std::vector<MSchedGraphNode*>::iterator I = IntersectCurrent.begin(), E = IntersectCurrent.end(); I != E; ++I)
    std::cerr << **I << ", ";
  std::cerr << ")\n";
}



void ModuloSchedulingPass::orderNodes() {
  
  int BOTTOM_UP = 0;
  int TOP_DOWN = 1;

  //Set default order
  int order = BOTTOM_UP;


  //Loop over all the sets and place them in the final node order
  for(std::vector<std::vector<MSchedGraphNode*> >::iterator CurrentSet = partialOrder.begin(), E= partialOrder.end(); CurrentSet != E; ++CurrentSet) {

    DEBUG(std::cerr << "Processing set in S\n");
    dumpIntersection(*CurrentSet);
    //Result of intersection
    std::vector<MSchedGraphNode*> IntersectCurrent;

    predIntersect(*CurrentSet, IntersectCurrent);

    //If the intersection of predecessor and current set is not empty
    //sort nodes bottom up
    if(IntersectCurrent.size() != 0) {
      DEBUG(std::cerr << "Final Node Order Predecessors and Current Set interesection is NOT empty\n");
      order = BOTTOM_UP;
    }
    //If empty, use successors
    else {
      DEBUG(std::cerr << "Final Node Order Predecessors and Current Set interesection is empty\n");

      succIntersect(*CurrentSet, IntersectCurrent);

      //sort top-down
      if(IntersectCurrent.size() != 0) {
	 DEBUG(std::cerr << "Final Node Order Successors and Current Set interesection is NOT empty\n");
	order = TOP_DOWN;
      }
      else {
	DEBUG(std::cerr << "Final Node Order Successors and Current Set interesection is empty\n");
	//Find node with max ASAP in current Set
	MSchedGraphNode *node;
	int maxASAP = 0;
	DEBUG(std::cerr << "Using current set of size " << CurrentSet->size() << "to find max ASAP\n");
	for(unsigned j=0; j < CurrentSet->size(); ++j) {
	  //Get node attributes
	  MSNodeAttributes nodeAttr= nodeToAttributesMap.find((*CurrentSet)[j])->second;
	  //assert(nodeAttr != nodeToAttributesMap.end() && "Node not in attributes map!");
	  DEBUG(std::cerr << "CurrentSet index " << j << "has ASAP: " << nodeAttr.ASAP << "\n");
	  if(maxASAP < nodeAttr.ASAP) {
	    maxASAP = nodeAttr.ASAP;
	    node = (*CurrentSet)[j];
	  }
	}
	assert(node != 0 && "In node ordering node should not be null");
	IntersectCurrent.push_back(node);
	order = BOTTOM_UP;
      }
    }
      
    //Repeat until all nodes are put into the final order from current set
    while(IntersectCurrent.size() > 0) {

      if(order == TOP_DOWN) {
	DEBUG(std::cerr << "Order is TOP DOWN\n");

	while(IntersectCurrent.size() > 0) {
	  DEBUG(std::cerr << "Intersection is not empty, so find heighest height\n");
	  
	  int MOB = 0;
	  int height = 0;
	  MSchedGraphNode *highestHeightNode = IntersectCurrent[0];
	  	  
	  //Find node in intersection with highest heigh and lowest MOB
	  for(std::vector<MSchedGraphNode*>::iterator I = IntersectCurrent.begin(), 
		E = IntersectCurrent.end(); I != E; ++I) {
	    
	    //Get current nodes properties
	    MSNodeAttributes nodeAttr= nodeToAttributesMap.find(*I)->second;

	    if(height < nodeAttr.height) {
	      highestHeightNode = *I;
	      height = nodeAttr.height;
	      MOB = nodeAttr.MOB;
	    }
	    else if(height ==  nodeAttr.height) {
	      if(MOB > nodeAttr.height) {
		highestHeightNode = *I;
		height =  nodeAttr.height;
		MOB = nodeAttr.MOB;
	      }
	    }
	  }
	  
	  //Append our node with greatest height to the NodeOrder
	  if(find(FinalNodeOrder.begin(), FinalNodeOrder.end(), highestHeightNode) == FinalNodeOrder.end()) {
	    DEBUG(std::cerr << "Adding node to Final Order: " << *highestHeightNode << "\n");
	    FinalNodeOrder.push_back(highestHeightNode);
	  }

	  //Remove V from IntersectOrder
	  IntersectCurrent.erase(find(IntersectCurrent.begin(), 
				      IntersectCurrent.end(), highestHeightNode));


	  //Intersect V's successors with CurrentSet
	  for(MSchedGraphNode::succ_iterator P = highestHeightNode->succ_begin(),
		E = highestHeightNode->succ_end(); P != E; ++P) {
	    //if(lower_bound(CurrentSet->begin(), 
	    //	   CurrentSet->end(), *P) != CurrentSet->end()) {
	    if(find(CurrentSet->begin(), CurrentSet->end(), *P) != CurrentSet->end()) {  
	      if(ignoreEdge(highestHeightNode, *P))
		continue;
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
	predIntersect(*CurrentSet, IntersectCurrent);
	
      } //End If TOP_DOWN
	
	//Begin if BOTTOM_UP
      else {
	DEBUG(std::cerr << "Order is BOTTOM UP\n");
	while(IntersectCurrent.size() > 0) {
	  DEBUG(std::cerr << "Intersection of size " << IntersectCurrent.size() << ", finding highest depth\n");

	  //dump intersection
	  DEBUG(dumpIntersection(IntersectCurrent));
	  //Get node with highest depth, if a tie, use one with lowest
	  //MOB
	  int MOB = 0;
	  int depth = 0;
	  MSchedGraphNode *highestDepthNode = IntersectCurrent[0];
	  
	  for(std::vector<MSchedGraphNode*>::iterator I = IntersectCurrent.begin(), 
		E = IntersectCurrent.end(); I != E; ++I) {
	    //Find node attribute in graph
	    MSNodeAttributes nodeAttr= nodeToAttributesMap.find(*I)->second;
	    
	    if(depth < nodeAttr.depth) {
	      highestDepthNode = *I;
	      depth = nodeAttr.depth;
	      MOB = nodeAttr.MOB;
	    }
	    else if(depth == nodeAttr.depth) {
	      if(MOB > nodeAttr.MOB) {
		highestDepthNode = *I;
		depth = nodeAttr.depth;
		MOB = nodeAttr.MOB;
	      }
	    }
	  }
	  
	  

	  //Append highest depth node to the NodeOrder
	   if(find(FinalNodeOrder.begin(), FinalNodeOrder.end(), highestDepthNode) == FinalNodeOrder.end()) {
	     DEBUG(std::cerr << "Adding node to Final Order: " << *highestDepthNode << "\n");
	     FinalNodeOrder.push_back(highestDepthNode);
	   }
	  //Remove heightestDepthNode from IntersectOrder
	  IntersectCurrent.erase(find(IntersectCurrent.begin(), 
				      IntersectCurrent.end(),highestDepthNode));
	  

	  //Intersect heightDepthNode's pred with CurrentSet
	  for(MSchedGraphNode::pred_iterator P = highestDepthNode->pred_begin(), 
		E = highestDepthNode->pred_end(); P != E; ++P) {
	    //if(lower_bound(CurrentSet->begin(), 
	    //	   CurrentSet->end(), *P) != CurrentSet->end()) {
	    if(find(CurrentSet->begin(), CurrentSet->end(), *P) != CurrentSet->end()) {
	    
	      if(ignoreEdge(*P, highestDepthNode))
		continue;
	    
	    //If not already in Intersect, add
	    if(find(IntersectCurrent.begin(), 
		      IntersectCurrent.end(), *P) == IntersectCurrent.end())
		IntersectCurrent.push_back(*P);
	    }
	  }
	  
	} //End while loop over Intersect Size
	
	  //Change order
	order = TOP_DOWN;
	
	//Reset IntersectCurrent to reflect changes in OrderNodes
	IntersectCurrent.clear();
	succIntersect(*CurrentSet, IntersectCurrent);
	} //End if BOTTOM_DOWN
	
    }
    //End Wrapping while loop
      
  }//End for over all sets of nodes
   
  //Return final Order
  //return FinalNodeOrder;
}

void ModuloSchedulingPass::computeSchedule() {

  bool success = false;
  
  while(!success) {

    //Loop over the final node order and process each node
    for(std::vector<MSchedGraphNode*>::iterator I = FinalNodeOrder.begin(), 
	  E = FinalNodeOrder.end(); I != E; ++I) {
      
      //CalculateEarly and Late start
      int EarlyStart = -1;
      int LateStart = 99999; //Set to something higher then we would ever expect (FIXME)
      bool hasSucc = false;
      bool hasPred = false;
      std::set<MSchedGraphNode*> seenNodes;

      for(std::map<unsigned, std::vector<std::pair<unsigned, std::vector<MSchedGraphNode*> > > >::iterator J = schedule.begin(), 
	    JE = schedule.end(); J != JE; ++J) {
	
	//For each resource with nodes scheduled, loop over the nodes and see if they
	//are a predecessor or successor of this current node we are trying
	//to schedule.
	for(std::vector<std::pair<unsigned, std::vector<MSchedGraphNode*> > >::iterator schedNodeVec = J->second.begin(), SNE = J->second.end(); schedNodeVec != SNE; ++schedNodeVec) {
	  
	  for(std::vector<MSchedGraphNode*>::iterator schedNode = schedNodeVec->second.begin(), schedNodeEnd = schedNodeVec->second.end(); schedNode != schedNodeEnd; ++schedNode) {
	    if((*I)->isPredecessor(*schedNode) && !seenNodes.count(*schedNode)) {
	      if(!ignoreEdge(*schedNode, *I)) {
		int diff = (*I)->getInEdge(*schedNode).getIteDiff();
		int ES_Temp = J->first + (*schedNode)->getLatency() - diff * II;
		DEBUG(std::cerr << "Diff: " << diff << " Cycle: " << J->first << "\n");
		DEBUG(std::cerr << "Temp EarlyStart: " << ES_Temp << " Prev EarlyStart: " << EarlyStart << "\n");
		EarlyStart = std::max(EarlyStart, ES_Temp);
		hasPred = true;
	      }
	    }
	    if((*I)->isSuccessor(*schedNode) && !seenNodes.count(*schedNode)) {
	      if(!ignoreEdge(*I,*schedNode)) {
		int diff = (*schedNode)->getInEdge(*I).getIteDiff();
		int LS_Temp = J->first - (*I)->getLatency() + diff * II;
		DEBUG(std::cerr << "Diff: " << diff << " Cycle: " << J->first << "\n");
		DEBUG(std::cerr << "Temp LateStart: " << LS_Temp << " Prev LateStart: " << LateStart << "\n");
		LateStart = std::min(LateStart, LS_Temp);
		hasSucc = true;
	      }
	    }
	    seenNodes.insert(*schedNode);
	  }
	}
      }
      seenNodes.clear();
      
      DEBUG(std::cerr << "Has Successors: " << hasSucc << ", Has Pred: " << hasPred << "\n");
      DEBUG(std::cerr << "EarlyStart: " << EarlyStart << ", LateStart: " << LateStart << "\n");

      //Check if the node has no pred or successors and set Early Start to its ASAP
      if(!hasSucc && !hasPred)
	EarlyStart = nodeToAttributesMap.find(*I)->second.ASAP;
      
      //Now, try to schedule this node depending upon its pred and successor in the schedule
      //already
      if(!hasSucc && hasPred)
	success = scheduleNode(*I, EarlyStart, (EarlyStart + II -1));
      else if(!hasPred && hasSucc)
	success = scheduleNode(*I, LateStart, (LateStart - II +1));
      else if(hasPred && hasSucc)
	success = scheduleNode(*I, EarlyStart, std::min(LateStart, (EarlyStart + II -1)));
      else
	success = scheduleNode(*I, EarlyStart, EarlyStart + II - 1);
      
      if(!success) {
	++II; 
	schedule.clear();
	break;
      }
     
    }
  } 
}


bool ModuloSchedulingPass::scheduleNode(MSchedGraphNode *node, 
				      int start, int end) {
  bool success = false;

  DEBUG(std::cerr << *node << " (Start Cycle: " << start << ", End Cycle: " << end << ")\n");

  /*std::cerr << "CURRENT SCHEDULE\n";
  //Dump out current schedule
  for(std::map<unsigned, std::vector<std::pair<unsigned, MSchedGraphNode*> > >::iterator J = schedule.begin(), 
	JE = schedule.end(); J != JE; ++J) {
    std::cerr << "Cycle " << J->first << ":\n";
    for(std::vector<std::pair<unsigned, MSchedGraphNode*> >::iterator VI = J->second.begin(), VE = J->second.end(); VI != VE; ++VI)
      std::cerr << "Resource ID: " << VI->first << " by node " << *(VI->second) << "\n";
  }
  std::cerr << "END CURRENT SCHEDULE\n";
  */

  //Make sure start and end are not negative
  if(start < 0)
    start = 0;
  if(end < 0)
    end = 0;

  bool forward = true;
  if(start > end)
    forward = false;

  const TargetSchedInfo & msi = target.getSchedInfo();

  bool increaseSC = true;
 
  int cycle = start ;


  while(increaseSC) {
    
    increaseSC = false;

    //Get the resource used by this instruction
    //Get resource usage for this instruction
    InstrRUsage rUsage = msi.getInstrRUsage(node->getInst()->getOpcode());
    std::vector<std::vector<resourceId_t> > resources = rUsage.resourcesByCycle;

    //Loop over each resource and see if we can put it into the schedule
    for(unsigned r=0; r < resources.size(); ++r) {
      unsigned intermediateCycle = cycle + r;
      
      for(unsigned j=0; j < resources[r].size(); ++j) {
	//Put it into the schedule
	DEBUG(std::cerr << "Attempting to put resource " << resources[r][j] << " in schedule at cycle: " << intermediateCycle << "\n");
	
	//Check if resource is free at this cycle
	std::vector<std::pair<unsigned, std::vector<MSchedGraphNode*> > > resourceForCycle = schedule[intermediateCycle]; 
      
	//Vector of nodes using this resource
	std::vector<MSchedGraphNode*> *nodesUsingResource;

	for(std::vector<std::pair<unsigned, std::vector<MSchedGraphNode*> > >::iterator I = resourceForCycle.begin(), E= resourceForCycle.end(); I != E; ++I) {
	
	  if(I->first == resources[r][j]) {
	    //Get the number of available for this resource
	    unsigned numResource = CPUResource::getCPUResource(resources[r][j])->maxNumUsers;
	    nodesUsingResource = &(I->second);

	    //Check that there are enough of this resource, otherwise
	    //we need to increase/decrease the cycle
	    if(I->second.size() >= numResource) {
	      DEBUG(std::cerr << "No open spot for this resource in this cycle\n");
	      increaseSC = true;
	    }
	    break;
		
	  }
	  //safe to put into schedule
	}

	if(increaseSC)
	  break;

	else {
	  DEBUG(std::cerr << "Found spot in schedule\n");
	  //Add node to resource vector
	  if(nodesUsingResource == 0) {
	    nodesUsingResource = new std::vector<MSchedGraphNode*>;
	    resourceForCycle.push_back(std::make_pair(resources[r][j], *nodesUsingResource));
	  }
	  
	  nodesUsingResource->push_back(node);
	  
	  schedule[intermediateCycle] = resourceForCycle;
	}
      }
      if(increaseSC) {
	/*for(unsigned x = 0; x < r; ++x) {
	  unsigned removeCycle = x + start;
	  for(unsigned j=0; j < resources[x].size(); ++j) {
	    std::vector<std::pair<unsigned, MSchedGraphNode*> > resourceForCycle = schedule[removeCycle]; 
	    for(std::vector<std::pair<unsigned,MSchedGraphNode*> >::iterator I = resourceForCycle.begin(), E= resourceForCycle.end(); I != E; ++I) {
	      if(I->first == resources[x][j]) {
		//remove it
		resourceForCycle.erase(I);
	      }
	    }
	    //Put vector back
	    schedule[removeCycle] = resourceForCycle;
	  }
	  }*/
	
	break;
      }
    }
    if(!increaseSC) 
      return true;

    //Increment cycle to try again
    if(forward) {
      ++cycle;
      DEBUG(std::cerr << "Increase cycle: " << cycle << "\n");
      if(cycle > end)
	return false;
    }
    else {
      --cycle;
      DEBUG(std::cerr << "Decrease cycle: " << cycle << "\n");
      if(cycle < end)
	return false;
    }
  }
  return success;
}
