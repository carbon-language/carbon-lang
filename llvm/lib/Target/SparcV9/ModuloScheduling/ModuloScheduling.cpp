//===-- ModuloScheduling.cpp - ModuloScheduling  ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// 
//  This ModuloScheduling pass is based on the Swing Modulo Scheduling 
//  algorithm. 
// 
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ModuloSched"

#include "ModuloScheduling.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CFG.h"
#include "llvm/Target/TargetSchedInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Timer.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include "../MachineCodeForInstruction.h"
#include "../SparcV9TmpInstr.h"
#include "../SparcV9Internals.h"
#include "../SparcV9RegisterInfo.h"
using namespace llvm;

/// Create ModuloSchedulingPass
///
FunctionPass *llvm::createModuloSchedulingPass(TargetMachine & targ) {
  DEBUG(std::cerr << "Created ModuloSchedulingPass\n");
  return new ModuloSchedulingPass(targ); 
}


//Graph Traits for printing out the dependence graph
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


#if 1
#define TIME_REGION(VARNAME, DESC) \
   NamedRegionTimer VARNAME(DESC)
#else
#define TIME_REGION(VARNAME, DESC)
#endif


//Graph Traits for printing out the dependence graph
namespace llvm {
  Statistic<> ValidLoops("modulosched-validLoops", "Number of candidate loops modulo-scheduled");
  Statistic<> MSLoops("modulosched-schedLoops", "Number of loops successfully modulo-scheduled");
  Statistic<> IncreasedII("modulosched-increasedII", "Number of times we had to increase II");
  Statistic<> SingleBBLoops("modulosched-singeBBLoops", "Number of single basic block loops");
  Statistic<> NoSched("modulosched-noSched", "No schedule");
  Statistic<> SameStage("modulosched-sameStage", "Max stage is 0");

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
/// The Swing Modulo Schedule algorithm has three basic steps:
/// 1) Computation and Analysis of the dependence graph
/// 2) Ordering of the nodes
/// 3) Scheduling
/// 
bool ModuloSchedulingPass::runOnFunction(Function &F) {
  
  bool Changed = false;
  int numMS = 0;
  
  DEBUG(std::cerr << "Creating ModuloSchedGraph for each valid BasicBlock in " + F.getName() + "\n");
  
  //Get MachineFunction
  MachineFunction &MF = MachineFunction::get(&F);
 
 
  //Worklist
  std::vector<MachineBasicBlock*> Worklist;
  
  //Iterate over BasicBlocks and put them into our worklist if they are valid
  for (MachineFunction::iterator BI = MF.begin(); BI != MF.end(); ++BI)
    if(MachineBBisValid(BI)) { 
      Worklist.push_back(&*BI);
      ++ValidLoops;
    }
  
  defaultInst = 0;

  DEBUG(if(Worklist.size() == 0) std::cerr << "No single basic block loops in function to ModuloSchedule\n");

  //Iterate over the worklist and perform scheduling
  for(std::vector<MachineBasicBlock*>::iterator BI = Worklist.begin(),  
	BE = Worklist.end(); BI != BE; ++BI) {

    //Print out BB for debugging
    DEBUG(std::cerr << "ModuloScheduling BB: \n"; (*BI)->print(std::cerr));

    //Catch the odd case where we only have TmpInstructions and no real Value*s
    if(!CreateDefMap(*BI)) {
      //Clear out our maps for the next basic block that is processed
      nodeToAttributesMap.clear();
      partialOrder.clear();
      recurrenceList.clear();
      FinalNodeOrder.clear();
      schedule.clear();
      defMap.clear();
      continue;
    }

    MSchedGraph *MSG = new MSchedGraph(*BI, target);
    
    //Write Graph out to file
    DEBUG(WriteGraphToFile(std::cerr, F.getName(), MSG));
    
    //Calculate Resource II
    int ResMII = calculateResMII(*BI);
    
    //Calculate Recurrence II
    int RecMII = calculateRecMII(MSG, ResMII);

    DEBUG(std::cerr << "Number of reccurrences found: " << recurrenceList.size() << "\n");
       
    


    //Our starting initiation interval is the maximum of RecMII and ResMII
    II = std::max(RecMII, ResMII);
    
    //Print out II, RecMII, and ResMII
    DEBUG(std::cerr << "II starts out as " << II << " ( RecMII=" << RecMII << " and ResMII=" << ResMII << ")\n");
    
    //Dump node properties if in debug mode
    DEBUG(for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I =  nodeToAttributesMap.begin(), 
		E = nodeToAttributesMap.end(); I !=E; ++I) {
	    std::cerr << "Node: " << *(I->first) << " ASAP: " << I->second.ASAP << " ALAP: " 
		      << I->second.ALAP << " MOB: " << I->second.MOB << " Depth: " << I->second.depth 
		      << " Height: " << I->second.height << "\n";
	  });

    //Calculate Node Properties
    calculateNodeAttributes(MSG, ResMII);
    
    //Dump node properties if in debug mode
    DEBUG(for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I =  nodeToAttributesMap.begin(), 
		E = nodeToAttributesMap.end(); I !=E; ++I) {
	    std::cerr << "Node: " << *(I->first) << " ASAP: " << I->second.ASAP << " ALAP: " 
		      << I->second.ALAP << " MOB: " << I->second.MOB << " Depth: " << I->second.depth 
		      << " Height: " << I->second.height << "\n";
	  });
    
    //Put nodes in order to schedule them
    computePartialOrder();
    
    //Dump out partial order
    DEBUG(for(std::vector<std::set<MSchedGraphNode*> >::iterator I = partialOrder.begin(), 
		E = partialOrder.end(); I !=E; ++I) {
	    std::cerr << "Start set in PO\n";
	    for(std::set<MSchedGraphNode*>::iterator J = I->begin(), JE = I->end(); J != JE; ++J)
	      std::cerr << "PO:" << **J << "\n";
	  });
    
    //Place nodes in final order
    orderNodes();
    
    //Dump out order of nodes
    DEBUG(for(std::vector<MSchedGraphNode*>::iterator I = FinalNodeOrder.begin(), E = FinalNodeOrder.end(); I != E; ++I) {
	    std::cerr << "FO:" << **I << "\n";
	  });
    
    //Finally schedule nodes
    bool haveSched = computeSchedule();
    
    //Print out final schedule
    DEBUG(schedule.print(std::cerr));
    
    //Final scheduling step is to reconstruct the loop only if we actual have
    //stage > 0
    if(schedule.getMaxStage() != 0 && haveSched) {
      reconstructLoop(*BI);
      ++MSLoops;
      Changed = true;
    }
    else {
      if(!haveSched)
	++NoSched;
      else
	++SameStage;
      DEBUG(std::cerr << "Max stage is 0, so no change in loop or reached cap\n");
    }
    //Clear out our maps for the next basic block that is processed
    nodeToAttributesMap.clear();
    partialOrder.clear();
    recurrenceList.clear();
    FinalNodeOrder.clear();
    schedule.clear();
    defMap.clear();
    //Clean up. Nuke old MachineBB and llvmBB
    //BasicBlock *llvmBB = (BasicBlock*) (*BI)->getBasicBlock();
    //Function *parent = (Function*) llvmBB->getParent();
    //Should't std::find work??
    //parent->getBasicBlockList().erase(std::find(parent->getBasicBlockList().begin(), parent->getBasicBlockList().end(), *llvmBB));
    //parent->getBasicBlockList().erase(llvmBB);
    
    //delete(llvmBB);
    //delete(*BI);
  }
  
  return Changed;
}

bool ModuloSchedulingPass::CreateDefMap(MachineBasicBlock *BI) {
  defaultInst = 0;

  for(MachineBasicBlock::iterator I = BI->begin(), E = BI->end(); I != E; ++I) {
    for(unsigned opNum = 0; opNum < I->getNumOperands(); ++opNum) {
      const MachineOperand &mOp = I->getOperand(opNum);
      if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isDef()) {
	//assert if this is the second def we have seen
	//DEBUG(std::cerr << "Putting " << *(mOp.getVRegValue()) << " into map\n"); 
	assert(!defMap.count(mOp.getVRegValue()) && "Def already in the map");

	defMap[mOp.getVRegValue()] = &*I;
      }
      
      //See if we can use this Value* as our defaultInst
      if(!defaultInst && mOp.getType() == MachineOperand::MO_VirtualRegister) {
	Value *V = mOp.getVRegValue();
	if(!isa<TmpInstruction>(V) && !isa<Argument>(V) && !isa<Constant>(V) && !isa<PHINode>(V))
	  defaultInst = (Instruction*) V;
      }
    }
  }
  
  if(!defaultInst)
    return false;
  
  return true;
  
}
/// This function checks if a Machine Basic Block is valid for modulo
/// scheduling. This means that it has no control flow (if/else or
/// calls) in the block.  Currently ModuloScheduling only works on
/// single basic block loops.
bool ModuloSchedulingPass::MachineBBisValid(const MachineBasicBlock *BI) {

  bool isLoop = false;
  
  //Check first if its a valid loop
  for(succ_const_iterator I = succ_begin(BI->getBasicBlock()), 
	E = succ_end(BI->getBasicBlock()); I != E; ++I) {
    if (*I == BI->getBasicBlock())    // has single block loop
      isLoop = true;
  }
  
  if(!isLoop)
    return false;

  //Check that we have a conditional branch (avoiding MS infinite loops)
  if(BranchInst *b = dyn_cast<BranchInst>(((BasicBlock*) BI->getBasicBlock())->getTerminator()))
    if(b->isUnconditional())
      return false;

  //Check size of our basic block.. make sure we have more then just the terminator in it
  if(BI->getBasicBlock()->size() == 1)
    return false;


  //Increase number of single basic block loops for stats
  ++SingleBBLoops;

  //Get Target machine instruction info
  const TargetInstrInfo *TMI = target.getInstrInfo();
    
  //Check each instruction and look for calls
  for(MachineBasicBlock::const_iterator I = BI->begin(), E = BI->end(); I != E; ++I) {
    //Get opcode to check instruction type
    MachineOpCode OC = I->getOpcode();
    if(TMI->isCall(OC))
      return false;
    //Look for conditional move
    if(OC == V9::MOVRZr || OC == V9::MOVRZi || OC == V9::MOVRLEZr || OC == V9::MOVRLEZi 
       || OC == V9::MOVRLZr || OC == V9::MOVRLZi || OC == V9::MOVRNZr || OC == V9::MOVRNZi
       || OC == V9::MOVRGZr || OC == V9::MOVRGZi || OC == V9::MOVRGEZr 
       || OC == V9::MOVRGEZi || OC == V9::MOVLEr || OC == V9::MOVLEi || OC == V9::MOVLEUr
       || OC == V9::MOVLEUi || OC == V9::MOVFLEr || OC == V9::MOVFLEi
       || OC == V9::MOVNEr || OC == V9::MOVNEi || OC == V9::MOVNEGr || OC == V9::MOVNEGi
       || OC == V9::MOVFNEr || OC == V9::MOVFNEi)
      return false;
  }
  return true;
}

//ResMII is calculated by determining the usage count for each resource
//and using the maximum.
//FIXME: In future there should be a way to get alternative resources
//for each instruction
int ModuloSchedulingPass::calculateResMII(const MachineBasicBlock *BI) {
  
  TIME_REGION(X, "calculateResMII");

  const TargetInstrInfo *mii = target.getInstrInfo();
  const TargetSchedInfo *msi = target.getSchedInfo();

  int ResMII = 0;
  
  //Map to keep track of usage count of each resource
  std::map<unsigned, unsigned> resourceUsageCount;

  for(MachineBasicBlock::const_iterator I = BI->begin(), E = BI->end(); I != E; ++I) {

    //Get resource usage for this instruction
    InstrRUsage rUsage = msi->getInstrRUsage(I->getOpcode());
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
  int issueSlots = msi->maxNumIssueTotal;

  for(std::map<unsigned,unsigned>::iterator RB = resourceUsageCount.begin(), RE = resourceUsageCount.end(); RB != RE; ++RB) {
    
    //Get the total number of the resources in our cpu
    int resourceNum = CPUResource::getCPUResource(RB->first)->maxNumUsers;
    
    //Get total usage count for this resources
    unsigned usageCount = RB->second;
    
    //Divide the usage count by either the max number we can issue or the number of
    //resources (whichever is its upper bound)
    double finalUsageCount;
    if( resourceNum <= issueSlots)
      finalUsageCount = ceil(1.0 * usageCount / resourceNum);
    else
      finalUsageCount = ceil(1.0 * usageCount / issueSlots);
    
    
    //Only keep track of the max
    ResMII = std::max( (int) finalUsageCount, ResMII);

  }

  return ResMII;

}

/// calculateRecMII - Calculates the value of the highest recurrence
/// By value we mean the total latency
int ModuloSchedulingPass::calculateRecMII(MSchedGraph *graph, int MII) {
  /*std::vector<MSchedGraphNode*> vNodes;
  //Loop over all nodes in the graph
  for(MSchedGraph::iterator I = graph->begin(), E = graph->end(); I != E; ++I) {
    findAllReccurrences(I->second, vNodes, MII);
    vNodes.clear();
  }*/
  
  TIME_REGION(X, "calculateRecMII");

  findAllCircuits(graph, MII);
  int RecMII = 0;
  
  for(std::set<std::pair<int, std::vector<MSchedGraphNode*> > >::iterator I = recurrenceList.begin(), E=recurrenceList.end(); I !=E; ++I) {
    RecMII = std::max(RecMII, I->first);
  }
 
  return MII;
}

/// calculateNodeAttributes - The following properties are calculated for
/// each node in the dependence graph: ASAP, ALAP, Depth, Height, and
/// MOB.
void ModuloSchedulingPass::calculateNodeAttributes(MSchedGraph *graph, int MII) {

  TIME_REGION(X, "calculateNodeAttributes");

  assert(nodeToAttributesMap.empty() && "Node attribute map was not cleared");

  //Loop over the nodes and add them to the map
  for(MSchedGraph::iterator I = graph->begin(), E = graph->end(); I != E; ++I) {

    DEBUG(std::cerr << "Inserting node into attribute map: " << *I->second << "\n");

    //Assert if its already in the map
    assert(nodeToAttributesMap.count(I->second) == 0 &&
	   "Node attributes are already in the map");
    
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

/// ignoreEdge - Checks to see if this edge of a recurrence should be ignored or not
bool ModuloSchedulingPass::ignoreEdge(MSchedGraphNode *srcNode, MSchedGraphNode *destNode) {
  if(destNode == 0 || srcNode ==0)
    return false;
  
  bool findEdge = edgesToIgnore.count(std::make_pair(srcNode, destNode->getInEdgeNum(srcNode)));
  
  return findEdge;
}


/// calculateASAP - Calculates the 
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
	if(std::find(recurrence.begin(), recurrence.end(), *node) == recurrence.end()) {
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
    srcBENode = recurrence.back();
    destBENode = recurrence.front();
    
    //FIXME
    if(destBENode->getInEdge(srcBENode).getIteDiff() == 0) {
      //DEBUG(std::cerr << "NOT A BACKEDGE\n");
      //find actual backedge HACK HACK 
      for(unsigned i=0; i< recurrence.size()-1; ++i) {
	if(recurrence[i+1]->getInEdge(recurrence[i]).getIteDiff() == 1) {
	  srcBENode = recurrence[i];
	  destBENode = recurrence[i+1];
	  break;
	}
	  
      }
      
    }
    DEBUG(std::cerr << "Back Edge to Remove: " << *srcBENode << " to " << *destBENode << "\n");
    edgesToIgnore.insert(std::make_pair(srcBENode, destBENode->getInEdgeNum(srcBENode)));
    recurrenceList.insert(std::make_pair(II, recurrence));
  }
  
}

int CircCount;

void ModuloSchedulingPass::unblock(MSchedGraphNode *u, std::set<MSchedGraphNode*> &blocked,
	     std::map<MSchedGraphNode*, std::set<MSchedGraphNode*> > &B) {

  //Unblock u
  DEBUG(std::cerr << "Unblocking: " << *u << "\n");
  blocked.erase(u);

  //std::set<MSchedGraphNode*> toErase;
  while (!B[u].empty()) {
    MSchedGraphNode *W = *B[u].begin();
    B[u].erase(W);
    //toErase.insert(*W);
    DEBUG(std::cerr << "Removed: " << *W << "from B-List\n");
    if(blocked.count(W))
      unblock(W, blocked, B);
  }

}

bool ModuloSchedulingPass::circuit(MSchedGraphNode *v, std::vector<MSchedGraphNode*> &stack, 
	     std::set<MSchedGraphNode*> &blocked, std::vector<MSchedGraphNode*> &SCC, 
	     MSchedGraphNode *s, std::map<MSchedGraphNode*, std::set<MSchedGraphNode*> > &B, 
				   int II, std::map<MSchedGraphNode*, MSchedGraphNode*> &newNodes) {
  bool f = false;
  
  DEBUG(std::cerr << "Finding Circuits Starting with: ( " << v << ")"<< *v << "\n");

  //Push node onto the stack
  stack.push_back(v);

  //block this node
  blocked.insert(v);

  //Loop over all successors of node v that are in the scc, create Adjaceny list
  std::set<MSchedGraphNode*> AkV;
  for(MSchedGraphNode::succ_iterator I = v->succ_begin(), E = v->succ_end(); I != E; ++I) {
    if((std::find(SCC.begin(), SCC.end(), *I) != SCC.end())) {
      AkV.insert(*I);
    }
  }

  for(std::set<MSchedGraphNode*>::iterator I = AkV.begin(), E = AkV.end(); I != E; ++I) {
    if(*I == s) {
      //We have a circuit, so add it to our list
      
      std::vector<MSchedGraphNode*> recc;
      //Dump recurrence for now
      DEBUG(std::cerr << "Starting Recc\n");
	
      int totalDelay = 0;
      int totalDistance = 0;
      MSchedGraphNode *lastN = 0;

      //Loop over recurrence, get delay and distance
      for(std::vector<MSchedGraphNode*>::iterator N = stack.begin(), NE = stack.end(); N != NE; ++N) {
	totalDelay += (*N)->getLatency();
	if(lastN) {
	  totalDistance += (*N)->getInEdge(lastN).getIteDiff();
	}

	//Get the original node
	lastN = *N;
	recc.push_back(newNodes[*N]);

	DEBUG(std::cerr << *lastN << "\n");
      }

      //Get the loop edge
      totalDistance += lastN->getIteDiff(*stack.begin());

      DEBUG(std::cerr << "End Recc\n");
      f = true;
      CircCount++;

      //Insert reccurrence into the list
      DEBUG(std::cerr << "Ignore Edge from: " << *lastN << " to " << **stack.begin() << "\n");
      edgesToIgnore.insert(std::make_pair(newNodes[lastN], newNodes[(*stack.begin())]->getInEdgeNum(newNodes[lastN])));
	
      //Adjust II until we get close to the inequality delay - II*distance <= 0
      int RecMII = II; //Starting value
      int value = totalDelay-(RecMII * totalDistance);
      int lastII = II;
      while(value <= 0) {
	  
	lastII = RecMII;
	RecMII--;
	value = totalDelay-(RecMII * totalDistance);
      }

      recurrenceList.insert(std::make_pair(lastII, recc));

    }
    else if(!blocked.count(*I)) {
      if(circuit(*I, stack, blocked, SCC, s, B, II, newNodes))
	f = true;
    }
    else
      DEBUG(std::cerr << "Blocked: " << **I << "\n");
  }


  if(f) {
    unblock(v, blocked, B);
  }
  else {
    for(std::set<MSchedGraphNode*>::iterator I = AkV.begin(), E = AkV.end(); I != E; ++I) 
      B[*I].insert(v);

  }

  //Pop v
  stack.pop_back();

  return f;

}

void ModuloSchedulingPass::findAllCircuits(MSchedGraph *g, int II) {

  CircCount = 0;

  //Keep old to new node mapping information 
  std::map<MSchedGraphNode*, MSchedGraphNode*> newNodes;

  //copy the graph
  MSchedGraph *MSG = new MSchedGraph(*g, newNodes);

  DEBUG(std::cerr << "Finding All Circuits\n");

  //Set of blocked nodes
  std::set<MSchedGraphNode*> blocked;

  //Stack holding current circuit
  std::vector<MSchedGraphNode*> stack;

  //Map for B Lists
  std::map<MSchedGraphNode*, std::set<MSchedGraphNode*> > B;

  //current node
  MSchedGraphNode *s;


  //Iterate over the graph until its down to one node or empty
  while(MSG->size() > 1) {
    
    //Write Graph out to file
    //WriteGraphToFile(std::cerr, "Graph" + utostr(MSG->size()), MSG);

    DEBUG(std::cerr << "Graph Size: " << MSG->size() << "\n");
    DEBUG(std::cerr << "Finding strong component Vk with least vertex\n");

    //Iterate over all the SCCs in the graph
    std::set<MSchedGraphNode*> Visited;
    std::vector<MSchedGraphNode*> Vk;
    MSchedGraphNode* s = 0;

    //Find scc with the least vertex
    for (MSchedGraph::iterator GI = MSG->begin(), E = MSG->end(); GI != E; ++GI)
      if (Visited.insert(GI->second).second) {
	for (scc_iterator<MSchedGraphNode*> SCCI = scc_begin(GI->second),
	       E = scc_end(GI->second); SCCI != E; ++SCCI) {
	  std::vector<MSchedGraphNode*> &nextSCC = *SCCI;

	  if (Visited.insert(nextSCC[0]).second) {
	    Visited.insert(nextSCC.begin()+1, nextSCC.end());

	    DEBUG(std::cerr << "SCC size: " << nextSCC.size() << "\n");

	    //Ignore self loops
	    if(nextSCC.size() > 1) {
	    
	      //Get least vertex in Vk
	      if(!s) {
		s = nextSCC[0];
		Vk = nextSCC;
	      }

	      for(unsigned i = 0; i < nextSCC.size(); ++i) {
		if(nextSCC[i] < s) {
		  s = nextSCC[i];
		  Vk = nextSCC;
		}
	      }
	    }
	  }
	}
      }
  
  

    //Process SCC
    DEBUG(for(std::vector<MSchedGraphNode*>::iterator N = Vk.begin(), NE = Vk.end();
	      N != NE; ++N) { std::cerr << *((*N)->getInst()); });
    
    //Iterate over all nodes in this scc
    for(std::vector<MSchedGraphNode*>::iterator N = Vk.begin(), NE = Vk.end();
	N != NE; ++N) {
      blocked.erase(*N);
      B[*N].clear();
    }
    if(Vk.size() > 1) {
      circuit(s, stack, blocked, Vk, s, B, II, newNodes);
      
      //Find all nodes up to s and delete them
      std::vector<MSchedGraphNode*> nodesToRemove;
      nodesToRemove.push_back(s);
      for(MSchedGraph::iterator N = MSG->begin(), NE = MSG->end(); N != NE; ++N) {
	if(N->second < s )
	  nodesToRemove.push_back(N->second);
      }
      for(std::vector<MSchedGraphNode*>::iterator N = nodesToRemove.begin(), NE = nodesToRemove.end(); N != NE; ++N) {
	DEBUG(std::cerr << "Deleting Node: " << **N << "\n");
	MSG->deleteNode(*N);
      }
    }
    else
      break;
  }
  DEBUG(std::cerr << "Num Circuits found: " << CircCount << "\n");
}


void ModuloSchedulingPass::findAllReccurrences(MSchedGraphNode *node, 
					       std::vector<MSchedGraphNode*> &visitedNodes,
					       int II) {
  

  if(std::find(visitedNodes.begin(), visitedNodes.end(), node) != visitedNodes.end()) {
    std::vector<MSchedGraphNode*> recurrence;
    bool first = true;
    int delay = 0;
    int distance = 0;
    int RecMII = II; //Starting value
    MSchedGraphNode *last = node;
    MSchedGraphNode *srcBackEdge = 0;
    MSchedGraphNode *destBackEdge = 0;
    


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
    DEBUG(std::cerr << "Reccurrence Distance: " << distance << "\n");

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

  unsigned count = 0;
  for(MSchedGraphNode::succ_iterator I = node->succ_begin(), E = node->succ_end(); I != E; ++I) {
    visitedNodes.push_back(node);
    //if(!edgesToIgnore.count(std::make_pair(node, count)))
    findAllReccurrences(*I, visitedNodes, II);
    visitedNodes.pop_back();
    count++;
  }
}





void ModuloSchedulingPass::computePartialOrder() {

  TIME_REGION(X, "calculatePartialOrder");
  
  //Only push BA branches onto the final node order, we put other branches after it
  //FIXME: Should we really be pushing branches on it a specific order instead of relying
  //on BA being there?
  std::vector<MSchedGraphNode*> branches;
  
  //Loop over all recurrences and add to our partial order
  //be sure to remove nodes that are already in the partial order in
  //a different recurrence and don't add empty recurrences.
  for(std::set<std::pair<int, std::vector<MSchedGraphNode*> > >::reverse_iterator I = recurrenceList.rbegin(), E=recurrenceList.rend(); I !=E; ++I) {
    
    //Add nodes that connect this recurrence to the previous recurrence
    
    //If this is the first recurrence in the partial order, add all predecessors
    for(std::vector<MSchedGraphNode*>::const_iterator N = I->second.begin(), NE = I->second.end(); N != NE; ++N) {

    }


    std::set<MSchedGraphNode*> new_recurrence;
    //Loop through recurrence and remove any nodes already in the partial order
    for(std::vector<MSchedGraphNode*>::const_iterator N = I->second.begin(), NE = I->second.end(); N != NE; ++N) {
      bool found = false;
      for(std::vector<std::set<MSchedGraphNode*> >::iterator PO = partialOrder.begin(), PE = partialOrder.end(); PO != PE; ++PO) {
	if(PO->count(*N))
	  found = true;
      }
      if(!found) {
	if((*N)->isBranch()) {
	  branches.push_back(*N);
	}
	else
	  new_recurrence.insert(*N);
	}
	if(partialOrder.size() == 0)
	  //For each predecessors, add it to this recurrence ONLY if it is not already in it
	  for(MSchedGraphNode::pred_iterator P = (*N)->pred_begin(), 
		PE = (*N)->pred_end(); P != PE; ++P) {
	    
	    //Check if we are supposed to ignore this edge or not
	    if(!ignoreEdge(*P, *N))
	      //Check if already in this recurrence
	      if(std::find(I->second.begin(), I->second.end(), *P) == I->second.end()) {
		//Also need to check if in partial order
		bool predFound = false;
		for(std::vector<std::set<MSchedGraphNode*> >::iterator PO = partialOrder.begin(), PEND = partialOrder.end(); PO != PEND; ++PO) {
		  if(PO->count(*P))
		    predFound = true;
		}
		
		if(!predFound)
		  if(!new_recurrence.count(*P)) {
		    if((*P)->isBranch()) {
		      branches.push_back(*P);
		    }
		    else
		      new_recurrence.insert(*P);
		    
		  }
	      }
	  }
    }
    
    if(new_recurrence.size() > 0)
      partialOrder.push_back(new_recurrence);
  }
  
  //Add any nodes that are not already in the partial order
  //Add them in a set, one set per connected component
  std::set<MSchedGraphNode*> lastNodes;
  for(std::map<MSchedGraphNode*, MSNodeAttributes>::iterator I = nodeToAttributesMap.begin(), E = nodeToAttributesMap.end(); I != E; ++I) {
    bool found = false;
    //Check if its already in our partial order, if not add it to the final vector
    for(std::vector<std::set<MSchedGraphNode*> >::iterator PO = partialOrder.begin(), PE = partialOrder.end(); PO != PE; ++PO) {
      if(PO->count(I->first))
	found = true;
    }
    if(!found) {
      if(I->first->isBranch()) {
	if(std::find(branches.begin(), branches.end(), I->first) == branches.end())
	  branches.push_back(I->first); 
      }
      else
	lastNodes.insert(I->first);
    }
  }

  //Break up remaining nodes that are not in the partial order
  //into their connected compoenents
  while(lastNodes.size() > 0) {
    std::set<MSchedGraphNode*> ccSet;
    connectedComponentSet(*(lastNodes.begin()),ccSet, lastNodes);
    if(ccSet.size() > 0)
      partialOrder.push_back(ccSet);
  }
  //if(lastNodes.size() > 0)
  //partialOrder.push_back(lastNodes);
  
  //Clean up branches by putting them in final order
  std::map<unsigned, MSchedGraphNode*> branchOrder;
  for(std::vector<MSchedGraphNode*>::iterator I = branches.begin(), E = branches.end(); I != E; ++I)
    branchOrder[(*I)->getIndex()] = *I;

  for(std::map<unsigned, MSchedGraphNode*>::reverse_iterator I = branchOrder.rbegin(), E = branchOrder.rend(); I != E; ++I)
    FinalNodeOrder.push_back(I->second);

}


void ModuloSchedulingPass::connectedComponentSet(MSchedGraphNode *node, std::set<MSchedGraphNode*> &ccSet, std::set<MSchedGraphNode*> &lastNodes) {

//Add to final set
if( !ccSet.count(node) && lastNodes.count(node)) {
    lastNodes.erase(node);
if(node->isBranch())
      FinalNodeOrder.push_back(node);
    else
      ccSet.insert(node);
  }
  else
    return;

  //Loop over successors and recurse if we have not seen this node before
  for(MSchedGraphNode::succ_iterator node_succ = node->succ_begin(), end=node->succ_end(); node_succ != end; ++node_succ) {
    connectedComponentSet(*node_succ, ccSet, lastNodes);
  }
  
}

void ModuloSchedulingPass::predIntersect(std::set<MSchedGraphNode*> &CurrentSet, std::set<MSchedGraphNode*> &IntersectResult) {
  
  for(unsigned j=0; j < FinalNodeOrder.size(); ++j) {
    for(MSchedGraphNode::pred_iterator P = FinalNodeOrder[j]->pred_begin(), 
	  E = FinalNodeOrder[j]->pred_end(); P != E; ++P) {
   
      //Check if we are supposed to ignore this edge or not
      if(ignoreEdge(*P,FinalNodeOrder[j]))
	continue;
	 
      if(CurrentSet.count(*P))
	if(std::find(FinalNodeOrder.begin(), FinalNodeOrder.end(), *P) == FinalNodeOrder.end())
	  IntersectResult.insert(*P);
    }
  } 
}


    


void ModuloSchedulingPass::succIntersect(std::set<MSchedGraphNode*> &CurrentSet, std::set<MSchedGraphNode*> &IntersectResult) {

  for(unsigned j=0; j < FinalNodeOrder.size(); ++j) {
    for(MSchedGraphNode::succ_iterator P = FinalNodeOrder[j]->succ_begin(), 
	  E = FinalNodeOrder[j]->succ_end(); P != E; ++P) {

      //Check if we are supposed to ignore this edge or not
      if(ignoreEdge(FinalNodeOrder[j],*P))
	continue;

      if(CurrentSet.count(*P))
	if(std::find(FinalNodeOrder.begin(), FinalNodeOrder.end(), *P) == FinalNodeOrder.end())
	  IntersectResult.insert(*P);
    }
  }
}

void dumpIntersection(std::set<MSchedGraphNode*> &IntersectCurrent) {
  std::cerr << "Intersection (";
  for(std::set<MSchedGraphNode*>::iterator I = IntersectCurrent.begin(), E = IntersectCurrent.end(); I != E; ++I)
    std::cerr << **I << ", ";
  std::cerr << ")\n";
}



void ModuloSchedulingPass::orderNodes() {
  
  TIME_REGION(X, "orderNodes");

  int BOTTOM_UP = 0;
  int TOP_DOWN = 1;

  //Set default order
  int order = BOTTOM_UP;

  
  //Loop over all the sets and place them in the final node order
  for(std::vector<std::set<MSchedGraphNode*> >::iterator CurrentSet = partialOrder.begin(), E= partialOrder.end(); CurrentSet != E; ++CurrentSet) {

    DEBUG(std::cerr << "Processing set in S\n");
    DEBUG(dumpIntersection(*CurrentSet));

    //Result of intersection
    std::set<MSchedGraphNode*> IntersectCurrent;

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
	for(std::set<MSchedGraphNode*>::iterator J = CurrentSet->begin(), JE = CurrentSet->end(); J != JE; ++J) {
	  //Get node attributes
	  MSNodeAttributes nodeAttr= nodeToAttributesMap.find(*J)->second;
	  //assert(nodeAttr != nodeToAttributesMap.end() && "Node not in attributes map!");
	  
	  if(maxASAP <= nodeAttr.ASAP) {
	    maxASAP = nodeAttr.ASAP;
	    node = *J;
	  }
	}
	assert(node != 0 && "In node ordering node should not be null");
	IntersectCurrent.insert(node);
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
	  MSchedGraphNode *highestHeightNode = *(IntersectCurrent.begin());
	  	  
	  //Find node in intersection with highest heigh and lowest MOB
	  for(std::set<MSchedGraphNode*>::iterator I = IntersectCurrent.begin(), 
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
	  if(std::find(FinalNodeOrder.begin(), FinalNodeOrder.end(), highestHeightNode) == FinalNodeOrder.end()) {
	    DEBUG(std::cerr << "Adding node to Final Order: " << *highestHeightNode << "\n");
	    FinalNodeOrder.push_back(highestHeightNode);
	  }

	  //Remove V from IntersectOrder
	  IntersectCurrent.erase(std::find(IntersectCurrent.begin(), 
				      IntersectCurrent.end(), highestHeightNode));


	  //Intersect V's successors with CurrentSet
	  for(MSchedGraphNode::succ_iterator P = highestHeightNode->succ_begin(),
		E = highestHeightNode->succ_end(); P != E; ++P) {
	    //if(lower_bound(CurrentSet->begin(), 
	    //	   CurrentSet->end(), *P) != CurrentSet->end()) {
	    if(std::find(CurrentSet->begin(), CurrentSet->end(), *P) != CurrentSet->end()) {  
	      if(ignoreEdge(highestHeightNode, *P))
		continue;
	      //If not already in Intersect, add
	      if(!IntersectCurrent.count(*P))
		IntersectCurrent.insert(*P);
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
	  MSchedGraphNode *highestDepthNode = *(IntersectCurrent.begin());
	  
	  for(std::set<MSchedGraphNode*>::iterator I = IntersectCurrent.begin(), 
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
	   if(std::find(FinalNodeOrder.begin(), FinalNodeOrder.end(), highestDepthNode) == FinalNodeOrder.end()) {
	     DEBUG(std::cerr << "Adding node to Final Order: " << *highestDepthNode << "\n");
	     FinalNodeOrder.push_back(highestDepthNode);
	   }
	  //Remove heightestDepthNode from IntersectOrder
	   IntersectCurrent.erase(highestDepthNode);
	  

	  //Intersect heightDepthNode's pred with CurrentSet
	  for(MSchedGraphNode::pred_iterator P = highestDepthNode->pred_begin(), 
		E = highestDepthNode->pred_end(); P != E; ++P) {
	    if(CurrentSet->count(*P)) {
	      if(ignoreEdge(*P, highestDepthNode))
		continue;
	    
	    //If not already in Intersect, add
	    if(!IntersectCurrent.count(*P))
	      IntersectCurrent.insert(*P);
	    }
	  }
	  
	} //End while loop over Intersect Size
	
	  //Change order
	order = TOP_DOWN;
	
	//Reset IntersectCurrent to reflect changes in OrderNodes
	IntersectCurrent.clear();
	succIntersect(*CurrentSet, IntersectCurrent);
	} //End if BOTTOM_DOWN
	
      DEBUG(std::cerr << "Current Intersection Size: " << IntersectCurrent.size() << "\n");
    }
    //End Wrapping while loop
    DEBUG(std::cerr << "Ending Size of Current Set: " << CurrentSet->size() << "\n");  
  }//End for over all sets of nodes
  
  //FIXME: As the algorithm stands it will NEVER add an instruction such as ba (with no
  //data dependencies) to the final order. We add this manually. It will always be
  //in the last set of S since its not part of a recurrence
    //Loop over all the sets and place them in the final node order
  std::vector<std::set<MSchedGraphNode*> > ::reverse_iterator LastSet = partialOrder.rbegin();
  for(std::set<MSchedGraphNode*>::iterator CurrentNode = LastSet->begin(), LastNode = LastSet->end();
      CurrentNode != LastNode; ++CurrentNode) {
    if((*CurrentNode)->getInst()->getOpcode() == V9::BA)
      FinalNodeOrder.push_back(*CurrentNode);
  }
  //Return final Order
  //return FinalNodeOrder;
}

bool ModuloSchedulingPass::computeSchedule() {

  TIME_REGION(X, "computeSchedule");

  bool success = false;
  
  //FIXME: Should be set to max II of the original loop
  //Cap II in order to prevent infinite loop
  int capII = 100;

  while(!success) {

    int branchES = II - 1;
    int branchLS = II - 1;
    bool lastBranch = true;

    //Loop over the final node order and process each node
    for(std::vector<MSchedGraphNode*>::iterator I = FinalNodeOrder.begin(), 
	  E = FinalNodeOrder.end(); I != E; ++I) {
      
      //CalculateEarly and Late start
      int EarlyStart = -1;
      int LateStart = 99999; //Set to something higher then we would ever expect (FIXME)
      bool hasSucc = false;
      bool hasPred = false;
      
      if(!(*I)->isBranch()) {
	//Loop over nodes in the schedule and determine if they are predecessors
	//or successors of the node we are trying to schedule
	for(MSSchedule::schedule_iterator nodesByCycle = schedule.begin(), nodesByCycleEnd = schedule.end(); 
	    nodesByCycle != nodesByCycleEnd; ++nodesByCycle) {
	  
	  //For this cycle, get the vector of nodes schedule and loop over it
	  for(std::vector<MSchedGraphNode*>::iterator schedNode = nodesByCycle->second.begin(), SNE = nodesByCycle->second.end(); schedNode != SNE; ++schedNode) {
	    
	    if((*I)->isPredecessor(*schedNode)) {
	      //if(!ignoreEdge(*schedNode, *I)) {
		int diff = (*I)->getInEdge(*schedNode).getIteDiff();
		int ES_Temp = nodesByCycle->first + (*schedNode)->getLatency() - diff * II;
		DEBUG(std::cerr << "Diff: " << diff << " Cycle: " << nodesByCycle->first << "\n");
		DEBUG(std::cerr << "Temp EarlyStart: " << ES_Temp << " Prev EarlyStart: " << EarlyStart << "\n");
		EarlyStart = std::max(EarlyStart, ES_Temp);
		hasPred = true;
		//}
	    }
	    if((*I)->isSuccessor(*schedNode)) {
	      //if(!ignoreEdge(*I,*schedNode)) {
		int diff = (*schedNode)->getInEdge(*I).getIteDiff();
		int LS_Temp = nodesByCycle->first - (*I)->getLatency() + diff * II;
		DEBUG(std::cerr << "Diff: " << diff << " Cycle: " << nodesByCycle->first << "\n");
		DEBUG(std::cerr << "Temp LateStart: " << LS_Temp << " Prev LateStart: " << LateStart << "\n");
		LateStart = std::min(LateStart, LS_Temp);
		hasSucc = true;
		//}
	    }
	  }
	}
      }
      else {
	if(lastBranch) {
	  EarlyStart = branchES;
	  LateStart = branchLS;
	  lastBranch = false;
	  --branchES;
	  branchLS = 0;
	}
	else {
	  EarlyStart = branchLS;
	  LateStart = branchES;
	  assert( (EarlyStart >= 0) && (LateStart >=0) && "EarlyStart and LateStart must be greater then 0"); 
	  --branchES;
	}
	hasPred = 0;
	hasSucc = 1;
      }
 
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
      else if(hasPred && hasSucc) {
	if(EarlyStart > LateStart)
	  success = false;
      	else
	  success = scheduleNode(*I, EarlyStart, std::min(LateStart, (EarlyStart + II -1)));
      }
      else
	success = scheduleNode(*I, EarlyStart, EarlyStart + II - 1);
      
      if(!success) {
	++IncreasedII;
	++II; 
	schedule.clear();
	break;
      }
     
    }

    if(success) {
      DEBUG(std::cerr << "Constructing Schedule Kernel\n");
      success = schedule.constructKernel(II);
      DEBUG(std::cerr << "Done Constructing Schedule Kernel\n");
      if(!success) {
	++IncreasedII;
	++II;
	schedule.clear();
      }
    }
    
    if(II >= capII)
      return false;

    assert(II < capII && "The II should not exceed the original loop number of cycles");
  } 
  return true;
}


bool ModuloSchedulingPass::scheduleNode(MSchedGraphNode *node, 
				      int start, int end) {
  bool success = false;

  DEBUG(std::cerr << *node << " (Start Cycle: " << start << ", End Cycle: " << end << ")\n");

  //Make sure start and end are not negative
  if(start < 0) {
    start = 0;
    
  }
  if(end < 0)
    end = 0;

  bool forward = true;
  if(start > end)
    forward = false;

  bool increaseSC = true;
  int cycle = start ;


  while(increaseSC) {
    
    increaseSC = false;

    increaseSC = schedule.insert(node, cycle);
    
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

void ModuloSchedulingPass::writePrologues(std::vector<MachineBasicBlock *> &prologues, MachineBasicBlock *origBB, std::vector<BasicBlock*> &llvm_prologues, std::map<const Value*, std::pair<const MSchedGraphNode*, int> > &valuesToSave, std::map<Value*, std::map<int, Value*> > &newValues, std::map<Value*, MachineBasicBlock*> &newValLocation) {

  //Keep a map to easily know whats in the kernel
  std::map<int, std::set<const MachineInstr*> > inKernel;
  int maxStageCount = 0;

  //Keep a map of new values we consumed in case they need to be added back
  std::map<Value*, std::map<int, Value*> > consumedValues;

  MSchedGraphNode *branch = 0;
  MSchedGraphNode *BAbranch = 0;

  schedule.print(std::cerr);

  std::vector<MSchedGraphNode*> branches;

  for(MSSchedule::kernel_iterator I = schedule.kernel_begin(), E = schedule.kernel_end(); I != E; ++I) {
    maxStageCount = std::max(maxStageCount, I->second);
    
    //Ignore the branch, we will handle this separately
    if(I->first->isBranch()) {
      branches.push_back(I->first);
      continue;
    }

    //Put int the map so we know what instructions in each stage are in the kernel
    DEBUG(std::cerr << "Inserting instruction " << *(I->first->getInst()) << " into map at stage " << I->second << "\n");
    inKernel[I->second].insert(I->first->getInst());
  }

  //Get target information to look at machine operands
  const TargetInstrInfo *mii = target.getInstrInfo();

 //Now write the prologues
  for(int i = 0; i < maxStageCount; ++i) {
    BasicBlock *llvmBB = new BasicBlock("PROLOGUE", (Function*) (origBB->getBasicBlock()->getParent()));
    MachineBasicBlock *machineBB = new MachineBasicBlock(llvmBB);
  
    DEBUG(std::cerr << "i=" << i << "\n");
    for(int j = 0; j <= i; ++j) {
      for(MachineBasicBlock::const_iterator MI = origBB->begin(), ME = origBB->end(); ME != MI; ++MI) {
	if(inKernel[j].count(&*MI)) {
	  MachineInstr *instClone = MI->clone();
	  machineBB->push_back(instClone);
	  
	  DEBUG(std::cerr << "Cloning: " << *MI << "\n");

	  Instruction *tmp;

	  //After cloning, we may need to save the value that this instruction defines
	  for(unsigned opNum=0; opNum < MI->getNumOperands(); ++opNum) {
	    //get machine operand
	    MachineOperand &mOp = instClone->getOperand(opNum);
	    if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isDef()) {

	      //Check if this is a value we should save
	      if(valuesToSave.count(mOp.getVRegValue())) {
		//Save copy in tmpInstruction
		tmp = new TmpInstruction(mOp.getVRegValue());
		
		//Add TmpInstruction to safe LLVM Instruction MCFI
		MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
		tempMvec.addTemp((Value*) tmp);

		DEBUG(std::cerr << "Value: " << *(mOp.getVRegValue()) << " New Value: " << *tmp << " Stage: " << i << "\n");
		
		newValues[mOp.getVRegValue()][i]= tmp;
		newValLocation[tmp] = machineBB;

		DEBUG(std::cerr << "Machine Instr Operands: " << *(mOp.getVRegValue()) << ", 0, " << *tmp << "\n");
		
		//Create machine instruction and put int machineBB
		MachineInstr *saveValue = BuildMI(machineBB, V9::ORr, 3).addReg(mOp.getVRegValue()).addImm(0).addRegDef(tmp);
		
		DEBUG(std::cerr << "Created new machine instr: " << *saveValue << "\n");
	      }
	    }

	    //We may also need to update the value that we use if its from an earlier prologue
	    if(j != 0) {
	      if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isUse()) {
		if(newValues.count(mOp.getVRegValue())) {
		  if(newValues[mOp.getVRegValue()].count(i-1)) {
		    Value *oldV =  mOp.getVRegValue();
		    DEBUG(std::cerr << "Replaced this value: " << mOp.getVRegValue() << " With:" << (newValues[mOp.getVRegValue()][i-1]) << "\n");
		    //Update the operand with the right value
		    mOp.setValueReg(newValues[mOp.getVRegValue()][i-1]);

		    //Remove this value since we have consumed it
		    //NOTE: Should this only be done if j != maxStage?
		    consumedValues[oldV][i-1] = (newValues[oldV][i-1]);
		    DEBUG(std::cerr << "Deleted value: " << consumedValues[oldV][i-1] << "\n");
		    newValues[oldV].erase(i-1);
		  }
		}
		else
		  if(consumedValues.count(mOp.getVRegValue()))
		    assert(!consumedValues[mOp.getVRegValue()].count(i-1) && "Found a case where we need the value");
	      }
	    }
	  }
	}
      }
    }


    for(std::vector<MSchedGraphNode*>::iterator BR = branches.begin(), BE = branches.end(); BR != BE; ++BR) {
      
      //Stick in branch at the end
      machineBB->push_back((*BR)->getInst()->clone());

      //Add nop
      BuildMI(machineBB, V9::NOP, 0);
    }


  (((MachineBasicBlock*)origBB)->getParent())->getBasicBlockList().push_back(machineBB);  
    prologues.push_back(machineBB);
    llvm_prologues.push_back(llvmBB);
  }
}

void ModuloSchedulingPass::writeEpilogues(std::vector<MachineBasicBlock *> &epilogues, const MachineBasicBlock *origBB, std::vector<BasicBlock*> &llvm_epilogues, std::map<const Value*, std::pair<const MSchedGraphNode*, int> > &valuesToSave, std::map<Value*, std::map<int, Value*> > &newValues,std::map<Value*, MachineBasicBlock*> &newValLocation, std::map<Value*, std::map<int, Value*> > &kernelPHIs ) {
  
  std::map<int, std::set<const MachineInstr*> > inKernel;
  
  for(MSSchedule::kernel_iterator I = schedule.kernel_begin(), E = schedule.kernel_end(); I != E; ++I) {
    
    //Ignore the branch, we will handle this separately
    if(I->first->isBranch())
      continue;

    //Put int the map so we know what instructions in each stage are in the kernel
    inKernel[I->second].insert(I->first->getInst());
  }

  std::map<Value*, Value*> valPHIs;

  //some debug stuff, will remove later
  DEBUG(for(std::map<Value*, std::map<int, Value*> >::iterator V = newValues.begin(), E = newValues.end(); V !=E; ++V) {
    std::cerr << "Old Value: " << *(V->first) << "\n";
    for(std::map<int, Value*>::iterator I = V->second.begin(), IE = V->second.end(); I != IE; ++I)
      std::cerr << "Stage: " << I->first << " Value: " << *(I->second) << "\n";
  });

  //some debug stuff, will remove later
  DEBUG(for(std::map<Value*, std::map<int, Value*> >::iterator V = kernelPHIs.begin(), E = kernelPHIs.end(); V !=E; ++V) {
    std::cerr << "Old Value: " << *(V->first) << "\n";
    for(std::map<int, Value*>::iterator I = V->second.begin(), IE = V->second.end(); I != IE; ++I)
      std::cerr << "Stage: " << I->first << " Value: " << *(I->second) << "\n";
  });

  //Now write the epilogues
  for(int i = schedule.getMaxStage()-1; i >= 0; --i) {
    BasicBlock *llvmBB = new BasicBlock("EPILOGUE", (Function*) (origBB->getBasicBlock()->getParent()));
    MachineBasicBlock *machineBB = new MachineBasicBlock(llvmBB);
   
    DEBUG(std::cerr << " Epilogue #: " << i << "\n");


    std::map<Value*, int> inEpilogue;

     for(MachineBasicBlock::const_iterator MI = origBB->begin(), ME = origBB->end(); ME != MI; ++MI) {
      for(int j=schedule.getMaxStage(); j > i; --j) {
	if(inKernel[j].count(&*MI)) {
	  DEBUG(std::cerr << "Cloning instruction " << *MI << "\n");
	  MachineInstr *clone = MI->clone();
	  
	  //Update operands that need to use the result from the phi
	  for(unsigned opNum=0; opNum < clone->getNumOperands(); ++opNum) {
	    //get machine operand
	    const MachineOperand &mOp = clone->getOperand(opNum);
	    
	    if((mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isUse())) {
	      
	      DEBUG(std::cerr << "Writing PHI for " << *(mOp.getVRegValue()) << "\n");
	    
	      //If this is the last instructions for the max iterations ago, don't update operands
	      if(inEpilogue.count(mOp.getVRegValue()))
		if(inEpilogue[mOp.getVRegValue()] == i)
		  continue;
	      
	      //Quickly write appropriate phis for this operand
	      if(newValues.count(mOp.getVRegValue())) {
		if(newValues[mOp.getVRegValue()].count(i)) {
		  Instruction *tmp = new TmpInstruction(newValues[mOp.getVRegValue()][i]);
		 
		  //Get machine code for this instruction
		  MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
		  tempMvec.addTemp((Value*) tmp);

		  MachineInstr *saveValue = BuildMI(machineBB, V9::PHI, 3).addReg(newValues[mOp.getVRegValue()][i]).addReg(kernelPHIs[mOp.getVRegValue()][i]).addRegDef(tmp);
		  DEBUG(std::cerr << "Resulting PHI: " << *saveValue << "\n");
		  valPHIs[mOp.getVRegValue()] = tmp;
		}
	      }
	      
	      if(valPHIs.count(mOp.getVRegValue())) {
		//Update the operand in the cloned instruction
		clone->getOperand(opNum).setValueReg(valPHIs[mOp.getVRegValue()]); 
	      }
	    }
	    else if((mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isDef())) {
	      inEpilogue[mOp.getVRegValue()] = i;
	    }
	  }
	  machineBB->push_back(clone);
	}
      }
     }

    (((MachineBasicBlock*)origBB)->getParent())->getBasicBlockList().push_back(machineBB);
    epilogues.push_back(machineBB);
    llvm_epilogues.push_back(llvmBB);
  
    DEBUG(std::cerr << "EPILOGUE #" << i << "\n");
    DEBUG(machineBB->print(std::cerr));
  }
}

void ModuloSchedulingPass::writeKernel(BasicBlock *llvmBB, MachineBasicBlock *machineBB, std::map<const Value*, std::pair<const MSchedGraphNode*, int> > &valuesToSave, std::map<Value*, std::map<int, Value*> > &newValues, std::map<Value*, MachineBasicBlock*> &newValLocation, std::map<Value*, std::map<int, Value*> > &kernelPHIs) {
  
  //Keep track of operands that are read and saved from a previous iteration. The new clone
  //instruction will use the result of the phi instead.
  std::map<Value*, Value*> finalPHIValue;
  std::map<Value*, Value*> kernelValue;

  //Branches are a special case
  std::vector<MachineInstr*> branches;

    //Create TmpInstructions for the final phis
 for(MSSchedule::kernel_iterator I = schedule.kernel_begin(), E = schedule.kernel_end(); I != E; ++I) {

   DEBUG(std::cerr << "Stage: " << I->second << " Inst: " << *(I->first->getInst()) << "\n";);

   if(I->first->isBranch()) {
     //Clone instruction
     const MachineInstr *inst = I->first->getInst();
     MachineInstr *instClone = inst->clone();
     branches.push_back(instClone);
     continue;
   }
   
   //Clone instruction
   const MachineInstr *inst = I->first->getInst();
   MachineInstr *instClone = inst->clone();

   //Insert into machine basic block
   machineBB->push_back(instClone);

   DEBUG(std::cerr <<  "Cloned Inst: " << *instClone << "\n");

   //Loop over Machine Operands
   for(unsigned i=0; i < inst->getNumOperands(); ++i) {
     //get machine operand
     const MachineOperand &mOp = inst->getOperand(i);
   
     if(I->second != 0) {
       if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isUse()) {

	 //Check to see where this operand is defined if this instruction is from max stage
	 if(I->second == schedule.getMaxStage()) {
	   DEBUG(std::cerr << "VREG: " << *(mOp.getVRegValue()) << "\n");
	 }

	 //If its in the value saved, we need to create a temp instruction and use that instead
	 if(valuesToSave.count(mOp.getVRegValue())) {

	   //Check if we already have a final PHI value for this
	   if(!finalPHIValue.count(mOp.getVRegValue())) {
	     TmpInstruction *tmp = new TmpInstruction(mOp.getVRegValue());
	     
	     //Get machine code for this instruction
	     MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
	     tempMvec.addTemp((Value*) tmp);
	     
	     //Update the operand in the cloned instruction
	     instClone->getOperand(i).setValueReg(tmp);
	     
	     //save this as our final phi
	     finalPHIValue[mOp.getVRegValue()] = tmp;
	     newValLocation[tmp] = machineBB;
	   }
	   else {
	     //Use the previous final phi value
	     instClone->getOperand(i).setValueReg(finalPHIValue[mOp.getVRegValue()]); 
	   }
	 }
       }
     }
     if(I->second != schedule.getMaxStage()) {
       if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isDef()) {
	 if(valuesToSave.count(mOp.getVRegValue())) {
	   
	   TmpInstruction *tmp = new TmpInstruction(mOp.getVRegValue());
	   
	   //Get machine code for this instruction
	   MachineCodeForInstruction & tempVec = MachineCodeForInstruction::get(defaultInst);
	   tempVec.addTemp((Value*) tmp);

	   //Create new machine instr and put in MBB
	   MachineInstr *saveValue = BuildMI(machineBB, V9::ORr, 3).addReg(mOp.getVRegValue()).addImm(0).addRegDef(tmp);
	   
	   //Save for future cleanup
	   kernelValue[mOp.getVRegValue()] = tmp;
	   newValLocation[tmp] = machineBB;
	   kernelPHIs[mOp.getVRegValue()][schedule.getMaxStage()-1] = tmp;
	 }
       }
     }
   }
   
 }

 //Add branches
 for(std::vector<MachineInstr*>::iterator I = branches.begin(), E = branches.end(); I != E; ++I) {
   machineBB->push_back(*I);
   BuildMI(machineBB, V9::NOP, 0);
 }


  DEBUG(std::cerr << "KERNEL before PHIs\n");
  DEBUG(machineBB->print(std::cerr));


 //Loop over each value we need to generate phis for
 for(std::map<Value*, std::map<int, Value*> >::iterator V = newValues.begin(), 
       E = newValues.end(); V != E; ++V) {


   DEBUG(std::cerr << "Writing phi for" << *(V->first));
   DEBUG(std::cerr << "\nMap of Value* for this phi\n");
   DEBUG(for(std::map<int, Value*>::iterator I = V->second.begin(), 
	       IE = V->second.end(); I != IE; ++I) { 
     std::cerr << "Stage: " << I->first;
     std::cerr << " Value: " << *(I->second) << "\n";
   });

   //If we only have one current iteration live, its safe to set lastPhi = to kernel value
   if(V->second.size() == 1) {
     assert(kernelValue[V->first] != 0 && "Kernel value* must exist to create phi");
     MachineInstr *saveValue = BuildMI(*machineBB, machineBB->begin(),V9::PHI, 3).addReg(V->second.begin()->second).addReg(kernelValue[V->first]).addRegDef(finalPHIValue[V->first]); 
     DEBUG(std::cerr << "Resulting PHI: " << *saveValue << "\n");
     kernelPHIs[V->first][schedule.getMaxStage()-1] = kernelValue[V->first];
   }
   else {

     //Keep track of last phi created.
     Instruction *lastPhi = 0;
     
     unsigned count = 1;
     //Loop over the the map backwards to generate phis
     for(std::map<int, Value*>::reverse_iterator I = V->second.rbegin(), IE = V->second.rend(); 
	 I != IE; ++I) {

       if(count < (V->second).size()) {
	 if(lastPhi == 0) {
	   lastPhi = new TmpInstruction(I->second);

	   //Get machine code for this instruction
	   MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
	   tempMvec.addTemp((Value*) lastPhi);

	   MachineInstr *saveValue = BuildMI(*machineBB, machineBB->begin(), V9::PHI, 3).addReg(kernelValue[V->first]).addReg(I->second).addRegDef(lastPhi);
	   DEBUG(std::cerr << "Resulting PHI: " << *saveValue << "\n");
	   newValLocation[lastPhi] = machineBB;
	 }
	 else {
	   Instruction *tmp = new TmpInstruction(I->second);

	   //Get machine code for this instruction
	   MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
	   tempMvec.addTemp((Value*) tmp);
	   

	   MachineInstr *saveValue = BuildMI(*machineBB, machineBB->begin(), V9::PHI, 3).addReg(lastPhi).addReg(I->second).addRegDef(tmp);
	   DEBUG(std::cerr << "Resulting PHI: " << *saveValue << "\n");
	   lastPhi = tmp;
	   kernelPHIs[V->first][I->first] = lastPhi;
	   newValLocation[lastPhi] = machineBB;
	 }
       }
       //Final phi value
       else {
	 //The resulting value must be the Value* we created earlier
	 assert(lastPhi != 0 && "Last phi is NULL!\n");
	 MachineInstr *saveValue = BuildMI(*machineBB, machineBB->begin(), V9::PHI, 3).addReg(lastPhi).addReg(I->second).addRegDef(finalPHIValue[V->first]);
	 DEBUG(std::cerr << "Resulting PHI: " << *saveValue << "\n");
	 kernelPHIs[V->first][I->first] = finalPHIValue[V->first];
       }

       ++count;
     }

   }
 } 

  DEBUG(std::cerr << "KERNEL after PHIs\n");
  DEBUG(machineBB->print(std::cerr));
}


void ModuloSchedulingPass::removePHIs(const MachineBasicBlock *origBB, std::vector<MachineBasicBlock *> &prologues, std::vector<MachineBasicBlock *> &epilogues, MachineBasicBlock *kernelBB, std::map<Value*, MachineBasicBlock*> &newValLocation) {

  //Worklist to delete things
  std::vector<std::pair<MachineBasicBlock*, MachineBasicBlock::iterator> > worklist;

  //Worklist of TmpInstructions that need to be added to a MCFI
  std::vector<Instruction*> addToMCFI;
  
  //Worklist to add OR instructions to end of kernel so not to invalidate the iterator
  //std::vector<std::pair<Instruction*, Value*> > newORs;

  const TargetInstrInfo *TMI = target.getInstrInfo();

  //Start with the kernel and for each phi insert a copy for the phi def and for each arg
  for(MachineBasicBlock::iterator I = kernelBB->begin(), E = kernelBB->end(); I != E; ++I) {
  
    DEBUG(std::cerr << "Looking at Instr: " << *I << "\n");
    //Get op code and check if its a phi
    if(I->getOpcode() == V9::PHI) {
      
      DEBUG(std::cerr << "Replacing PHI: " << *I << "\n");
      Instruction *tmp = 0;

      for(unsigned i = 0; i < I->getNumOperands(); ++i) {
	//Get Operand
	const MachineOperand &mOp = I->getOperand(i);
	assert(mOp.getType() == MachineOperand::MO_VirtualRegister && "Should be a Value*\n");
	
	if(!tmp) {
	  tmp = new TmpInstruction(mOp.getVRegValue());
	  addToMCFI.push_back(tmp);
	}

	//Now for all our arguments we read, OR to the new TmpInstruction that we created
	if(mOp.isUse()) {
	  DEBUG(std::cerr << "Use: " << mOp << "\n");
	  //Place a copy at the end of its BB but before the branches
	  assert(newValLocation.count(mOp.getVRegValue()) && "We must know where this value is located\n");
	  //Reverse iterate to find the branches, we can safely assume no instructions have been
	  //put in the nop positions
	  for(MachineBasicBlock::iterator inst = --(newValLocation[mOp.getVRegValue()])->end(), endBB = (newValLocation[mOp.getVRegValue()])->begin(); inst != endBB; --inst) {
	    MachineOpCode opc = inst->getOpcode();
	    if(TMI->isBranch(opc) || TMI->isNop(opc))
	      continue;
	    else {
	      BuildMI(*(newValLocation[mOp.getVRegValue()]), ++inst, V9::ORr, 3).addReg(mOp.getVRegValue()).addImm(0).addRegDef(tmp);
	      break;
	    }
	    
	  }

	}
	else {
	  //Remove the phi and replace it with an OR
	  DEBUG(std::cerr << "Def: " << mOp << "\n");
	  //newORs.push_back(std::make_pair(tmp, mOp.getVRegValue()));
	  BuildMI(*kernelBB, I, V9::ORr, 3).addReg(tmp).addImm(0).addRegDef(mOp.getVRegValue());
	  worklist.push_back(std::make_pair(kernelBB, I));
	}
	
      }
      
    }

    
  }

  //Add TmpInstructions to some MCFI
  if(addToMCFI.size() > 0) {
    MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
    for(unsigned x = 0; x < addToMCFI.size(); ++x) {
      tempMvec.addTemp(addToMCFI[x]);
    }
    addToMCFI.clear();
  }


  //Remove phis from epilogue
  for(std::vector<MachineBasicBlock*>::iterator MB = epilogues.begin(), ME = epilogues.end(); MB != ME; ++MB) {
    for(MachineBasicBlock::iterator I = (*MB)->begin(), E = (*MB)->end(); I != E; ++I) {
      
      DEBUG(std::cerr << "Looking at Instr: " << *I << "\n");
      //Get op code and check if its a phi
      if(I->getOpcode() == V9::PHI) {
	Instruction *tmp = 0;

	for(unsigned i = 0; i < I->getNumOperands(); ++i) {
	  //Get Operand
	  const MachineOperand &mOp = I->getOperand(i);
	  assert(mOp.getType() == MachineOperand::MO_VirtualRegister && "Should be a Value*\n");
	  
	  if(!tmp) {
	    tmp = new TmpInstruction(mOp.getVRegValue());
	    addToMCFI.push_back(tmp);
	  }
	  
	  //Now for all our arguments we read, OR to the new TmpInstruction that we created
	  if(mOp.isUse()) {
	    DEBUG(std::cerr << "Use: " << mOp << "\n");
	    //Place a copy at the end of its BB but before the branches
	    assert(newValLocation.count(mOp.getVRegValue()) && "We must know where this value is located\n");
	    //Reverse iterate to find the branches, we can safely assume no instructions have been
	    //put in the nop positions
	    for(MachineBasicBlock::iterator inst = --(newValLocation[mOp.getVRegValue()])->end(), endBB = (newValLocation[mOp.getVRegValue()])->begin(); inst != endBB; --inst) {
	      MachineOpCode opc = inst->getOpcode();
	      if(TMI->isBranch(opc) || TMI->isNop(opc))
		continue;
	      else {
		BuildMI(*(newValLocation[mOp.getVRegValue()]), ++inst, V9::ORr, 3).addReg(mOp.getVRegValue()).addImm(0).addRegDef(tmp);
		break;
	      }
	      
	    }
	  	  	    
	  }
	  else {
	    //Remove the phi and replace it with an OR
	    DEBUG(std::cerr << "Def: " << mOp << "\n");
	    BuildMI(**MB, I, V9::ORr, 3).addReg(tmp).addImm(0).addRegDef(mOp.getVRegValue());
	    worklist.push_back(std::make_pair(*MB,I));
	  }
	  
	}
      }

  
    }
  }


  if(addToMCFI.size() > 0) {
    MachineCodeForInstruction & tempMvec = MachineCodeForInstruction::get(defaultInst);
    for(unsigned x = 0; x < addToMCFI.size(); ++x) {
      tempMvec.addTemp(addToMCFI[x]);
    }
    addToMCFI.clear();
  }

    //Delete the phis
  for(std::vector<std::pair<MachineBasicBlock*, MachineBasicBlock::iterator> >::iterator I =  worklist.begin(), E = worklist.end(); I != E; ++I) {
  
    DEBUG(std::cerr << "Deleting PHI " << *I->second << "\n");
    I->first->erase(I->second);
		    
  }


  assert((addToMCFI.size() == 0) && "We should have added all TmpInstructions to some MachineCodeForInstruction");
}


void ModuloSchedulingPass::reconstructLoop(MachineBasicBlock *BB) {

  TIME_REGION(X, "reconstructLoop");


  DEBUG(std::cerr << "Reconstructing Loop\n");

  //First find the value *'s that we need to "save"
  std::map<const Value*, std::pair<const MSchedGraphNode*, int> > valuesToSave;

  //Keep track of instructions we have already seen and their stage because
  //we don't want to "save" values if they are used in the kernel immediately
  std::map<const MachineInstr*, int> lastInstrs;

  //Loop over kernel and only look at instructions from a stage > 0
  //Look at its operands and save values *'s that are read
  for(MSSchedule::kernel_iterator I = schedule.kernel_begin(), E = schedule.kernel_end(); I != E; ++I) {

    if(I->second !=0) {
      //For this instruction, get the Value*'s that it reads and put them into the set.
      //Assert if there is an operand of another type that we need to save
      const MachineInstr *inst = I->first->getInst();
      lastInstrs[inst] = I->second;

      for(unsigned i=0; i < inst->getNumOperands(); ++i) {
	//get machine operand
	const MachineOperand &mOp = inst->getOperand(i);
	
	if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isUse()) {
	  //find the value in the map
	  if (const Value* srcI = mOp.getVRegValue()) {

	    if(isa<Constant>(srcI) || isa<Argument>(srcI) || isa<PHINode>(srcI))
	      continue;

	    //Before we declare this Value* one that we should save
	    //make sure its def is not of the same stage as this instruction
	    //because it will be consumed before its used
	    Instruction *defInst = (Instruction*) srcI;
	    
	    //Should we save this value?
	    bool save = true;

	    //Continue if not in the def map, loop invariant code does not need to be saved
	    if(!defMap.count(srcI))
	      continue;

	    MachineInstr *defInstr = defMap[srcI];
	    

	    if(lastInstrs.count(defInstr)) {
	      if(lastInstrs[defInstr] == I->second) {
		save = false;
		
	      }
	    }
	    
	    if(save)
	      valuesToSave[srcI] = std::make_pair(I->first, i);
	  }	  
	}
	
	if(mOp.getType() != MachineOperand::MO_VirtualRegister && mOp.isUse()) {
	  assert("Our assumption is wrong. We have another type of register that needs to be saved\n");
	}
      }
    }
  }

  //The new loop will consist of one or more prologues, the kernel, and one or more epilogues.

  //Map to keep track of old to new values
  std::map<Value*, std::map<int, Value*> > newValues;
 
  //Map to keep track of old to new values in kernel
  std::map<Value*, std::map<int, Value*> > kernelPHIs;

  //Another map to keep track of what machine basic blocks these new value*s are in since
  //they have no llvm instruction equivalent
  std::map<Value*, MachineBasicBlock*> newValLocation;

  std::vector<MachineBasicBlock*> prologues;
  std::vector<BasicBlock*> llvm_prologues;


  //Write prologue
  writePrologues(prologues, BB, llvm_prologues, valuesToSave, newValues, newValLocation);
    
  //Print out epilogues and prologue
  DEBUG(for(std::vector<MachineBasicBlock*>::iterator I = prologues.begin(), E = prologues.end(); 
      I != E; ++I) {
    std::cerr << "PROLOGUE\n";
    (*I)->print(std::cerr);
  });

  BasicBlock *llvmKernelBB = new BasicBlock("Kernel", (Function*) (BB->getBasicBlock()->getParent()));
  MachineBasicBlock *machineKernelBB = new MachineBasicBlock(llvmKernelBB);
  (((MachineBasicBlock*)BB)->getParent())->getBasicBlockList().push_back(machineKernelBB);
  writeKernel(llvmKernelBB, machineKernelBB, valuesToSave, newValues, newValLocation, kernelPHIs);
  
 
  std::vector<MachineBasicBlock*> epilogues;
  std::vector<BasicBlock*> llvm_epilogues;

  //Write epilogues
  writeEpilogues(epilogues, BB, llvm_epilogues, valuesToSave, newValues, newValLocation, kernelPHIs);


  //Fix our branches
  fixBranches(prologues, llvm_prologues, machineKernelBB, llvmKernelBB, epilogues, llvm_epilogues, BB);

  //Remove phis
  removePHIs(BB, prologues, epilogues, machineKernelBB, newValLocation);
    
  //Print out epilogues and prologue
  DEBUG(for(std::vector<MachineBasicBlock*>::iterator I = prologues.begin(), E = prologues.end(); 
      I != E; ++I) {
    std::cerr << "PROLOGUE\n";
    (*I)->print(std::cerr);
  });
  
  DEBUG(std::cerr << "KERNEL\n");
  DEBUG(machineKernelBB->print(std::cerr));

  DEBUG(for(std::vector<MachineBasicBlock*>::iterator I = epilogues.begin(), E = epilogues.end(); 
      I != E; ++I) {
    std::cerr << "EPILOGUE\n";
    (*I)->print(std::cerr);
  });


  DEBUG(std::cerr << "New Machine Function" << "\n");
  DEBUG(std::cerr << BB->getParent() << "\n");


}

void ModuloSchedulingPass::fixBranches(std::vector<MachineBasicBlock *> &prologues, std::vector<BasicBlock*> &llvm_prologues, MachineBasicBlock *machineKernelBB, BasicBlock *llvmKernelBB, std::vector<MachineBasicBlock *> &epilogues, std::vector<BasicBlock*> &llvm_epilogues, MachineBasicBlock *BB) {

  const TargetInstrInfo *TMI = target.getInstrInfo();

  //Fix prologue branches
  for(unsigned I = 0; I <  prologues.size(); ++I) {
   
     //Find terminator since getFirstTerminator does not work!
    for(MachineBasicBlock::reverse_iterator mInst = prologues[I]->rbegin(), mInstEnd = prologues[I]->rend(); mInst != mInstEnd; ++mInst) {
      MachineOpCode OC = mInst->getOpcode();
      //If its a branch update its branchto
      if(TMI->isBranch(OC)) {
	for(unsigned opNum = 0; opNum < mInst->getNumOperands(); ++opNum) {
	  MachineOperand &mOp = mInst->getOperand(opNum);
	  if (mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
	    //Check if we are branching to the kernel, if not branch to epilogue
	    if(mOp.getVRegValue() == BB->getBasicBlock()) { 
	      if(I == prologues.size()-1)
		mOp.setValueReg(llvmKernelBB);
	      else
		mOp.setValueReg(llvm_prologues[I+1]);
	    }
	    else {
	      mOp.setValueReg(llvm_epilogues[(llvm_epilogues.size()-1-I)]);
	    }
	  }
	}

	DEBUG(std::cerr << "New Prologue Branch: " << *mInst << "\n");
      }
    }


    //Update llvm basic block with our new branch instr
    DEBUG(std::cerr << BB->getBasicBlock()->getTerminator() << "\n");
    const BranchInst *branchVal = dyn_cast<BranchInst>(BB->getBasicBlock()->getTerminator());
   
    if(I == prologues.size()-1) {
      TerminatorInst *newBranch = new BranchInst(llvmKernelBB,
						 llvm_epilogues[(llvm_epilogues.size()-1-I)], 
						 branchVal->getCondition(), 
						 llvm_prologues[I]);
    }
    else
      TerminatorInst *newBranch = new BranchInst(llvm_prologues[I+1],
						 llvm_epilogues[(llvm_epilogues.size()-1-I)], 
						 branchVal->getCondition(), 
						 llvm_prologues[I]);

  }

  Value *origBranchExit = 0;

  //Fix up kernel machine branches
  for(MachineBasicBlock::reverse_iterator mInst = machineKernelBB->rbegin(), mInstEnd = machineKernelBB->rend(); mInst != mInstEnd; ++mInst) {
    MachineOpCode OC = mInst->getOpcode();
    if(TMI->isBranch(OC)) {
      for(unsigned opNum = 0; opNum < mInst->getNumOperands(); ++opNum) {
	MachineOperand &mOp = mInst->getOperand(opNum);
	
	if(mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
	  if(mOp.getVRegValue() == BB->getBasicBlock())
	    mOp.setValueReg(llvmKernelBB);
	  else
	    if(llvm_epilogues.size() > 0) {
	      assert(origBranchExit == 0 && "There should only be one branch out of the loop");
	      	     
	      origBranchExit = mOp.getVRegValue();
	      mOp.setValueReg(llvm_epilogues[0]);
	    }
	}
      }
    }
  }
   
  //Update kernelLLVM branches
  const BranchInst *branchVal = dyn_cast<BranchInst>(BB->getBasicBlock()->getTerminator());
  
  assert(llvm_epilogues.size() != 0 && "We must have epilogues!");
  
  TerminatorInst *newBranch = new BranchInst(llvmKernelBB,
					     llvm_epilogues[0], 
					     branchVal->getCondition(), 
					     llvmKernelBB);


   //Lastly add unconditional branches for the epilogues
   for(unsigned I = 0; I <  epilogues.size(); ++I) {
     
    //Now since we don't have fall throughs, add a unconditional branch to the next prologue
     if(I != epilogues.size()-1) {
       BuildMI(epilogues[I], V9::BA, 1).addPCDisp(llvm_epilogues[I+1]);
       //Add unconditional branch to end of epilogue
       TerminatorInst *newBranch = new BranchInst(llvm_epilogues[I+1], 
						  llvm_epilogues[I]);

     }
     else {
       BuildMI(epilogues[I], V9::BA, 1).addPCDisp(origBranchExit);
       
      
       //Update last epilogue exit branch
       BranchInst *branchVal = (BranchInst*) dyn_cast<BranchInst>(BB->getBasicBlock()->getTerminator());
       //Find where we are supposed to branch to
       BasicBlock *nextBlock = 0;
       for(unsigned j=0; j <branchVal->getNumSuccessors(); ++j) {
	 if(branchVal->getSuccessor(j) != BB->getBasicBlock())
	   nextBlock = branchVal->getSuccessor(j);
       }
       
       assert((nextBlock != 0) && "Next block should not be null!");
       TerminatorInst *newBranch = new BranchInst(nextBlock, llvm_epilogues[I]);
     }
     //Add one more nop!
     BuildMI(epilogues[I], V9::NOP, 0);
     
   }

   //FIX UP Machine BB entry!!
   //We are looking at the predecesor of our loop basic block and we want to change its ba instruction
   

   //Find all llvm basic blocks that branch to the loop entry and change to our first prologue.
   const BasicBlock *llvmBB = BB->getBasicBlock();

   std::vector<const BasicBlock*>Preds (pred_begin(llvmBB), pred_end(llvmBB));

   //for(pred_const_iterator P = pred_begin(llvmBB), PE = pred_end(llvmBB); P != PE; ++PE) {
   for(std::vector<const BasicBlock*>::iterator P = Preds.begin(), PE = Preds.end(); P != PE; ++P) { 
     if(*P == llvmBB)
       continue;
     else {
       DEBUG(std::cerr << "Found our entry BB\n");
       //Get the Terminator instruction for this basic block and print it out
       DEBUG(std::cerr << *((*P)->getTerminator()) << "\n");
       //Update the terminator
       TerminatorInst *term = ((BasicBlock*)*P)->getTerminator();
       for(unsigned i=0; i < term->getNumSuccessors(); ++i) {
	 if(term->getSuccessor(i) == llvmBB) {
	   DEBUG(std::cerr << "Replacing successor bb\n");
	   if(llvm_prologues.size() > 0) {
	     term->setSuccessor(i, llvm_prologues[0]);
	     //Also update its corresponding machine instruction
	     MachineCodeForInstruction & tempMvec =
	       MachineCodeForInstruction::get(term);
	     for (unsigned j = 0; j < tempMvec.size(); j++) {
	       MachineInstr *temp = tempMvec[j];
	       MachineOpCode opc = temp->getOpcode();
	       if(TMI->isBranch(opc)) {
		 DEBUG(std::cerr << *temp << "\n");
		 //Update branch
		 for(unsigned opNum = 0; opNum < temp->getNumOperands(); ++opNum) {
		   MachineOperand &mOp = temp->getOperand(opNum);
		   if (mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
		     mOp.setValueReg(llvm_prologues[0]);
		   }
		 }
	       }
	     }        
	   }
	   else {
	     term->setSuccessor(i, llvmKernelBB);
	   //Also update its corresponding machine instruction
	     MachineCodeForInstruction & tempMvec =
	       MachineCodeForInstruction::get(term);
	     for (unsigned j = 0; j < tempMvec.size(); j++) {
	       MachineInstr *temp = tempMvec[j];
	       MachineOpCode opc = temp->getOpcode();
	       if(TMI->isBranch(opc)) {
		 DEBUG(std::cerr << *temp << "\n");
		 //Update branch
		 for(unsigned opNum = 0; opNum < temp->getNumOperands(); ++opNum) {
		   MachineOperand &mOp = temp->getOperand(opNum);
		   if (mOp.getType() == MachineOperand::MO_PCRelativeDisp) {
		     mOp.setValueReg(llvmKernelBB);
		   }
		 }
	       }
	     }
	   }
	 }
       }
       break;
     }
   }
   

  //BB->getParent()->getBasicBlockList().erase(BB);

}

