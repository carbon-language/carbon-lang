//===-- MSSchedule.cpp Schedule ---------------------------------*- C++ -*-===//
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

#include "MSSchedule.h"
#include "Support/Debug.h"
#include "llvm/Target/TargetSchedInfo.h"
#include <iostream>

using namespace llvm;

bool MSSchedule::insert(MSchedGraphNode *node, int cycle) {
  
  //First, check if the cycle has a spot free to start
  if(schedule.find(cycle) != schedule.end()) {
    if (schedule[cycle].size() < numIssue) {
      if(resourcesFree(node, cycle)) {
	schedule[cycle].push_back(node);
	DEBUG(std::cerr << "Found spot in map, and there is an issue slot\n");
	return false;
      }
    }
  }
  //Not in the map yet so put it in
  else {
    if(resourcesFree(node,cycle)) {
      std::vector<MSchedGraphNode*> nodes;
      nodes.push_back(node);
      schedule[cycle] = nodes;
      DEBUG(std::cerr << "Nothing in map yet so taking an issue slot\n");
      return false;
    }
  }

  DEBUG(std::cerr << "All issue slots taken\n");
  return true;

}

bool MSSchedule::resourcesFree(MSchedGraphNode *node, int cycle) {

  //Get Resource usage for this instruction
  const TargetSchedInfo & msi = node->getParent()->getTarget()->getSchedInfo();
  int currentCycle = cycle;
  bool success = true;

    //Get resource usage for this instruction
    InstrRUsage rUsage = msi.getInstrRUsage(node->getInst()->getOpcode());
    std::vector<std::vector<resourceId_t> > resources = rUsage.resourcesByCycle;

    //Loop over resources in each cycle and increments their usage count
    for(unsigned i=0; i < resources.size(); ++i) {
      for(unsigned j=0; j < resources[i].size(); ++j) {
	int resourceNum = resources[i][j];

	//Check if this resource is available for this cycle
	std::map<int, std::map<int,int> >::iterator resourcesForCycle = resourceNumPerCycle.find(currentCycle);

	//First check map of resources for this cycle
	if(resourcesForCycle != resourceNumPerCycle.end()) {
	  //A map exists for this cycle, so lets check for the resource
	  std::map<int, int>::iterator resourceUse = resourcesForCycle->second.find(resourceNum);
	  if(resourceUse != resourcesForCycle->second.end()) {
	    //Check if there are enough of this resource and if so, increase count and move on
	    if(resourceUse->second < CPUResource::getCPUResource(resourceNum)->maxNumUsers)
	      ++resourceUse->second;
	    else {
	      success = false;
	    }
	  }
	  //Not in the map yet, so put it
	  else
	    resourcesForCycle->second[resourceNum] = 1;
	
	}
	else {
	  //Create a new map and put in our resource
	  std::map<int, int> resourceMap;
	  resourceMap[resourceNum] = 1;
	  resourceNumPerCycle[cycle] = resourceMap;
	}
	if(!success)
	  break;
      }
      if(!success)
	break;
	//Increase cycle
	currentCycle++;
      }

    if(!success) {
      int oldCycle = cycle;
      DEBUG(std::cerr << "Backtrack\n");
      //Get resource usage for this instruction
      InstrRUsage rUsage = msi.getInstrRUsage(node->getInst()->getOpcode());
      std::vector<std::vector<resourceId_t> > resources = rUsage.resourcesByCycle;
      
      //Loop over resources in each cycle and increments their usage count
      for(unsigned i=0; i < resources.size(); ++i) {
	if(oldCycle < currentCycle) {

	  //Check if this resource is available for this cycle
	  std::map<int, std::map<int,int> >::iterator resourcesForCycle = resourceNumPerCycle.find(oldCycle);

	  for(unsigned j=0; j < resources[i].size(); ++j) {
	    int resourceNum = resources[i][j];
	    //remove from map
	    std::map<int, int>::iterator resourceUse = resourcesForCycle->second.find(resourceNum);
	    //assert if not in the map.. since it should be!
	    //assert(resourceUse != resourcesForCycle.end() && "Resource should be in map!");
	    --resourceUse->second;
	  }
	}
	else
	  break;
	oldCycle++;
      }
      return false;

    }

    return true;

}

bool MSSchedule::constructKernel(int II) {
  MSchedGraphNode *branchNode;

  int stageNum = (schedule.rbegin()->first)/ II;
  DEBUG(std::cerr << "Number of Stages: " << stageNum << "\n");
  
  for(int index = 0; index < II; ++index) {
    int count = 0;
    for(int i = index; i <= (schedule.rbegin()->first); i+=II) {  
      if(schedule.count(i)) {
	for(std::vector<MSchedGraphNode*>::iterator I = schedule[i].begin(), 
	      E = schedule[i].end(); I != E; ++I) {
	  //Check if its a branch
	  if((*I)->isBranch()) {
	    branchNode = *I;
	    assert(count == 0 && "Branch can not be from a previous iteration");
	  }
	  else
	  //FIXME: Check if the instructions in the earlier stage conflict
	  kernel.push_back(std::make_pair(*I, count));
	}
      }
      ++count;
    }
  }
  
  //Add Branch to the end
  kernel.push_back(std::make_pair(branchNode, 0));
  
  return true;
}


void MSSchedule::print(std::ostream &os) const {
  os << "Schedule:\n";
  
  for(schedule_const_iterator I =  schedule.begin(), E = schedule.end(); I != E; ++I) {
    os << "Cycle: " << I->first << "\n";
    for(std::vector<MSchedGraphNode*>::const_iterator node = I->second.begin(), nodeEnd = I->second.end(); node != nodeEnd; ++node)
    os << **node << "\n";
  }

  os << "Kernel:\n";
  for(std::vector<std::pair<MSchedGraphNode*, int> >::const_iterator I = kernel.begin(),
	E = kernel.end(); I != E; ++I)
    os << "Node: " << *(I->first) << " Stage: " << I->second << "\n";
}
  
