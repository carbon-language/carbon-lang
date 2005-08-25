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
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetSchedInfo.h"
#include "../SparcV9Internals.h"
#include "llvm/CodeGen/MachineInstr.h"

using namespace llvm;

//Check if all resources are free
bool resourcesFree(MSchedGraphNode*, int,
std::map<int, std::map<int, int> > &resourceNumPerCycle);

//Returns a boolean indicating if the start cycle needs to be increased/decreased
bool MSSchedule::insert(MSchedGraphNode *node, int cycle, int II) {

  //First, check if the cycle has a spot free to start
  if(schedule.find(cycle) != schedule.end()) {
    //Check if we have a free issue slot at this cycle
    if (schedule[cycle].size() < numIssue) {
      //Now check if all the resources in their respective cycles are available
      if(resourcesFree(node, cycle, II)) {
        //Insert to preserve dependencies
        addToSchedule(cycle,node);
        DEBUG(std::cerr << "Found spot in map, and there is an issue slot\n");
        return false;
      }
    }
  }
  //Not in the map yet so put it in
  else {
    if(resourcesFree(node,cycle,II)) {
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

void MSSchedule::addToSchedule(int cycle, MSchedGraphNode *node) {
  std::vector<MSchedGraphNode*> nodesAtCycle = schedule[cycle];

  std::map<unsigned, MSchedGraphNode*> indexMap;
  for(unsigned i=0; i < nodesAtCycle.size(); ++i) {
    indexMap[nodesAtCycle[i]->getIndex()] = nodesAtCycle[i];
  }

  indexMap[node->getIndex()] = node;

  std::vector<MSchedGraphNode*> nodes;
  for(std::map<unsigned, MSchedGraphNode*>::iterator I = indexMap.begin(), E = indexMap.end(); I != E; ++I)
    nodes.push_back(I->second);

  schedule[cycle] =  nodes;
}

bool MSSchedule::resourceAvailable(int resourceNum, int cycle) {
  bool isFree = true;

  //Get Map for this cycle
  if(resourceNumPerCycle.count(cycle)) {
    if(resourceNumPerCycle[cycle].count(resourceNum)) {
      int maxRes = CPUResource::getCPUResource(resourceNum)->maxNumUsers;
      if(resourceNumPerCycle[cycle][resourceNum] >= maxRes)
        isFree = false;
    }
  }

  return isFree;
}

void MSSchedule::useResource(int resourceNum, int cycle) {

  //Get Map for this cycle
  if(resourceNumPerCycle.count(cycle)) {
    if(resourceNumPerCycle[cycle].count(resourceNum)) {
      resourceNumPerCycle[cycle][resourceNum]++;
    }
    else {
      resourceNumPerCycle[cycle][resourceNum] = 1;
    }
  }
  //If no map, create one!
  else {
    std::map<int, int> resourceUse;
    resourceUse[resourceNum] = 1;
    resourceNumPerCycle[cycle] = resourceUse;
  }

}

bool MSSchedule::resourcesFree(MSchedGraphNode *node, int cycle, int II) {

  //Get Resource usage for this instruction
  const TargetSchedInfo *msi = node->getParent()->getTarget()->getSchedInfo();
  int currentCycle = cycle;
  bool success = true;

  //Create vector of starting cycles
  std::vector<int> cyclesMayConflict;
  cyclesMayConflict.push_back(cycle);

  if(resourceNumPerCycle.size() > 0) {
    for(int i = cycle-II; i >= (resourceNumPerCycle.begin()->first); i-=II)
      cyclesMayConflict.push_back(i);
    for(int i = cycle+II; i <= resourceNumPerCycle.end()->first; i+=II)
      cyclesMayConflict.push_back(i);
  }

  //Now check all cycles for conflicts
  for(int index = 0; index < (int) cyclesMayConflict.size(); ++index) {
    currentCycle = cyclesMayConflict[index];

    //Get resource usage for this instruction
    InstrRUsage rUsage = msi->getInstrRUsage(node->getInst()->getOpcode());
    std::vector<std::vector<resourceId_t> > resources = rUsage.resourcesByCycle;

    //Loop over resources in each cycle and increments their usage count
    for(unsigned i=0; i < resources.size(); ++i) {
      for(unsigned j=0; j < resources[i].size(); ++j) {

        //Get Resource to check its availability
        int resourceNum = resources[i][j];

        DEBUG(std::cerr << "Attempting to schedule Resource Num: " << resourceNum << " in cycle: " << currentCycle << "\n");

        success = resourceAvailable(resourceNum, currentCycle);

        if(!success)
          break;

      }

      if(!success)
        break;

      //Increase cycle
      currentCycle++;
    }

    if(!success)
      return false;
  }

  //Actually put resources into the map
  if(success) {

    int currentCycle = cycle;
    //Get resource usage for this instruction
    InstrRUsage rUsage = msi->getInstrRUsage(node->getInst()->getOpcode());
    std::vector<std::vector<resourceId_t> > resources = rUsage.resourcesByCycle;

    //Loop over resources in each cycle and increments their usage count
    for(unsigned i=0; i < resources.size(); ++i) {
      for(unsigned j=0; j < resources[i].size(); ++j) {
        int resourceNum = resources[i][j];
        useResource(resourceNum, currentCycle);
      }
      currentCycle++;
    }
  }


  return true;

}

bool MSSchedule::constructKernel(int II, std::vector<MSchedGraphNode*> &branches, std::map<const MachineInstr*, unsigned> &indVar) {

  //Our schedule is allowed to have negative numbers, so lets calculate this offset
  int offset = schedule.begin()->first;
  if(offset > 0)
    offset = 0;

  DEBUG(std::cerr << "Offset: " << offset << "\n");

  //Using the schedule, fold up into kernel and check resource conflicts as we go
  std::vector<std::pair<MSchedGraphNode*, int> > tempKernel;

  int stageNum = ((schedule.rbegin()->first-offset)+1)/ II;
  int maxSN = 0;

  DEBUG(std::cerr << "Number of Stages: " << stageNum << "\n");

  for(int index = offset; index < (II+offset); ++index) {
    int count = 0;
    for(int i = index; i <= (schedule.rbegin()->first); i+=II) {
      if(schedule.count(i)) {
        for(std::vector<MSchedGraphNode*>::iterator I = schedule[i].begin(),
              E = schedule[i].end(); I != E; ++I) {
          //Check if its a branch
          assert(!(*I)->isBranch() && "Branch should not be schedule!");

          tempKernel.push_back(std::make_pair(*I, count));
          maxSN = std::max(maxSN, count);

        }
      }
      ++count;
    }
  }


  //Add in induction var code
  for(std::vector<std::pair<MSchedGraphNode*, int> >::iterator I = tempKernel.begin(), IE = tempKernel.end();
      I != IE; ++I) {
    //Add indVar instructions before this one for the current iteration
    if(I->second == 0) {
      std::map<unsigned, MachineInstr*> tmpMap;

      //Loop over induction variable instructions in the map that come before this instr
      for(std::map<const MachineInstr*, unsigned>::iterator N = indVar.begin(), NE = indVar.end(); N != NE; ++N) {


        if(N->second < I->first->getIndex())
          tmpMap[N->second] = (MachineInstr*) N->first;
      }

      //Add to kernel, and delete from indVar
      for(std::map<unsigned, MachineInstr*>::iterator N = tmpMap.begin(), NE = tmpMap.end(); N != NE; ++N) {
        kernel.push_back(std::make_pair(N->second, 0));
        indVar.erase(N->second);
      }
    }

    kernel.push_back(std::make_pair((MachineInstr*) I->first->getInst(), I->second));

  }

  std::map<unsigned, MachineInstr*> tmpMap;

  //Add remaining invar instructions
  for(std::map<const MachineInstr*, unsigned>::iterator N = indVar.begin(), NE = indVar.end(); N != NE; ++N) {
    tmpMap[N->second] = (MachineInstr*) N->first;
  }

  //Add to kernel, and delete from indVar
  for(std::map<unsigned, MachineInstr*>::iterator N = tmpMap.begin(), NE = tmpMap.end(); N != NE; ++N) {
    kernel.push_back(std::make_pair(N->second, 0));
    indVar.erase(N->second);
  }


  maxStage = maxSN;


  return true;
}

bool MSSchedule::defPreviousStage(Value *def, int stage) {

  //Loop over kernel and determine if value is being defined in previous stage
  for(std::vector<std::pair<MachineInstr*, int> >::iterator P = kernel.begin(), PE = kernel.end(); P != PE; ++P) {
    MachineInstr* inst = P->first;

    //Loop over Machine Operands
    for(unsigned i=0; i < inst->getNumOperands(); ++i) {
      //get machine operand
     const MachineOperand &mOp = inst->getOperand(i);
     if(mOp.getType() == MachineOperand::MO_VirtualRegister && mOp.isDef()) {
       if(def == mOp.getVRegValue()) {
         if(P->second >= stage)
           return false;
         else
           return true;
       }
     }
    }
  }

  assert(0 && "We should always have found the def in our kernel\n");
  abort();
}


void MSSchedule::print(std::ostream &os) const {
  os << "Schedule:\n";

  for(schedule_const_iterator I =  schedule.begin(), E = schedule.end(); I != E; ++I) {
    os << "Cycle: " << I->first << "\n";
    for(std::vector<MSchedGraphNode*>::const_iterator node = I->second.begin(), nodeEnd = I->second.end(); node != nodeEnd; ++node)
    os << **node << "\n";
  }

  os << "Kernel:\n";
  for(std::vector<std::pair<MachineInstr*, int> >::const_iterator I = kernel.begin(),
        E = kernel.end(); I != E; ++I)
    os << "Node: " << *(I->first) << " Stage: " << I->second << "\n";
}

