//===- ModuloScheduling.cpp - Modulo Software Pipelining ------------------===//
//
// Implements the llvm/CodeGen/ModuloScheduling.h interface
//
//===----------------------------------------------------------------------===//

//#include "llvm/CodeGen/MachineCodeForBasicBlock.h"
//#include "llvm/CodeGen/MachineCodeForMethod.h"
//#include "llvm/Analysis/LiveVar/FunctionLiveVarInfo.h" // FIXME: Remove when modularized better
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Instruction.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/Target/TargetSchedInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/CommandLine.h"
#include "Support/Statistic.h"
#include "ModuloSchedGraph.h"
#include "ModuloScheduling.h"
#include <algorithm>
#include <fstream>
#include <iostream>

using std::endl;

//************************************************************
// printing Debug information
// ModuloSchedDebugLevel stores the value of debug level
// modsched_os is the ostream to dump debug information, which is written into
// the file 'moduloSchedDebugInfo.output'
// see ModuloSchedulingPass::runOnFunction()
//************************************************************

ModuloSchedDebugLevel_t ModuloSchedDebugLevel;

cl::opt<ModuloSchedDebugLevel_t,true>
SDL_opt("modsched", cl::Hidden, cl::location(ModuloSchedDebugLevel),
	cl::desc("enable modulo scheduling debugging information"),
	cl::values(clEnumValN(ModuloSchedDebugLevel_NoDebugInfo,
			      "none", "disable debug output"),
		   clEnumValN(ModuloSchedDebugLevel_PrintSchedule,
			      "psched", "print original and new schedule"),
		   clEnumValN(ModuloSchedDebugLevel_PrintScheduleProcess,
			      "pschedproc",
			      "print how the new schdule is produced"),
		   0));

// Computes the schedule and inserts epilogue and prologue
//
void ModuloScheduling::instrScheduling()
{

  printf(" instrScheduling \n");

  if (ModuloScheduling::printScheduleProcess())
    DEBUG_PRINT(std::cerr << "************ computing modulo schedule ***********\n");

  const TargetSchedInfo & msi = target.getSchedInfo();

  //number of issue slots in the in each cycle
  int numIssueSlots = msi.maxNumIssueTotal;

  //compute the schedule
  bool success = false;
  while (!success) {
    //clear memory from the last round and initialize if necessary
    clearInitMem(msi);

    //compute schedule and coreSchedule with the current II
    success = computeSchedule();

    if (!success) {
      II++;
      if (ModuloScheduling::printScheduleProcess())
        DEBUG_PRINT(std::cerr << "increase II  to " << II << "\n");
    }
  }

  //print the final schedule if necessary
  if (ModuloScheduling::printSchedule())
    dumpScheduling();

  //the schedule has been computed
  //create epilogue, prologue and kernel BasicBlock

  //find the successor for this BasicBlock
  BasicBlock *succ_bb = getSuccBB(bb);

  //print the original BasicBlock if necessary
  if (ModuloScheduling::printSchedule()) {
    DEBUG_PRINT(std::cerr << "dumping the orginal block\n");
    graph.dump(bb);
  }
  //construction of prologue, kernel and epilogue
  BasicBlock *kernel = bb->splitBasicBlock(bb->begin());
  BasicBlock *prologue = bb;
  BasicBlock *epilogue = kernel->splitBasicBlock(kernel->begin());

  // Construct prologue
  constructPrologue(prologue);

  // Construct kernel
  constructKernel(prologue, kernel, epilogue);

  // Construct epilogue
  constructEpilogue(epilogue, succ_bb);

  //print the BasicBlocks if necessary
  if (ModuloScheduling::printSchedule()) {
    DEBUG_PRINT(std::cerr << "dumping the prologue block:\n");
    graph.dump(prologue);
    DEBUG_PRINT(std::cerr << "dumping the kernel block\n");
    graph.dump(kernel);
    DEBUG_PRINT(std::cerr << "dumping the epilogue block\n");
    graph.dump(epilogue);
  }
}

// Clear memory from the last round and initialize if necessary
//
void ModuloScheduling::clearInitMem(const TargetSchedInfo & msi)
{
  unsigned numIssueSlots = msi.maxNumIssueTotal;
  // clear nodeScheduled from the last round
  if (ModuloScheduling::printScheduleProcess()) {
    DEBUG_PRINT(std::cerr << "***** new round  with II= " << II << " ***********\n");
    DEBUG_PRINT(std::cerr <<
        " ************clear the vector nodeScheduled*************\n");
  }
  nodeScheduled.clear();

  // clear resourceTable from the last round and reset it 
  resourceTable.clear();
  for (unsigned i = 0; i < II; ++i)
    resourceTable.push_back(msi.resourceNumVector);

  // clear the schdule and coreSchedule from the last round 
  schedule.clear();
  coreSchedule.clear();

  // create a coreSchedule of size II*numIssueSlots
  // each entry is NULL
  while (coreSchedule.size() < II) {
    std::vector < ModuloSchedGraphNode * >*newCycle =
        new std::vector < ModuloSchedGraphNode * >();
    for (unsigned k = 0; k < numIssueSlots; ++k)
      newCycle->push_back(NULL);
    coreSchedule.push_back(*newCycle);
  }
}

// Compute schedule and coreSchedule with the current II
//
bool ModuloScheduling::computeSchedule()
{

  if (ModuloScheduling::printScheduleProcess())
    DEBUG_PRINT(std::cerr << "start to compute schedule\n");

  // Loop over the ordered nodes
  for (NodeVec::const_iterator I = oNodes.begin(); I != oNodes.end(); ++I) {
    // Try to schedule for node I
    if (ModuloScheduling::printScheduleProcess())
      dumpScheduling();
    ModuloSchedGraphNode *node = *I;

    // Compute whether this node has successor(s)
    bool succ = true;

    // Compute whether this node has predessor(s)
    bool pred = true;

    NodeVec schSucc = graph.vectorConj(nodeScheduled, graph.succSet(node));
    if (schSucc.empty())
      succ = false;
    NodeVec schPred = graph.vectorConj(nodeScheduled, graph.predSet(node));
    if (schPred.empty())
      pred = false;

    //startTime: the earliest time we will try to schedule this node
    //endTime: the latest time we will try to schedule this node
    int startTime, endTime;

    //node's earlyStart: possible earliest time to schedule this node
    //node's lateStart: possible latest time to schedule this node
    node->setEarlyStart(-1);
    node->setLateStart(9999);

    //this node has predessor but no successor
    if (!succ && pred) {
      // This node's earlyStart is it's predessor's schedule time + the edge
      // delay - the iteration difference* II
      for (unsigned i = 0; i < schPred.size(); i++) {
        ModuloSchedGraphNode *predNode = schPred[i];
        SchedGraphEdge *edge =
            graph.getMaxDelayEdge(predNode->getNodeId(),
                                  node->getNodeId());
        int temp =
            predNode->getSchTime() + edge->getMinDelay() -
            edge->getIteDiff() * II;
        node->setEarlyStart(std::max(node->getEarlyStart(), temp));
      }
      startTime = node->getEarlyStart();
      endTime = node->getEarlyStart() + II - 1;
    }
    // This node has a successor but no predecessor
    if (succ && !pred) {
      for (unsigned i = 0; i < schSucc.size(); ++i) {
        ModuloSchedGraphNode *succNode = schSucc[i];
        SchedGraphEdge *edge =
            graph.getMaxDelayEdge(succNode->getNodeId(),
                                  node->getNodeId());
        int temp =
            succNode->getSchTime() - edge->getMinDelay() +
            edge->getIteDiff() * II;
        node->setLateStart(std::min(node->getEarlyStart(), temp));
      }
      startTime = node->getLateStart() - II + 1;
      endTime = node->getLateStart();
    }
    // This node has both successors and predecessors
    if (succ && pred) {
      for (unsigned i = 0; i < schPred.size(); ++i) {
        ModuloSchedGraphNode *predNode = schPred[i];
        SchedGraphEdge *edge =
            graph.getMaxDelayEdge(predNode->getNodeId(),
                                  node->getNodeId());
        int temp =
            predNode->getSchTime() + edge->getMinDelay() -
            edge->getIteDiff() * II;
        node->setEarlyStart(std::max(node->getEarlyStart(), temp));
      }
      for (unsigned i = 0; i < schSucc.size(); ++i) {
        ModuloSchedGraphNode *succNode = schSucc[i];
        SchedGraphEdge *edge =
            graph.getMaxDelayEdge(succNode->getNodeId(),
                                  node->getNodeId());
        int temp =
            succNode->getSchTime() - edge->getMinDelay() +
            edge->getIteDiff() * II;
        node->setLateStart(std::min(node->getEarlyStart(), temp));
      }
      startTime = node->getEarlyStart();
      endTime = std::min(node->getLateStart(),
                         node->getEarlyStart() + ((int) II) - 1);
    }
    //this node has no successor or predessor
    if (!succ && !pred) {
      node->setEarlyStart(node->getASAP());
      startTime = node->getEarlyStart();
      endTime = node->getEarlyStart() + II - 1;
    }
    //try to schedule this node based on the startTime and endTime
    if (ModuloScheduling::printScheduleProcess())
      DEBUG_PRINT(std::cerr << "scheduling the node " << (*I)->getNodeId() << "\n");

    bool success =
        this->ScheduleNode(node, startTime, endTime, nodeScheduled);
    if (!success)
      return false;
  }
  return true;
}


// Get the successor of the BasicBlock
//
BasicBlock *ModuloScheduling::getSuccBB(BasicBlock *bb)
{
  BasicBlock *succ_bb;
  for (unsigned i = 0; i < II; ++i)
    for (unsigned j = 0; j < coreSchedule[i].size(); ++j)
      if (coreSchedule[i][j]) {
        const Instruction *ist = coreSchedule[i][j]->getInst();

        //we can get successor from the BranchInst instruction
        //assume we only have one successor (besides itself) here
        if (BranchInst::classof(ist)) {
          BranchInst *bi = (BranchInst *) ist;
          assert(bi->isConditional() &&
                 "the branchInst is not a conditional one");
          assert(bi->getNumSuccessors() == 2
                 && " more than two successors?");
          BasicBlock *bb1 = bi->getSuccessor(0);
          BasicBlock *bb2 = bi->getSuccessor(1);
          assert((bb1 == bb || bb2 == bb) &&
                 " None of its successors is itself?");
          if (bb1 == bb)
            succ_bb = bb2;
          else
            succ_bb = bb1;
          return succ_bb;
        }
      }
  assert(0 && "NO Successor?");
  return NULL;
}


// Get the predecessor of the BasicBlock
//
BasicBlock *ModuloScheduling::getPredBB(BasicBlock *bb)
{
  BasicBlock *pred_bb;
  for (unsigned i = 0; i < II; ++i)
    for (unsigned j = 0; j < coreSchedule[i].size(); ++j)
      if (coreSchedule[i][j]) {
        const Instruction *ist = coreSchedule[i][j]->getInst();

        //we can get predecessor from the PHINode instruction
        //assume we only have one predecessor (besides itself) here
        if (PHINode::classof(ist)) {
          PHINode *phi = (PHINode *) ist;
          assert(phi->getNumIncomingValues() == 2 &&
                 " the number of incoming value is not equal to two? ");
          BasicBlock *bb1 = phi->getIncomingBlock(0);
          BasicBlock *bb2 = phi->getIncomingBlock(1);
          assert((bb1 == bb || bb2 == bb) &&
                 " None of its predecessor is itself?");
          if (bb1 == bb)
            pred_bb = bb2;
          else
            pred_bb = bb1;
          return pred_bb;
        }
      }
  assert(0 && " no predecessor?");
  return NULL;
}


// Construct the prologue
//
void ModuloScheduling::constructPrologue(BasicBlock *prologue)
{
  InstListType & prologue_ist = prologue->getInstList();
  vvNodeType & tempSchedule_prologue =
      *(new std::vector<std::vector<ModuloSchedGraphNode*> >(schedule));

  //compute the schedule for prologue
  unsigned round = 0;
  unsigned scheduleSize = schedule.size();
  while (round < scheduleSize / II) {
    round++;
    for (unsigned i = 0; i < scheduleSize; ++i) {
      if (round * II + i >= scheduleSize)
        break;
      for (unsigned j = 0; j < schedule[i].size(); ++j) {
        if (schedule[i][j]) {
          assert(tempSchedule_prologue[round * II + i][j] == NULL &&
                 "table not consitent with core table");
          // move the schedule one iteration ahead and overlap with the original
          tempSchedule_prologue[round * II + i][j] = schedule[i][j];
        }
      }
    }
  }

  // Clear the clone memory in the core schedule instructions
  clearCloneMemory();

  // Fill in the prologue
  for (unsigned i = 0; i < ceil(1.0 * scheduleSize / II - 1) * II; ++i)
    for (unsigned j = 0; j < tempSchedule_prologue[i].size(); ++j)
      if (tempSchedule_prologue[i][j]) {

        //get the instruction
        Instruction *orn =
            (Instruction *) tempSchedule_prologue[i][j]->getInst();

        //made a clone of it
        Instruction *cln = cloneInstSetMemory(orn);

        //insert the instruction
        prologue_ist.insert(prologue_ist.back(), cln);

        //if there is PHINode in the prologue, the incoming value from itself
        //should be removed because it is not a loop any longer
        if (PHINode::classof(cln)) {
          PHINode *phi = (PHINode *) cln;
          phi->removeIncomingValue(phi->getParent());
        }
      }
}


// Construct the kernel BasicBlock
//
void ModuloScheduling::constructKernel(BasicBlock *prologue,
                                       BasicBlock *kernel,
                                       BasicBlock *epilogue)
{
  //*************fill instructions in the kernel****************
  InstListType & kernel_ist = kernel->getInstList();
  BranchInst *brchInst;
  PHINode *phiInst, *phiCln;

  for (unsigned i = 0; i < coreSchedule.size(); ++i)
    for (unsigned j = 0; j < coreSchedule[i].size(); ++j)
      if (coreSchedule[i][j]) {

        // Take care of branch instruction differently with normal instructions
        if (BranchInst::classof(coreSchedule[i][j]->getInst())) {
          brchInst = (BranchInst *) coreSchedule[i][j]->getInst();
          continue;
        }
        // Take care of PHINode instruction differently with normal instructions
        if (PHINode::classof(coreSchedule[i][j]->getInst())) {
          phiInst = (PHINode *) coreSchedule[i][j]->getInst();
          Instruction *cln = cloneInstSetMemory(phiInst);
          kernel_ist.insert(kernel_ist.back(), cln);
          phiCln = (PHINode *) cln;
          continue;
        }
        //for normal instructions: made a clone and insert it in the kernel_ist
        Instruction *cln =
            cloneInstSetMemory((Instruction *) coreSchedule[i][j]->
                               getInst());
        kernel_ist.insert(kernel_ist.back(), cln);
      }
  // The two incoming BasicBlock for PHINode is the prologue and the kernel
  // (itself)
  phiCln->setIncomingBlock(0, prologue);
  phiCln->setIncomingBlock(1, kernel);

  // The incoming value for the kernel (itself) is the new value which is
  // computed in the kernel
  Instruction *originalVal = (Instruction *) phiInst->getIncomingValue(1);
  phiCln->setIncomingValue(1, originalVal->getClone());

  // Make a clone of the branch instruction and insert it in the end
  BranchInst *cln = (BranchInst *) cloneInstSetMemory(brchInst);
  kernel_ist.insert(kernel_ist.back(), cln);

  // delete the unconditional branch instruction, which is generated when
  // splitting the basicBlock
  kernel_ist.erase(--kernel_ist.end());

  // set the first successor to itself
  ((BranchInst *) cln)->setSuccessor(0, kernel);
  // set the second successor to eiplogue
  ((BranchInst *) cln)->setSuccessor(1, epilogue);

  //*****change the condition*******

  //get the condition instruction
  Instruction *cond = (Instruction *) cln->getCondition();

  //get the condition's second operand, it should be a constant
  Value *operand = cond->getOperand(1);
  assert(ConstantSInt::classof(operand));

  //change the constant in the condtion instruction
  ConstantSInt *iteTimes =
      ConstantSInt::get(operand->getType(),
                        ((ConstantSInt *) operand)->getValue() - II + 1);
  cond->setOperand(1, iteTimes);

}


// Construct the epilogue 
//
void ModuloScheduling::constructEpilogue(BasicBlock *epilogue,
                                         BasicBlock *succ_bb)
{

  //compute the schedule for epilogue
  vvNodeType &tempSchedule_epilogue =
      *(new std::vector<std::vector<ModuloSchedGraphNode*> >(schedule));
  unsigned scheduleSize = schedule.size();
  int round = 0;
  while (round < ceil(1.0 * scheduleSize / II) - 1) {
    round++;
    for (unsigned i = 0; i < scheduleSize; i++) {
      if (i + round * II >= scheduleSize)
        break;
      for (unsigned j = 0; j < schedule[i].size(); j++)
        if (schedule[i + round * II][j]) {
          assert(tempSchedule_epilogue[i][j] == NULL
                 && "table not consitant with core table");

          //move the schdule one iteration behind and overlap
          tempSchedule_epilogue[i][j] = schedule[i + round * II][j];
        }
    }
  }

  //fill in the epilogue
  InstListType & epilogue_ist = epilogue->getInstList();
  for (unsigned i = II; i < scheduleSize; i++)
    for (unsigned j = 0; j < tempSchedule_epilogue[i].size(); j++)
      if (tempSchedule_epilogue[i][j]) {
        Instruction *inst =
            (Instruction *) tempSchedule_epilogue[i][j]->getInst();

        //BranchInst and PHINode should be treated differently
        //BranchInst:unecessary, simly omitted
        //PHINode: omitted
        if (!BranchInst::classof(inst) && !PHINode::classof(inst)) {
          //make a clone instruction and insert it into the epilogue
          Instruction *cln = cloneInstSetMemory(inst);
          epilogue_ist.push_front(cln);
        }
      }

  //*************delete the original instructions****************//
  //to delete the original instructions, we have to make sure their use is zero

  //update original core instruction's uses, using its clone instread
  for (unsigned i = 0; i < II; i++)
    for (unsigned j = 0; j < coreSchedule[i].size(); j++) {
      if (coreSchedule[i][j])
        updateUseWithClone((Instruction *) coreSchedule[i][j]->getInst());
    }

  //erase these instructions
  for (unsigned i = 0; i < II; i++)
    for (unsigned j = 0; j < coreSchedule[i].size(); j++)
      if (coreSchedule[i][j]) {
        Instruction *ist = (Instruction *) coreSchedule[i][j]->getInst();
        ist->getParent()->getInstList().erase(ist);
      }
  //**************************************************************//


  //finally, insert an unconditional branch instruction at the end
  epilogue_ist.push_back(new BranchInst(succ_bb));

}


//------------------------------------------------------------------------------
//this function replace the value(instruction) ist in other instructions with
//its latest clone i.e. after this function is called, the ist is not used
//anywhere and it can be erased.
//------------------------------------------------------------------------------
void ModuloScheduling::updateUseWithClone(Instruction * ist)
{

  while (ist->use_size() > 0) {
    bool destroyed = false;

    //other instruction is using this value ist
    assert(Instruction::classof(*ist->use_begin()));
    Instruction *inst = (Instruction *) (*ist->use_begin());

    for (unsigned i = 0; i < inst->getNumOperands(); i++)
      if (inst->getOperand(i) == ist && ist->getClone()) {
        // if the instruction is TmpInstruction, simly delete it because it has
        // no parent and it does not belongs to any BasicBlock
        if (TmpInstruction::classof(inst)) {
          delete inst;
          destroyed = true;
          break;
        }

        //otherwise, set the instruction's operand to the value's clone
        inst->setOperand(i, ist->getClone());

        //the use from the original value ist is destroyed
        destroyed = true;
        break;
      }
    if (!destroyed) {
      //if the use can not be destroyed , something is wrong
      inst->dump();
      assert(0 && "this use can not be destroyed");
    }
  }

}


//********************************************************
//this function clear all clone mememoy
//i.e. set all instruction's clone memory to NULL
//*****************************************************
void ModuloScheduling::clearCloneMemory()
{
  for (unsigned i = 0; i < coreSchedule.size(); i++)
    for (unsigned j = 0; j < coreSchedule[i].size(); j++)
      if (coreSchedule[i][j])
        ((Instruction *) coreSchedule[i][j]->getInst())->clearClone();

}


//******************************************************************************
// this function make a clone of the instruction orn the cloned instruction will
// use the orn's operands' latest clone as its operands it is done this way
// because LLVM is in SSA form and we should use the correct value
//this fuction also update the instruction orn's latest clone memory
//******************************************************************************
Instruction *ModuloScheduling::cloneInstSetMemory(Instruction * orn)
{
  // make a clone instruction
  Instruction *cln = orn->clone();

  // update the operands
  for (unsigned k = 0; k < orn->getNumOperands(); k++) {
    const Value *op = orn->getOperand(k);
    if (Instruction::classof(op) && ((Instruction *) op)->getClone()) {
      Instruction *op_inst = (Instruction *) op;
      cln->setOperand(k, op_inst->getClone());
    }
  }

  // update clone memory
  orn->setClone(cln);
  return cln;
}



bool ModuloScheduling::ScheduleNode(ModuloSchedGraphNode * node,
                                    unsigned start, unsigned end,
                                    NodeVec & nodeScheduled)
{
  const TargetSchedInfo & msi = target.getSchedInfo();
  unsigned int numIssueSlots = msi.maxNumIssueTotal;

  if (ModuloScheduling::printScheduleProcess())
    DEBUG_PRINT(std::cerr << "startTime= " << start << " endTime= " << end << "\n");
  bool isScheduled = false;
  for (unsigned i = start; i <= end; i++) {
    if (ModuloScheduling::printScheduleProcess())
      DEBUG_PRINT(std::cerr << " now try cycle " << i << ":" << "\n");
    for (unsigned j = 0; j < numIssueSlots; j++) {
      unsigned int core_i = i % II;
      unsigned int core_j = j;
      if (ModuloScheduling::printScheduleProcess())
        DEBUG_PRINT(std::cerr << "\t Trying slot " << j << "...........");
      //check the resouce table, make sure there is no resource conflicts
      const Instruction *instr = node->getInst();
      MachineCodeForInstruction & tempMvec =
          MachineCodeForInstruction::get(instr);
      bool resourceConflict = false;
      const TargetInstrInfo & mii = msi.getInstrInfo();

      if (coreSchedule.size() < core_i + 1
          || !coreSchedule[core_i][core_j]) {
        //this->dumpResourceUsageTable();
        int latency = 0;
        for (unsigned k = 0; k < tempMvec.size(); k++) {
          MachineInstr *minstr = tempMvec[k];
          InstrRUsage rUsage = msi.getInstrRUsage(minstr->getOpCode());
          std::vector < std::vector < resourceId_t > >resources
              = rUsage.resourcesByCycle;
          updateResourceTable(resources, i + latency);
          latency += std::max(mii.minLatency(minstr->getOpCode()), 1);
        }

        //this->dumpResourceUsageTable();

        latency = 0;
        if (resourceTableNegative()) {

          //undo-update the resource table
          for (unsigned k = 0; k < tempMvec.size(); k++) {
            MachineInstr *minstr = tempMvec[k];
            InstrRUsage rUsage = msi.getInstrRUsage(minstr->getOpCode());
            std::vector < std::vector < resourceId_t > >resources
                = rUsage.resourcesByCycle;
            undoUpdateResourceTable(resources, i + latency);
            latency += std::max(mii.minLatency(minstr->getOpCode()), 1);
          }
          resourceConflict = true;
        }
      }
      if (!resourceConflict && !coreSchedule[core_i][core_j]) {
        if (ModuloScheduling::printScheduleProcess()) {
          DEBUG_PRINT(std::cerr << " OK!" << "\n");
          DEBUG_PRINT(std::cerr << "Node " << node->getNodeId() << " scheduled.\n");
        }
        //schedule[i][j]=node;
        while (schedule.size() <= i) {
          std::vector < ModuloSchedGraphNode * >*newCycle =
              new std::vector < ModuloSchedGraphNode * >();
          for (unsigned k = 0; k < numIssueSlots; k++)
            newCycle->push_back(NULL);
          schedule.push_back(*newCycle);
        }
        std::vector<ModuloSchedGraphNode*>::iterator startIterator;
        startIterator = schedule[i].begin();
        schedule[i].insert(startIterator + j, node);
        startIterator = schedule[i].begin();
        schedule[i].erase(startIterator + j + 1);

        //update coreSchedule
        //coreSchedule[core_i][core_j]=node;
        while (coreSchedule.size() <= core_i) {
          std::vector<ModuloSchedGraphNode*> *newCycle =
              new std::vector<ModuloSchedGraphNode*>();
          for (unsigned k = 0; k < numIssueSlots; k++)
            newCycle->push_back(NULL);
          coreSchedule.push_back(*newCycle);
        }

        startIterator = coreSchedule[core_i].begin();
        coreSchedule[core_i].insert(startIterator + core_j, node);
        startIterator = coreSchedule[core_i].begin();
        coreSchedule[core_i].erase(startIterator + core_j + 1);

        node->setSchTime(i);
        isScheduled = true;
        nodeScheduled.push_back(node);

        break;
      } else if (coreSchedule[core_i][core_j]) {
        if (ModuloScheduling::printScheduleProcess())
          DEBUG_PRINT(std::cerr << " Slot not available\n");
      } else {
        if (ModuloScheduling::printScheduleProcess())
          DEBUG_PRINT(std::cerr << " Resource conflicts\n");
      }
    }
    if (isScheduled)
      break;
  }
  //assert(nodeScheduled &&"this node can not be scheduled?");
  return isScheduled;
}


void ModuloScheduling::updateResourceTable(Resources useResources,
                                           int startCycle)
{
  for (unsigned i = 0; i < useResources.size(); i++) {
    int absCycle = startCycle + i;
    int coreCycle = absCycle % II;
    std::vector<std::pair<int,int> > &resourceRemained =
        resourceTable[coreCycle];
    std::vector < unsigned int >&resourceUsed = useResources[i];
    for (unsigned j = 0; j < resourceUsed.size(); j++) {
      for (unsigned k = 0; k < resourceRemained.size(); k++)
        if ((int) resourceUsed[j] == resourceRemained[k].first) {
          resourceRemained[k].second--;
        }
    }
  }
}

void ModuloScheduling::undoUpdateResourceTable(Resources useResources,
                                               int startCycle)
{
  for (unsigned i = 0; i < useResources.size(); i++) {
    int absCycle = startCycle + i;
    int coreCycle = absCycle % II;
    std::vector<std::pair<int,int> > &resourceRemained =
        resourceTable[coreCycle];
    std::vector < unsigned int >&resourceUsed = useResources[i];
    for (unsigned j = 0; j < resourceUsed.size(); j++) {
      for (unsigned k = 0; k < resourceRemained.size(); k++)
        if ((int) resourceUsed[j] == resourceRemained[k].first) {
          resourceRemained[k].second++;
        }
    }
  }
}


//-----------------------------------------------------------------------
// Function: resourceTableNegative
// return value:
//   return false if any element in the resouceTable is negative
//   otherwise return true
// Purpose:

//   this function is used to determine if an instruction is eligible for
//   schedule at certain cycle
//-----------------------------------------------------------------------


bool ModuloScheduling::resourceTableNegative()
{
  assert(resourceTable.size() == (unsigned) II
         && "resouceTable size must be equal to II");
  bool isNegative = false;
  for (unsigned i = 0; i < resourceTable.size(); i++)
    for (unsigned j = 0; j < resourceTable[i].size(); j++) {
      if (resourceTable[i][j].second < 0) {
        isNegative = true;
        break;
      }
    }
  return isNegative;
}


//----------------------------------------------------------------------
// Function: dumpResouceUsageTable
// Purpose:
//   print out ResouceTable for debug
//
//------------------------------------------------------------------------

void ModuloScheduling::dumpResourceUsageTable()
{
  DEBUG_PRINT(std::cerr << "dumping resource usage table\n");
  for (unsigned i = 0; i < resourceTable.size(); i++) {
    for (unsigned j = 0; j < resourceTable[i].size(); j++)
      DEBUG_PRINT(std::cerr << resourceTable[i][j].first 
            << ":" << resourceTable[i][j].second << " ");
    DEBUG_PRINT(std::cerr << "\n");
  }

}

//----------------------------------------------------------------------
//Function: dumpSchedule
//Purpose:
//       print out thisSchedule for debug
//
//-----------------------------------------------------------------------
void ModuloScheduling::dumpSchedule(vvNodeType thisSchedule)
{
  const TargetSchedInfo & msi = target.getSchedInfo();
  unsigned numIssueSlots = msi.maxNumIssueTotal;
  for (unsigned i = 0; i < numIssueSlots; i++)
    DEBUG_PRINT(std::cerr << "\t#");
  DEBUG_PRINT(std::cerr << "\n");
  for (unsigned i = 0; i < thisSchedule.size(); i++) {
    DEBUG_PRINT(std::cerr << "cycle" << i << ": ");
    for (unsigned j = 0; j < thisSchedule[i].size(); j++)
      if (thisSchedule[i][j] != NULL)
        DEBUG_PRINT(std::cerr << thisSchedule[i][j]->getNodeId() << "\t");
      else
        DEBUG_PRINT(std::cerr << "\t");
    DEBUG_PRINT(std::cerr << "\n");
  }
}


//----------------------------------------------------
//Function: dumpScheduling
//Purpose:
//   print out the schedule and coreSchedule for debug      
//
//-------------------------------------------------------

void ModuloScheduling::dumpScheduling()
{
  DEBUG_PRINT(std::cerr << "dump schedule:" << "\n");
  const TargetSchedInfo & msi = target.getSchedInfo();
  unsigned numIssueSlots = msi.maxNumIssueTotal;
  for (unsigned i = 0; i < numIssueSlots; i++)
    DEBUG_PRINT(std::cerr << "\t#");
  DEBUG_PRINT(std::cerr << "\n");
  for (unsigned i = 0; i < schedule.size(); i++) {
    DEBUG_PRINT(std::cerr << "cycle" << i << ": ");
    for (unsigned j = 0; j < schedule[i].size(); j++)
      if (schedule[i][j] != NULL)
        DEBUG_PRINT(std::cerr << schedule[i][j]->getNodeId() << "\t");
      else
        DEBUG_PRINT(std::cerr << "\t");
    DEBUG_PRINT(std::cerr << "\n");
  }

  DEBUG_PRINT(std::cerr << "dump coreSchedule:" << "\n");
  for (unsigned i = 0; i < numIssueSlots; i++)
    DEBUG_PRINT(std::cerr << "\t#");
  DEBUG_PRINT(std::cerr << "\n");
  for (unsigned i = 0; i < coreSchedule.size(); i++) {
    DEBUG_PRINT(std::cerr << "cycle" << i << ": ");
    for (unsigned j = 0; j < coreSchedule[i].size(); j++)
      if (coreSchedule[i][j] != NULL)
        DEBUG_PRINT(std::cerr << coreSchedule[i][j]->getNodeId() << "\t");
      else
        DEBUG_PRINT(std::cerr << "\t");
    DEBUG_PRINT(std::cerr << "\n");
  }
}



//---------------------------------------------------------------------------
// Function: ModuloSchedulingPass
// 
// Purpose:
//   Entry point for Modulo Scheduling
//   Schedules LLVM instruction
//   
//---------------------------------------------------------------------------

namespace {
  class ModuloSchedulingPass:public FunctionPass {
    const TargetMachine &target;

  public:
    ModuloSchedulingPass(const TargetMachine &T):target(T) {}

    const char *getPassName() const {
      return "Modulo Scheduling";
    }

    // getAnalysisUsage - We use LiveVarInfo...
        virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      //AU.addRequired(FunctionLiveVarInfo::ID);
    } bool runOnFunction(Function & F);
  };
}                               // end anonymous namespace


bool ModuloSchedulingPass::runOnFunction(Function &F)
{
  ModuloSchedGraphSet *graphSet = new ModuloSchedGraphSet(&F, target);
  ModuloSchedulingSet ModuloSchedulingSet(*graphSet);

  return false;
}


Pass *createModuloSchedulingPass(const TargetMachine & tgt)
{
  return new ModuloSchedulingPass(tgt);
}
