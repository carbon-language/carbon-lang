//===-- SchedPriorities.h - Encapsulate scheduling heuristics -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// Strategy:
//    Priority ordering rules:
//    (1) Max delay, which is the order of the heap S.candsAsHeap.
//    (2) Instruction that frees up a register.
//    (3) Instruction that has the maximum number of dependent instructions.
//    Note that rules 2 and 3 are only used if issue conflicts prevent
//    choosing a higher priority instruction by rule 1.
//
//===----------------------------------------------------------------------===//

#include "SchedPriorities.h"
#include "../../Target/SparcV9/LiveVar/FunctionLiveVarInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/CFG.h"
#include "Support/PostOrderIterator.h"

namespace llvm {

std::ostream &operator<<(std::ostream &os, const NodeDelayPair* nd) {
  return os << "Delay for node " << nd->node->getNodeId()
	    << " = " << (long)nd->delay << "\n";
}


SchedPriorities::SchedPriorities(const Function *, const SchedGraph *G,
                                 FunctionLiveVarInfo &LVI)
  : curTime(0), graph(G), methodLiveVarInfo(LVI),
    nodeDelayVec(G->getNumNodes(), INVALID_LATENCY), // make errors obvious
    earliestReadyTimeForNode(G->getNumNodes(), 0),
    earliestReadyTime(0),
    nextToTry(candsAsHeap.begin())
{
  computeDelays(graph);
}


void
SchedPriorities::initialize() {
  initializeReadyHeap(graph);
}


void
SchedPriorities::computeDelays(const SchedGraph* graph) {
  po_iterator<const SchedGraph*> poIter = po_begin(graph), poEnd =po_end(graph);
  for ( ; poIter != poEnd; ++poIter) {
    const SchedGraphNode* node = *poIter;
    cycles_t nodeDelay;
    if (node->beginOutEdges() == node->endOutEdges())
      nodeDelay = node->getLatency();
    else {
      // Iterate over the out-edges of the node to compute delay
      nodeDelay = 0;
      for (SchedGraphNode::const_iterator E=node->beginOutEdges();
           E != node->endOutEdges(); ++E) {
        cycles_t sinkDelay = getNodeDelay((SchedGraphNode*)(*E)->getSink());
        nodeDelay = std::max(nodeDelay, sinkDelay + (*E)->getMinDelay());
      }
    }
    getNodeDelayRef(node) = nodeDelay;
  }
}


void
SchedPriorities::initializeReadyHeap(const SchedGraph* graph) {
  const SchedGraphNode* graphRoot = (const SchedGraphNode*)graph->getRoot();
  assert(graphRoot->getMachineInstr() == NULL && "Expect dummy root");
  
  // Insert immediate successors of dummy root, which are the actual roots
  sg_succ_const_iterator SEnd = succ_end(graphRoot);
  for (sg_succ_const_iterator S = succ_begin(graphRoot); S != SEnd; ++S)
    this->insertReady(*S);
  
#undef TEST_HEAP_CONVERSION
#ifdef TEST_HEAP_CONVERSION
  std::cerr << "Before heap conversion:\n";
  copy(candsAsHeap.begin(), candsAsHeap.end(),
       ostream_iterator<NodeDelayPair*>(std::cerr,"\n"));
#endif
  
  candsAsHeap.makeHeap();
  
  nextToTry = candsAsHeap.begin();
  
#ifdef TEST_HEAP_CONVERSION
  std::cerr << "After heap conversion:\n";
  copy(candsAsHeap.begin(), candsAsHeap.end(),
       ostream_iterator<NodeDelayPair*>(std::cerr,"\n"));
#endif
}

void
SchedPriorities::insertReady(const SchedGraphNode* node) {
  candsAsHeap.insert(node, nodeDelayVec[node->getNodeId()]);
  candsAsSet.insert(node);
  mcands.clear(); // ensure reset choices is called before any more choices
  earliestReadyTime = std::min(earliestReadyTime,
                       getEarliestReadyTimeForNode(node));
  
  if (SchedDebugLevel >= Sched_PrintSchedTrace) {
    std::cerr << " Node " << node->getNodeId() << " will be ready in Cycle "
              << getEarliestReadyTimeForNode(node) << "; "
              << " Delay = " <<(long)getNodeDelay(node) << "; Instruction: \n"
              << "        " << *node->getMachineInstr() << "\n";
  }
}

void
SchedPriorities::issuedReadyNodeAt(cycles_t curTime,
				   const SchedGraphNode* node) {
  candsAsHeap.removeNode(node);
  candsAsSet.erase(node);
  mcands.clear(); // ensure reset choices is called before any more choices
  
  if (earliestReadyTime == getEarliestReadyTimeForNode(node)) {
    // earliestReadyTime may have been due to this node, so recompute it
    earliestReadyTime = HUGE_LATENCY;
    for (NodeHeap::const_iterator I=candsAsHeap.begin();
         I != candsAsHeap.end(); ++I)
      if (candsAsHeap.getNode(I)) {
        earliestReadyTime = 
          std::min(earliestReadyTime, 
                   getEarliestReadyTimeForNode(candsAsHeap.getNode(I)));
      }
  }
  
  // Now update ready times for successors
  for (SchedGraphNode::const_iterator E=node->beginOutEdges();
       E != node->endOutEdges(); ++E) {
    cycles_t& etime =
      getEarliestReadyTimeForNodeRef((SchedGraphNode*)(*E)->getSink());
    etime = std::max(etime, curTime + (*E)->getMinDelay());
  }    
}


//----------------------------------------------------------------------
// Priority ordering rules:
// (1) Max delay, which is the order of the heap S.candsAsHeap.
// (2) Instruction that frees up a register.
// (3) Instruction that has the maximum number of dependent instructions.
// Note that rules 2 and 3 are only used if issue conflicts prevent
// choosing a higher priority instruction by rule 1.
//----------------------------------------------------------------------

inline int
SchedPriorities::chooseByRule1(std::vector<candIndex>& mcands) {
  return (mcands.size() == 1)? 0	// only one choice exists so take it
			     : -1;	// -1 indicates multiple choices
}

inline int
SchedPriorities::chooseByRule2(std::vector<candIndex>& mcands) {
  assert(mcands.size() >= 1 && "Should have at least one candidate here.");
  for (unsigned i=0, N = mcands.size(); i < N; i++)
    if (instructionHasLastUse(methodLiveVarInfo,
			      candsAsHeap.getNode(mcands[i])))
      return i;
  return -1;
}

inline int
SchedPriorities::chooseByRule3(std::vector<candIndex>& mcands) {
  assert(mcands.size() >= 1 && "Should have at least one candidate here.");
  int maxUses = candsAsHeap.getNode(mcands[0])->getNumOutEdges();	
  int indexWithMaxUses = 0;
  for (unsigned i=1, N = mcands.size(); i < N; i++) {
    int numUses = candsAsHeap.getNode(mcands[i])->getNumOutEdges();
    if (numUses > maxUses) {
      maxUses = numUses;
      indexWithMaxUses = i;
    }
  }
  return indexWithMaxUses; 
}

const SchedGraphNode*
SchedPriorities::getNextHighest(const SchedulingManager& S,
				cycles_t curTime) {
  int nextIdx = -1;
  const SchedGraphNode* nextChoice = NULL;
  
  if (mcands.size() == 0)
    findSetWithMaxDelay(mcands, S);
  
  while (nextIdx < 0 && mcands.size() > 0) {
    nextIdx = chooseByRule1(mcands);	 // rule 1
      
    if (nextIdx == -1)
      nextIdx = chooseByRule2(mcands); // rule 2
      
    if (nextIdx == -1)
      nextIdx = chooseByRule3(mcands); // rule 3
      
    if (nextIdx == -1)
      nextIdx = 0;			 // default to first choice by delays
      
    // We have found the next best candidate.  Check if it ready in
    // the current cycle, and if it is feasible.
    // If not, remove it from mcands and continue.  Refill mcands if
    // it becomes empty.
    nextChoice = candsAsHeap.getNode(mcands[nextIdx]);
    if (getEarliestReadyTimeForNode(nextChoice) > curTime
        || ! instrIsFeasible(S, nextChoice->getMachineInstr()->getOpcode()))
    {
      mcands.erase(mcands.begin() + nextIdx);
      nextIdx = -1;
      if (mcands.size() == 0)
        findSetWithMaxDelay(mcands, S);
    }
  }
  
  if (nextIdx >= 0) {
    mcands.erase(mcands.begin() + nextIdx);
    return nextChoice;
  } else
    return NULL;
}


void
SchedPriorities::findSetWithMaxDelay(std::vector<candIndex>& mcands,
				     const SchedulingManager& S)
{
  if (mcands.size() == 0 && nextToTry != candsAsHeap.end())
    { // out of choices at current maximum delay;
      // put nodes with next highest delay in mcands
      candIndex next = nextToTry;
      cycles_t maxDelay = candsAsHeap.getDelay(next);
      for (; next != candsAsHeap.end()
	     && candsAsHeap.getDelay(next) == maxDelay; ++next)
	mcands.push_back(next);
      
      nextToTry = next;
      
      if (SchedDebugLevel >= Sched_PrintSchedTrace) {
        std::cerr << "    Cycle " << (long)getTime() << ": "
                  << "Next highest delay = " << (long)maxDelay << " : "
                  << mcands.size() << " Nodes with this delay: ";
        for (unsigned i=0; i < mcands.size(); i++)
          std::cerr << candsAsHeap.getNode(mcands[i])->getNodeId() << ", ";
        std::cerr << "\n";
      }
    }
}


bool
SchedPriorities::instructionHasLastUse(FunctionLiveVarInfo &LVI,
				       const SchedGraphNode* graphNode) {
  const MachineInstr *MI = graphNode->getMachineInstr();
  
  hash_map<const MachineInstr*, bool>::const_iterator
    ui = lastUseMap.find(MI);
  if (ui != lastUseMap.end())
    return ui->second;
  
  // else check if instruction is a last use and save it in the hash_map
  bool hasLastUse = false;
  const BasicBlock* bb = graphNode->getMachineBasicBlock().getBasicBlock();
  const ValueSet &LVs = LVI.getLiveVarSetBeforeMInst(MI, bb);
  
  for (MachineInstr::const_val_op_iterator OI = MI->begin(), OE = MI->end();
       OI != OE; ++OI)
    if (!LVs.count(*OI)) {
      hasLastUse = true;
      break;
    }

  return lastUseMap[MI] = hasLastUse;
}

} // End llvm namespace
