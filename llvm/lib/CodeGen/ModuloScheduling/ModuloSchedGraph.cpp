//===- ModuloSchedGraph.cpp - Graph datastructure for Modulo Scheduling ---===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetSchedInfo.h"
#include "Support/StringExtras.h"
#include "Support/STLExtras.h"
#include "Support/hash_map"
#include "Support/Statistic.h"
#include "ModuloScheduling.h"
#include "ModuloSchedGraph.h"
#include <algorithm>
#include <ostream>
#include <vector>
// FIXME: Should be using #include <cmath>
#include <math.h>
//#include <swig.h>

#define UNIDELAY 1

//*********************** Internal Data Structures *************************/

// The following two types need to be classes, not typedefs, so we can use
// opaque declarations in SchedGraph.h
// 
struct RefVec:public std::vector<std::pair<ModuloSchedGraphNode*,int> > {
  typedef std::vector<std::pair<ModuloSchedGraphNode*,
                                int> >::iterator iterator;
  typedef std::vector<std::pair<ModuloSchedGraphNode*,
                                int> >::const_iterator const_iterator;
};

struct RegToRefVecMap:public hash_map<int,RefVec> {
  typedef hash_map<int,RefVec>::iterator iterator;
  typedef hash_map<int,RefVec>::const_iterator const_iterator;
};

struct ValueToDefVecMap:public hash_map<const Instruction*,RefVec> {
  typedef hash_map<const Instruction*, RefVec>::iterator iterator;
  typedef hash_map<const Instruction*,
                        RefVec>::const_iterator const_iterator;
};

// class Modulo SchedGraphNode

ModuloSchedGraphNode::ModuloSchedGraphNode(unsigned int in_nodeId,
                                           const BasicBlock * in_bb,
                                           const Instruction * in_inst,
                                           int indexInBB,
                                           const TargetMachine & target)
:SchedGraphNodeCommon(in_nodeId, indexInBB), inst(in_inst)
{
  if (inst) {
    //FIXME: find the latency 
    //currently setthe latency to zero
    latency = 0;
  }
}

//class ModuloScheGraph

void ModuloSchedGraph::addDummyEdges()
{
  assert(graphRoot->outEdges.size() == 0);

  for (const_iterator I = begin(); I != end(); ++I) {
    ModuloSchedGraphNode *node = (ModuloSchedGraphNode *) ((*I).second);
    assert(node != graphRoot && node != graphLeaf);
    if (node->beginInEdges() == node->endInEdges())
      (void) new SchedGraphEdge(graphRoot, node, SchedGraphEdge::CtrlDep,
                                SchedGraphEdge::NonDataDep, 0);
    if (node->beginOutEdges() == node->endOutEdges())
      (void) new SchedGraphEdge(node, graphLeaf, SchedGraphEdge::CtrlDep,
                                SchedGraphEdge::NonDataDep, 0);
  }
}

bool isDefinition(const Instruction *I)
{
  //if(TerminatorInst::classof(I)||FreeInst::classof(I) || StoreInst::classof(I) || CallInst::classof(I))
  if (!I->hasName())
    return false;
  else
    return true;
}

void ModuloSchedGraph::addDefUseEdges(const BasicBlock *bb)
{
  //collect def instructions, store them in vector
  //  const TargetInstrInfo& mii = target.getInstrInfo();
  const TargetInstrInfo & mii = target.getInstrInfo();

  typedef std::vector < ModuloSchedGraphNode * >DefVec;
  DefVec defVec;

  //find those def instructions
  for (BasicBlock::const_iterator I = bb->begin(), E = bb->end(); I != E; ++I) {
    if (I->getType() != Type::VoidTy) {
      defVec.push_back(this->getGraphNodeForInst(I));
    }
  }

  for (unsigned int i = 0; i < defVec.size(); i++) {
    for (Value::use_const_iterator I = defVec[i]->getInst()->use_begin();
         I != defVec[i]->getInst()->use_end(); I++) {
      //for each use of a def, add a flow edge from the def instruction to the
      //ref instruction

      const Instruction *value = defVec[i]->getInst();
      Instruction *inst = (Instruction *) (*I);
      ModuloSchedGraphNode *node = NULL;

      for (BasicBlock::const_iterator I = bb->begin(), E = bb->end();
           I != E; ++I)
        if ((const Instruction *) I == inst) {
          node = (*this)[inst];
          break;
        }

      assert(inst != NULL && " Use not an Instruction ?");

      if (node == NULL)         //inst is not an instruction in this block
      {
      } else {
        // Add a flow edge from the def instruction to the ref instruction

        // self loop will not happen in SSA form
        assert(defVec[i] != node && "same node?");

        // This is a true dependence, so the delay is equal to the delay of the
        // pred node.
        int delay = 0;
        MachineCodeForInstruction & tempMvec =
            MachineCodeForInstruction::get(value);
        for (unsigned j = 0; j < tempMvec.size(); j++) {
          MachineInstr *temp = tempMvec[j];
          //delay +=mii.minLatency(temp->getOpCode());
          delay = std::max(delay, mii.minLatency(temp->getOpCode()));
        }

        SchedGraphEdge *trueEdge =
            new SchedGraphEdge(defVec[i], node, value,
                               SchedGraphEdge::TrueDep, delay);

        // if the ref instruction is before the def instrution
        // then the def instruction must be a phi instruction 
        // add an anti-dependence edge to from the ref instruction to the def
        // instruction
        if (node->getOrigIndexInBB() < defVec[i]->getOrigIndexInBB()) {
          assert(PHINode::classof(inst)
                 && "the ref instruction befre def is not PHINode?");
          trueEdge->setIteDiff(1);
        }

      }

    }
  }
}

void ModuloSchedGraph::addCDEdges(const BasicBlock * bb) {
  // find the last instruction in the basic block
  // see if it is an branch instruction. 
  // If yes, then add an edge from each node expcept the last node to the last
  // node
  const Instruction *inst = &(bb->back());
  ModuloSchedGraphNode *lastNode = (*this)[inst];
  if (TerminatorInst::classof(inst))
    for (BasicBlock::const_iterator I = bb->begin(), E = bb->end(); I != E;
         I++) {
      if (inst != I) {
        ModuloSchedGraphNode *node = (*this)[I];
        //use latency of 0
        (void) new SchedGraphEdge(node, lastNode, SchedGraphEdge::CtrlDep,
                                  SchedGraphEdge::NonDataDep, 0);
      }

    }
}

static const int SG_LOAD_REF = 0;
static const int SG_STORE_REF = 1;
static const int SG_CALL_REF = 2;

static const unsigned int SG_DepOrderArray[][3] = {
  {SchedGraphEdge::NonDataDep,
   SchedGraphEdge::AntiDep,
   SchedGraphEdge::AntiDep},
  {SchedGraphEdge::TrueDep,
   SchedGraphEdge::OutputDep,
   SchedGraphEdge::TrueDep | SchedGraphEdge::OutputDep},
  {SchedGraphEdge::TrueDep,
   SchedGraphEdge::AntiDep | SchedGraphEdge::OutputDep,
   SchedGraphEdge::TrueDep | SchedGraphEdge::AntiDep
   | SchedGraphEdge::OutputDep}
};


// Add a dependence edge between every pair of machine load/store/call
// instructions, where at least one is a store or a call.
// Use latency 1 just to ensure that memory operations are ordered;
// latency does not otherwise matter (true dependences enforce that).
// 
void ModuloSchedGraph::addMemEdges(const BasicBlock * bb) {

  std::vector<ModuloSchedGraphNode*> memNodeVec;

  //construct the memNodeVec
  for (BasicBlock::const_iterator I = bb->begin(), E = bb->end(); I != E; ++I) {
    if (LoadInst::classof(I) || StoreInst::classof(I)
        || CallInst::classof(I)) {
      ModuloSchedGraphNode *node = (*this)[(const Instruction *) I];
      memNodeVec.push_back(node);
    }
  }

  // Instructions in memNodeVec are in execution order within the basic block,
  // so simply look at all pairs <memNodeVec[i], memNodeVec[j: j > i]>.
  // 
  for (unsigned im = 0, NM = memNodeVec.size(); im < NM; im++) {
    const Instruction *fromInst = memNodeVec[im]->getInst();
    int fromType = CallInst::classof(fromInst) ? SG_CALL_REF
        : LoadInst::classof(fromInst) ? SG_LOAD_REF : SG_STORE_REF;
    for (unsigned jm = im + 1; jm < NM; jm++) {
      const Instruction *toInst = memNodeVec[jm]->getInst();
      int toType = CallInst::classof(toInst) ? SG_CALL_REF
          : LoadInst::classof(toInst) ? SG_LOAD_REF : SG_STORE_REF;
      if (fromType != SG_LOAD_REF || toType != SG_LOAD_REF) {
        (void) new SchedGraphEdge(memNodeVec[im], memNodeVec[jm],
                                  SchedGraphEdge::MemoryDep,
                                  SG_DepOrderArray[fromType][toType], 1);

        SchedGraphEdge *edge =
            new SchedGraphEdge(memNodeVec[jm], memNodeVec[im],
                               SchedGraphEdge::MemoryDep,
                               SG_DepOrderArray[toType][fromType], 1);
        edge->setIteDiff(1);

      }
    }
  }
}



void ModuloSchedGraph::buildNodesforBB(const TargetMachine &target,
                                       const BasicBlock *bb,
                                    std::vector<ModuloSchedGraphNode*> &memNode,
                                       RegToRefVecMap &regToRefVecMap,
                                       ValueToDefVecMap &valueToDefVecMap)
{
  //const TargetInstrInfo& mii=target.getInstrInfo();

  //Build graph nodes for each LLVM instruction and gather def/use info.
  //Do both together in a single pass over all machine instructions.

  int i = 0;
  for (BasicBlock::const_iterator I = bb->begin(), E = bb->end(); I != E; 
       ++I) {
    ModuloSchedGraphNode *node =
        new ModuloSchedGraphNode(getNumNodes(), bb, I, i, target);
    i++;
    this->noteModuloSchedGraphNodeForInst(I, node);
  }

  //this function finds some info about instruction in basic block for later use
  //findDefUseInfoAtInstr(target, node,
  //memNode,regToRefVecMap,valueToDefVecMap);
}


bool ModuloSchedGraph::isLoop(const BasicBlock *bb) {
  //only if the last instruction in the basicblock is branch instruction and 
  //there is at least an option to branch itself

  const Instruction *inst = &(bb->back());
  if (BranchInst::classof(inst)) {
    for (unsigned i = 0; i < ((BranchInst *) inst)->getNumSuccessors();
         i++) {
      BasicBlock *sb = ((BranchInst *) inst)->getSuccessor(i);
      if (sb == bb)
        return true;
    }
  }

  return false;

}

bool ModuloSchedGraph::isLoop() {
  //only if the last instruction in the basicblock is branch instruction and 
  //there is at least an option to branch itself

  assert(this->bb&& "the basicblock is not empty");
  const Instruction *inst = &(bb->back());
  if (BranchInst::classof(inst))
    for (unsigned i = 0; i < ((BranchInst *) inst)->getNumSuccessors();
         i++) {
      BasicBlock *sb = ((BranchInst *) inst)->getSuccessor(i);
      if (sb == bb)
        return true;
    }
  
  return false;

}

void ModuloSchedGraph::computeNodeASAP(const BasicBlock *bb) {

  //FIXME: now assume the only backward edges come from the edges from other
  //nodes to the phi Node so i will ignore all edges to the phi node; after
  //this, there shall be no recurrence.

  unsigned numNodes = bb->size();
  for (unsigned i = 2; i < numNodes + 2; i++) {
    ModuloSchedGraphNode *node = getNode(i);
    node->setPropertyComputed(false);
  }

  for (unsigned i = 2; i < numNodes + 2; i++) {
    ModuloSchedGraphNode *node = getNode(i);
    node->ASAP = 0;
    if (i == 2 || node->getNumInEdges() == 0) {
      node->setPropertyComputed(true);
      continue;
    }
    for (ModuloSchedGraphNode::const_iterator I = node->beginInEdges(), E =
         node->endInEdges(); I != E; I++) {
      SchedGraphEdge *edge = *I;
      ModuloSchedGraphNode *pred =
          (ModuloSchedGraphNode *) (edge->getSrc());
      assert(pred->getPropertyComputed()
             && "pred node property is not computed!");
      int temp =
          pred->ASAP + edge->getMinDelay() -
          edge->getIteDiff() * this->MII;
      node->ASAP = std::max(node->ASAP, temp);
    }
    node->setPropertyComputed(true);
  }
}

void ModuloSchedGraph::computeNodeALAP(const BasicBlock *bb) {
  unsigned numNodes = bb->size();
  int maxASAP = 0;
  for (unsigned i = numNodes + 1; i >= 2; i--) {
    ModuloSchedGraphNode *node = getNode(i);
    node->setPropertyComputed(false);
    //cerr<< " maxASAP= " <<maxASAP<<" node->ASAP= "<< node->ASAP<<"\n";
    maxASAP = std::max(maxASAP, node->ASAP);
  }

  //cerr<<"maxASAP is "<<maxASAP<<"\n";

  for (unsigned i = numNodes + 1; i >= 2; i--) {
    ModuloSchedGraphNode *node = getNode(i);
    node->ALAP = maxASAP;
    for (ModuloSchedGraphNode::const_iterator I =
         node->beginOutEdges(), E = node->endOutEdges(); I != E; I++) {
      SchedGraphEdge *edge = *I;
      ModuloSchedGraphNode *succ =
          (ModuloSchedGraphNode *) (edge->getSink());
      if (PHINode::classof(succ->getInst()))
        continue;
      assert(succ->getPropertyComputed()
             && "succ node property is not computed!");
      int temp =
          succ->ALAP - edge->getMinDelay() +
          edge->getIteDiff() * this->MII;
      node->ALAP = std::min(node->ALAP, temp);
    }
    node->setPropertyComputed(true);
  }
}

void ModuloSchedGraph::computeNodeMov(const BasicBlock *bb)
{
  unsigned numNodes = bb->size();
  for (unsigned i = 2; i < numNodes + 2; i++) {
    ModuloSchedGraphNode *node = getNode(i);
    node->mov = node->ALAP - node->ASAP;
    assert(node->mov >= 0
           && "move freedom for this node is less than zero? ");
  }
}


void ModuloSchedGraph::computeNodeDepth(const BasicBlock * bb)
{

  unsigned numNodes = bb->size();
  for (unsigned i = 2; i < numNodes + 2; i++) {
    ModuloSchedGraphNode *node = getNode(i);
    node->setPropertyComputed(false);
  }

  for (unsigned i = 2; i < numNodes + 2; i++) {
    ModuloSchedGraphNode *node = getNode(i);
    node->depth = 0;
    if (i == 2 || node->getNumInEdges() == 0) {
      node->setPropertyComputed(true);
      continue;
    }
    for (ModuloSchedGraphNode::const_iterator I = node->beginInEdges(), E =
         node->endInEdges(); I != E; I++) {
      SchedGraphEdge *edge = *I;
      ModuloSchedGraphNode *pred =
          (ModuloSchedGraphNode *) (edge->getSrc());
      assert(pred->getPropertyComputed()
             && "pred node property is not computed!");
      int temp = pred->depth + edge->getMinDelay();
      node->depth = std::max(node->depth, temp);
    }
    node->setPropertyComputed(true);
  }

}


void ModuloSchedGraph::computeNodeHeight(const BasicBlock *bb)
{
  unsigned numNodes = bb->size();
  for (unsigned i = numNodes + 1; i >= 2; i--) {
    ModuloSchedGraphNode *node = getNode(i);
    node->setPropertyComputed(false);
  }

  for (unsigned i = numNodes + 1; i >= 2; i--) {
    ModuloSchedGraphNode *node = getNode(i);
    node->height = 0;
    for (ModuloSchedGraphNode::const_iterator I =
         node->beginOutEdges(), E = node->endOutEdges(); I != E; ++I) {
      SchedGraphEdge *edge = *I;
      ModuloSchedGraphNode *succ =
          (ModuloSchedGraphNode *) (edge->getSink());
      if (PHINode::classof(succ->getInst()))
        continue;
      assert(succ->getPropertyComputed()
             && "succ node property is not computed!");
      node->height = std::max(node->height, succ->height + edge->getMinDelay());

    }
    node->setPropertyComputed(true);

  }

}

void ModuloSchedGraph::computeNodeProperty(const BasicBlock * bb)
{
  //FIXME: now assume the only backward edges come from the edges from other
  //nodes to the phi Node so i will ignore all edges to the phi node; after
  //this, there shall be no recurrence.

  this->computeNodeASAP(bb);
  this->computeNodeALAP(bb);
  this->computeNodeMov(bb);
  this->computeNodeDepth(bb);
  this->computeNodeHeight(bb);
}


//do not consider the backedge in these two functions:
// i.e. don't consider the edge with destination in phi node
std::vector<ModuloSchedGraphNode*>
ModuloSchedGraph::predSet(std::vector<ModuloSchedGraphNode*> set,
                          unsigned backEdgeSrc,
                          unsigned
                          backEdgeSink)
{
  std::vector<ModuloSchedGraphNode*> predS;
  for (unsigned i = 0; i < set.size(); i++) {
    ModuloSchedGraphNode *node = set[i];
    for (ModuloSchedGraphNode::const_iterator I = node->beginInEdges(), E =
         node->endInEdges(); I != E; I++) {
      SchedGraphEdge *edge = *I;
      if (edge->getSrc()->getNodeId() == backEdgeSrc
          && edge->getSink()->getNodeId() == backEdgeSink)
        continue;
      ModuloSchedGraphNode *pred =
          (ModuloSchedGraphNode *) (edge->getSrc());
      //if pred is not in the predSet, push it in vector
      bool alreadyInset = false;
      for (unsigned j = 0; j < predS.size(); ++j)
        if (predS[j]->getNodeId() == pred->getNodeId()) {
          alreadyInset = true;
          break;
        }

      for (unsigned j = 0; j < set.size(); ++j)
        if (set[j]->getNodeId() == pred->getNodeId()) {
          alreadyInset = true;
          break;
        }

      if (!alreadyInset)
        predS.push_back(pred);
    }
  }
  return predS;
}

ModuloSchedGraph::NodeVec ModuloSchedGraph::predSet(NodeVec set)
{
  //node number increases from 2
  return predSet(set, 0, 0);
}

std::vector <ModuloSchedGraphNode*>
ModuloSchedGraph::predSet(ModuloSchedGraphNode *_node,
                          unsigned backEdgeSrc, unsigned backEdgeSink)
{
  std::vector<ModuloSchedGraphNode*> set;
  set.push_back(_node);
  return predSet(set, backEdgeSrc, backEdgeSink);
}

std::vector <ModuloSchedGraphNode*>
ModuloSchedGraph::predSet(ModuloSchedGraphNode * _node)
{
  return predSet(_node, 0, 0);
}

std::vector<ModuloSchedGraphNode*>
ModuloSchedGraph::succSet(std::vector<ModuloSchedGraphNode*> set, 
                          unsigned src, unsigned sink)
{
  std::vector<ModuloSchedGraphNode*> succS;
  for (unsigned i = 0; i < set.size(); i++) {
    ModuloSchedGraphNode *node = set[i];
    for (ModuloSchedGraphNode::const_iterator I =
         node->beginOutEdges(), E = node->endOutEdges(); I != E; I++) {
      SchedGraphEdge *edge = *I;
      if (edge->getSrc()->getNodeId() == src
          && edge->getSink()->getNodeId() == sink)
        continue;
      ModuloSchedGraphNode *succ =
          (ModuloSchedGraphNode *) (edge->getSink());
      //if pred is not in the predSet, push it in vector
      bool alreadyInset = false;
      for (unsigned j = 0; j < succS.size(); j++)
        if (succS[j]->getNodeId() == succ->getNodeId()) {
          alreadyInset = true;
          break;
        }

      for (unsigned j = 0; j < set.size(); j++)
        if (set[j]->getNodeId() == succ->getNodeId()) {
          alreadyInset = true;
          break;
        }
      if (!alreadyInset)
        succS.push_back(succ);
    }
  }
  return succS;
}

ModuloSchedGraph::NodeVec ModuloSchedGraph::succSet(NodeVec set)
{
  return succSet(set, 0, 0);
}


std::vector<ModuloSchedGraphNode*>
ModuloSchedGraph::succSet(ModuloSchedGraphNode *_node,
                          unsigned src, unsigned sink)
{
  std::vector<ModuloSchedGraphNode*>set;
  set.push_back(_node);
  return succSet(set, src, sink);
}

std::vector<ModuloSchedGraphNode*>
ModuloSchedGraph::succSet(ModuloSchedGraphNode * _node)
{
  return succSet(_node, 0, 0);
}

SchedGraphEdge *ModuloSchedGraph::getMaxDelayEdge(unsigned srcId,
                                                  unsigned sinkId)
{
  ModuloSchedGraphNode *node = getNode(srcId);
  SchedGraphEdge *maxDelayEdge = NULL;
  int maxDelay = -1;
  for (ModuloSchedGraphNode::const_iterator I = node->beginOutEdges(), E =
       node->endOutEdges(); I != E; I++) {
    SchedGraphEdge *edge = *I;
    if (edge->getSink()->getNodeId() == sinkId)
      if (edge->getMinDelay() > maxDelay) {
        maxDelayEdge = edge;
        maxDelay = edge->getMinDelay();
      }
  }
  assert(maxDelayEdge != NULL && "no edge between the srcId and sinkId?");
  return maxDelayEdge;
}

void ModuloSchedGraph::dumpCircuits()
{
  DEBUG(std::cerr << "dumping circuits for graph:\n");
  int j = -1;
  while (circuits[++j][0] != 0) {
    int k = -1;
    while (circuits[j][++k] != 0)
      DEBUG(std::cerr << circuits[j][k] << "\t");
    DEBUG(std::cerr << "\n");
  }
}

void ModuloSchedGraph::dumpSet(std::vector < ModuloSchedGraphNode * >set)
{
  for (unsigned i = 0; i < set.size(); i++)
    DEBUG(std::cerr << set[i]->getNodeId() << "\t");
  DEBUG(std::cerr << "\n");
}

std::vector<ModuloSchedGraphNode*>
ModuloSchedGraph::vectorUnion(std::vector<ModuloSchedGraphNode*> set1,
                              std::vector<ModuloSchedGraphNode*> set2)
{
  std::vector<ModuloSchedGraphNode*> unionVec;
  for (unsigned i = 0; i < set1.size(); i++)
    unionVec.push_back(set1[i]);
  for (unsigned j = 0; j < set2.size(); j++) {
    bool inset = false;
    for (unsigned i = 0; i < unionVec.size(); i++)
      if (set2[j] == unionVec[i])
        inset = true;
    if (!inset)
      unionVec.push_back(set2[j]);
  }
  return unionVec;
}

std::vector<ModuloSchedGraphNode*>
ModuloSchedGraph::vectorConj(std::vector<ModuloSchedGraphNode*> set1,
                             std::vector<ModuloSchedGraphNode*> set2)
{
  std::vector<ModuloSchedGraphNode*> conjVec;
  for (unsigned i = 0; i < set1.size(); i++)
    for (unsigned j = 0; j < set2.size(); j++)
      if (set1[i] == set2[j])
        conjVec.push_back(set1[i]);
  return conjVec;
}

ModuloSchedGraph::NodeVec ModuloSchedGraph::vectorSub(NodeVec set1,
                                                      NodeVec set2)
{
  NodeVec newVec;
  for (NodeVec::iterator I = set1.begin(); I != set1.end(); I++) {
    bool inset = false;
    for (NodeVec::iterator II = set2.begin(); II != set2.end(); II++)
      if ((*I)->getNodeId() == (*II)->getNodeId()) {
        inset = true;
        break;
      }
    if (!inset)
      newVec.push_back(*I);
  }
  return newVec;
}

void ModuloSchedGraph::orderNodes() {
  oNodes.clear();

  std::vector < ModuloSchedGraphNode * >set;
  unsigned numNodes = bb->size();

  // first order all the sets
  int j = -1;
  int totalDelay = -1;
  int preDelay = -1;
  while (circuits[++j][0] != 0) {
    int k = -1;
    preDelay = totalDelay;

    while (circuits[j][++k] != 0) {
      ModuloSchedGraphNode *node = getNode(circuits[j][k]);
      unsigned nextNodeId;
      nextNodeId =
          circuits[j][k + 1] != 0 ? circuits[j][k + 1] : circuits[j][0];
      SchedGraphEdge *edge = getMaxDelayEdge(circuits[j][k], nextNodeId);
      totalDelay += edge->getMinDelay();
    }
    if (preDelay != -1 && totalDelay > preDelay) {
      // swap circuits[j][] and cuicuits[j-1][]
      unsigned temp[MAXNODE];
      for (int k = 0; k < MAXNODE; k++) {
        temp[k] = circuits[j - 1][k];
        circuits[j - 1][k] = circuits[j][k];
        circuits[j][k] = temp[k];
      }
      //restart
      j = -1;
    }
  }

  // build the first set
  int backEdgeSrc;
  int backEdgeSink;
  if (ModuloScheduling::printScheduleProcess())
    DEBUG(std::cerr << "building the first set" << "\n");
  int setSeq = -1;
  int k = -1;
  setSeq++;
  while (circuits[setSeq][++k] != 0)
    set.push_back(getNode(circuits[setSeq][k]));
  if (circuits[setSeq][0] != 0) {
    backEdgeSrc = circuits[setSeq][k - 1];
    backEdgeSink = circuits[setSeq][0];
  }
  if (ModuloScheduling::printScheduleProcess()) {
    DEBUG(std::cerr << "the first set is:");
    dumpSet(set);
  }
  // implement the ordering algorithm
  enum OrderSeq { bottom_up, top_down };
  OrderSeq order;
  std::vector<ModuloSchedGraphNode*> R;
  while (!set.empty()) {
    std::vector<ModuloSchedGraphNode*> pset = predSet(oNodes);
    std::vector<ModuloSchedGraphNode*> sset = succSet(oNodes);

    if (!pset.empty() && !vectorConj(pset, set).empty()) {
      R = vectorConj(pset, set);
      order = bottom_up;
    } else if (!sset.empty() && !vectorConj(sset, set).empty()) {
      R = vectorConj(sset, set);
      order = top_down;
    } else {
      int maxASAP = -1;
      int position = -1;
      for (unsigned i = 0; i < set.size(); i++) {
        int temp = set[i]->getASAP();
        if (temp > maxASAP) {
          maxASAP = temp;
          position = i;
        }
      }
      R.push_back(set[position]);
      order = bottom_up;
    }

    while (!R.empty()) {
      if (order == top_down) {
        if (ModuloScheduling::printScheduleProcess())
          DEBUG(std::cerr << "in top_down round\n");
        while (!R.empty()) {
          int maxHeight = -1;
          NodeVec::iterator chosenI;
          for (NodeVec::iterator I = R.begin(); I != R.end(); I++) {
            int temp = (*I)->height;
            if ((temp > maxHeight)
                || (temp == maxHeight && (*I)->mov <= (*chosenI)->mov)) {

              if ((temp > maxHeight)
                  || (temp == maxHeight && (*I)->mov < (*chosenI)->mov)) {
                maxHeight = temp;
                chosenI = I;
                continue;
              }
              //possible case: instruction A and B has the same height and mov,
              //but A has dependence to B e.g B is the branch instruction in the
              //end, or A is the phi instruction at the beginning
              if ((*I)->mov == (*chosenI)->mov)
                for (ModuloSchedGraphNode::const_iterator oe =
                     (*I)->beginOutEdges(), end = (*I)->endOutEdges();
                     oe != end; oe++) {
                  if ((*oe)->getSink() == (*chosenI)) {
                    maxHeight = temp;
                    chosenI = I;
                    continue;
                  }
                }
            }
          }
          ModuloSchedGraphNode *mu = *chosenI;
          oNodes.push_back(mu);
          R.erase(chosenI);
          std::vector<ModuloSchedGraphNode*> succ_mu =
            succSet(mu, backEdgeSrc, backEdgeSink);
          std::vector<ModuloSchedGraphNode*> comm =
            vectorConj(succ_mu, set);
          comm = vectorSub(comm, oNodes);
          R = vectorUnion(comm, R);
        }
        order = bottom_up;
        R = vectorConj(predSet(oNodes), set);
      } else {
        if (ModuloScheduling::printScheduleProcess())
          DEBUG(std::cerr << "in bottom up round\n");
        while (!R.empty()) {
          int maxDepth = -1;
          NodeVec::iterator chosenI;
          for (NodeVec::iterator I = R.begin(); I != R.end(); I++) {
            int temp = (*I)->depth;
            if ((temp > maxDepth)
                || (temp == maxDepth && (*I)->mov < (*chosenI)->mov)) {
              maxDepth = temp;
              chosenI = I;
            }
          }
          ModuloSchedGraphNode *mu = *chosenI;
          oNodes.push_back(mu);
          R.erase(chosenI);
          std::vector<ModuloSchedGraphNode*> pred_mu =
            predSet(mu, backEdgeSrc, backEdgeSink);
          std::vector<ModuloSchedGraphNode*> comm =
            vectorConj(pred_mu, set);
          comm = vectorSub(comm, oNodes);
          R = vectorUnion(comm, R);
        }
        order = top_down;
        R = vectorConj(succSet(oNodes), set);
      }
    }
    if (ModuloScheduling::printScheduleProcess()) {
      DEBUG(std::cerr << "order finished\n");
      DEBUG(std::cerr << "dumping the ordered nodes:\n");
      dumpSet(oNodes);
      dumpCircuits();
    }
    //create a new set
    //FIXME: the nodes between onodes and this circuit should also be include in
    //this set
    if (ModuloScheduling::printScheduleProcess())
      DEBUG(std::cerr << "building the next set\n");
    set.clear();
    int k = -1;
    setSeq++;
    while (circuits[setSeq][++k] != 0)
      set.push_back(getNode(circuits[setSeq][k]));
    if (circuits[setSeq][0] != 0) {
      backEdgeSrc = circuits[setSeq][k - 1];
      backEdgeSink = circuits[setSeq][0];
    }
    if (set.empty()) {
      //no circuits any more
      //collect all other nodes
      if (ModuloScheduling::printScheduleProcess())
        DEBUG(std::cerr << "no circuits any more, collect the rest nodes\n");
      for (unsigned i = 2; i < numNodes + 2; i++) {
        bool inset = false;
        for (unsigned j = 0; j < oNodes.size(); j++)
          if (oNodes[j]->getNodeId() == i) {
            inset = true;
            break;
          }
        if (!inset)
          set.push_back(getNode(i));
      }
    }
    if (ModuloScheduling::printScheduleProcess()) {
      DEBUG(std::cerr << "next set is:\n");
      dumpSet(set);
    }
  }                             //while(!set.empty())

}



void ModuloSchedGraph::buildGraph(const TargetMachine & target)
{

  assert(this->bb && "The basicBlock is NULL?");

  // Use this data structure to note all machine operands that compute
  // ordinary LLVM values.  These must be computed defs (i.e., instructions). 
  // Note that there may be multiple machine instructions that define
  // each Value.
  ValueToDefVecMap valueToDefVecMap;

  // Use this data structure to note all memory instructions.
  // We use this to add memory dependence edges without a second full walk.
  // 
  // vector<const Instruction*> memVec;
  std::vector<ModuloSchedGraphNode*> memNodeVec;

  // Use this data structure to note any uses or definitions of
  // machine registers so we can add edges for those later without
  // extra passes over the nodes.
  // The vector holds an ordered list of references to the machine reg,
  // ordered according to control-flow order.  This only works for a
  // single basic block, hence the assertion.  Each reference is identified
  // by the pair: <node, operand-number>.
  // 
  RegToRefVecMap regToRefVecMap;

  // Make a dummy root node.  We'll add edges to the real roots later.
  graphRoot = new ModuloSchedGraphNode(0, NULL, NULL, -1, target);
  graphLeaf = new ModuloSchedGraphNode(1, NULL, NULL, -1, target);

  //----------------------------------------------------------------
  // First add nodes for all the LLVM instructions in the basic block because
  // this greatly simplifies identifying which edges to add.  Do this one VM
  // instruction at a time since the ModuloSchedGraphNode needs that.
  // Also, remember the load/store instructions to add memory deps later.
  //----------------------------------------------------------------

  //FIXME:if there is call instruction, then we shall quit somewhere
  //      currently we leave call instruction and continue construct graph

  //dump only the blocks which are from loops


  if (ModuloScheduling::printScheduleProcess())
    this->dump(bb);

  if (!isLoop(bb)) {
    DEBUG(std::cerr << " dumping non-loop BB:\n");
    dump(bb);
  }
  if (isLoop(bb)) {
    buildNodesforBB(target, bb, memNodeVec, regToRefVecMap,
                    valueToDefVecMap);

    this->addDefUseEdges(bb);
    this->addCDEdges(bb);
    this->addMemEdges(bb);

    //this->dump();

    int ResII = this->computeResII(bb);
    if (ModuloScheduling::printScheduleProcess())
      DEBUG(std::cerr << "ResII is " << ResII << "\n");
    int RecII = this->computeRecII(bb);
    if (ModuloScheduling::printScheduleProcess())
      DEBUG(std::cerr << "RecII is " << RecII << "\n");

    this->MII = std::max(ResII, RecII);

    this->computeNodeProperty(bb);
    if (ModuloScheduling::printScheduleProcess())
      this->dumpNodeProperty();

    this->orderNodes();

    if (ModuloScheduling::printScheduleProcess())
      this->dump();
    //this->instrScheduling();

    //this->dumpScheduling();
  }
}

ModuloSchedGraphNode *ModuloSchedGraph::getNode(const unsigned nodeId) const
{
  for (const_iterator I = begin(), E = end(); I != E; I++)
    if ((*I).second->getNodeId() == nodeId)
      return (ModuloSchedGraphNode *) (*I).second;
  return NULL;
}

int ModuloSchedGraph::computeRecII(const BasicBlock *bb)
{
  int RecII = 0;

  //os<<"begining computerRecII()"<<"\n";

  //FIXME: only deal with circuits starting at the first node: the phi node
  //nodeId=2;

  //search all elementary circuits in the dependance graph
  //assume maximum number of nodes is MAXNODE

  unsigned path[MAXNODE];
  unsigned stack[MAXNODE][MAXNODE];

  for (int j = 0; j < MAXNODE; j++) {
    path[j] = 0;
    for (int k = 0; k < MAXNODE; k++)
      stack[j][k] = 0;
  }
  //in our graph, the node number starts at 2

  //number of nodes, because each instruction will result in one node
  const unsigned numNodes = bb->size();

  int i = 0;
  path[i] = 2;

  ModuloSchedGraphNode *initNode = getNode(path[0]);
  unsigned initNodeId = initNode->getNodeId();
  ModuloSchedGraphNode *currentNode = initNode;

  while (currentNode != NULL) {
    unsigned currentNodeId = currentNode->getNodeId();
    // DEBUG(std::cerr<<"current node is "<<currentNodeId<<"\n");

    ModuloSchedGraphNode *nextNode = NULL;
    for (ModuloSchedGraphNode::const_iterator I =
         currentNode->beginOutEdges(), E = currentNode->endOutEdges();
         I != E; I++) {
      //DEBUG(std::cerr <<" searching in outgoint edges of node
      //"<<currentNodeId<<"\n";
      unsigned nodeId = ((SchedGraphEdge *) * I)->getSink()->getNodeId();
      bool inpath = false, instack = false;
      int k;

      //DEBUG(std::cerr<<"nodeId is "<<nodeId<<"\n");

      k = -1;
      while (path[++k] != 0)
        if (nodeId == path[k]) {
          inpath = true;
          break;
        }

      k = -1;
      while (stack[i][++k] != 0)
        if (nodeId == stack[i][k]) {
          instack = true;
          break;
        }

      if (nodeId > currentNodeId && !inpath && !instack) {
        nextNode =
            (ModuloSchedGraphNode *) ((SchedGraphEdge *) * I)->getSink();
        break;
      }
    }

    if (nextNode != NULL) {
      //DEBUG(std::cerr<<"find the next Node "<<nextNode->getNodeId()<<"\n");

      int j = 0;
      while (stack[i][j] != 0)
        j++;
      stack[i][j] = nextNode->getNodeId();

      i++;
      path[i] = nextNode->getNodeId();
      currentNode = nextNode;
    } else {
      //DEBUG(std::cerr<<"no expansion any more"<<"\n");
      //confirmCircuit();
      for (ModuloSchedGraphNode::const_iterator I =
           currentNode->beginOutEdges(), E = currentNode->endOutEdges();
           I != E; I++) {
        unsigned nodeId = ((SchedGraphEdge *) * I)->getSink()->getNodeId();
        if (nodeId == initNodeId) {

          int j = -1;
          while (circuits[++j][0] != 0);
          for (int k = 0; k < MAXNODE; k++)
            circuits[j][k] = path[k];

        }
      }
      //remove this node in the path and clear the corresponding entries in the
      //stack
      path[i] = 0;
      int j = 0;
      for (j = 0; j < MAXNODE; j++)
        stack[i][j] = 0;
      i--;
      currentNode = getNode(path[i]);
    }
    if (i == 0) {

      if (ModuloScheduling::printScheduleProcess())
        DEBUG(std::cerr << "circuits found are:\n");
      int j = -1;
      while (circuits[++j][0] != 0) {
        int k = -1;
        while (circuits[j][++k] != 0)
          if (ModuloScheduling::printScheduleProcess())
            DEBUG(std::cerr << circuits[j][k] << "\t");
        if (ModuloScheduling::printScheduleProcess())
          DEBUG(std::cerr << "\n");

        //for this circuit, compute the sum of all edge delay
        int sumDelay = 0;
        k = -1;
        while (circuits[j][++k] != 0) {
          //ModuloSchedGraphNode* node =getNode(circuits[j][k]);
          unsigned nextNodeId;
          nextNodeId =
              circuits[j][k + 1] !=
              0 ? circuits[j][k + 1] : circuits[j][0];

          /*
             for(ModuloSchedGraphNode::const_iterator I=node->beginOutEdges(),
             E=node->endOutEdges();I !=E; I++)
             {
             SchedGraphEdge* edge= *I;
             if(edge->getSink()->getNodeId() == nextNodeId)
             {sumDelay  += edge->getMinDelay();break;}
             }
           */

          sumDelay +=
              getMaxDelayEdge(circuits[j][k], nextNodeId)->getMinDelay();

        }
        //       assume we have distance 1, in this case the sumDelay is RecII
        //       this is correct for SSA form only
        //      
        if (ModuloScheduling::printScheduleProcess())
          DEBUG(std::cerr << "The total Delay in the circuit is " << sumDelay
                << "\n");

        RecII = RecII > sumDelay ? RecII : sumDelay;

      }
      return RecII;
    }

  }

  return -1;
}

void ModuloSchedGraph::addResourceUsage(std::vector<std::pair<int,int> > &ruVec,
                                        int rid)
{
  //DEBUG(std::cerr<<"\nadding a resouce , current resouceUsage vector size is
  //"<<ruVec.size()<<"\n";
  bool alreadyExists = false;
  for (unsigned i = 0; i < ruVec.size(); i++) {
    if (rid == ruVec[i].first) {
      ruVec[i].second++;
      alreadyExists = true;
      break;
    }
  }
  if (!alreadyExists)
    ruVec.push_back(std::make_pair(rid, 1));
  //DEBUG(std::cerr<<"current resouceUsage vector size is "<<ruVec.size()<<"\n";

}
void ModuloSchedGraph::dumpResourceUsage(std::vector<std::pair<int,int> > &ru)
{
  TargetSchedInfo & msi = (TargetSchedInfo &) target.getSchedInfo();

  std::vector<std::pair<int,int> > resourceNumVector = msi.resourceNumVector;
  DEBUG(std::cerr << "resourceID\t" << "resourceNum\n");
  for (unsigned i = 0; i < resourceNumVector.size(); i++)
    DEBUG(std::cerr << resourceNumVector[i].
        first << "\t" << resourceNumVector[i].second << "\n");

  DEBUG(std::cerr << " maxNumIssueTotal(issue slot in one cycle) = " << msi.
        maxNumIssueTotal << "\n");
  DEBUG(std::cerr << "resourceID\t resourceUsage\t ResourceNum\n");
  for (unsigned i = 0; i < ru.size(); i++) {
    DEBUG(std::cerr << ru[i].first << "\t" << ru[i].second);
    const unsigned resNum = msi.getCPUResourceNum(ru[i].first);
    DEBUG(std::cerr << "\t" << resNum << "\n");
  }
}

int ModuloSchedGraph::computeResII(const BasicBlock * bb)
{

  const TargetInstrInfo & mii = target.getInstrInfo();
  const TargetSchedInfo & msi = target.getSchedInfo();

  int ResII;
  std::vector<std::pair<int,int> > resourceUsage;
  //pair<int resourceid, int resourceUsageTimes_in_the_whole_block>

  //FIXME: number of functional units the target machine can provide should be
  //get from Target here I temporary hardcode it

  for (BasicBlock::const_iterator I = bb->begin(), E = bb->end(); I != E;
       I++) {
    if (ModuloScheduling::printScheduleProcess()) {
      DEBUG(std::cerr << "machine instruction for llvm instruction( node " <<
            getGraphNodeForInst(I)->getNodeId() << ")\n");
      DEBUG(std::cerr << "\t" << *I);
    }
    MachineCodeForInstruction & tempMvec =
        MachineCodeForInstruction::get(I);
    if (ModuloScheduling::printScheduleProcess())
      DEBUG(std::cerr << "size =" << tempMvec.size() << "\n");
    for (unsigned i = 0; i < tempMvec.size(); i++) {
      MachineInstr *minstr = tempMvec[i];

      unsigned minDelay = mii.minLatency(minstr->getOpCode());
      InstrRUsage rUsage = msi.getInstrRUsage(minstr->getOpCode());
      InstrClassRUsage classRUsage =
          msi.getClassRUsage(mii.getSchedClass(minstr->getOpCode()));
      unsigned totCycles = classRUsage.totCycles;

      std::vector<std::vector<resourceId_t> > resources=rUsage.resourcesByCycle;
      assert(totCycles == resources.size());
      if (ModuloScheduling::printScheduleProcess())
        DEBUG(std::cerr << "resources Usage for this Instr(totCycles="
              << totCycles << ",mindLatency="
              << mii.minLatency(minstr->getOpCode()) << "): " << *minstr
              << "\n");
      for (unsigned j = 0; j < resources.size(); j++) {
        if (ModuloScheduling::printScheduleProcess())
          DEBUG(std::cerr << "cycle " << j << ": ");
        for (unsigned k = 0; k < resources[j].size(); k++) {
          if (ModuloScheduling::printScheduleProcess())
            DEBUG(std::cerr << "\t" << resources[j][k]);
          addResourceUsage(resourceUsage, resources[j][k]);
        }
        if (ModuloScheduling::printScheduleProcess())
          DEBUG(std::cerr << "\n");
      }
    }
  }
  if (ModuloScheduling::printScheduleProcess())
    this->dumpResourceUsage(resourceUsage);

  //compute ResII
  ResII = 0;
  int issueSlots = msi.maxNumIssueTotal;
  for (unsigned i = 0; i < resourceUsage.size(); i++) {
    int resourceNum = msi.getCPUResourceNum(resourceUsage[i].first);
    int useNum = resourceUsage[i].second;
    double tempII;
    if (resourceNum <= issueSlots)
      tempII = ceil(1.0 * useNum / resourceNum);
    else
      tempII = ceil(1.0 * useNum / issueSlots);
    ResII = std::max((int) tempII, ResII);
  }
  return ResII;
}

ModuloSchedGraphSet::ModuloSchedGraphSet(const Function *function,
                                         const TargetMachine &target)
:  method(function)
{
  buildGraphsForMethod(method, target);
}


ModuloSchedGraphSet::~ModuloSchedGraphSet()
{
  //delete all the graphs
  for (iterator I = begin(), E = end(); I != E; ++I)
    delete *I;
}

void ModuloSchedGraphSet::dump() const
{
  DEBUG(std::cerr << " ====== ModuloSched graphs for function `" << 
        method->getName() << "' =========\n\n");
  for (const_iterator I = begin(); I != end(); ++I)
    (*I)->dump();

  DEBUG(std::cerr << "\n=========End graphs for function `" << method->getName()
        << "' ==========\n\n");
}

void ModuloSchedGraph::dump(const BasicBlock * bb)
{
  DEBUG(std::cerr << "dumping basic block:");
  DEBUG(std::cerr << (bb->hasName()? bb->getName() : "block")
        << " (" << bb << ")" << "\n");
}

void ModuloSchedGraph::dump(const BasicBlock * bb, std::ostream & os)
{
  os << "dumping basic block:";
  os << (bb->hasName()? bb->getName() : "block")
      << " (" << bb << ")" << "\n";
}

void ModuloSchedGraph::dump() const
{
  DEBUG(std::cerr << " ModuloSchedGraph for basic Blocks:");

  DEBUG(std::cerr << (bb->hasName()? bb->getName() : "block")
	<< " (" << bb << ")" <<  "");

  DEBUG(std::cerr << "\n\n    Actual Root nodes : ");
  for (unsigned i = 0, N = graphRoot->outEdges.size(); i < N; i++)
    DEBUG(std::cerr << graphRoot->outEdges[i]->getSink()->getNodeId()
          << ((i == N - 1) ? "" : ", "));

  DEBUG(std::cerr << "\n    Graph Nodes:\n");
  //for (const_iterator I=begin(); I != end(); ++I)
  //DEBUG(std::cerr << "\n" << *I->second;
  unsigned numNodes = bb->size();
  for (unsigned i = 2; i < numNodes + 2; i++) {
    ModuloSchedGraphNode *node = getNode(i);
    DEBUG(std::cerr << "\n" << *node);
  }

  DEBUG(std::cerr << "\n");
}

void ModuloSchedGraph::dumpNodeProperty() const
{

  unsigned numNodes = bb->size();
  for (unsigned i = 2; i < numNodes + 2; i++) {
    ModuloSchedGraphNode *node = getNode(i);
    DEBUG(std::cerr << "NodeId " << node->getNodeId() << "\t");
    DEBUG(std::cerr << "ASAP " << node->getASAP() << "\t");
    DEBUG(std::cerr << "ALAP " << node->getALAP() << "\t");
    DEBUG(std::cerr << "mov " << node->getMov() << "\t");
    DEBUG(std::cerr << "depth " << node->getDepth() << "\t");
    DEBUG(std::cerr << "height " << node->getHeight() << "\t\n");
  }
}

void ModuloSchedGraphSet::buildGraphsForMethod(const Function *F,
                                               const TargetMachine &target)
{
  for (Function::const_iterator BI = F->begin(); BI != F->end(); ++BI){
    const BasicBlock* local_bb;
    local_bb=BI;
    addGraph(new ModuloSchedGraph((BasicBlock*)local_bb, target));
  }
}

std::ostream& operator<<(std::ostream &os,
                         const ModuloSchedGraphNode &node)
{
  os << std::string(8, ' ')
      << "Node " << node.nodeId << " : "
      << "latency = " << node.latency << "\n" << std::string(12, ' ');

  if (node.getInst() == NULL)
    os << "(Dummy node)\n";
  else {
    os << *node.getInst() << "\n" << std::string(12, ' ');
    os << node.inEdges.size() << " Incoming Edges:\n";
    for (unsigned i = 0, N = node.inEdges.size(); i < N; i++)
      os << std::string(16, ' ') << *node.inEdges[i];

    os << std::string(12, ' ') << node.outEdges.size()
        << " Outgoing Edges:\n";
    for (unsigned i = 0, N = node.outEdges.size(); i < N; i++)
      os << std::string(16, ' ') << *node.outEdges[i];
  }

  return os;
}
