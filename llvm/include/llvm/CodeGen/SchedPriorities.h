/* -*-C++-*-
 ****************************************************************************
 * File:
 *	SchedPriorities.h
 * 
 * Purpose:
 *	Encapsulate heuristics for instruction scheduling.
 * 
 * Strategy:
 *    Priority ordering rules:
 *    (1) Max delay, which is the order of the heap S.candsAsHeap.
 *    (2) Instruction that frees up a register.
 *    (3) Instruction that has the maximum number of dependent instructions.
 *    Note that rules 2 and 3 are only used if issue conflicts prevent
 *    choosing a higher priority instruction by rule 1.
 * 
 * History:
 *	7/30/01	 -  Vikram Adve  -  Created
 ***************************************************************************/

#ifndef LLVM_CODEGEN_SCHEDPRIORITIES_H
#define LLVM_CODEGEN_SCHEDPRIORITIES_H

//************************** System Include Files **************************/

#include <hash_map>
#include <list>
#include <vector>
#include <algorithm>

//*************************** User Include Files ***************************/

#include "llvm/CFG.h"			// just for graph iterators
#include "llvm/Support/NonCopyable.h"
#include "llvm/Support/HashExtras.h"
#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"
#include "llvm/CodeGen/SchedGraph.h"
#include "llvm/CodeGen/InstrScheduling.h"

//************************* Opaque Declarations ****************************/

class Method;
class MachineInstr;
class SchedulingManager;

/******************** Exported Data Types and Constants ********************/


//*********************** Public Class Declarations ************************/

struct NodeDelayPair {
  const SchedGraphNode* node;
  cycles_t delay;
  NodeDelayPair(const SchedGraphNode* n, cycles_t d) :  node(n), delay(d) {}
  inline bool operator< (const NodeDelayPair& np) { return delay < np.delay; }
};

inline bool
NDPLessThan(const NodeDelayPair* np1, const NodeDelayPair* np2)
{
  return (np1->delay < np2->delay);
}

class NodeHeap: public list<NodeDelayPair*>, public NonCopyable {
public:
  typedef list<NodeDelayPair*>::iterator iterator;
  typedef list<NodeDelayPair*>::const_iterator const_iterator;
  
public:
  /*ctor*/	  NodeHeap	() : list<NodeDelayPair*>(), _size(0) {}
  /*dtor*/	  ~NodeHeap	() {}
  
  inline unsigned int	size	() const { return _size; }
  
  const SchedGraphNode* getNode	(const_iterator i) const { return (*i)->node; }
  cycles_t		getDelay(const_iterator i) const { return (*i)->delay;}
  
  inline void		makeHeap() { 
    // make_heap(begin(), end(), NDPLessThan);
  }
  
  inline iterator	findNode(const SchedGraphNode* node) {
    for (iterator I=begin(); I != end(); ++I)
      if (getNode(I) == node)
	return I;
    return end();
  }
  
  inline void	  removeNode	(const SchedGraphNode* node) {
    iterator ndpPtr = findNode(node);
    if (ndpPtr != end())
      {
	delete *ndpPtr;
	erase(ndpPtr);
	--_size;
      }
  };
  
  void		  insert(const SchedGraphNode* node, cycles_t delay) {
    NodeDelayPair* ndp = new NodeDelayPair(node, delay);
    if (_size == 0 || front()->delay < delay)
      push_front(ndp);
    else
      {
	iterator I=begin();
	for ( ; I != end() && getDelay(I) >= delay; ++I)
	  ;
	list<NodeDelayPair*>::insert(I, ndp);
      }
    _size++;
  }
private:
  unsigned int _size;
};


class SchedPriorities: public NonCopyable {
public:
  /*ctor*/	SchedPriorities		(const Method* method,
					 const SchedGraph* _graph);
  
  // This must be called before scheduling begins.
  void		initialize		();
  
  cycles_t	getTime			() const { return curTime; }
  cycles_t	getEarliestReadyTime	() const { return earliestReadyTime; }
  unsigned	getNumReady		() const { return candsAsHeap.size(); }
  bool		nodeIsReady		(const SchedGraphNode* node) const {
    return (candsAsSet.find(node) != candsAsSet.end());
  }
  
  void		issuedReadyNodeAt	(cycles_t curTime,
					 const SchedGraphNode* node);
  
  void		insertReady		(const SchedGraphNode* node);
  
  void		updateTime		(cycles_t /*unused*/);
  
  const SchedGraphNode* getNextHighest	(const SchedulingManager& S,
					 cycles_t curTime);
					// choose next highest priority instr
  
private:
  typedef NodeHeap::iterator candIndex;
  
private:
  cycles_t curTime;
  const SchedGraph* graph;
  MethodLiveVarInfo methodLiveVarInfo;
  hash_map<const MachineInstr*, bool> lastUseMap;
  vector<cycles_t> nodeDelayVec;
  vector<cycles_t> earliestForNode;
  cycles_t earliestReadyTime;
  NodeHeap candsAsHeap;				// candidate nodes, ready to go
  hash_set<const SchedGraphNode*> candsAsSet;	// same entries as candsAsHeap,
						//   but as set for fast lookup
  vector<candIndex> mcands;			// holds pointers into cands
  candIndex nextToTry;				// next cand after the last
						//   one tried in this cycle
  
  int		chooseByRule1		(vector<candIndex>& mcands);
  int		chooseByRule2		(vector<candIndex>& mcands);
  int		chooseByRule3		(vector<candIndex>& mcands);
  
  void		findSetWithMaxDelay	(vector<candIndex>& mcands,
					 const SchedulingManager& S);
  
  void		computeDelays		(const SchedGraph* graph);
  
  void		initializeReadyHeap	(const SchedGraph* graph);
  
  bool		instructionHasLastUse	(MethodLiveVarInfo& methodLiveVarInfo,
					 const SchedGraphNode* graphNode);
  
  // NOTE: The next two return references to the actual vector entries.
  //       Use with care.
  cycles_t&	getNodeDelayRef		(const SchedGraphNode* node) {
    assert(node->getNodeId() < nodeDelayVec.size());
    return nodeDelayVec[node->getNodeId()];
  }
  cycles_t&	getEarliestForNodeRef	(const SchedGraphNode* node) {
    assert(node->getNodeId() < earliestForNode.size());
    return earliestForNode[node->getNodeId()];
  }
};


inline void
SchedPriorities::insertReady(const SchedGraphNode* node)
{
  candsAsHeap.insert(node, nodeDelayVec[node->getNodeId()]);
  candsAsSet.insert(node);
  mcands.clear(); // ensure reset choices is called before any more choices
  earliestReadyTime = min(earliestReadyTime,
			  earliestForNode[node->getNodeId()]);
  
  if (SchedDebugLevel >= Sched_PrintSchedTrace)
    {
      printIndent(2);
      cout << "Cycle " << this->getTime() << ": "
	   << " Node " << node->getNodeId() << " is ready; "
	   << " Delay = " << this->getNodeDelayRef(node) << "; Instruction: "
	   << endl;
      printIndent(4);
      cout << * node->getMachineInstr() << endl;
    }
}

inline void
SchedPriorities::updateTime(cycles_t c)
{
  curTime = c;
  nextToTry = candsAsHeap.begin();
  mcands.clear();
}

inline ostream& operator<< (ostream& os, const NodeDelayPair* nd) {
  os << "Delay for node " << nd->node->getNodeId()
     << " = " << nd->delay << endl;
  return os;
}

/***************************************************************************/

#endif
