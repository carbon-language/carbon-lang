//===- SchedGraphCommon.cpp - Scheduling Graphs Base Class- ---------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Scheduling graph base class that contains common information for SchedGraph
// and ModuloSchedGraph scheduling graphs.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SchedGraphCommon.h"
#include "Support/STLExtras.h"

namespace llvm {

class SchedGraphCommon;

//
// class SchedGraphEdge
// 
SchedGraphEdge::SchedGraphEdge(SchedGraphNodeCommon* _src,
			       SchedGraphNodeCommon* _sink,
			       SchedGraphEdgeDepType _depType,
			       unsigned int     _depOrderType,
			       int _minDelay)
  : src(_src), sink(_sink), depType(_depType), depOrderType(_depOrderType),
    minDelay((_minDelay >= 0)? _minDelay : _src->getLatency()), val(NULL) {
  
  iteDiff=0;
  assert(src != sink && "Self-loop in scheduling graph!");
  src->addOutEdge(this);
  sink->addInEdge(this);
}

SchedGraphEdge::SchedGraphEdge(SchedGraphNodeCommon*  _src,
			       SchedGraphNodeCommon*  _sink,
			       const Value*     _val,
			       unsigned int     _depOrderType,
			       int              _minDelay)
  : src(_src), sink(_sink), depType(ValueDep), depOrderType(_depOrderType),
    minDelay((_minDelay >= 0)? _minDelay : _src->getLatency()), val(_val) {
  iteDiff=0;
  assert(src != sink && "Self-loop in scheduling graph!");
  src->addOutEdge(this);
  sink->addInEdge(this);
}

SchedGraphEdge::SchedGraphEdge(SchedGraphNodeCommon*  _src,
			       SchedGraphNodeCommon*  _sink,
			       unsigned int     _regNum,
			       unsigned int     _depOrderType,
			       int             _minDelay)
  : src(_src), sink(_sink), depType(MachineRegister),
    depOrderType(_depOrderType),
    minDelay((_minDelay >= 0)? _minDelay : _src->getLatency()),
    machineRegNum(_regNum) {
  iteDiff=0;
  assert(src != sink && "Self-loop in scheduling graph!");
  src->addOutEdge(this);
  sink->addInEdge(this);
}

SchedGraphEdge::SchedGraphEdge(SchedGraphNodeCommon* _src,
			       SchedGraphNodeCommon* _sink,
			       ResourceId      _resourceId,
			       int             _minDelay)
  : src(_src), sink(_sink), depType(MachineResource), depOrderType(NonDataDep),
    minDelay((_minDelay >= 0)? _minDelay : _src->getLatency()),
    resourceId(_resourceId) {
  iteDiff=0;
  assert(src != sink && "Self-loop in scheduling graph!");
  src->addOutEdge(this);
  sink->addInEdge(this);
}




void SchedGraphEdge::dump(int indent) const {
  std::cerr << std::string(indent*2, ' ') << *this; 
}

/*dtor*/
SchedGraphNodeCommon::~SchedGraphNodeCommon()
{
  // for each node, delete its out-edges
  std::for_each(beginOutEdges(), endOutEdges(),
                deleter<SchedGraphEdge>);
}

void SchedGraphNodeCommon::removeInEdge(const SchedGraphEdge* edge) {
  assert(edge->getSink() == this);
  
  for (iterator I = beginInEdges(); I != endInEdges(); ++I)
    if ((*I) == edge) {
      inEdges.erase(I);
      break;
    }
}

void SchedGraphNodeCommon::removeOutEdge(const SchedGraphEdge* edge) {
  assert(edge->getSrc() == this);
  
  for (iterator I = beginOutEdges(); I != endOutEdges(); ++I)
    if ((*I) == edge) {
      outEdges.erase(I);
      break;
    }
}

void SchedGraphNodeCommon::dump(int indent) const {
  std::cerr << std::string(indent*2, ' ') << *this; 
}

//class SchedGraphCommon

SchedGraphCommon::~SchedGraphCommon() {
  delete graphRoot;
  delete graphLeaf;
}


void SchedGraphCommon::eraseIncomingEdges(SchedGraphNodeCommon* node, 
					  bool addDummyEdges) {
  // Delete and disconnect all in-edges for the node
  for (SchedGraphNodeCommon::iterator I = node->beginInEdges();
       I != node->endInEdges(); ++I) {
    SchedGraphNodeCommon* srcNode = (*I)->getSrc();
    srcNode->removeOutEdge(*I);
    delete *I;
    
    if (addDummyEdges && srcNode != getRoot() &&
	srcNode->beginOutEdges() == srcNode->endOutEdges()) { 
      
      // srcNode has no more out edges, so add an edge to dummy EXIT node
      assert(node != getLeaf() && "Adding edge that was just removed?");
      (void) new SchedGraphEdge(srcNode, getLeaf(),
				SchedGraphEdge::CtrlDep, 
				SchedGraphEdge::NonDataDep, 0);
    }
  }
  
  node->inEdges.clear();
}

void SchedGraphCommon::eraseOutgoingEdges(SchedGraphNodeCommon* node, 
					  bool addDummyEdges) {
  // Delete and disconnect all out-edges for the node
  for (SchedGraphNodeCommon::iterator I = node->beginOutEdges();
       I != node->endOutEdges(); ++I) {
    SchedGraphNodeCommon* sinkNode = (*I)->getSink();
    sinkNode->removeInEdge(*I);
    delete *I;
    
    if (addDummyEdges &&
	sinkNode != getLeaf() &&
	sinkNode->beginInEdges() == sinkNode->endInEdges()) {
      
      //sinkNode has no more in edges, so add an edge from dummy ENTRY node
      assert(node != getRoot() && "Adding edge that was just removed?");
      (void) new SchedGraphEdge(getRoot(), sinkNode,
				SchedGraphEdge::CtrlDep, 
				SchedGraphEdge::NonDataDep, 0);
    }
  }
  
  node->outEdges.clear();
}

void SchedGraphCommon::eraseIncidentEdges(SchedGraphNodeCommon* node, 
					  bool addDummyEdges) {
  this->eraseIncomingEdges(node, addDummyEdges);	
  this->eraseOutgoingEdges(node, addDummyEdges);	
}

} // End llvm namespace
