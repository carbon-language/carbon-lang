//===- DependenceGraph.cpp - Dependence graph for a function ----*- C++ -*-===//
//
// This file implements an explicit representation for the dependence graph
// of a function, with one node per instruction and one edge per dependence.
// Dependences include both data and control dependences.
// 
// Each dep. graph node (class DepGraphNode) keeps lists of incoming and
// outgoing dependence edges.
// 
// Each dep. graph edge (class Dependence) keeps a pointer to one end-point
// of the dependence.  This saves space and is important because dep. graphs
// can grow quickly.  It works just fine because the standard idiom is to
// start with a known node and enumerate the dependences to or from that node.
//===----------------------------------------------------------------------===//


#include "llvm/Analysis/DependenceGraph.h"
#include "llvm/Function.h"


//----------------------------------------------------------------------------
// class Dependence:
// 
// A representation of a simple (non-loop-related) dependence
//----------------------------------------------------------------------------

void Dependence::print(std::ostream &O) const
{
  assert(depType != NoDependence && "This dependence should never be created!");
  switch (depType) {
  case TrueDependence:    O << "TRUE dependence"; break;
  case AntiDependence:    O << "ANTI dependence"; break;
  case OutputDependence:  O << "OUTPUT dependence"; break;
  case ControlDependence: O << "CONTROL dependence"; break;
  default: assert(0 && "Invalid dependence type"); break;
  }
}


//----------------------------------------------------------------------------
// class DepGraphNode
//----------------------------------------------------------------------------

void DepGraphNode::print(std::ostream &O) const
{
  const_iterator DI = outDepBegin(), DE = outDepEnd();

  O << "\nDeps. from instr:" << getInstr();

  for ( ; DI != DE; ++DI)
    {
      O << "\t";
      DI->print(O);
      O << " to instruction:";
      O << DI->getSink()->getInstr();
    }
}

//----------------------------------------------------------------------------
// class DependenceGraph
//----------------------------------------------------------------------------

DependenceGraph::~DependenceGraph()
{
  // Free all DepGraphNode objects created for this graph
  for (map_iterator I = depNodeMap.begin(), E = depNodeMap.end(); I != E; ++I)
    delete I->second;
}

void DependenceGraph::print(const Function& func, std::ostream &O) const
{
  O << "DEPENDENCE GRAPH FOR FUNCTION " << func.getName() << ":\n";
  for (Function::const_iterator BB=func.begin(), FE=func.end(); BB != FE; ++BB)
    for (BasicBlock::const_iterator II=BB->begin(), IE=BB->end(); II !=IE; ++II)
      if (const DepGraphNode* dgNode = this->getNode(*II))
        dgNode->print(O);
}
