//===-- SimpleGraph.h - Simple PBQP Graph ----------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simple PBQP graph class representing a PBQP problem. Graphs of this type
// can be passed to a PBQPSolver instance to solve the PBQP problem.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_SIMPLEGRAPH_H
#define LLVM_CODEGEN_PBQP_SIMPLEGRAPH_H

#include "GraphBase.h"

namespace PBQP {

class SimpleEdge;

class SimpleNode : public NodeBase<SimpleNode, SimpleEdge> {
public:
  SimpleNode(const Vector &costs) :
    NodeBase<SimpleNode, SimpleEdge>(costs) {}
};

class SimpleEdge : public EdgeBase<SimpleNode, SimpleEdge> {
public:
  SimpleEdge(const NodeIterator &node1Itr, const NodeIterator &node2Itr,
             const Matrix &costs) :
    EdgeBase<SimpleNode, SimpleEdge>(node1Itr, node2Itr, costs) {}
};

class SimpleGraph : public GraphBase<SimpleNode, SimpleEdge> {
private:

  typedef GraphBase<SimpleNode, SimpleEdge> PGraph;

  void copyFrom(const SimpleGraph &other) {
    assert(other.areNodeIDsValid() &&
           "Cannot copy from another graph unless IDs have been assigned.");
   
    std::vector<NodeIterator> newNodeItrs(other.getNumNodes());

    for (ConstNodeIterator nItr = other.nodesBegin(), nEnd = other.nodesEnd();
         nItr != nEnd; ++nItr) {
      newNodeItrs[other.getNodeID(nItr)] = addNode(other.getNodeCosts(nItr));
    }

    for (ConstEdgeIterator eItr = other.edgesBegin(), eEnd = other.edgesEnd();
         eItr != eEnd; ++eItr) {

      unsigned node1ID = other.getNodeID(other.getEdgeNode1Itr(eItr)),
               node2ID = other.getNodeID(other.getEdgeNode2Itr(eItr));

      addEdge(newNodeItrs[node1ID], newNodeItrs[node2ID],
              other.getEdgeCosts(eItr));
    }
  }

  void copyFrom(SimpleGraph &other) {
    if (!other.areNodeIDsValid()) {
      other.assignNodeIDs();
    }
    copyFrom(const_cast<const SimpleGraph&>(other));
  }

public:

  SimpleGraph() {}


  SimpleGraph(const SimpleGraph &other) : PGraph() {
    copyFrom(other);
  }

  SimpleGraph& operator=(const SimpleGraph &other) {
    clear();
    copyFrom(other);
    return *this;
  }

  NodeIterator addNode(const Vector &costs) {
    return PGraph::addConstructedNode(SimpleNode(costs));
  }

  EdgeIterator addEdge(const NodeIterator &node1Itr,
                       const NodeIterator &node2Itr,
                       const Matrix &costs) {
    return PGraph::addConstructedEdge(SimpleEdge(node1Itr, node2Itr, costs));
  }

};

}

#endif // LLVM_CODEGEN_PBQP_SIMPLEGRAPH_H
