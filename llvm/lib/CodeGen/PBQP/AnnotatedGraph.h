//===-- AnnotatedGraph.h - Annotated PBQP Graph ----------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Annotated PBQP Graph class. This class is used internally by the PBQP solver
// to cache information to speed up reduction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_ANNOTATEDGRAPH_H
#define LLVM_CODEGEN_PBQP_ANNOTATEDGRAPH_H

#include "GraphBase.h"

namespace PBQP {


template <typename NodeData, typename EdgeData> class AnnotatedEdge;

template <typename NodeData, typename EdgeData>
class AnnotatedNode : public NodeBase<AnnotatedNode<NodeData, EdgeData>,
                                      AnnotatedEdge<NodeData, EdgeData> > {
private:

  NodeData nodeData; 

public:

  AnnotatedNode(const Vector &costs, const NodeData &nodeData) :
    NodeBase<AnnotatedNode<NodeData, EdgeData>,
             AnnotatedEdge<NodeData, EdgeData> >(costs),
             nodeData(nodeData) {}

  NodeData& getNodeData() { return nodeData; }
  const NodeData& getNodeData() const { return nodeData; }

};

template <typename NodeData, typename EdgeData>
class AnnotatedEdge : public EdgeBase<AnnotatedNode<NodeData, EdgeData>,
                                      AnnotatedEdge<NodeData, EdgeData> > {
private:

  typedef typename GraphBase<AnnotatedNode<NodeData, EdgeData>,
                             AnnotatedEdge<NodeData, EdgeData> >::NodeIterator
    NodeIterator;

  EdgeData edgeData; 

public:


  AnnotatedEdge(const NodeIterator &node1Itr, const NodeIterator &node2Itr,
                const Matrix &costs, const EdgeData &edgeData) :
    EdgeBase<AnnotatedNode<NodeData, EdgeData>,
             AnnotatedEdge<NodeData, EdgeData> >(node1Itr, node2Itr, costs),
    edgeData(edgeData) {}

  EdgeData& getEdgeData() { return edgeData; }
  const EdgeData& getEdgeData() const { return edgeData; }

};

template <typename NodeData, typename EdgeData>
class AnnotatedGraph : public GraphBase<AnnotatedNode<NodeData, EdgeData>,
                                        AnnotatedEdge<NodeData, EdgeData> > {
private:

  typedef GraphBase<AnnotatedNode<NodeData, EdgeData>,
                    AnnotatedEdge<NodeData, EdgeData> > PGraph;

  typedef AnnotatedNode<NodeData, EdgeData> NodeEntry;
  typedef AnnotatedEdge<NodeData, EdgeData> EdgeEntry;


  void copyFrom(const AnnotatedGraph &other) {
    if (!other.areNodeIDsValid()) {
      other.assignNodeIDs();
    }
    std::vector<NodeIterator> newNodeItrs(other.getNumNodes());

    for (ConstNodeIterator nItr = other.nodesBegin(), nEnd = other.nodesEnd();
         nItr != nEnd; ++nItr) {
      newNodeItrs[other.getNodeID(nItr)] = addNode(other.getNodeCosts(nItr));
    }

    for (ConstEdgeIterator eItr = other.edgesBegin(), eEnd = other.edgesEnd();
         eItr != eEnd; ++eItr) {

      unsigned node1ID = other.getNodeID(other.getEdgeNode1(eItr)),
               node2ID = other.getNodeID(other.getEdgeNode2(eItr));

      addEdge(newNodeItrs[node1ID], newNodeItrs[node2ID],
              other.getEdgeCosts(eItr), other.getEdgeData(eItr));
    }

  }

public:

  typedef typename PGraph::NodeIterator NodeIterator;
  typedef typename PGraph::ConstNodeIterator ConstNodeIterator;
  typedef typename PGraph::EdgeIterator EdgeIterator;
  typedef typename PGraph::ConstEdgeIterator ConstEdgeIterator;

  AnnotatedGraph() {}

  AnnotatedGraph(const AnnotatedGraph &other) {
    copyFrom(other);
  }

  AnnotatedGraph& operator=(const AnnotatedGraph &other) {
    PGraph::clear();
    copyFrom(other);
    return *this;
  }

  NodeIterator addNode(const Vector &costs, const NodeData &data) {
    return PGraph::addConstructedNode(NodeEntry(costs, data));
  }

  EdgeIterator addEdge(const NodeIterator &node1Itr,
                       const NodeIterator &node2Itr,
                       const Matrix &costs, const EdgeData &data) {
    return PGraph::addConstructedEdge(EdgeEntry(node1Itr, node2Itr,
                                                costs, data));
  }

  NodeData& getNodeData(const NodeIterator &nodeItr) {
    return getNodeEntry(nodeItr).getNodeData();
  }

  const NodeData& getNodeData(const NodeIterator &nodeItr) const {
    return getNodeEntry(nodeItr).getNodeData();
  }

  EdgeData& getEdgeData(const EdgeIterator &edgeItr) {
    return getEdgeEntry(edgeItr).getEdgeData();
  }

  const EdgeEntry& getEdgeData(const EdgeIterator &edgeItr) const {
    return getEdgeEntry(edgeItr).getEdgeData();
  }

  SimpleGraph toSimpleGraph() const {
    SimpleGraph g;

    if (!PGraph::areNodeIDsValid()) {
      PGraph::assignNodeIDs();
    }
    std::vector<SimpleGraph::NodeIterator> newNodeItrs(PGraph::getNumNodes());

    for (ConstNodeIterator nItr = PGraph::nodesBegin(), 
         nEnd = PGraph::nodesEnd();
         nItr != nEnd; ++nItr) {

      newNodeItrs[getNodeID(nItr)] = g.addNode(getNodeCosts(nItr));
    }

    for (ConstEdgeIterator
         eItr = PGraph::edgesBegin(), eEnd = PGraph::edgesEnd();
         eItr != eEnd; ++eItr) {

      unsigned node1ID = getNodeID(getEdgeNode1(eItr)),
               node2ID = getNodeID(getEdgeNode2(eItr));

        g.addEdge(newNodeItrs[node1ID], newNodeItrs[node2ID],
                  getEdgeCosts(eItr));
    }

    return g;
  }

};


}

#endif // LLVM_CODEGEN_PBQP_ANNOTATEDGRAPH_H
