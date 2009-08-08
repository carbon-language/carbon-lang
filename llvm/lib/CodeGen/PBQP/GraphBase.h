//===-- GraphBase.h - Abstract Base PBQP Graph -----------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Base class for PBQP Graphs.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CODEGEN_PBQP_GRAPHBASE_H
#define LLVM_CODEGEN_PBQP_GRAPHBASE_H

#include "PBQPMath.h"

#include <list>
#include <vector>

namespace PBQP {

// UGLY, but I'm not sure there's a good way around this: We need to be able to
// look up a Node's "adjacent edge list" structure type before the Node type is
// fully constructed.  We can enable this by pushing the choice of data type
// out into this traits class.
template <typename Graph>
class NodeBaseTraits {
  public:
    typedef std::list<typename Graph::EdgeIterator> AdjEdgeList;
    typedef typename AdjEdgeList::iterator AdjEdgeIterator;
    typedef typename AdjEdgeList::const_iterator ConstAdjEdgeIterator;
};

/// \brief Base for concrete graph classes. Provides a basic set of graph
///        operations which are useful for PBQP solvers.
template <typename NodeEntry, typename EdgeEntry>
class GraphBase {
private:

  typedef GraphBase<NodeEntry, EdgeEntry> ThisGraphT;

  typedef std::list<NodeEntry> NodeList;
  typedef std::list<EdgeEntry> EdgeList;

  NodeList nodeList;
  unsigned nodeListSize;

  EdgeList edgeList;
  unsigned edgeListSize;

  GraphBase(const ThisGraphT &other) { abort(); }
  void operator=(const ThisGraphT &other) { abort(); } 
  
public:

  /// \brief Iterates over the nodes of a graph.
  typedef typename NodeList::iterator NodeIterator;
  /// \brief Iterates over the nodes of a const graph.
  typedef typename NodeList::const_iterator ConstNodeIterator;
  /// \brief Iterates over the edges of a graph.
  typedef typename EdgeList::iterator EdgeIterator;
  /// \brief Iterates over the edges of a const graph.
  typedef typename EdgeList::const_iterator ConstEdgeIterator;

  /// \brief Iterates over the edges attached to a node.
  typedef typename NodeBaseTraits<ThisGraphT>::AdjEdgeIterator
    AdjEdgeIterator;

  /// \brief Iterates over the edges attached to a node in a const graph.
  typedef typename NodeBaseTraits<ThisGraphT>::ConstAdjEdgeIterator
    ConstAdjEdgeIterator;

private:

  typedef std::vector<NodeIterator> IDToNodeMap;

  IDToNodeMap idToNodeMap;
  bool nodeIDsValid;

  void invalidateNodeIDs() {
    if (nodeIDsValid) {
      idToNodeMap.clear();
      nodeIDsValid = false;
    }
  }

  template <typename ItrT>
  bool iteratorInRange(ItrT itr, const ItrT &begin, const ItrT &end) {
    for (ItrT t = begin; t != end; ++t) {
      if (itr == t)
        return true;
    }

    return false;
  }

protected:

  GraphBase() : nodeListSize(0), edgeListSize(0), nodeIDsValid(false) {}
  
  NodeEntry& getNodeEntry(const NodeIterator &nodeItr) { return *nodeItr; }
  const NodeEntry& getNodeEntry(const ConstNodeIterator &nodeItr) const {
    return *nodeItr;
  }

  EdgeEntry& getEdgeEntry(const EdgeIterator &edgeItr) { return *edgeItr; }
  const EdgeEntry& getEdgeEntry(const ConstEdgeIterator &edgeItr) const {
    return *edgeItr;
  }

  NodeIterator addConstructedNode(const NodeEntry &nodeEntry) {
    ++nodeListSize;

    invalidateNodeIDs();

    NodeIterator newNodeItr = nodeList.insert(nodeList.end(), nodeEntry);

    return newNodeItr;
  }

  EdgeIterator addConstructedEdge(const EdgeEntry &edgeEntry) {

    assert((findEdge(edgeEntry.getNode1Itr(), edgeEntry.getNode2Itr())
          == edgeList.end()) && "Attempt to add duplicate edge.");

    ++edgeListSize;

    // Add the edge to the graph.
    EdgeIterator edgeItr = edgeList.insert(edgeList.end(), edgeEntry);

    // Get a reference to the version in the graph.
    EdgeEntry &newEdgeEntry = getEdgeEntry(edgeItr);

    // Node entries:
    NodeEntry &node1Entry = getNodeEntry(newEdgeEntry.getNode1Itr()),
              &node2Entry = getNodeEntry(newEdgeEntry.getNode2Itr());

    // Sanity check on matrix dimensions.
    assert((node1Entry.getCosts().getLength() == 
            newEdgeEntry.getCosts().getRows()) && 
           (node2Entry.getCosts().getLength() == 
            newEdgeEntry.getCosts().getCols()) &&
        "Matrix dimensions do not match cost vector dimensions.");

    // Create links between nodes and edges.
    newEdgeEntry.setNode1ThisEdgeItr(
        node1Entry.addAdjEdge(edgeItr));
    newEdgeEntry.setNode2ThisEdgeItr(
        node2Entry.addAdjEdge(edgeItr));

    return edgeItr;
  }

public:

  /// \brief Returns the number of nodes in this graph.
  unsigned getNumNodes() const { return nodeListSize; }

  /// \brief Returns the number of edges in this graph.
  unsigned getNumEdges() const { return edgeListSize; } 

  /// \brief Return the cost vector for the given node.
  Vector& getNodeCosts(const NodeIterator &nodeItr) {
    return getNodeEntry(nodeItr).getCosts();
  }

  /// \brief Return the cost vector for the give node. 
  const Vector& getNodeCosts(const ConstNodeIterator &nodeItr) const {
    return getNodeEntry(nodeItr).getCosts();
  }

  /// \brief Return the degree of the given node.
  unsigned getNodeDegree(const NodeIterator &nodeItr) const {
    return getNodeEntry(nodeItr).getDegree();
  }

  /// \brief Assigns sequential IDs to the nodes, starting at 0, which
  /// remain valid until the next addition or removal of a node.
  void assignNodeIDs() {
    unsigned curID = 0;
    idToNodeMap.resize(getNumNodes());
    for (NodeIterator nodeItr = nodesBegin(), nodeEnd = nodesEnd();
         nodeItr != nodeEnd; ++nodeItr, ++curID) {
      getNodeEntry(nodeItr).setID(curID);
      idToNodeMap[curID] = nodeItr;
    }
    nodeIDsValid = true;
  }

  /// \brief Assigns sequential IDs to the nodes using the ordering of the
  /// given vector.
  void assignNodeIDs(const std::vector<NodeIterator> &nodeOrdering) {
    assert((getNumNodes() == nodeOrdering.size()) && 
           "Wrong number of nodes in node ordering.");
    idToNodeMap = nodeOrdering;
    for (unsigned nodeID = 0; nodeID < idToNodeMap.size(); ++nodeID) {
      getNodeEntry(idToNodeMap[nodeID]).setID(nodeID);
    }
    nodeIDsValid = true;
  }

  /// \brief Returns true if valid node IDs are assigned, false otherwise.
  bool areNodeIDsValid() const { return nodeIDsValid; }

  /// \brief Return the numeric ID of the given node.
  ///
  /// Calls to this method will result in an assertion failure if there have
  /// been any node additions or removals since the last call to
  /// assignNodeIDs().
  unsigned getNodeID(const ConstNodeIterator &nodeItr) const {
    assert(nodeIDsValid && "Attempt to retrieve invalid ID.");
    return getNodeEntry(nodeItr).getID();
  }

  /// \brief Returns the iterator associated with the given node ID.
  NodeIterator getNodeItr(unsigned nodeID) {
    assert(nodeIDsValid && "Attempt to retrieve iterator with invalid ID.");
    return idToNodeMap[nodeID];
  }

  /// \brief Returns the iterator associated with the given node ID.
  ConstNodeIterator getNodeItr(unsigned nodeID) const {
    assert(nodeIDsValid && "Attempt to retrieve iterator with invalid ID.");
    return idToNodeMap[nodeID];
  }

  /// \brief Removes the given node (and all attached edges) from the graph.
  void removeNode(const NodeIterator &nodeItr) {
    assert(iteratorInRange(nodeItr, nodeList.begin(), nodeList.end()) &&
           "Iterator does not belong to this graph!");

    invalidateNodeIDs();
    
    NodeEntry &nodeEntry = getNodeEntry(nodeItr);

    // We need to copy this out because it will be destroyed as the edges are
    // removed.
    typedef std::vector<EdgeIterator> AdjEdgeList;
    typedef typename AdjEdgeList::iterator AdjEdgeListItr;

    AdjEdgeList adjEdges;
    adjEdges.reserve(nodeEntry.getDegree());
    std::copy(nodeEntry.adjEdgesBegin(), nodeEntry.adjEdgesEnd(),
              std::back_inserter(adjEdges));

    // Iterate over the copied out edges and remove them from the graph.
    for (AdjEdgeListItr itr = adjEdges.begin(), end = adjEdges.end();
         itr != end; ++itr) {
      removeEdge(*itr);
    }

    // Erase the node from the nodelist.
    nodeList.erase(nodeItr);
    --nodeListSize;
  }

  NodeIterator nodesBegin() { return nodeList.begin(); }
  ConstNodeIterator nodesBegin() const { return nodeList.begin(); }
  NodeIterator nodesEnd() { return nodeList.end(); }
  ConstNodeIterator nodesEnd() const { return nodeList.end(); }

  AdjEdgeIterator adjEdgesBegin(const NodeIterator &nodeItr) {
    return getNodeEntry(nodeItr).adjEdgesBegin();
  }

  ConstAdjEdgeIterator adjEdgesBegin(const ConstNodeIterator &nodeItr) const {
    return getNodeEntry(nodeItr).adjEdgesBegin();
  }

  AdjEdgeIterator adjEdgesEnd(const NodeIterator &nodeItr) {
    return getNodeEntry(nodeItr).adjEdgesEnd();
  }
  
  ConstAdjEdgeIterator adjEdgesEnd(const ConstNodeIterator &nodeItr) const {
    getNodeEntry(nodeItr).adjEdgesEnd();
  }

  EdgeIterator findEdge(const NodeIterator &node1Itr,
                        const NodeIterator &node2Itr) {

    for (AdjEdgeIterator adjEdgeItr = adjEdgesBegin(node1Itr),
         adjEdgeEnd = adjEdgesEnd(node1Itr);
         adjEdgeItr != adjEdgeEnd; ++adjEdgeItr) {
      if ((getEdgeNode1Itr(*adjEdgeItr) == node2Itr) ||
          (getEdgeNode2Itr(*adjEdgeItr) == node2Itr)) {
        return *adjEdgeItr;
      }
    }

    return edgeList.end();
  }

  ConstEdgeIterator findEdge(const ConstNodeIterator &node1Itr,
                             const ConstNodeIterator &node2Itr) const {

    for (ConstAdjEdgeIterator adjEdgeItr = adjEdgesBegin(node1Itr),
         adjEdgeEnd = adjEdgesEnd(node1Itr);
         adjEdgeItr != adjEdgesEnd; ++adjEdgeItr) {
      if ((getEdgeNode1Itr(*adjEdgeItr) == node2Itr) ||
          (getEdgeNode2Itr(*adjEdgeItr) == node2Itr)) {
        return *adjEdgeItr;
      }
    }

    return edgeList.end();
  }

  Matrix& getEdgeCosts(const EdgeIterator &edgeItr) {
    return getEdgeEntry(edgeItr).getCosts();
  }

  const Matrix& getEdgeCosts(const ConstEdgeIterator &edgeItr) const {
    return getEdgeEntry(edgeItr).getCosts();
  }

  NodeIterator getEdgeNode1Itr(const EdgeIterator &edgeItr) {
    return getEdgeEntry(edgeItr).getNode1Itr();
  }

  ConstNodeIterator getEdgeNode1Itr(const ConstEdgeIterator &edgeItr) const {
    return getEdgeEntry(edgeItr).getNode1Itr();
  }

  NodeIterator getEdgeNode2Itr(const EdgeIterator &edgeItr) {
    return getEdgeEntry(edgeItr).getNode2Itr();
  }

  ConstNodeIterator getEdgeNode2Itr(const ConstEdgeIterator &edgeItr) const {
    return getEdgeEntry(edgeItr).getNode2Itr();
  }

  NodeIterator getEdgeOtherNode(const EdgeIterator &edgeItr,
                                const NodeIterator &nodeItr) {

    EdgeEntry &edgeEntry = getEdgeEntry(edgeItr);
    if (nodeItr == edgeEntry.getNode1Itr()) {
      return edgeEntry.getNode2Itr();
    }
    //else
    return edgeEntry.getNode1Itr();
  }

  ConstNodeIterator getEdgeOtherNode(const ConstEdgeIterator &edgeItr,
                                     const ConstNodeIterator &nodeItr) const {

    const EdgeEntry &edgeEntry = getEdgeEntry(edgeItr);
    if (nodeItr == edgeEntry.getNode1Itr()) {
      return edgeEntry.getNode2Itr();
    }
    //else
    return edgeEntry.getNode1Itr();
  }

  void removeEdge(const EdgeIterator &edgeItr) {
    assert(iteratorInRange(edgeItr, edgeList.begin(), edgeList.end()) &&
           "Iterator does not belong to this graph!");

    --edgeListSize;

    // Get the edge entry.
    EdgeEntry &edgeEntry = getEdgeEntry(edgeItr);

    // Get the nodes entry.
    NodeEntry &node1Entry(getNodeEntry(edgeEntry.getNode1Itr())),
              &node2Entry(getNodeEntry(edgeEntry.getNode2Itr()));

    // Disconnect the edge from the nodes.
    node1Entry.removeAdjEdge(edgeEntry.getNode1ThisEdgeItr());
    node2Entry.removeAdjEdge(edgeEntry.getNode2ThisEdgeItr());

    // Remove the edge from the graph.
    edgeList.erase(edgeItr);
  }

  EdgeIterator edgesBegin() { return edgeList.begin(); }
  ConstEdgeIterator edgesBegin() const { return edgeList.begin(); }
  EdgeIterator edgesEnd() { return edgeList.end(); }
  ConstEdgeIterator edgesEnd() const { return edgeList.end(); }

  void clear() {
    nodeList.clear();
    nodeListSize = 0;
    edgeList.clear();
    edgeListSize = 0;
    idToNodeMap.clear();
  }

  template <typename OStream>
  void printDot(OStream &os) const {
    
    assert(areNodeIDsValid() &&
           "Cannot print a .dot of a graph unless IDs have been assigned.");
    
    os << "graph {\n";

    for (ConstNodeIterator nodeItr = nodesBegin(), nodeEnd = nodesEnd();
         nodeItr != nodeEnd; ++nodeItr) {

      os << "  node" << getNodeID(nodeItr) << " [ label=\""
         << getNodeID(nodeItr) << ": " << getNodeCosts(nodeItr) << "\" ]\n";
    }

    os << "  edge [ len=" << getNumNodes() << " ]\n";

    for (ConstEdgeIterator edgeItr = edgesBegin(), edgeEnd = edgesEnd();
         edgeItr != edgeEnd; ++edgeItr) {

      os << "  node" << getNodeID(getEdgeNode1Itr(edgeItr))
         << " -- node" << getNodeID(getEdgeNode2Itr(edgeItr))
         << " [ label=\"";

      const Matrix &edgeCosts = getEdgeCosts(edgeItr);

      for (unsigned i = 0; i < edgeCosts.getRows(); ++i) {
        os << edgeCosts.getRowAsVector(i) << "\\n";
      }

      os << "\" ]\n";
    }

    os << "}\n";
  }

  template <typename OStream>
  void printDot(OStream &os) {
    if (!areNodeIDsValid()) {
      assignNodeIDs();
    }

    const_cast<const ThisGraphT*>(this)->printDot(os);
  }

  template <typename OStream>
  void dumpTo(OStream &os) const {
    typedef ConstNodeIterator ConstNodeID;
    
    assert(areNodeIDsValid() &&
           "Cannot dump a graph unless IDs have been assigned.");

    for (ConstNodeIterator nItr = nodesBegin(), nEnd = nodesEnd();
         nItr != nEnd; ++nItr) {
      os << getNodeID(nItr) << "\n";
    }

    unsigned edgeNumber = 1;
    for (ConstEdgeIterator eItr = edgesBegin(), eEnd = edgesEnd();
         eItr != eEnd; ++eItr) {

      os << edgeNumber++ << ": { "
         << getNodeID(getEdgeNode1Itr(eItr)) << ", "
         << getNodeID(getEdgeNode2Itr(eItr)) << " }\n";
    }

  }

  template <typename OStream>
  void dumpTo(OStream &os) {
    if (!areNodeIDsValid()) {
      assignNodeIDs();
    }

    const_cast<const ThisGraphT*>(this)->dumpTo(os);
  }

};

/// \brief Provides a base from which to derive nodes for GraphBase.
template <typename NodeImpl, typename EdgeImpl>
class NodeBase {
private:

  typedef GraphBase<NodeImpl, EdgeImpl> GraphBaseT;
  typedef NodeBaseTraits<GraphBaseT> ThisNodeBaseTraits;

public:
  typedef typename GraphBaseT::EdgeIterator EdgeIterator;

private:
  typedef typename ThisNodeBaseTraits::AdjEdgeList AdjEdgeList;

  unsigned degree, id;
  Vector costs;
  AdjEdgeList adjEdges;

  void operator=(const NodeBase& other) {
    assert(false && "Can't assign NodeEntrys.");
  }

public:

  typedef typename ThisNodeBaseTraits::AdjEdgeIterator AdjEdgeIterator;
  typedef typename ThisNodeBaseTraits::ConstAdjEdgeIterator
    ConstAdjEdgeIterator;

  NodeBase(const Vector &costs) : degree(0), costs(costs) {
    assert((costs.getLength() > 0) && "Can't have zero-length cost vector.");
  }

  Vector& getCosts() { return costs; }
  const Vector& getCosts() const { return costs; }

  unsigned getDegree() const { return degree;  }

  void setID(unsigned id) { this->id = id; }
  unsigned getID() const { return id; }

  AdjEdgeIterator addAdjEdge(const EdgeIterator &edgeItr) {
    ++degree;
    return adjEdges.insert(adjEdges.end(), edgeItr);
  }

  void removeAdjEdge(const AdjEdgeIterator &adjEdgeItr) {
    --degree;
    adjEdges.erase(adjEdgeItr);
  }

  AdjEdgeIterator adjEdgesBegin() { return adjEdges.begin(); } 
  ConstAdjEdgeIterator adjEdgesBegin() const { return adjEdges.begin(); }
  AdjEdgeIterator adjEdgesEnd() { return adjEdges.end(); }
  ConstAdjEdgeIterator adjEdgesEnd() const { return adjEdges.end(); }

};

template <typename NodeImpl, typename EdgeImpl>
class EdgeBase {
public:
  typedef typename GraphBase<NodeImpl, EdgeImpl>::NodeIterator NodeIterator;
  typedef typename GraphBase<NodeImpl, EdgeImpl>::EdgeIterator EdgeIterator;

  typedef typename NodeImpl::AdjEdgeIterator NodeAdjEdgeIterator;

private:

  NodeIterator node1Itr, node2Itr;
  NodeAdjEdgeIterator node1ThisEdgeItr, node2ThisEdgeItr;
  Matrix costs;

  void operator=(const EdgeBase &other) {
    assert(false && "Can't assign EdgeEntrys.");
  }

public:

  EdgeBase(const NodeIterator &node1Itr, const NodeIterator &node2Itr,
           const Matrix &costs) :
    node1Itr(node1Itr), node2Itr(node2Itr), costs(costs) {

    assert((costs.getRows() > 0) && (costs.getCols() > 0) &&
           "Can't have zero-dimensioned cost matrices");
  }

  Matrix& getCosts() { return costs; }
  const Matrix& getCosts() const { return costs; }

  const NodeIterator& getNode1Itr() const { return node1Itr; }
  const NodeIterator& getNode2Itr() const { return node2Itr; }

  void setNode1ThisEdgeItr(const NodeAdjEdgeIterator &node1ThisEdgeItr) {
    this->node1ThisEdgeItr = node1ThisEdgeItr;
  }

  const NodeAdjEdgeIterator& getNode1ThisEdgeItr() const {
    return node1ThisEdgeItr;
  }

  void setNode2ThisEdgeItr(const NodeAdjEdgeIterator &node2ThisEdgeItr) {
    this->node2ThisEdgeItr = node2ThisEdgeItr;
  }

  const NodeAdjEdgeIterator& getNode2ThisEdgeItr() const {
    return node2ThisEdgeItr;
  }

};


}

#endif // LLVM_CODEGEN_PBQP_GRAPHBASE_HPP
