//===-------------------- Graph.h - PBQP Graph ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PBQP Graph class.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CODEGEN_PBQP_GRAPH_H
#define LLVM_CODEGEN_PBQP_GRAPH_H

#include "Math.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include <list>
#include <map>
#include <set>

namespace PBQP {

  /// PBQP Graph class.
  /// Instances of this class describe PBQP problems.
  class Graph {
  public:

    typedef unsigned NodeId;
    typedef unsigned EdgeId;

  private:

    typedef std::set<NodeId> AdjEdgeList;

  public:

    typedef AdjEdgeList::iterator AdjEdgeItr;

  private:

    class NodeEntry {
    private:
      Vector costs;
      AdjEdgeList adjEdges;
      void *data;
      NodeEntry() : costs(0, 0) {}
    public:
      NodeEntry(const Vector &costs) : costs(costs), data(0) {}
      Vector& getCosts() { return costs; }
      const Vector& getCosts() const { return costs; }
      unsigned getDegree() const { return adjEdges.size(); }
      AdjEdgeItr edgesBegin() { return adjEdges.begin(); }
      AdjEdgeItr edgesEnd() { return adjEdges.end(); }
      AdjEdgeItr addEdge(EdgeId e) {
        return adjEdges.insert(adjEdges.end(), e);
      }
      void removeEdge(AdjEdgeItr ae) {
        adjEdges.erase(ae);
      }
      void setData(void *data) { this->data = data; }
      void* getData() { return data; }
    };

    class EdgeEntry {
    private:
      NodeId node1, node2;
      Matrix costs;
      AdjEdgeItr node1AEItr, node2AEItr;
      void *data;
      EdgeEntry() : costs(0, 0, 0), data(0) {}
    public:
      EdgeEntry(NodeId node1, NodeId node2, const Matrix &costs)
        : node1(node1), node2(node2), costs(costs) {}
      NodeId getNode1() const { return node1; }
      NodeId getNode2() const { return node2; }
      Matrix& getCosts() { return costs; }
      const Matrix& getCosts() const { return costs; }
      void setNode1AEItr(AdjEdgeItr ae) { node1AEItr = ae; }
      AdjEdgeItr getNode1AEItr() { return node1AEItr; }
      void setNode2AEItr(AdjEdgeItr ae) { node2AEItr = ae; }
      AdjEdgeItr getNode2AEItr() { return node2AEItr; }
      void setData(void *data) { this->data = data; }
      void *getData() { return data; }
    };

    // ----- MEMBERS -----

    typedef std::vector<NodeEntry> NodeVector;
    typedef std::vector<NodeId> FreeNodeVector;
    NodeVector nodes;
    FreeNodeVector freeNodes;

    typedef std::vector<EdgeEntry> EdgeVector;
    typedef std::vector<EdgeId> FreeEdgeVector;
    EdgeVector edges;
    FreeEdgeVector freeEdges;

    // ----- INTERNAL METHODS -----

    NodeEntry& getNode(NodeId nId) { return nodes[nId]; }
    const NodeEntry& getNode(NodeId nId) const { return nodes[nId]; }

    EdgeEntry& getEdge(EdgeId eId) { return edges[eId]; }
    const EdgeEntry& getEdge(EdgeId eId) const { return edges[eId]; }

    NodeId addConstructedNode(const NodeEntry &n) {
      NodeId nodeId = 0;
      if (!freeNodes.empty()) {
        nodeId = freeNodes.back();
        freeNodes.pop_back();
        nodes[nodeId] = n;
      } else {
        nodeId = nodes.size();
        nodes.push_back(n);
      }
      return nodeId;
    }

    EdgeId addConstructedEdge(const EdgeEntry &e) {
      assert(findEdge(e.getNode1(), e.getNode2()) == invalidEdgeId() &&
             "Attempt to add duplicate edge.");
      EdgeId edgeId = 0;
      if (!freeEdges.empty()) {
        edgeId = freeEdges.back();
        freeEdges.pop_back();
        edges[edgeId] = e;
      } else {
        edgeId = edges.size();
        edges.push_back(e);
      }

      EdgeEntry &ne = getEdge(edgeId);
      NodeEntry &n1 = getNode(ne.getNode1());
      NodeEntry &n2 = getNode(ne.getNode2());

      // Sanity check on matrix dimensions:
      assert((n1.getCosts().getLength() == ne.getCosts().getRows()) &&
             (n2.getCosts().getLength() == ne.getCosts().getCols()) &&
             "Edge cost dimensions do not match node costs dimensions.");

      ne.setNode1AEItr(n1.addEdge(edgeId));
      ne.setNode2AEItr(n2.addEdge(edgeId));
      return edgeId;
    }

    Graph(const Graph &other) {}
    void operator=(const Graph &other) {}

  public:

    class NodeItr {
    public:
      NodeItr(NodeId nodeId, const Graph &g)
        : nodeId(nodeId), endNodeId(g.nodes.size()), freeNodes(g.freeNodes) {
        this->nodeId = findNextInUse(nodeId); // Move to the first in-use nodeId
      }

      bool operator==(const NodeItr& n) const { return nodeId == n.nodeId; }
      bool operator!=(const NodeItr& n) const { return !(*this == n); }
      NodeItr& operator++() { nodeId = findNextInUse(++nodeId); return *this; }
      NodeId operator*() const { return nodeId; }

    private:
      NodeId findNextInUse(NodeId n) const {
        while (n < endNodeId &&
               std::find(freeNodes.begin(), freeNodes.end(), n) !=
                 freeNodes.end()) {
          ++n;
        }
        return n;
      }

      NodeId nodeId, endNodeId;
      const FreeNodeVector& freeNodes;
    };

    class EdgeItr {
    public:
      EdgeItr(EdgeId edgeId, const Graph &g)
        : edgeId(edgeId), endEdgeId(g.edges.size()), freeEdges(g.freeEdges) {
        this->edgeId = findNextInUse(edgeId); // Move to the first in-use edgeId
      }

      bool operator==(const EdgeItr& n) const { return edgeId == n.edgeId; }
      bool operator!=(const EdgeItr& n) const { return !(*this == n); }
      EdgeItr& operator++() { edgeId = findNextInUse(++edgeId); return *this; }
      EdgeId operator*() const { return edgeId; }

    private:
      EdgeId findNextInUse(EdgeId n) const {
        while (n < endEdgeId &&
               std::find(freeEdges.begin(), freeEdges.end(), n) !=
                 freeEdges.end()) {
          ++n;
        }
        return n;
      }

      EdgeId edgeId, endEdgeId;
      const FreeEdgeVector& freeEdges;
    };

    /// \brief Construct an empty PBQP graph.
    Graph() {}

    /// \brief Add a node with the given costs.
    /// @param costs Cost vector for the new node.
    /// @return Node iterator for the added node.
    NodeId addNode(const Vector &costs) {
      return addConstructedNode(NodeEntry(costs));
    }

    /// \brief Add an edge between the given nodes with the given costs.
    /// @param n1Id First node.
    /// @param n2Id Second node.
    /// @return Edge iterator for the added edge.
    EdgeId addEdge(NodeId n1Id, NodeId n2Id, const Matrix &costs) {
      assert(getNodeCosts(n1Id).getLength() == costs.getRows() &&
             getNodeCosts(n2Id).getLength() == costs.getCols() &&
             "Matrix dimensions mismatch.");
      return addConstructedEdge(EdgeEntry(n1Id, n2Id, costs));
    }

    /// \brief Get the number of nodes in the graph.
    /// @return Number of nodes in the graph.
    unsigned getNumNodes() const { return nodes.size() - freeNodes.size(); }

    /// \brief Get the number of edges in the graph.
    /// @return Number of edges in the graph.
    unsigned getNumEdges() const { return edges.size() - freeEdges.size(); }

    /// \brief Get a node's cost vector.
    /// @param nId Node id.
    /// @return Node cost vector.
    Vector& getNodeCosts(NodeId nId) { return getNode(nId).getCosts(); }

    /// \brief Get a node's cost vector (const version).
    /// @param nId Node id.
    /// @return Node cost vector.
    const Vector& getNodeCosts(NodeId nId) const {
      return getNode(nId).getCosts();
    }

    /// \brief Set a node's data pointer.
    /// @param nId Node id.
    /// @param data Pointer to node data.
    ///
    /// Typically used by a PBQP solver to attach data to aid in solution.
    void setNodeData(NodeId nId, void *data) { getNode(nId).setData(data); }

    /// \brief Get the node's data pointer.
    /// @param nId Node id.
    /// @return Pointer to node data.
    void* getNodeData(NodeId nId) { return getNode(nId).getData(); }

    /// \brief Get an edge's cost matrix.
    /// @param eId Edge id.
    /// @return Edge cost matrix.
    Matrix& getEdgeCosts(EdgeId eId) { return getEdge(eId).getCosts(); }

    /// \brief Get an edge's cost matrix (const version).
    /// @param eId Edge id.
    /// @return Edge cost matrix.
    const Matrix& getEdgeCosts(EdgeId eId) const {
      return getEdge(eId).getCosts();
    }

    /// \brief Set an edge's data pointer.
    /// @param eId Edge id.
    /// @param data Pointer to edge data.
    ///
    /// Typically used by a PBQP solver to attach data to aid in solution.
    void setEdgeData(EdgeId eId, void *data) { getEdge(eId).setData(data); }

    /// \brief Get an edge's data pointer.
    /// @param eId Edge id.
    /// @return Pointer to edge data.
    void* getEdgeData(EdgeId eId) { return getEdge(eId).getData(); }

    /// \brief Get a node's degree.
    /// @param nId Node id.
    /// @return The degree of the node.
    unsigned getNodeDegree(NodeId nId) const {
      return getNode(nId).getDegree();
    }

    /// \brief Begin iterator for node set.
    NodeItr nodesBegin() const { return NodeItr(0, *this);  }

    /// \brief End iterator for node set.
    NodeItr nodesEnd() const { return NodeItr(nodes.size(), *this); }

    /// \brief Begin iterator for edge set.
    EdgeItr edgesBegin() const { return EdgeItr(0, *this); }

    /// \brief End iterator for edge set.
    EdgeItr edgesEnd() const { return EdgeItr(edges.size(), *this); }

    /// \brief Get begin iterator for adjacent edge set.
    /// @param nId Node id.
    /// @return Begin iterator for the set of edges connected to the given node.
    AdjEdgeItr adjEdgesBegin(NodeId nId) {
      return getNode(nId).edgesBegin();
    }

    /// \brief Get end iterator for adjacent edge set.
    /// @param nId Node id.
    /// @return End iterator for the set of edges connected to the given node.
    AdjEdgeItr adjEdgesEnd(NodeId nId) {
      return getNode(nId).edgesEnd();
    }

    /// \brief Get the first node connected to this edge.
    /// @param eId Edge id.
    /// @return The first node connected to the given edge.
    NodeId getEdgeNode1(EdgeId eId) {
      return getEdge(eId).getNode1();
    }

    /// \brief Get the second node connected to this edge.
    /// @param eId Edge id.
    /// @return The second node connected to the given edge.
    NodeId getEdgeNode2(EdgeId eId) {
      return getEdge(eId).getNode2();
    }

    /// \brief Get the "other" node connected to this edge.
    /// @param eId Edge id.
    /// @param nId Node id for the "given" node.
    /// @return The iterator for the "other" node connected to this edge.
    NodeId getEdgeOtherNode(EdgeId eId, NodeId nId) {
      EdgeEntry &e = getEdge(eId);
      if (e.getNode1() == nId) {
        return e.getNode2();
      } // else
      return e.getNode1();
    }

    EdgeId invalidEdgeId() const {
      return std::numeric_limits<EdgeId>::max();
    }

    /// \brief Get the edge connecting two nodes.
    /// @param n1Id First node id.
    /// @param n2Id Second node id.
    /// @return An id for edge (n1Id, n2Id) if such an edge exists,
    ///         otherwise returns an invalid edge id.
    EdgeId findEdge(NodeId n1Id, NodeId n2Id) {
      for (AdjEdgeItr aeItr = adjEdgesBegin(n1Id), aeEnd = adjEdgesEnd(n1Id);
         aeItr != aeEnd; ++aeItr) {
        if ((getEdgeNode1(*aeItr) == n2Id) ||
            (getEdgeNode2(*aeItr) == n2Id)) {
          return *aeItr;
        }
      }
      return invalidEdgeId();
    }

    /// \brief Remove a node from the graph.
    /// @param nId Node id.
    void removeNode(NodeId nId) {
      NodeEntry &n = getNode(nId);
      for (AdjEdgeItr itr = n.edgesBegin(), end = n.edgesEnd(); itr != end; ++itr) {
        EdgeId eId = *itr;
        removeEdge(eId);
      }
      freeNodes.push_back(nId);
    }

    /// \brief Remove an edge from the graph.
    /// @param eId Edge id.
    void removeEdge(EdgeId eId) {
      EdgeEntry &e = getEdge(eId);
      NodeEntry &n1 = getNode(e.getNode1());
      NodeEntry &n2 = getNode(e.getNode2());
      n1.removeEdge(e.getNode1AEItr());
      n2.removeEdge(e.getNode2AEItr());
      freeEdges.push_back(eId);
    }

    /// \brief Remove all nodes and edges from the graph.
    void clear() {
      nodes.clear();
      freeNodes.clear();
      edges.clear();
      freeEdges.clear();
    }

    /// \brief Dump a graph to an output stream.
    template <typename OStream>
    void dump(OStream &os) {
      os << getNumNodes() << " " << getNumEdges() << "\n";

      for (NodeItr nodeItr = nodesBegin(), nodeEnd = nodesEnd();
           nodeItr != nodeEnd; ++nodeItr) {
        const Vector& v = getNodeCosts(*nodeItr);
        os << "\n" << v.getLength() << "\n";
        assert(v.getLength() != 0 && "Empty vector in graph.");
        os << v[0];
        for (unsigned i = 1; i < v.getLength(); ++i) {
          os << " " << v[i];
        }
        os << "\n";
      }

      for (EdgeItr edgeItr = edgesBegin(), edgeEnd = edgesEnd();
           edgeItr != edgeEnd; ++edgeItr) {
        NodeId n1 = getEdgeNode1(*edgeItr);
        NodeId n2 = getEdgeNode2(*edgeItr);
        assert(n1 != n2 && "PBQP graphs shound not have self-edges.");
        const Matrix& m = getEdgeCosts(*edgeItr);
        os << "\n" << n1 << " " << n2 << "\n"
           << m.getRows() << " " << m.getCols() << "\n";
        assert(m.getRows() != 0 && "No rows in matrix.");
        assert(m.getCols() != 0 && "No cols in matrix.");
        for (unsigned i = 0; i < m.getRows(); ++i) {
          os << m[i][0];
          for (unsigned j = 1; j < m.getCols(); ++j) {
            os << " " << m[i][j];
          }
          os << "\n";
        }
      }
    }

    /// \brief Print a representation of this graph in DOT format.
    /// @param os Output stream to print on.
    template <typename OStream>
    void printDot(OStream &os) {

      os << "graph {\n";

      for (NodeItr nodeItr = nodesBegin(), nodeEnd = nodesEnd();
           nodeItr != nodeEnd; ++nodeItr) {

        os << "  node" << *nodeItr << " [ label=\""
           << *nodeItr << ": " << getNodeCosts(*nodeItr) << "\" ]\n";
      }

      os << "  edge [ len=" << getNumNodes() << " ]\n";

      for (EdgeItr edgeItr = edgesBegin(), edgeEnd = edgesEnd();
           edgeItr != edgeEnd; ++edgeItr) {

        os << "  node" << getEdgeNode1(*edgeItr)
           << " -- node" << getEdgeNode2(*edgeItr)
           << " [ label=\"";

        const Matrix &edgeCosts = getEdgeCosts(*edgeItr);

        for (unsigned i = 0; i < edgeCosts.getRows(); ++i) {
          os << edgeCosts.getRowAsVector(i) << "\\n";
        }
        os << "\" ]\n";
      }
      os << "}\n";
    }

  };

//  void Graph::copyFrom(const Graph &other) {
//     std::map<Graph::ConstNodeItr, Graph::NodeItr,
//              NodeItrComparator> nodeMap;

//      for (Graph::ConstNodeItr nItr = other.nodesBegin(),
//                              nEnd = other.nodesEnd();
//          nItr != nEnd; ++nItr) {
//       nodeMap[nItr] = addNode(other.getNodeCosts(nItr));
//     }
//  }

}

#endif // LLVM_CODEGEN_PBQP_GRAPH_HPP
