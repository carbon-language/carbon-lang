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

#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/Compiler.h"
#include <list>
#include <map>
#include <set>

namespace PBQP {

  class GraphBase {
  public:
    typedef unsigned NodeId;
    typedef unsigned EdgeId;
  };

  /// PBQP Graph class.
  /// Instances of this class describe PBQP problems.
  ///
  template <typename SolverT>
  class Graph : public GraphBase {
  private:
    typedef typename SolverT::CostAllocator CostAllocator;
  public:
    typedef typename SolverT::RawVector RawVector;
    typedef typename SolverT::RawMatrix RawMatrix;
    typedef typename SolverT::Vector Vector;
    typedef typename SolverT::Matrix Matrix;
    typedef typename CostAllocator::VectorPtr VectorPtr;
    typedef typename CostAllocator::MatrixPtr MatrixPtr;
    typedef typename SolverT::NodeMetadata NodeMetadata;
    typedef typename SolverT::EdgeMetadata EdgeMetadata;

  private:

    class NodeEntry {
    public:
      typedef std::set<NodeId> AdjEdgeList;
      typedef AdjEdgeList::const_iterator AdjEdgeItr;
      NodeEntry(VectorPtr Costs) : Costs(Costs) {}

      VectorPtr Costs;
      NodeMetadata Metadata;
      AdjEdgeList AdjEdgeIds;
    };

    class EdgeEntry {
    public:
      EdgeEntry(NodeId N1Id, NodeId N2Id, MatrixPtr Costs)
        : Costs(Costs), N1Id(N1Id), N2Id(N2Id) {}
      void invalidate() {
        N1Id = N2Id = Graph::invalidNodeId();
        Costs = nullptr;
      }
      NodeId getN1Id() const { return N1Id; }
      NodeId getN2Id() const { return N2Id; }
      MatrixPtr Costs;
      EdgeMetadata Metadata;
    private:
      NodeId N1Id, N2Id;
    };

    // ----- MEMBERS -----

    CostAllocator CostAlloc;
    SolverT *Solver;

    typedef std::vector<NodeEntry> NodeVector;
    typedef std::vector<NodeId> FreeNodeVector;
    NodeVector Nodes;
    FreeNodeVector FreeNodeIds;

    typedef std::vector<EdgeEntry> EdgeVector;
    typedef std::vector<EdgeId> FreeEdgeVector;
    EdgeVector Edges;
    FreeEdgeVector FreeEdgeIds;

    // ----- INTERNAL METHODS -----

    NodeEntry& getNode(NodeId NId) { return Nodes[NId]; }
    const NodeEntry& getNode(NodeId NId) const { return Nodes[NId]; }

    EdgeEntry& getEdge(EdgeId EId) { return Edges[EId]; }
    const EdgeEntry& getEdge(EdgeId EId) const { return Edges[EId]; }

    NodeId addConstructedNode(const NodeEntry &N) {
      NodeId NId = 0;
      if (!FreeNodeIds.empty()) {
        NId = FreeNodeIds.back();
        FreeNodeIds.pop_back();
        Nodes[NId] = std::move(N);
      } else {
        NId = Nodes.size();
        Nodes.push_back(std::move(N));
      }
      return NId;
    }

    EdgeId addConstructedEdge(const EdgeEntry &E) {
      assert(findEdge(E.getN1Id(), E.getN2Id()) == invalidEdgeId() &&
             "Attempt to add duplicate edge.");
      EdgeId EId = 0;
      if (!FreeEdgeIds.empty()) {
        EId = FreeEdgeIds.back();
        FreeEdgeIds.pop_back();
        Edges[EId] = std::move(E);
      } else {
        EId = Edges.size();
        Edges.push_back(std::move(E));
      }

      EdgeEntry &NE = getEdge(EId);
      NodeEntry &N1 = getNode(NE.getN1Id());
      NodeEntry &N2 = getNode(NE.getN2Id());

      // Sanity check on matrix dimensions:
      assert((N1.Costs->getLength() == NE.Costs->getRows()) &&
             (N2.Costs->getLength() == NE.Costs->getCols()) &&
             "Edge cost dimensions do not match node costs dimensions.");

      N1.AdjEdgeIds.insert(EId);
      N2.AdjEdgeIds.insert(EId);
      return EId;
    }

    Graph(const Graph &Other) {}
    void operator=(const Graph &Other) {}

  public:

    typedef typename NodeEntry::AdjEdgeItr AdjEdgeItr;

    class NodeItr {
    public:
      NodeItr(NodeId CurNId, const Graph &G)
        : CurNId(CurNId), EndNId(G.Nodes.size()), FreeNodeIds(G.FreeNodeIds) {
        this->CurNId = findNextInUse(CurNId); // Move to first in-use node id
      }

      bool operator==(const NodeItr &O) const { return CurNId == O.CurNId; }
      bool operator!=(const NodeItr &O) const { return !(*this == O); }
      NodeItr& operator++() { CurNId = findNextInUse(++CurNId); return *this; }
      NodeId operator*() const { return CurNId; }

    private:
      NodeId findNextInUse(NodeId NId) const {
        while (NId < EndNId &&
               std::find(FreeNodeIds.begin(), FreeNodeIds.end(), NId) !=
                 FreeNodeIds.end()) {
          ++NId;
        }
        return NId;
      }

      NodeId CurNId, EndNId;
      const FreeNodeVector &FreeNodeIds;
    };

    class EdgeItr {
    public:
      EdgeItr(EdgeId CurEId, const Graph &G)
        : CurEId(CurEId), EndEId(G.Edges.size()), FreeEdgeIds(G.FreeEdgeIds) {
        this->CurEId = findNextInUse(CurEId); // Move to first in-use edge id
      }

      bool operator==(const EdgeItr &O) const { return CurEId == O.CurEId; }
      bool operator!=(const EdgeItr &O) const { return !(*this == O); }
      EdgeItr& operator++() { CurEId = findNextInUse(++CurEId); return *this; }
      EdgeId operator*() const { return CurEId; }

    private:
      EdgeId findNextInUse(EdgeId EId) const {
        while (EId < EndEId &&
               std::find(FreeEdgeIds.begin(), FreeEdgeIds.end(), EId) !=
               FreeEdgeIds.end()) {
          ++EId;
        }
        return EId;
      }

      EdgeId CurEId, EndEId;
      const FreeEdgeVector &FreeEdgeIds;
    };

    class NodeIdSet {
    public:
      NodeIdSet(const Graph &G) : G(G) { }
      NodeItr begin() const { return NodeItr(0, G); }
      NodeItr end() const { return NodeItr(G.Nodes.size(), G); }
      bool empty() const { return G.Nodes.empty(); }
      typename NodeVector::size_type size() const {
        return G.Nodes.size() - G.FreeNodeIds.size();
      }
    private:
      const Graph& G;
    };

    class EdgeIdSet {
    public:
      EdgeIdSet(const Graph &G) : G(G) { }
      EdgeItr begin() const { return EdgeItr(0, G); }
      EdgeItr end() const { return EdgeItr(G.Edges.size(), G); }
      bool empty() const { return G.Edges.empty(); }
      typename NodeVector::size_type size() const {
        return G.Edges.size() - G.FreeEdgeIds.size();
      }
    private:
      const Graph& G;
    };

    class AdjEdgeIdSet {
    public:
      AdjEdgeIdSet(const NodeEntry &NE) : NE(NE) { }
      typename NodeEntry::AdjEdgeItr begin() const {
        return NE.AdjEdgeIds.begin();
      }
      typename NodeEntry::AdjEdgeItr end() const {
        return NE.AdjEdgeIds.end();
      }
      bool empty() const { return NE.AdjEdges.empty(); }
      typename NodeEntry::AdjEdgeList::size_type size() const {
        return NE.AdjEdgeIds.size();
      }
    private:
      const NodeEntry &NE;
    };

    /// \brief Construct an empty PBQP graph.
    Graph() : Solver(nullptr) { }

    /// \brief Lock this graph to the given solver instance in preparation
    /// for running the solver. This method will call solver.handleAddNode for
    /// each node in the graph, and handleAddEdge for each edge, to give the
    /// solver an opportunity to set up any requried metadata.
    void setSolver(SolverT &S) {
      assert(Solver == nullptr && "Solver already set. Call unsetSolver().");
      Solver = &S;
      for (auto NId : nodeIds())
        Solver->handleAddNode(NId);
      for (auto EId : edgeIds())
        Solver->handleAddEdge(EId);
    }

    /// \brief Release from solver instance.
    void unsetSolver() {
      assert(Solver != nullptr && "Solver not set.");
      Solver = nullptr;
    }

    /// \brief Add a node with the given costs.
    /// @param Costs Cost vector for the new node.
    /// @return Node iterator for the added node.
    template <typename OtherVectorT>
    NodeId addNode(OtherVectorT Costs) {
      // Get cost vector from the problem domain
      VectorPtr AllocatedCosts = CostAlloc.getVector(std::move(Costs));
      NodeId NId = addConstructedNode(NodeEntry(AllocatedCosts));
      if (Solver)
        Solver->handleAddNode(NId);
      return NId;
    }

    /// \brief Add an edge between the given nodes with the given costs.
    /// @param N1Id First node.
    /// @param N2Id Second node.
    /// @return Edge iterator for the added edge.
    template <typename OtherVectorT>
    EdgeId addEdge(NodeId N1Id, NodeId N2Id, OtherVectorT Costs) {
      assert(getNodeCosts(N1Id).getLength() == Costs.getRows() &&
             getNodeCosts(N2Id).getLength() == Costs.getCols() &&
             "Matrix dimensions mismatch.");
      // Get cost matrix from the problem domain.
      MatrixPtr AllocatedCosts = CostAlloc.getMatrix(std::move(Costs));
      EdgeId EId = addConstructedEdge(EdgeEntry(N1Id, N2Id, AllocatedCosts));
      if (Solver)
        Solver->handleAddEdge(EId);
      return EId;
    }

    /// \brief Returns true if the graph is empty.
    bool empty() const { return NodeIdSet(*this).empty(); }

    NodeIdSet nodeIds() const { return NodeIdSet(*this); }
    EdgeIdSet edgeIds() const { return EdgeIdSet(*this); }

    AdjEdgeIdSet adjEdgeIds(NodeId NId) { return AdjEdgeIdSet(getNode(NId)); }

    /// \brief Get the number of nodes in the graph.
    /// @return Number of nodes in the graph.
    unsigned getNumNodes() const { return NodeIdSet(*this).size(); }

    /// \brief Get the number of edges in the graph.
    /// @return Number of edges in the graph.
    unsigned getNumEdges() const { return EdgeIdSet(*this).size(); }

    /// \brief Set a node's cost vector.
    /// @param NId Node to update.
    /// @param Costs New costs to set.
    template <typename OtherVectorT>
    void setNodeCosts(NodeId NId, OtherVectorT Costs) {
      VectorPtr AllocatedCosts = CostAlloc.getVector(std::move(Costs));
      if (Solver)
        Solver->handleSetNodeCosts(NId, *AllocatedCosts);
      getNode(NId).Costs = AllocatedCosts;
    }

    /// \brief Get a node's cost vector (const version).
    /// @param NId Node id.
    /// @return Node cost vector.
    const Vector& getNodeCosts(NodeId NId) const {
      return *getNode(NId).Costs;
    }

    NodeMetadata& getNodeMetadata(NodeId NId) {
      return getNode(NId).Metadata;
    }

    const NodeMetadata& getNodeMetadata(NodeId NId) const {
      return getNode(NId).Metadata;
    }

    typename NodeEntry::AdjEdgeList::size_type getNodeDegree(NodeId NId) const {
      return getNode(NId).AdjEdgeIds.size();
    }

    /// \brief Set an edge's cost matrix.
    /// @param EId Edge id.
    /// @param Costs New cost matrix.
    template <typename OtherMatrixT>
    void setEdgeCosts(EdgeId EId, OtherMatrixT Costs) {
      MatrixPtr AllocatedCosts = CostAlloc.getMatrix(std::move(Costs));
      if (Solver)
        Solver->handleSetEdgeCosts(EId, *AllocatedCosts);
      getEdge(EId).Costs = AllocatedCosts;
    }

    /// \brief Get an edge's cost matrix (const version).
    /// @param EId Edge id.
    /// @return Edge cost matrix.
    const Matrix& getEdgeCosts(EdgeId EId) const { return *getEdge(EId).Costs; }

    EdgeMetadata& getEdgeMetadata(EdgeId NId) {
      return getEdge(NId).Metadata;
    }

    const EdgeMetadata& getEdgeMetadata(EdgeId NId) const {
      return getEdge(NId).Metadata;
    }

    /// \brief Get the first node connected to this edge.
    /// @param EId Edge id.
    /// @return The first node connected to the given edge.
    NodeId getEdgeNode1Id(EdgeId EId) {
      return getEdge(EId).getN1Id();
    }

    /// \brief Get the second node connected to this edge.
    /// @param EId Edge id.
    /// @return The second node connected to the given edge.
    NodeId getEdgeNode2Id(EdgeId EId) {
      return getEdge(EId).getN2Id();
    }

    /// \brief Get the "other" node connected to this edge.
    /// @param EId Edge id.
    /// @param NId Node id for the "given" node.
    /// @return The iterator for the "other" node connected to this edge.
    NodeId getEdgeOtherNodeId(EdgeId EId, NodeId NId) {
      EdgeEntry &E = getEdge(EId);
      if (E.getN1Id() == NId) {
        return E.getN2Id();
      } // else
      return E.getN1Id();
    }

    /// \brief Returns a value representing an invalid (non-existant) node.
    static NodeId invalidNodeId() {
      return std::numeric_limits<NodeId>::max();
    }

    /// \brief Returns a value representing an invalid (non-existant) edge.
    static EdgeId invalidEdgeId() {
      return std::numeric_limits<EdgeId>::max();
    }

    /// \brief Get the edge connecting two nodes.
    /// @param N1Id First node id.
    /// @param N2Id Second node id.
    /// @return An id for edge (N1Id, N2Id) if such an edge exists,
    ///         otherwise returns an invalid edge id.
    EdgeId findEdge(NodeId N1Id, NodeId N2Id) {
      for (auto AEId : adjEdgeIds(N1Id)) {
        if ((getEdgeNode1Id(AEId) == N2Id) ||
            (getEdgeNode2Id(AEId) == N2Id)) {
          return AEId;
        }
      }
      return invalidEdgeId();
    }

    /// \brief Remove a node from the graph.
    /// @param NId Node id.
    void removeNode(NodeId NId) {
      if (Solver)
        Solver->handleRemoveNode(NId);
      NodeEntry &N = getNode(NId);
      // TODO: Can this be for-each'd?
      for (AdjEdgeItr AEItr = N.adjEdgesBegin(),
                      AEEnd = N.adjEdgesEnd();
           AEItr != AEEnd;) {
        EdgeId EId = *AEItr;
        ++AEItr;
        removeEdge(EId);
      }
      FreeNodeIds.push_back(NId);
    }

    /// \brief Disconnect an edge from the given node.
    ///
    /// Removes the given edge from the adjacency list of the given node.
    /// This operation leaves the edge in an 'asymmetric' state: It will no
    /// longer appear in an iteration over the given node's (NId's) edges, but
    /// will appear in an iteration over the 'other', unnamed node's edges.
    ///
    /// This does not correspond to any normal graph operation, but exists to
    /// support efficient PBQP graph-reduction based solvers. It is used to
    /// 'effectively' remove the unnamed node from the graph while the solver
    /// is performing the reduction. The solver will later call reconnectNode
    /// to restore the edge in the named node's adjacency list.
    ///
    /// Since the degree of a node is the number of connected edges,
    /// disconnecting an edge from a node 'u' will cause the degree of 'u' to
    /// drop by 1.
    ///
    /// A disconnected edge WILL still appear in an iteration over the graph
    /// edges.
    ///
    /// A disconnected edge should not be removed from the graph, it should be
    /// reconnected first.
    ///
    /// A disconnected edge can be reconnected by calling the reconnectEdge
    /// method.
    void disconnectEdge(EdgeId EId, NodeId NId) {
      if (Solver)
        Solver->handleDisconnectEdge(EId, NId);
      NodeEntry &N = getNode(NId);
      N.AdjEdgeIds.erase(EId);
    }

    /// \brief Convenience method to disconnect all neighbours from the given
    ///        node.
    void disconnectAllNeighborsFromNode(NodeId NId) {
      for (auto AEId : adjEdgeIds(NId))
        disconnectEdge(AEId, getEdgeOtherNodeId(AEId, NId));
    }

    /// \brief Re-attach an edge to its nodes.
    ///
    /// Adds an edge that had been previously disconnected back into the
    /// adjacency set of the nodes that the edge connects.
    void reconnectEdge(EdgeId EId, NodeId NId) {
      NodeEntry &N = getNode(NId);
      N.addAdjEdge(EId);
      if (Solver)
        Solver->handleReconnectEdge(EId, NId);
    }

    /// \brief Remove an edge from the graph.
    /// @param EId Edge id.
    void removeEdge(EdgeId EId) {
      if (Solver)
        Solver->handleRemoveEdge(EId);
      EdgeEntry &E = getEdge(EId);
      NodeEntry &N1 = getNode(E.getNode1());
      NodeEntry &N2 = getNode(E.getNode2());
      N1.removeEdge(EId);
      N2.removeEdge(EId);
      FreeEdgeIds.push_back(EId);
      Edges[EId].invalidate();
    }

    /// \brief Remove all nodes and edges from the graph.
    void clear() {
      Nodes.clear();
      FreeNodeIds.clear();
      Edges.clear();
      FreeEdgeIds.clear();
    }

    /// \brief Dump a graph to an output stream.
    template <typename OStream>
    void dump(OStream &OS) {
      OS << nodeIds().size() << " " << edgeIds().size() << "\n";

      for (auto NId : nodeIds()) {
        const Vector& V = getNodeCosts(NId);
        OS << "\n" << V.getLength() << "\n";
        assert(V.getLength() != 0 && "Empty vector in graph.");
        OS << V[0];
        for (unsigned i = 1; i < V.getLength(); ++i) {
          OS << " " << V[i];
        }
        OS << "\n";
      }

      for (auto EId : edgeIds()) {
        NodeId N1Id = getEdgeNode1Id(EId);
        NodeId N2Id = getEdgeNode2Id(EId);
        assert(N1Id != N2Id && "PBQP graphs shound not have self-edges.");
        const Matrix& M = getEdgeCosts(EId);
        OS << "\n" << N1Id << " " << N2Id << "\n"
           << M.getRows() << " " << M.getCols() << "\n";
        assert(M.getRows() != 0 && "No rows in matrix.");
        assert(M.getCols() != 0 && "No cols in matrix.");
        for (unsigned i = 0; i < M.getRows(); ++i) {
          OS << M[i][0];
          for (unsigned j = 1; j < M.getCols(); ++j) {
            OS << " " << M[i][j];
          }
          OS << "\n";
        }
      }
    }

    /// \brief Print a representation of this graph in DOT format.
    /// @param OS Output stream to print on.
    template <typename OStream>
    void printDot(OStream &OS) {
      OS << "graph {\n";
      for (auto NId : nodeIds()) {
        OS << "  node" << NId << " [ label=\""
           << NId << ": " << getNodeCosts(NId) << "\" ]\n";
      }
      OS << "  edge [ len=" << nodeIds().size() << " ]\n";
      for (auto EId : edgeIds()) {
        OS << "  node" << getEdgeNode1Id(EId)
           << " -- node" << getEdgeNode2Id(EId)
           << " [ label=\"";
        const Matrix &EdgeCosts = getEdgeCosts(EId);
        for (unsigned i = 0; i < EdgeCosts.getRows(); ++i) {
          OS << EdgeCosts.getRowAsVector(i) << "\\n";
        }
        OS << "\" ]\n";
      }
      OS << "}\n";
    }
  };

}

#endif // LLVM_CODEGEN_PBQP_GRAPH_HPP
