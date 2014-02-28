//===-- HeuristicSolver.h - Heuristic PBQP Solver --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Heuristic PBQP solver. This solver is able to perform optimal reductions for
// nodes of degree 0, 1 or 2. For nodes of degree >2 a plugable heuristic is
// used to select a node for reduction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_HEURISTICSOLVER_H
#define LLVM_CODEGEN_PBQP_HEURISTICSOLVER_H

#include "Graph.h"
#include "Solution.h"
#include <limits>
#include <vector>

namespace PBQP {

  /// \brief Heuristic PBQP solver implementation.
  ///
  /// This class should usually be created (and destroyed) indirectly via a call
  /// to HeuristicSolver<HImpl>::solve(Graph&).
  /// See the comments for HeuristicSolver.
  ///
  /// HeuristicSolverImpl provides the R0, R1 and R2 reduction rules,
  /// backpropagation phase, and maintains the internal copy of the graph on
  /// which the reduction is carried out (the original being kept to facilitate
  /// backpropagation).
  template <typename HImpl>
  class HeuristicSolverImpl {
  private:

    typedef typename HImpl::NodeData HeuristicNodeData;
    typedef typename HImpl::EdgeData HeuristicEdgeData;

    typedef std::list<Graph::EdgeId> SolverEdges;

  public:

    /// \brief Iterator type for edges in the solver graph.
    typedef SolverEdges::iterator SolverEdgeItr;

  private:

    class NodeData {
    public:
      NodeData() : solverDegree(0) {}

      HeuristicNodeData& getHeuristicData() { return hData; }

      SolverEdgeItr addSolverEdge(Graph::EdgeId eId) {
        ++solverDegree;
        return solverEdges.insert(solverEdges.end(), eId);
      }

      void removeSolverEdge(SolverEdgeItr seItr) {
        --solverDegree;
        solverEdges.erase(seItr);
      }

      SolverEdgeItr solverEdgesBegin() { return solverEdges.begin(); }
      SolverEdgeItr solverEdgesEnd() { return solverEdges.end(); }
      unsigned getSolverDegree() const { return solverDegree; }
      void clearSolverEdges() {
        solverDegree = 0;
        solverEdges.clear();
      }

    private:
      HeuristicNodeData hData;
      unsigned solverDegree;
      SolverEdges solverEdges;
    };

    class EdgeData {
    public:
      HeuristicEdgeData& getHeuristicData() { return hData; }

      void setN1SolverEdgeItr(SolverEdgeItr n1SolverEdgeItr) {
        this->n1SolverEdgeItr = n1SolverEdgeItr;
      }

      SolverEdgeItr getN1SolverEdgeItr() { return n1SolverEdgeItr; }

      void setN2SolverEdgeItr(SolverEdgeItr n2SolverEdgeItr){
        this->n2SolverEdgeItr = n2SolverEdgeItr;
      }

      SolverEdgeItr getN2SolverEdgeItr() { return n2SolverEdgeItr; }

    private:

      HeuristicEdgeData hData;
      SolverEdgeItr n1SolverEdgeItr, n2SolverEdgeItr;
    };

    Graph &g;
    HImpl h;
    Solution s;
    std::vector<Graph::NodeId> stack;

    typedef std::list<NodeData> NodeDataList;
    NodeDataList nodeDataList;

    typedef std::list<EdgeData> EdgeDataList;
    EdgeDataList edgeDataList;

  public:

    /// \brief Construct a heuristic solver implementation to solve the given
    ///        graph.
    /// @param g The graph representing the problem instance to be solved.
    HeuristicSolverImpl(Graph &g) : g(g), h(*this) {}

    /// \brief Get the graph being solved by this solver.
    /// @return The graph representing the problem instance being solved by this
    ///         solver.
    Graph& getGraph() { return g; }

    /// \brief Get the heuristic data attached to the given node.
    /// @param nId Node id.
    /// @return The heuristic data attached to the given node.
    HeuristicNodeData& getHeuristicNodeData(Graph::NodeId nId) {
      return getSolverNodeData(nId).getHeuristicData();
    }

    /// \brief Get the heuristic data attached to the given edge.
    /// @param eId Edge id.
    /// @return The heuristic data attached to the given node.
    HeuristicEdgeData& getHeuristicEdgeData(Graph::EdgeId eId) {
      return getSolverEdgeData(eId).getHeuristicData();
    }

    /// \brief Begin iterator for the set of edges adjacent to the given node in
    ///        the solver graph.
    /// @param nId Node id.
    /// @return Begin iterator for the set of edges adjacent to the given node
    ///         in the solver graph.
    SolverEdgeItr solverEdgesBegin(Graph::NodeId nId) {
      return getSolverNodeData(nId).solverEdgesBegin();
    }

    /// \brief End iterator for the set of edges adjacent to the given node in
    ///        the solver graph.
    /// @param nId Node id.
    /// @return End iterator for the set of edges adjacent to the given node in
    ///         the solver graph.
    SolverEdgeItr solverEdgesEnd(Graph::NodeId nId) {
      return getSolverNodeData(nId).solverEdgesEnd();
    }

    /// \brief Remove a node from the solver graph.
    /// @param eId Edge id for edge to be removed.
    ///
    /// Does <i>not</i> notify the heuristic of the removal. That should be
    /// done manually if necessary.
    void removeSolverEdge(Graph::EdgeId eId) {
      EdgeData &eData = getSolverEdgeData(eId);
      NodeData &n1Data = getSolverNodeData(g.getEdgeNode1(eId)),
               &n2Data = getSolverNodeData(g.getEdgeNode2(eId));

      n1Data.removeSolverEdge(eData.getN1SolverEdgeItr());
      n2Data.removeSolverEdge(eData.getN2SolverEdgeItr());
    }

    /// \brief Compute a solution to the PBQP problem instance with which this
    ///        heuristic solver was constructed.
    /// @return A solution to the PBQP problem.
    ///
    /// Performs the full PBQP heuristic solver algorithm, including setup,
    /// calls to the heuristic (which will call back to the reduction rules in
    /// this class), and cleanup.
    Solution computeSolution() {
      setup();
      h.setup();
      h.reduce();
      backpropagate();
      h.cleanup();
      cleanup();
      return s;
    }

    /// \brief Add to the end of the stack.
    /// @param nId Node id to add to the reduction stack.
    void pushToStack(Graph::NodeId nId) {
      getSolverNodeData(nId).clearSolverEdges();
      stack.push_back(nId);
    }

    /// \brief Returns the solver degree of the given node.
    /// @param nId Node id for which degree is requested.
    /// @return Node degree in the <i>solver</i> graph (not the original graph).
    unsigned getSolverDegree(Graph::NodeId nId) {
      return  getSolverNodeData(nId).getSolverDegree();
    }

    /// \brief Set the solution of the given node.
    /// @param nId Node id to set solution for.
    /// @param selection Selection for node.
    void setSolution(const Graph::NodeId &nId, unsigned selection) {
      s.setSelection(nId, selection);

      for (Graph::AdjEdgeItr aeItr = g.adjEdgesBegin(nId),
                             aeEnd = g.adjEdgesEnd(nId);
           aeItr != aeEnd; ++aeItr) {
        Graph::EdgeId eId(*aeItr);
        Graph::NodeId anId(g.getEdgeOtherNode(eId, nId));
        getSolverNodeData(anId).addSolverEdge(eId);
      }
    }

    /// \brief Apply rule R0.
    /// @param nId Node id for node to apply R0 to.
    ///
    /// Node will be automatically pushed to the solver stack.
    void applyR0(Graph::NodeId nId) {
      assert(getSolverNodeData(nId).getSolverDegree() == 0 &&
             "R0 applied to node with degree != 0.");

      // Nothing to do. Just push the node onto the reduction stack.
      pushToStack(nId);

      s.recordR0();
    }

    /// \brief Apply rule R1.
    /// @param xnId Node id for node to apply R1 to.
    ///
    /// Node will be automatically pushed to the solver stack.
    void applyR1(Graph::NodeId xnId) {
      NodeData &nd = getSolverNodeData(xnId);
      assert(nd.getSolverDegree() == 1 &&
             "R1 applied to node with degree != 1.");

      Graph::EdgeId eId = *nd.solverEdgesBegin();

      const Matrix &eCosts = g.getEdgeCosts(eId);
      const Vector &xCosts = g.getNodeCosts(xnId);

      // Duplicate a little to avoid transposing matrices.
      if (xnId == g.getEdgeNode1(eId)) {
        Graph::NodeId ynId = g.getEdgeNode2(eId);
        Vector &yCosts = g.getNodeCosts(ynId);
        for (unsigned j = 0; j < yCosts.getLength(); ++j) {
          PBQPNum min = eCosts[0][j] + xCosts[0];
          for (unsigned i = 1; i < xCosts.getLength(); ++i) {
            PBQPNum c = eCosts[i][j] + xCosts[i];
            if (c < min)
              min = c;
          }
          yCosts[j] += min;
        }
        h.handleRemoveEdge(eId, ynId);
     } else {
        Graph::NodeId ynId = g.getEdgeNode1(eId);
        Vector &yCosts = g.getNodeCosts(ynId);
        for (unsigned i = 0; i < yCosts.getLength(); ++i) {
          PBQPNum min = eCosts[i][0] + xCosts[0];
          for (unsigned j = 1; j < xCosts.getLength(); ++j) {
            PBQPNum c = eCosts[i][j] + xCosts[j];
            if (c < min)
              min = c;
          }
          yCosts[i] += min;
        }
        h.handleRemoveEdge(eId, ynId);
      }
      removeSolverEdge(eId);
      assert(nd.getSolverDegree() == 0 &&
             "Degree 1 with edge removed should be 0.");
      pushToStack(xnId);
      s.recordR1();
    }

    /// \brief Apply rule R2.
    /// @param xnId Node id for node to apply R2 to.
    ///
    /// Node will be automatically pushed to the solver stack.
    void applyR2(Graph::NodeId xnId) {
      assert(getSolverNodeData(xnId).getSolverDegree() == 2 &&
             "R2 applied to node with degree != 2.");

      NodeData &nd = getSolverNodeData(xnId);
      const Vector &xCosts = g.getNodeCosts(xnId);

      SolverEdgeItr aeItr = nd.solverEdgesBegin();
      Graph::EdgeId yxeId = *aeItr,
                    zxeId = *(++aeItr);

      Graph::NodeId ynId = g.getEdgeOtherNode(yxeId, xnId),
                    znId = g.getEdgeOtherNode(zxeId, xnId);

      bool flipEdge1 = (g.getEdgeNode1(yxeId) == xnId),
           flipEdge2 = (g.getEdgeNode1(zxeId) == xnId);

      const Matrix *yxeCosts = flipEdge1 ?
        new Matrix(g.getEdgeCosts(yxeId).transpose()) :
        &g.getEdgeCosts(yxeId);

      const Matrix *zxeCosts = flipEdge2 ?
        new Matrix(g.getEdgeCosts(zxeId).transpose()) :
        &g.getEdgeCosts(zxeId);

      unsigned xLen = xCosts.getLength(),
               yLen = yxeCosts->getRows(),
               zLen = zxeCosts->getRows();

      Matrix delta(yLen, zLen);

      for (unsigned i = 0; i < yLen; ++i) {
        for (unsigned j = 0; j < zLen; ++j) {
          PBQPNum min = (*yxeCosts)[i][0] + (*zxeCosts)[j][0] + xCosts[0];
          for (unsigned k = 1; k < xLen; ++k) {
            PBQPNum c = (*yxeCosts)[i][k] + (*zxeCosts)[j][k] + xCosts[k];
            if (c < min) {
              min = c;
            }
          }
          delta[i][j] = min;
        }
      }

      if (flipEdge1)
        delete yxeCosts;

      if (flipEdge2)
        delete zxeCosts;

      Graph::EdgeId yzeId = g.findEdge(ynId, znId);
      bool addedEdge = false;

      if (yzeId == g.invalidEdgeId()) {
        yzeId = g.addEdge(ynId, znId, delta);
        addedEdge = true;
      } else {
        Matrix &yzeCosts = g.getEdgeCosts(yzeId);
        h.preUpdateEdgeCosts(yzeId);
        if (ynId == g.getEdgeNode1(yzeId)) {
          yzeCosts += delta;
        } else {
          yzeCosts += delta.transpose();
        }
      }

      bool nullCostEdge = tryNormaliseEdgeMatrix(yzeId);

      if (!addedEdge) {
        // If we modified the edge costs let the heuristic know.
        h.postUpdateEdgeCosts(yzeId);
      }

      if (nullCostEdge) {
        // If this edge ended up null remove it.
        if (!addedEdge) {
          // We didn't just add it, so we need to notify the heuristic
          // and remove it from the solver.
          h.handleRemoveEdge(yzeId, ynId);
          h.handleRemoveEdge(yzeId, znId);
          removeSolverEdge(yzeId);
        }
        g.removeEdge(yzeId);
      } else if (addedEdge) {
        // If the edge was added, and non-null, finish setting it up, add it to
        // the solver & notify heuristic.
        edgeDataList.push_back(EdgeData());
        g.setEdgeData(yzeId, &edgeDataList.back());
        addSolverEdge(yzeId);
        h.handleAddEdge(yzeId);
      }

      h.handleRemoveEdge(yxeId, ynId);
      removeSolverEdge(yxeId);
      h.handleRemoveEdge(zxeId, znId);
      removeSolverEdge(zxeId);

      pushToStack(xnId);
      s.recordR2();
    }

    /// \brief Record an application of the RN rule.
    ///
    /// For use by the HeuristicBase.
    void recordRN() { s.recordRN(); }

  private:

    NodeData& getSolverNodeData(Graph::NodeId nId) {
      return *static_cast<NodeData*>(g.getNodeData(nId));
    }

    EdgeData& getSolverEdgeData(Graph::EdgeId eId) {
      return *static_cast<EdgeData*>(g.getEdgeData(eId));
    }

    void addSolverEdge(Graph::EdgeId eId) {
      EdgeData &eData = getSolverEdgeData(eId);
      NodeData &n1Data = getSolverNodeData(g.getEdgeNode1(eId)),
               &n2Data = getSolverNodeData(g.getEdgeNode2(eId));

      eData.setN1SolverEdgeItr(n1Data.addSolverEdge(eId));
      eData.setN2SolverEdgeItr(n2Data.addSolverEdge(eId));
    }

    void setup() {
      if (h.solverRunSimplify()) {
        simplify();
      }

      // Create node data objects.
      for (Graph::NodeItr nItr = g.nodesBegin(), nEnd = g.nodesEnd();
           nItr != nEnd; ++nItr) {
        nodeDataList.push_back(NodeData());
        g.setNodeData(*nItr, &nodeDataList.back());
      }

      // Create edge data objects.
      for (Graph::EdgeItr eItr = g.edgesBegin(), eEnd = g.edgesEnd();
           eItr != eEnd; ++eItr) {
        edgeDataList.push_back(EdgeData());
        g.setEdgeData(*eItr, &edgeDataList.back());
        addSolverEdge(*eItr);
      }
    }

    void simplify() {
      disconnectTrivialNodes();
      eliminateIndependentEdges();
    }

    // Eliminate trivial nodes.
    void disconnectTrivialNodes() {
      unsigned numDisconnected = 0;

      for (Graph::NodeItr nItr = g.nodesBegin(), nEnd = g.nodesEnd();
           nItr != nEnd; ++nItr) {

        Graph::NodeId nId = *nItr;

        if (g.getNodeCosts(nId).getLength() == 1) {

          std::vector<Graph::EdgeId> edgesToRemove;

          for (Graph::AdjEdgeItr aeItr = g.adjEdgesBegin(nId),
                                 aeEnd = g.adjEdgesEnd(nId);
               aeItr != aeEnd; ++aeItr) {

            Graph::EdgeId eId = *aeItr;

            if (g.getEdgeNode1(eId) == nId) {
              Graph::NodeId otherNodeId = g.getEdgeNode2(eId);
              g.getNodeCosts(otherNodeId) +=
                g.getEdgeCosts(eId).getRowAsVector(0);
            }
            else {
              Graph::NodeId otherNodeId = g.getEdgeNode1(eId);
              g.getNodeCosts(otherNodeId) +=
                g.getEdgeCosts(eId).getColAsVector(0);
            }

            edgesToRemove.push_back(eId);
          }

          if (!edgesToRemove.empty())
            ++numDisconnected;

          while (!edgesToRemove.empty()) {
            g.removeEdge(edgesToRemove.back());
            edgesToRemove.pop_back();
          }
        }
      }
    }

    void eliminateIndependentEdges() {
      std::vector<Graph::EdgeId> edgesToProcess;
      unsigned numEliminated = 0;

      for (Graph::EdgeItr eItr = g.edgesBegin(), eEnd = g.edgesEnd();
           eItr != eEnd; ++eItr) {
        edgesToProcess.push_back(*eItr);
      }

      while (!edgesToProcess.empty()) {
        if (tryToEliminateEdge(edgesToProcess.back()))
          ++numEliminated;
        edgesToProcess.pop_back();
      }
    }

    bool tryToEliminateEdge(Graph::EdgeId eId) {
      if (tryNormaliseEdgeMatrix(eId)) {
        g.removeEdge(eId);
        return true;
      }
      return false;
    }

    bool tryNormaliseEdgeMatrix(Graph::EdgeId &eId) {

      const PBQPNum infinity = std::numeric_limits<PBQPNum>::infinity();

      Matrix &edgeCosts = g.getEdgeCosts(eId);
      Vector &uCosts = g.getNodeCosts(g.getEdgeNode1(eId)),
             &vCosts = g.getNodeCosts(g.getEdgeNode2(eId));

      for (unsigned r = 0; r < edgeCosts.getRows(); ++r) {
        PBQPNum rowMin = infinity;

        for (unsigned c = 0; c < edgeCosts.getCols(); ++c) {
          if (vCosts[c] != infinity && edgeCosts[r][c] < rowMin)
            rowMin = edgeCosts[r][c];
        }

        uCosts[r] += rowMin;

        if (rowMin != infinity) {
          edgeCosts.subFromRow(r, rowMin);
        }
        else {
          edgeCosts.setRow(r, 0);
        }
      }

      for (unsigned c = 0; c < edgeCosts.getCols(); ++c) {
        PBQPNum colMin = infinity;

        for (unsigned r = 0; r < edgeCosts.getRows(); ++r) {
          if (uCosts[r] != infinity && edgeCosts[r][c] < colMin)
            colMin = edgeCosts[r][c];
        }

        vCosts[c] += colMin;

        if (colMin != infinity) {
          edgeCosts.subFromCol(c, colMin);
        }
        else {
          edgeCosts.setCol(c, 0);
        }
      }

      return edgeCosts.isZero();
    }

    void backpropagate() {
      while (!stack.empty()) {
        computeSolution(stack.back());
        stack.pop_back();
      }
    }

    void computeSolution(Graph::NodeId nId) {

      NodeData &nodeData = getSolverNodeData(nId);

      Vector v(g.getNodeCosts(nId));

      // Solve based on existing solved edges.
      for (SolverEdgeItr solvedEdgeItr = nodeData.solverEdgesBegin(),
                         solvedEdgeEnd = nodeData.solverEdgesEnd();
           solvedEdgeItr != solvedEdgeEnd; ++solvedEdgeItr) {

        Graph::EdgeId eId(*solvedEdgeItr);
        Matrix &edgeCosts = g.getEdgeCosts(eId);

        if (nId == g.getEdgeNode1(eId)) {
          Graph::NodeId adjNode(g.getEdgeNode2(eId));
          unsigned adjSolution = s.getSelection(adjNode);
          v += edgeCosts.getColAsVector(adjSolution);
        }
        else {
          Graph::NodeId adjNode(g.getEdgeNode1(eId));
          unsigned adjSolution = s.getSelection(adjNode);
          v += edgeCosts.getRowAsVector(adjSolution);
        }

      }

      setSolution(nId, v.minIndex());
    }

    void cleanup() {
      h.cleanup();
      nodeDataList.clear();
      edgeDataList.clear();
    }
  };

  /// \brief PBQP heuristic solver class.
  ///
  /// Given a PBQP Graph g representing a PBQP problem, you can find a solution
  /// by calling
  /// <tt>Solution s = HeuristicSolver<H>::solve(g);</tt>
  ///
  /// The choice of heuristic for the H parameter will affect both the solver
  /// speed and solution quality. The heuristic should be chosen based on the
  /// nature of the problem being solved.
  /// Currently the only solver included with LLVM is the Briggs heuristic for
  /// register allocation.
  template <typename HImpl>
  class HeuristicSolver {
  public:
    static Solution solve(Graph &g) {
      HeuristicSolverImpl<HImpl> hs(g);
      return hs.computeSolution();
    }
  };

}

#endif // LLVM_CODEGEN_PBQP_HEURISTICSOLVER_H
