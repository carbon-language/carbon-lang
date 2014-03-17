//===-- RegAllocSolver.h - Heuristic PBQP Solver for reg alloc --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Heuristic PBQP solver for register allocation problems. This solver uses a
// graph reduction approach. Nodes of degree 0, 1 and 2 are eliminated with
// optimality-preserving rules (see ReductionRules.h). When no low-degree (<3)
// nodes are present, a heuristic derived from Brigg's graph coloring approach
// is used.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_REGALLOCSOLVER_H
#define LLVM_CODEGEN_PBQP_REGALLOCSOLVER_H

#include "CostAllocator.h"
#include "Graph.h"
#include "ReductionRules.h"
#include "Solution.h"
#include "llvm/Support/ErrorHandling.h"
#include <limits>
#include <vector>

namespace PBQP {

  namespace RegAlloc {

    /// \brief Metadata to speed allocatability test.
    ///
    /// Keeps track of the number of infinities in each row and column.
    class MatrixMetadata {
    private:
      MatrixMetadata(const MatrixMetadata&);
      void operator=(const MatrixMetadata&);
    public:
      MatrixMetadata(const PBQP::Matrix& M)
        : WorstRow(0), WorstCol(0),
          UnsafeRows(new bool[M.getRows() - 1]()),
          UnsafeCols(new bool[M.getCols() - 1]()) {

        unsigned* ColCounts = new unsigned[M.getCols() - 1]();

        for (unsigned i = 1; i < M.getRows(); ++i) {
          unsigned RowCount = 0;
          for (unsigned j = 1; j < M.getCols(); ++j) {
            if (M[i][j] == std::numeric_limits<PBQP::PBQPNum>::infinity()) {
              ++RowCount;
              ++ColCounts[j - 1];
              UnsafeRows[i - 1] = true;
              UnsafeCols[j - 1] = true;
            }
          }
          WorstRow = std::max(WorstRow, RowCount);
        }
        unsigned WorstColCountForCurRow =
          *std::max_element(ColCounts, ColCounts + M.getCols() - 1);
        WorstCol = std::max(WorstCol, WorstColCountForCurRow);
        delete[] ColCounts;
      }

      ~MatrixMetadata() {
        delete[] UnsafeRows;
        delete[] UnsafeCols;
      }

      unsigned getWorstRow() const { return WorstRow; }
      unsigned getWorstCol() const { return WorstCol; }
      const bool* getUnsafeRows() const { return UnsafeRows; }
      const bool* getUnsafeCols() const { return UnsafeCols; }

    private:
      unsigned WorstRow, WorstCol;
      bool* UnsafeRows;
      bool* UnsafeCols;
    };

    class NodeMetadata {
    public:
      typedef enum { Unprocessed,
                     OptimallyReducible,
                     ConservativelyAllocatable,
                     NotProvablyAllocatable } ReductionState;

      NodeMetadata() : RS(Unprocessed), DeniedOpts(0), OptUnsafeEdges(0) {}
      ~NodeMetadata() { delete[] OptUnsafeEdges; }

      void setup(const Vector& Costs) {
        NumOpts = Costs.getLength() - 1;
        OptUnsafeEdges = new unsigned[NumOpts]();
      }

      ReductionState getReductionState() const { return RS; }
      void setReductionState(ReductionState RS) { this->RS = RS; }

      void handleAddEdge(const MatrixMetadata& MD, bool Transpose) {
        DeniedOpts += Transpose ? MD.getWorstCol() : MD.getWorstRow();
        const bool* UnsafeOpts =
          Transpose ? MD.getUnsafeCols() : MD.getUnsafeRows();
        for (unsigned i = 0; i < NumOpts; ++i)
          OptUnsafeEdges[i] += UnsafeOpts[i];
      }

      void handleRemoveEdge(const MatrixMetadata& MD, bool Transpose) {
        DeniedOpts -= Transpose ? MD.getWorstCol() : MD.getWorstRow();
        const bool* UnsafeOpts =
          Transpose ? MD.getUnsafeCols() : MD.getUnsafeRows();
        for (unsigned i = 0; i < NumOpts; ++i)
          OptUnsafeEdges[i] -= UnsafeOpts[i];
      }

      bool isConservativelyAllocatable() const {
        return (DeniedOpts < NumOpts) ||
               (std::find(OptUnsafeEdges, OptUnsafeEdges + NumOpts, 0) !=
                  OptUnsafeEdges + NumOpts);
      }

    private:
      ReductionState RS;
      unsigned NumOpts;
      unsigned DeniedOpts;
      unsigned* OptUnsafeEdges;
    };

    class RegAllocSolverImpl {
    private:
      typedef PBQP::MDMatrix<MatrixMetadata> RAMatrix;
    public:
      typedef PBQP::Vector RawVector;
      typedef PBQP::Matrix RawMatrix;
      typedef PBQP::Vector Vector;
      typedef RAMatrix     Matrix;
      typedef PBQP::PoolCostAllocator<
                Vector, PBQP::VectorComparator,
                Matrix, PBQP::MatrixComparator> CostAllocator;

      typedef PBQP::GraphBase::NodeId NodeId;
      typedef PBQP::GraphBase::EdgeId EdgeId;

      typedef RegAlloc::NodeMetadata NodeMetadata;

      struct EdgeMetadata { };

      typedef PBQP::Graph<RegAllocSolverImpl> Graph;

      RegAllocSolverImpl(Graph &G) : G(G) {}

      Solution solve() {
        G.setSolver(*this);
        Solution S;
        setup();
        S = backpropagate(G, reduce());
        G.unsetSolver();
        return S;
      }

      void handleAddNode(NodeId NId) {
        G.getNodeMetadata(NId).setup(G.getNodeCosts(NId));
      }
      void handleRemoveNode(NodeId NId) {}
      void handleSetNodeCosts(NodeId NId, const Vector& newCosts) {}

      void handleAddEdge(EdgeId EId) {
        handleReconnectEdge(EId, G.getEdgeNode1Id(EId));
        handleReconnectEdge(EId, G.getEdgeNode2Id(EId));
      }

      void handleRemoveEdge(EdgeId EId) {
        handleDisconnectEdge(EId, G.getEdgeNode1Id(EId));
        handleDisconnectEdge(EId, G.getEdgeNode2Id(EId));
      }

      void handleDisconnectEdge(EdgeId EId, NodeId NId) {
        NodeMetadata& NMd = G.getNodeMetadata(NId);
        const MatrixMetadata& MMd = G.getEdgeCosts(EId).getMetadata();
        NMd.handleRemoveEdge(MMd, NId == G.getEdgeNode2Id(EId));
        if (G.getNodeDegree(NId) == 3) {
          // This node is becoming optimally reducible.
          moveToOptimallyReducibleNodes(NId);
        } else if (NMd.getReductionState() ==
                     NodeMetadata::NotProvablyAllocatable &&
                   NMd.isConservativelyAllocatable()) {
          // This node just became conservatively allocatable.
          moveToConservativelyAllocatableNodes(NId);
        }
      }

      void handleReconnectEdge(EdgeId EId, NodeId NId) {
        NodeMetadata& NMd = G.getNodeMetadata(NId);
        const MatrixMetadata& MMd = G.getEdgeCosts(EId).getMetadata();
        NMd.handleAddEdge(MMd, NId == G.getEdgeNode2Id(EId));
      }

      void handleSetEdgeCosts(EdgeId EId, const Matrix& NewCosts) {
        handleRemoveEdge(EId);

        NodeId N1Id = G.getEdgeNode1Id(EId);
        NodeId N2Id = G.getEdgeNode2Id(EId);
        NodeMetadata& N1Md = G.getNodeMetadata(N1Id);
        NodeMetadata& N2Md = G.getNodeMetadata(N2Id);
        const MatrixMetadata& MMd = NewCosts.getMetadata();
        N1Md.handleAddEdge(MMd, N1Id != G.getEdgeNode1Id(EId));
        N2Md.handleAddEdge(MMd, N2Id != G.getEdgeNode1Id(EId));
      }

    private:

      void removeFromCurrentSet(NodeId NId) {
        switch (G.getNodeMetadata(NId).getReductionState()) {
          case NodeMetadata::Unprocessed: break;
          case NodeMetadata::OptimallyReducible:
            assert(OptimallyReducibleNodes.find(NId) !=
                     OptimallyReducibleNodes.end() &&
                   "Node not in optimally reducible set.");
            OptimallyReducibleNodes.erase(NId);
            break;
          case NodeMetadata::ConservativelyAllocatable:
            assert(ConservativelyAllocatableNodes.find(NId) !=
                     ConservativelyAllocatableNodes.end() &&
                   "Node not in conservatively allocatable set.");
            ConservativelyAllocatableNodes.erase(NId);
            break;
          case NodeMetadata::NotProvablyAllocatable:
            assert(NotProvablyAllocatableNodes.find(NId) !=
                     NotProvablyAllocatableNodes.end() &&
                   "Node not in not-provably-allocatable set.");
            NotProvablyAllocatableNodes.erase(NId);
            break;
        }
      }

      void moveToOptimallyReducibleNodes(NodeId NId) {
        removeFromCurrentSet(NId);
        OptimallyReducibleNodes.insert(NId);
        G.getNodeMetadata(NId).setReductionState(
          NodeMetadata::OptimallyReducible);
      }

      void moveToConservativelyAllocatableNodes(NodeId NId) {
        removeFromCurrentSet(NId);
        ConservativelyAllocatableNodes.insert(NId);
        G.getNodeMetadata(NId).setReductionState(
          NodeMetadata::ConservativelyAllocatable);
      }

      void moveToNotProvablyAllocatableNodes(NodeId NId) {
        removeFromCurrentSet(NId);
        NotProvablyAllocatableNodes.insert(NId);
        G.getNodeMetadata(NId).setReductionState(
          NodeMetadata::NotProvablyAllocatable);
      }

      void setup() {
        // Set up worklists.
        for (auto NId : G.nodeIds()) {
          if (G.getNodeDegree(NId) < 3)
            moveToOptimallyReducibleNodes(NId);
          else if (G.getNodeMetadata(NId).isConservativelyAllocatable())
            moveToConservativelyAllocatableNodes(NId);
          else
            moveToNotProvablyAllocatableNodes(NId);
        }
      }

      // Compute a reduction order for the graph by iteratively applying PBQP
      // reduction rules. Locally optimal rules are applied whenever possible (R0,
      // R1, R2). If no locally-optimal rules apply then any conservatively
      // allocatable node is reduced. Finally, if no conservatively allocatable
      // node exists then the node with the lowest spill-cost:degree ratio is
      // selected.
      std::vector<GraphBase::NodeId> reduce() {
        assert(!G.empty() && "Cannot reduce empty graph.");

        typedef GraphBase::NodeId NodeId;
        std::vector<NodeId> NodeStack;

        // Consume worklists.
        while (true) {
          if (!OptimallyReducibleNodes.empty()) {
            NodeSet::iterator NItr = OptimallyReducibleNodes.begin();
            NodeId NId = *NItr;
            OptimallyReducibleNodes.erase(NItr);
            NodeStack.push_back(NId);
            switch (G.getNodeDegree(NId)) {
              case 0:
                break;
              case 1:
                applyR1(G, NId);
                break;
              case 2:
                applyR2(G, NId);
                break;
              default: llvm_unreachable("Not an optimally reducible node.");
            }
          } else if (!ConservativelyAllocatableNodes.empty()) {
            // Conservatively allocatable nodes will never spill. For now just
            // take the first node in the set and push it on the stack. When we
            // start optimizing more heavily for register preferencing, it may
            // would be better to push nodes with lower 'expected' or worst-case
            // register costs first (since early nodes are the most
            // constrained).
            NodeSet::iterator NItr = ConservativelyAllocatableNodes.begin();
            NodeId NId = *NItr;
            ConservativelyAllocatableNodes.erase(NItr);
            NodeStack.push_back(NId);
            G.disconnectAllNeighborsFromNode(NId);

          } else if (!NotProvablyAllocatableNodes.empty()) {
            NodeSet::iterator NItr =
              std::min_element(NotProvablyAllocatableNodes.begin(),
                               NotProvablyAllocatableNodes.end(),
                               SpillCostComparator(G));
            NodeId NId = *NItr;
            NotProvablyAllocatableNodes.erase(NItr);
            NodeStack.push_back(NId);
            G.disconnectAllNeighborsFromNode(NId);
          } else
            break;
        }

        return NodeStack;
      }

      class SpillCostComparator {
      public:
        SpillCostComparator(const Graph& G) : G(G) {}
        bool operator()(NodeId N1Id, NodeId N2Id) {
          PBQPNum N1SC = G.getNodeCosts(N1Id)[0] / G.getNodeDegree(N1Id);
          PBQPNum N2SC = G.getNodeCosts(N2Id)[0] / G.getNodeDegree(N2Id);
          return N1SC < N2SC;
        }
      private:
        const Graph& G;
      };

      Graph& G;
      typedef std::set<NodeId> NodeSet;
      NodeSet OptimallyReducibleNodes;
      NodeSet ConservativelyAllocatableNodes;
      NodeSet NotProvablyAllocatableNodes;
    };

    typedef Graph<RegAllocSolverImpl> Graph;

    Solution solve(Graph& G) {
      if (G.empty())
        return Solution();
      RegAllocSolverImpl RegAllocSolver(G);
      return RegAllocSolver.solve();
    }

  }
}

#endif // LLVM_CODEGEN_PBQP_REGALLOCSOLVER_H
