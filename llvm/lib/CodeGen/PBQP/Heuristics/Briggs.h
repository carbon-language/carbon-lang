//===-- Briggs.h --- Briggs Heuristic for PBQP -----------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the Briggs test for "allocability" of nodes in a
// PBQP graph representing a register allocation problem. Nodes which can be
// proven allocable (by a safe and relatively accurate test) are removed from
// the PBQP graph first. If no provably allocable node is present in the graph
// then the node with the minimal spill-cost to degree ratio is removed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_HEURISTICS_BRIGGS_H
#define LLVM_CODEGEN_PBQP_HEURISTICS_BRIGGS_H

#include "../HeuristicSolver.h"

#include <set>

namespace PBQP {
namespace Heuristics {

class Briggs {
  public:

    class NodeData;
    class EdgeData;

  private:

    typedef HeuristicSolverImpl<Briggs> Solver;
    typedef HSITypes<NodeData, EdgeData> HSIT;
    typedef HSIT::SolverGraph SolverGraph;
    typedef HSIT::GraphNodeIterator GraphNodeIterator;
    typedef HSIT::GraphEdgeIterator GraphEdgeIterator;

    class LinkDegreeComparator {
      public:
        LinkDegreeComparator() : g(0) {}
        LinkDegreeComparator(SolverGraph *g) : g(g) {}

        bool operator()(const GraphNodeIterator &node1Itr,
                        const GraphNodeIterator &node2Itr) const {
          assert((g != 0) && "Graph object not set, cannot access node data.");
          unsigned n1Degree = g->getNodeData(node1Itr).getLinkDegree(),
                   n2Degree = g->getNodeData(node2Itr).getLinkDegree();
          if (n1Degree > n2Degree) {
            return true;
          }
          else if (n1Degree < n2Degree) {
            return false;
          }
          // else they're "equal" by degree, differentiate based on ID.
          return g->getNodeID(node1Itr) < g->getNodeID(node2Itr);
        }

      private:
        SolverGraph *g;
    };

    class SpillPriorityComparator {
      public:
        SpillPriorityComparator() : g(0) {}
        SpillPriorityComparator(SolverGraph *g) : g(g) {}

        bool operator()(const GraphNodeIterator &node1Itr,
                        const GraphNodeIterator &node2Itr) const {
          assert((g != 0) && "Graph object not set, cannot access node data.");
          PBQPNum cost1 =
            g->getNodeCosts(node1Itr)[0] /
            g->getNodeData(node1Itr).getLinkDegree(),
            cost2 =
              g->getNodeCosts(node2Itr)[0] /
              g->getNodeData(node2Itr).getLinkDegree();

          if (cost1 < cost2) {
            return true;
          }
          else if (cost1 > cost2) {
            return false;
          }
          // else they'er "equal" again, differentiate based on address again.
          return g->getNodeID(node1Itr) < g->getNodeID(node2Itr);
        }

      private:
        SolverGraph *g;
    };

    typedef std::set<GraphNodeIterator, LinkDegreeComparator>
      RNAllocableNodeList;
    typedef RNAllocableNodeList::iterator RNAllocableNodeListIterator;

    typedef std::set<GraphNodeIterator, SpillPriorityComparator>
      RNUnallocableNodeList;
    typedef RNUnallocableNodeList::iterator RNUnallocableNodeListIterator;

  public:

    class NodeData {
      private:
        RNAllocableNodeListIterator rNAllocableNodeListItr;
        RNUnallocableNodeListIterator rNUnallocableNodeListItr;
        unsigned numRegOptions, numDenied, numSafe;
        std::vector<unsigned> unsafeDegrees;
        bool allocable;

        void addRemoveLink(SolverGraph &g, const GraphNodeIterator &nodeItr,
            const GraphEdgeIterator &edgeItr, bool add) {

          //assume we're adding...
          unsigned udTarget = 0, dir = 1;

          if (!add) {
            udTarget = 1;
            dir = -1;
          }

          EdgeData &linkEdgeData = g.getEdgeData(edgeItr).getHeuristicData();

          EdgeData::ConstUnsafeIterator edgeUnsafeBegin, edgeUnsafeEnd;

          if (nodeItr == g.getEdgeNode1Itr(edgeItr)) {
            numDenied += (dir * linkEdgeData.getWorstDegree());
            edgeUnsafeBegin = linkEdgeData.unsafeBegin();
            edgeUnsafeEnd = linkEdgeData.unsafeEnd();
          }
          else {
            numDenied += (dir * linkEdgeData.getReverseWorstDegree());
            edgeUnsafeBegin = linkEdgeData.reverseUnsafeBegin();
            edgeUnsafeEnd = linkEdgeData.reverseUnsafeEnd();
          }

          assert((unsafeDegrees.size() ==
                static_cast<unsigned>(
                  std::distance(edgeUnsafeBegin, edgeUnsafeEnd)))
              && "Unsafe array size mismatch.");

          std::vector<unsigned>::iterator unsafeDegreesItr =
            unsafeDegrees.begin();

          for (EdgeData::ConstUnsafeIterator edgeUnsafeItr = edgeUnsafeBegin;
              edgeUnsafeItr != edgeUnsafeEnd;
              ++edgeUnsafeItr, ++unsafeDegreesItr) {

            if ((*edgeUnsafeItr == 1) && (*unsafeDegreesItr == udTarget))  {
              numSafe -= dir;
            }
            *unsafeDegreesItr += (dir * (*edgeUnsafeItr));
          }

          allocable = (numDenied < numRegOptions) || (numSafe > 0);
        }

      public:

        void setup(SolverGraph &g, const GraphNodeIterator &nodeItr) {

          numRegOptions = g.getNodeCosts(nodeItr).getLength() - 1;

          numSafe = numRegOptions; // Optimistic, correct below.
          numDenied = 0; // Also optimistic.
          unsafeDegrees.resize(numRegOptions, 0);

          HSIT::NodeData &nodeData = g.getNodeData(nodeItr);

          for (HSIT::NodeData::AdjLinkIterator
              adjLinkItr = nodeData.adjLinksBegin(),
              adjLinkEnd = nodeData.adjLinksEnd();
              adjLinkItr != adjLinkEnd; ++adjLinkItr) {

            addRemoveLink(g, nodeItr, *adjLinkItr, true);
          }
        }

        bool isAllocable() const { return allocable; }

        void handleAddLink(SolverGraph &g, const GraphNodeIterator &nodeItr,
            const GraphEdgeIterator &adjEdge) {
          addRemoveLink(g, nodeItr, adjEdge, true);
        }

        void handleRemoveLink(SolverGraph &g, const GraphNodeIterator &nodeItr,
            const GraphEdgeIterator &adjEdge) {
          addRemoveLink(g, nodeItr, adjEdge, false);
        }

        void setRNAllocableNodeListItr(
            const RNAllocableNodeListIterator &rNAllocableNodeListItr) {

          this->rNAllocableNodeListItr = rNAllocableNodeListItr;
        }

        RNAllocableNodeListIterator getRNAllocableNodeListItr() const {
          return rNAllocableNodeListItr;
        }

        void setRNUnallocableNodeListItr(
            const RNUnallocableNodeListIterator &rNUnallocableNodeListItr) {

          this->rNUnallocableNodeListItr = rNUnallocableNodeListItr;
        }

        RNUnallocableNodeListIterator getRNUnallocableNodeListItr() const {
          return rNUnallocableNodeListItr;
        }


    };

    class EdgeData {
      private:

        typedef std::vector<unsigned> UnsafeArray;

        unsigned worstDegree,
                 reverseWorstDegree;
        UnsafeArray unsafe, reverseUnsafe;

      public:

        EdgeData() : worstDegree(0), reverseWorstDegree(0) {}

        typedef UnsafeArray::const_iterator ConstUnsafeIterator;

        void setup(SolverGraph &g, const GraphEdgeIterator &edgeItr) {
          const Matrix &edgeCosts = g.getEdgeCosts(edgeItr);
          unsigned numRegs = edgeCosts.getRows() - 1,
                   numReverseRegs = edgeCosts.getCols() - 1;

          unsafe.resize(numRegs, 0);
          reverseUnsafe.resize(numReverseRegs, 0);

          std::vector<unsigned> rowInfCounts(numRegs, 0),
                                colInfCounts(numReverseRegs, 0);

          for (unsigned i = 0; i < numRegs; ++i) {
            for (unsigned j = 0; j < numReverseRegs; ++j) {
              if (edgeCosts[i + 1][j + 1] ==
                  std::numeric_limits<PBQPNum>::infinity()) {
                unsafe[i] = 1;
                reverseUnsafe[j] = 1;
                ++rowInfCounts[i];
                ++colInfCounts[j];

                if (colInfCounts[j] > worstDegree) {
                  worstDegree = colInfCounts[j];
                }

                if (rowInfCounts[i] > reverseWorstDegree) {
                  reverseWorstDegree = rowInfCounts[i];
                }
              }
            }
          }
        }

        unsigned getWorstDegree() const { return worstDegree; }
        unsigned getReverseWorstDegree() const { return reverseWorstDegree; }
        ConstUnsafeIterator unsafeBegin() const { return unsafe.begin(); }
        ConstUnsafeIterator unsafeEnd() const { return unsafe.end(); }
        ConstUnsafeIterator reverseUnsafeBegin() const {
          return reverseUnsafe.begin();
        }
        ConstUnsafeIterator reverseUnsafeEnd() const {
          return reverseUnsafe.end();
        }
    };

  void initialise(Solver &solver) {
    this->s = &solver;
    g = &s->getGraph();
    rNAllocableBucket = RNAllocableNodeList(LinkDegreeComparator(g));
    rNUnallocableBucket =
      RNUnallocableNodeList(SpillPriorityComparator(g));
    
    for (GraphEdgeIterator
         edgeItr = g->edgesBegin(), edgeEnd = g->edgesEnd();
         edgeItr != edgeEnd; ++edgeItr) {

      g->getEdgeData(edgeItr).getHeuristicData().setup(*g, edgeItr);
    }

    for (GraphNodeIterator
         nodeItr = g->nodesBegin(), nodeEnd = g->nodesEnd();
         nodeItr != nodeEnd; ++nodeItr) {

      g->getNodeData(nodeItr).getHeuristicData().setup(*g, nodeItr);
    }
  }

  void addToRNBucket(const GraphNodeIterator &nodeItr) {
    NodeData &nodeData = g->getNodeData(nodeItr).getHeuristicData();

    if (nodeData.isAllocable()) {
      nodeData.setRNAllocableNodeListItr(
        rNAllocableBucket.insert(rNAllocableBucket.begin(), nodeItr));
    }
    else {
      nodeData.setRNUnallocableNodeListItr(
        rNUnallocableBucket.insert(rNUnallocableBucket.begin(), nodeItr));
    }
  }

  void removeFromRNBucket(const GraphNodeIterator &nodeItr) {
    NodeData &nodeData = g->getNodeData(nodeItr).getHeuristicData();

    if (nodeData.isAllocable()) {
      rNAllocableBucket.erase(nodeData.getRNAllocableNodeListItr());
    }
    else {
      rNUnallocableBucket.erase(nodeData.getRNUnallocableNodeListItr());
    }
  }

  void handleAddLink(const GraphEdgeIterator &edgeItr) {
    // We assume that if we got here this edge is attached to at least
    // one high degree node.
    g->getEdgeData(edgeItr).getHeuristicData().setup(*g, edgeItr);

    GraphNodeIterator n1Itr = g->getEdgeNode1Itr(edgeItr),
                      n2Itr = g->getEdgeNode2Itr(edgeItr);
   
    HSIT::NodeData &n1Data = g->getNodeData(n1Itr),
                   &n2Data = g->getNodeData(n2Itr);

    if (n1Data.getLinkDegree() > 2) {
      n1Data.getHeuristicData().handleAddLink(*g, n1Itr, edgeItr);
    }
    if (n2Data.getLinkDegree() > 2) {
      n2Data.getHeuristicData().handleAddLink(*g, n2Itr, edgeItr);
    }
  }

  void handleRemoveLink(const GraphEdgeIterator &edgeItr,
                        const GraphNodeIterator &nodeItr) {
    NodeData &nodeData = g->getNodeData(nodeItr).getHeuristicData();
    nodeData.handleRemoveLink(*g, nodeItr, edgeItr);
  }

  void processRN() {
    
    if (!rNAllocableBucket.empty()) {
      GraphNodeIterator selectedNodeItr = *rNAllocableBucket.begin();
      //std::cerr << "RN safely pushing " << g->getNodeID(selectedNodeItr) << "\n";
      rNAllocableBucket.erase(rNAllocableBucket.begin());
      s->pushStack(selectedNodeItr);
      s->unlinkNode(selectedNodeItr);
    }
    else {
      GraphNodeIterator selectedNodeItr = *rNUnallocableBucket.begin();
      //std::cerr << "RN optimistically pushing " << g->getNodeID(selectedNodeItr) << "\n";
      rNUnallocableBucket.erase(rNUnallocableBucket.begin());
      s->pushStack(selectedNodeItr);
      s->unlinkNode(selectedNodeItr);
    }
 
  }

  bool rNBucketEmpty() const {
    return (rNAllocableBucket.empty() && rNUnallocableBucket.empty());
  }

private:

  Solver *s;
  SolverGraph *g;
  RNAllocableNodeList rNAllocableBucket;
  RNUnallocableNodeList rNUnallocableBucket;
};



}
}


#endif // LLVM_CODEGEN_PBQP_HEURISTICS_BRIGGS_H
