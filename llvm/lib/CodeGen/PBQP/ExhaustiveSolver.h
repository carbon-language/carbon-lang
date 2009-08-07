//===-- ExhaustiveSolver.h - Brute Force PBQP Solver -----------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Uses a trivial brute force algorithm to solve a PBQP problem.
// PBQP is NP-HARD - This solver should only be used for debugging small
// problems.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_EXHAUSTIVESOLVER_H
#define LLVM_CODEGEN_PBQP_EXHAUSTIVESOLVER_H

#include "Solver.h"

namespace PBQP {

/// A brute force PBQP solver. This solver takes exponential time. It should
/// only be used for debugging purposes. 
class ExhaustiveSolverImpl {
private:

  const SimpleGraph &g;

  PBQPNum getSolutionCost(const Solution &solution) const {
    PBQPNum cost = 0.0;
    
    for (SimpleGraph::ConstNodeIterator
         nodeItr = g.nodesBegin(), nodeEnd = g.nodesEnd();
         nodeItr != nodeEnd; ++nodeItr) {
      
      unsigned nodeId = g.getNodeID(nodeItr);

      cost += g.getNodeCosts(nodeItr)[solution.getSelection(nodeId)];
    }

    for (SimpleGraph::ConstEdgeIterator
         edgeItr = g.edgesBegin(), edgeEnd = g.edgesEnd();
         edgeItr != edgeEnd; ++edgeItr) {
      
      SimpleGraph::ConstNodeIterator n1 = g.getEdgeNode1Itr(edgeItr),
                                     n2 = g.getEdgeNode2Itr(edgeItr);
      unsigned sol1 = solution.getSelection(g.getNodeID(n1)),
               sol2 = solution.getSelection(g.getNodeID(n2));

      cost += g.getEdgeCosts(edgeItr)[sol1][sol2];
    }

    return cost;
  }

public:

  ExhaustiveSolverImpl(const SimpleGraph &g) : g(g) {}

  Solution solve() const {
    Solution current(g.getNumNodes(), true), optimal(current);

    PBQPNum bestCost = std::numeric_limits<PBQPNum>::infinity();
    bool finished = false;

    while (!finished) {
      PBQPNum currentCost = getSolutionCost(current);

      if (currentCost < bestCost) {
        optimal = current;
        bestCost = currentCost;
      }

      // assume we're done.
      finished = true;

      for (unsigned i = 0; i < g.getNumNodes(); ++i) {
        if (current.getSelection(i) ==
            (g.getNodeCosts(g.getNodeItr(i)).getLength() - 1)) {
          current.setSelection(i, 0);
        }
        else {
          current.setSelection(i, current.getSelection(i) + 1);
          finished = false;
          break;
        }
      }

    }

    optimal.setSolutionCost(bestCost);

    return optimal;
  }

};

class ExhaustiveSolver : public Solver {
public:
  ~ExhaustiveSolver() {}
  Solution solve(const SimpleGraph &g) const {
    ExhaustiveSolverImpl solver(g);
    return solver.solve();
  }
};

}

#endif // LLVM_CODGEN_PBQP_EXHAUSTIVESOLVER_HPP
