#ifndef LLVM_CODEGEN_PBQP_EXHAUSTIVESOLVER_H
#define LLVM_CODEGEN_PBQP_EXHAUSTIVESOLVER_H

#include "Solver.h"

namespace PBQP {

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
