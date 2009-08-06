#ifndef LLVM_CODEGEN_PBQP_GRAPHGENERATOR_H
#define LLVM_CODEGEN_PBQP_GRAPHGENERATOR_H

#include "PBQPMath.h"

namespace PBQP {

unsigned randRange(unsigned min, unsigned max) {
  return min + (rand() % (max - min + 1));
}

class BasicNodeCostsGenerator {
private:

  unsigned maxDegree, minCost, maxCost;


public:

  BasicNodeCostsGenerator(unsigned maxDegree, unsigned minCost,
                          unsigned maxCost) :
    maxDegree(maxDegree), minCost(minCost), maxCost(maxCost) { }

  Vector operator()() const {
    Vector v(randRange(1, maxDegree));
    for (unsigned i = 0; i < v.getLength(); ++i) {
      v[i] = randRange(minCost, maxCost);
    }
    return v;
  };

};

class FixedDegreeSpillCostGenerator {
private:

  unsigned degree, spillCostMin, spillCostMax;

public:

  FixedDegreeSpillCostGenerator(unsigned degree, unsigned spillCostMin,
                                unsigned spillCostMax) :
    degree(degree), spillCostMin(spillCostMin), spillCostMax(spillCostMax) { }

  Vector operator()() const {
    Vector v(degree, 0);
    v[0] = randRange(spillCostMin, spillCostMax);
    return v;
  }

};

class BasicEdgeCostsGenerator {
private:

  unsigned minCost, maxCost;

public:

  BasicEdgeCostsGenerator(unsigned minCost, unsigned maxCost) :
    minCost(minCost), maxCost(maxCost) {}

  Matrix operator()(const SimpleGraph &g,
                        const SimpleGraph::ConstNodeIterator &n1,
                        const SimpleGraph::ConstNodeIterator &n2) const {

    Matrix m(g.getNodeCosts(n1).getLength(),
                 g.getNodeCosts(n2).getLength());

    for (unsigned i = 0; i < m.getRows(); ++i) {
      for (unsigned j = 0; j < m.getCols(); ++j) {
        m[i][j] = randRange(minCost, maxCost);
      }
    }

    return m;
  }

};

class InterferenceCostsGenerator {
public:

  Matrix operator()(const SimpleGraph &g,
                        const SimpleGraph::ConstNodeIterator &n1,
                        const SimpleGraph::ConstNodeIterator &n2) const {

    unsigned len = g.getNodeCosts(n1).getLength();

    assert(len == g.getNodeCosts(n2).getLength());

    Matrix m(len, len);

    m[0][0] = 0;
    for (unsigned i = 1; i < len; ++i) {
      m[i][i] = std::numeric_limits<PBQPNum>::infinity();
    }

    return m;
  }
};

class RingEdgeGenerator {
public:

  template <typename EdgeCostsGenerator>
  void operator()(SimpleGraph &g, EdgeCostsGenerator &edgeCostsGen) {

    assert(g.areNodeIDsValid() && "Graph must have valid node IDs.");

    if (g.getNumNodes() < 2)
      return;

    if (g.getNumNodes() == 2) {
      SimpleGraph::NodeIterator n1 = g.getNodeItr(0),
                                n2 = g.getNodeItr(1);
      g.addEdge(n1, n2, edgeCostsGen(g, n1, n2));
      return;
    }

    // Else |V| > 2:
    for (unsigned i = 0; i < g.getNumNodes(); ++i) {
      SimpleGraph::NodeIterator
        n1 = g.getNodeItr(i),
        n2 = g.getNodeItr((i + 1) % g.getNumNodes());
      g.addEdge(n1, n2, edgeCostsGen(g, n1, n2));
    }
  }

};

class FullyConnectedEdgeGenerator {
public:
    
  template <typename EdgeCostsGenerator>
  void operator()(SimpleGraph &g, EdgeCostsGenerator &edgeCostsGen) {
    assert(g.areNodeIDsValid() && "Graph must have valid node IDs.");
    
    for (unsigned i = 0; i < g.getNumNodes(); ++i) {
      for (unsigned j = i + 1; j < g.getNumNodes(); ++j) {
        SimpleGraph::NodeIterator
          n1 = g.getNodeItr(i),
          n2 = g.getNodeItr(j);
        g.addEdge(n1, n2, edgeCostsGen(g, n1, n2));
      }
    }
  }

};

class RandomEdgeGenerator {
public:

  template <typename EdgeCostsGenerator>
  void operator()(SimpleGraph &g, EdgeCostsGenerator &edgeCostsGen) {
    
    assert(g.areNodeIDsValid() && "Graph must have valid node IDs.");
    
    for (unsigned i = 0; i < g.getNumNodes(); ++i) {
      for (unsigned j = i + 1; j < g.getNumNodes(); ++j) {
        if (rand() % 2 == 0) {
          SimpleGraph::NodeIterator
            n1 = g.getNodeItr(i),
            n2 = g.getNodeItr(j);
          g.addEdge(n1, n2, edgeCostsGen(g, n1, n2));
        }
      }
    }
  }

};

template <typename NodeCostsGenerator,
          typename EdgesGenerator,
          typename EdgeCostsGenerator>
SimpleGraph createRandomGraph(unsigned numNodes,
                              NodeCostsGenerator nodeCostsGen,
                              EdgesGenerator edgeGen,
                              EdgeCostsGenerator edgeCostsGen) {

  SimpleGraph g;
  for (unsigned n = 0; n < numNodes; ++n) {
    g.addNode(nodeCostsGen());
  }

  g.assignNodeIDs();

  edgeGen(g, edgeCostsGen);

  return g;
}

}

#endif // LLVM_CODEGEN_PBQP_GRAPHGENERATOR_H
