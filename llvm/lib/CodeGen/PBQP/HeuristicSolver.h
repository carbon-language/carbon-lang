//===-- HeuristicSolver.h - Heuristic PBQP Solver ---------------*- C++ -*-===//
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
// used to to select a node for reduction. 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_HEURISTICSOLVER_H
#define LLVM_CODEGEN_PBQP_HEURISTICSOLVER_H

#include "Solver.h"
#include "AnnotatedGraph.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace PBQP {

/// \brief Important types for the HeuristicSolverImpl.
/// 
/// Declared seperately to allow access to heuristic classes before the solver
/// is fully constructed.
template <typename HeuristicNodeData, typename HeuristicEdgeData>
class HSITypes {
public:

  class NodeData;
  class EdgeData;

  typedef AnnotatedGraph<NodeData, EdgeData> SolverGraph;
  typedef typename SolverGraph::NodeIterator GraphNodeIterator;
  typedef typename SolverGraph::EdgeIterator GraphEdgeIterator;
  typedef typename SolverGraph::AdjEdgeIterator GraphAdjEdgeIterator;

  typedef std::list<GraphNodeIterator> NodeList;
  typedef typename NodeList::iterator NodeListIterator;

  typedef std::vector<GraphNodeIterator> NodeStack;
  typedef typename NodeStack::iterator NodeStackIterator;

  class NodeData {
    friend class EdgeData;

  private:

    typedef std::list<GraphEdgeIterator> LinksList;

    unsigned numLinks;
    LinksList links, solvedLinks;
    NodeListIterator bucketItr;
    HeuristicNodeData heuristicData;

  public:

    typedef typename LinksList::iterator AdjLinkIterator;

  private:

    AdjLinkIterator addLink(const GraphEdgeIterator &edgeItr) {
      ++numLinks;
      return links.insert(links.end(), edgeItr);
    }

    void delLink(const AdjLinkIterator &adjLinkItr) {
      --numLinks;
      links.erase(adjLinkItr);
    }

  public:

    NodeData() : numLinks(0) {}

    unsigned getLinkDegree() const { return numLinks; }

    HeuristicNodeData& getHeuristicData() { return heuristicData; }
    const HeuristicNodeData& getHeuristicData() const {
      return heuristicData;
    }

    void setBucketItr(const NodeListIterator &bucketItr) {
      this->bucketItr = bucketItr;
    }

    const NodeListIterator& getBucketItr() const {
      return bucketItr;
    }

    AdjLinkIterator adjLinksBegin() {
      return links.begin();
    }

    AdjLinkIterator adjLinksEnd() {
      return links.end();
    }

    void addSolvedLink(const GraphEdgeIterator &solvedLinkItr) {
      solvedLinks.push_back(solvedLinkItr);
    }

    AdjLinkIterator solvedLinksBegin() {
      return solvedLinks.begin();
    }

    AdjLinkIterator solvedLinksEnd() {
      return solvedLinks.end();
    }

  };

  class EdgeData {
  private:

    SolverGraph &g;
    GraphNodeIterator node1Itr, node2Itr;
    HeuristicEdgeData heuristicData;
    typename NodeData::AdjLinkIterator node1ThisEdgeItr, node2ThisEdgeItr;

  public:

    EdgeData(SolverGraph &g) : g(g) {}

    HeuristicEdgeData& getHeuristicData() { return heuristicData; }
    const HeuristicEdgeData& getHeuristicData() const {
      return heuristicData;
    }

    void setup(const GraphEdgeIterator &thisEdgeItr) {
      node1Itr = g.getEdgeNode1Itr(thisEdgeItr);
      node2Itr = g.getEdgeNode2Itr(thisEdgeItr);

      node1ThisEdgeItr = g.getNodeData(node1Itr).addLink(thisEdgeItr);
      node2ThisEdgeItr = g.getNodeData(node2Itr).addLink(thisEdgeItr);
    }

    void unlink() {
      g.getNodeData(node1Itr).delLink(node1ThisEdgeItr);
      g.getNodeData(node2Itr).delLink(node2ThisEdgeItr);
    }

  };

};

template <typename Heuristic>
class HeuristicSolverImpl {
public:
  // Typedefs to make life easier:
  typedef HSITypes<typename Heuristic::NodeData,
                   typename Heuristic::EdgeData> HSIT;
  typedef typename HSIT::SolverGraph SolverGraph;
  typedef typename HSIT::NodeData NodeData;
  typedef typename HSIT::EdgeData EdgeData;
  typedef typename HSIT::GraphNodeIterator GraphNodeIterator;
  typedef typename HSIT::GraphEdgeIterator GraphEdgeIterator;
  typedef typename HSIT::GraphAdjEdgeIterator GraphAdjEdgeIterator;

  typedef typename HSIT::NodeList NodeList;
  typedef typename HSIT::NodeListIterator NodeListIterator;

  typedef std::vector<GraphNodeIterator> NodeStack;
  typedef typename NodeStack::iterator NodeStackIterator;

  /// \brief Constructor, which performs all the actual solver work.
  HeuristicSolverImpl(const SimpleGraph &orig) :
    solution(orig.getNumNodes(), true)
  {
    copyGraph(orig);
    simplify();
    setup();
    computeSolution();
    computeSolutionCost(orig);
  }

  /// \brief Returns the graph for this solver.
  SolverGraph& getGraph() { return g; }

  /// \brief Return the solution found by this solver.
  const Solution& getSolution() const { return solution; }

private:

  /// \brief Add the given node to the appropriate bucket for its link
  /// degree.
  void addToBucket(const GraphNodeIterator &nodeItr) {
    NodeData &nodeData = g.getNodeData(nodeItr);

    switch (nodeData.getLinkDegree()) {
      case 0: nodeData.setBucketItr(
                r0Bucket.insert(r0Bucket.end(), nodeItr));
              break;                                            
      case 1: nodeData.setBucketItr(
                r1Bucket.insert(r1Bucket.end(), nodeItr));
              break;
      case 2: nodeData.setBucketItr(
                r2Bucket.insert(r2Bucket.end(), nodeItr));
              break;
      default: heuristic.addToRNBucket(nodeItr);
               break;
    }
  }

  /// \brief Remove the given node from the appropriate bucket for its link
  /// degree.
  void removeFromBucket(const GraphNodeIterator &nodeItr) {
    NodeData &nodeData = g.getNodeData(nodeItr);

    switch (nodeData.getLinkDegree()) {
      case 0: r0Bucket.erase(nodeData.getBucketItr()); break;
      case 1: r1Bucket.erase(nodeData.getBucketItr()); break;
      case 2: r2Bucket.erase(nodeData.getBucketItr()); break;
      default: heuristic.removeFromRNBucket(nodeItr); break;
    }
  }

public:

  /// \brief Add a link.
  void addLink(const GraphEdgeIterator &edgeItr) {
    g.getEdgeData(edgeItr).setup(edgeItr);

    if ((g.getNodeData(g.getEdgeNode1Itr(edgeItr)).getLinkDegree() > 2) ||
        (g.getNodeData(g.getEdgeNode2Itr(edgeItr)).getLinkDegree() > 2)) {
      heuristic.handleAddLink(edgeItr);
    }
  }

  /// \brief Remove link, update info for node.
  ///
  /// Only updates information for the given node, since usually the other
  /// is about to be removed.
  void removeLink(const GraphEdgeIterator &edgeItr,
                  const GraphNodeIterator &nodeItr) {

    if (g.getNodeData(nodeItr).getLinkDegree() > 2) {
      heuristic.handleRemoveLink(edgeItr, nodeItr);
    }
    g.getEdgeData(edgeItr).unlink();
  }

  /// \brief Remove link, update info for both nodes. Useful for R2 only.
  void removeLinkR2(const GraphEdgeIterator &edgeItr) {
    GraphNodeIterator node1Itr = g.getEdgeNode1Itr(edgeItr);

    if (g.getNodeData(node1Itr).getLinkDegree() > 2) {
      heuristic.handleRemoveLink(edgeItr, node1Itr);
    }
    removeLink(edgeItr, g.getEdgeNode2Itr(edgeItr));
  }

  /// \brief Removes all links connected to the given node.
  void unlinkNode(const GraphNodeIterator &nodeItr) {
    NodeData &nodeData = g.getNodeData(nodeItr);

    typedef std::vector<GraphEdgeIterator> TempEdgeList;

    TempEdgeList edgesToUnlink;
    edgesToUnlink.reserve(nodeData.getLinkDegree());

    // Copy adj edges into a temp vector. We want to destroy them during
    // the unlink, and we can't do that while we're iterating over them.
    std::copy(nodeData.adjLinksBegin(), nodeData.adjLinksEnd(),
              std::back_inserter(edgesToUnlink));

    for (typename TempEdgeList::iterator
         edgeItr = edgesToUnlink.begin(), edgeEnd = edgesToUnlink.end();
         edgeItr != edgeEnd; ++edgeItr) {

      GraphNodeIterator otherNode = g.getEdgeOtherNode(*edgeItr, nodeItr);

      removeFromBucket(otherNode);
      removeLink(*edgeItr, otherNode);
      addToBucket(otherNode);
    }
  }

  /// \brief Push the given node onto the stack to be solved with
  /// backpropagation.
  void pushStack(const GraphNodeIterator &nodeItr) {
    stack.push_back(nodeItr);
  }

  /// \brief Set the solution of the given node.
  void setSolution(const GraphNodeIterator &nodeItr, unsigned solIndex) {
    solution.setSelection(g.getNodeID(nodeItr), solIndex);

    for (GraphAdjEdgeIterator adjEdgeItr = g.adjEdgesBegin(nodeItr),
         adjEdgeEnd = g.adjEdgesEnd(nodeItr);
         adjEdgeItr != adjEdgeEnd; ++adjEdgeItr) {
      GraphEdgeIterator edgeItr(*adjEdgeItr);
      GraphNodeIterator adjNodeItr(g.getEdgeOtherNode(edgeItr, nodeItr));
      g.getNodeData(adjNodeItr).addSolvedLink(edgeItr);
    }
  }

private:

  SolverGraph g;
  Heuristic heuristic;
  Solution solution;

  NodeList r0Bucket,
           r1Bucket,
           r2Bucket;

  NodeStack stack;

  // Copy the SimpleGraph into an annotated graph which we can use for reduction.
  void copyGraph(const SimpleGraph &orig) {

    assert((g.getNumEdges() == 0) && (g.getNumNodes() == 0) &&
           "Graph should be empty prior to solver setup.");

    assert(orig.areNodeIDsValid() &&
           "Cannot copy from a graph with invalid node IDs.");

    std::vector<GraphNodeIterator> newNodeItrs;

    for (unsigned nodeID = 0; nodeID < orig.getNumNodes(); ++nodeID) {
      newNodeItrs.push_back(
        g.addNode(orig.getNodeCosts(orig.getNodeItr(nodeID)), NodeData()));
    }

    for (SimpleGraph::ConstEdgeIterator
         origEdgeItr = orig.edgesBegin(), origEdgeEnd = orig.edgesEnd();
         origEdgeItr != origEdgeEnd; ++origEdgeItr) {

      unsigned id1 = orig.getNodeID(orig.getEdgeNode1Itr(origEdgeItr)),
               id2 = orig.getNodeID(orig.getEdgeNode2Itr(origEdgeItr));

      g.addEdge(newNodeItrs[id1], newNodeItrs[id2],
                orig.getEdgeCosts(origEdgeItr), EdgeData(g));
    }

    // Assign IDs to the new nodes using the ordering from the old graph,
    // this will lead to nodes in the new graph getting the same ID as the
    // corresponding node in the old graph.
    g.assignNodeIDs(newNodeItrs);
  }

  // Simplify the annotated graph by eliminating independent edges and trivial
  // nodes. 
  void simplify() {
    disconnectTrivialNodes();
    eliminateIndependentEdges();
  }

  // Eliminate trivial nodes.
  void disconnectTrivialNodes() {
    for (GraphNodeIterator nodeItr = g.nodesBegin(), nodeEnd = g.nodesEnd();
         nodeItr != nodeEnd; ++nodeItr) {

      if (g.getNodeCosts(nodeItr).getLength() == 1) {

        std::vector<GraphEdgeIterator> edgesToRemove;

        for (GraphAdjEdgeIterator adjEdgeItr = g.adjEdgesBegin(nodeItr),
             adjEdgeEnd = g.adjEdgesEnd(nodeItr);
             adjEdgeItr != adjEdgeEnd; ++adjEdgeItr) {

          GraphEdgeIterator edgeItr = *adjEdgeItr;

          if (g.getEdgeNode1Itr(edgeItr) == nodeItr) {
            GraphNodeIterator otherNodeItr = g.getEdgeNode2Itr(edgeItr);
            g.getNodeCosts(otherNodeItr) +=
              g.getEdgeCosts(edgeItr).getRowAsVector(0);
          }
          else {
            GraphNodeIterator otherNodeItr = g.getEdgeNode1Itr(edgeItr);
            g.getNodeCosts(otherNodeItr) +=
              g.getEdgeCosts(edgeItr).getColAsVector(0);
          }

          edgesToRemove.push_back(edgeItr);
        }

        while (!edgesToRemove.empty()) {
          g.removeEdge(edgesToRemove.back());
          edgesToRemove.pop_back();
        }
      }
    }
  }

  void eliminateIndependentEdges() {
    std::vector<GraphEdgeIterator> edgesToProcess;

    for (GraphEdgeIterator edgeItr = g.edgesBegin(), edgeEnd = g.edgesEnd();
         edgeItr != edgeEnd; ++edgeItr) {
      edgesToProcess.push_back(edgeItr);
    }

    while (!edgesToProcess.empty()) {
      tryToEliminateEdge(edgesToProcess.back());
      edgesToProcess.pop_back();
    }
  }

  void tryToEliminateEdge(const GraphEdgeIterator &edgeItr) {
    if (tryNormaliseEdgeMatrix(edgeItr)) {
      g.removeEdge(edgeItr); 
    }
  }

  bool tryNormaliseEdgeMatrix(const GraphEdgeIterator &edgeItr) {

    Matrix &edgeCosts = g.getEdgeCosts(edgeItr);
    Vector &uCosts = g.getNodeCosts(g.getEdgeNode1Itr(edgeItr)),
               &vCosts = g.getNodeCosts(g.getEdgeNode2Itr(edgeItr));

    for (unsigned r = 0; r < edgeCosts.getRows(); ++r) {
      PBQPNum rowMin = edgeCosts.getRowMin(r);
      uCosts[r] += rowMin;
      if (rowMin != std::numeric_limits<PBQPNum>::infinity()) {
        edgeCosts.subFromRow(r, rowMin);
      }
      else {
        edgeCosts.setRow(r, 0);
      }
    }

    for (unsigned c = 0; c < edgeCosts.getCols(); ++c) {
      PBQPNum colMin = edgeCosts.getColMin(c);
      vCosts[c] += colMin;
      if (colMin != std::numeric_limits<PBQPNum>::infinity()) {
        edgeCosts.subFromCol(c, colMin);
      }
      else {
        edgeCosts.setCol(c, 0);
      }
    }

    return edgeCosts.isZero();
  }

  void setup() {
    setupLinks();
    heuristic.initialise(*this);
    setupBuckets();
  }

  void setupLinks() {
    for (GraphEdgeIterator edgeItr = g.edgesBegin(), edgeEnd = g.edgesEnd();
         edgeItr != edgeEnd; ++edgeItr) {
      g.getEdgeData(edgeItr).setup(edgeItr);
    }
  }

  void setupBuckets() {
    for (GraphNodeIterator nodeItr = g.nodesBegin(), nodeEnd = g.nodesEnd();
         nodeItr != nodeEnd; ++nodeItr) {
      addToBucket(nodeItr);
    }
  }

  void computeSolution() {
    assert(g.areNodeIDsValid() &&
           "Nodes cannot be added/removed during reduction.");

    reduce();
    computeTrivialSolutions();
    backpropagate();
  }

  void printNode(const GraphNodeIterator &nodeItr) {
    llvm::errs() << "Node " << g.getNodeID(nodeItr) << " (" << &*nodeItr << "):\n"
                 << "  costs = " << g.getNodeCosts(nodeItr) << "\n"
                 << "  link degree = " << g.getNodeData(nodeItr).getLinkDegree() << "\n"
                 << "  links = [ ";

    for (typename HSIT::NodeData::AdjLinkIterator 
         aeItr = g.getNodeData(nodeItr).adjLinksBegin(),
         aeEnd = g.getNodeData(nodeItr).adjLinksEnd();
         aeItr != aeEnd; ++aeItr) {
      llvm::errs() << "(" << g.getNodeID(g.getEdgeNode1Itr(*aeItr))
                   << ", " << g.getNodeID(g.getEdgeNode2Itr(*aeItr))
                   << ") ";
    }
    llvm::errs() << "]\n";
  }

  void dumpState() {
    llvm::errs() << "\n";

    for (GraphNodeIterator nodeItr = g.nodesBegin(), nodeEnd = g.nodesEnd();
         nodeItr != nodeEnd; ++nodeItr) {
      printNode(nodeItr);
    }

    NodeList* buckets[] = { &r0Bucket, &r1Bucket, &r2Bucket };

    for (unsigned b = 0; b < 3; ++b) {
      NodeList &bucket = *buckets[b];

      llvm::errs() << "Bucket " << b << ": [ ";

      for (NodeListIterator nItr = bucket.begin(), nEnd = bucket.end();
           nItr != nEnd; ++nItr) {
        llvm::errs() << g.getNodeID(*nItr) << " ";
      }

      llvm::errs() << "]\n";
    }

    llvm::errs() << "Stack: [ ";
    for (NodeStackIterator nsItr = stack.begin(), nsEnd = stack.end();
         nsItr != nsEnd; ++nsItr) {
      llvm::errs() << g.getNodeID(*nsItr) << " ";
    }
    llvm::errs() << "]\n";
  }

  void reduce() {
    bool reductionFinished = r1Bucket.empty() && r2Bucket.empty() &&
      heuristic.rNBucketEmpty();

    while (!reductionFinished) {

      if (!r1Bucket.empty()) {
        processR1();
      }
      else if (!r2Bucket.empty()) {
        processR2();
      }
      else if (!heuristic.rNBucketEmpty()) {
        solution.setProvedOptimal(false);
        solution.incRNReductions();
        heuristic.processRN();
      } 
      else reductionFinished = true;
    }
      
  }

  void processR1() {

    // Remove the first node in the R0 bucket:
    GraphNodeIterator xNodeItr = r1Bucket.front();
    r1Bucket.pop_front();

    solution.incR1Reductions();

    //llvm::errs() << "Applying R1 to " << g.getNodeID(xNodeItr) << "\n";

    assert((g.getNodeData(xNodeItr).getLinkDegree() == 1) &&
           "Node in R1 bucket has degree != 1");

    GraphEdgeIterator edgeItr = *g.getNodeData(xNodeItr).adjLinksBegin();

    const Matrix &edgeCosts = g.getEdgeCosts(edgeItr);

    const Vector &xCosts = g.getNodeCosts(xNodeItr);
    unsigned xLen = xCosts.getLength();

    // Duplicate a little code to avoid transposing matrices:
    if (xNodeItr == g.getEdgeNode1Itr(edgeItr)) {
      GraphNodeIterator yNodeItr = g.getEdgeNode2Itr(edgeItr);
      Vector &yCosts = g.getNodeCosts(yNodeItr);
      unsigned yLen = yCosts.getLength();

      for (unsigned j = 0; j < yLen; ++j) {
        PBQPNum min = edgeCosts[0][j] + xCosts[0];
        for (unsigned i = 1; i < xLen; ++i) {
          PBQPNum c = edgeCosts[i][j] + xCosts[i];
          if (c < min)
            min = c;
        }
        yCosts[j] += min;
      }
    }
    else {
      GraphNodeIterator yNodeItr = g.getEdgeNode1Itr(edgeItr);
      Vector &yCosts = g.getNodeCosts(yNodeItr);
      unsigned yLen = yCosts.getLength();

      for (unsigned i = 0; i < yLen; ++i) {
        PBQPNum min = edgeCosts[i][0] + xCosts[0];

        for (unsigned j = 1; j < xLen; ++j) {
          PBQPNum c = edgeCosts[i][j] + xCosts[j];
          if (c < min)
            min = c;
        }
        yCosts[i] += min;
      }
    }

    unlinkNode(xNodeItr);
    pushStack(xNodeItr);
  }

  void processR2() {

    GraphNodeIterator xNodeItr = r2Bucket.front();
    r2Bucket.pop_front();

    solution.incR2Reductions();

    // Unlink is unsafe here. At some point it may optimistically more a node
    // to a lower-degree list when its degree will later rise, or vice versa,
    // violating the assumption that node degrees monotonically decrease
    // during the reduction phase. Instead we'll bucket shuffle manually.
    pushStack(xNodeItr);

    assert((g.getNodeData(xNodeItr).getLinkDegree() == 2) &&
           "Node in R2 bucket has degree != 2");

    const Vector &xCosts = g.getNodeCosts(xNodeItr);

    typename NodeData::AdjLinkIterator tempItr =
      g.getNodeData(xNodeItr).adjLinksBegin();

    GraphEdgeIterator yxEdgeItr = *tempItr,
                      zxEdgeItr = *(++tempItr);

    GraphNodeIterator yNodeItr = g.getEdgeOtherNode(yxEdgeItr, xNodeItr),
                      zNodeItr = g.getEdgeOtherNode(zxEdgeItr, xNodeItr);

    removeFromBucket(yNodeItr);
    removeFromBucket(zNodeItr);

    removeLink(yxEdgeItr, yNodeItr);
    removeLink(zxEdgeItr, zNodeItr);

    // Graph some of the costs:
    bool flipEdge1 = (g.getEdgeNode1Itr(yxEdgeItr) == xNodeItr),
         flipEdge2 = (g.getEdgeNode1Itr(zxEdgeItr) == xNodeItr);

    const Matrix *yxCosts = flipEdge1 ?
      new Matrix(g.getEdgeCosts(yxEdgeItr).transpose()) :
      &g.getEdgeCosts(yxEdgeItr),
                     *zxCosts = flipEdge2 ?
      new Matrix(g.getEdgeCosts(zxEdgeItr).transpose()) :
        &g.getEdgeCosts(zxEdgeItr);

    unsigned xLen = xCosts.getLength(),
             yLen = yxCosts->getRows(),
             zLen = zxCosts->getRows();

    // Compute delta:
    Matrix delta(yLen, zLen);

    for (unsigned i = 0; i < yLen; ++i) {
      for (unsigned j = 0; j < zLen; ++j) {
        PBQPNum min = (*yxCosts)[i][0] + (*zxCosts)[j][0] + xCosts[0];
        for (unsigned k = 1; k < xLen; ++k) {
          PBQPNum c = (*yxCosts)[i][k] + (*zxCosts)[j][k] + xCosts[k];
          if (c < min) {
            min = c;
          }
        }
        delta[i][j] = min;
      }
    }

    if (flipEdge1)
      delete yxCosts;

    if (flipEdge2)
      delete zxCosts;

    // Deal with the potentially induced yz edge.
    GraphEdgeIterator yzEdgeItr = g.findEdge(yNodeItr, zNodeItr);
    if (yzEdgeItr == g.edgesEnd()) {
      yzEdgeItr = g.addEdge(yNodeItr, zNodeItr, delta, EdgeData(g));
    }
    else {
      // There was an edge, but we're going to screw with it. Delete the old
      // link, update the costs. We'll re-link it later.
      removeLinkR2(yzEdgeItr);
      g.getEdgeCosts(yzEdgeItr) +=
        (yNodeItr == g.getEdgeNode1Itr(yzEdgeItr)) ?
        delta : delta.transpose();
    }

    bool nullCostEdge = tryNormaliseEdgeMatrix(yzEdgeItr);

    // Nulled the edge, remove it entirely.
    if (nullCostEdge) {
      g.removeEdge(yzEdgeItr);
    }
    else {
      // Edge remains - re-link it.
      addLink(yzEdgeItr);
    }

    addToBucket(yNodeItr);
    addToBucket(zNodeItr);
    }

  void computeTrivialSolutions() {

    for (NodeListIterator r0Itr = r0Bucket.begin(), r0End = r0Bucket.end();
         r0Itr != r0End; ++r0Itr) {
      GraphNodeIterator nodeItr = *r0Itr;

      solution.incR0Reductions();
      setSolution(nodeItr, g.getNodeCosts(nodeItr).minIndex());
    }

  }

  void backpropagate() {
    while (!stack.empty()) {
      computeSolution(stack.back());
      stack.pop_back();
    }
  }

  void computeSolution(const GraphNodeIterator &nodeItr) {

    NodeData &nodeData = g.getNodeData(nodeItr);

    Vector v(g.getNodeCosts(nodeItr));

    // Solve based on existing links.
    for (typename NodeData::AdjLinkIterator
         solvedLinkItr = nodeData.solvedLinksBegin(),
         solvedLinkEnd = nodeData.solvedLinksEnd();
         solvedLinkItr != solvedLinkEnd; ++solvedLinkItr) {

      GraphEdgeIterator solvedEdgeItr(*solvedLinkItr);
      Matrix &edgeCosts = g.getEdgeCosts(solvedEdgeItr);

      if (nodeItr == g.getEdgeNode1Itr(solvedEdgeItr)) {
        GraphNodeIterator adjNode(g.getEdgeNode2Itr(solvedEdgeItr));
        unsigned adjSolution =
          solution.getSelection(g.getNodeID(adjNode));
        v += edgeCosts.getColAsVector(adjSolution);
      }
      else {
        GraphNodeIterator adjNode(g.getEdgeNode1Itr(solvedEdgeItr));
        unsigned adjSolution =
          solution.getSelection(g.getNodeID(adjNode));
        v += edgeCosts.getRowAsVector(adjSolution);
      }

    }

    setSolution(nodeItr, v.minIndex());
  }

  void computeSolutionCost(const SimpleGraph &orig) {
    PBQPNum cost = 0.0;

    for (SimpleGraph::ConstNodeIterator
         nodeItr = orig.nodesBegin(), nodeEnd = orig.nodesEnd();
         nodeItr != nodeEnd; ++nodeItr) {

      unsigned nodeId = orig.getNodeID(nodeItr);

      cost += orig.getNodeCosts(nodeItr)[solution.getSelection(nodeId)];
    }

    for (SimpleGraph::ConstEdgeIterator
         edgeItr = orig.edgesBegin(), edgeEnd = orig.edgesEnd();
         edgeItr != edgeEnd; ++edgeItr) {

      SimpleGraph::ConstNodeIterator n1 = orig.getEdgeNode1Itr(edgeItr),
                                     n2 = orig.getEdgeNode2Itr(edgeItr);
      unsigned sol1 = solution.getSelection(orig.getNodeID(n1)),
               sol2 = solution.getSelection(orig.getNodeID(n2));

      cost += orig.getEdgeCosts(edgeItr)[sol1][sol2];
    }

    solution.setSolutionCost(cost);
  }

};

template <typename Heuristic>
class HeuristicSolver : public Solver {
public:
  Solution solve(const SimpleGraph &g) const {
    HeuristicSolverImpl<Heuristic> solverImpl(g);
    return solverImpl.getSolution();
  }
};

}

#endif // LLVM_CODEGEN_PBQP_HEURISTICSOLVER_H
