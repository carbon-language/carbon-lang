//===-- Support/TarjanSCCIterator.h - Tarjan SCC iterator -------*- C++ -*-===//
//
// This builds on the Support/GraphTraits.h file to find the strongly 
// connected components (SCCs) of a graph in O(N+E) time using
// Tarjan's DFS algorithm.
//
// The SCC iterator has the important property that if a node in SCC S1
// has an edge to a node in SCC S2, then it visits S1 *after* S2.
// 
// To visit S1 *before* S2, use the TarjanSCCIterator on the Inverse graph.
// (NOTE: This requires some simple wrappers and is not supported yet.)
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TARJANSCCITERATOR_H
#define SUPPORT_TARJANSCCITERATOR_H

#include "Support/GraphTraits.h"
#include "Support/Statistic.h"
#include "Support/iterator"
#include <vector>
#include <stack>
#include <map>

//--------------------------------------------------------------------------
// class SCC : A simple representation of an SCC in a generic Graph.
//--------------------------------------------------------------------------

template<class GraphT, class GT = GraphTraits<GraphT> >
struct SCC: public std::vector<typename GT::NodeType*> {

  typedef typename GT::NodeType NodeType;
  typedef typename GT::ChildIteratorType ChildItTy;

  typedef std::vector<typename GT::NodeType*> super;
  typedef typename super::iterator               iterator;
  typedef typename super::const_iterator         const_iterator;
  typedef typename super::reverse_iterator       reverse_iterator;
  typedef typename super::const_reverse_iterator const_reverse_iterator;

  // HasLoop() -- Test if this SCC has a loop.  If it has more than one
  // node, this is trivially true.  If not, it may still contain a loop
  // if the node has an edge back to itself.
  bool HasLoop() const {
    if (size() > 1) return true;
    NodeType* N = front();
    for (ChildItTy CI=GT::child_begin(N), CE=GT::child_end(N); CI != CE; ++CI)
      if (*CI == N)
        return true;
    return false;
  }
};

//--------------------------------------------------------------------------
// class TarjanSCC_iterator: Enumerate the SCCs of a directed graph, in
// reverse topological order of the SCC DAG.
//--------------------------------------------------------------------------

namespace {
  Statistic<> NumSCCs("NumSCCs", "Number of Strongly Connected Components");
  Statistic<> MaxSCCSize("MaxSCCSize", "Size of largest Strongly Connected Component");
}

template<class GraphT, class GT = GraphTraits<GraphT> >
class TarjanSCC_iterator : public forward_iterator<SCC<GraphT, GT>, ptrdiff_t>
{
  typedef SCC<GraphT, GT> SccTy;
  typedef forward_iterator<SccTy, ptrdiff_t> super;
  typedef typename super::reference reference;
  typedef typename super::pointer pointer;
  typedef typename GT::NodeType          NodeType;
  typedef typename GT::ChildIteratorType ChildItTy;

  // The visit counters used to detect when a complete SCC is on the stack.
  // visitNum is the global counter.
  // nodeVisitNumbers are per-node visit numbers, also used as DFS flags.
  unsigned long visitNum;
  std::map<NodeType *, unsigned long> nodeVisitNumbers;

  // SCCNodeStack - Stack holding nodes of the SCC.
  std::stack<NodeType *> SCCNodeStack;

  // CurrentSCC - The current SCC, retrieved using operator*().
  SccTy CurrentSCC;

  // VisitStack - Used to maintain the ordering.  Top = current block
  // First element is basic block pointer, second is the 'next child' to visit
  std::stack<std::pair<NodeType *, ChildItTy> > VisitStack;

  // MinVistNumStack - Stack holding the "min" values for each node in the DFS.
  // This is used to track the minimum uplink values for all children of
  // the corresponding node on the VisitStack.
  std::stack<unsigned long> MinVisitNumStack;

  // A single "visit" within the non-recursive DFS traversal.
  void DFSVisitOne(NodeType* N) {
    ++visitNum;                         // Global counter for the visit order
    nodeVisitNumbers[N] = visitNum;
    SCCNodeStack.push(N);
    MinVisitNumStack.push(visitNum);
    VisitStack.push(make_pair(N, GT::child_begin(N)));
    DEBUG(std::cerr << "TarjanSCC: Node " << N <<
          " : visitNum = " << visitNum << "\n");
  }

  // The stack-based DFS traversal; defined below.
  void DFSVisitChildren() {
    assert(!VisitStack.empty());
    while (VisitStack.top().second != GT::child_end(VisitStack.top().first))
      { // TOS has at least one more child so continue DFS
        NodeType *childN = *VisitStack.top().second++;
        if (nodeVisitNumbers.find(childN) == nodeVisitNumbers.end())
          { // this node has never been seen
            DFSVisitOne(childN);
          }
        else
          {
            unsigned long childNum = nodeVisitNumbers[childN];
            if (MinVisitNumStack.top() > childNum)
              MinVisitNumStack.top() = childNum;
          }
      }
  }

  // Compute the next SCC using the DFS traversal.
  void GetNextSCC() {
    assert(VisitStack.size() == MinVisitNumStack.size());
    CurrentSCC.clear();                 // Prepare to compute the next SCC
    while (! VisitStack.empty())
      {
        DFSVisitChildren();

        assert(VisitStack.top().second==GT::child_end(VisitStack.top().first));
        NodeType* visitingN = VisitStack.top().first;
        unsigned long minVisitNum = MinVisitNumStack.top();
        VisitStack.pop();
        MinVisitNumStack.pop();
        if (! MinVisitNumStack.empty() && MinVisitNumStack.top() > minVisitNum)
          MinVisitNumStack.top() = minVisitNum;

        DEBUG(std::cerr << "TarjanSCC: Popped node " << visitingN <<
              " : minVisitNum = " << minVisitNum << "; Node visit num = " <<
              nodeVisitNumbers[visitingN] << "\n");

        if (minVisitNum == nodeVisitNumbers[visitingN])
          { // A full SCC is on the SCCNodeStack!  It includes all nodes below
            // visitingN on the stack.  Copy those nodes to CurrentSCC,
            // reset their minVisit values, and return (this suspends
            // the DFS traversal till the next ++).
            do {
              CurrentSCC.push_back(SCCNodeStack.top());
              SCCNodeStack.pop();
              nodeVisitNumbers[CurrentSCC.back()] = ~0UL; 
            } while (CurrentSCC.back() != visitingN);

            ++NumSCCs;
            if (CurrentSCC.size() > MaxSCCSize) MaxSCCSize = CurrentSCC.size();
            
            return;
          }
      }
  }

  inline TarjanSCC_iterator(NodeType *entryN) : visitNum(0) {
    DFSVisitOne(entryN);
    GetNextSCC();
  }
  inline TarjanSCC_iterator() { /* End is when DFS stack is empty */ }

public:
  typedef TarjanSCC_iterator<GraphT, GT> _Self;

  // Provide static "constructors"...
  static inline _Self begin(GraphT& G) { return _Self(GT::getEntryNode(G)); }
  static inline _Self end  (GraphT& G) { return _Self(); }

  // Direct loop termination test (I.fini() is more efficient than I == end())
  inline bool fini() const {
    assert(!CurrentSCC.empty() || VisitStack.empty());
    return CurrentSCC.empty();
  }

  inline bool operator==(const _Self& x) const { 
    return VisitStack == x.VisitStack && CurrentSCC == x.CurrentSCC;
  }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }

  // Iterator traversal: forward iteration only
  inline _Self& operator++() {          // Preincrement
    GetNextSCC();
    return *this; 
  }
  inline _Self operator++(int) {        // Postincrement
    _Self tmp = *this; ++*this; return tmp; 
  }

  // Retrieve a pointer to the current SCC.  Returns NULL when done.
  inline const SccTy* operator*() const { 
    assert(!CurrentSCC.empty() || VisitStack.empty());
    return CurrentSCC.empty()? NULL : &CurrentSCC;
  }
  inline SccTy* operator*() { 
    assert(!CurrentSCC.empty() || VisitStack.empty());
    return CurrentSCC.empty()? NULL : &CurrentSCC;
  }
};


// Global constructor for the Tarjan SCC iterator.  Use *I == NULL or I.fini()
// to test termination efficiently, instead of I == the "end" iterator.
template <class T>
TarjanSCC_iterator<T> tarj_begin(T G)
{
  return TarjanSCC_iterator<T>::begin(G);
}

template <class T>
TarjanSCC_iterator<T> tarj_end(T G)
{
  return TarjanSCC_iterator<T>::end(G);
}

//===----------------------------------------------------------------------===//

#endif
