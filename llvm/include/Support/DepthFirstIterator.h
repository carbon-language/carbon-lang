//===- Support/DepthFirstIterator.h - Depth First iterator -------*- C++ -*--=//
//
// This file builds on the Support/GraphTraits.h file to build generic depth
// first graph iterator.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEPTHFIRSTITERATOR_H
#define SUPPORT_DEPTHFIRSTITERATOR_H

#include "Support/GraphTraits.h"
#include "Support/iterator"
#include <stack>
#include <set>

// Generic Depth First Iterator
template<class GraphT, class GT = GraphTraits<GraphT> >
class df_iterator : public forward_iterator<typename GT::NodeType, ptrdiff_t> {
  typedef forward_iterator<typename GT::NodeType, ptrdiff_t> super;

  typedef typename GT::NodeType          NodeType;
  typedef typename GT::ChildIteratorType ChildItTy;

  std::set<NodeType *> Visited;    // All of the blocks visited so far...
  // VisitStack - Used to maintain the ordering.  Top = current block
  // First element is node pointer, second is the 'next child' to visit
  std::stack<std::pair<NodeType *, ChildItTy> > VisitStack;
  const bool Reverse;         // Iterate over children before self?
private:
  void reverseEnterNode() {
    std::pair<NodeType *, ChildItTy> &Top = VisitStack.top();
    NodeType *Node = Top.first;
    ChildItTy &It  = Top.second;
    for (; It != GT::child_end(Node); ++It) {
      NodeType *Child = *It;
      if (!Visited.count(Child)) {
	Visited.insert(Child);
	VisitStack.push(std::make_pair(Child, GT::child_begin(Child)));
	reverseEnterNode();
	return;
      }
    }
  }

  inline df_iterator(NodeType *Node, bool reverse) : Reverse(reverse) {
    Visited.insert(Node);
    VisitStack.push(std::make_pair(Node, GT::child_begin(Node)));
    if (Reverse) reverseEnterNode();
  }
  inline df_iterator() { /* End is when stack is empty */ }

public:
  typedef typename super::pointer pointer;
  typedef df_iterator<GraphT, GT> _Self;

  // Provide static begin and end methods as our public "constructors"
  static inline _Self begin(GraphT G, bool Reverse = false) {
    return _Self(GT::getEntryNode(G), Reverse);
  }
  static inline _Self end(GraphT G) { return _Self(); }


  inline bool operator==(const _Self& x) const { 
    return VisitStack == x.VisitStack;
  }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }

  inline pointer operator*() const { 
    return VisitStack.top().first;
  }

  // This is a nonstandard operator-> that dereferences the pointer an extra
  // time... so that you can actually call methods ON the Node, because
  // the contained type is a pointer.  This allows BBIt->getTerminator() f.e.
  //
  inline NodeType *operator->() const { return operator*(); }

  inline _Self& operator++() {   // Preincrement
    if (Reverse) {               // Reverse Depth First Iterator
      if (VisitStack.top().second == GT::child_end(VisitStack.top().first))
	VisitStack.pop();
      if (!VisitStack.empty())
	reverseEnterNode();
    } else {                     // Normal Depth First Iterator
      do {
	std::pair<NodeType *, ChildItTy> &Top = VisitStack.top();
	NodeType *Node = Top.first;
	ChildItTy &It  = Top.second;

	while (It != GT::child_end(Node)) {
	  NodeType *Next = *It++;
	  if (!Visited.count(Next)) {  // Has our next sibling been visited?
	    // No, do it now.
	    Visited.insert(Next);
	    VisitStack.push(std::make_pair(Next, GT::child_begin(Next)));
	    return *this;
	  }
	}
	
	// Oops, ran out of successors... go up a level on the stack.
	VisitStack.pop();
      } while (!VisitStack.empty());
    }
    return *this; 
  }

  inline _Self operator++(int) { // Postincrement
    _Self tmp = *this; ++*this; return tmp; 
  }

  // nodeVisited - return true if this iterator has already visited the
  // specified node.  This is public, and will probably be used to iterate over
  // nodes that a depth first iteration did not find: ie unreachable nodes.
  //
  inline bool nodeVisited(NodeType *Node) const { 
    return Visited.count(Node) != 0;
  }
};


// Provide global constructors that automatically figure out correct types...
//
template <class T>
df_iterator<T> df_begin(T G, bool Reverse = false) {
  return df_iterator<T>::begin(G, Reverse);
}

template <class T>
df_iterator<T> df_end(T G) {
  return df_iterator<T>::end(G);
}

// Provide global definitions of inverse depth first iterators...
template <class T>
struct idf_iterator : public df_iterator<Inverse<T> > {
  idf_iterator(const df_iterator<Inverse<T> > &V) :df_iterator<Inverse<T> >(V){}
};

template <class T>
idf_iterator<T> idf_begin(T G, bool Reverse = false) {
  return idf_iterator<T>::begin(G, Reverse);
}

template <class T>
idf_iterator<T> idf_end(T G){
  return idf_iterator<T>::end(G);
}

#endif
