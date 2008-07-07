//==- llvm/ADT/alist.h - Linked lists with hooks -----------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the alist class template, and related infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ALIST_H
#define LLVM_ADT_ALIST_H

#include <cassert>
#include "llvm/ADT/alist_node.h"
#include "llvm/ADT/STLExtras.h"

namespace llvm {

/// alist_iterator - An iterator class for alist.
///
template<class T, class LargestT = T, class ValueT = T,
         class NodeIterT = ilist_iterator<alist_node<T, LargestT> > >
class alist_iterator : public bidirectional_iterator<ValueT, ptrdiff_t> {
  typedef bidirectional_iterator<ValueT, ptrdiff_t> super;
  typedef alist_node<T, LargestT> NodeTy;

  /// NodeIter - The underlying iplist iterator that is being wrapped.
  NodeIterT NodeIter;

public:
  typedef size_t size_type;
  typedef typename super::pointer pointer;
  typedef typename super::reference reference;

  alist_iterator(NodeIterT NI) : NodeIter(NI) {}
  alist_iterator(pointer EP) : NodeIter(NodeTy::getNode(EP)) {}
  alist_iterator() : NodeIter() {}

  // This is templated so that we can allow constructing a const iterator from
  // a nonconst iterator...
  template<class V, class W, class X, class Y>
  alist_iterator(const alist_iterator<V, W, X, Y> &RHS)
    : NodeIter(RHS.getNodeIterUnchecked()) {}

  // This is templated so that we can allow assigning to a const iterator from
  // a nonconst iterator...
  template<class V, class W, class X, class Y>
  const alist_iterator &operator=(const alist_iterator<V, W, X, Y> &RHS) {
    NodeIter = RHS.getNodeIterUnchecked();
    return *this;
  }

  operator pointer() const { return NodeIter->getElement((T*)0); }

  reference operator*() const { return *NodeIter->getElement((T*)0); }
  pointer   operator->() const { return &operator*(); }

  bool operator==(const alist_iterator &RHS) const {
    return NodeIter == RHS.NodeIter;
  }
  bool operator!=(const alist_iterator &RHS) const {
    return NodeIter != RHS.NodeIter;
  }

  alist_iterator &operator--() {
    --NodeIter;
    return *this;
  }
  alist_iterator &operator++() {
    ++NodeIter;
    return *this;
  }
  alist_iterator operator--(int) {
    alist_iterator tmp = *this;
    --*this;
    return tmp;
  }
  alist_iterator operator++(int) {
    alist_iterator tmp = *this;
    ++*this;
    return tmp;
  }

  NodeIterT getNodeIterUnchecked() const { return NodeIter; }
};

// do not implement. this is to catch errors when people try to use
// them as random access iterators
template<class T, class LargestT, class ValueT, class NodeIterT>
void operator-(int, alist_iterator<T, LargestT, ValueT, NodeIterT>);
template<class T, class LargestT, class ValueT, class NodeIterT>
void operator-(alist_iterator<T, LargestT, ValueT, NodeIterT>,int);

template<class T, class LargestT, class ValueT, class NodeIterT>
void operator+(int, alist_iterator<T, LargestT, ValueT, NodeIterT>);
template<class T, class LargestT, class ValueT, class NodeIterT>
void operator+(alist_iterator<T, LargestT, ValueT, NodeIterT>,int);

// operator!=/operator== - Allow mixed comparisons without dereferencing
// the iterator, which could very likely be pointing to end().
template<class T, class V, class W, class X, class Y>
bool operator!=(T* LHS, const alist_iterator<V, W, X, Y> &RHS) {
  return LHS != RHS.getNodeIterUnchecked().getNodePtrUnchecked()
                                                            ->getElement((T*)0);
}
template<class T, class V, class W, class X, class Y>
bool operator==(T* LHS, const alist_iterator<V, W, X, Y> &RHS) {
  return LHS == RHS.getNodeIterUnchecked().getNodePtrUnchecked()
                                                            ->getElement((T*)0);
}

// Allow alist_iterators to convert into pointers to a node automatically when
// used by the dyn_cast, cast, isa mechanisms...

template<class From> struct simplify_type;

template<class V, class W, class X, class Y>
struct simplify_type<alist_iterator<V, W, X, Y> > {
  typedef alist_node<V, W> NodeTy;
  typedef NodeTy* SimpleType;

  static SimpleType
  getSimplifiedValue(const alist_iterator<V, W, X, Y> &Node) {
    return &*Node;
  }
};
template<class V, class W, class X, class Y>
struct simplify_type<const alist_iterator<V, W, X, Y> > {
  typedef alist_node<V, W> NodeTy;
  typedef NodeTy* SimpleType;

  static SimpleType
  getSimplifiedValue(const alist_iterator<V, W, X, Y> &Node) {
    return &*Node;
  }
};

/// Template traits for alist.  By specializing this template class, you
/// can register custom actions to be run when a node is added to or removed
/// from an alist. A common use of this is to update parent pointers.
///
template<class T, class LargestT = T>
class alist_traits {
  typedef alist_iterator<T, LargestT> iterator;

public:
  void addNodeToList(T *) {}
  void removeNodeFromList(T *) {}
  void transferNodesFromList(alist_traits &, iterator, iterator) {}
  void deleteNode(T *E) { delete alist_node<T, LargestT>::getNode(E); }
};

/// alist - This class is an ilist-style container that automatically
/// adds the next/prev pointers. It is designed to work in cooperation
/// with <llvm/Support/Recycler.h>.
///
template<class T, class LargestT = T>
class alist {
  typedef alist_node<T, LargestT> NodeTy;

public:
  typedef typename ilist<NodeTy>::size_type size_type;

private:
  /// NodeListTraits - ilist traits for NodeList.
  ///
  struct NodeListTraits : ilist_traits<alist_node<T, LargestT> > {
    alist_traits<T, LargestT> UserTraits;

    void addNodeToList(NodeTy *N) {
      UserTraits.addNodeToList(N->getElement((T*)0));
    }
    void removeNodeFromList(NodeTy *N) {
      UserTraits.removeNodeFromList(N->getElement((T*)0));
    }
    void transferNodesFromList(iplist<NodeTy, NodeListTraits> &L2,
                               ilist_iterator<NodeTy> first,
                               ilist_iterator<NodeTy> last) {
      UserTraits.transferNodesFromList(L2.UserTraits,
                                       iterator(first),
                                       iterator(last));
    }
  };

  /// NodeList - Doubly-linked list of nodes that have constructed
  /// contents and may be in active use.
  ///
  iplist<NodeTy, NodeListTraits> NodeList;

public:
  ~alist() { clear(); }

  typedef alist_iterator<T, LargestT, T, ilist_iterator<NodeTy> >
    iterator;
  typedef alist_iterator<T, LargestT, const T, ilist_iterator<const NodeTy> >
    const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  iterator begin() { return iterator(NodeList.begin()); }
  iterator end() { return iterator(NodeList.end()); }
  const_iterator begin() const { return const_iterator(NodeList.begin()); }
  const_iterator end() const { return const_iterator(NodeList.end()); }
  reverse_iterator rbegin() { return reverse_iterator(NodeList.rbegin()); }
  reverse_iterator rend() { return reverse_iterator(NodeList.rend()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(NodeList.rbegin());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(NodeList.rend());
  }

  typedef T& reference;
  typedef const T& const_reference;
  reference front() { return *NodeList.front().getElement((T*)0); }
  reference back()  { return *NodeList.back().getElement((T*)0); }
  const_reference front() const { return *NodeList.front().getElement((T*)0); }
  const_reference back()  const { return *NodeList.back().getElement((T*)0); }

  bool empty() const { return NodeList.empty(); }
  size_type size() const { return NodeList.size(); }

  void push_front(T *E) {
    NodeTy *N = alist_node<T, LargestT>::getNode(E);
    assert(N->getPrev() == 0);
    assert(N->getNext() == 0);
    NodeList.push_front(N);
  }
  void push_back(T *E) {
    NodeTy *N = alist_node<T, LargestT>::getNode(E);
    assert(N->getPrev() == 0);
    assert(N->getNext() == 0);
    NodeList.push_back(N);
  }
  iterator insert(iterator I, T *E) {
    NodeTy *N = alist_node<T, LargestT>::getNode(E);
    assert(N->getPrev() == 0);
    assert(N->getNext() == 0);
    return iterator(NodeList.insert(I.getNodeIterUnchecked(), N));
  }
  void splice(iterator where, alist &Other) {
    NodeList.splice(where.getNodeIterUnchecked(), Other.NodeList);
  }
  void splice(iterator where, alist &Other, iterator From) {
    NodeList.splice(where.getNodeIterUnchecked(), Other.NodeList,
                    From.getNodeIterUnchecked());
  }
  void splice(iterator where, alist &Other, iterator From,
              iterator To) {
    NodeList.splice(where.getNodeIterUnchecked(), Other.NodeList,
                    From.getNodeIterUnchecked(), To.getNodeIterUnchecked());
  }

  void pop_front() {
    erase(NodeList.begin());
  }
  void pop_back() {
    erase(prior(NodeList.end()));
  }
  iterator erase(iterator I) {
    iterator Next = next(I);
    NodeTy *N = NodeList.remove(I.getNodeIterUnchecked());
    NodeList.UserTraits.deleteNode(N->getElement((T*)0));
    return Next;
  }
  iterator erase(iterator first, iterator last) {
    while (first != last)
      first = erase(first);
    return last;
  }

  T *remove(T *E) {
    NodeTy *N = alist_node<T, LargestT>::getNode(E);
    return NodeList.remove(N)->getElement((T*)0);
  }

  void clear() {
    while (!empty()) pop_front();
  }

  alist_traits<T, LargestT> &getTraits() {
    return NodeList.UserTraits;
  }
};

}

#endif
