//==- llvm/ADT/alist_node.h - Next/Prev helper class for alist ---*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the alist_node class template, which is used by the alist
// class template to provide next/prev pointers for arbitrary objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ALIST_NODE_H
#define LLVM_ADT_ALIST_NODE_H

#include "llvm/ADT/ilist.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {

/// alist_node - This is a utility class used by alist. It holds prev and next
/// pointers for use with ilists, as well as storage for objects as large as
/// LargestT, that are in T's inheritance tree.
///
template<class T, class LargestT = T>
class alist_node {
  alist_node *Prev, *Next;

public:
  alist_node() : Prev(0), Next(0) {}

  alist_node *getPrev() const { return Prev; }
  alist_node *getNext() const { return Next; }
  void setPrev(alist_node *N) { Prev = N; }
  void setNext(alist_node *N) { Next = N; }

  union {
    char Bytes[sizeof(LargestT)];
    long long L;
    void *P;
  } Storage;

  template<class SubClass>
  SubClass *getElement(SubClass *) {
    assert(sizeof(SubClass) <= sizeof(LargestT));
    return reinterpret_cast<SubClass*>(&Storage.Bytes);
  }

  template<class SubClass>
  const SubClass *getElement(SubClass *) const {
    assert(sizeof(SubClass) <= sizeof(LargestT));
    return reinterpret_cast<const SubClass*>(&Storage.Bytes);
  }

  // This code essentially does offsetof, but actual offsetof hits an ICE in
  // GCC 4.0 relating to offsetof being used inside a template.
  static alist_node* getNode(T *D) {
    return reinterpret_cast<alist_node*>(reinterpret_cast<char*>(D) -
                                                (uintptr_t)&getNull()->Storage);
  }
  static const alist_node* getNode(const T *D) {
    return reinterpret_cast<alist_node*>(reinterpret_cast<char*>(D) -
                                                (uintptr_t)&getNull()->Storage);
  }
private:
  static alist_node* getNull() { return 0; }
};

// A specialization of ilist_traits for alist_nodes.
template<class T, class LargestT>
class ilist_traits<alist_node<T, LargestT> > {
public:
  typedef alist_node<T, LargestT> NodeTy;

protected:
  // Allocate a sentinel inside the traits class. This works
  // because iplist carries an instance of the traits class.
  NodeTy Sentinel;

public:
  static NodeTy *getPrev(NodeTy *N) { return N->getPrev(); }
  static NodeTy *getNext(NodeTy *N) { return N->getNext(); }
  static const NodeTy *getPrev(const NodeTy *N) { return N->getPrev(); }
  static const NodeTy *getNext(const NodeTy *N) { return N->getNext(); }

  static void setPrev(NodeTy *N, NodeTy *Prev) { N->setPrev(Prev); }
  static void setNext(NodeTy *N, NodeTy *Next) { N->setNext(Next); }

  NodeTy *createSentinel() const {
    assert(Sentinel.getPrev() == 0);
    assert(Sentinel.getNext() == 0);
    return const_cast<NodeTy*>(&Sentinel);
  }

  void destroySentinel(NodeTy *N) {
    assert(N == &Sentinel); N = N;
    Sentinel.setPrev(0);
    Sentinel.setNext(0);
  }

  void addNodeToList(NodeTy *) {}
  void removeNodeFromList(NodeTy *) {}
  void transferNodesFromList(iplist<NodeTy, ilist_traits> &,
                             ilist_iterator<NodeTy> /*first*/,
                             ilist_iterator<NodeTy> /*last*/) {}

  // Ideally we wouldn't implement this, but ilist's clear calls it,
  // which is called from ilist's destructor. We won't ever call
  // either of those with a non-empty list, but statically this
  // method needs to exist.
  void deleteNode(NodeTy *) { assert(0); }

private:
  static NodeTy *createNode(const NodeTy &V); // do not implement
};

}

#endif
