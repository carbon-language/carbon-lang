//===- llvm/ADT/Trie.h ---- Generic trie structure --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Anton Korobeynikov and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class defines a generic trie structure. The trie structure
// is immutable after creation, but the payload contained within it is not.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_TRIE_H
#define LLVM_ADT_TRIE_H

#include <map>
#include <vector>

namespace llvm {

// FIXME:
// - Labels are usually small, maybe it's better to use SmallString
// - Should we use char* during construction?
// - Should we templatize Empty with traits-like interface?
// - GraphTraits interface

template<class Payload>
class Trie {
  class Node {
    friend class Trie;

    typedef enum {
      Same           = -3,
      StringIsPrefix = -2,
      LabelIsPrefix  = -1,
      DontMatch      = 0,
      HaveCommonPart
    } QueryResult;
    typedef std::vector<Node*> NodeVector;
    typedef typename std::vector<Node*>::iterator NodeVectorIter;

    struct NodeCmp {
      bool operator() (Node* N1, Node* N2) {
        return (N1->Label[0] < N2->Label[0]);
      }
      bool operator() (Node* N, char Id) {
        return (N->Label[0] < Id);
      }
    };

    std::string Label;
    Payload Data;
    NodeVector Children;
  public:
    inline explicit Node(const Payload& data, const std::string& label = ""):
        Label(label), Data(data) { }

    inline Node(const Node& n) {
      Data = n.Data;
      Children = n.Children;
      Label = n.Label;
    }
    inline Node& operator=(const Node& n) {
      if (&n != this) {
        Data = n.Data;
        Children = n.Children;
        Label = n.Label;
      }

      return *this;
    }

    inline bool isLeaf() const { return Children.empty(); }

    inline const Payload& getData() const { return Data; }
    inline void setData(const Payload& data) { Data = data; }

    inline void setLabel(const std::string& label) { Label = label; }
    inline const std::string& getLabel() const { return Label; }

#if 0
    inline void dump() {
      std::cerr << "Node: " << this << "\n"
                << "Label: " << Label << "\n"
                << "Children:\n";

      for (NodeVectorIter I = Children.begin(), E = Children.end(); I != E; ++I)
        std::cerr << (*I)->Label << "\n";
    }
#endif

    inline void addEdge(Node* N) {
      if (Children.empty())
        Children.push_back(N);
      else {
        NodeVectorIter I = std::lower_bound(Children.begin(), Children.end(),
                                            N, NodeCmp());
        // FIXME: no dups are allowed
        Children.insert(I, N);
      }
    }

    inline Node* getEdge(char Id) {
      Node* fNode = NULL;
      NodeVectorIter I = std::lower_bound(Children.begin(), Children.end(),
                                          Id, NodeCmp());
      if (I != Children.end() && (*I)->Label[0] == Id)
        fNode = *I;

      return fNode;
    }

    inline void setEdge(Node* N) {
      char Id = N->Label[0];
      NodeVectorIter I = std::lower_bound(Children.begin(), Children.end(),
                                          Id, NodeCmp());
      assert(I != Children.end() && "Node does not exists!");
      *I = N;
    }

    QueryResult query(const std::string& s) const {
      unsigned i, l;
      unsigned l1 = s.length();
      unsigned l2 = Label.length();

      // Find the length of common part
      l = std::min(l1, l2);
      i = 0;
      while ((i < l) && (s[i] == Label[i]))
        ++i;

      if (i == l) { // One is prefix of another, find who is who
        if (l1 == l2)
          return Same;
        else if (i == l1)
          return StringIsPrefix;
        else
          return LabelIsPrefix;
      } else // s and Label have common (possible empty) part, return its length
        return (QueryResult)i;
    }
  };

  std::vector<Node*> Nodes;
  Payload Empty;

  inline Node* getRoot() const { return Nodes[0]; }

  inline Node* addNode(const Payload& data, const std::string label = "") {
    Node* N = new Node(data, label);
    Nodes.push_back(N);
    return N;
  }

  inline Node* splitEdge(Node* N, char Id, size_t index) {
    Node* eNode = N->getEdge(Id);
    assert(eNode && "Node doesn't exist");

    const std::string &l = eNode->Label;
    assert(index > 0 && index < l.length() && "Trying to split too far!");
    std::string l1 = l.substr(0, index);
    std::string l2 = l.substr(index);

    Node* nNode = addNode(Empty, l1);
    N->setEdge(nNode);

    eNode->Label = l2;
    nNode->addEdge(eNode);

    return nNode;
  }

public:
  inline explicit Trie(const Payload& empty):Empty(empty) {
    addNode(Empty);
  }
  inline ~Trie() {
    for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
      delete Nodes[i];
  }

  bool addString(const std::string& s, const Payload& data) {
    Node* cNode = getRoot();
    Node* tNode = NULL;
    std::string s1(s);

    while (tNode == NULL) {
      char Id = s1[0];
      if (Node* nNode = cNode->getEdge(Id)) {
        typename Node::QueryResult r = nNode->query(s1);

        switch (r) {
        case Node::Same:
        case Node::StringIsPrefix:
          // Currently we don't allow to have two strings in the trie one
          // being a prefix of another. This should be fixed.
          assert(0 && "FIXME!");
          return false;
        case Node::DontMatch:
          assert(0 && "Impossible!");
          return false;
        case Node::LabelIsPrefix:
          s1 = s1.substr(nNode->getLabel().length());
          cNode = nNode;
          break;
        default:
         nNode = splitEdge(cNode, Id, r);
         tNode = addNode(data, s1.substr(r));
         nNode->addEdge(tNode);
       }
      } else {
        tNode = addNode(data, s1);
        cNode->addEdge(tNode);
      }
    }

    return true;
  }

  const Payload& lookup(const std::string& s) const {
    Node* cNode = getRoot();
    Node* tNode = NULL;
    std::string s1(s);

    while (tNode == NULL) {
      char Id = s1[0];
      if (Node* nNode = cNode->getEdge(Id)) {
        typename Node::QueryResult r = nNode->query(s1);

        switch (r) {
        case Node::Same:
          tNode = nNode;
          break;
        case Node::StringIsPrefix:
          return Empty;
        case Node::DontMatch:
          assert(0 && "Impossible!");
          return Empty;
        case Node::LabelIsPrefix:
          s1 = s1.substr(nNode->getLabel().length());
          cNode = nNode;
          break;
        default:
          return Empty;
        }
      } else
        return Empty;
    }

    return tNode->getData();
  }

};

}

#endif // LLVM_ADT_TRIE_H
