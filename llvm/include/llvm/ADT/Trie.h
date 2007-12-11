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
// - Something efficient for child storage
// - Should we use char* during construction?
// - GraphTraits interface
// - Eliminate Edge class, which is ok for debugging, but not for end code

template<class Payload>
class Trie {
  class Edge;
  class Node;

  class Edge {
    std::string Label;
    Node *Parent, *Child;

  public:
    typedef enum {
      Same           = -3,
      StringIsPrefix = -2,
      LabelIsPrefix  = -1,
      DontMatch      = 0,
      HaveCommonPart
    } QueryResult;

    inline explicit Edge(std::string label = "",
                         Node* parent = NULL, Node* child = NULL):
      Label(label), Parent(parent), Child(child) { }

    inline void setParent(Node* parent) { Parent = parent; }
    inline Node* getParent() const { return Parent; }
    inline void setChild(Node* child) { Child = child; }
    inline Node* getChild() const { return Child; }
    inline void setLabel(const std::string& label) { Label = label; }
    inline const std::string& getLabel() const { return Label; }

    QueryResult query(const std::string& string) const {
      unsigned i, l;
      unsigned l1 = string.length();
      unsigned l2 = Label.length();

      // Find the length of common part
      l = std::min(l1, l2);
      i = 0;
      while ((i < l) && (string[i] == Label[i]))
        ++i;

      if (i == l) { // One is prefix of another, find who is who
        if (l1 == l2)
          return Same;
        else if (i == l1)
          return StringIsPrefix;
        else
          return LabelIsPrefix;
      } else // String and Label just have common part, return its length
        return (QueryResult)i;
    }
  };

  class Node {
    friend class Trie;

    std::map<char, Edge> Edges;
    Payload Data;
  public:
    inline explicit Node(const Payload& data):Data(data) { }
    inline Node(const Node& n) {
      Data = n.Data;
      Edges = n.Edges;
    }
    inline Node& operator=(const Node& n) {
      if (&n != this) {
        Data = n.Data;
        Edges = n.Edges;
      }

      return *this;
    }

    inline bool isLeaf() const { return Edges.empty(); }

    inline const Payload& getData() const { return Data; }
    inline void setData(const Payload& data) { Data = data; }

    inline Edge* addEdge(const std::string& Label) {
      if (!Edges.insert(std::make_pair(Label[0],
                                       Edge(Label, this))).second) {
        assert(0 && "Edge already exists!");
        return NULL;
      } else
        return &Edges[Label[0]];
    }
  };

  std::vector<Node*> Nodes;
  Payload Empty;

  inline Node* addNode(const Payload& data) {
    Node* N = new Node(data);
    Nodes.push_back(N);
    return N;
  }

  inline Node* splitEdge(Edge& cEdge, size_t index) {
    const std::string& l = cEdge.getLabel();
    assert(index < l.length() && "Trying to split too far!");

    std::string l1 = l.substr(0, index);
    std::string l2 = l.substr(index);

    Node* nNode = addNode(Empty);
    Edge* nEdge = nNode->addEdge(l2);
    nEdge->setChild(cEdge.getChild());
    cEdge.setChild(nNode);
    cEdge.setLabel(l1);

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

  inline Node* getRoot() const { return Nodes[0]; }

  bool addString(const std::string& s, const Payload& data) {
    Node* cNode = getRoot();
    Edge* nEdge = NULL;
    std::string s1(s);

    while (nEdge == NULL) {
      if (cNode->Edges.count(s1[0])) {
        Edge& cEdge = cNode->Edges[s1[0]];
        typename Edge::QueryResult r = cEdge.query(s1);

        switch (r) {
        case Edge::Same:
        case Edge::StringIsPrefix:
        case Edge::DontMatch:
          assert(0 && "Impossible!");
          return false;
        case Edge::LabelIsPrefix:
          s1 = s1.substr(cEdge.getLabel().length());
          cNode = cEdge.getChild();
          break;
        default:
          nEdge = splitEdge(cEdge, r)->addEdge(s1.substr(r));
        }
      } else
        nEdge = cNode->addEdge(s1);
    }

    Node* tNode = addNode(data);
    nEdge->setChild(tNode);

    return true;
  }

  const Payload& lookup(const std::string& s) const {
    Node* cNode = getRoot();
    Node* tNode = NULL;
    std::string s1(s);

    while (tNode == NULL) {
      if (cNode->Edges.count(s1[0])) {
        Edge& cEdge = cNode->Edges[s1[0]];
        typename Edge::QueryResult r = cEdge.query(s1);

        switch (r) {
        case Edge::Same:
          tNode = cEdge.getChild();
          break;
        case Edge::StringIsPrefix:
          return Empty;
        case Edge::DontMatch:
          assert(0 && "Impossible!");
          return Empty;
        case Edge::LabelIsPrefix:
          s1 = s1.substr(cEdge.getLabel().length());
          cNode = cEdge.getChild();
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
