//===-- PredicateSimplifier.cpp - Path Sensitive Simplifier ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nick Lewycky and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Path-sensitive optimizer. In a branch where x == y, replace uses of
// x with y. Permits further optimization, such as the elimination of
// the unreachable call:
//
// void test(int *p, int *q)
// {
//   if (p != q)
//     return;
// 
//   if (*p != *q)
//     foo(); // unreachable
// }
//
//===----------------------------------------------------------------------===//
//
// The InequalityGraph focusses on four properties; equals, not equals,
// less-than and less-than-or-equals-to. The greater-than forms are also held
// just to allow walking from a lesser node to a greater one. These properties
// are stored in a lattice; LE can become LT or EQ, NE can become LT or GT.
//
// These relationships define a graph between values of the same type. Each
// Value is stored in a map table that retrieves the associated Node. This
// is how EQ relationships are stored; the map contains pointers from equal
// Value to the same node. The node contains a most canonical Value* form
// and the list of known relationships with other nodes.
//
// If two nodes are known to be inequal, then they will contain pointers to
// each other with an "NE" relationship. If node getNode(%x) is less than
// getNode(%y), then the %x node will contain <%y, GT> and %y will contain
// <%x, LT>. This allows us to tie nodes together into a graph like this:
//
//   %a < %b < %c < %d
//
// with four nodes representing the properties. The InequalityGraph provides
// querying with "isRelatedBy" and mutators "addEquality" and "addInequality".
// To find a relationship, we start with one of the nodes any binary search
// through its list to find where the relationships with the second node start.
// Then we iterate through those to find the first relationship that dominates
// our context node.
//
// To create these properties, we wait until a branch or switch instruction
// implies that a particular value is true (or false). The VRPSolver is
// responsible for analyzing the variable and seeing what new inferences
// can be made from each property. For example:
//
//   %P = icmp ne i32* %ptr, null
//   %a = and i1 %P, %Q
//   br i1 %a label %cond_true, label %cond_false
//
// For the true branch, the VRPSolver will start with %a EQ true and look at
// the definition of %a and find that it can infer that %P and %Q are both
// true. From %P being true, it can infer that %ptr NE null. For the false
// branch it can't infer anything from the "and" instruction.
//
// Besides branches, we can also infer properties from instruction that may
// have undefined behaviour in certain cases. For example, the dividend of
// a division may never be zero. After the division instruction, we may assume
// that the dividend is not equal to zero.
//
//===----------------------------------------------------------------------===//
//
// The ValueRanges class stores the known integer bounds of a Value. When we
// encounter i8 %a u< %b, the ValueRanges stores that %a = [1, 255] and
// %b = [0, 254]. Because we store these by Value*, you should always
// canonicalize through the InequalityGraph first.
//
// It never stores an empty range, because that means that the code is
// unreachable. It never stores a single-element range since that's an equality
// relationship and better stored in the InequalityGraph, nor an empty range
// since that is better stored in UnreachableBlocks.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "predsimplify"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ET-Forest.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <deque>
#include <sstream>
using namespace llvm;

STATISTIC(NumVarsReplaced, "Number of argument substitutions");
STATISTIC(NumInstruction , "Number of instructions removed");
STATISTIC(NumSimple      , "Number of simple replacements");
STATISTIC(NumBlocks      , "Number of blocks marked unreachable");
STATISTIC(NumSnuggle     , "Number of comparisons snuggled");

namespace {
  // SLT SGT ULT UGT EQ
  //   0   1   0   1  0 -- GT                  10
  //   0   1   0   1  1 -- GE                  11
  //   0   1   1   0  0 -- SGTULT              12
  //   0   1   1   0  1 -- SGEULE              13
  //   0   1   1   1  0 -- SGT                 14
  //   0   1   1   1  1 -- SGE                 15
  //   1   0   0   1  0 -- SLTUGT              18
  //   1   0   0   1  1 -- SLEUGE              19
  //   1   0   1   0  0 -- LT                  20
  //   1   0   1   0  1 -- LE                  21
  //   1   0   1   1  0 -- SLT                 22
  //   1   0   1   1  1 -- SLE                 23
  //   1   1   0   1  0 -- UGT                 26
  //   1   1   0   1  1 -- UGE                 27
  //   1   1   1   0  0 -- ULT                 28
  //   1   1   1   0  1 -- ULE                 29
  //   1   1   1   1  0 -- NE                  30
  enum LatticeBits {
    EQ_BIT = 1, UGT_BIT = 2, ULT_BIT = 4, SGT_BIT = 8, SLT_BIT = 16
  };
  enum LatticeVal {
    GT = SGT_BIT | UGT_BIT,
    GE = GT | EQ_BIT,
    LT = SLT_BIT | ULT_BIT,
    LE = LT | EQ_BIT,
    NE = SLT_BIT | SGT_BIT | ULT_BIT | UGT_BIT,
    SGTULT = SGT_BIT | ULT_BIT,
    SGEULE = SGTULT | EQ_BIT,
    SLTUGT = SLT_BIT | UGT_BIT,
    SLEUGE = SLTUGT | EQ_BIT,
    ULT = SLT_BIT | SGT_BIT | ULT_BIT,
    UGT = SLT_BIT | SGT_BIT | UGT_BIT,
    SLT = SLT_BIT | ULT_BIT | UGT_BIT,
    SGT = SGT_BIT | ULT_BIT | UGT_BIT,
    SLE = SLT | EQ_BIT,
    SGE = SGT | EQ_BIT,
    ULE = ULT | EQ_BIT,
    UGE = UGT | EQ_BIT
  };

  static bool validPredicate(LatticeVal LV) {
    switch (LV) {
      case GT: case GE: case LT: case LE: case NE:
      case SGTULT: case SGT: case SGEULE:
      case SLTUGT: case SLT: case SLEUGE:
      case ULT: case UGT:
      case SLE: case SGE: case ULE: case UGE:
        return true;
      default:
        return false;
    }
  }

  /// reversePredicate - reverse the direction of the inequality
  static LatticeVal reversePredicate(LatticeVal LV) {
    unsigned reverse = LV ^ (SLT_BIT|SGT_BIT|ULT_BIT|UGT_BIT); //preserve EQ_BIT

    if ((reverse & (SLT_BIT|SGT_BIT)) == 0)
      reverse |= (SLT_BIT|SGT_BIT);

    if ((reverse & (ULT_BIT|UGT_BIT)) == 0)
      reverse |= (ULT_BIT|UGT_BIT);

    LatticeVal Rev = static_cast<LatticeVal>(reverse);
    assert(validPredicate(Rev) && "Failed reversing predicate.");
    return Rev;
  }

  /// This is a StrictWeakOrdering predicate that sorts ETNodes by how many
  /// descendants they have. With this, you can iterate through a list sorted
  /// by this operation and the first matching entry is the most specific
  /// match for your basic block. The order provided is stable; ETNodes with
  /// the same number of children are sorted by pointer address.
  struct VISIBILITY_HIDDEN OrderByDominance {
    bool operator()(const ETNode *LHS, const ETNode *RHS) const {
      unsigned LHS_spread = LHS->getDFSNumOut() - LHS->getDFSNumIn();
      unsigned RHS_spread = RHS->getDFSNumOut() - RHS->getDFSNumIn();
      if (LHS_spread != RHS_spread) return LHS_spread < RHS_spread;
      else return LHS < RHS;
    }
  };

  /// The InequalityGraph stores the relationships between values.
  /// Each Value in the graph is assigned to a Node. Nodes are pointer
  /// comparable for equality. The caller is expected to maintain the logical
  /// consistency of the system.
  ///
  /// The InequalityGraph class may invalidate Node*s after any mutator call.
  /// @brief The InequalityGraph stores the relationships between values.
  class VISIBILITY_HIDDEN InequalityGraph {
    ETNode *TreeRoot;

    InequalityGraph();                  // DO NOT IMPLEMENT
    InequalityGraph(InequalityGraph &); // DO NOT IMPLEMENT
  public:
    explicit InequalityGraph(ETNode *TreeRoot) : TreeRoot(TreeRoot) {}

    class Node;

    /// An Edge is contained inside a Node making one end of the edge implicit
    /// and contains a pointer to the other end. The edge contains a lattice
    /// value specifying the relationship and an ETNode specifying the root
    /// in the dominator tree to which this edge applies.
    class VISIBILITY_HIDDEN Edge {
    public:
      Edge(unsigned T, LatticeVal V, ETNode *ST)
        : To(T), LV(V), Subtree(ST) {}

      unsigned To;
      LatticeVal LV;
      ETNode *Subtree;

      bool operator<(const Edge &edge) const {
        if (To != edge.To) return To < edge.To;
        else return OrderByDominance()(Subtree, edge.Subtree);
      }
      bool operator<(unsigned to) const {
        return To < to;
      }
    };

    /// A single node in the InequalityGraph. This stores the canonical Value
    /// for the node, as well as the relationships with the neighbours.
    ///
    /// @brief A single node in the InequalityGraph.
    class VISIBILITY_HIDDEN Node {
      friend class InequalityGraph;

      typedef SmallVector<Edge, 4> RelationsType;
      RelationsType Relations;

      Value *Canonical;

      // TODO: can this idea improve performance?
      //friend class std::vector<Node>;
      //Node(Node &N) { RelationsType.swap(N.RelationsType); }

    public:
      typedef RelationsType::iterator       iterator;
      typedef RelationsType::const_iterator const_iterator;

      Node(Value *V) : Canonical(V) {}

    private:
#ifndef NDEBUG
    public:
      virtual ~Node() {}
      virtual void dump() const {
        dump(*cerr.stream());
      }
    private:
      void dump(std::ostream &os) const  {
        os << *getValue() << ":\n";
        for (Node::const_iterator NI = begin(), NE = end(); NI != NE; ++NI) {
          static const std::string names[32] =
            { "000000", "000001", "000002", "000003", "000004", "000005",
              "000006", "000007", "000008", "000009", "     >", "    >=",
              "  s>u<", "s>=u<=", "    s>", "   s>=", "000016", "000017",
              "  s<u>", "s<=u>=", "     <", "    <=", "    s<", "   s<=",
              "000024", "000025", "    u>", "   u>=", "    u<", "   u<=",
              "    !=", "000031" };
          os << "  " << names[NI->LV] << " " << NI->To
             << " (" << NI->Subtree->getDFSNumIn() << ")\n";
        }
      }
#endif

    public:
      iterator begin()             { return Relations.begin(); }
      iterator end()               { return Relations.end();   }
      const_iterator begin() const { return Relations.begin(); }
      const_iterator end()   const { return Relations.end();   }

      iterator find(unsigned n, ETNode *Subtree) {
        iterator E = end();
        for (iterator I = std::lower_bound(begin(), E, n);
             I != E && I->To == n; ++I) {
          if (Subtree->DominatedBy(I->Subtree))
            return I;
        }
        return E;
      }

      const_iterator find(unsigned n, ETNode *Subtree) const {
        const_iterator E = end();
        for (const_iterator I = std::lower_bound(begin(), E, n);
             I != E && I->To == n; ++I) {
          if (Subtree->DominatedBy(I->Subtree))
            return I;
        }
        return E;
      }

      Value *getValue() const
      {
        return Canonical;
      }

      /// Updates the lattice value for a given node. Create a new entry if
      /// one doesn't exist, otherwise it merges the values. The new lattice
      /// value must not be inconsistent with any previously existing value.
      void update(unsigned n, LatticeVal R, ETNode *Subtree) {
        assert(validPredicate(R) && "Invalid predicate.");
        iterator I = find(n, Subtree);
        if (I == end()) {
          Edge edge(n, R, Subtree);
          iterator Insert = std::lower_bound(begin(), end(), edge);
          Relations.insert(Insert, edge);
        } else {
          LatticeVal LV = static_cast<LatticeVal>(I->LV & R);
          assert(validPredicate(LV) && "Invalid union of lattice values.");
          if (LV != I->LV) {
            if (Subtree != I->Subtree) {
              assert(Subtree->DominatedBy(I->Subtree) &&
                     "Find returned subtree that doesn't apply.");

              Edge edge(n, R, Subtree);
              iterator Insert = std::lower_bound(begin(), end(), edge);
              Relations.insert(Insert, edge); // invalidates I
              I = find(n, Subtree);
            }

            // Also, we have to tighten any edge that Subtree dominates.
            for (iterator B = begin(); I->To == n; --I) {
              if (I->Subtree->DominatedBy(Subtree)) {
                LatticeVal LV = static_cast<LatticeVal>(I->LV & R);
                assert(validPredicate(LV) && "Invalid union of lattice values");
                I->LV = LV;
              }
              if (I == B) break;
            }
          }
        }
      }
    };

  private:
    struct VISIBILITY_HIDDEN NodeMapEdge {
      Value *V;
      unsigned index;
      ETNode *Subtree;

      NodeMapEdge(Value *V, unsigned index, ETNode *Subtree)
        : V(V), index(index), Subtree(Subtree) {}

      bool operator==(const NodeMapEdge &RHS) const {
        return V == RHS.V &&
               Subtree == RHS.Subtree;
      }

      bool operator<(const NodeMapEdge &RHS) const {
        if (V != RHS.V) return V < RHS.V;
        return OrderByDominance()(Subtree, RHS.Subtree);
      }

      bool operator<(Value *RHS) const {
        return V < RHS;
      }
    };

    typedef std::vector<NodeMapEdge> NodeMapType;
    NodeMapType NodeMap;

    std::vector<Node> Nodes;

  public:
    /// node - returns the node object at a given index retrieved from getNode.
    /// Index zero is reserved and may not be passed in here. The pointer
    /// returned is valid until the next call to newNode or getOrInsertNode.
    Node *node(unsigned index) {
      assert(index != 0 && "Zero index is reserved for not found.");
      assert(index <= Nodes.size() && "Index out of range.");
      return &Nodes[index-1];
    }

    /// Returns the node currently representing Value V, or zero if no such
    /// node exists.
    unsigned getNode(Value *V, ETNode *Subtree) {
      NodeMapType::iterator E = NodeMap.end();
      NodeMapEdge Edge(V, 0, Subtree);
      NodeMapType::iterator I = std::lower_bound(NodeMap.begin(), E, Edge);
      while (I != E && I->V == V) {
        if (Subtree->DominatedBy(I->Subtree))
          return I->index;
        ++I;
      }
      return 0;
    }

    /// getOrInsertNode - always returns a valid node index, creating a node
    /// to match the Value if needed.
    unsigned getOrInsertNode(Value *V, ETNode *Subtree) {
      if (unsigned n = getNode(V, Subtree))
        return n;
      else
        return newNode(V);
    }

    /// newNode - creates a new node for a given Value and returns the index.
    unsigned newNode(Value *V) {
      Nodes.push_back(Node(V));

      NodeMapEdge MapEntry = NodeMapEdge(V, Nodes.size(), TreeRoot);
      assert(!std::binary_search(NodeMap.begin(), NodeMap.end(), MapEntry) &&
             "Attempt to create a duplicate Node.");
      NodeMap.insert(std::lower_bound(NodeMap.begin(), NodeMap.end(),
                                      MapEntry), MapEntry);
      return MapEntry.index;
    }

    /// If the Value is in the graph, return the canonical form. Otherwise,
    /// return the original Value.
    Value *canonicalize(Value *V, ETNode *Subtree) {
      if (isa<Constant>(V)) return V;

      if (unsigned n = getNode(V, Subtree))
        return node(n)->getValue();
      else 
        return V;
    }

    /// isRelatedBy - true iff n1 op n2
    bool isRelatedBy(unsigned n1, unsigned n2, ETNode *Subtree, LatticeVal LV) {
      if (n1 == n2) return LV & EQ_BIT;

      Node *N1 = node(n1);
      Node::iterator I = N1->find(n2, Subtree), E = N1->end();
      if (I != E) return (I->LV & LV) == I->LV;

      return false;
    }

    // The add* methods assume that your input is logically valid and may 
    // assertion-fail or infinitely loop if you attempt a contradiction.

    void addEquality(unsigned n, Value *V, ETNode *Subtree) {
      assert(canonicalize(node(n)->getValue(), Subtree) == node(n)->getValue()
             && "Node's 'canonical' choice isn't best within this subtree.");

      // Suppose that we are given "%x -> node #1 (%y)". The problem is that
      // we may already have "%z -> node #2 (%x)" somewhere above us in the
      // graph. We need to find those edges and add "%z -> node #1 (%y)"
      // to keep the lookups canonical.

      std::vector<Value *> ToRepoint;
      ToRepoint.push_back(V);

      if (unsigned Conflict = getNode(V, Subtree)) {
        for (NodeMapType::iterator I = NodeMap.begin(), E = NodeMap.end();
             I != E; ++I) {
          if (I->index == Conflict && Subtree->DominatedBy(I->Subtree))
            ToRepoint.push_back(I->V);
        }
      }

      for (std::vector<Value *>::iterator VI = ToRepoint.begin(),
           VE = ToRepoint.end(); VI != VE; ++VI) {
        Value *V = *VI;

        // XXX: review this code. This may be doing too many insertions.
        NodeMapEdge Edge(V, n, Subtree);
        NodeMapType::iterator E = NodeMap.end();
        NodeMapType::iterator I = std::lower_bound(NodeMap.begin(), E, Edge);
        if (I == E || I->V != V || I->Subtree != Subtree) {
          // New Value
          NodeMap.insert(I, Edge);
        } else if (I != E && I->V == V && I->Subtree == Subtree) {
          // Update best choice
          I->index = n;
        }

#ifndef NDEBUG
        Node *N = node(n);
        if (isa<Constant>(V)) {
          if (isa<Constant>(N->getValue())) {
            assert(V == N->getValue() && "Constant equals different constant?");
          }
        }
#endif
      }
    }

    /// addInequality - Sets n1 op n2.
    /// It is also an error to call this on an inequality that is already true.
    void addInequality(unsigned n1, unsigned n2, ETNode *Subtree,
                       LatticeVal LV1) {
      assert(n1 != n2 && "A node can't be inequal to itself.");

      if (LV1 != NE)
        assert(!isRelatedBy(n1, n2, Subtree, reversePredicate(LV1)) &&
               "Contradictory inequality.");

      Node *N1 = node(n1);
      Node *N2 = node(n2);

      // Suppose we're adding %n1 < %n2. Find all the %a < %n1 and
      // add %a < %n2 too. This keeps the graph fully connected.
      if (LV1 != NE) {
        // Break up the relationship into signed and unsigned comparison parts.
        // If the signed parts of %a op1 %n1 match that of %n1 op2 %n2, and
        // op1 and op2 aren't NE, then add %a op3 %n2. The new relationship
        // should have the EQ_BIT iff it's set for both op1 and op2.

        unsigned LV1_s = LV1 & (SLT_BIT|SGT_BIT);
        unsigned LV1_u = LV1 & (ULT_BIT|UGT_BIT);

        for (Node::iterator I = N1->begin(), E = N1->end(); I != E; ++I) {
          if (I->LV != NE && I->To != n2) {

            ETNode *Local_Subtree = NULL;
            if (Subtree->DominatedBy(I->Subtree))
              Local_Subtree = Subtree;
            else if (I->Subtree->DominatedBy(Subtree))
              Local_Subtree = I->Subtree;

            if (Local_Subtree) {
              unsigned new_relationship = 0;
              LatticeVal ILV = reversePredicate(I->LV);
              unsigned ILV_s = ILV & (SLT_BIT|SGT_BIT);
              unsigned ILV_u = ILV & (ULT_BIT|UGT_BIT);

              if (LV1_s != (SLT_BIT|SGT_BIT) && ILV_s == LV1_s)
                new_relationship |= ILV_s;
              if (LV1_u != (ULT_BIT|UGT_BIT) && ILV_u == LV1_u)
                new_relationship |= ILV_u;

              if (new_relationship) {
                if ((new_relationship & (SLT_BIT|SGT_BIT)) == 0)
                  new_relationship |= (SLT_BIT|SGT_BIT);
                if ((new_relationship & (ULT_BIT|UGT_BIT)) == 0)
                  new_relationship |= (ULT_BIT|UGT_BIT);
                if ((LV1 & EQ_BIT) && (ILV & EQ_BIT))
                  new_relationship |= EQ_BIT;

                LatticeVal NewLV = static_cast<LatticeVal>(new_relationship);

                node(I->To)->update(n2, NewLV, Local_Subtree);
                N2->update(I->To, reversePredicate(NewLV), Local_Subtree);
              }
            }
          }
        }

        for (Node::iterator I = N2->begin(), E = N2->end(); I != E; ++I) {
          if (I->LV != NE && I->To != n1) {
            ETNode *Local_Subtree = NULL;
            if (Subtree->DominatedBy(I->Subtree))
              Local_Subtree = Subtree;
            else if (I->Subtree->DominatedBy(Subtree))
              Local_Subtree = I->Subtree;

            if (Local_Subtree) {
              unsigned new_relationship = 0;
              unsigned ILV_s = I->LV & (SLT_BIT|SGT_BIT);
              unsigned ILV_u = I->LV & (ULT_BIT|UGT_BIT);

              if (LV1_s != (SLT_BIT|SGT_BIT) && ILV_s == LV1_s)
                new_relationship |= ILV_s;

              if (LV1_u != (ULT_BIT|UGT_BIT) && ILV_u == LV1_u)
                new_relationship |= ILV_u;

              if (new_relationship) {
                if ((new_relationship & (SLT_BIT|SGT_BIT)) == 0)
                  new_relationship |= (SLT_BIT|SGT_BIT);
                if ((new_relationship & (ULT_BIT|UGT_BIT)) == 0)
                  new_relationship |= (ULT_BIT|UGT_BIT);
                if ((LV1 & EQ_BIT) && (I->LV & EQ_BIT))
                  new_relationship |= EQ_BIT;

                LatticeVal NewLV = static_cast<LatticeVal>(new_relationship);

                N1->update(I->To, NewLV, Local_Subtree);
                node(I->To)->update(n1, reversePredicate(NewLV), Local_Subtree);
              }
            }
          }
        }
      }

      N1->update(n2, LV1, Subtree);
      N2->update(n1, reversePredicate(LV1), Subtree);
    }

    /// remove - Removes a Value from the graph. If the value is the canonical
    /// choice for a Node, destroys the Node from the graph deleting all edges
    /// to and from it. This method does not renumber the nodes.
    void remove(Value *V) {
      for (unsigned i = 0; i < NodeMap.size();) {
        NodeMapType::iterator I = NodeMap.begin()+i;
        if (I->V == V) {
          Node *N = node(I->index);
          if (node(I->index)->getValue() == V) {
            for (Node::iterator NI = N->begin(), NE = N->end(); NI != NE; ++NI){
              Node::iterator Iter = node(NI->To)->find(I->index, TreeRoot);
              do {
                node(NI->To)->Relations.erase(Iter);
                Iter = node(NI->To)->find(I->index, TreeRoot);
              } while (Iter != node(NI->To)->end());
            }
            N->Canonical = NULL;
          }
          N->Relations.clear();
          NodeMap.erase(I);
        } else ++i;
      }
    }

#ifndef NDEBUG
    virtual ~InequalityGraph() {}
    virtual void dump() {
      dump(*cerr.stream());
    }

    void dump(std::ostream &os) {
    std::set<Node *> VisitedNodes;
    for (NodeMapType::const_iterator I = NodeMap.begin(), E = NodeMap.end();
         I != E; ++I) {
      Node *N = node(I->index);
      os << *I->V << " == " << I->index
         << "(" << I->Subtree->getDFSNumIn() << ")\n";
      if (VisitedNodes.insert(N).second) {
        os << I->index << ". ";
        if (!N->getValue()) os << "(deleted node)\n";
        else N->dump(os);
      }
    }
  }
#endif
  };

  class VRPSolver;

  /// ValueRanges tracks the known integer ranges and anti-ranges of the nodes
  /// in the InequalityGraph.
  class VISIBILITY_HIDDEN ValueRanges {

    /// A ScopedRange ties an InequalityGraph node with a ConstantRange under
    /// the scope of a rooted subtree in the dominator tree.
    class VISIBILITY_HIDDEN ScopedRange {
    public:
      ScopedRange(Value *V, ConstantRange CR, ETNode *ST)
        : V(V), CR(CR), Subtree(ST) {}

      Value *V;
      ConstantRange CR;
      ETNode *Subtree;

      bool operator<(const ScopedRange &range) const {
        if (V != range.V) return V < range.V;
        else return OrderByDominance()(Subtree, range.Subtree);
      }

      bool operator<(const Value *value) const {
        return V < value;
      }
    };

    TargetData *TD;

    std::vector<ScopedRange> Ranges;
    typedef std::vector<ScopedRange>::iterator iterator;

    // XXX: this is a copy of the code in InequalityGraph::Node. Perhaps a
    // intrusive domtree-scoped container is in order?

    iterator begin() { return Ranges.begin(); }
    iterator end()   { return Ranges.end();   }

    iterator find(Value *V, ETNode *Subtree) {
      iterator E = end();
      for (iterator I = std::lower_bound(begin(), E, V);
           I != E && I->V == V; ++I) {
        if (Subtree->DominatedBy(I->Subtree))
          return I;
      }
      return E;
    }

    void update(Value *V, ConstantRange CR, ETNode *Subtree) {
      assert(!CR.isEmptySet() && "Empty ConstantRange!");
      if (CR.isFullSet()) return;

      iterator I = find(V, Subtree);
      if (I == end()) {
        ScopedRange range(V, CR, Subtree);
        iterator Insert = std::lower_bound(begin(), end(), range);
        Ranges.insert(Insert, range);
      } else {
        CR = CR.intersectWith(I->CR);
        assert(!CR.isEmptySet() && "Empty intersection of ConstantRanges!");

        if (CR != I->CR) {
          if (Subtree != I->Subtree) {
            assert(Subtree->DominatedBy(I->Subtree) &&
                   "Find returned subtree that doesn't apply.");

            ScopedRange range(V, CR, Subtree);
            iterator Insert = std::lower_bound(begin(), end(), range);
            Ranges.insert(Insert, range); // invalidates I
            I = find(V, Subtree);
          }

          // Also, we have to tighten any edge that Subtree dominates.
          for (iterator B = begin(); I->V == V; --I) {
            if (I->Subtree->DominatedBy(Subtree)) {
              I->CR = CR.intersectWith(I->CR);
              assert(!I->CR.isEmptySet() &&
                     "Empty intersection of ConstantRanges!");
            }
            if (I == B) break;
          }
        }
      }
    }

    /// range - Creates a ConstantRange representing the set of all values
    /// that match the ICmpInst::Predicate with any of the values in CR.
    ConstantRange range(ICmpInst::Predicate ICmpOpcode,
                        const ConstantRange &CR) {
      uint32_t W = CR.getBitWidth();
      switch (ICmpOpcode) {
        default: assert(!"Invalid ICmp opcode to range()");
        case ICmpInst::ICMP_EQ:
          return ConstantRange(CR.getLower(), CR.getUpper());
        case ICmpInst::ICMP_NE:
          if (CR.isSingleElement())
            return ConstantRange(CR.getUpper(), CR.getLower());
          return ConstantRange(W);
        case ICmpInst::ICMP_ULT:
          return ConstantRange(APInt::getMinValue(W), CR.getUnsignedMax());
        case ICmpInst::ICMP_SLT:
          return ConstantRange(APInt::getSignedMinValue(W), CR.getSignedMax());
        case ICmpInst::ICMP_ULE: {
          APInt UMax(CR.getUnsignedMax());
          if (UMax.isMaxValue())
            return ConstantRange(W);
          return ConstantRange(APInt::getMinValue(W), UMax + 1);
        }
        case ICmpInst::ICMP_SLE: {
          APInt SMax(CR.getSignedMax());
          if (SMax.isMaxSignedValue() || (SMax+1).isMaxSignedValue())
            return ConstantRange(W);
          return ConstantRange(APInt::getSignedMinValue(W), SMax + 1);
        }
        case ICmpInst::ICMP_UGT:
          return ConstantRange(CR.getUnsignedMin() + 1, APInt::getNullValue(W));
        case ICmpInst::ICMP_SGT:
          return ConstantRange(CR.getSignedMin() + 1,
                               APInt::getSignedMinValue(W));
        case ICmpInst::ICMP_UGE: {
          APInt UMin(CR.getUnsignedMin());
          if (UMin.isMinValue())
            return ConstantRange(W);
          return ConstantRange(UMin, APInt::getNullValue(W));
        }
        case ICmpInst::ICMP_SGE: {
          APInt SMin(CR.getSignedMin());
          if (SMin.isMinSignedValue())
            return ConstantRange(W);
          return ConstantRange(SMin, APInt::getSignedMinValue(W));
        }
      }
    }

    /// create - Creates a ConstantRange that matches the given LatticeVal
    /// relation with a given integer.
    ConstantRange create(LatticeVal LV, const ConstantRange &CR) {
      assert(!CR.isEmptySet() && "Can't deal with empty set.");

      if (LV == NE)
        return range(ICmpInst::ICMP_NE, CR);

      unsigned LV_s = LV & (SGT_BIT|SLT_BIT);
      unsigned LV_u = LV & (UGT_BIT|ULT_BIT);
      bool hasEQ = LV & EQ_BIT;

      ConstantRange Range(CR.getBitWidth());

      if (LV_s == SGT_BIT) {
        Range = Range.intersectWith(range(
                    hasEQ ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_SGT, CR));
      } else if (LV_s == SLT_BIT) {
        Range = Range.intersectWith(range(
                    hasEQ ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_SLT, CR));
      }

      if (LV_u == UGT_BIT) {
        Range = Range.intersectWith(range(
                    hasEQ ? ICmpInst::ICMP_UGE : ICmpInst::ICMP_UGT, CR));
      } else if (LV_u == ULT_BIT) {
        Range = Range.intersectWith(range(
                    hasEQ ? ICmpInst::ICMP_ULE : ICmpInst::ICMP_ULT, CR));
      }

      return Range;
    }

#ifndef NDEBUG
    bool isCanonical(Value *V, ETNode *Subtree, VRPSolver *VRP);
#endif

  public:

    explicit ValueRanges(TargetData *TD) : TD(TD) {}

    // rangeFromValue - converts a Value into a range. If the value is a
    // constant it constructs the single element range, otherwise it performs
    // a lookup. The width W must be retrieved from typeToWidth and may not
    // be zero.
    ConstantRange rangeFromValue(Value *V, ETNode *Subtree, uint32_t W) {
      if (ConstantInt *C = dyn_cast<ConstantInt>(V)) {
        return ConstantRange(C->getValue());
      } else if (isa<ConstantPointerNull>(V)) {
        return ConstantRange(APInt::getNullValue(W));
      } else {
        iterator I = find(V, Subtree);
        if (I != end())
          return I->CR;
      }
      return ConstantRange(W);
    }

    // typeToWidth - returns the number of bits necessary to store a value of
    // this type, or zero if unknown.
    uint32_t typeToWidth(const Type *Ty) const {
      if (TD)
        return TD->getTypeSizeInBits(Ty);

      if (const IntegerType *ITy = dyn_cast<IntegerType>(Ty))
        return ITy->getBitWidth();

      return 0;
    }

    bool isRelatedBy(Value *V1, Value *V2, ETNode *Subtree, LatticeVal LV) {
      uint32_t W = typeToWidth(V1->getType());
      if (!W) return false;

      ConstantRange CR1 = rangeFromValue(V1, Subtree, W);
      ConstantRange CR2 = rangeFromValue(V2, Subtree, W);

      // True iff all values in CR1 are LV to all values in CR2.
      switch (LV) {
      default: assert(!"Impossible lattice value!");
      case NE:
        return CR1.intersectWith(CR2).isEmptySet();
      case ULT:
        return CR1.getUnsignedMax().ult(CR2.getUnsignedMin());
      case ULE:
        return CR1.getUnsignedMax().ule(CR2.getUnsignedMin());
      case UGT:
        return CR1.getUnsignedMin().ugt(CR2.getUnsignedMax());
      case UGE:
        return CR1.getUnsignedMin().uge(CR2.getUnsignedMax());
      case SLT:
        return CR1.getSignedMax().slt(CR2.getSignedMin());
      case SLE:
        return CR1.getSignedMax().sle(CR2.getSignedMin());
      case SGT:
        return CR1.getSignedMin().sgt(CR2.getSignedMax());
      case SGE:
        return CR1.getSignedMin().sge(CR2.getSignedMax());
      case LT:
        return CR1.getUnsignedMax().ult(CR2.getUnsignedMin()) &&
               CR1.getSignedMax().slt(CR2.getUnsignedMin());
      case LE:
        return CR1.getUnsignedMax().ule(CR2.getUnsignedMin()) &&
               CR1.getSignedMax().sle(CR2.getUnsignedMin());
      case GT:
        return CR1.getUnsignedMin().ugt(CR2.getUnsignedMax()) &&
               CR1.getSignedMin().sgt(CR2.getSignedMax());
      case GE:
        return CR1.getUnsignedMin().uge(CR2.getUnsignedMax()) &&
               CR1.getSignedMin().sge(CR2.getSignedMax());
      case SLTUGT:
        return CR1.getSignedMax().slt(CR2.getSignedMin()) &&
               CR1.getUnsignedMin().ugt(CR2.getUnsignedMax());
      case SLEUGE:
        return CR1.getSignedMax().sle(CR2.getSignedMin()) &&
               CR1.getUnsignedMin().uge(CR2.getUnsignedMax());
      case SGTULT:
        return CR1.getSignedMin().sgt(CR2.getSignedMax()) &&
               CR1.getUnsignedMax().ult(CR2.getUnsignedMin());
      case SGEULE:
        return CR1.getSignedMin().sge(CR2.getSignedMax()) &&
               CR1.getUnsignedMax().ule(CR2.getUnsignedMin());
      }
    }

    void addToWorklist(Value *V, Constant *C, ICmpInst::Predicate Pred,
                       VRPSolver *VRP);
    void markBlock(VRPSolver *VRP);

    void mergeInto(Value **I, unsigned n, Value *New, ETNode *Subtree,
                   VRPSolver *VRP) {
      assert(isCanonical(New, Subtree, VRP) && "Best choice not canonical?");

      uint32_t W = typeToWidth(New->getType());
      if (!W) return;

      ConstantRange CR_New = rangeFromValue(New, Subtree, W);
      ConstantRange Merged = CR_New;

      for (; n != 0; ++I, --n) {
        ConstantRange CR_Kill = rangeFromValue(*I, Subtree, W);
        if (CR_Kill.isFullSet()) continue;
        Merged = Merged.intersectWith(CR_Kill);
      }

      if (Merged.isFullSet() || Merged == CR_New) return;

      applyRange(New, Merged, Subtree, VRP);
    }

    void applyRange(Value *V, const ConstantRange &CR, ETNode *Subtree,
                    VRPSolver *VRP) {
      assert(isCanonical(V, Subtree, VRP) && "Value not canonical.");

      if (const APInt *I = CR.getSingleElement()) {
        const Type *Ty = V->getType();
        if (Ty->isInteger()) {
          addToWorklist(V, ConstantInt::get(*I), ICmpInst::ICMP_EQ, VRP);
          return;
        } else if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
          assert(*I == 0 && "Pointer is null but not zero?");
          addToWorklist(V, ConstantPointerNull::get(PTy),
                      ICmpInst::ICMP_EQ, VRP);
          return;
        }
      }

      ConstantRange Merged = CR.intersectWith(
                                rangeFromValue(V, Subtree, CR.getBitWidth()));
      if (Merged.isEmptySet()) {
        markBlock(VRP);
        return;
      }

      update(V, Merged, Subtree);
    }

    void addNotEquals(Value *V1, Value *V2, ETNode *Subtree, VRPSolver *VRP) {
      uint32_t W = typeToWidth(V1->getType());
      if (!W) return;

      ConstantRange CR1 = rangeFromValue(V1, Subtree, W);
      ConstantRange CR2 = rangeFromValue(V2, Subtree, W);

      if (const APInt *I = CR1.getSingleElement()) {
        if (CR2.isFullSet()) {
          ConstantRange NewCR2(CR1.getUpper(), CR1.getLower());
          applyRange(V2, NewCR2, Subtree, VRP);
        } else if (*I == CR2.getLower()) {
          APInt NewLower(CR2.getLower() + 1),
                NewUpper(CR2.getUpper());
          if (NewLower == NewUpper)
            NewLower = NewUpper = APInt::getMinValue(W);

          ConstantRange NewCR2(NewLower, NewUpper);
          applyRange(V2, NewCR2, Subtree, VRP);
        } else if (*I == CR2.getUpper() - 1) {
          APInt NewLower(CR2.getLower()),
                NewUpper(CR2.getUpper() - 1);
          if (NewLower == NewUpper)
            NewLower = NewUpper = APInt::getMinValue(W);

          ConstantRange NewCR2(NewLower, NewUpper);
          applyRange(V2, NewCR2, Subtree, VRP);
        }
      }

      if (const APInt *I = CR2.getSingleElement()) {
        if (CR1.isFullSet()) {
          ConstantRange NewCR1(CR2.getUpper(), CR2.getLower());
          applyRange(V1, NewCR1, Subtree, VRP);
        } else if (*I == CR1.getLower()) {
          APInt NewLower(CR1.getLower() + 1),
                NewUpper(CR1.getUpper());
          if (NewLower == NewUpper)
            NewLower = NewUpper = APInt::getMinValue(W);

          ConstantRange NewCR1(NewLower, NewUpper);
          applyRange(V1, NewCR1, Subtree, VRP);
        } else if (*I == CR1.getUpper() - 1) {
          APInt NewLower(CR1.getLower()),
                NewUpper(CR1.getUpper() - 1);
          if (NewLower == NewUpper)
            NewLower = NewUpper = APInt::getMinValue(W);

          ConstantRange NewCR1(NewLower, NewUpper);
          applyRange(V1, NewCR1, Subtree, VRP);
        }
      }
    }

    void addInequality(Value *V1, Value *V2, ETNode *Subtree, LatticeVal LV,
                       VRPSolver *VRP) {
      assert(!isRelatedBy(V1, V2, Subtree, LV) && "Asked to do useless work.");

      assert(isCanonical(V1, Subtree, VRP) && "Value not canonical.");
      assert(isCanonical(V2, Subtree, VRP) && "Value not canonical.");

      if (LV == NE) {
        addNotEquals(V1, V2, Subtree, VRP);
        return;
      }

      uint32_t W = typeToWidth(V1->getType());
      if (!W) return;

      ConstantRange CR1 = rangeFromValue(V1, Subtree, W);
      ConstantRange CR2 = rangeFromValue(V2, Subtree, W);

      if (!CR1.isSingleElement()) {
        ConstantRange NewCR1 = CR1.intersectWith(create(LV, CR2));
        if (NewCR1 != CR1)
          applyRange(V1, NewCR1, Subtree, VRP);
      }

      if (!CR2.isSingleElement()) {
        ConstantRange NewCR2 = CR2.intersectWith(create(reversePredicate(LV),
                                                        CR1));
        if (NewCR2 != CR2)
          applyRange(V2, NewCR2, Subtree, VRP);
      }
    }
  };

  /// UnreachableBlocks keeps tracks of blocks that are for one reason or
  /// another discovered to be unreachable. This is used to cull the graph when
  /// analyzing instructions, and to mark blocks with the "unreachable"
  /// terminator instruction after the function has executed.
  class VISIBILITY_HIDDEN UnreachableBlocks {
  private:
    std::vector<BasicBlock *> DeadBlocks;

  public:
    /// mark - mark a block as dead
    void mark(BasicBlock *BB) {
      std::vector<BasicBlock *>::iterator E = DeadBlocks.end();
      std::vector<BasicBlock *>::iterator I =
        std::lower_bound(DeadBlocks.begin(), E, BB);

      if (I == E || *I != BB) DeadBlocks.insert(I, BB);
    }

    /// isDead - returns whether a block is known to be dead already
    bool isDead(BasicBlock *BB) {
      std::vector<BasicBlock *>::iterator E = DeadBlocks.end();
      std::vector<BasicBlock *>::iterator I =
        std::lower_bound(DeadBlocks.begin(), E, BB);

      return I != E && *I == BB;
    }

    /// kill - replace the dead blocks' terminator with an UnreachableInst.
    bool kill() {
      bool modified = false;
      for (std::vector<BasicBlock *>::iterator I = DeadBlocks.begin(),
           E = DeadBlocks.end(); I != E; ++I) {
        BasicBlock *BB = *I;

        DOUT << "unreachable block: " << BB->getName() << "\n";

        for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB);
             SI != SE; ++SI) {
          BasicBlock *Succ = *SI;
          Succ->removePredecessor(BB);
        }

        TerminatorInst *TI = BB->getTerminator();
        TI->replaceAllUsesWith(UndefValue::get(TI->getType()));
        TI->eraseFromParent();
        new UnreachableInst(BB);
        ++NumBlocks;
        modified = true;
      }
      DeadBlocks.clear();
      return modified;
    }
  };

  /// VRPSolver keeps track of how changes to one variable affect other
  /// variables, and forwards changes along to the InequalityGraph. It
  /// also maintains the correct choice for "canonical" in the IG.
  /// @brief VRPSolver calculates inferences from a new relationship.
  class VISIBILITY_HIDDEN VRPSolver {
  private:
    friend class ValueRanges;

    struct Operation {
      Value *LHS, *RHS;
      ICmpInst::Predicate Op;

      BasicBlock *ContextBB;
      Instruction *ContextInst;
    };
    std::deque<Operation> WorkList;

    InequalityGraph &IG;
    UnreachableBlocks &UB;
    ValueRanges &VR;

    ETForest *Forest;
    ETNode *Top;
    BasicBlock *TopBB;
    Instruction *TopInst;
    bool &modified;

    typedef InequalityGraph::Node Node;

    /// IdomI - Determines whether one Instruction dominates another.
    bool IdomI(Instruction *I1, Instruction *I2) const {
      BasicBlock *BB1 = I1->getParent(),
                 *BB2 = I2->getParent();
      if (BB1 == BB2) {
        if (isa<TerminatorInst>(I1)) return false;
        if (isa<TerminatorInst>(I2)) return true;
        if (isa<PHINode>(I1) && !isa<PHINode>(I2)) return true;
        if (!isa<PHINode>(I1) && isa<PHINode>(I2)) return false;

        for (BasicBlock::const_iterator I = BB1->begin(), E = BB1->end();
             I != E; ++I) {
          if (&*I == I1) return true;
          if (&*I == I2) return false;
        }
        assert(!"Instructions not found in parent BasicBlock?");
      } else {
        return Forest->properlyDominates(BB1, BB2);
      }
      return false;
    }

    /// Returns true if V1 is a better canonical value than V2.
    bool compare(Value *V1, Value *V2) const {
      if (isa<Constant>(V1))
        return !isa<Constant>(V2);
      else if (isa<Constant>(V2))
        return false;
      else if (isa<Argument>(V1))
        return !isa<Argument>(V2);
      else if (isa<Argument>(V2))
        return false;

      Instruction *I1 = dyn_cast<Instruction>(V1);
      Instruction *I2 = dyn_cast<Instruction>(V2);

      if (!I1 || !I2)
        return V1->getNumUses() < V2->getNumUses();

      return IdomI(I1, I2);
    }

    // below - true if the Instruction is dominated by the current context
    // block or instruction
    bool below(Instruction *I) {
      if (TopInst)
        return IdomI(TopInst, I);
      else {
        ETNode *Node = Forest->getNodeForBlock(I->getParent());
        return Node->DominatedBy(Top);
      }
    }

    bool makeEqual(Value *V1, Value *V2) {
      DOUT << "makeEqual(" << *V1 << ", " << *V2 << ")\n";

      assert(V1->getType() == V2->getType() &&
             "Can't make two values with different types equal.");

      if (V1 == V2) return true;

      if (isa<Constant>(V1) && isa<Constant>(V2))
        return false;

      unsigned n1 = IG.getNode(V1, Top), n2 = IG.getNode(V2, Top);

      if (n1 && n2) {
        if (n1 == n2) return true;
        if (IG.isRelatedBy(n1, n2, Top, NE)) return false;
      }

      if (n1) assert(V1 == IG.node(n1)->getValue() && "Value isn't canonical.");
      if (n2) assert(V2 == IG.node(n2)->getValue() && "Value isn't canonical.");

      assert(!compare(V2, V1) && "Please order parameters to makeEqual.");

      assert(!isa<Constant>(V2) && "Tried to remove a constant.");

      SetVector<unsigned> Remove;
      if (n2) Remove.insert(n2);

      if (n1 && n2) {
        // Suppose we're being told that %x == %y, and %x <= %z and %y >= %z.
        // We can't just merge %x and %y because the relationship with %z would
        // be EQ and that's invalid. What we're doing is looking for any nodes
        // %z such that %x <= %z and %y >= %z, and vice versa.

        Node *N1 = IG.node(n1);
        Node *N2 = IG.node(n2);
        Node::iterator end = N2->end();

        // Find the intersection between N1 and N2 which is dominated by
        // Top. If we find %x where N1 <= %x <= N2 (or >=) then add %x to
        // Remove.
        for (Node::iterator I = N1->begin(), E = N1->end(); I != E; ++I) {
          if (!(I->LV & EQ_BIT) || !Top->DominatedBy(I->Subtree)) continue;

          unsigned ILV_s = I->LV & (SLT_BIT|SGT_BIT);
          unsigned ILV_u = I->LV & (ULT_BIT|UGT_BIT);
          Node::iterator NI = N2->find(I->To, Top);
          if (NI != end) {
            LatticeVal NILV = reversePredicate(NI->LV);
            unsigned NILV_s = NILV & (SLT_BIT|SGT_BIT);
            unsigned NILV_u = NILV & (ULT_BIT|UGT_BIT);

            if ((ILV_s != (SLT_BIT|SGT_BIT) && ILV_s == NILV_s) ||
                (ILV_u != (ULT_BIT|UGT_BIT) && ILV_u == NILV_u))
              Remove.insert(I->To);
          }
        }

        // See if one of the nodes about to be removed is actually a better
        // canonical choice than n1.
        unsigned orig_n1 = n1;
        SetVector<unsigned>::iterator DontRemove = Remove.end();
        for (SetVector<unsigned>::iterator I = Remove.begin()+1 /* skip n2 */,
             E = Remove.end(); I != E; ++I) {
          unsigned n = *I;
          Value *V = IG.node(n)->getValue();
          if (compare(V, V1)) {
            V1 = V;
            n1 = n;
            DontRemove = I;
          }
        }
        if (DontRemove != Remove.end()) {
          unsigned n = *DontRemove;
          Remove.remove(n);
          Remove.insert(orig_n1);
        }
      }

      // We'd like to allow makeEqual on two values to perform a simple
      // substitution without every creating nodes in the IG whenever possible.
      //
      // The first iteration through this loop operates on V2 before going
      // through the Remove list and operating on those too. If all of the
      // iterations performed simple replacements then we exit early.
      bool mergeIGNode = false;
      unsigned i = 0;
      for (Value *R = V2; i == 0 || i < Remove.size(); ++i) {
        if (i) R = IG.node(Remove[i])->getValue(); // skip n2.

        // Try to replace the whole instruction. If we can, we're done.
        Instruction *I2 = dyn_cast<Instruction>(R);
        if (I2 && below(I2)) {
          std::vector<Instruction *> ToNotify;
          for (Value::use_iterator UI = R->use_begin(), UE = R->use_end();
               UI != UE;) {
            Use &TheUse = UI.getUse();
            ++UI;
            if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser()))
              ToNotify.push_back(I);
          }

          DOUT << "Simply removing " << *I2
               << ", replacing with " << *V1 << "\n";
          I2->replaceAllUsesWith(V1);
          // leave it dead; it'll get erased later.
          ++NumInstruction;
          modified = true;

          for (std::vector<Instruction *>::iterator II = ToNotify.begin(),
               IE = ToNotify.end(); II != IE; ++II) {
            opsToDef(*II);
          }

          continue;
        }

        // Otherwise, replace all dominated uses.
        for (Value::use_iterator UI = R->use_begin(), UE = R->use_end();
             UI != UE;) {
          Use &TheUse = UI.getUse();
          ++UI;
          if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser())) {
            if (below(I)) {
              TheUse.set(V1);
              modified = true;
              ++NumVarsReplaced;
              opsToDef(I);
            }
          }
        }

        // If that killed the instruction, stop here.
        if (I2 && isInstructionTriviallyDead(I2)) {
          DOUT << "Killed all uses of " << *I2
               << ", replacing with " << *V1 << "\n";
          continue;
        }

        // If we make it to here, then we will need to create a node for N1.
        // Otherwise, we can skip out early!
        mergeIGNode = true;
      }

      if (!isa<Constant>(V1)) {
        if (Remove.empty()) {
          VR.mergeInto(&V2, 1, V1, Top, this);
        } else {
          std::vector<Value*> RemoveVals;
          RemoveVals.reserve(Remove.size());

          for (SetVector<unsigned>::iterator I = Remove.begin(),
               E = Remove.end(); I != E; ++I) {
            Value *V = IG.node(*I)->getValue();
            if (!V->use_empty())
              RemoveVals.push_back(V);
          }
          VR.mergeInto(&RemoveVals[0], RemoveVals.size(), V1, Top, this);
        }
      }

      if (mergeIGNode) {
        // Create N1.
        if (!n1) n1 = IG.newNode(V1);

        // Migrate relationships from removed nodes to N1.
        Node *N1 = IG.node(n1);
        for (SetVector<unsigned>::iterator I = Remove.begin(), E = Remove.end();
             I != E; ++I) {
          unsigned n = *I;
          Node *N = IG.node(n);
          for (Node::iterator NI = N->begin(), NE = N->end(); NI != NE; ++NI) {
            if (NI->Subtree->DominatedBy(Top)) {
              if (NI->To == n1) {
                assert((NI->LV & EQ_BIT) && "Node inequal to itself.");
                continue;
              }
              if (Remove.count(NI->To))
                continue;

              IG.node(NI->To)->update(n1, reversePredicate(NI->LV), Top);
              N1->update(NI->To, NI->LV, Top);
            }
          }
        }

        // Point V2 (and all items in Remove) to N1.
        if (!n2)
          IG.addEquality(n1, V2, Top);
        else {
          for (SetVector<unsigned>::iterator I = Remove.begin(),
               E = Remove.end(); I != E; ++I) {
            IG.addEquality(n1, IG.node(*I)->getValue(), Top);
          }
        }

        // If !Remove.empty() then V2 = Remove[0]->getValue().
        // Even when Remove is empty, we still want to process V2.
        i = 0;
        for (Value *R = V2; i == 0 || i < Remove.size(); ++i) {
          if (i) R = IG.node(Remove[i])->getValue(); // skip n2.

          if (Instruction *I2 = dyn_cast<Instruction>(R)) {
            if (below(I2) ||
                Top->DominatedBy(Forest->getNodeForBlock(I2->getParent())))
            defToOps(I2);
          }
          for (Value::use_iterator UI = V2->use_begin(), UE = V2->use_end();
               UI != UE;) {
            Use &TheUse = UI.getUse();
            ++UI;
            if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser())) {
              if (below(I) ||
                  Top->DominatedBy(Forest->getNodeForBlock(I->getParent())))
                opsToDef(I);
            }
          }
        }
      }

      // re-opsToDef all dominated users of V1.
      if (Instruction *I = dyn_cast<Instruction>(V1)) {
        for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
             UI != UE;) {
          Use &TheUse = UI.getUse();
          ++UI;
          Value *V = TheUse.getUser();
          if (!V->use_empty()) {
            if (Instruction *Inst = dyn_cast<Instruction>(V)) {
              if (below(Inst) ||
                  Top->DominatedBy(Forest->getNodeForBlock(Inst->getParent())))
                opsToDef(Inst);
            }
          }
        }
      }

      return true;
    }

    /// cmpInstToLattice - converts an CmpInst::Predicate to lattice value
    /// Requires that the lattice value be valid; does not accept ICMP_EQ.
    static LatticeVal cmpInstToLattice(ICmpInst::Predicate Pred) {
      switch (Pred) {
        case ICmpInst::ICMP_EQ:
          assert(!"No matching lattice value.");
          return static_cast<LatticeVal>(EQ_BIT);
        default:
          assert(!"Invalid 'icmp' predicate.");
        case ICmpInst::ICMP_NE:
          return NE;
        case ICmpInst::ICMP_UGT:
          return UGT;
        case ICmpInst::ICMP_UGE:
          return UGE;
        case ICmpInst::ICMP_ULT:
          return ULT;
        case ICmpInst::ICMP_ULE:
          return ULE;
        case ICmpInst::ICMP_SGT:
          return SGT;
        case ICmpInst::ICMP_SGE:
          return SGE;
        case ICmpInst::ICMP_SLT:
          return SLT;
        case ICmpInst::ICMP_SLE:
          return SLE;
      }
    }

  public:
    VRPSolver(InequalityGraph &IG, UnreachableBlocks &UB, ValueRanges &VR,
              ETForest *Forest, bool &modified, BasicBlock *TopBB)
      : IG(IG),
        UB(UB),
        VR(VR),
        Forest(Forest),
        Top(Forest->getNodeForBlock(TopBB)),
        TopBB(TopBB),
        TopInst(NULL),
        modified(modified) {}

    VRPSolver(InequalityGraph &IG, UnreachableBlocks &UB, ValueRanges &VR,
              ETForest *Forest, bool &modified, Instruction *TopInst)
      : IG(IG),
        UB(UB),
        VR(VR),
        Forest(Forest),
        TopInst(TopInst),
        modified(modified)
    {
      TopBB = TopInst->getParent();
      Top = Forest->getNodeForBlock(TopBB);
    }

    bool isRelatedBy(Value *V1, Value *V2, ICmpInst::Predicate Pred) const {
      if (Constant *C1 = dyn_cast<Constant>(V1))
        if (Constant *C2 = dyn_cast<Constant>(V2))
          return ConstantExpr::getCompare(Pred, C1, C2) ==
                 ConstantInt::getTrue();

      if (unsigned n1 = IG.getNode(V1, Top))
        if (unsigned n2 = IG.getNode(V2, Top)) {
          if (n1 == n2) return Pred == ICmpInst::ICMP_EQ ||
                               Pred == ICmpInst::ICMP_ULE ||
                               Pred == ICmpInst::ICMP_UGE ||
                               Pred == ICmpInst::ICMP_SLE ||
                               Pred == ICmpInst::ICMP_SGE;
          if (Pred == ICmpInst::ICMP_EQ) return false;
          if (IG.isRelatedBy(n1, n2, Top, cmpInstToLattice(Pred))) return true;
        }

      if (Pred == ICmpInst::ICMP_EQ) return V1 == V2;
      return VR.isRelatedBy(V1, V2, Top, cmpInstToLattice(Pred));
    }

    /// add - adds a new property to the work queue
    void add(Value *V1, Value *V2, ICmpInst::Predicate Pred,
             Instruction *I = NULL) {
      DOUT << "adding " << *V1 << " " << Pred << " " << *V2;
      if (I) DOUT << " context: " << *I;
      else DOUT << " default context";
      DOUT << "\n";

      assert(V1->getType() == V2->getType() &&
             "Can't relate two values with different types.");

      WorkList.push_back(Operation());
      Operation &O = WorkList.back();
      O.LHS = V1, O.RHS = V2, O.Op = Pred, O.ContextInst = I;
      O.ContextBB = I ? I->getParent() : TopBB;
    }

    /// defToOps - Given an instruction definition that we've learned something
    /// new about, find any new relationships between its operands.
    void defToOps(Instruction *I) {
      Instruction *NewContext = below(I) ? I : TopInst;
      Value *Canonical = IG.canonicalize(I, Top);

      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
        const Type *Ty = BO->getType();
        assert(!Ty->isFPOrFPVector() && "Float in work queue!");

        Value *Op0 = IG.canonicalize(BO->getOperand(0), Top);
        Value *Op1 = IG.canonicalize(BO->getOperand(1), Top);

        // TODO: "and i32 -1, %x" EQ %y then %x EQ %y.

        switch (BO->getOpcode()) {
          case Instruction::And: {
            // "and i32 %a, %b" EQ -1 then %a EQ -1 and %b EQ -1
            ConstantInt *CI = ConstantInt::getAllOnesValue(Ty);
            if (Canonical == CI) {
              add(CI, Op0, ICmpInst::ICMP_EQ, NewContext);
              add(CI, Op1, ICmpInst::ICMP_EQ, NewContext);
            }
          } break;
          case Instruction::Or: {
            // "or i32 %a, %b" EQ 0 then %a EQ 0 and %b EQ 0
            Constant *Zero = Constant::getNullValue(Ty);
            if (Canonical == Zero) {
              add(Zero, Op0, ICmpInst::ICMP_EQ, NewContext);
              add(Zero, Op1, ICmpInst::ICMP_EQ, NewContext);
            }
          } break;
          case Instruction::Xor: {
            // "xor i32 %c, %a" EQ %b then %a EQ %c ^ %b
            // "xor i32 %c, %a" EQ %c then %a EQ 0
            // "xor i32 %c, %a" NE %c then %a NE 0
            // Repeat the above, with order of operands reversed.
            Value *LHS = Op0;
            Value *RHS = Op1;
            if (!isa<Constant>(LHS)) std::swap(LHS, RHS);

            if (ConstantInt *CI = dyn_cast<ConstantInt>(Canonical)) {
              if (ConstantInt *Arg = dyn_cast<ConstantInt>(LHS)) {
                add(RHS, ConstantInt::get(CI->getValue() ^ Arg->getValue()),
                    ICmpInst::ICMP_EQ, NewContext);
              }
            }
            if (Canonical == LHS) {
              if (isa<ConstantInt>(Canonical))
                add(RHS, Constant::getNullValue(Ty), ICmpInst::ICMP_EQ,
                    NewContext);
            } else if (isRelatedBy(LHS, Canonical, ICmpInst::ICMP_NE)) {
              add(RHS, Constant::getNullValue(Ty), ICmpInst::ICMP_NE,
                  NewContext);
            }
          } break;
          default:
            break;
        }
      } else if (ICmpInst *IC = dyn_cast<ICmpInst>(I)) {
        // "icmp ult i32 %a, %y" EQ true then %a u< y
        // etc.

        if (Canonical == ConstantInt::getTrue()) {
          add(IC->getOperand(0), IC->getOperand(1), IC->getPredicate(),
              NewContext);
        } else if (Canonical == ConstantInt::getFalse()) {
          add(IC->getOperand(0), IC->getOperand(1),
              ICmpInst::getInversePredicate(IC->getPredicate()), NewContext);
        }
      } else if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
        if (I->getType()->isFPOrFPVector()) return;

        // Given: "%a = select i1 %x, i32 %b, i32 %c"
        // %a EQ %b and %b NE %c then %x EQ true
        // %a EQ %c and %b NE %c then %x EQ false

        Value *True  = SI->getTrueValue();
        Value *False = SI->getFalseValue();
        if (isRelatedBy(True, False, ICmpInst::ICMP_NE)) {
          if (Canonical == IG.canonicalize(True, Top) ||
              isRelatedBy(Canonical, False, ICmpInst::ICMP_NE))
            add(SI->getCondition(), ConstantInt::getTrue(),
                ICmpInst::ICMP_EQ, NewContext);
          else if (Canonical == IG.canonicalize(False, Top) ||
                   isRelatedBy(Canonical, True, ICmpInst::ICMP_NE))
            add(SI->getCondition(), ConstantInt::getFalse(),
                ICmpInst::ICMP_EQ, NewContext);
        }
      } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
        for (GetElementPtrInst::op_iterator OI = GEPI->idx_begin(),
             OE = GEPI->idx_end(); OI != OE; ++OI) {
          ConstantInt *Op = dyn_cast<ConstantInt>(IG.canonicalize(*OI, Top));
          if (!Op || !Op->isZero()) return;
        }
        // TODO: The GEPI indices are all zero. Copy from definition to operand,
        // jumping the type plane as needed.
        if (isRelatedBy(GEPI, Constant::getNullValue(GEPI->getType()),
                        ICmpInst::ICMP_NE)) {
          Value *Ptr = GEPI->getPointerOperand();
          add(Ptr, Constant::getNullValue(Ptr->getType()), ICmpInst::ICMP_NE,
              NewContext);
        }
      } else if (CastInst *CI = dyn_cast<CastInst>(I)) {
        const Type *SrcTy = CI->getSrcTy();

        Value *TheCI = IG.canonicalize(CI, Top);
        uint32_t W = VR.typeToWidth(SrcTy);
        if (!W) return;
        ConstantRange CR = VR.rangeFromValue(TheCI, Top, W);

        if (CR.isFullSet()) return;

        switch (CI->getOpcode()) {
          default: break;
          case Instruction::ZExt:
          case Instruction::SExt:
            VR.applyRange(IG.canonicalize(CI->getOperand(0), Top),
                          CR.truncate(W), Top, this);
            break;
          case Instruction::BitCast:
            VR.applyRange(IG.canonicalize(CI->getOperand(0), Top),
                          CR, Top, this);
            break;
        }
      }
    }

    /// opsToDef - A new relationship was discovered involving one of this
    /// instruction's operands. Find any new relationship involving the
    /// definition, or another operand.
    void opsToDef(Instruction *I) {
      Instruction *NewContext = below(I) ? I : TopInst;

      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
        Value *Op0 = IG.canonicalize(BO->getOperand(0), Top);
        Value *Op1 = IG.canonicalize(BO->getOperand(1), Top);

        if (ConstantInt *CI0 = dyn_cast<ConstantInt>(Op0))
          if (ConstantInt *CI1 = dyn_cast<ConstantInt>(Op1)) {
            add(BO, ConstantExpr::get(BO->getOpcode(), CI0, CI1),
                ICmpInst::ICMP_EQ, NewContext);
            return;
          }

        // "%y = and i1 true, %x" then %x EQ %y
        // "%y = or i1 false, %x" then %x EQ %y
        // "%x = add i32 %y, 0" then %x EQ %y
        // "%x = mul i32 %y, 0" then %x EQ 0

        Instruction::BinaryOps Opcode = BO->getOpcode();
        const Type *Ty = BO->getType();
        assert(!Ty->isFPOrFPVector() && "Float in work queue!");

        Constant *Zero = Constant::getNullValue(Ty);
        ConstantInt *AllOnes = ConstantInt::getAllOnesValue(Ty);

        switch (Opcode) {
          default: break;
          case Instruction::LShr:
          case Instruction::AShr:
          case Instruction::Shl:
          case Instruction::Sub:
            if (Op1 == Zero) {
              add(BO, Op0, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            break;
          case Instruction::Or:
            if (Op0 == AllOnes || Op1 == AllOnes) {
              add(BO, AllOnes, ICmpInst::ICMP_EQ, NewContext);
              return;
            } // fall-through
          case Instruction::Xor:
          case Instruction::Add:
            if (Op0 == Zero) {
              add(BO, Op1, ICmpInst::ICMP_EQ, NewContext);
              return;
            } else if (Op1 == Zero) {
              add(BO, Op0, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            break;
          case Instruction::And:
            if (Op0 == AllOnes) {
              add(BO, Op1, ICmpInst::ICMP_EQ, NewContext);
              return;
            } else if (Op1 == AllOnes) {
              add(BO, Op0, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            // fall-through
          case Instruction::Mul:
            if (Op0 == Zero || Op1 == Zero) {
              add(BO, Zero, ICmpInst::ICMP_EQ, NewContext);
              return;
            }
            break;
        }

        // "%x = add i32 %y, %z" and %x EQ %y then %z EQ 0
        // "%x = add i32 %y, %z" and %x EQ %z then %y EQ 0
        // "%x = shl i32 %y, %z" and %x EQ %y and %y NE 0 then %z EQ 0
        // "%x = udiv i32 %y, %z" and %x EQ %y then %z EQ 1

        Value *Known = Op0, *Unknown = Op1,
              *TheBO = IG.canonicalize(BO, Top);
        if (Known != TheBO) std::swap(Known, Unknown);
        if (Known == TheBO) {
          switch (Opcode) {
            default: break;
            case Instruction::LShr:
            case Instruction::AShr:
            case Instruction::Shl:
              if (!isRelatedBy(Known, Zero, ICmpInst::ICMP_NE)) break;
              // otherwise, fall-through.
            case Instruction::Sub:
              if (Unknown == Op1) break;
              // otherwise, fall-through.
            case Instruction::Xor:
            case Instruction::Add:
              add(Unknown, Zero, ICmpInst::ICMP_EQ, NewContext);
              break;
            case Instruction::UDiv:
            case Instruction::SDiv:
              if (Unknown == Op1) break;
              if (isRelatedBy(Known, Zero, ICmpInst::ICMP_NE)) {
                Constant *One = ConstantInt::get(Ty, 1);
                add(Unknown, One, ICmpInst::ICMP_EQ, NewContext);
              }
              break;
          }
        }

        // TODO: "%a = add i32 %b, 1" and %b > %z then %a >= %z.

      } else if (ICmpInst *IC = dyn_cast<ICmpInst>(I)) {
        // "%a = icmp ult i32 %b, %c" and %b u<  %c then %a EQ true
        // "%a = icmp ult i32 %b, %c" and %b u>= %c then %a EQ false
        // etc.

        Value *Op0 = IG.canonicalize(IC->getOperand(0), Top);
        Value *Op1 = IG.canonicalize(IC->getOperand(1), Top);

        ICmpInst::Predicate Pred = IC->getPredicate();
        if (isRelatedBy(Op0, Op1, Pred)) {
          add(IC, ConstantInt::getTrue(), ICmpInst::ICMP_EQ, NewContext);
        } else if (isRelatedBy(Op0, Op1, ICmpInst::getInversePredicate(Pred))) {
          add(IC, ConstantInt::getFalse(), ICmpInst::ICMP_EQ, NewContext);
        }

      } else if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
        if (I->getType()->isFPOrFPVector()) return;

        // Given: "%a = select i1 %x, i32 %b, i32 %c"
        // %x EQ true  then %a EQ %b
        // %x EQ false then %a EQ %c
        // %b EQ %c then %a EQ %b

        Value *Canonical = IG.canonicalize(SI->getCondition(), Top);
        if (Canonical == ConstantInt::getTrue()) {
          add(SI, SI->getTrueValue(), ICmpInst::ICMP_EQ, NewContext);
        } else if (Canonical == ConstantInt::getFalse()) {
          add(SI, SI->getFalseValue(), ICmpInst::ICMP_EQ, NewContext);
        } else if (IG.canonicalize(SI->getTrueValue(), Top) ==
                   IG.canonicalize(SI->getFalseValue(), Top)) {
          add(SI, SI->getTrueValue(), ICmpInst::ICMP_EQ, NewContext);
        }
      } else if (CastInst *CI = dyn_cast<CastInst>(I)) {
        const Type *DestTy = CI->getDestTy();
        if (DestTy->isFPOrFPVector()) return;

        Value *Op = IG.canonicalize(CI->getOperand(0), Top);
        Instruction::CastOps Opcode = CI->getOpcode();

        if (Constant *C = dyn_cast<Constant>(Op)) {
          add(CI, ConstantExpr::getCast(Opcode, C, DestTy),
              ICmpInst::ICMP_EQ, NewContext);
        }

        uint32_t W = VR.typeToWidth(DestTy);
        Value *TheCI = IG.canonicalize(CI, Top);
        ConstantRange CR = VR.rangeFromValue(Op, Top, W);

        if (!CR.isFullSet()) {
          switch (Opcode) {
            default: break;
            case Instruction::ZExt:
              VR.applyRange(TheCI, CR.zeroExtend(W), Top, this);
              break;
            case Instruction::SExt:
              VR.applyRange(TheCI, CR.signExtend(W), Top, this);
              break;
            case Instruction::Trunc: {
              ConstantRange Result = CR.truncate(W);
              if (!Result.isFullSet())
                VR.applyRange(TheCI, Result, Top, this);
            } break;
            case Instruction::BitCast:
              VR.applyRange(TheCI, CR, Top, this);
              break;
            // TODO: other casts?
          }
        }
      } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
        for (GetElementPtrInst::op_iterator OI = GEPI->idx_begin(),
             OE = GEPI->idx_end(); OI != OE; ++OI) {
          ConstantInt *Op = dyn_cast<ConstantInt>(IG.canonicalize(*OI, Top));
          if (!Op || !Op->isZero()) return;
        }
        // TODO: The GEPI indices are all zero. Copy from operand to definition,
        // jumping the type plane as needed.
        Value *Ptr = GEPI->getPointerOperand();
        if (isRelatedBy(Ptr, Constant::getNullValue(Ptr->getType()),
                        ICmpInst::ICMP_NE)) {
          add(GEPI, Constant::getNullValue(GEPI->getType()), ICmpInst::ICMP_NE,
              NewContext);
        }
      }
    }

    /// solve - process the work queue
    void solve() {
      //DOUT << "WorkList entry, size: " << WorkList.size() << "\n";
      while (!WorkList.empty()) {
        //DOUT << "WorkList size: " << WorkList.size() << "\n";

        Operation &O = WorkList.front();
        TopInst = O.ContextInst;
        TopBB = O.ContextBB;
        Top = Forest->getNodeForBlock(TopBB);

        O.LHS = IG.canonicalize(O.LHS, Top);
        O.RHS = IG.canonicalize(O.RHS, Top);

        assert(O.LHS == IG.canonicalize(O.LHS, Top) && "Canonicalize isn't.");
        assert(O.RHS == IG.canonicalize(O.RHS, Top) && "Canonicalize isn't.");

        DOUT << "solving " << *O.LHS << " " << O.Op << " " << *O.RHS;
        if (O.ContextInst) DOUT << " context inst: " << *O.ContextInst;
        else DOUT << " context block: " << O.ContextBB->getName();
        DOUT << "\n";

        DEBUG(IG.dump());

        // If they're both Constant, skip it. Check for contradiction and mark
        // the BB as unreachable if so.
        if (Constant *CI_L = dyn_cast<Constant>(O.LHS)) {
          if (Constant *CI_R = dyn_cast<Constant>(O.RHS)) {
            if (ConstantExpr::getCompare(O.Op, CI_L, CI_R) ==
                ConstantInt::getFalse())
              UB.mark(TopBB);

            WorkList.pop_front();
            continue;
          }
        }

        if (compare(O.LHS, O.RHS)) {
          std::swap(O.LHS, O.RHS);
          O.Op = ICmpInst::getSwappedPredicate(O.Op);
        }

        if (O.Op == ICmpInst::ICMP_EQ) {
          if (!makeEqual(O.RHS, O.LHS))
            UB.mark(TopBB);
        } else {
          LatticeVal LV = cmpInstToLattice(O.Op);

          if ((LV & EQ_BIT) &&
              isRelatedBy(O.LHS, O.RHS, ICmpInst::getSwappedPredicate(O.Op))) {
            if (!makeEqual(O.RHS, O.LHS))
              UB.mark(TopBB);
          } else {
            if (isRelatedBy(O.LHS, O.RHS, ICmpInst::getInversePredicate(O.Op))){
              UB.mark(TopBB);
              WorkList.pop_front();
              continue;
            }

            unsigned n1 = IG.getNode(O.LHS, Top);
            unsigned n2 = IG.getNode(O.RHS, Top);

            if (n1 && n1 == n2) {
              if (O.Op != ICmpInst::ICMP_UGE && O.Op != ICmpInst::ICMP_ULE &&
                  O.Op != ICmpInst::ICMP_SGE && O.Op != ICmpInst::ICMP_SLE)
                UB.mark(TopBB);

              WorkList.pop_front();
              continue;
            }

            if (VR.isRelatedBy(O.LHS, O.RHS, Top, LV) ||
                (n1 && n2 && IG.isRelatedBy(n1, n2, Top, LV))) {
              WorkList.pop_front();
              continue;
            }

            VR.addInequality(O.LHS, O.RHS, Top, LV, this);
            if ((!isa<ConstantInt>(O.RHS) && !isa<ConstantInt>(O.LHS)) ||
                LV == NE) {
              if (!n1) n1 = IG.newNode(O.LHS);
              if (!n2) n2 = IG.newNode(O.RHS);
              IG.addInequality(n1, n2, Top, LV);
            }

            if (Instruction *I1 = dyn_cast<Instruction>(O.LHS)) {
              if (below(I1) ||
                  Top->DominatedBy(Forest->getNodeForBlock(I1->getParent())))
                defToOps(I1);
            }
            if (isa<Instruction>(O.LHS) || isa<Argument>(O.LHS)) {
              for (Value::use_iterator UI = O.LHS->use_begin(),
                   UE = O.LHS->use_end(); UI != UE;) {
                Use &TheUse = UI.getUse();
                ++UI;
                if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser())) {
                  if (below(I) ||
                      Top->DominatedBy(Forest->getNodeForBlock(I->getParent())))
                    opsToDef(I);
                }
              }
            }
            if (Instruction *I2 = dyn_cast<Instruction>(O.RHS)) {
              if (below(I2) ||
                  Top->DominatedBy(Forest->getNodeForBlock(I2->getParent())))
              defToOps(I2);
            }
            if (isa<Instruction>(O.RHS) || isa<Argument>(O.RHS)) {
              for (Value::use_iterator UI = O.RHS->use_begin(),
                   UE = O.RHS->use_end(); UI != UE;) {
                Use &TheUse = UI.getUse();
                ++UI;
                if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser())) {
                  if (below(I) ||
                      Top->DominatedBy(Forest->getNodeForBlock(I->getParent())))

                    opsToDef(I);
                }
              }
            }
          }
        }
        WorkList.pop_front();
      }
    }
  };

  void ValueRanges::addToWorklist(Value *V, Constant *C,
                                  ICmpInst::Predicate Pred, VRPSolver *VRP) {
    VRP->add(V, C, Pred, VRP->TopInst);
  }

  void ValueRanges::markBlock(VRPSolver *VRP) {
    VRP->UB.mark(VRP->TopBB);
  }

#ifndef NDEBUG
  bool ValueRanges::isCanonical(Value *V, ETNode *Subtree, VRPSolver *VRP) {
    return V == VRP->IG.canonicalize(V, Subtree);
  }
#endif

  /// PredicateSimplifier - This class is a simplifier that replaces
  /// one equivalent variable with another. It also tracks what
  /// can't be equal and will solve setcc instructions when possible.
  /// @brief Root of the predicate simplifier optimization.
  class VISIBILITY_HIDDEN PredicateSimplifier : public FunctionPass {
    DominatorTree *DT;
    ETForest *Forest;
    bool modified;
    InequalityGraph *IG;
    UnreachableBlocks UB;
    ValueRanges *VR;

    std::vector<DominatorTree::Node *> WorkList;

  public:
    static char ID; // Pass identification, replacement for typeid
    PredicateSimplifier() : FunctionPass((intptr_t)&ID) {}

    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addRequired<DominatorTree>();
      AU.addRequired<ETForest>();
      AU.addRequired<TargetData>();
      AU.addPreserved<TargetData>();
    }

  private:
    /// Forwards - Adds new properties into PropertySet and uses them to
    /// simplify instructions. Because new properties sometimes apply to
    /// a transition from one BasicBlock to another, this will use the
    /// PredicateSimplifier::proceedToSuccessor(s) interface to enter the
    /// basic block with the new PropertySet.
    /// @brief Performs abstract execution of the program.
    class VISIBILITY_HIDDEN Forwards : public InstVisitor<Forwards> {
      friend class InstVisitor<Forwards>;
      PredicateSimplifier *PS;
      DominatorTree::Node *DTNode;

    public:
      InequalityGraph &IG;
      UnreachableBlocks &UB;
      ValueRanges &VR;

      Forwards(PredicateSimplifier *PS, DominatorTree::Node *DTNode)
        : PS(PS), DTNode(DTNode), IG(*PS->IG), UB(PS->UB), VR(*PS->VR) {}

      void visitTerminatorInst(TerminatorInst &TI);
      void visitBranchInst(BranchInst &BI);
      void visitSwitchInst(SwitchInst &SI);

      void visitAllocaInst(AllocaInst &AI);
      void visitLoadInst(LoadInst &LI);
      void visitStoreInst(StoreInst &SI);

      void visitSExtInst(SExtInst &SI);
      void visitZExtInst(ZExtInst &ZI);

      void visitBinaryOperator(BinaryOperator &BO);
      void visitICmpInst(ICmpInst &IC);
    };
  
    // Used by terminator instructions to proceed from the current basic
    // block to the next. Verifies that "current" dominates "next",
    // then calls visitBasicBlock.
    void proceedToSuccessors(DominatorTree::Node *Current) {
      for (DominatorTree::Node::iterator I = Current->begin(),
           E = Current->end(); I != E; ++I) {
        WorkList.push_back(*I);
      }
    }

    void proceedToSuccessor(DominatorTree::Node *Next) {
      WorkList.push_back(Next);
    }

    // Visits each instruction in the basic block.
    void visitBasicBlock(DominatorTree::Node *Node) {
      BasicBlock *BB = Node->getBlock();
      ETNode *ET = Forest->getNodeForBlock(BB);
      DOUT << "Entering Basic Block: " << BB->getName()
           << " (" << ET->getDFSNumIn() << ")\n";
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
        visitInstruction(I++, Node, ET);
      }
    }

    // Tries to simplify each Instruction and add new properties to
    // the PropertySet.
    void visitInstruction(Instruction *I, DominatorTree::Node *DT, ETNode *ET) {
      DOUT << "Considering instruction " << *I << "\n";
      DEBUG(IG->dump());

      // Sometimes instructions are killed in earlier analysis.
      if (isInstructionTriviallyDead(I)) {
        ++NumSimple;
        modified = true;
        IG->remove(I);
        I->eraseFromParent();
        return;
      }

#ifndef NDEBUG
      // Try to replace the whole instruction.
      Value *V = IG->canonicalize(I, ET);
      assert(V == I && "Late instruction canonicalization.");
      if (V != I) {
        modified = true;
        ++NumInstruction;
        DOUT << "Removing " << *I << ", replacing with " << *V << "\n";
        IG->remove(I);
        I->replaceAllUsesWith(V);
        I->eraseFromParent();
        return;
      }

      // Try to substitute operands.
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
        Value *Oper = I->getOperand(i);
        Value *V = IG->canonicalize(Oper, ET);
        assert(V == Oper && "Late operand canonicalization.");
        if (V != Oper) {
          modified = true;
          ++NumVarsReplaced;
          DOUT << "Resolving " << *I;
          I->setOperand(i, V);
          DOUT << " into " << *I;
        }
      }
#endif

      std::string name = I->getParent()->getName();
      DOUT << "push (%" << name << ")\n";
      Forwards visit(this, DT);
      visit.visit(*I);
      DOUT << "pop (%" << name << ")\n";
    }
  };

  bool PredicateSimplifier::runOnFunction(Function &F) {
    DT = &getAnalysis<DominatorTree>();
    Forest = &getAnalysis<ETForest>();

    TargetData *TD = &getAnalysis<TargetData>();

    // XXX: should only act when numbers are out of date
    Forest->updateDFSNumbers();

    DOUT << "Entering Function: " << F.getName() << "\n";

    modified = false;
    BasicBlock *RootBlock = &F.getEntryBlock();
    IG = new InequalityGraph(Forest->getNodeForBlock(RootBlock));
    VR = new ValueRanges(TD);
    WorkList.push_back(DT->getRootNode());

    do {
      DominatorTree::Node *DTNode = WorkList.back();
      WorkList.pop_back();
      if (!UB.isDead(DTNode->getBlock())) visitBasicBlock(DTNode);
    } while (!WorkList.empty());

    delete VR;
    delete IG;

    modified |= UB.kill();

    return modified;
  }

  void PredicateSimplifier::Forwards::visitTerminatorInst(TerminatorInst &TI) {
    PS->proceedToSuccessors(DTNode);
  }

  void PredicateSimplifier::Forwards::visitBranchInst(BranchInst &BI) {
    if (BI.isUnconditional()) {
      PS->proceedToSuccessors(DTNode);
      return;
    }

    Value *Condition = BI.getCondition();
    BasicBlock *TrueDest  = BI.getSuccessor(0);
    BasicBlock *FalseDest = BI.getSuccessor(1);

    if (isa<Constant>(Condition) || TrueDest == FalseDest) {
      PS->proceedToSuccessors(DTNode);
      return;
    }

    for (DominatorTree::Node::iterator I = DTNode->begin(), E = DTNode->end();
         I != E; ++I) {
      BasicBlock *Dest = (*I)->getBlock();
      DOUT << "Branch thinking about %" << Dest->getName()
           << "(" << PS->Forest->getNodeForBlock(Dest)->getDFSNumIn() << ")\n";

      if (Dest == TrueDest) {
        DOUT << "(" << DTNode->getBlock()->getName() << ") true set:\n";
        VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, Dest);
        VRP.add(ConstantInt::getTrue(), Condition, ICmpInst::ICMP_EQ);
        VRP.solve();
        DEBUG(IG.dump());
      } else if (Dest == FalseDest) {
        DOUT << "(" << DTNode->getBlock()->getName() << ") false set:\n";
        VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, Dest);
        VRP.add(ConstantInt::getFalse(), Condition, ICmpInst::ICMP_EQ);
        VRP.solve();
        DEBUG(IG.dump());
      }

      PS->proceedToSuccessor(*I);
    }
  }

  void PredicateSimplifier::Forwards::visitSwitchInst(SwitchInst &SI) {
    Value *Condition = SI.getCondition();

    // Set the EQProperty in each of the cases BBs, and the NEProperties
    // in the default BB.

    for (DominatorTree::Node::iterator I = DTNode->begin(), E = DTNode->end();
         I != E; ++I) {
      BasicBlock *BB = (*I)->getBlock();
      DOUT << "Switch thinking about BB %" << BB->getName()
           << "(" << PS->Forest->getNodeForBlock(BB)->getDFSNumIn() << ")\n";

      VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, BB);
      if (BB == SI.getDefaultDest()) {
        for (unsigned i = 1, e = SI.getNumCases(); i < e; ++i)
          if (SI.getSuccessor(i) != BB)
            VRP.add(Condition, SI.getCaseValue(i), ICmpInst::ICMP_NE);
        VRP.solve();
      } else if (ConstantInt *CI = SI.findCaseDest(BB)) {
        VRP.add(Condition, CI, ICmpInst::ICMP_EQ);
        VRP.solve();
      }
      PS->proceedToSuccessor(*I);
    }
  }

  void PredicateSimplifier::Forwards::visitAllocaInst(AllocaInst &AI) {
    VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &AI);
    VRP.add(Constant::getNullValue(AI.getType()), &AI, ICmpInst::ICMP_NE);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitLoadInst(LoadInst &LI) {
    Value *Ptr = LI.getPointerOperand();
    // avoid "load uint* null" -> null NE null.
    if (isa<Constant>(Ptr)) return;

    VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &LI);
    VRP.add(Constant::getNullValue(Ptr->getType()), Ptr, ICmpInst::ICMP_NE);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitStoreInst(StoreInst &SI) {
    Value *Ptr = SI.getPointerOperand();
    if (isa<Constant>(Ptr)) return;

    VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &SI);
    VRP.add(Constant::getNullValue(Ptr->getType()), Ptr, ICmpInst::ICMP_NE);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitSExtInst(SExtInst &SI) {
    VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &SI);
    uint32_t SrcBitWidth = cast<IntegerType>(SI.getSrcTy())->getBitWidth();
    uint32_t DstBitWidth = cast<IntegerType>(SI.getDestTy())->getBitWidth();
    APInt Min(APInt::getHighBitsSet(DstBitWidth, DstBitWidth-SrcBitWidth+1));
    APInt Max(APInt::getLowBitsSet(DstBitWidth, SrcBitWidth-1));
    VRP.add(ConstantInt::get(Min), &SI, ICmpInst::ICMP_SLE);
    VRP.add(ConstantInt::get(Max), &SI, ICmpInst::ICMP_SGE);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitZExtInst(ZExtInst &ZI) {
    VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &ZI);
    uint32_t SrcBitWidth = cast<IntegerType>(ZI.getSrcTy())->getBitWidth();
    uint32_t DstBitWidth = cast<IntegerType>(ZI.getDestTy())->getBitWidth();
    APInt Max(APInt::getLowBitsSet(DstBitWidth, SrcBitWidth));
    VRP.add(ConstantInt::get(Max), &ZI, ICmpInst::ICMP_UGE);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitBinaryOperator(BinaryOperator &BO) {
    Instruction::BinaryOps ops = BO.getOpcode();

    switch (ops) {
    default: break;
      case Instruction::URem:
      case Instruction::SRem:
      case Instruction::UDiv:
      case Instruction::SDiv: {
        Value *Divisor = BO.getOperand(1);
        VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &BO);
        VRP.add(Constant::getNullValue(Divisor->getType()), Divisor,
                ICmpInst::ICMP_NE);
        VRP.solve();
        break;
      }
    }

    switch (ops) {
      default: break;
      case Instruction::Shl: {
        VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(0), ICmpInst::ICMP_UGE);
        VRP.solve();
      } break;
      case Instruction::AShr: {
        VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(0), ICmpInst::ICMP_SLE);
        VRP.solve();
      } break;
      case Instruction::LShr:
      case Instruction::UDiv: {
        VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(0), ICmpInst::ICMP_ULE);
        VRP.solve();
      } break;
      case Instruction::URem: {
        VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(1), ICmpInst::ICMP_ULE);
        VRP.solve();
      } break;
      case Instruction::And: {
        VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(0), ICmpInst::ICMP_ULE);
        VRP.add(&BO, BO.getOperand(1), ICmpInst::ICMP_ULE);
        VRP.solve();
      } break;
      case Instruction::Or: {
        VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &BO);
        VRP.add(&BO, BO.getOperand(0), ICmpInst::ICMP_UGE);
        VRP.add(&BO, BO.getOperand(1), ICmpInst::ICMP_UGE);
        VRP.solve();
      } break;
    }
  }

  void PredicateSimplifier::Forwards::visitICmpInst(ICmpInst &IC) {
    // If possible, squeeze the ICmp predicate into something simpler.
    // Eg., if x = [0, 4) and we're being asked icmp uge %x, 3 then change
    // the predicate to eq.

    // XXX: once we do full PHI handling, modifying the instruction in the
    // Forwards visitor will cause missed optimizations.

    ICmpInst::Predicate Pred = IC.getPredicate();

    switch (Pred) {
      default: break;
      case ICmpInst::ICMP_ULE: Pred = ICmpInst::ICMP_ULT; break;
      case ICmpInst::ICMP_UGE: Pred = ICmpInst::ICMP_UGT; break;
      case ICmpInst::ICMP_SLE: Pred = ICmpInst::ICMP_SLT; break;
      case ICmpInst::ICMP_SGE: Pred = ICmpInst::ICMP_SGT; break;
    }
    if (Pred != IC.getPredicate()) {
      VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &IC);
      if (VRP.isRelatedBy(IC.getOperand(1), IC.getOperand(0),
                          ICmpInst::ICMP_NE)) {
        ++NumSnuggle;
        PS->modified = true;
        IC.setPredicate(Pred);
      }
    }

    Pred = IC.getPredicate();

    if (ConstantInt *Op1 = dyn_cast<ConstantInt>(IC.getOperand(1))) {
      ConstantInt *NextVal = 0;
      switch (Pred) {
        default: break;
        case ICmpInst::ICMP_SLT:
        case ICmpInst::ICMP_ULT:
          if (Op1->getValue() != 0)
            NextVal = ConstantInt::get(Op1->getValue()-1);
         break;
        case ICmpInst::ICMP_SGT:
        case ICmpInst::ICMP_UGT:
          if (!Op1->getValue().isAllOnesValue())
            NextVal = ConstantInt::get(Op1->getValue()+1);
         break;

      }
      if (NextVal) {
        VRPSolver VRP(IG, UB, VR, PS->Forest, PS->modified, &IC);
        if (VRP.isRelatedBy(IC.getOperand(0), NextVal,
                            ICmpInst::getInversePredicate(Pred))) {
          ICmpInst *NewIC = new ICmpInst(ICmpInst::ICMP_EQ, IC.getOperand(0),
                                         NextVal, "", &IC);
          NewIC->takeName(&IC);
          IC.replaceAllUsesWith(NewIC);
          IG.remove(&IC); // XXX: prove this isn't necessary
          IC.eraseFromParent();
          ++NumSnuggle;
          PS->modified = true;
        }
      }
    }
  }

  char PredicateSimplifier::ID = 0;
  RegisterPass<PredicateSimplifier> X("predsimplify",
                                      "Predicate Simplifier");
}

FunctionPass *llvm::createPredicateSimplifierPass() {
  return new PredicateSimplifier();
}
