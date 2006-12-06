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
// This pass focusses on four properties; equals, not equals, less-than
// and less-than-or-equals-to. The greater-than forms are also held just
// to allow walking from a lesser node to a greater one. These properties
// are stored in a lattice; LE can become LT or EQ, NE can become LT or GT.
//
// These relationships define a graph between values of the same type. Each
// Value is stored in a map table that retrieves the associated Node. This
// is how EQ relationships are stored; the map contains pointers to the
// same node. The node contains a most canonical Value* form and the list of
// known relationships.
//
// If two nodes are known to be inequal, then they will contain pointers to
// each other with an "NE" relationship. If node getNode(%x) is less than
// getNode(%y), then the %x node will contain <%y, GT> and %y will contain
// <%x, LT>. This allows us to tie nodes together into a graph like this:
//
//   %a < %b < %c < %d
//
// with four nodes representing the properties. The InequalityGraph provides
// queries (such as "isEqual") and mutators (such as "addEqual"). To implement
// "isLess(%a, %c)", we start with getNode(%c) and walk downwards until
// we reach %a or the leaf node. Note that the graph is directed and acyclic,
// but may contain joins, meaning that this walk is not a linear time
// algorithm.
//
// To create these properties, we wait until a branch or switch instruction
// implies that a particular value is true (or false). The VRPSolver is
// responsible for analyzing the variable and seeing what new inferences
// can be made from each property. For example:
//
//   %P = seteq int* %ptr, null
//   %a = or bool %P, %Q
//   br bool %a label %cond_true, label %cond_false
//
// For the true branch, the VRPSolver will start with %a EQ true and look at
// the definition of %a and find that it can infer that %P and %Q are both
// true. From %P being true, it can infer that %ptr NE null. For the false
// branch it can't infer anything from the "or" instruction.
//
// Besides branches, we can also infer properties from instruction that may
// have undefined behaviour in certain cases. For example, the dividend of
// a division may never be zero. After the division instruction, we may assume
// that the dividend is not equal to zero.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "predsimplify"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ET-Forest.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <deque>
#include <iostream>
#include <sstream>
#include <map>
using namespace llvm;

namespace {
  Statistic
  NumVarsReplaced("predsimplify", "Number of argument substitutions");
  Statistic
  NumInstruction("predsimplify", "Number of instructions removed");
  Statistic
  NumSimple("predsimplify", "Number of simple replacements");

  /// The InequalityGraph stores the relationships between values.
  /// Each Value in the graph is assigned to a Node. Nodes are pointer
  /// comparable for equality. The caller is expected to maintain the logical
  /// consistency of the system.
  ///
  /// The InequalityGraph class may invalidate Node*s after any mutator call.
  /// @brief The InequalityGraph stores the relationships between values.
  class VISIBILITY_HIDDEN InequalityGraph {
  public:
    class Node;

    // LT GT EQ
    //  0  0  0 -- invalid (false)
    //  0  0  1 -- invalid (EQ)
    //  0  1  0 -- GT
    //  0  1  1 -- GE
    //  1  0  0 -- LT
    //  1  0  1 -- LE
    //  1  1  0 -- NE
    //  1  1  1 -- invalid (true)
    enum LatticeBits {
      EQ_BIT = 1, GT_BIT = 2, LT_BIT = 4
    };
    enum LatticeVal {
      GT = GT_BIT, GE = GT_BIT | EQ_BIT,
      LT = LT_BIT, LE = LT_BIT | EQ_BIT,
      NE = GT_BIT | LT_BIT
    };

    static bool validPredicate(LatticeVal LV) {
      return LV > 1 && LV < 7;
    }

  private:
    typedef std::map<Value *, Node *> NodeMapType;
    NodeMapType Nodes;

    const InequalityGraph *ConcreteIG;

  public:
    /// A single node in the InequalityGraph. This stores the canonical Value
    /// for the node, as well as the relationships with the neighbours.
    ///
    /// Because the lists are intended to be used for traversal, it is invalid
    /// for the node to list itself in LessEqual or GreaterEqual lists. The
    /// fact that a node is equal to itself is implied, and may be checked
    /// with pointer comparison.
    /// @brief A single node in the InequalityGraph.
    class VISIBILITY_HIDDEN Node {
      friend class InequalityGraph;

      Value *Canonical;

      typedef SmallVector<std::pair<Node *, LatticeVal>, 4> RelationsType;
      RelationsType Relations;
    public:
      typedef RelationsType::iterator       iterator;
      typedef RelationsType::const_iterator const_iterator;

    private:
      /// Updates the lattice value for a given node. Create a new entry if
      /// one doesn't exist, otherwise it merges the values. The new lattice
      /// value must not be inconsistent with any previously existing value.
      void update(Node *N, LatticeVal R) {
        iterator I = find(N);
        if (I == end()) {
          Relations.push_back(std::make_pair(N, R));
        } else {
          I->second = static_cast<LatticeVal>(I->second & R);
          assert(validPredicate(I->second) &&
                 "Invalid union of lattice values.");
        }
      }

      void assign(Node *N, LatticeVal R) {
        iterator I = find(N);
        if (I != end()) I->second = R;

        Relations.push_back(std::make_pair(N, R));
      }

    public:
      iterator begin()       { return Relations.begin(); }
      iterator end()         { return Relations.end();   }
      iterator find(Node *N) {
        iterator I = begin();
        for (iterator E = end(); I != E; ++I)
          if (I->first == N) break;
        return I;
      }

      const_iterator begin()       const { return Relations.begin(); }
      const_iterator end()         const { return Relations.end();   }
      const_iterator find(Node *N) const {
        const_iterator I = begin();
        for (const_iterator E = end(); I != E; ++I)
          if (I->first == N) break;
        return I;
      }

      unsigned findIndex(Node *N) {
        unsigned i = 0;
        iterator I = begin();
        for (iterator E = end(); I != E; ++I, ++i)
          if (I->first == N) return i;
        return (unsigned)-1;
      }

      void erase(iterator i) { Relations.erase(i); }

      Value *getValue() const { return Canonical; }
      void setValue(Value *V) { Canonical = V; }

      void addNotEqual(Node *N)     { update(N, NE); }
      void addLess(Node *N)         { update(N, LT); }
      void addLessEqual(Node *N)    { update(N, LE); }
      void addGreater(Node *N)      { update(N, GT); }
      void addGreaterEqual(Node *N) { update(N, GE); }
    };

    InequalityGraph() : ConcreteIG(NULL) {}

    InequalityGraph(const InequalityGraph &_IG) {
#if 0 // disable COW
      if (_IG.ConcreteIG) ConcreteIG = _IG.ConcreteIG;
      else ConcreteIG = &_IG;
#else
      ConcreteIG = &_IG;
      materialize();
#endif
    }

    ~InequalityGraph();

  private:
    void materialize();

  public:
    /// If the Value is in the graph, return the canonical form. Otherwise,
    /// return the original Value.
    Value *canonicalize(Value *V) const {  
      if (const Node *N = getNode(V))
        return N->getValue();
      else 
        return V;
    }

    /// Returns the node currently representing Value V, or null if no such
    /// node exists.
    Node *getNode(Value *V) {
      materialize();

      NodeMapType::const_iterator I = Nodes.find(V);
      return (I != Nodes.end()) ? I->second : 0;
    }

    const Node *getNode(Value *V) const {
      if (ConcreteIG) return ConcreteIG->getNode(V);

      NodeMapType::const_iterator I = Nodes.find(V);
      return (I != Nodes.end()) ? I->second : 0;
    }

    Node *getOrInsertNode(Value *V) {
      if (Node *N = getNode(V))
        return N;
      else
        return newNode(V);
    }

    Node *newNode(Value *V) {
      //DEBUG(std::cerr << "new node: " << *V << "\n");
      materialize();
      Node *&N = Nodes[V];
      assert(N == 0 && "Node already exists for value.");
      N = new Node();
      N->setValue(V);
      return N;
    }

    /// Returns true iff the nodes are provably inequal.
    bool isNotEqual(const Node *N1, const Node *N2) const {
      if (N1 == N2) return false;
      for (Node::const_iterator I = N1->begin(), E = N1->end(); I != E; ++I) {
        if (I->first == N2)
          return (I->second & EQ_BIT) == 0;
      }
      return isLess(N1, N2) || isGreater(N1, N2);
    }

    /// Returns true iff N1 is provably less than N2.
    bool isLess(const Node *N1, const Node *N2) const {
      if (N1 == N2) return false;
      for (Node::const_iterator I = N2->begin(), E = N2->end(); I != E; ++I) {
        if (I->first == N1)
          return I->second == LT;
      }
      for (Node::const_iterator I = N2->begin(), E = N2->end(); I != E; ++I) {
        if ((I->second & (LT_BIT | GT_BIT)) == LT_BIT)
          if (isLess(N1, I->first)) return true;
      }
      return false;
    }

    /// Returns true iff N1 is provably less than or equal to N2.
    bool isLessEqual(const Node *N1, const Node *N2) const {
      if (N1 == N2) return true;
      for (Node::const_iterator I = N2->begin(), E = N2->end(); I != E; ++I) {
        if (I->first == N1)
          return (I->second & (LT_BIT | GT_BIT)) == LT_BIT;
      }
      for (Node::const_iterator I = N2->begin(), E = N2->end(); I != E; ++I) {
        if ((I->second & (LT_BIT | GT_BIT)) == LT_BIT)
          if (isLessEqual(N1, I->first)) return true;
      }
      return false;
    }

    /// Returns true iff N1 is provably greater than N2.
    bool isGreater(const Node *N1, const Node *N2) const {
      return isLess(N2, N1);
    }

    /// Returns true iff N1 is provably greater than or equal to N2.
    bool isGreaterEqual(const Node *N1, const Node *N2) const {
      return isLessEqual(N2, N1);
    }

    // The add* methods assume that your input is logically valid and may 
    // assertion-fail or infinitely loop if you attempt a contradiction.

    void addEqual(Node *N, Value *V) {
      materialize();
      Nodes[V] = N;
    }

    void addNotEqual(Node *N1, Node *N2) {
      assert(N1 != N2 && "A node can't be inequal to itself.");
      materialize();
      N1->addNotEqual(N2);
      N2->addNotEqual(N1);
    }

    /// N1 is less than N2.
    void addLess(Node *N1, Node *N2) {
      assert(N1 != N2 && !isLess(N2, N1) && "Attempt to create < cycle.");
      materialize();
      N2->addLess(N1);
      N1->addGreater(N2);
    }

    /// N1 is less than or equal to N2.
    void addLessEqual(Node *N1, Node *N2) {
      assert(N1 != N2 && "Nodes are equal. Use mergeNodes instead.");
      assert(!isGreater(N1, N2) && "Impossible: Adding x <= y when x > y.");
      materialize();
      N2->addLessEqual(N1);
      N1->addGreaterEqual(N2);
    }

    /// Find the transitive closure starting at a node walking down the edges
    /// of type Val. Type Inserter must be an inserter that accepts Node *.
    template <typename Inserter>
    void transitiveClosure(Node *N, LatticeVal Val, Inserter insert) {
      for (Node::iterator I = N->begin(), E = N->end(); I != E; ++I) {
        if (I->second == Val) {
          *insert = I->first;
          transitiveClosure(I->first, Val, insert);
        }
      }
    }

    /// Kills off all the nodes in Kill by replicating their properties into
    /// node N. The elements of Kill must be unique. After merging, N's new
    /// canonical value is NewCanonical. Type C must be a container of Node *.
    template <typename C>
    void mergeNodes(Node *N, C &Kill, Value *NewCanonical);

    /// Removes a Value from the graph, but does not delete any nodes. As this
    /// method does not delete Nodes, V may not be the canonical choice for
    /// any node.
    void remove(Value *V) {
      materialize();

      for (NodeMapType::iterator I = Nodes.begin(), E = Nodes.end(); I != E;) {
        NodeMapType::iterator J = I++;
        assert(J->second->getValue() != V && "Can't delete canonical choice.");
        if (J->first == V) Nodes.erase(J);
      }
    }

#ifndef NDEBUG
    void debug(std::ostream &os) const {
    std::set<Node *> VisitedNodes;
    for (NodeMapType::const_iterator I = Nodes.begin(), E = Nodes.end();
         I != E; ++I) {
      Node *N = I->second;
      os << *I->first << " == " << *N->getValue() << "\n";
      if (VisitedNodes.insert(N).second) {
        os << *N->getValue() << ":\n";
        for (Node::const_iterator NI = N->begin(), NE = N->end();
             NI != NE; ++NI) {
          static const std::string names[8] =
              { "00", "01", " <", "<=", " >", ">=", "!=", "07" };
          os << "  " << names[NI->second] << " "
             << *NI->first->getValue() << "\n";
        }
      }
    }
  }
#endif
  };

  InequalityGraph::~InequalityGraph() {
    if (ConcreteIG) return;

    std::vector<Node *> Remove;
    for (NodeMapType::iterator I = Nodes.begin(), E = Nodes.end();
         I != E; ++I) {
      if (I->first == I->second->getValue())
        Remove.push_back(I->second);
    }
    for (std::vector<Node *>::iterator I = Remove.begin(), E = Remove.end();
         I != E; ++I) {
      delete *I;
    }
  }

  template <typename C>
  void InequalityGraph::mergeNodes(Node *N, C &Kill, Value *NewCanonical) {
    materialize();

    // Merge the relationships from the members of Kill into N.
    for (typename C::iterator KI = Kill.begin(), KE = Kill.end();
         KI != KE; ++KI) {

      for (Node::iterator I = (*KI)->begin(), E = (*KI)->end(); I != E; ++I) {
        if (I->first == N) continue;

        Node::iterator NI = N->find(I->first);
        if (NI == N->end()) {
          N->Relations.push_back(std::make_pair(I->first, I->second));
        } else {
          unsigned char LV = NI->second & I->second;
          if (LV == EQ_BIT) {

            assert(std::find(Kill.begin(), Kill.end(), I->first) != Kill.end()
                    && "Lost EQ property.");
            N->erase(NI);
          } else {
            NI->second = static_cast<LatticeVal>(LV);
            assert(InequalityGraph::validPredicate(NI->second) &&
                   "Invalid union of lattice values.");
          }
        }

        // All edges are reciprocal; every Node that Kill points to also
        // contains a pointer to Kill. Replace those with pointers with N.
        unsigned iter = I->first->findIndex(*KI);
        assert(iter != (unsigned)-1 && "Edge not reciprocal.");
        I->first->assign(N, (I->first->begin()+iter)->second);
        I->first->erase(I->first->begin()+iter);
      }

      // Removing references from N to Kill.
      Node::iterator NI = N->find(*KI);
      if (NI != N->end()) {
        N->erase(NI); // breaks reciprocity until Kill is deleted.
      }
    }

    N->setValue(NewCanonical);

    // Update value mapping to point to the merged node.
    for (NodeMapType::iterator I = Nodes.begin(), E = Nodes.end();
         I != E; ++I) {
      if (std::find(Kill.begin(), Kill.end(), I->second) != Kill.end())
        I->second = N;
    }

    for (typename C::iterator KI = Kill.begin(), KE = Kill.end();
         KI != KE; ++KI) {
      delete *KI;
    }
  }

  void InequalityGraph::materialize() {
    if (!ConcreteIG) return;
    const InequalityGraph *IG = ConcreteIG;
    ConcreteIG = NULL;

    for (NodeMapType::const_iterator I = IG->Nodes.begin(),
         E = IG->Nodes.end(); I != E; ++I) {
      if (I->first == I->second->getValue()) {
        Node *N = newNode(I->first);
        N->Relations.reserve(N->Relations.size());
      }
    }
    for (NodeMapType::const_iterator I = IG->Nodes.begin(),
         E = IG->Nodes.end(); I != E; ++I) {
      if (I->first != I->second->getValue()) {
        Nodes[I->first] = getNode(I->second->getValue());
      } else {
        Node *Old = I->second;
        Node *N = getNode(I->first);
        for (Node::const_iterator NI = Old->begin(), NE = Old->end();
             NI != NE; ++NI) {
          N->assign(getNode(NI->first->getValue()), NI->second);
        }
      }
    }
  }

  /// VRPSolver keeps track of how changes to one variable affect other
  /// variables, and forwards changes along to the InequalityGraph. It
  /// also maintains the correct choice for "canonical" in the IG.
  /// @brief VRPSolver calculates inferences from a new relationship.
  class VISIBILITY_HIDDEN VRPSolver {
  private:
    std::deque<Instruction *> WorkList;

    InequalityGraph &IG;
    const InequalityGraph &cIG;
    ETForest *Forest;
    ETNode *Top;

    typedef InequalityGraph::Node Node;

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

      if (!I1 || !I2) return false;

      BasicBlock *BB1 = I1->getParent(),
                 *BB2 = I2->getParent();
      if (BB1 == BB2) {
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

    void addToWorklist(Instruction *I) {
      //DEBUG(std::cerr << "addToWorklist: " << *I << "\n");

      if (!isa<BinaryOperator>(I) && !isa<SelectInst>(I)) return;

      const Type *Ty = I->getType();
      if (Ty == Type::VoidTy || Ty->isFPOrFPVector()) return;

      if (isInstructionTriviallyDead(I)) return;

      WorkList.push_back(I);
    }

    void addRecursive(Value *V) {
      //DEBUG(std::cerr << "addRecursive: " << *V << "\n");

      Instruction *I = dyn_cast<Instruction>(V);
      if (I)
        addToWorklist(I);
      else if (!isa<Argument>(V))
        return;

      //DEBUG(std::cerr << "addRecursive uses...\n");
      for (Value::use_iterator UI = V->use_begin(), UE = V->use_end();
           UI != UE; ++UI) {
        // Use must be either be dominated by Top, or dominate Top.
        if (Instruction *Inst = dyn_cast<Instruction>(*UI)) {
          ETNode *INode = Forest->getNodeForBlock(Inst->getParent());
          if (INode->DominatedBy(Top) || Top->DominatedBy(INode))
            addToWorklist(Inst);
        }
      }

      if (I) {
        //DEBUG(std::cerr << "addRecursive ops...\n");
        for (User::op_iterator OI = I->op_begin(), OE = I->op_end();
             OI != OE; ++OI) {
          if (Instruction *Inst = dyn_cast<Instruction>(*OI))
            addToWorklist(Inst);
        }
      }
      //DEBUG(std::cerr << "exit addRecursive (" << *V << ").\n");
    }

  public:
    VRPSolver(InequalityGraph &IG, ETForest *Forest, BasicBlock *TopBB)
      : IG(IG), cIG(IG), Forest(Forest), Top(Forest->getNodeForBlock(TopBB)) {}

    bool isEqual(Value *V1, Value *V2) const {
      if (V1 == V2) return true;
      if (const Node *N1 = cIG.getNode(V1))
        return N1 == cIG.getNode(V2);
      return false;
    }

    bool isNotEqual(Value *V1, Value *V2) const {
      if (V1 == V2) return false;
      if (const Node *N1 = cIG.getNode(V1))
        if (const Node *N2 = cIG.getNode(V2))
          return cIG.isNotEqual(N1, N2);
      return false;
    }

    bool isLess(Value *V1, Value *V2) const {
      if (V1 == V2) return false;
      if (const Node *N1 = cIG.getNode(V1))
        if (const Node *N2 = cIG.getNode(V2))
          return cIG.isLess(N1, N2);
      return false;
    }

    bool isLessEqual(Value *V1, Value *V2) const {
      if (V1 == V2) return true;
      if (const Node *N1 = cIG.getNode(V1))
        if (const Node *N2 = cIG.getNode(V2))
          return cIG.isLessEqual(N1, N2);
      return false;
    }

    bool isGreater(Value *V1, Value *V2) const {
      if (V1 == V2) return false;
      if (const Node *N1 = cIG.getNode(V1))
        if (const Node *N2 = cIG.getNode(V2))
          return cIG.isGreater(N1, N2);
      return false;
    }

    bool isGreaterEqual(Value *V1, Value *V2) const {
      if (V1 == V2) return true;
      if (const Node *N1 = IG.getNode(V1))
        if (const Node *N2 = IG.getNode(V2))
          return cIG.isGreaterEqual(N1, N2);
      return false;
    }

    // All of the add* functions return true if the InequalityGraph represents
    // the property, and false if there is a logical contradiction. On false,
    // you may no longer perform any queries on the InequalityGraph.

    bool addEqual(Value *V1, Value *V2) {
      //DEBUG(std::cerr << "addEqual(" << *V1 << ", "
      //                               << *V2 << ")\n");
      if (isEqual(V1, V2)) return true;

      const Node *cN1 = cIG.getNode(V1), *cN2 = cIG.getNode(V2);

      if (cN1 && cN2 && cIG.isNotEqual(cN1, cN2))
          return false;

      if (compare(V2, V1)) { std::swap(V1, V2); std::swap(cN1, cN2); }

      if (cN1) {
        if (ConstantBool *CB = dyn_cast<ConstantBool>(V1)) {
          Node *N1 = IG.getNode(V1);
           
          // When "addEqual" is performed and the new value is a ConstantBool,
          // iterate through the NE set and fix them up to be EQ of the
          // opposite bool.

          for (Node::iterator I = N1->begin(), E = N1->end(); I != E; ++I)
            if ((I->second & 1) == 0) {
              assert(N1 != I->first && "Node related to itself?");
              addEqual(I->first->getValue(),
                       ConstantBool::get(!CB->getValue()));
            }
        }
      }

      if (!cN2) {
        if (Instruction *I2 = dyn_cast<Instruction>(V2)) {
          ETNode *Node_I2 = Forest->getNodeForBlock(I2->getParent());
          if (Top != Node_I2 && Node_I2->DominatedBy(Top)) {
            Value *V = V1;
            if (cN1 && compare(V1, cN1->getValue())) V = cN1->getValue();
            //DEBUG(std::cerr << "Simply removing " << *I2
            //                << ", replacing with " << *V << "\n");
            I2->replaceAllUsesWith(V);
            // leave it dead; it'll get erased later.
            ++NumSimple;
            addRecursive(V1);
            return true;
          }
        }
      }

      Node *N1 = IG.getNode(V1), *N2 = IG.getNode(V2);

      if ( N1 && !N2) {
        IG.addEqual(N1, V2);
        if (compare(V1, N1->getValue())) N1->setValue(V1);
      }
      if (!N1 &&  N2) {
        IG.addEqual(N2, V1);
        if (compare(V1, N2->getValue())) N2->setValue(V1);
      }
      if ( N1 &&  N2) {
        // Suppose we're being told that %x == %y, and %x <= %z and %y >= %z.
        // We can't just merge %x and %y because the relationship with %z would
        // be EQ and that's invalid; they need to be the same Node.
        //
        // What we're doing is looking for any chain of nodes reaching %z such
        // that %x <= %z and %y >= %z, and vice versa. The cool part is that
        // every node in between is also equal because of the squeeze principle.

        std::vector<Node *> N1_GE, N2_LE, N1_LE, N2_GE;
        IG.transitiveClosure(N1, InequalityGraph::GE, back_inserter(N1_GE));
        std::sort(N1_GE.begin(), N1_GE.end());
        N1_GE.erase(std::unique(N1_GE.begin(), N1_GE.end()), N1_GE.end());
        IG.transitiveClosure(N2, InequalityGraph::LE, back_inserter(N2_LE));
        std::sort(N1_LE.begin(), N1_LE.end());
        N1_LE.erase(std::unique(N1_LE.begin(), N1_LE.end()), N1_LE.end());
        IG.transitiveClosure(N1, InequalityGraph::LE, back_inserter(N1_LE));
        std::sort(N2_GE.begin(), N2_GE.end());
        N2_GE.erase(std::unique(N2_GE.begin(), N2_GE.end()), N2_GE.end());
        std::unique(N2_GE.begin(), N2_GE.end());
        IG.transitiveClosure(N2, InequalityGraph::GE, back_inserter(N2_GE));
        std::sort(N2_LE.begin(), N2_LE.end());
        N2_LE.erase(std::unique(N2_LE.begin(), N2_LE.end()), N2_LE.end());

        std::vector<Node *> Set1, Set2;
        std::set_intersection(N1_GE.begin(), N1_GE.end(),
                              N2_LE.begin(), N2_LE.end(),
                              back_inserter(Set1));
        std::set_intersection(N1_LE.begin(), N1_LE.end(),
                              N2_GE.begin(), N2_GE.end(),
                              back_inserter(Set2));

        std::vector<Node *> Equal;
        std::set_union(Set1.begin(), Set1.end(), Set2.begin(), Set2.end(),
                       back_inserter(Equal));

        Value *Best = N1->getValue();
        if (compare(N2->getValue(), Best)) Best = N2->getValue();

        for (std::vector<Node *>::iterator I = Equal.begin(), E = Equal.end();
             I != E; ++I) {
          Value *V = (*I)->getValue();
          if (compare(V, Best)) Best = V;
        }

        Equal.push_back(N2);
        IG.mergeNodes(N1, Equal, Best);
      }
      if (!N1 && !N2) IG.addEqual(IG.newNode(V1), V2);

      addRecursive(V1);
      addRecursive(V2);

      return true;
    }

    bool addNotEqual(Value *V1, Value *V2) {
      //DEBUG(std::cerr << "addNotEqual(" << *V1 << ", "
      //                                  << *V2 << ")\n");
      if (isNotEqual(V1, V2)) return true;

      // Never permit %x NE true/false.
      if (ConstantBool *B1 = dyn_cast<ConstantBool>(V1)) {
        return addEqual(ConstantBool::get(!B1->getValue()), V2);
      } else if (ConstantBool *B2 = dyn_cast<ConstantBool>(V2)) {
        return addEqual(V1, ConstantBool::get(!B2->getValue()));
      }

      Node *N1 = IG.getOrInsertNode(V1),
           *N2 = IG.getOrInsertNode(V2);

      if (N1 == N2) return false;

      IG.addNotEqual(N1, N2);

      addRecursive(V1);
      addRecursive(V2);

      return true;
    }

    /// Set V1 less than V2.
    bool addLess(Value *V1, Value *V2) {
      if (isLess(V1, V2)) return true;
      if (isGreaterEqual(V1, V2)) return false;

      Node *N1 = IG.getOrInsertNode(V1), *N2 = IG.getOrInsertNode(V2);

      if (N1 == N2) return false;

      IG.addLess(N1, N2);

      addRecursive(V1);
      addRecursive(V2);

      return true;
    }

    /// Set V1 less than or equal to V2.
    bool addLessEqual(Value *V1, Value *V2) {
      if (isLessEqual(V1, V2)) return true;
      if (V1 == V2) return true;

      if (isLessEqual(V2, V1))
        return addEqual(V1, V2);

      if (isGreater(V1, V2)) return false;

      Node *N1 = IG.getOrInsertNode(V1),
           *N2 = IG.getOrInsertNode(V2);

      if (N1 == N2) return true;

      IG.addLessEqual(N1, N2);

      addRecursive(V1);
      addRecursive(V2);

      return true;
    }

    void solve() {
      DEBUG(std::cerr << "WorkList entry, size: " << WorkList.size() << "\n");
      while (!WorkList.empty()) {
        DEBUG(std::cerr << "WorkList size: " << WorkList.size() << "\n");

        Instruction *I = WorkList.front();
        WorkList.pop_front();

        Value *Canonical = cIG.canonicalize(I);
        const Type *Ty = I->getType();

        //DEBUG(std::cerr << "solving: " << *I << "\n");
        //DEBUG(IG.debug(std::cerr));

        if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
          Value *Op0 = cIG.canonicalize(BO->getOperand(0)),
                *Op1 = cIG.canonicalize(BO->getOperand(1));

          ConstantIntegral *CI1 = dyn_cast<ConstantIntegral>(Op0),
                           *CI2 = dyn_cast<ConstantIntegral>(Op1);

          if (CI1 && CI2)
            addEqual(BO, ConstantExpr::get(BO->getOpcode(), CI1, CI2));

          switch (BO->getOpcode()) {
            case Instruction::SetEQ:
              // "seteq int %a, %b" EQ true  then %a EQ %b
              // "seteq int %a, %b" EQ false then %a NE %b
              if (Canonical == ConstantBool::getTrue())
                addEqual(Op0, Op1);
              else if (Canonical == ConstantBool::getFalse())
                addNotEqual(Op0, Op1);

              // %a EQ %b then "seteq int %a, %b" EQ true
              // %a NE %b then "seteq int %a, %b" EQ false
              if (isEqual(Op0, Op1))
                addEqual(BO, ConstantBool::getTrue());
              else if (isNotEqual(Op0, Op1))
                addEqual(BO, ConstantBool::getFalse());

              break;
            case Instruction::SetNE:
              // "setne int %a, %b" EQ true  then %a NE %b
              // "setne int %a, %b" EQ false then %a EQ %b
              if (Canonical == ConstantBool::getTrue())
                addNotEqual(Op0, Op1);
              else if (Canonical == ConstantBool::getFalse())
                addEqual(Op0, Op1);

              // %a EQ %b then "setne int %a, %b" EQ false
              // %a NE %b then "setne int %a, %b" EQ true
              if (isEqual(Op0, Op1))
                addEqual(BO, ConstantBool::getFalse());
              else if (isNotEqual(Op0, Op1))
                addEqual(BO, ConstantBool::getTrue());

              break;
            case Instruction::SetLT:
              // "setlt int %a, %b" EQ true  then %a LT %b
              // "setlt int %a, %b" EQ false then %b LE %a
              if (Canonical == ConstantBool::getTrue())
                addLess(Op0, Op1);
              else if (Canonical == ConstantBool::getFalse())
                addLessEqual(Op1, Op0);

              // %a LT %b then "setlt int %a, %b" EQ true
              // %a GE %b then "setlt int %a, %b" EQ false
              if (isLess(Op0, Op1))
                addEqual(BO, ConstantBool::getTrue());
              else if (isGreaterEqual(Op0, Op1))
                addEqual(BO, ConstantBool::getFalse());

              break;
            case Instruction::SetLE:
              // "setle int %a, %b" EQ true  then %a LE %b
              // "setle int %a, %b" EQ false then %b LT %a
              if (Canonical == ConstantBool::getTrue())
                addLessEqual(Op0, Op1);
              else if (Canonical == ConstantBool::getFalse())
                addLess(Op1, Op0);

              // %a LE %b then "setle int %a, %b" EQ true
              // %a GT %b then "setle int %a, %b" EQ false
              if (isLessEqual(Op0, Op1))
                addEqual(BO, ConstantBool::getTrue());
              else if (isGreater(Op0, Op1))
                addEqual(BO, ConstantBool::getFalse());

              break;
            case Instruction::SetGT:
              // "setgt int %a, %b" EQ true  then %b LT %a
              // "setgt int %a, %b" EQ false then %a LE %b
              if (Canonical == ConstantBool::getTrue())
                addLess(Op1, Op0);
              else if (Canonical == ConstantBool::getFalse())
                addLessEqual(Op0, Op1);

              // %a GT %b then "setgt int %a, %b" EQ true
              // %a LE %b then "setgt int %a, %b" EQ false
              if (isGreater(Op0, Op1))
                addEqual(BO, ConstantBool::getTrue());
              else if (isLessEqual(Op0, Op1))
                addEqual(BO, ConstantBool::getFalse());

              break;
            case Instruction::SetGE:
              // "setge int %a, %b" EQ true  then %b LE %a
              // "setge int %a, %b" EQ false then %a LT %b
              if (Canonical == ConstantBool::getTrue())
                addLessEqual(Op1, Op0);
              else if (Canonical == ConstantBool::getFalse())
                addLess(Op0, Op1);

              // %a GE %b then "setge int %a, %b" EQ true
              // %a LT %b then "setlt int %a, %b" EQ false
              if (isGreaterEqual(Op0, Op1))
                addEqual(BO, ConstantBool::getTrue());
              else if (isLess(Op0, Op1))
                addEqual(BO, ConstantBool::getFalse());

              break;
            case Instruction::And: {
              // "and int %a, %b"  EQ -1   then %a EQ -1   and %b EQ -1
              // "and bool %a, %b" EQ true then %a EQ true and %b EQ true
              ConstantIntegral *CI = ConstantIntegral::getAllOnesValue(Ty);
              if (Canonical == CI) {
                addEqual(CI, Op0);
                addEqual(CI, Op1);
              }
            } break;
            case Instruction::Or: {
              // "or int %a, %b"  EQ 0     then %a EQ 0     and %b EQ 0
              // "or bool %a, %b" EQ false then %a EQ false and %b EQ false
              Constant *Zero = Constant::getNullValue(Ty);
              if (Canonical == Zero) {
                addEqual(Zero, Op0);
                addEqual(Zero, Op1);
              }
            } break;
            case Instruction::Xor: {
              // "xor bool true,  %a" EQ true  then %a EQ false
              // "xor bool true,  %a" EQ false then %a EQ true
              // "xor bool false, %a" EQ true  then %a EQ true
              // "xor bool false, %a" EQ false then %a EQ false
              // "xor int %c, %a" EQ %c then %a EQ 0
              // "xor int %c, %a" NE %c then %a NE 0
              // 1. Repeat all of the above, with order of operands reversed.
              Value *LHS = Op0, *RHS = Op1;
              if (!isa<Constant>(LHS)) std::swap(LHS, RHS);

              if (ConstantBool *CB = dyn_cast<ConstantBool>(Canonical)) {
                if (ConstantBool *A = dyn_cast<ConstantBool>(LHS))
                  addEqual(RHS, ConstantBool::get(A->getValue() ^
                                                  CB->getValue()));
              }
              if (Canonical == LHS) {
                if (isa<ConstantIntegral>(Canonical))
                  addEqual(RHS, Constant::getNullValue(Ty));
              } else if (isNotEqual(LHS, Canonical)) {
                addNotEqual(RHS, Constant::getNullValue(Ty));
              }
            } break;
            default:
              break;
          }

          // "%x = add int %y, %z" and %x EQ %y then %z EQ 0
          // "%x = mul int %y, %z" and %x EQ %y then %z EQ 1
          // 1. Repeat all of the above, with order of operands reversed.
          // "%x = fdiv float %y, %z" and %x EQ %y then %z EQ 1
          Value *Known = Op0, *Unknown = Op1;
          if (Known != BO) std::swap(Known, Unknown);
          if (Known == BO) {
            switch (BO->getOpcode()) {
              default: break;
              case Instruction::Xor:
              case Instruction::Or:
              case Instruction::Add:
              case Instruction::Sub:
                if (!Ty->isFloatingPoint())
                  addEqual(Unknown, Constant::getNullValue(Ty));
                break;
              case Instruction::UDiv:
              case Instruction::SDiv:
              case Instruction::FDiv:
                if (Unknown == Op0) break; // otherwise, fallthrough
              case Instruction::And:
              case Instruction::Mul:
                Constant *One = NULL;
                if (isa<ConstantInt>(Unknown))
                  One = ConstantInt::get(Ty, 1);
                else if (isa<ConstantFP>(Unknown))
                  One = ConstantFP::get(Ty, 1);
                else if (isa<ConstantBool>(Unknown))
                  One = ConstantBool::getTrue();

                if (One) addEqual(Unknown, One);
                break;
            }
          }
        } else if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
          // Given: "%a = select bool %x, int %b, int %c"
          // %a EQ %b then %x EQ true
          // %a EQ %c then %x EQ false
          if (isEqual(I, SI->getTrueValue()) ||
              isNotEqual(I, SI->getFalseValue()))
            addEqual(SI->getCondition(), ConstantBool::getTrue());
          else if (isEqual(I, SI->getFalseValue()) ||
                   isNotEqual(I, SI->getTrueValue()))
            addEqual(SI->getCondition(), ConstantBool::getFalse());

          // %x EQ true  then %a EQ %b
          // %x EQ false then %a NE %b
          if (isEqual(SI->getCondition(), ConstantBool::getTrue()))
            addEqual(SI, SI->getTrueValue());
          else if (isEqual(SI->getCondition(), ConstantBool::getFalse()))
            addEqual(SI, SI->getFalseValue());
        }
      }
    }
  };

  /// PredicateSimplifier - This class is a simplifier that replaces
  /// one equivalent variable with another. It also tracks what
  /// can't be equal and will solve setcc instructions when possible.
  /// @brief Root of the predicate simplifier optimization.
  class VISIBILITY_HIDDEN PredicateSimplifier : public FunctionPass {
    DominatorTree *DT;
    ETForest *Forest;
    bool modified;

    class State {
    public:
      BasicBlock *ToVisit;
      InequalityGraph *IG;

      State(BasicBlock *BB, InequalityGraph *IG) : ToVisit(BB), IG(IG) {}
    };

    std::vector<State> WorkList;

  public:
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addRequired<DominatorTree>();
      AU.addRequired<ETForest>();
      AU.setPreservesCFG();
      AU.addPreservedID(BreakCriticalEdgesID);
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

    public:
      InequalityGraph &IG;

      Forwards(PredicateSimplifier *PS, InequalityGraph &IG)
        : PS(PS), IG(IG) {}

      void visitTerminatorInst(TerminatorInst &TI);
      void visitBranchInst(BranchInst &BI);
      void visitSwitchInst(SwitchInst &SI);

      void visitAllocaInst(AllocaInst &AI);
      void visitLoadInst(LoadInst &LI);
      void visitStoreInst(StoreInst &SI);

      void visitBinaryOperator(BinaryOperator &BO);
    };

    // Used by terminator instructions to proceed from the current basic
    // block to the next. Verifies that "current" dominates "next",
    // then calls visitBasicBlock.
    void proceedToSuccessors(const InequalityGraph &IG, BasicBlock *BBCurrent) {
      DominatorTree::Node *Current = DT->getNode(BBCurrent);
      for (DominatorTree::Node::iterator I = Current->begin(),
           E = Current->end(); I != E; ++I) {
        //visitBasicBlock((*I)->getBlock(), IG);
        WorkList.push_back(State((*I)->getBlock(), new InequalityGraph(IG)));
      }
    }

    void proceedToSuccessor(InequalityGraph *NextIG, BasicBlock *Next) {
      //visitBasicBlock(Next, NextIG);
      WorkList.push_back(State(Next, NextIG));
    }

    // Visits each instruction in the basic block.
    void visitBasicBlock(BasicBlock *BB, InequalityGraph &IG) {
     DEBUG(std::cerr << "Entering Basic Block: " << BB->getName() << "\n");
     for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
       visitInstruction(I++, IG);
      }
    }

    // Tries to simplify each Instruction and add new properties to
    // the PropertySet.
    void visitInstruction(Instruction *I, InequalityGraph &IG) {
      DEBUG(std::cerr << "Considering instruction " << *I << "\n");
      DEBUG(IG.debug(std::cerr));

      // Sometimes instructions are made dead due to earlier analysis.
      if (isInstructionTriviallyDead(I)) {
        I->eraseFromParent();
        return;
      }

      // Try to replace the whole instruction.
      Value *V = IG.canonicalize(I);
      if (V != I) {
        modified = true;
        ++NumInstruction;
        DEBUG(std::cerr << "Removing " << *I << ", replacing with "
                        << *V << "\n");
        IG.remove(I);
        I->replaceAllUsesWith(V);
        I->eraseFromParent();
        return;
      }

      // Try to substitute operands.
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
        Value *Oper = I->getOperand(i);
        Value *V = IG.canonicalize(Oper);
        if (V != Oper) {
          modified = true;
          ++NumVarsReplaced;
          DEBUG(std::cerr << "Resolving " << *I);
          I->setOperand(i, V);
          DEBUG(std::cerr << " into " << *I);
        }
      }

      //DEBUG(std::cerr << "push (%" << I->getParent()->getName() << ")\n");
      Forwards visit(this, IG);
      visit.visit(*I);
      //DEBUG(std::cerr << "pop (%" << I->getParent()->getName() << ")\n");
    }
  };

  bool PredicateSimplifier::runOnFunction(Function &F) {
    DT = &getAnalysis<DominatorTree>();
    Forest = &getAnalysis<ETForest>();

    DEBUG(std::cerr << "Entering Function: " << F.getName() << "\n");

    modified = false;
    WorkList.push_back(State(DT->getRoot(), new InequalityGraph()));

    do {
      State S = WorkList.back();
      WorkList.pop_back();
      visitBasicBlock(S.ToVisit, *S.IG);
      delete S.IG;
    } while (!WorkList.empty());

    //DEBUG(F.viewCFG());

    return modified;
  }

  void PredicateSimplifier::Forwards::visitTerminatorInst(TerminatorInst &TI) {
    PS->proceedToSuccessors(IG, TI.getParent());
  }

  void PredicateSimplifier::Forwards::visitBranchInst(BranchInst &BI) {
    BasicBlock *BB = BI.getParent();

    if (BI.isUnconditional()) {
      PS->proceedToSuccessors(IG, BB);
      return;
    }

    Value *Condition = BI.getCondition();
    BasicBlock *TrueDest  = BI.getSuccessor(0),
               *FalseDest = BI.getSuccessor(1);

    if (isa<ConstantBool>(Condition) || TrueDest == FalseDest) {
      PS->proceedToSuccessors(IG, BB);
      return;
    }

    DominatorTree::Node *Node = PS->DT->getNode(BB);
    for (DominatorTree::Node::iterator I = Node->begin(), E = Node->end();
         I != E; ++I) {
      BasicBlock *Dest = (*I)->getBlock();
      InequalityGraph *DestProperties = new InequalityGraph(IG);
      VRPSolver Solver(*DestProperties, PS->Forest, Dest);

      if (Dest == TrueDest) {
        DEBUG(std::cerr << "(" << BB->getName() << ") true set:\n");
        if (!Solver.addEqual(ConstantBool::getTrue(), Condition)) continue;
        Solver.solve();
        DEBUG(DestProperties->debug(std::cerr));
      } else if (Dest == FalseDest) {
        DEBUG(std::cerr << "(" << BB->getName() << ") false set:\n");
        if (!Solver.addEqual(ConstantBool::getFalse(), Condition)) continue;
        Solver.solve();
        DEBUG(DestProperties->debug(std::cerr));
      }

      PS->proceedToSuccessor(DestProperties, Dest);
    }
  }

  void PredicateSimplifier::Forwards::visitSwitchInst(SwitchInst &SI) {
    Value *Condition = SI.getCondition();

    // Set the EQProperty in each of the cases BBs, and the NEProperties
    // in the default BB.
    // InequalityGraph DefaultProperties(IG);

    DominatorTree::Node *Node = PS->DT->getNode(SI.getParent());
    for (DominatorTree::Node::iterator I = Node->begin(), E = Node->end();
         I != E; ++I) {
      BasicBlock *BB = (*I)->getBlock();

      InequalityGraph *BBProperties = new InequalityGraph(IG);
      VRPSolver Solver(*BBProperties, PS->Forest, BB);
      if (BB == SI.getDefaultDest()) {
        for (unsigned i = 1, e = SI.getNumCases(); i < e; ++i)
          if (SI.getSuccessor(i) != BB)
            if (!Solver.addNotEqual(Condition, SI.getCaseValue(i))) continue;
        Solver.solve();
      } else if (ConstantInt *CI = SI.findCaseDest(BB)) {
        if (!Solver.addEqual(Condition, CI)) continue;
        Solver.solve();
      }
      PS->proceedToSuccessor(BBProperties, BB);
    }
  }

  void PredicateSimplifier::Forwards::visitAllocaInst(AllocaInst &AI) {
    VRPSolver VRP(IG, PS->Forest, AI.getParent());
    VRP.addNotEqual(Constant::getNullValue(AI.getType()), &AI);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitLoadInst(LoadInst &LI) {
    Value *Ptr = LI.getPointerOperand();
    // avoid "load uint* null" -> null NE null.
    if (isa<Constant>(Ptr)) return;

    VRPSolver VRP(IG, PS->Forest, LI.getParent());
    VRP.addNotEqual(Constant::getNullValue(Ptr->getType()), Ptr);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitStoreInst(StoreInst &SI) {
    Value *Ptr = SI.getPointerOperand();
    if (isa<Constant>(Ptr)) return;

    VRPSolver VRP(IG, PS->Forest, SI.getParent());
    VRP.addNotEqual(Constant::getNullValue(Ptr->getType()), Ptr);
    VRP.solve();
  }

  void PredicateSimplifier::Forwards::visitBinaryOperator(BinaryOperator &BO) {
    Instruction::BinaryOps ops = BO.getOpcode();

    switch (ops) {
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv: {
      Value *Divisor = BO.getOperand(1);
      VRPSolver VRP(IG, PS->Forest, BO.getParent());
      VRP.addNotEqual(Constant::getNullValue(Divisor->getType()), Divisor);
      VRP.solve();
      break;
    }
    default:
      break;
    }
  }


  RegisterPass<PredicateSimplifier> X("predsimplify",
                                      "Predicate Simplifier");
}

FunctionPass *llvm::createPredicateSimplifierPass() {
  return new PredicateSimplifier();
}
