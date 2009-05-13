//=-- ExplodedGraph.h - Local, Path-Sens. "Exploded Graph" -*- C++ -*-------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the template classes ExplodedNode and ExplodedGraph,
//  which represent a path-sensitive, intra-procedural "exploded graph."
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_EXPLODEDGRAPH
#define LLVM_CLANG_ANALYSIS_EXPLODEDGRAPH

#include "clang/Analysis/ProgramPoint.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/Casting.h"

namespace clang {

class GRCoreEngineImpl;
class ExplodedNodeImpl;
class CFG;
class ASTContext;

class GRStmtNodeBuilderImpl;
class GRBranchNodeBuilderImpl;
class GRIndirectGotoNodeBuilderImpl;
class GRSwitchNodeBuilderImpl;
class GREndPathNodebuilderImpl;  

//===----------------------------------------------------------------------===//
// ExplodedGraph "implementation" classes.  These classes are not typed to
// contain a specific kind of state.  Typed-specialized versions are defined
// on top of these classes.
//===----------------------------------------------------------------------===//
  
class ExplodedNodeImpl : public llvm::FoldingSetNode {
protected:
  friend class ExplodedGraphImpl;
  friend class GRCoreEngineImpl;
  friend class GRStmtNodeBuilderImpl;
  friend class GRBranchNodeBuilderImpl;
  friend class GRIndirectGotoNodeBuilderImpl;
  friend class GRSwitchNodeBuilderImpl;
  friend class GREndPathNodeBuilderImpl;  
  
  class NodeGroup {
    enum { Size1 = 0x0, SizeOther = 0x1, AuxFlag = 0x2, Mask = 0x3 };
    uintptr_t P;
    
    unsigned getKind() const {
      return P & 0x1;
    }
    
    void* getPtr() const {
      assert (!getFlag());
      return reinterpret_cast<void*>(P & ~Mask);
    }

    ExplodedNodeImpl* getNode() const {
      return reinterpret_cast<ExplodedNodeImpl*>(getPtr());
    }
    
  public:
    NodeGroup() : P(0) {}
    
    ~NodeGroup();
    
    ExplodedNodeImpl** begin() const;
    
    ExplodedNodeImpl** end() const;
    
    unsigned size() const;
    
    bool empty() const { return size() == 0; }
    
    void addNode(ExplodedNodeImpl* N);
    
    void setFlag() {
      assert (P == 0);
      P = AuxFlag;
    }
    
    bool getFlag() const {
      return P & AuxFlag ? true : false;
    }
  };  
  
  /// Location - The program location (within a function body) associated
  ///  with this node.
  const ProgramPoint Location;
  
  /// State - The state associated with this node.
  const void* State;
  
  /// Preds - The predecessors of this node.
  NodeGroup Preds;
  
  /// Succs - The successors of this node.
  NodeGroup Succs;
  
  /// Construct a ExplodedNodeImpl with the provided location and state.
  explicit ExplodedNodeImpl(const ProgramPoint& loc, const void* state)
  : Location(loc), State(state) {}
  
  /// addPredeccessor - Adds a predecessor to the current node, and 
  ///  in tandem add this node as a successor of the other node.
  void addPredecessor(ExplodedNodeImpl* V);
  
public:
  
  /// getLocation - Returns the edge associated with the given node.
  ProgramPoint getLocation() const { return Location; }
  
  template <typename T>
  const T* getLocationAs() const { return llvm::dyn_cast<T>(&Location); }
  
  unsigned succ_size() const { return Succs.size(); }
  unsigned pred_size() const { return Preds.size(); }
  bool succ_empty() const { return Succs.empty(); }
  bool pred_empty() const { return Preds.empty(); }
  
  bool isSink() const { return Succs.getFlag(); }
  void markAsSink() { Succs.setFlag(); } 
  
  // For debugging.
  
public:
  
  class Auditor {
  public:
    virtual ~Auditor();
    virtual void AddEdge(ExplodedNodeImpl* Src, ExplodedNodeImpl* Dst) = 0;
  };
  
  static void SetAuditor(Auditor* A);
};


template <typename StateTy>
struct GRTrait {
  static inline void Profile(llvm::FoldingSetNodeID& ID, const StateTy* St) {
    St->Profile(ID);
  }
};


template <typename StateTy>
class ExplodedNode : public ExplodedNodeImpl {
public:
  /// Construct a ExplodedNodeImpl with the given node ID, program edge,
  ///  and state.
  explicit ExplodedNode(const ProgramPoint& loc, const StateTy* St)
    : ExplodedNodeImpl(loc, St) {}
  
  /// getState - Returns the state associated with the node.  
  inline const StateTy* getState() const {
    return static_cast<const StateTy*>(State);
  }
  
  // Profiling (for FoldingSet).
  
  static inline void Profile(llvm::FoldingSetNodeID& ID,
                             const ProgramPoint& Loc,
                             const StateTy* state) {
    ID.Add(Loc);
    GRTrait<StateTy>::Profile(ID, state);
  }
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, getLocation(), getState());
  }
  
  void addPredecessor(ExplodedNode* V) {
    ExplodedNodeImpl::addPredecessor(V);
  }
  
  ExplodedNode* getFirstPred() {
    return pred_empty() ? NULL : *(pred_begin());
  }
  
  const ExplodedNode* getFirstPred() const {
    return const_cast<ExplodedNode*>(this)->getFirstPred();
  }
  
  // Iterators over successor and predecessor vertices.
  typedef ExplodedNode**       succ_iterator;
  typedef const ExplodedNode* const * const_succ_iterator;
  typedef ExplodedNode**       pred_iterator;
  typedef const ExplodedNode* const * const_pred_iterator;

  pred_iterator pred_begin() { return (ExplodedNode**) Preds.begin(); }
  pred_iterator pred_end() { return (ExplodedNode**) Preds.end(); }

  const_pred_iterator pred_begin() const {
    return const_cast<ExplodedNode*>(this)->pred_begin();
  }  
  const_pred_iterator pred_end() const {
    return const_cast<ExplodedNode*>(this)->pred_end();
  }

  succ_iterator succ_begin() { return (ExplodedNode**) Succs.begin(); }
  succ_iterator succ_end() { return (ExplodedNode**) Succs.end(); }

  const_succ_iterator succ_begin() const {
    return const_cast<ExplodedNode*>(this)->succ_begin();
  }
  const_succ_iterator succ_end() const {
    return const_cast<ExplodedNode*>(this)->succ_end();
  }
};

class InterExplodedGraphMapImpl;

class ExplodedGraphImpl {
protected:
  friend class GRCoreEngineImpl;
  friend class GRStmtNodeBuilderImpl;
  friend class GRBranchNodeBuilderImpl;
  friend class GRIndirectGotoNodeBuilderImpl;
  friend class GRSwitchNodeBuilderImpl;
  friend class GREndPathNodeBuilderImpl;
  
  // Type definitions.
  typedef llvm::SmallVector<ExplodedNodeImpl*,2>    RootsTy;
  typedef llvm::SmallVector<ExplodedNodeImpl*,10>   EndNodesTy;
    
  /// Roots - The roots of the simulation graph. Usually there will be only
  /// one, but clients are free to establish multiple subgraphs within a single
  /// SimulGraph. Moreover, these subgraphs can often merge when paths from
  /// different roots reach the same state at the same program location.
  RootsTy Roots;

  /// EndNodes - The nodes in the simulation graph which have been
  ///  specially marked as the endpoint of an abstract simulation path.
  EndNodesTy EndNodes;
  
  /// Allocator - BumpPtrAllocator to create nodes.
  llvm::BumpPtrAllocator Allocator;
  
  /// cfg - The CFG associated with this analysis graph.
  CFG& cfg;
  
  /// CodeDecl - The declaration containing the code being analyzed.  This
  ///  can be a FunctionDecl or and ObjCMethodDecl.
  Decl& CodeDecl;
  
  /// Ctx - The ASTContext used to "interpret" CodeDecl.
  ASTContext& Ctx;
  
  /// NumNodes - The number of nodes in the graph.
  unsigned NumNodes;

  /// getNodeImpl - Retrieve the node associated with a (Location,State)
  ///  pair, where 'State' is represented as an opaque void*.  This method
  ///  is intended to be used only by GRCoreEngineImpl.
  virtual ExplodedNodeImpl* getNodeImpl(const ProgramPoint& L,
                                        const void* State,
                                        bool* IsNew) = 0;
  
  virtual ExplodedGraphImpl* MakeEmptyGraph() const = 0;

  /// addRoot - Add an untyped node to the set of roots.
  ExplodedNodeImpl* addRoot(ExplodedNodeImpl* V) {
    Roots.push_back(V);
    return V;
  }

  /// addEndOfPath - Add an untyped node to the set of EOP nodes.
  ExplodedNodeImpl* addEndOfPath(ExplodedNodeImpl* V) {
    EndNodes.push_back(V);
    return V;
  }
  
  // ctor.
  ExplodedGraphImpl(CFG& c, Decl& cd, ASTContext& ctx)
    : cfg(c), CodeDecl(cd), Ctx(ctx), NumNodes(0) {}

public:
  virtual ~ExplodedGraphImpl() {}

  unsigned num_roots() const { return Roots.size(); }
  unsigned num_eops() const { return EndNodes.size(); }
  
  bool empty() const { return NumNodes == 0; }
  unsigned size() const { return NumNodes; }
  
  llvm::BumpPtrAllocator& getAllocator() { return Allocator; }
  CFG& getCFG() { return cfg; }
  ASTContext& getContext() { return Ctx; }

  Decl& getCodeDecl() { return CodeDecl; }
  const Decl& getCodeDecl() const { return CodeDecl; }

  const FunctionDecl* getFunctionDecl() const {
    return llvm::dyn_cast<FunctionDecl>(&CodeDecl);
  }
  
  typedef llvm::DenseMap<const ExplodedNodeImpl*, ExplodedNodeImpl*> NodeMap;

  ExplodedGraphImpl* Trim(const ExplodedNodeImpl* const * NBeg,
                          const ExplodedNodeImpl* const * NEnd,
                          InterExplodedGraphMapImpl *M,
                          llvm::DenseMap<const void*, const void*> *InverseMap)
                        const;
};
  
class InterExplodedGraphMapImpl {
  llvm::DenseMap<const ExplodedNodeImpl*, ExplodedNodeImpl*> M;
  friend class ExplodedGraphImpl;  
  void add(const ExplodedNodeImpl* From, ExplodedNodeImpl* To);
  
protected:
  ExplodedNodeImpl* getMappedImplNode(const ExplodedNodeImpl* N) const;
  
  InterExplodedGraphMapImpl();
public:
  virtual ~InterExplodedGraphMapImpl() {}
};
  
//===----------------------------------------------------------------------===//
// Type-specialized ExplodedGraph classes.
//===----------------------------------------------------------------------===//
  
template <typename STATE>
class InterExplodedGraphMap : public InterExplodedGraphMapImpl {
public:
  InterExplodedGraphMap() {};
  ~InterExplodedGraphMap() {};

  ExplodedNode<STATE>* getMappedNode(const ExplodedNode<STATE>* N) const {
    return static_cast<ExplodedNode<STATE>*>(getMappedImplNode(N));
  }
};
  
template <typename STATE>
class ExplodedGraph : public ExplodedGraphImpl {
public:
  typedef STATE                       StateTy;
  typedef ExplodedNode<StateTy>       NodeTy;  
  typedef llvm::FoldingSet<NodeTy>    AllNodesTy;
  
protected:  
  /// Nodes - The nodes in the graph.
  AllNodesTy Nodes;
  
protected:
  virtual ExplodedNodeImpl* getNodeImpl(const ProgramPoint& L,
                                        const void* State,
                                        bool* IsNew) {
    
    return getNode(L, static_cast<const StateTy*>(State), IsNew);
  }
  
  virtual ExplodedGraphImpl* MakeEmptyGraph() const {
    return new ExplodedGraph(cfg, CodeDecl, Ctx);
  }  
    
public:
  ExplodedGraph(CFG& c, Decl& cd, ASTContext& ctx)
    : ExplodedGraphImpl(c, cd, ctx) {}
  
  /// getNode - Retrieve the node associated with a (Location,State) pair,
  ///  where the 'Location' is a ProgramPoint in the CFG.  If no node for
  ///  this pair exists, it is created.  IsNew is set to true if
  ///  the node was freshly created.
  NodeTy* getNode(const ProgramPoint& L, const StateTy* State,
                  bool* IsNew = NULL) {
    
    // Profile 'State' to determine if we already have an existing node.
    llvm::FoldingSetNodeID profile;    
    void* InsertPos = 0;
    
    NodeTy::Profile(profile, L, State);
    NodeTy* V = Nodes.FindNodeOrInsertPos(profile, InsertPos);

    if (!V) {
      // Allocate a new node.
      V = (NodeTy*) Allocator.Allocate<NodeTy>();
      new (V) NodeTy(L, State);
      
      // Insert the node into the node set and return it.
      Nodes.InsertNode(V, InsertPos);
      
      ++NumNodes;
      
      if (IsNew) *IsNew = true;
    }
    else
      if (IsNew) *IsNew = false;

    return V;
  }
  
  // Iterators.
  typedef NodeTy**                            roots_iterator;
  typedef const NodeTy**                      const_roots_iterator;
  typedef NodeTy**                            eop_iterator;
  typedef const NodeTy**                      const_eop_iterator;
  typedef typename AllNodesTy::iterator       node_iterator;
  typedef typename AllNodesTy::const_iterator const_node_iterator;
  
  node_iterator nodes_begin() {
    return Nodes.begin();
  }

  node_iterator nodes_end() {
    return Nodes.end();
  }
  
  const_node_iterator nodes_begin() const {
    return Nodes.begin();
  }
  
  const_node_iterator nodes_end() const {
    return Nodes.end();
  }
  
  roots_iterator roots_begin() {
    return reinterpret_cast<roots_iterator>(Roots.begin());
  }
  
  roots_iterator roots_end() { 
    return reinterpret_cast<roots_iterator>(Roots.end());
  }
  
  const_roots_iterator roots_begin() const { 
    return const_cast<ExplodedGraph>(this)->roots_begin();
  }
  
  const_roots_iterator roots_end() const { 
    return const_cast<ExplodedGraph>(this)->roots_end();
  }  

  eop_iterator eop_begin() {
    return reinterpret_cast<eop_iterator>(EndNodes.begin());
  }
    
  eop_iterator eop_end() { 
    return reinterpret_cast<eop_iterator>(EndNodes.end());
  }
  
  const_eop_iterator eop_begin() const {
    return const_cast<ExplodedGraph>(this)->eop_begin();
  }
  
  const_eop_iterator eop_end() const {
    return const_cast<ExplodedGraph>(this)->eop_end();
  }
  
  std::pair<ExplodedGraph*, InterExplodedGraphMap<STATE>*>
  Trim(const NodeTy* const* NBeg, const NodeTy* const* NEnd,
       llvm::DenseMap<const void*, const void*> *InverseMap = 0) const {
    
    if (NBeg == NEnd)
      return std::make_pair((ExplodedGraph*) 0,
                            (InterExplodedGraphMap<STATE>*) 0);
    
    assert (NBeg < NEnd);
    
    const ExplodedNodeImpl* const* NBegImpl =
      (const ExplodedNodeImpl* const*) NBeg;
    const ExplodedNodeImpl* const* NEndImpl =
      (const ExplodedNodeImpl* const*) NEnd;
    
    llvm::OwningPtr<InterExplodedGraphMap<STATE> > 
      M(new InterExplodedGraphMap<STATE>());

    ExplodedGraphImpl* G = ExplodedGraphImpl::Trim(NBegImpl, NEndImpl, M.get(),
                                                   InverseMap);

    return std::make_pair(static_cast<ExplodedGraph*>(G), M.take());
  }
};

template <typename StateTy>
class ExplodedNodeSet {
  
  typedef ExplodedNode<StateTy>        NodeTy;
  typedef llvm::SmallPtrSet<NodeTy*,5> ImplTy;
  ImplTy Impl;
  
public:
  ExplodedNodeSet(NodeTy* N) {
    assert (N && !static_cast<ExplodedNodeImpl*>(N)->isSink());
    Impl.insert(N);
  }
  
  ExplodedNodeSet() {}
  
  inline void Add(NodeTy* N) {
    if (N && !static_cast<ExplodedNodeImpl*>(N)->isSink()) Impl.insert(N);
  }
  
  typedef typename ImplTy::iterator       iterator;
  typedef typename ImplTy::const_iterator const_iterator;

  inline unsigned size() const { return Impl.size();  }
  inline bool empty()    const { return Impl.empty(); }

  inline void clear() { Impl.clear(); }
  
  inline iterator begin() { return Impl.begin(); }
  inline iterator end()   { return Impl.end();   }
  
  inline const_iterator begin() const { return Impl.begin(); }
  inline const_iterator end()   const { return Impl.end();   }
};  
  
} // end clang namespace

// GraphTraits

namespace llvm {
  template<typename StateTy>
  struct GraphTraits<clang::ExplodedNode<StateTy>*> {
    typedef clang::ExplodedNode<StateTy>      NodeType;
    typedef typename NodeType::succ_iterator  ChildIteratorType;
    typedef llvm::df_iterator<NodeType*>      nodes_iterator;
    
    static inline NodeType* getEntryNode(NodeType* N) {
      return N;
    }
    
    static inline ChildIteratorType child_begin(NodeType* N) {
      return N->succ_begin();
    }
    
    static inline ChildIteratorType child_end(NodeType* N) {
      return N->succ_end();
    }
    
    static inline nodes_iterator nodes_begin(NodeType* N) {
      return df_begin(N);
    }
    
    static inline nodes_iterator nodes_end(NodeType* N) {
      return df_end(N);
    }
  };
  
  template<typename StateTy>
  struct GraphTraits<const clang::ExplodedNode<StateTy>*> {
    typedef const clang::ExplodedNode<StateTy> NodeType;
    typedef typename NodeType::succ_iterator   ChildIteratorType;
    typedef llvm::df_iterator<NodeType*>       nodes_iterator;
    
    static inline NodeType* getEntryNode(NodeType* N) {
      return N;
    }
    
    static inline ChildIteratorType child_begin(NodeType* N) {
      return N->succ_begin();
    }
    
    static inline ChildIteratorType child_end(NodeType* N) {
      return N->succ_end();
    }
    
    static inline nodes_iterator nodes_begin(NodeType* N) {
      return df_begin(N);
    }
    
    static inline nodes_iterator nodes_end(NodeType* N) {
      return df_end(N);
    }
  };
  
} // end llvm namespace

#endif
