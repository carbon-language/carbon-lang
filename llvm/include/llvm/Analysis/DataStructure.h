//===- DataStructure.h - Build data structure graphs -------------*- C++ -*--=//
//
// Implement the LLVM data structure analysis library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DATA_STRUCTURE_H
#define LLVM_ANALYSIS_DATA_STRUCTURE_H

#include "llvm/Pass.h"
#include <string>

class Type;
class CallInst;
class AllocationInst;
class Argument;
class DSNode;
class FunctionRepBuilder;
class GlobalValue;
class FunctionDSGraph;
class DataStructure;
class DSNodeIterator;
class ShadowDSNode;

// FIXME: move this somewhere private
unsigned countPointerFields(const Type *Ty);

// PointerVal - Represent a pointer to a datastructure.  The pointer points to
// a node, and can index into it.  This is used for getelementptr instructions,
// which do not affect which node a pointer points to, but does change the field
// index
//
struct PointerVal {
  DSNode *Node;
  unsigned Index;  // Index into Node->FieldLinks[]
public:
  PointerVal(DSNode *N, unsigned Idx = 0) : Node(N), Index(Idx) {}

  DSNode *getNode() const { return Node; }
  unsigned getIndex() const { return Index; }

  inline bool operator==(DSNode *N) const { return Node == N; }
  inline bool operator!=(DSNode *N) const { return Node != N; }

  // operator< - Allow insertion into a map...
  bool operator<(const PointerVal &PV) const {
    return Node < PV.Node || (Node == PV.Node && Index < PV.Index);
  }

  inline bool operator==(const PointerVal &PV) const {
    return Node == PV.Node && Index == PV.Index;
  }
  inline bool operator!=(const PointerVal &PV) const { return !operator==(PV); }

  void print(std::ostream &O) const;
};


// PointerValSet - This class represents a list of pointer values.  The add
// method is used to add values to the set, and ensures that duplicates cannot
// happen.
//
class PointerValSet {
  std::vector<PointerVal> Vals;
  void dropRefs();
  void addRefs();
public:
  PointerValSet() {}
  PointerValSet(const PointerValSet &PVS) : Vals(PVS.Vals) { addRefs(); }
  ~PointerValSet() { dropRefs(); }
  const PointerValSet &operator=(const PointerValSet &PVS);

  // operator< - Allow insertion into a map...
  bool operator<(const PointerValSet &PVS) const;
  bool operator==(const PointerValSet &PVS) const;

  const PointerVal &operator[](unsigned i) const { return Vals[i]; }

  unsigned size() const { return Vals.size(); }
  bool empty() const { return Vals.empty(); }
  void clear() { dropRefs(); Vals.clear(); }

  // add - Add the specified pointer, or contents of the specified PVS to this
  // pointer set.  If a 'Pointer' value is provided, notify the underlying data
  // structure node that the pointer is pointing to it, so that it can be
  // invalidated if neccesary later.  True is returned if the value is new to
  // this pointer.
  //
  bool add(const PointerVal &PV, Value *Pointer = 0);
  bool add(const PointerValSet &PVS, Value *Pointer = 0) {
    bool Changed = false;
    for (unsigned i = 0, e = PVS.size(); i != e; ++i)
      Changed |= add(PVS[i], Pointer);
    return Changed;
  }

  // removePointerTo - Remove a single pointer val that points to the specified
  // node...
  void removePointerTo(DSNode *Node);

  void print(std::ostream &O) const;
};


//===----------------------------------------------------------------------===//
// DSNode - Base class for all data structure nodes...
//
// This class keeps track of its type, the pointer fields in the data structure,
// and a list of LLVM values that are pointing to this node.
//
class DSNode {
  friend class FunctionDSGraph;
  const Type *Ty;
  std::vector<PointerValSet> FieldLinks;
  std::vector<Value*> Pointers;   // Values pointing to me...
  std::vector<PointerValSet*> Referrers;

  std::vector<std::pair<const Type *, ShadowDSNode *> > SynthNodes;
  
  DSNode(const DSNode &);         // DO NOT IMPLEMENT
  void operator=(const DSNode &); // DO NOT IMPLEMENT
public:
  enum NodeTy {
    NewNode, CallNode, ShadowNode, GlobalNode
  } NodeType;

  DSNode(enum NodeTy NT, const Type *T);
  virtual ~DSNode() {
    dropAllReferences();
    assert(Referrers.empty() && "Referrers to dead node exist!");
  }

  typedef DSNodeIterator iterator;
  inline iterator begin();   // Defined in DataStructureGraph.h
  inline iterator end();

  unsigned getNumLinks() const { return FieldLinks.size(); }
  PointerValSet &getLink(unsigned i) {
    assert(i < getNumLinks() && "Field links access out of range...");
    return FieldLinks[i];
  }
  const PointerValSet &getLink(unsigned i) const {
    assert(i < getNumLinks() && "Field links access out of range...");
    return FieldLinks[i];
  }

  // addReferrer - Keep the referrer set up to date...
  void addReferrer(PointerValSet *PVS) { Referrers.push_back(PVS); }
  void removeReferrer(PointerValSet *PVS);
  const std::vector<PointerValSet*> &getReferrers() const { return Referrers; }

  // removeAllIncomingEdges - Erase all edges in the graph that point to
  // this node
  void removeAllIncomingEdges();

  void addPointer(Value *V) { Pointers.push_back(V); }
  const std::vector<Value*> &getPointers() const { return Pointers; }

  const Type *getType() const { return Ty; }

  // getNumOutgoingLinks - Return the number of outgoing links, which is usually
  // the number of normal links, but for call nodes it also includes their
  // arguments.
  //
  virtual unsigned getNumOutgoingLinks() const { return getNumLinks(); }
  virtual PointerValSet &getOutgoingLink(unsigned Link) {
    return getLink(Link);
  }
  virtual const PointerValSet &getOutgoingLink(unsigned Link) const {
    return getLink(Link);
  }

  void print(std::ostream &O) const;
  void dump() const;

  virtual std::string getCaption() const = 0;
  virtual const std::vector<PointerValSet> *getAuxLinks() const {
    return 0;  // Default to nothing...
  }

  // isEquivalentTo - Return true if the nodes should be merged...
  virtual bool isEquivalentTo(DSNode *Node) const = 0;
  virtual void mergeInto(DSNode *Node) const {}

  DSNode *clone() const {
    DSNode *New = cloneImpl();
    // Add all of the pointers to the new node...
    for (unsigned pn = 0, pe = Pointers.size(); pn != pe; ++pn)
      New->addPointer(Pointers[pn]);
    return New;
  }

  // synthesizeNode - Create a new shadow node that is to be linked into this
  // chain..
  //
  ShadowDSNode *synthesizeNode(const Type *Ty, FunctionRepBuilder *Rep);

  virtual void dropAllReferences() {
    FieldLinks.clear();
  }

  static bool classof(const DSNode *N) { return true; }
protected:
  virtual DSNode *cloneImpl() const = 0;
  virtual void mapNode(std::map<const DSNode*, DSNode*> &NodeMap,
                       const DSNode *Old);
};


// AllocDSNode - Represent all allocation (malloc or alloca) in the program.
//
class AllocDSNode : public DSNode {
  AllocationInst *Allocation;
  bool isVarSize;                // Allocating variable sized objects
public:
  AllocDSNode(AllocationInst *V, bool isVarSize = false);

  virtual std::string getCaption() const;

  bool isAllocaNode() const;
  bool isMallocNode() const { return !isAllocaNode(); }

  AllocationInst *getAllocation() const { return Allocation; }
  bool isVariableSize() const { return isVarSize; }

  // isEquivalentTo - Return true if the nodes should be merged...
  virtual bool isEquivalentTo(DSNode *Node) const;
  virtual void mergeInto(DSNode *Node) const;

  // Support type inquiry through isa, cast, and dyn_cast...
  static bool classof(const AllocDSNode *) { return true; }
  static bool classof(const DSNode *N) { return N->NodeType == NewNode; }
protected:
  virtual AllocDSNode *cloneImpl() const { return new AllocDSNode(Allocation,
                                                                  isVarSize); }
};


// GlobalDSNode - Represent the memory location that a global variable occupies
//
class GlobalDSNode : public DSNode {
  GlobalValue *Val;
public:
  GlobalDSNode(GlobalValue *V);

  GlobalValue *getGlobal() const { return Val; }
  
  virtual std::string getCaption() const;

  // isEquivalentTo - Return true if the nodes should be merged...
  virtual bool isEquivalentTo(DSNode *Node) const;

  // Support type inquiry through isa, cast, and dyn_cast...
  static bool classof(const GlobalDSNode *) { return true; }
  static bool classof(const DSNode *N) { return N->NodeType == GlobalNode; }
private:
  virtual GlobalDSNode *cloneImpl() const { return new GlobalDSNode(Val); }
};


// CallDSNode - Represent a call instruction in the program...
//
class CallDSNode : public DSNode {
  friend class FunctionDSGraph;
  CallInst *CI;
  std::vector<PointerValSet> ArgLinks;
public:
  CallDSNode(CallInst *CI);
  ~CallDSNode() {
    ArgLinks.clear();
  }

  CallInst *getCall() const { return CI; }

  const std::vector<PointerValSet> *getAuxLinks() const { return &ArgLinks; }
  virtual std::string getCaption() const;

  bool addArgValue(unsigned ArgNo, const PointerValSet &PVS) {
    return ArgLinks[ArgNo].add(PVS);
  }

  unsigned getNumArgs() const { return ArgLinks.size(); }
  const PointerValSet &getArgValues(unsigned ArgNo) const {
    assert(ArgNo < ArgLinks.size() && "Arg # out of range!");
    return ArgLinks[ArgNo];
  }
  PointerValSet &getArgValues(unsigned ArgNo) {
    assert(ArgNo < ArgLinks.size() && "Arg # out of range!");
    return ArgLinks[ArgNo];
  }
  const std::vector<PointerValSet> &getArgs() const { return ArgLinks; }

  virtual void dropAllReferences() {
    DSNode::dropAllReferences();
    ArgLinks.clear();
  }

  // getNumOutgoingLinks - Return the number of outgoing links, which is usually
  // the number of normal links, but for call nodes it also includes their
  // arguments.
  //
  virtual unsigned getNumOutgoingLinks() const {
    return getNumLinks() + getNumArgs();
  }
  virtual PointerValSet &getOutgoingLink(unsigned Link) {
    if (Link < getNumLinks()) return getLink(Link);
    return getArgValues(Link-getNumLinks());
  }
  virtual const PointerValSet &getOutgoingLink(unsigned Link) const {
    if (Link < getNumLinks()) return getLink(Link);
    return getArgValues(Link-getNumLinks());
  }

  // isEquivalentTo - Return true if the nodes should be merged...
  virtual bool isEquivalentTo(DSNode *Node) const;

  // Support type inquiry through isa, cast, and dyn_cast...
  static bool classof(const CallDSNode *) { return true; }
  static bool classof(const DSNode *N) { return N->NodeType == CallNode; }
private:
  virtual CallDSNode *cloneImpl() const { return new CallDSNode(CI); }
  virtual void mapNode(std::map<const DSNode*, DSNode*> &NodeMap,
                       const DSNode *Old);
}; 


// ShadowDSNode - Represent a chunk of memory that we need to be able to
// address.  These are generated due to (for example) pointer type method
// arguments... if the pointer is dereferenced, we need to have a node to point
// to.  When functions are integrated into each other, shadow nodes are
// resolved.
//
class ShadowDSNode : public DSNode {
  friend class FunctionDSGraph;
  friend class FunctionRepBuilder;
  Module *Mod;
  DSNode *ShadowParent;              // Nonnull if this is a synthesized node...
public:
  ShadowDSNode(const Type *Ty, Module *M);
  virtual std::string getCaption() const;

  // isEquivalentTo - Return true if the nodes should be merged...
  virtual bool isEquivalentTo(DSNode *Node) const;

  DSNode *getShadowParent() const { return ShadowParent; }

  // Support type inquiry through isa, cast, and dyn_cast...
  static bool classof(const ShadowDSNode *) { return true; }
  static bool classof(const DSNode *N) { return N->NodeType == ShadowNode; }

private:
  ShadowDSNode(const Type *Ty, Module *M, DSNode *ShadParent);
protected:
  virtual ShadowDSNode *cloneImpl() const {
    if (ShadowParent)
      return new ShadowDSNode(getType(), Mod, ShadowParent);
    else
      return new ShadowDSNode(getType(), Mod);
  }
};


// FunctionDSGraph - The graph that represents a method.
//
class FunctionDSGraph {
  Function *Func;
  std::vector<AllocDSNode*>  AllocNodes;
  std::vector<ShadowDSNode*> ShadowNodes;
  std::vector<GlobalDSNode*> GlobalNodes;
  std::vector<CallDSNode*>   CallNodes;
  PointerValSet RetNode;             // Node that gets returned...
  std::map<Value*, PointerValSet> ValueMap;

  // cloneFunctionIntoSelf - Clone the specified method graph into the current
  // method graph, returning the Return's set of the graph.  If ValueMap is set
  // to true, the ValueMap of the function is cloned into this function as well
  // as the data structure graph itself.  Regardless, the arguments value sets
  // of DSG are copied into Args.
  //
  PointerValSet cloneFunctionIntoSelf(const FunctionDSGraph &G, bool ValueMap,
                                      std::vector<PointerValSet> &Args);

  bool RemoveUnreachableNodes();
  bool UnlinkUndistinguishableNodes();
  void MarkEscapeableNodesReachable(std::vector<bool> &RSN,
                                    std::vector<bool> &RAN);

private:
  // Define the interface only accessable to DataStructure
  friend class DataStructure;
  FunctionDSGraph(Function *F);
  FunctionDSGraph(const FunctionDSGraph &DSG);
  ~FunctionDSGraph();

  void computeClosure(const DataStructure &DS);
public:

  Function *getFunction() const { return Func; }

  // getEscapingAllocations - Add all allocations that escape the current
  // function to the specified vector.
  //
  void getEscapingAllocations(std::vector<AllocDSNode*> &Allocs);

  // getNonEscapingAllocations - Add all allocations that do not escape the
  // current function to the specified vector.
  //
  void getNonEscapingAllocations(std::vector<AllocDSNode*> &Allocs);

  // getValueMap - Get a map that describes what the nodes the scalars in this
  // function point to...
  //
  std::map<Value*, PointerValSet> &getValueMap() { return ValueMap; }
  const std::map<Value*, PointerValSet> &getValueMap() const { return ValueMap;}

  const PointerValSet &getRetNodes() const { return RetNode; }

  unsigned getGraphSize() const {
    return AllocNodes.size() + ShadowNodes.size() +
      GlobalNodes.size() + CallNodes.size();
  }

  void printFunction(std::ostream &O, const char *Label) const;
};


// FIXME: This should be a FunctionPass.  When the pass framework sees a 'Pass'
// that uses the output of a FunctionPass, it should automatically build a map
// of output from the method pass that the pass can use.
//
class DataStructure : public Pass {
  // DSInfo, one intraprocedural and one closed graph for each method...
  typedef std::map<Function*, std::pair<FunctionDSGraph*,
                                        FunctionDSGraph*> > InfoMap;
  mutable InfoMap DSInfo;
public:
  static AnalysisID ID;            // DataStructure Analysis ID 

  DataStructure(AnalysisID id) { assert(id == ID); }
  ~DataStructure() { releaseMemory(); }

  virtual const char *getPassName() const { return "Data Structure Analysis"; }

  // run - Do nothing, because methods are analyzed lazily
  virtual bool run(Module &TheModule) { return false; }

  // getDSGraph - Return the data structure graph for the specified method.
  // Since method graphs are lazily computed, we may have to create one on the
  // fly here.
  //
  FunctionDSGraph &getDSGraph(Function *F) const {
    std::pair<FunctionDSGraph*, FunctionDSGraph*> &N = DSInfo[F];
    if (N.first) return *N.first;
    return *(N.first = new FunctionDSGraph(F));
  }

  // getClosedDSGraph - Return the data structure graph for the specified
  // method. Since method graphs are lazily computed, we may have to create one
  // on the fly here. This is different than the normal DSGraph for the method
  // because any function calls that are resolvable will have the data structure
  // graphs of the called function incorporated into this function as well.
  //
  FunctionDSGraph &getClosedDSGraph(Function *F) const {
    std::pair<FunctionDSGraph*, FunctionDSGraph*> &N = DSInfo[F];
    if (N.second) return *N.second;
    N.second = new FunctionDSGraph(getDSGraph(F));
    N.second->computeClosure(*this);
    return *N.second;
  }

  // invalidateFunction - Inform this analysis that you changed the specified
  // function, so the graphs that depend on it are out of date.
  //
  void invalidateFunction(Function *F) const {
    // FIXME: THis should invalidate all functions who have inlined the
    // specified graph!
    //
    std::pair<FunctionDSGraph*, FunctionDSGraph*> &N = DSInfo[F];
    delete N.first;
    delete N.second;
    N.first = N.second = 0;
  }

  // print - Print out the analysis results...
  void print(std::ostream &O, Module *M) const;

  // If the pass pipeline is done with this pass, we can release our memory...
  virtual void releaseMemory();

  // getAnalysisUsage - This obviously provides a call graph
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addProvided(ID);
  }
};

#endif
