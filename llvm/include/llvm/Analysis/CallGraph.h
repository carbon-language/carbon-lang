//===- CallGraph.h - Build a Module's call graph -----------------*- C++ -*--=//
//
// This interface is used to build and manipulate a call graph, which is a very 
// useful tool for interprocedural optimization.
//
// Every function in a module is represented as a node in the call graph.  The
// callgraph node keeps track of which functions the are called by the function
// corresponding to the node.
//
// A call graph will contain nodes where the function that they correspond to is
// null.  This 'external' node is used to represent control flow that is not
// represented (or analyzable) in the module.  As such, the external node will
// have edges to functions with the following properties:
//   1. All functions in the module without internal linkage, since they could
//      be called by functions outside of the our analysis capability.
//   2. All functions whose address is used for something more than a direct
//      call, for example being stored into a memory location.  Since they may
//      be called by an unknown caller later, they must be tracked as such.
//
// Similarly, functions have a call edge to the external node iff:
//   1. The function is external, reflecting the fact that they could call
//      anything without internal linkage or that has its address taken.
//   2. The function contains an indirect function call.
//
// As an extension in the future, there may be multiple nodes with a null
// function.  These will be used when we can prove (through pointer analysis)
// that an indirect call site can call only a specific set of functions.
//
// Because of these properties, the CallGraph captures a conservative superset
// of all of the caller-callee relationships, which is useful for
// transformations.
//
// The CallGraph class also attempts to figure out what the root of the
// CallGraph is, which is currently does by looking for a function named 'main'.
// If no function named 'main' is found, the external node is used as the entry
// node, reflecting the fact that any function without internal linkage could
// be called into (which is common for libraries).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CALLGRAPH_H
#define LLVM_ANALYSIS_CALLGRAPH_H

#include "Support/GraphTraits.h"
#include "Support/STLExtras.h"
#include "llvm/Pass.h"
class Function;
class Module;
class CallGraphNode;

//===----------------------------------------------------------------------===//
// CallGraph class definition
//
class CallGraph : public Pass {
  Module *Mod;              // The module this call graph represents

  typedef std::map<const Function *, CallGraphNode *> FunctionMapTy;
  FunctionMapTy FunctionMap;    // Map from a function to its node

  // Root is root of the call graph, or the external node if a 'main' function
  // couldn't be found.  ExternalNode is equivalent to (*this)[0].
  //
  CallGraphNode *Root, *ExternalNode;
public:

  //===---------------------------------------------------------------------
  // Accessors...
  //
  typedef FunctionMapTy::iterator iterator;
  typedef FunctionMapTy::const_iterator const_iterator;

  // getExternalNode - Return the node that points to all functions that are
  // accessable from outside of the current program.
  //
        CallGraphNode *getExternalNode()       { return ExternalNode; }
  const CallGraphNode *getExternalNode() const { return ExternalNode; }

  // getRoot - Return the root of the call graph, which is either main, or if
  // main cannot be found, the external node.
  //
        CallGraphNode *getRoot()       { return Root; }
  const CallGraphNode *getRoot() const { return Root; }

  inline       iterator begin()       { return FunctionMap.begin(); }
  inline       iterator end()         { return FunctionMap.end();   }
  inline const_iterator begin() const { return FunctionMap.begin(); }
  inline const_iterator end()   const { return FunctionMap.end();   }


  // Subscripting operators, return the call graph node for the provided
  // function
  inline const CallGraphNode *operator[](const Function *F) const {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second;
  }
  inline CallGraphNode *operator[](const Function *F) {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second;
  }

  //===---------------------------------------------------------------------
  // Functions to keep a call graph up to date with a function that has been
  // modified
  //
  void addFunctionToModule(Function *Meth);


  // removeFunctionFromModule - Unlink the function from this module, returning
  // it.  Because this removes the function from the module, the call graph node
  // is destroyed.  This is only valid if the function does not call any other
  // functions (ie, there are no edges in it's CGN).  The easiest way to do this
  // is to dropAllReferences before calling this.
  //
  Function *removeFunctionFromModule(CallGraphNode *CGN);
  Function *removeFunctionFromModule(Function *Meth) {
    return removeFunctionFromModule((*this)[Meth]);
  }


  //===---------------------------------------------------------------------
  // Pass infrastructure interface glue code...
  //
  CallGraph() : Root(0) {}
  ~CallGraph() { destroy(); }

  // run - Compute the call graph for the specified module.
  virtual bool run(Module &M);

  // getAnalysisUsage - This obviously provides a call graph
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  // releaseMemory - Data structures can be large, so free memory aggressively.
  virtual void releaseMemory() {
    destroy();
  }

  /// Print the types found in the module.  If the optional Module parameter is
  /// passed in, then the types are printed symbolically if possible, using the
  /// symbol table from the module.
  ///
  void print(std::ostream &o, const Module *M) const;

private:
  //===---------------------------------------------------------------------
  // Implementation of CallGraph construction
  //

  // getNodeFor - Return the node for the specified function or create one if it
  // does not already exist.
  //
  CallGraphNode *getNodeFor(Function *F);

  // addToCallGraph - Add a function to the call graph, and link the node to all
  // of the functions that it calls.
  //
  void addToCallGraph(Function *F);

  // destroy - Release memory for the call graph
  void destroy();
};


//===----------------------------------------------------------------------===//
// CallGraphNode class definition
//
class CallGraphNode {
  Function *Meth;
  std::vector<CallGraphNode*> CalledFunctions;

  CallGraphNode(const CallGraphNode &);           // Do not implement
public:
  //===---------------------------------------------------------------------
  // Accessor methods...
  //

  typedef std::vector<CallGraphNode*>::iterator iterator;
  typedef std::vector<CallGraphNode*>::const_iterator const_iterator;

  // getFunction - Return the function that this call graph node represents...
  Function *getFunction() const { return Meth; }

  inline iterator begin() { return CalledFunctions.begin(); }
  inline iterator end()   { return CalledFunctions.end();   }
  inline const_iterator begin() const { return CalledFunctions.begin(); }
  inline const_iterator end()   const { return CalledFunctions.end();   }
  inline unsigned size() const { return CalledFunctions.size(); }

  // Subscripting operator - Return the i'th called function...
  //
  CallGraphNode *operator[](unsigned i) const { return CalledFunctions[i];}


  //===---------------------------------------------------------------------
  // Methods to keep a call graph up to date with a function that has been
  // modified
  //

  void removeAllCalledFunctions() {
    CalledFunctions.clear();
  }

private:                    // Stuff to construct the node, used by CallGraph
  friend class CallGraph;

  // CallGraphNode ctor - Create a node for the specified function...
  inline CallGraphNode(Function *F) : Meth(F) {}
  
  // addCalledFunction add a function to the list of functions called by this
  // one
  void addCalledFunction(CallGraphNode *M) {
    CalledFunctions.push_back(M);
  }
};



//===----------------------------------------------------------------------===//
// GraphTraits specializations for call graphs so that they can be treated as
// graphs by the generic graph algorithms...
//

// Provide graph traits for tranversing call graphs using standard graph
// traversals.
template <> struct GraphTraits<CallGraphNode*> {
  typedef CallGraphNode NodeType;
  typedef NodeType::iterator ChildIteratorType;

  static NodeType *getEntryNode(CallGraphNode *CGN) { return CGN; }
  static inline ChildIteratorType child_begin(NodeType *N) { return N->begin();}
  static inline ChildIteratorType child_end  (NodeType *N) { return N->end(); }
};

template <> struct GraphTraits<const CallGraphNode*> {
  typedef const CallGraphNode NodeType;
  typedef NodeType::const_iterator ChildIteratorType;

  static NodeType *getEntryNode(const CallGraphNode *CGN) { return CGN; }
  static inline ChildIteratorType child_begin(NodeType *N) { return N->begin();}
  static inline ChildIteratorType child_end  (NodeType *N) { return N->end(); }
};

template<> struct GraphTraits<CallGraph*> : public GraphTraits<CallGraphNode*> {
  static NodeType *getEntryNode(CallGraph *CGN) {
    return CGN->getExternalNode();  // Start at the external node!
  }
  typedef std::pair<const Function*, CallGraphNode*> PairTy;
  typedef std::pointer_to_unary_function<PairTy, CallGraphNode&> DerefFun;

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<CallGraph::iterator, DerefFun> nodes_iterator;
  static nodes_iterator nodes_begin(CallGraph *CG) {
    return map_iterator(CG->begin(), DerefFun(CGdereference));
  }
  static nodes_iterator nodes_end  (CallGraph *CG) {
    return map_iterator(CG->end(), DerefFun(CGdereference));
  }

  static CallGraphNode &CGdereference (std::pair<const Function*,
                                       CallGraphNode*> P) {
    return *P.second;
  }
};
template<> struct GraphTraits<const CallGraph*> :
  public GraphTraits<const CallGraphNode*> {
  static NodeType *getEntryNode(const CallGraph *CGN) {
    return CGN->getExternalNode();
  }
  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef CallGraph::const_iterator nodes_iterator;
  static nodes_iterator nodes_begin(const CallGraph *CG) { return CG->begin(); }
  static nodes_iterator nodes_end  (const CallGraph *CG) { return CG->end(); }
};

#endif
