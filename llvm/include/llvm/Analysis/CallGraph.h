//===- CallGraph.h - Build a Module's call graph -----------------*- C++ -*--=//
//
// This interface is used to build and manipulate a call graph, which is a very 
// useful tool for interprocedural optimization.
//
// Every method in a module is represented as a node in the call graph.  The
// callgraph node keeps track of which methods the are called by the method
// corresponding to the node.
//
// A call graph will contain nodes where the method that they correspond to is
// null.  This 'external' node is used to represent control flow that is not
// represented (or analyzable) in the module.  As such, the external node will
// have edges to methods with the following properties:
//   1. All methods in the module without internal linkage, since they could
//      be called by methods outside of the our analysis capability.
//   2. All methods whose address is used for something more than a direct call,
//      for example being stored into a memory location.  Since they may be
//      called by an unknown caller later, they must be tracked as such.
//
// Similarly, methods have a call edge to the external node iff:
//   1. The method is external, reflecting the fact that they could call
//      anything without internal linkage or that has its address taken.
//   2. The method contains an indirect method call.
//
// As an extension in the future, there may be multiple nodes with a null
// method.  These will be used when we can prove (through pointer analysis) that
// an indirect call site can call only a specific set of methods.
//
// Because of these properties, the CallGraph captures a conservative superset
// of all of the caller-callee relationships, which is useful for
// transformations.
//
// The CallGraph class also attempts to figure out what the root of the
// CallGraph is, which is currently does by looking for a method named 'main'.
// If no method named 'main' is found, the external node is used as the entry
// node, reflecting the fact that any method without internal linkage could
// be called into (which is common for libraries).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CALLGRAPH_H
#define LLVM_ANALYSIS_CALLGRAPH_H

#include "Support/GraphTraits.h"
#include "llvm/Pass.h"
class Function;
class Module;
class CallGraphNode;

//===----------------------------------------------------------------------===//
// CallGraph class definition
//
class CallGraph : public Pass {
  Module *Mod;              // The module this call graph represents

  typedef std::map<const Function *, CallGraphNode *> MethodMapTy;
  MethodMapTy MethodMap;    // Map from a method to its node

  // Root is root of the call graph, or the external node if a 'main' function
  // couldn't be found.  ExternalNode is equivalent to (*this)[0].
  //
  CallGraphNode *Root, *ExternalNode;
public:

  //===---------------------------------------------------------------------
  // Accessors...
  //
  typedef MethodMapTy::iterator iterator;
  typedef MethodMapTy::const_iterator const_iterator;

  inline       CallGraphNode *getRoot()       { return Root; }
  inline const CallGraphNode *getRoot() const { return Root; }
  inline       iterator begin()       { return MethodMap.begin(); }
  inline       iterator end()         { return MethodMap.end();   }
  inline const_iterator begin() const { return MethodMap.begin(); }
  inline const_iterator end()   const { return MethodMap.end();   }


  // Subscripting operators, return the call graph node for the provided method
  inline const CallGraphNode *operator[](const Function *F) const {
    const_iterator I = MethodMap.find(F);
    assert(I != MethodMap.end() && "Method not in callgraph!");
    return I->second;
  }
  inline CallGraphNode *operator[](const Function *F) {
    const_iterator I = MethodMap.find(F);
    assert(I != MethodMap.end() && "Method not in callgraph!");
    return I->second;
  }

  //===---------------------------------------------------------------------
  // Methods to keep a call graph up to date with a method that has been
  // modified
  //
  void addMethodToModule(Function *Meth);


  // removeMethodFromModule - Unlink the method from this module, returning it.
  // Because this removes the method from the module, the call graph node is
  // destroyed.  This is only valid if the method does not call any other
  // methods (ie, there are no edges in it's CGN).  The easiest way to do this
  // is to dropAllReferences before calling this.
  //
  Function *removeMethodFromModule(CallGraphNode *CGN);
  Function *removeMethodFromModule(Function *Meth) {
    return removeMethodFromModule((*this)[Meth]);
  }


  //===---------------------------------------------------------------------
  // Pass infrastructure interface glue code...
  //
  static AnalysisID ID;    // We are an analysis, we must have an ID

  CallGraph(AnalysisID AID) : Root(0) { assert(AID == ID); }
  ~CallGraph() { destroy(); }

  virtual const char *getPassName() const { return "Call Graph Construction"; }

  // run - Compute the call graph for the specified module.
  virtual bool run(Module &M);

  // getAnalysisUsage - This obviously provides a call graph
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addProvided(ID);
  }

  // releaseMemory - Data structures can be large, so free memory aggressively.
  virtual void releaseMemory() {
    destroy();
  }

private:
  //===---------------------------------------------------------------------
  // Implementation of CallGraph construction
  //

  // getNodeFor - Return the node for the specified function or create one if it
  // does not already exist.
  //
  CallGraphNode *getNodeFor(Function *F);

  // addToCallGraph - Add a function to the call graph, and link the node to all
  // of the methods that it calls.
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
  std::vector<CallGraphNode*> CalledMethods;

  CallGraphNode(const CallGraphNode &);           // Do not implement
public:
  //===---------------------------------------------------------------------
  // Accessor methods...
  //

  typedef std::vector<CallGraphNode*>::iterator iterator;
  typedef std::vector<CallGraphNode*>::const_iterator const_iterator;

  // getMethod - Return the method that this call graph node represents...
  Function *getMethod() const { return Meth; }

  inline iterator begin() { return CalledMethods.begin(); }
  inline iterator end()   { return CalledMethods.end();   }
  inline const_iterator begin() const { return CalledMethods.begin(); }
  inline const_iterator end()   const { return CalledMethods.end();   }
  inline unsigned size() const { return CalledMethods.size(); }

  // Subscripting operator - Return the i'th called method...
  //
  inline CallGraphNode *operator[](unsigned i) const { return CalledMethods[i];}


  //===---------------------------------------------------------------------
  // Methods to keep a call graph up to date with a method that has been
  // modified
  //

  void removeAllCalledMethods() {
    CalledMethods.clear();
  }

private:                    // Stuff to construct the node, used by CallGraph
  friend class CallGraph;

  // CallGraphNode ctor - Create a node for the specified method...
  inline CallGraphNode(Function *F) : Meth(F) {}
  
  // addCalledMethod add a method to the list of methods called by this one
  void addCalledMethod(CallGraphNode *M) {
    CalledMethods.push_back(M);
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


template<> struct GraphTraits<CallGraph*> :
  public GraphTraits<CallGraphNode*> {
  static NodeType *getEntryNode(CallGraph *CGN) {
    return CGN->getRoot();
  }
};
template<> struct GraphTraits<const CallGraph*> :
  public GraphTraits<const CallGraphNode*> {
  static NodeType *getEntryNode(const CallGraph *CGN) {
    return CGN->getRoot();
  }
};


//===----------------------------------------------------------------------===//
// Printing support for Call Graphs
//

// Stuff for printing out a callgraph...

void WriteToOutput(const CallGraph &, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o, const CallGraph &CG) {
  WriteToOutput(CG, o); return o;
}
  
void WriteToOutput(const CallGraphNode *, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o, const CallGraphNode *CGN) {
  WriteToOutput(CGN, o); return o;
}

#endif
