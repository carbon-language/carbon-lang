//===- llvm/Analysis/CallGraph.h - Build a Module's call graph ---*- C++ -*--=//
//
// This interface is used to build and manipulate a call graph, which is a very 
// useful tool for interprocedural optimization.
//
// This call graph represents a dynamic method invocation as a null method node.
// A call graph may only have up to one null method node that represents all of
// the dynamic method invocations.
//
// Additionally, the 'root' node of a call graph represents the "entry point"
// node of the graph, which has an edge to every external method in the graph.
// This node has a null method pointer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CALLGRAPH_H
#define LLVM_ANALYSIS_CALLGRAPH_H

#include "Support/GraphTraits.h"
#include <map>
#include <vector>
class Method;
class Module;

namespace cfg {

class CallGraph;
class CallGraphNode {
  Method *Meth;
  vector<CallGraphNode*> CalledMethods;

  CallGraphNode(const CallGraphNode &);           // Do not implement
public:
  typedef vector<CallGraphNode*>::iterator iterator;
  typedef vector<CallGraphNode*>::const_iterator const_iterator;

  // getMethod - Return the method that this call graph node represents...
  Method *getMethod() const { return Meth; }

  inline iterator begin() { return CalledMethods.begin(); }
  inline iterator end()   { return CalledMethods.end();   }
  inline const_iterator begin() const { return CalledMethods.begin(); }
  inline const_iterator end()   const { return CalledMethods.end();   }
  inline unsigned size() const { return CalledMethods.size(); }

  inline CallGraphNode *operator[](unsigned i) const { return CalledMethods[i];}

  void removeAllCalledMethods() {
    CalledMethods.clear();
  }

private:                    // Stuff to construct the node, used by CallGraph
  friend class CallGraph;

  // CallGraphNode ctor - Create a node for the specified method...
  inline CallGraphNode(Method *M) : Meth(M) {}
  
  // addCalledMethod add a method to the list of methods called by this one
  void addCalledMethod(CallGraphNode *M) {
    CalledMethods.push_back(M);
  }
};


class CallGraph {
  Module *Mod;              // The module this call graph represents

  typedef map<const Method *, CallGraphNode *> MethodMapTy;
  MethodMapTy MethodMap;    // Map from a method to its node

  CallGraphNode *Root;
public:
  CallGraph(Module *TheModule);
  ~CallGraph();

  typedef MethodMapTy::iterator iterator;
  typedef MethodMapTy::const_iterator const_iterator;

  inline       CallGraphNode *getRoot()       { return Root; }
  inline const CallGraphNode *getRoot() const { return Root; }
  inline       iterator begin()       { return MethodMap.begin(); }
  inline       iterator end()         { return MethodMap.end();   }
  inline const_iterator begin() const { return MethodMap.begin(); }
  inline const_iterator end()   const { return MethodMap.end();   }

  inline const CallGraphNode *operator[](const Method *M) const {
    const_iterator I = MethodMap.find(M);
    assert(I != MethodMap.end() && "Method not in callgraph!");
    return I->second;
  }
  inline CallGraphNode *operator[](const Method *M) {
    const_iterator I = MethodMap.find(M);
    assert(I != MethodMap.end() && "Method not in callgraph!");
    return I->second;
  }

  // Methods to keep a call graph up to date with a method that has been
  // modified
  //
  void addMethodToModule(Method *Meth);  // TODO IMPLEMENT


  // removeMethodFromModule - Unlink the method from this module, returning it.
  // Because this removes the method from the module, the call graph node is
  // destroyed.  This is only valid if the method does not call any other
  // methods (ie, there are no edges in it's CGN).  The easiest way to do this
  // is to dropAllReferences before calling this.
  //
  Method *removeMethodFromModule(CallGraphNode *CGN);
  Method *removeMethodFromModule(Method *Meth) {
    return removeMethodFromModule((*this)[Meth]);
  }

private:   // Implementation of CallGraph construction

  // getNodeFor - Return the node for the specified method or create one if it
  // does not already exist.
  //
  CallGraphNode *getNodeFor(Method *M);

  // addToCallGraph - Add a method to the call graph, and link the node to all
  // of the methods that it calls.
  //
  void addToCallGraph(Method *M);
};

}  // end namespace cfg




// Provide graph traits for tranversing call graphs using standard graph
// traversals.
template <> struct GraphTraits<cfg::CallGraphNode*> {
  typedef cfg::CallGraphNode NodeType;
  typedef NodeType::iterator ChildIteratorType;

  static NodeType *getEntryNode(cfg::CallGraphNode *CGN) { return CGN; }
  static inline ChildIteratorType child_begin(NodeType *N) { return N->begin();}
  static inline ChildIteratorType child_end  (NodeType *N) { return N->end(); }
};

template <> struct GraphTraits<const cfg::CallGraphNode*> {
  typedef const cfg::CallGraphNode NodeType;
  typedef NodeType::const_iterator ChildIteratorType;

  static NodeType *getEntryNode(const cfg::CallGraphNode *CGN) { return CGN; }
  static inline ChildIteratorType child_begin(NodeType *N) { return N->begin();}
  static inline ChildIteratorType child_end  (NodeType *N) { return N->end(); }
};


template<> struct GraphTraits<cfg::CallGraph*> :
  public GraphTraits<cfg::CallGraphNode*> {
  static NodeType *getEntryNode(cfg::CallGraph *CGN) {
    return CGN->getRoot();
  }
};
template<> struct GraphTraits<const cfg::CallGraph*> :
  public GraphTraits<const cfg::CallGraphNode*> {
  static NodeType *getEntryNode(const cfg::CallGraph *CGN) {
    return CGN->getRoot();
  }
};


// Checks if a method contains any call instructions.
// Note that this uses the call graph only if one is provided.
// It does not build the call graph.
// 
bool isLeafMethod(const Method* method, const cfg::CallGraph *callGraph = 0);

#endif
