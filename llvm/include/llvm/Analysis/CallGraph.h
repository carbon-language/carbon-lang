//===- llvm/Analysis/CallGraph.h - Build a Module's call graph ---*- C++ -*--=//
//
// This interface is used to build and manipulate a call graph, which is a very 
// useful tool for interprocedural optimization.
//
// This call graph represents a dynamic method invocation as a null method node.
// A call graph may only have up to one null method node that represents all of
// the dynamic method invocations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CALLGRAPH_H
#define LLVM_ANALYSIS_CALLGRAPH_H

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
  Module *Mod;
  typedef map<const Method *, CallGraphNode *> MethodMapTy;
  MethodMapTy MethodMap;
public:
  CallGraph(Module *TheModule);

  typedef MethodMapTy::iterator iterator;
  typedef MethodMapTy::const_iterator const_iterator;

  inline const_iterator begin() const { return MethodMap.begin(); }
  inline const_iterator end()   const { return MethodMap.end();   }

  inline const CallGraphNode *operator[](const Method *M) const {
    const_iterator I = MethodMap.find(M);
    assert(I != MethodMap.end() && "Method not in callgraph!");
    return I->second;
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

#endif
