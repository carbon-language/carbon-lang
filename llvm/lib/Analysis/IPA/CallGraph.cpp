//===- CallGraph.cpp - Build a Module's call graph ------------------------===//
//
// This file implements call graph construction (from a module), and will
// eventually implement call graph serialization and deserialization for
// annotation support.
//
// This call graph represents a dynamic method invocation as a null method node.
// A call graph may only have up to one null method node that represents all of
// the dynamic method invocations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/Support/InstIterator.h"// FIXME: CallGraph should use method uses
#include "Support/STLExtras.h"
#include <algorithm>
#include <iostream>

AnalysisID CallGraph::ID(AnalysisID::create<CallGraph>());

// getNodeFor - Return the node for the specified method or create one if it
// does not already exist.
//
CallGraphNode *CallGraph::getNodeFor(Method *M) {
  iterator I = MethodMap.find(M);
  if (I != MethodMap.end()) return I->second;

  assert(M->getParent() == Mod && "Method not in current module!");
  CallGraphNode *New = new CallGraphNode(M);

  MethodMap.insert(std::make_pair(M, New));
  return New;
}

// addToCallGraph - Add a method to the call graph, and link the node to all of
// the methods that it calls.
//
void CallGraph::addToCallGraph(Method *M) {
  CallGraphNode *Node = getNodeFor(M);

  // If this method has external linkage, 
  if (!M->hasInternalLinkage())
    Root->addCalledMethod(Node);

  for (inst_iterator I = inst_begin(M), E = inst_end(M); I != E; ++I) {
    // Dynamic calls will cause Null nodes to be created
    if (CallInst *CI = dyn_cast<CallInst>(*I))
      Node->addCalledMethod(getNodeFor(CI->getCalledMethod()));
    else if (InvokeInst *II = dyn_cast<InvokeInst>(*I))
      Node->addCalledMethod(getNodeFor(II->getCalledMethod()));
  }
}

bool CallGraph::run(Module *TheModule) {
  destroy();

  Mod = TheModule;

  // Create the root node of the module...
  Root = new CallGraphNode(0);

  // Add every method to the call graph...
  for_each(Mod->begin(), Mod->end(), bind_obj(this,&CallGraph::addToCallGraph));
  
  return false;
}

void CallGraph::destroy() {
  for (MethodMapTy::iterator I = MethodMap.begin(), E = MethodMap.end();
       I != E; ++I) {
    delete I->second;
  }
  MethodMap.clear();
}


void WriteToOutput(const CallGraphNode *CGN, std::ostream &o) {
  if (CGN->getMethod())
    o << "Call graph node for method: '" << CGN->getMethod()->getName() <<"'\n";
  else
    o << "Call graph node null method:\n";

  for (unsigned i = 0; i < CGN->size(); ++i)
    o << "  Calls method '" << (*CGN)[i]->getMethod()->getName() << "'\n";
  o << "\n";
}

void WriteToOutput(const CallGraph &CG, std::ostream &o) {
  WriteToOutput(CG.getRoot(), o);
  for (CallGraph::const_iterator I = CG.begin(), E = CG.end(); I != E; ++I)
    o << I->second;
}


//===----------------------------------------------------------------------===//
// Implementations of public modification methods
//

// Methods to keep a call graph up to date with a method that has been
// modified
//
void CallGraph::addMethodToModule(Method *Meth) {
  assert(0 && "not implemented");
  abort();
}

// removeMethodFromModule - Unlink the method from this module, returning it.
// Because this removes the method from the module, the call graph node is
// destroyed.  This is only valid if the method does not call any other
// methods (ie, there are no edges in it's CGN).  The easiest way to do this
// is to dropAllReferences before calling this.
//
Method *CallGraph::removeMethodFromModule(CallGraphNode *CGN) {
  assert(CGN->CalledMethods.empty() && "Cannot remove method from call graph"
	 " if it references other methods!");
  Method *M = CGN->getMethod();  // Get the method for the call graph node
  delete CGN;                    // Delete the call graph node for this method
  MethodMap.erase(M);            // Remove the call graph node from the map

  Mod->getMethodList().remove(M);
  return M;
}

