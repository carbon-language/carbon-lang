//===- CallGraph.cpp - Build a Module's call graph ------------------------===//
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

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <iostream>

AnalysisID CallGraph::ID(AnalysisID::create<CallGraph>());

// getNodeFor - Return the node for the specified method or create one if it
// does not already exist.
//
CallGraphNode *CallGraph::getNodeFor(Function *F) {
  CallGraphNode *&CGN = MethodMap[F];
  if (CGN) return CGN;

  assert((!F || F->getParent() == Mod) && "Function not in current module!");
  return CGN = new CallGraphNode(F);
}

// addToCallGraph - Add a method to the call graph, and link the node to all of
// the methods that it calls.
//
void CallGraph::addToCallGraph(Function *M) {
  CallGraphNode *Node = getNodeFor(M);

  // If this method has external linkage, 
  if (!M->hasInternalLinkage()) {
    ExternalNode->addCalledMethod(Node);

    // Found the entry point?
    if (M->getName() == "main") {
      if (Root)
        Root = ExternalNode;  // Found multiple external mains?  Don't pick one.
      else
        Root = Node;          // Found a main, keep track of it!
    }
  } else if (M->isExternal()) { // Not defined in this xlation unit?
    Node->addCalledMethod(ExternalNode);  // It could call anything...
  }

  // Loop over all of the users of the method... looking for callers...
  //
  for (Value::use_iterator I = M->use_begin(), E = M->use_end(); I != E; ++I) {
    User *U = *I;
    if (CallInst *CI = dyn_cast<CallInst>(U))
      getNodeFor(CI->getParent()->getParent())->addCalledMethod(Node);
    else if (InvokeInst *II = dyn_cast<InvokeInst>(U))
      getNodeFor(II->getParent()->getParent())->addCalledMethod(Node);
    else                         // Can't classify the user!
      ExternalNode->addCalledMethod(Node);
  }

  // Look for an indirect method call...
  for (Function::iterator BB = M->begin(), BBE = M->end(); BB != BBE; ++BB)
    for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE; ++II){
      Instruction &I = *II;

      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        if (CI->getCalledFunction() == 0)
          Node->addCalledMethod(ExternalNode);
      } else if (InvokeInst *II = dyn_cast<InvokeInst>(&I)) {
        if (II->getCalledFunction() == 0)
          Node->addCalledMethod(ExternalNode);
      }
    }
}

bool CallGraph::run(Module &M) {
  destroy();

  Mod = &M;
  ExternalNode = getNodeFor(0);
  Root = 0;

  // Add every method to the call graph...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    addToCallGraph(I);

  // If we didn't find a main method, use the external call graph node
  if (Root == 0) Root = ExternalNode;
  
  return false;
}

void CallGraph::destroy() {
  for (MethodMapTy::iterator I = MethodMap.begin(), E = MethodMap.end();
       I != E; ++I)
    delete I->second;
  MethodMap.clear();
}


void WriteToOutput(const CallGraphNode *CGN, std::ostream &o) {
  if (CGN->getMethod())
    o << "Call graph node for method: '" << CGN->getMethod()->getName() <<"'\n";
  else
    o << "Call graph node null method:\n";

  for (unsigned i = 0; i < CGN->size(); ++i)
    if ((*CGN)[i]->getMethod())
      o << "  Calls method '" << (*CGN)[i]->getMethod()->getName() << "'\n";
    else
      o << "  Calls external node\n";
  o << "\n";
}

void WriteToOutput(const CallGraph &CG, std::ostream &o) {
  o << "CallGraph Root is:\n" << CG.getRoot();

  for (CallGraph::const_iterator I = CG.begin(), E = CG.end(); I != E; ++I)
    o << I->second;
}


//===----------------------------------------------------------------------===//
// Implementations of public modification methods
//

// Methods to keep a call graph up to date with a method that has been
// modified
//
void CallGraph::addMethodToModule(Function *Meth) {
  assert(0 && "not implemented");
  abort();
}

// removeMethodFromModule - Unlink the method from this module, returning it.
// Because this removes the method from the module, the call graph node is
// destroyed.  This is only valid if the method does not call any other
// methods (ie, there are no edges in it's CGN).  The easiest way to do this
// is to dropAllReferences before calling this.
//
Function *CallGraph::removeMethodFromModule(CallGraphNode *CGN) {
  assert(CGN->CalledMethods.empty() && "Cannot remove method from call graph"
	 " if it references other methods!");
  Function *M = CGN->getMethod(); // Get the function for the call graph node
  delete CGN;                     // Delete the call graph node for this func
  MethodMap.erase(M);             // Remove the call graph node from the map

  Mod->getFunctionList().remove(M);
  return M;
}

