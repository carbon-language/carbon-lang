//===- CallGraph.cpp - Build a Module's call graph ------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Module.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "Support/STLExtras.h"
#include <algorithm>

static RegisterAnalysis<CallGraph> X("callgraph", "Call Graph Construction");

// getNodeFor - Return the node for the specified function or create one if it
// does not already exist.
//
CallGraphNode *CallGraph::getNodeFor(Function *F) {
  CallGraphNode *&CGN = FunctionMap[F];
  if (CGN) return CGN;

  assert((!F || F->getParent() == Mod) && "Function not in current module!");
  return CGN = new CallGraphNode(F);
}

// addToCallGraph - Add a function to the call graph, and link the node to all
// of the functions that it calls.
//
void CallGraph::addToCallGraph(Function *F) {
  CallGraphNode *Node = getNodeFor(F);

  // If this function has external linkage, anything could call it...
  if (!F->hasInternalLinkage()) {
    ExternalNode->addCalledFunction(Node);

    // Found the entry point?
    if (F->getName() == "main") {
      if (Root)
        Root = ExternalNode;  // Found multiple external mains?  Don't pick one.
      else
        Root = Node;          // Found a main, keep track of it!
    }
  }
  
  // If this function is not defined in this translation unit, it could call
  // anything.
  if (F->isExternal())
    Node->addCalledFunction(ExternalNode);

  // Loop over all of the users of the function... looking for callers...
  //
  for (Value::use_iterator I = F->use_begin(), E = F->use_end(); I != E; ++I) {
    User *U = *I;
    if (CallInst *CI = dyn_cast<CallInst>(U))
      getNodeFor(CI->getParent()->getParent())->addCalledFunction(Node);
    else if (InvokeInst *II = dyn_cast<InvokeInst>(U))
      getNodeFor(II->getParent()->getParent())->addCalledFunction(Node);
    else                         // Can't classify the user!
      ExternalNode->addCalledFunction(Node);
  }

  // Look for an indirect function call...
  for (Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB)
    for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE; ++II){
      Instruction &I = *II;

      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        if (CI->getCalledFunction() == 0)
          Node->addCalledFunction(ExternalNode);
      } else if (InvokeInst *II = dyn_cast<InvokeInst>(&I)) {
        if (II->getCalledFunction() == 0)
          Node->addCalledFunction(ExternalNode);
      }
    }
}

bool CallGraph::run(Module &M) {
  destroy();

  Mod = &M;
  ExternalNode = getNodeFor(0);
  Root = 0;

  // Add every function to the call graph...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    addToCallGraph(I);

  // If we didn't find a main function, use the external call graph node
  if (Root == 0) Root = ExternalNode;
  
  return false;
}

void CallGraph::destroy() {
  for (FunctionMapTy::iterator I = FunctionMap.begin(), E = FunctionMap.end();
       I != E; ++I)
    delete I->second;
  FunctionMap.clear();
}

static void WriteToOutput(const CallGraphNode *CGN, std::ostream &o) {
  if (CGN->getFunction())
    o << "Call graph node for function: '"
      << CGN->getFunction()->getName() <<"'\n";
  else
    o << "Call graph node <<null function: 0x" << CGN << ">>:\n";

  for (unsigned i = 0; i < CGN->size(); ++i)
    if ((*CGN)[i]->getFunction())
      o << "  Calls function '" << (*CGN)[i]->getFunction()->getName() << "'\n";
    else
      o << "  Calls external node\n";
  o << "\n";
}

void CallGraph::print(std::ostream &o, const Module *M) const {
  o << "CallGraph Root is: ";
  if (getRoot()->getFunction())
    o << getRoot()->getFunction()->getName() << "\n";
  else
    o << "<<null function: 0x" << getRoot() << ">>\n";
  
  for (CallGraph::const_iterator I = begin(), E = end(); I != E; ++I)
    WriteToOutput(I->second, o);
}


//===----------------------------------------------------------------------===//
// Implementations of public modification methods
//

// Functions to keep a call graph up to date with a function that has been
// modified
//
void CallGraph::addFunctionToModule(Function *Meth) {
  assert(0 && "not implemented");
  abort();
}

// removeFunctionFromModule - Unlink the function from this module, returning
// it.  Because this removes the function from the module, the call graph node
// is destroyed.  This is only valid if the function does not call any other
// functions (ie, there are no edges in it's CGN).  The easiest way to do this
// is to dropAllReferences before calling this.
//
Function *CallGraph::removeFunctionFromModule(CallGraphNode *CGN) {
  assert(CGN->CalledFunctions.empty() && "Cannot remove function from call "
         "graph if it references other functions!");
  Function *F = CGN->getFunction(); // Get the function for the call graph node
  delete CGN;                       // Delete the call graph node for this func
  FunctionMap.erase(F);             // Remove the call graph node from the map

  Mod->getFunctionList().remove(F);
  return F;
}

