//===- CallGraph.cpp - Build a Module's call graph ------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This file implements the CallGraph class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Constants.h"     // Remove when ConstantPointerRefs are gone
#include "llvm/Module.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/Support/CallSite.h"
#include "Support/STLExtras.h"
using namespace llvm;

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

static bool isOnlyADirectCall(Function *F, CallSite CS) {
  if (!CS.getInstruction()) return false;
  for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end(); I != E; ++I)
    if (*I == F) return false;
  return true;
}

// addToCallGraph - Add a function to the call graph, and link the node to all
// of the functions that it calls.
//
void CallGraph::addToCallGraph(Function *F) {
  CallGraphNode *Node = getNodeFor(F);

  // If this function has external linkage, anything could call it...
  if (!F->hasInternalLinkage()) {
    ExternalCallingNode->addCalledFunction(Node);

    // Found the entry point?
    if (F->getName() == "main") {
      if (Root)    // Found multiple external mains?  Don't pick one.
        Root = ExternalCallingNode;
      else
        Root = Node;          // Found a main, keep track of it!
    }
  }
  
  // If this function is not defined in this translation unit, it could call
  // anything.
  if (F->isExternal() && !F->getIntrinsicID())
    Node->addCalledFunction(CallsExternalNode);

  // Loop over all of the users of the function... looking for callers...
  //
  bool isUsedExternally = false;
  for (Value::use_iterator I = F->use_begin(), E = F->use_end(); I != E; ++I) {
    if (Instruction *Inst = dyn_cast<Instruction>(*I)) {
      if (isOnlyADirectCall(F, CallSite::get(Inst)))
        getNodeFor(Inst->getParent()->getParent())->addCalledFunction(Node);
      else
        isUsedExternally = true;
    } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(*I)) {
      // THIS IS A DISGUSTING HACK.  Brought to you by the power of
      // ConstantPointerRefs!
      for (Value::use_iterator I = CPR->use_begin(), E = CPR->use_end();
           I != E; ++I)
        if (Instruction *Inst = dyn_cast<Instruction>(*I)) {
          if (isOnlyADirectCall(F, CallSite::get(Inst)))
            getNodeFor(Inst->getParent()->getParent())->addCalledFunction(Node);
          else
            isUsedExternally = true;
        } else {
          isUsedExternally = true;
        }
    } else {                        // Can't classify the user!
      isUsedExternally = true;
    }
  }
  if (isUsedExternally)
    ExternalCallingNode->addCalledFunction(Node);

  // Look for an indirect function call...
  for (Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB)
    for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE; ++II){
      CallSite CS = CallSite::get(II);
      if (CS.getInstruction() && !CS.getCalledFunction())
        Node->addCalledFunction(CallsExternalNode);
    }
}

bool CallGraph::run(Module &M) {
  destroy();

  Mod = &M;
  ExternalCallingNode = getNodeFor(0);
  CallsExternalNode = new CallGraphNode(0);
  Root = 0;

  // Add every function to the call graph...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    addToCallGraph(I);

  // If we didn't find a main function, use the external call graph node
  if (Root == 0) Root = ExternalCallingNode;
  
  return false;
}

void CallGraph::destroy() {
  for (FunctionMapTy::iterator I = FunctionMap.begin(), E = FunctionMap.end();
       I != E; ++I)
    delete I->second;
  FunctionMap.clear();
  delete CallsExternalNode;
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
void CallGraph::addFunctionToModule(Function *F) {
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

void CallGraph::stub() {}

void CallGraphNode::removeCallEdgeTo(CallGraphNode *Callee) {
  for (unsigned i = CalledFunctions.size(); ; --i) {
    assert(i && "Cannot find callee to remove!");
    if (CalledFunctions[i-1] == Callee) {
      CalledFunctions.erase(CalledFunctions.begin()+i-1);
      return;
    }
  }
}
