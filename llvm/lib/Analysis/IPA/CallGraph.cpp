//===- CallGraph.cpp - Build a Module's call graph ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CallGraph class and provides the BasicCallGraph
// default implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Support/CallSite.h"
#include <iostream>
using namespace llvm;

static bool isOnlyADirectCall(Function *F, CallSite CS) {
  if (!CS.getInstruction()) return false;
  for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end(); I != E; ++I)
    if (*I == F) return false;
  return true;
}

namespace {

//===----------------------------------------------------------------------===//
// BasicCallGraph class definition
//
class BasicCallGraph : public CallGraph, public ModulePass {
  // Root is root of the call graph, or the external node if a 'main' function
  // couldn't be found.
  //
  CallGraphNode *Root;

  // ExternalCallingNode - This node has edges to all external functions and
  // those internal functions that have their address taken.
  CallGraphNode *ExternalCallingNode;

  // CallsExternalNode - This node has edges to it from all functions making
  // indirect calls or calling an external function.
  CallGraphNode *CallsExternalNode;

public:
  BasicCallGraph() : Root(0), ExternalCallingNode(0), CallsExternalNode(0) {}
  ~BasicCallGraph() { destroy(); }

  // runOnModule - Compute the call graph for the specified module.
  virtual bool runOnModule(Module &M) {
    destroy();
    CallGraph::initialize(M);
    
    ExternalCallingNode = getOrInsertFunction(0);
    CallsExternalNode = new CallGraphNode(0);
    Root = 0;
  
    // Add every function to the call graph...
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      addToCallGraph(I);
  
    // If we didn't find a main function, use the external call graph node
    if (Root == 0) Root = ExternalCallingNode;
    
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  virtual void print(std::ostream &o, const Module *M) const {
    o << "CallGraph Root is: ";
    if (Function *F = getRoot()->getFunction())
      o << F->getName() << "\n";
    else
      o << "<<null function: 0x" << getRoot() << ">>\n";
    
    CallGraph::print(o, M);
  }

  virtual void releaseMemory() {
    destroy();
  }
  
  /// dump - Print out this call graph.
  ///
  inline void dump() const {
    print(std::cerr, Mod);
  }

  CallGraphNode* getExternalCallingNode() const { return ExternalCallingNode; }
  CallGraphNode* getCallsExternalNode()   const { return CallsExternalNode; }

  // getRoot - Return the root of the call graph, which is either main, or if
  // main cannot be found, the external node.
  //
  CallGraphNode *getRoot()             { return Root; }
  const CallGraphNode *getRoot() const { return Root; }

private:
  //===---------------------------------------------------------------------
  // Implementation of CallGraph construction
  //

  // addToCallGraph - Add a function to the call graph, and link the node to all
  // of the functions that it calls.
  //
  void addToCallGraph(Function *F) {
    CallGraphNode *Node = getOrInsertFunction(F);

    // If this function has external linkage, anything could call it.
    if (!F->hasInternalLinkage()) {
      ExternalCallingNode->addCalledFunction(CallSite(), Node);

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
      Node->addCalledFunction(CallSite(), CallsExternalNode);

    // Loop over all of the users of the function... looking for callers...
    //
    bool isUsedExternally = false;
    for (Value::use_iterator I = F->use_begin(), E = F->use_end(); I != E; ++I){
      if (Instruction *Inst = dyn_cast<Instruction>(*I)) {
        CallSite CS = CallSite::get(Inst);
        if (isOnlyADirectCall(F, CS))
          getOrInsertFunction(Inst->getParent()->getParent())
              ->addCalledFunction(CS, Node);
        else
          isUsedExternally = true;
      } else if (GlobalValue *GV = dyn_cast<GlobalValue>(*I)) {
        for (Value::use_iterator I = GV->use_begin(), E = GV->use_end();
             I != E; ++I)
          if (Instruction *Inst = dyn_cast<Instruction>(*I)) {
            CallSite CS = CallSite::get(Inst);
            if (isOnlyADirectCall(F, CS))
              getOrInsertFunction(Inst->getParent()->getParent())
                ->addCalledFunction(CS, Node);
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
      ExternalCallingNode->addCalledFunction(CallSite(), Node);

    // Look for an indirect function call.
    for (Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB)
      for (BasicBlock::iterator II = BB->begin(), IE = BB->end();
           II != IE; ++II) {
      CallSite CS = CallSite::get(II);
      if (CS.getInstruction() && !CS.getCalledFunction())
        Node->addCalledFunction(CS, CallsExternalNode);
      }
  }

  //
  // destroy - Release memory for the call graph
  virtual void destroy() {
    if (!CallsExternalNode) {
      delete CallsExternalNode;
      CallsExternalNode = 0;
    }
  }
};

RegisterAnalysisGroup<CallGraph> X("Call Graph");
RegisterOpt<BasicCallGraph> Y("basiccg", "Basic CallGraph Construction");
RegisterAnalysisGroup<CallGraph, BasicCallGraph, true> Z;

} //End anonymous namespace

void CallGraph::initialize(Module &M) {
  destroy();
  Mod = &M;
}

void CallGraph::destroy() {
  if(!FunctionMap.size()) {
    for (FunctionMapTy::iterator I = FunctionMap.begin(), E = FunctionMap.end();
        I != E; ++I)
      delete I->second;
    FunctionMap.clear();
  }
}

void CallGraph::print(std::ostream &OS, const Module *M) const {
  for (CallGraph::const_iterator I = begin(), E = end(); I != E; ++I)
    I->second->print(OS);
}

void CallGraph::dump() const {
  print(std::cerr, 0);
}

//===----------------------------------------------------------------------===//
// Implementations of public modification methods
//

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

// changeFunction - This method changes the function associated with this
// CallGraphNode, for use by transformations that need to change the prototype
// of a Function (thus they must create a new Function and move the old code
// over).
void CallGraph::changeFunction(Function *OldF, Function *NewF) {
  iterator I = FunctionMap.find(OldF);
  CallGraphNode *&New = FunctionMap[NewF];
  assert(I != FunctionMap.end() && I->second && !New &&
         "OldF didn't exist in CG or NewF already does!");
  New = I->second;
  New->F = NewF;
  FunctionMap.erase(I);
}

// getOrInsertFunction - This method is identical to calling operator[], but
// it will insert a new CallGraphNode for the specified function if one does
// not already exist.
CallGraphNode *CallGraph::getOrInsertFunction(const Function *F) {
  CallGraphNode *&CGN = FunctionMap[F];
  if (CGN) return CGN;
  
  assert((!F || F->getParent() == Mod) && "Function not in current module!");
  return CGN = new CallGraphNode(const_cast<Function*>(F));
}

void CallGraphNode::print(std::ostream &OS) const {
  if (Function *F = getFunction())
    OS << "Call graph node for function: '" << F->getName() <<"'\n";
  else
    OS << "Call graph node <<null function: 0x" << this << ">>:\n";

  for (const_iterator I = begin(), E = end(); I != E; ++I)
    if (I->second->getFunction())
      OS << "  Calls function '" << I->second->getFunction()->getName() <<"'\n";
  else
    OS << "  Calls external node\n";
  OS << "\n";
}

void CallGraphNode::dump() const { print(std::cerr); }

void CallGraphNode::removeCallEdgeTo(CallGraphNode *Callee) {
  for (unsigned i = CalledFunctions.size(); ; --i) {
    assert(i && "Cannot find callee to remove!");
    if (CalledFunctions[i-1].second == Callee) {
      CalledFunctions.erase(CalledFunctions.begin()+i-1);
      return;
    }
  }
}

// removeAnyCallEdgeTo - This method removes any call edges from this node to
// the specified callee function.  This takes more time to execute than
// removeCallEdgeTo, so it should not be used unless necessary.
void CallGraphNode::removeAnyCallEdgeTo(CallGraphNode *Callee) {
  for (unsigned i = 0, e = CalledFunctions.size(); i != e; ++i)
    if (CalledFunctions[i].second == Callee) {
      CalledFunctions[i] = CalledFunctions.back();
      CalledFunctions.pop_back();
      --i; --e;
    }
}

// Enuse that users of CallGraph.h also link with this file
DEFINING_FILE_FOR(CallGraph)
