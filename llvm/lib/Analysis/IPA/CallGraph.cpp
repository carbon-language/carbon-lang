//===- CallGraph.cpp - Build a Module's call graph --------------------------=//
//
// This file implements call graph construction (from a module), and will
// eventually implement call graph serialization and deserialization for
// annotation support.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/Writer.h"
#include "llvm/Support/STLExtras.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/iOther.h"
#include <algorithm>

using namespace cfg;

// getNodeFor - Return the node for the specified method or create one if it
// does not already exist.
//
CallGraphNode *CallGraph::getNodeFor(Method *M) {
  iterator I = MethodMap.find(M);
  if (I != MethodMap.end()) return I->second;

  assert(M->getParent() == Mod && "Method not in current module!");
  CallGraphNode *New = new CallGraphNode(M);

  MethodMap.insert(pair<const Method*, CallGraphNode*>(M, New));
  return New;
}

// addToCallGraph - Add a method to the call graph, and link the node to all of
// the methods that it calls.
//
void CallGraph::addToCallGraph(Method *M) {
  CallGraphNode *Node = getNodeFor(M);

  for (Method::inst_iterator II = M->inst_begin(), IE = M->inst_end();
       II != IE; ++II) {
    if (II->getOpcode() == Instruction::Call) {
      CallInst *CI = (CallInst*)*II;
      Node->addCalledMethod(getNodeFor(CI->getCalledMethod()));
    }
  }
}

CallGraph::CallGraph(Module *TheModule) {
  Mod = TheModule;

  // Add every method to the call graph...
  for_each(Mod->begin(), Mod->end(), bind_obj(this,&CallGraph::addToCallGraph));
}


void cfg::WriteToOutput(const CallGraphNode *CGN, ostream &o) {
  o << "Call graph node for method: '" << CGN->getMethod()->getName() << "'\n";
  for (unsigned i = 0; i < CGN->size(); ++i)
    o << "  Calls method '" << (*CGN)[i]->getMethod()->getName() << "'\n";
  o << endl;
}

void cfg::WriteToOutput(const CallGraph &CG, ostream &o) {
  for (CallGraph::const_iterator I = CG.begin(), E = CG.end(); I != E; ++I)
    o << I->second;
}
