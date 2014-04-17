//===- LazyCallGraph.cpp - Analysis of a Module's call graph --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void findCallees(
    SmallVectorImpl<Constant *> &Worklist, SmallPtrSetImpl<Constant *> &Visited,
    SmallVectorImpl<PointerUnion<Function *, LazyCallGraph::Node *>> &Callees,
    SmallPtrSetImpl<Function *> &CalleeSet) {
  while (!Worklist.empty()) {
    Constant *C = Worklist.pop_back_val();

    if (Function *F = dyn_cast<Function>(C)) {
      // Note that we consider *any* function with a definition to be a viable
      // edge. Even if the function's definition is subject to replacement by
      // some other module (say, a weak definition) there may still be
      // optimizations which essentially speculate based on the definition and
      // a way to check that the specific definition is in fact the one being
      // used. For example, this could be done by moving the weak definition to
      // a strong (internal) definition and making the weak definition be an
      // alias. Then a test of the address of the weak function against the new
      // strong definition's address would be an effective way to determine the
      // safety of optimizing a direct call edge.
      if (!F->isDeclaration() && CalleeSet.insert(F))
        Callees.push_back(F);
      continue;
    }

    for (Value *Op : C->operand_values())
      if (Visited.insert(cast<Constant>(Op)))
        Worklist.push_back(cast<Constant>(Op));
  }
}

LazyCallGraph::Node::Node(LazyCallGraph &G, Function &F) : G(G), F(F) {
  SmallVector<Constant *, 16> Worklist;
  SmallPtrSet<Constant *, 16> Visited;
  // Find all the potential callees in this function. First walk the
  // instructions and add every operand which is a constant to the worklist.
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      for (Value *Op : I.operand_values())
        if (Constant *C = dyn_cast<Constant>(Op))
          if (Visited.insert(C))
            Worklist.push_back(C);

  // We've collected all the constant (and thus potentially function or
  // function containing) operands to all of the instructions in the function.
  // Process them (recursively) collecting every function found.
  findCallees(Worklist, Visited, Callees, CalleeSet);
}

LazyCallGraph::Node::Node(LazyCallGraph &G, const Node &OtherN)
    : G(G), F(OtherN.F), CalleeSet(OtherN.CalleeSet) {
  // Loop over the other node's callees, adding the Function*s to our list
  // directly, and recursing to add the Node*s.
  Callees.reserve(OtherN.Callees.size());
  for (const auto &OtherCallee : OtherN.Callees)
    if (Function *Callee = OtherCallee.dyn_cast<Function *>())
      Callees.push_back(Callee);
    else
      Callees.push_back(G.copyInto(*OtherCallee.get<Node *>()));
}

LazyCallGraph::Node::Node(LazyCallGraph &G, Node &&OtherN)
    : G(G), F(OtherN.F), Callees(std::move(OtherN.Callees)),
      CalleeSet(std::move(OtherN.CalleeSet)) {
  // Loop over our Callees. They've been moved from another node, but we need
  // to move the Node*s to live under our bump ptr allocator.
  for (auto &Callee : Callees)
    if (Node *ChildN = Callee.dyn_cast<Node *>())
      Callee = G.moveInto(std::move(*ChildN));
}

LazyCallGraph::LazyCallGraph(Module &M) {
  for (Function &F : M)
    if (!F.isDeclaration() && !F.hasLocalLinkage())
      if (EntryNodeSet.insert(&F))
        EntryNodes.push_back(&F);

  // Now add entry nodes for functions reachable via initializers to globals.
  SmallVector<Constant *, 16> Worklist;
  SmallPtrSet<Constant *, 16> Visited;
  for (GlobalVariable &GV : M.globals())
    if (GV.hasInitializer())
      if (Visited.insert(GV.getInitializer()))
        Worklist.push_back(GV.getInitializer());

  findCallees(Worklist, Visited, EntryNodes, EntryNodeSet);
}

LazyCallGraph::LazyCallGraph(const LazyCallGraph &G)
    : EntryNodeSet(G.EntryNodeSet) {
  EntryNodes.reserve(G.EntryNodes.size());
  for (const auto &EntryNode : G.EntryNodes)
    if (Function *Callee = EntryNode.dyn_cast<Function *>())
      EntryNodes.push_back(Callee);
    else
      EntryNodes.push_back(copyInto(*EntryNode.get<Node *>()));
}

// FIXME: This would be crazy simpler if BumpPtrAllocator were movable without
// invalidating any of the allocated memory. We should make that be the case at
// some point and delete this.
LazyCallGraph::LazyCallGraph(LazyCallGraph &&G)
    : EntryNodes(std::move(G.EntryNodes)),
      EntryNodeSet(std::move(G.EntryNodeSet)) {
  // Loop over our EntryNodes. They've been moved from another graph, so we
  // need to move the Node*s to live under our bump ptr allocator. We can just
  // do this in-place.
  for (auto &Entry : EntryNodes)
    if (Node *EntryN = Entry.dyn_cast<Node *>())
      Entry = moveInto(std::move(*EntryN));
}

LazyCallGraph::Node *LazyCallGraph::insertInto(Function &F, Node *&MappedN) {
  return new (MappedN = BPA.Allocate()) Node(*this, F);
}

LazyCallGraph::Node *LazyCallGraph::copyInto(const Node &OtherN) {
  Node *&N = NodeMap[&OtherN.F];
  if (N)
    return N;

  return new (N = BPA.Allocate()) Node(*this, OtherN);
}

LazyCallGraph::Node *LazyCallGraph::moveInto(Node &&OtherN) {
  Node *&N = NodeMap[&OtherN.F];
  if (N)
    return N;

  return new (N = BPA.Allocate()) Node(*this, std::move(OtherN));
}

char LazyCallGraphAnalysis::PassID;

LazyCallGraphPrinterPass::LazyCallGraphPrinterPass(raw_ostream &OS) : OS(OS) {}

static void printNodes(raw_ostream &OS, LazyCallGraph::Node &N,
                       SmallPtrSetImpl<LazyCallGraph::Node *> &Printed) {
  // Recurse depth first through the nodes.
  for (LazyCallGraph::Node *ChildN : N)
    if (Printed.insert(ChildN))
      printNodes(OS, *ChildN, Printed);

  OS << "  Call edges in function: " << N.getFunction().getName() << "\n";
  for (LazyCallGraph::iterator I = N.begin(), E = N.end(); I != E; ++I)
    OS << "    -> " << I->getFunction().getName() << "\n";

  OS << "\n";
}

PreservedAnalyses LazyCallGraphPrinterPass::run(Module *M,
                                                ModuleAnalysisManager *AM) {
  LazyCallGraph &G = AM->getResult<LazyCallGraphAnalysis>(M);

  OS << "Printing the call graph for module: " << M->getModuleIdentifier()
     << "\n\n";

  SmallPtrSet<LazyCallGraph::Node *, 16> Printed;
  for (LazyCallGraph::Node *N : G)
    if (Printed.insert(N))
      printNodes(OS, *N, Printed);

  return PreservedAnalyses::all();
}
