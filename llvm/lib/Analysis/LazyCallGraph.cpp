//===- LazyCallGraph.cpp - Analysis of a Module's call graph --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/ADT/STLExtras.h"
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

LazyCallGraph::Node::Node(LazyCallGraph &G, Function &F)
    : G(&G), F(F), DFSNumber(0), LowLink(0) {
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

LazyCallGraph::LazyCallGraph(Module &M) : NextDFSNumber(0) {
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

  for (auto &Entry : EntryNodes)
    if (Function *F = Entry.dyn_cast<Function *>())
      SCCEntryNodes.insert(F);
    else
      SCCEntryNodes.insert(&Entry.get<Node *>()->getFunction());
}

LazyCallGraph::LazyCallGraph(LazyCallGraph &&G)
    : BPA(std::move(G.BPA)), NodeMap(std::move(G.NodeMap)),
      EntryNodes(std::move(G.EntryNodes)),
      EntryNodeSet(std::move(G.EntryNodeSet)), SCCBPA(std::move(G.SCCBPA)),
      SCCMap(std::move(G.SCCMap)), LeafSCCs(std::move(G.LeafSCCs)),
      DFSStack(std::move(G.DFSStack)),
      SCCEntryNodes(std::move(G.SCCEntryNodes)),
      NextDFSNumber(G.NextDFSNumber) {
  updateGraphPtrs();
}

LazyCallGraph &LazyCallGraph::operator=(LazyCallGraph &&G) {
  BPA = std::move(G.BPA);
  NodeMap = std::move(G.NodeMap);
  EntryNodes = std::move(G.EntryNodes);
  EntryNodeSet = std::move(G.EntryNodeSet);
  SCCBPA = std::move(G.SCCBPA);
  SCCMap = std::move(G.SCCMap);
  LeafSCCs = std::move(G.LeafSCCs);
  DFSStack = std::move(G.DFSStack);
  SCCEntryNodes = std::move(G.SCCEntryNodes);
  NextDFSNumber = G.NextDFSNumber;
  updateGraphPtrs();
  return *this;
}

LazyCallGraph::Node *LazyCallGraph::insertInto(Function &F, Node *&MappedN) {
  return new (MappedN = BPA.Allocate()) Node(*this, F);
}

void LazyCallGraph::updateGraphPtrs() {
  // Process all nodes updating the graph pointers.
  SmallVector<Node *, 16> Worklist;
  for (auto &Entry : EntryNodes)
    if (Node *EntryN = Entry.dyn_cast<Node *>())
      Worklist.push_back(EntryN);

  while (!Worklist.empty()) {
    Node *N = Worklist.pop_back_val();
    N->G = this;
    for (auto &Callee : N->Callees)
      if (Node *CalleeN = Callee.dyn_cast<Node *>())
        Worklist.push_back(CalleeN);
  }
}

LazyCallGraph::SCC *LazyCallGraph::getNextSCCInPostOrder() {
  // When the stack is empty, there are no more SCCs to walk in this graph.
  if (DFSStack.empty()) {
    // If we've handled all candidate entry nodes to the SCC forest, we're done.
    if (SCCEntryNodes.empty())
      return nullptr;

    Node *N = get(*SCCEntryNodes.pop_back_val());
    DFSStack.push_back(std::make_pair(N, N->begin()));
  }

  Node *N = DFSStack.back().first;
  if (N->DFSNumber == 0) {
    // This node hasn't been visited before, assign it a DFS number and remove
    // it from the entry set.
    N->LowLink = N->DFSNumber = NextDFSNumber++;
    SCCEntryNodes.remove(&N->getFunction());
  }

  for (auto I = DFSStack.back().second, E = N->end(); I != E; ++I) {
    Node *ChildN = *I;
    if (ChildN->DFSNumber == 0) {
      // Mark that we should start at this child when next this node is the
      // top of the stack. We don't start at the next child to ensure this
      // child's lowlink is reflected.
      // FIXME: I don't actually think this is required, and we could start
      // at the next child.
      DFSStack.back().second = I;

      // Recurse onto this node via a tail call.
      DFSStack.push_back(std::make_pair(ChildN, ChildN->begin()));
      return LazyCallGraph::getNextSCCInPostOrder();
    }

    // Track the lowest link of the childen, if any are still in the stack.
    if (ChildN->LowLink < N->LowLink && !SCCMap.count(&ChildN->getFunction()))
      N->LowLink = ChildN->LowLink;
  }

  // The tail of the stack is the new SCC. Allocate the SCC and pop the stack
  // into it.
  SCC *NewSCC = new (SCCBPA.Allocate()) SCC();

  // Because we don't follow the strict Tarjan recursive formulation, walk
  // from the top of the stack down, propagating the lowest link and stopping
  // when the DFS number is the lowest link.
  int LowestLink = N->LowLink;
  do {
    Node *SCCN = DFSStack.pop_back_val().first;
    SCCMap.insert(std::make_pair(&SCCN->getFunction(), NewSCC));
    NewSCC->Nodes.push_back(SCCN);
    LowestLink = std::min(LowestLink, SCCN->LowLink);
    bool Inserted =
        NewSCC->NodeSet.insert(&SCCN->getFunction());
    (void)Inserted;
    assert(Inserted && "Cannot have duplicates in the DFSStack!");
  } while (!DFSStack.empty() && LowestLink <= DFSStack.back().first->DFSNumber);
  assert(LowestLink == NewSCC->Nodes.back()->DFSNumber &&
         "Cannot stop with a DFS number greater than the lowest link!");

  // A final pass over all edges in the SCC (this remains linear as we only
  // do this once when we build the SCC) to connect it to the parent sets of
  // its children.
  bool IsLeafSCC = true;
  for (Node *SCCN : NewSCC->Nodes)
    for (Node *SCCChildN : *SCCN) {
      if (NewSCC->NodeSet.count(&SCCChildN->getFunction()))
        continue;
      SCC *ChildSCC = SCCMap.lookup(&SCCChildN->getFunction());
      assert(ChildSCC &&
             "Must have all child SCCs processed when building a new SCC!");
      ChildSCC->ParentSCCs.insert(NewSCC);
      IsLeafSCC = false;
    }

  // For the SCCs where we fine no child SCCs, add them to the leaf list.
  if (IsLeafSCC)
    LeafSCCs.push_back(NewSCC);

  return NewSCC;
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

static void printSCC(raw_ostream &OS, LazyCallGraph::SCC &SCC) {
  ptrdiff_t SCCSize = std::distance(SCC.begin(), SCC.end());
  OS << "  SCC with " << SCCSize << " functions:\n";

  for (LazyCallGraph::Node *N : SCC)
    OS << "    " << N->getFunction().getName() << "\n";

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

  for (LazyCallGraph::SCC *SCC : G.postorder_sccs())
    printSCC(OS, *SCC);

  return PreservedAnalyses::all();

}
