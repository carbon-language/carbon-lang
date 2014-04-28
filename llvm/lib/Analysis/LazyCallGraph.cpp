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
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "lcg"

static void findCallees(
    SmallVectorImpl<Constant *> &Worklist, SmallPtrSetImpl<Constant *> &Visited,
    SmallVectorImpl<PointerUnion<Function *, LazyCallGraph::Node *>> &Callees,
    DenseMap<Function *, size_t> &CalleeIndexMap) {
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
      if (!F->isDeclaration() &&
          CalleeIndexMap.insert(std::make_pair(F, Callees.size())).second) {
        DEBUG(dbgs() << "    Added callable function: " << F->getName()
                     << "\n");
        Callees.push_back(F);
      }
      continue;
    }

    for (Value *Op : C->operand_values())
      if (Visited.insert(cast<Constant>(Op)))
        Worklist.push_back(cast<Constant>(Op));
  }
}

LazyCallGraph::Node::Node(LazyCallGraph &G, Function &F)
    : G(&G), F(F), DFSNumber(0), LowLink(0) {
  DEBUG(dbgs() << "  Adding functions called by '" << F.getName()
               << "' to the graph.\n");

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
  findCallees(Worklist, Visited, Callees, CalleeIndexMap);
}

void LazyCallGraph::Node::removeEdgeInternal(Function &Callee) {
  auto IndexMapI = CalleeIndexMap.find(&Callee);
  assert(IndexMapI != CalleeIndexMap.end() &&
         "Callee not in the callee set for this caller?");

  Callees.erase(Callees.begin() + IndexMapI->second);
  CalleeIndexMap.erase(IndexMapI);
}

LazyCallGraph::LazyCallGraph(Module &M) : NextDFSNumber(0) {
  DEBUG(dbgs() << "Building CG for module: " << M.getModuleIdentifier()
               << "\n");
  for (Function &F : M)
    if (!F.isDeclaration() && !F.hasLocalLinkage())
      if (EntryIndexMap.insert(std::make_pair(&F, EntryNodes.size())).second) {
        DEBUG(dbgs() << "  Adding '" << F.getName()
                     << "' to entry set of the graph.\n");
        EntryNodes.push_back(&F);
      }

  // Now add entry nodes for functions reachable via initializers to globals.
  SmallVector<Constant *, 16> Worklist;
  SmallPtrSet<Constant *, 16> Visited;
  for (GlobalVariable &GV : M.globals())
    if (GV.hasInitializer())
      if (Visited.insert(GV.getInitializer()))
        Worklist.push_back(GV.getInitializer());

  DEBUG(dbgs() << "  Adding functions referenced by global initializers to the "
                  "entry set.\n");
  findCallees(Worklist, Visited, EntryNodes, EntryIndexMap);

  for (auto &Entry : EntryNodes)
    if (Function *F = Entry.dyn_cast<Function *>())
      SCCEntryNodes.push_back(F);
    else
      SCCEntryNodes.push_back(&Entry.get<Node *>()->getFunction());
}

LazyCallGraph::LazyCallGraph(LazyCallGraph &&G)
    : BPA(std::move(G.BPA)), NodeMap(std::move(G.NodeMap)),
      EntryNodes(std::move(G.EntryNodes)),
      EntryIndexMap(std::move(G.EntryIndexMap)), SCCBPA(std::move(G.SCCBPA)),
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
  EntryIndexMap = std::move(G.EntryIndexMap);
  SCCBPA = std::move(G.SCCBPA);
  SCCMap = std::move(G.SCCMap);
  LeafSCCs = std::move(G.LeafSCCs);
  DFSStack = std::move(G.DFSStack);
  SCCEntryNodes = std::move(G.SCCEntryNodes);
  NextDFSNumber = G.NextDFSNumber;
  updateGraphPtrs();
  return *this;
}

void LazyCallGraph::SCC::insert(Node &N) {
  N.DFSNumber = N.LowLink = -1;
  Nodes.push_back(&N);
  G->SCCMap[&N] = this;
}

void LazyCallGraph::SCC::removeInterSCCEdge(Node &CallerN, Node &CalleeN) {
  // First remove it from the node.
  CallerN.removeEdgeInternal(CalleeN.getFunction());

  assert(G->SCCMap.lookup(&CallerN) == this &&
         "The caller must be a member of this SCC.");

  SCC &CalleeC = *G->SCCMap.lookup(&CalleeN);
  assert(&CalleeC != this &&
         "This API only supports the rmoval of inter-SCC edges.");

  assert(std::find(G->LeafSCCs.begin(), G->LeafSCCs.end(), this) ==
             G->LeafSCCs.end() &&
         "Cannot have a leaf SCC caller with a different SCC callee.");

  bool HasOtherCallToCalleeC = false;
  bool HasOtherCallOutsideSCC = false;
  for (Node *N : *this) {
    for (Node &OtherCalleeN : *N) {
      SCC &OtherCalleeC = *G->SCCMap.lookup(&OtherCalleeN);
      if (&OtherCalleeC == &CalleeC) {
        HasOtherCallToCalleeC = true;
        break;
      }
      if (&OtherCalleeC != this)
        HasOtherCallOutsideSCC = true;
    }
    if (HasOtherCallToCalleeC)
      break;
  }
  // Because the SCCs form a DAG, deleting such an edge cannot change the set
  // of SCCs in the graph. However, it may cut an edge of the SCC DAG, making
  // the caller no longer a parent of the callee. Walk the other call edges
  // in the caller to tell.
  if (!HasOtherCallToCalleeC) {
    bool Removed = CalleeC.ParentSCCs.erase(this);
    (void)Removed;
    assert(Removed &&
           "Did not find the caller SCC in the callee SCC's parent list!");

    // It may orphan an SCC if it is the last edge reaching it, but that does
    // not violate any invariants of the graph.
    if (CalleeC.ParentSCCs.empty())
      DEBUG(dbgs() << "LCG: Update removing " << CallerN.getFunction().getName()
                   << " -> " << CalleeN.getFunction().getName()
                   << " edge orphaned the callee's SCC!\n");
  }

  // It may make the Caller SCC a leaf SCC.
  if (!HasOtherCallOutsideSCC)
    G->LeafSCCs.push_back(this);
}

void LazyCallGraph::SCC::internalDFS(
    SmallVectorImpl<std::pair<Node *, Node::iterator>> &DFSStack,
    SmallVectorImpl<Node *> &PendingSCCStack, Node *N,
    SmallVectorImpl<SCC *> &ResultSCCs) {
  Node::iterator I = N->begin();
  N->LowLink = N->DFSNumber = 1;
  int NextDFSNumber = 2;
  for (;;) {
    assert(N->DFSNumber != 0 && "We should always assign a DFS number "
                                "before processing a node.");

    // We simulate recursion by popping out of the nested loop and continuing.
    Node::iterator E = N->end();
    while (I != E) {
      Node &ChildN = *I;
      if (SCC *ChildSCC = G->SCCMap.lookup(&ChildN)) {
        // Check if we have reached a node in the new (known connected) set of
        // this SCC. If so, the entire stack is necessarily in that set and we
        // can re-start.
        if (ChildSCC == this) {
          insert(*N);
          while (!PendingSCCStack.empty())
            insert(*PendingSCCStack.pop_back_val());
          while (!DFSStack.empty())
            insert(*DFSStack.pop_back_val().first);
          return;
        }

        // If this child isn't currently in this SCC, no need to process it.
        // However, we do need to remove this SCC from its SCC's parent set.
        ChildSCC->ParentSCCs.erase(this);
        ++I;
        continue;
      }

      if (ChildN.DFSNumber == 0) {
        // Mark that we should start at this child when next this node is the
        // top of the stack. We don't start at the next child to ensure this
        // child's lowlink is reflected.
        DFSStack.push_back(std::make_pair(N, I));

        // Continue, resetting to the child node.
        ChildN.LowLink = ChildN.DFSNumber = NextDFSNumber++;
        N = &ChildN;
        I = ChildN.begin();
        E = ChildN.end();
        continue;
      }

      // Track the lowest link of the childen, if any are still in the stack.
      // Any child not on the stack will have a LowLink of -1.
      assert(ChildN.LowLink != 0 &&
             "Low-link must not be zero with a non-zero DFS number.");
      if (ChildN.LowLink >= 0 && ChildN.LowLink < N->LowLink)
        N->LowLink = ChildN.LowLink;
      ++I;
    }

    if (N->LowLink == N->DFSNumber) {
      ResultSCCs.push_back(G->formSCC(N, PendingSCCStack));
      if (DFSStack.empty())
        return;
    } else {
      // At this point we know that N cannot ever be an SCC root. Its low-link
      // is not its dfs-number, and we've processed all of its children. It is
      // just sitting here waiting until some node further down the stack gets
      // low-link == dfs-number and pops it off as well. Move it to the pending
      // stack which is pulled into the next SCC to be formed.
      PendingSCCStack.push_back(N);

      assert(!DFSStack.empty() && "We shouldn't have an empty stack!");
    }

    N = DFSStack.back().first;
    I = DFSStack.back().second;
    DFSStack.pop_back();
  }
}

SmallVector<LazyCallGraph::SCC *, 1>
LazyCallGraph::SCC::removeIntraSCCEdge(Node &CallerN,
                                       Node &CalleeN) {
  // First remove it from the node.
  CallerN.removeEdgeInternal(CalleeN.getFunction());

  // We return a list of the resulting *new* SCCs in postorder.
  SmallVector<SCC *, 1> ResultSCCs;

  // Direct recursion doesn't impact the SCC graph at all.
  if (&CallerN == &CalleeN)
    return ResultSCCs;

  // The worklist is every node in the original SCC.
  SmallVector<Node *, 1> Worklist;
  Worklist.swap(Nodes);
  for (Node *N : Worklist) {
    // The nodes formerly in this SCC are no longer in any SCC.
    N->DFSNumber = 0;
    N->LowLink = 0;
    G->SCCMap.erase(N);
  }
  assert(Worklist.size() > 1 && "We have to have at least two nodes to have an "
                                "edge between them that is within the SCC.");

  // The callee can already reach every node in this SCC (by definition). It is
  // the only node we know will stay inside this SCC. Everything which
  // transitively reaches Callee will also remain in the SCC. To model this we
  // incrementally add any chain of nodes which reaches something in the new
  // node set to the new node set. This short circuits one side of the Tarjan's
  // walk.
  insert(CalleeN);

  // We're going to do a full mini-Tarjan's walk using a local stack here.
  SmallVector<std::pair<Node *, Node::iterator>, 4> DFSStack;
  SmallVector<Node *, 4> PendingSCCStack;
  do {
    Node *N = Worklist.pop_back_val();
    if (N->DFSNumber == 0)
      internalDFS(DFSStack, PendingSCCStack, N, ResultSCCs);

    assert(DFSStack.empty() && "Didn't flush the entire DFS stack!");
    assert(PendingSCCStack.empty() && "Didn't flush all pending SCC nodes!");
  } while (!Worklist.empty());

  // Now we need to reconnect the current SCC to the graph.
  bool IsLeafSCC = true;
  for (Node *N : Nodes) {
    for (Node &ChildN : *N) {
      SCC &ChildSCC = *G->SCCMap.lookup(&ChildN);
      if (&ChildSCC == this)
        continue;
      ChildSCC.ParentSCCs.insert(this);
      IsLeafSCC = false;
    }
  }
#ifndef NDEBUG
  if (!ResultSCCs.empty())
    assert(!IsLeafSCC && "This SCC cannot be a leaf as we have split out new "
                         "SCCs by removing this edge.");
  if (!std::any_of(G->LeafSCCs.begin(), G->LeafSCCs.end(),
                   [&](SCC *C) { return C == this; }))
    assert(!IsLeafSCC && "This SCC cannot be a leaf as it already had child "
                         "SCCs before we removed this edge.");
#endif
  // If this SCC stopped being a leaf through this edge removal, remove it from
  // the leaf SCC list.
  if (!IsLeafSCC && !ResultSCCs.empty())
    G->LeafSCCs.erase(std::remove(G->LeafSCCs.begin(), G->LeafSCCs.end(), this),
                     G->LeafSCCs.end());

  // Return the new list of SCCs.
  return ResultSCCs;
}

void LazyCallGraph::removeEdge(Node &CallerN, Function &Callee) {
  assert(SCCMap.empty() && DFSStack.empty() &&
         "This method cannot be called after SCCs have been formed!");

  return CallerN.removeEdgeInternal(Callee);
}

LazyCallGraph::Node &LazyCallGraph::insertInto(Function &F, Node *&MappedN) {
  return *new (MappedN = BPA.Allocate()) Node(*this, F);
}

void LazyCallGraph::updateGraphPtrs() {
  // Process all nodes updating the graph pointers.
  {
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

  // Process all SCCs updating the graph pointers.
  {
    SmallVector<SCC *, 16> Worklist(LeafSCCs.begin(), LeafSCCs.end());

    while (!Worklist.empty()) {
      SCC *C = Worklist.pop_back_val();
      C->G = this;
      Worklist.insert(Worklist.end(), C->ParentSCCs.begin(),
                      C->ParentSCCs.end());
    }
  }
}

LazyCallGraph::SCC *LazyCallGraph::formSCC(Node *RootN,
                                           SmallVectorImpl<Node *> &NodeStack) {
  // The tail of the stack is the new SCC. Allocate the SCC and pop the stack
  // into it.
  SCC *NewSCC = new (SCCBPA.Allocate()) SCC(*this);

  while (!NodeStack.empty() && NodeStack.back()->DFSNumber > RootN->DFSNumber) {
    assert(NodeStack.back()->LowLink >= RootN->LowLink &&
           "We cannot have a low link in an SCC lower than its root on the "
           "stack!");
    NewSCC->insert(*NodeStack.pop_back_val());
  }
  NewSCC->insert(*RootN);

  // A final pass over all edges in the SCC (this remains linear as we only
  // do this once when we build the SCC) to connect it to the parent sets of
  // its children.
  bool IsLeafSCC = true;
  for (Node *SCCN : NewSCC->Nodes)
    for (Node &SCCChildN : *SCCN) {
      if (SCCMap.lookup(&SCCChildN) == NewSCC)
        continue;
      SCC &ChildSCC = *SCCMap.lookup(&SCCChildN);
      ChildSCC.ParentSCCs.insert(NewSCC);
      IsLeafSCC = false;
    }

  // For the SCCs where we fine no child SCCs, add them to the leaf list.
  if (IsLeafSCC)
    LeafSCCs.push_back(NewSCC);

  return NewSCC;
}

LazyCallGraph::SCC *LazyCallGraph::getNextSCCInPostOrder() {
  Node *N;
  Node::iterator I;
  if (!DFSStack.empty()) {
    N = DFSStack.back().first;
    I = DFSStack.back().second;
    DFSStack.pop_back();
  } else {
    // If we've handled all candidate entry nodes to the SCC forest, we're done.
    do {
      if (SCCEntryNodes.empty())
        return nullptr;

      N = &get(*SCCEntryNodes.pop_back_val());
    } while (N->DFSNumber != 0);
    I = N->begin();
    N->LowLink = N->DFSNumber = 1;
    NextDFSNumber = 2;
  }

  for (;;) {
    assert(N->DFSNumber != 0 && "We should always assign a DFS number "
                                "before placing a node onto the stack.");

    Node::iterator E = N->end();
    while (I != E) {
      Node &ChildN = *I;
      if (ChildN.DFSNumber == 0) {
        // Mark that we should start at this child when next this node is the
        // top of the stack. We don't start at the next child to ensure this
        // child's lowlink is reflected.
        DFSStack.push_back(std::make_pair(N, N->begin()));

        // Recurse onto this node via a tail call.
        assert(!SCCMap.count(&ChildN) &&
               "Found a node with 0 DFS number but already in an SCC!");
        ChildN.LowLink = ChildN.DFSNumber = NextDFSNumber++;
        N = &ChildN;
        I = ChildN.begin();
        E = ChildN.end();
        continue;
      }

      // Track the lowest link of the childen, if any are still in the stack.
      assert(ChildN.LowLink != 0 &&
             "Low-link must not be zero with a non-zero DFS number.");
      if (ChildN.LowLink >= 0 && ChildN.LowLink < N->LowLink)
        N->LowLink = ChildN.LowLink;
      ++I;
    }

    if (N->LowLink == N->DFSNumber)
      // Form the new SCC out of the top of the DFS stack.
      return formSCC(N, PendingSCCStack);

    // At this point we know that N cannot ever be an SCC root. Its low-link
    // is not its dfs-number, and we've processed all of its children. It is
    // just sitting here waiting until some node further down the stack gets
    // low-link == dfs-number and pops it off as well. Move it to the pending
    // stack which is pulled into the next SCC to be formed.
    PendingSCCStack.push_back(N);

    assert(!DFSStack.empty() && "We never found a viable root!");
    N = DFSStack.back().first;
    I = DFSStack.back().second;
    DFSStack.pop_back();
  }
}

char LazyCallGraphAnalysis::PassID;

LazyCallGraphPrinterPass::LazyCallGraphPrinterPass(raw_ostream &OS) : OS(OS) {}

static void printNodes(raw_ostream &OS, LazyCallGraph::Node &N,
                       SmallPtrSetImpl<LazyCallGraph::Node *> &Printed) {
  // Recurse depth first through the nodes.
  for (LazyCallGraph::Node &ChildN : N)
    if (Printed.insert(&ChildN))
      printNodes(OS, ChildN, Printed);

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
  for (LazyCallGraph::Node &N : G)
    if (Printed.insert(&N))
      printNodes(OS, N, Printed);

  for (LazyCallGraph::SCC &SCC : G.postorder_sccs())
    printSCC(OS, SCC);

  return PreservedAnalyses::all();

}
