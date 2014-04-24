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
      SCCEntryNodes.insert(F);
    else
      SCCEntryNodes.insert(&Entry.get<Node *>()->getFunction());
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

void LazyCallGraph::SCC::removeEdge(LazyCallGraph &G, Function &Caller,
                                    Function &Callee, SCC &CalleeC) {
  assert(std::find(G.LeafSCCs.begin(), G.LeafSCCs.end(), this) ==
             G.LeafSCCs.end() &&
         "Cannot have a leaf SCC caller with a different SCC callee.");

  bool HasOtherCallToCalleeC = false;
  bool HasOtherCallOutsideSCC = false;
  for (Node *N : *this) {
    for (Node &Callee : *N) {
      SCC &OtherCalleeC = *G.SCCMap.lookup(&Callee);
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
      DEBUG(dbgs() << "LCG: Update removing " << Caller.getName() << " -> "
                   << Callee.getName() << " edge orphaned the callee's SCC!\n");
  }

  // It may make the Caller SCC a leaf SCC.
  if (!HasOtherCallOutsideSCC)
    G.LeafSCCs.push_back(this);
}

SmallVector<LazyCallGraph::SCC *, 1>
LazyCallGraph::SCC::removeInternalEdge(LazyCallGraph &G, Node &Caller,
                                       Node &Callee) {
  // We return a list of the resulting SCCs, where 'this' is always the first
  // element.
  SmallVector<SCC *, 1> ResultSCCs;
  ResultSCCs.push_back(this);

  // We're going to do a full mini-Tarjan's walk using a local stack here.
  int NextDFSNumber = 1;
  SmallVector<std::pair<Node *, Node::iterator>, 4> DFSStack;

  // The worklist is every node in the original SCC. FIXME: switch the SCC to
  // use a SmallSetVector and swap here.
  SmallSetVector<Node *, 1> Worklist;
  for (Node *N : Nodes) {
    // Clear these to 0 while we re-run Tarjan's over the SCC.
    N->DFSNumber = 0;
    N->LowLink = 0;
    Worklist.insert(N);
  }

  // The callee can already reach every node in this SCC (by definition). It is
  // the only node we know will stay inside this SCC. Everything which
  // transitively reaches Callee will also remain in the SCC. To model this we
  // incrementally add any chain of nodes which reaches something in the new
  // node set to the new node set. This short circuits one side of the Tarjan's
  // walk.
  SmallSetVector<Node *, 1> NewNodes;
  NewNodes.insert(&Callee);

  for (;;) {
    if (DFSStack.empty()) {
      if (Worklist.empty())
        break;
      Node *N = Worklist.pop_back_val();
      DFSStack.push_back(std::make_pair(N, N->begin()));
    }

    Node *N = DFSStack.back().first;

    // Check if we have reached a node in the new (known connected) set. If so,
    // the entire stack is necessarily in that set and we can re-start.
    if (NewNodes.count(N)) {
      DFSStack.pop_back();
      while (!DFSStack.empty())
        NewNodes.insert(DFSStack.pop_back_val().first);
      continue;
    }

    if (N->DFSNumber == 0) {
      N->LowLink = N->DFSNumber = NextDFSNumber++;
      Worklist.remove(N);
    }

    auto SI = DFSStack.rbegin();
    bool PushedChildNode = false;
    do {
      N = SI->first;
      for (auto I = SI->second, E = N->end(); I != E; ++I) {
        Node &ChildN = *I;
        // If this child isn't currently in this SCC, no need to process it.
        // However, we do need to remove this SCC from its SCC's parent set.
        SCC &ChildSCC = *G.SCCMap.lookup(&ChildN);
        if (&ChildSCC != this) {
          ChildSCC.ParentSCCs.erase(this);
          continue;
        }

        if (ChildN.DFSNumber == 0) {
          // Mark that we should start at this child when next this node is the
          // top of the stack. We don't start at the next child to ensure this
          // child's lowlink is reflected.
          SI->second = I;

          // Recurse onto this node via a tail call.
          DFSStack.push_back(std::make_pair(&ChildN, ChildN.begin()));
          PushedChildNode = true;
          break;
        }

        // Track the lowest link of the childen, if any are still in the stack.
        // Any child not on the stack will have a LowLink of -1.
        assert(ChildN.LowLink != 0 &&
               "Low-link must not be zero with a non-zero DFS number.");
        if (ChildN.LowLink >= 0 && ChildN.LowLink < N->LowLink)
          N->LowLink = ChildN.LowLink;
      }
      if (!PushedChildNode)
        // No more children to process for this stack entry.
        SI->second = N->end();

      ++SI;
      // If nothing is new on the stack and this isn't the SCC root, walk
      // upward.
    } while (!PushedChildNode && N->LowLink != N->DFSNumber &&
             SI != DFSStack.rend());

    if (PushedChildNode)
      continue;

    // Form the new SCC out of the top of the DFS stack.
    ResultSCCs.push_back(G.formSCCFromDFSStack(DFSStack, SI.base()));
  }

  // Replace this SCC with the NewNodes we collected above.
  // FIXME: Simplify this when the SCC's datastructure is just a list.
  Nodes.clear();

  // Now we need to reconnect the current SCC to the graph.
  bool IsLeafSCC = true;
  for (Node *N : NewNodes) {
    N->DFSNumber = -1;
    N->LowLink = -1;
    Nodes.push_back(N);
    for (Node &ChildN : *N) {
      if (NewNodes.count(&ChildN))
        continue;
      SCC &ChildSCC = *G.SCCMap.lookup(&ChildN);
      ChildSCC.ParentSCCs.insert(this);
      IsLeafSCC = false;
    }
  }
#ifndef NDEBUG
  if (ResultSCCs.size() > 1)
    assert(!IsLeafSCC && "This SCC cannot be a leaf as we have split out new "
                         "SCCs by removing this edge.");
  if (!std::any_of(G.LeafSCCs.begin(), G.LeafSCCs.end(),
                   [&](SCC *C) { return C == this; }))
    assert(!IsLeafSCC && "This SCC cannot be a leaf as it already had child "
                         "SCCs before we removed this edge.");
#endif
  // If this SCC stopped being a leaf through this edge removal, remove it from
  // the leaf SCC list.
  if (!IsLeafSCC && ResultSCCs.size() > 1)
    G.LeafSCCs.erase(std::remove(G.LeafSCCs.begin(), G.LeafSCCs.end(), this),
                     G.LeafSCCs.end());

  // Return the new list of SCCs.
  return ResultSCCs;
}

void LazyCallGraph::removeEdge(Node &CallerN, Function &Callee) {
  auto IndexMapI = CallerN.CalleeIndexMap.find(&Callee);
  assert(IndexMapI != CallerN.CalleeIndexMap.end() &&
         "Callee not in the callee set for the caller?");

  Node *CalleeN = CallerN.Callees[IndexMapI->second].dyn_cast<Node *>();
  CallerN.Callees.erase(CallerN.Callees.begin() + IndexMapI->second);
  CallerN.CalleeIndexMap.erase(IndexMapI);

  SCC *CallerC = SCCMap.lookup(&CallerN);
  if (!CallerC) {
    // We can only remove edges when the edge isn't actively participating in
    // a DFS walk. Either it must have been popped into an SCC, or it must not
    // yet have been reached by the DFS walk. Assert the latter here.
    assert(std::all_of(DFSStack.begin(), DFSStack.end(),
                       [&](const std::pair<Node *, iterator> &StackEntry) {
             return StackEntry.first != &CallerN;
           }) &&
           "Found the caller on the DFSStack!");
    return;
  }

  assert(CalleeN && "If the caller is in an SCC, we have to have explored all "
                    "its transitively called functions.");

  SCC *CalleeC = SCCMap.lookup(CalleeN);
  assert(CalleeC &&
         "The caller has an SCC, and thus by necessity so does the callee.");

  // The easy case is when they are different SCCs.
  if (CallerC != CalleeC) {
    CallerC->removeEdge(*this, CallerN.getFunction(), Callee, *CalleeC);
    return;
  }

  // The hard case is when we remove an edge within a SCC. This may cause new
  // SCCs to need to be added to the graph.
  CallerC->removeInternalEdge(*this, CallerN, *CalleeN);
}

LazyCallGraph::Node &LazyCallGraph::insertInto(Function &F, Node *&MappedN) {
  return *new (MappedN = BPA.Allocate()) Node(*this, F);
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

LazyCallGraph::SCC *LazyCallGraph::formSCCFromDFSStack(
    SmallVectorImpl<std::pair<Node *, Node::iterator>> &DFSStack,
    SmallVectorImpl<std::pair<Node *, Node::iterator>>::iterator SCCBegin) {
  // The tail of the stack is the new SCC. Allocate the SCC and pop the stack
  // into it.
  SCC *NewSCC = new (SCCBPA.Allocate()) SCC();

  for (auto I = SCCBegin, E = DFSStack.end(); I != E; ++I) {
    Node *SCCN = I->first;
    assert(SCCN->LowLink >= SCCBegin->first->LowLink &&
           "We cannot have a low link in an SCC lower than its root on the "
           "stack!");

    SCCMap[SCCN] = NewSCC;
    NewSCC->Nodes.push_back(SCCN);
  }
  DFSStack.erase(SCCBegin, DFSStack.end());

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
  // When the stack is empty, there are no more SCCs to walk in this graph.
  if (DFSStack.empty()) {
    // If we've handled all candidate entry nodes to the SCC forest, we're done.
    if (SCCEntryNodes.empty())
      return nullptr;

    // Reset the DFS numbering.
    NextDFSNumber = 1;
    Node &N = get(*SCCEntryNodes.pop_back_val());
    DFSStack.push_back(std::make_pair(&N, N.begin()));
  }

  auto SI = DFSStack.rbegin();
  if (SI->first->DFSNumber == 0) {
    // This node hasn't been visited before, assign it a DFS number and remove
    // it from the entry set.
    assert(!SCCMap.count(SI->first) &&
           "Found a node with 0 DFS number but already in an SCC!");
    SI->first->LowLink = SI->first->DFSNumber = NextDFSNumber++;
    SCCEntryNodes.remove(&SI->first->getFunction());
  }

  do {
    Node *N = SI->first;
    for (auto I = SI->second, E = N->end(); I != E; ++I) {
      Node &ChildN = *I;
      if (ChildN.DFSNumber == 0) {
        // Mark that we should start at this child when next this node is the
        // top of the stack. We don't start at the next child to ensure this
        // child's lowlink is reflected.
        SI->second = I;

        // Recurse onto this node via a tail call.
        DFSStack.push_back(std::make_pair(&ChildN, ChildN.begin()));
        return LazyCallGraph::getNextSCCInPostOrder();
      }

      // Track the lowest link of the childen, if any are still in the stack.
      assert(ChildN.LowLink != 0 &&
             "Low-link must not be zero with a non-zero DFS number.");
      if (ChildN.LowLink >= 0 && ChildN.LowLink < N->LowLink)
        N->LowLink = ChildN.LowLink;
    }
    // No more children to process for this stack entry.
    SI->second = N->end();

    if (N->LowLink == N->DFSNumber)
      // Form the new SCC out of the top of the DFS stack.
      return formSCCFromDFSStack(DFSStack, std::prev(SI.base()));

    ++SI;
  } while (SI != DFSStack.rend());

  llvm_unreachable(
      "We cannot reach the bottom of the stack without popping an SCC.");
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
