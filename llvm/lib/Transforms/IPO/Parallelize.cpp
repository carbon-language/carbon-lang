//===- Parallelize.cpp - Auto parallelization using DS Graphs -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a pass that automatically parallelizes a program,
// using the Cilk multi-threaded runtime system to execute parallel code.
// 
// The pass uses the Program Dependence Graph (class PDGIterator) to
// identify parallelizable function calls, i.e., calls whose instances
// can be executed in parallel with instances of other function calls.
// (In the future, this should also execute different instances of the same
// function call in parallel, but that requires parallelizing across
// loop iterations.)
//
// The output of the pass is LLVM code with:
// (1) all parallelizable functions renamed to flag them as parallelizable;
// (2) calls to a sync() function introduced at synchronization points.
// The CWriter recognizes these functions and inserts the appropriate Cilk
// keywords when writing out C code.  This C code must be compiled with cilk2c.
// 
// Current algorithmic limitations:
// -- no array dependence analysis
// -- no parallelization for function calls in different loop iterations
//    (except in unlikely trivial cases)
//
// Limitations of using Cilk:
// -- No parallelism within a function body, e.g., in a loop;
// -- Simplistic synchronization model requiring all parallel threads 
//    created within a function to block at a sync().
// -- Excessive overhead at "spawned" function calls, which has no benefit
//    once all threads are busy (especially common when the degree of
//    parallelism is low).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/DemoteRegToStack.h"
#include "llvm/Analysis/PgmDependenceGraph.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include "Support/hash_set"
#include "Support/hash_map"
#include <functional>
#include <algorithm>
using namespace llvm;

//---------------------------------------------------------------------------- 
// Global constants used in marking Cilk functions and function calls.
//---------------------------------------------------------------------------- 

static const char * const CilkSuffix = ".llvm2cilk";
static const char * const DummySyncFuncName = "__sync.llvm2cilk";

//---------------------------------------------------------------------------- 
// Routines to identify Cilk functions, calls to Cilk functions, and syncs.
//---------------------------------------------------------------------------- 

static bool isCilk(const Function& F) {
  return (F.getName().rfind(CilkSuffix) ==
          F.getName().size() - std::strlen(CilkSuffix));
}

static bool isCilkMain(const Function& F) {
  return F.getName() == "main" + std::string(CilkSuffix);
}


static bool isCilk(const CallInst& CI) {
  return CI.getCalledFunction() && isCilk(*CI.getCalledFunction());
}

static bool isSync(const CallInst& CI) { 
  return CI.getCalledFunction() &&
         CI.getCalledFunction()->getName() == DummySyncFuncName;
}


//---------------------------------------------------------------------------- 
// class Cilkifier
//
// Code generation pass that transforms code to identify where Cilk keywords
// should be inserted.  This relies on `llvm-dis -c' to print out the keywords.
//---------------------------------------------------------------------------- 


class Cilkifier: public InstVisitor<Cilkifier>
{
  Function* DummySyncFunc;

  // Data used when transforming each function.
  hash_set<const Instruction*>  stmtsVisited;    // Flags for recursive DFS
  hash_map<const CallInst*, hash_set<CallInst*> > spawnToSyncsMap;

  // Input data for the transformation.
  const hash_set<Function*>*    cilkFunctions;   // Set of parallel functions
  PgmDependenceGraph*           depGraph;

  void          DFSVisitInstr   (Instruction* I,
                                 Instruction* root,
                                 hash_set<const Instruction*>& depsOfRoot);

public:
  /*ctor*/      Cilkifier       (Module& M);

  // Transform a single function including its name, its call sites, and syncs
  // 
  void          TransformFunc   (Function* F,
                                 const hash_set<Function*>& cilkFunctions,
                                 PgmDependenceGraph&  _depGraph);

  // The visitor function that does most of the hard work, via DFSVisitInstr
  // 
  void visitCallInst(CallInst& CI);
};


Cilkifier::Cilkifier(Module& M)
{
  // create the dummy Sync function and add it to the Module
  DummySyncFunc = M.getOrInsertFunction(DummySyncFuncName, Type::VoidTy, 0);
}

void Cilkifier::TransformFunc(Function* F,
                              const hash_set<Function*>& _cilkFunctions,
                              PgmDependenceGraph& _depGraph)
{
  // Memoize the information for this function
  cilkFunctions = &_cilkFunctions;
  depGraph = &_depGraph;

  // Add the marker suffix to the Function name
  // This should automatically mark all calls to the function also!
  F->setName(F->getName() + CilkSuffix);

  // Insert sync operations for each separate spawn
  visit(*F);

  // Now traverse the CFG in rPostorder and eliminate redundant syncs, i.e.,
  // two consecutive sync's on a straight-line path with no intervening spawn.
  
}


void Cilkifier::DFSVisitInstr(Instruction* I,
                              Instruction* root,
                              hash_set<const Instruction*>& depsOfRoot)
{
  assert(stmtsVisited.find(I) == stmtsVisited.end());
  stmtsVisited.insert(I);

  // If there is a dependence from root to I, insert Sync and return
  if (depsOfRoot.find(I) != depsOfRoot.end())
    { // Insert a sync before I and stop searching along this path.
      // If I is a Phi instruction, the dependence can only be an SSA dep.
      // and we need to insert the sync in the predecessor on the appropriate
      // incoming edge!
      CallInst* syncI = 0;
      if (PHINode* phiI = dyn_cast<PHINode>(I))
        { // check all operands of the Phi and insert before each one
          for (unsigned i = 0, N = phiI->getNumIncomingValues(); i < N; ++i)
            if (phiI->getIncomingValue(i) == root)
              syncI = new CallInst(DummySyncFunc, std::vector<Value*>(), "",
                                   phiI->getIncomingBlock(i)->getTerminator());
        }
      else
        syncI = new CallInst(DummySyncFunc, std::vector<Value*>(), "", I);

      // Remember the sync for each spawn to eliminate redundant ones later
      spawnToSyncsMap[cast<CallInst>(root)].insert(syncI);

      return;
    }

  // else visit unvisited successors
  if (BranchInst* brI = dyn_cast<BranchInst>(I))
    { // visit first instruction in each successor BB
      for (unsigned i = 0, N = brI->getNumSuccessors(); i < N; ++i)
        if (stmtsVisited.find(&brI->getSuccessor(i)->front())
            == stmtsVisited.end())
          DFSVisitInstr(&brI->getSuccessor(i)->front(), root, depsOfRoot);
    }
  else
    if (Instruction* nextI = I->getNext())
      if (stmtsVisited.find(nextI) == stmtsVisited.end())
        DFSVisitInstr(nextI, root, depsOfRoot);
}


void Cilkifier::visitCallInst(CallInst& CI)
{
  assert(CI.getCalledFunction() != 0 && "Only direct calls can be spawned.");
  if (cilkFunctions->find(CI.getCalledFunction()) == cilkFunctions->end())
    return;                             // not a spawn

  // Find all the outgoing memory dependences.
  hash_set<const Instruction*> depsOfRoot;
  for (PgmDependenceGraph::iterator DI =
         depGraph->outDepBegin(CI, MemoryDeps); ! DI.fini(); ++DI)
    depsOfRoot.insert(&DI->getSink()->getInstr());

  // Now find all outgoing SSA dependences to the eventual non-Phi users of
  // the call value (i.e., direct users that are not phis, and for any
  // user that is a Phi, direct non-Phi users of that Phi, and recursively).
  std::vector<const PHINode*> phiUsers;
  hash_set<const PHINode*> phisSeen;    // ensures we don't visit a phi twice
  for (Value::use_iterator UI=CI.use_begin(), UE=CI.use_end(); UI != UE; ++UI)
    if (const PHINode* phiUser = dyn_cast<PHINode>(*UI))
      {
        if (phisSeen.find(phiUser) == phisSeen.end())
          {
            phiUsers.push_back(phiUser);
            phisSeen.insert(phiUser);
          }
      }
    else
      depsOfRoot.insert(cast<Instruction>(*UI));

  // Now we've found the non-Phi users and immediate phi users.
  // Recursively walk the phi users and add their non-phi users.
  for (const PHINode* phiUser; !phiUsers.empty(); phiUsers.pop_back())
    {
      phiUser = phiUsers.back();
      for (Value::use_const_iterator UI=phiUser->use_begin(),
             UE=phiUser->use_end(); UI != UE; ++UI)
        if (const PHINode* pn = dyn_cast<PHINode>(*UI))
          {
            if (phisSeen.find(pn) == phisSeen.end())
              {
                phiUsers.push_back(pn);
                phisSeen.insert(pn);
              }
          }
        else
          depsOfRoot.insert(cast<Instruction>(*UI));
    }

  // Walk paths of the CFG starting at the call instruction and insert
  // one sync before the first dependence on each path, if any.
  if (! depsOfRoot.empty())
    {
      stmtsVisited.clear();             // start a new DFS for this CallInst
      assert(CI.getNext() && "Call instruction cannot be a terminator!");
      DFSVisitInstr(CI.getNext(), &CI, depsOfRoot);
    }

  // Now, eliminate all users of the SSA value of the CallInst, i.e., 
  // if the call instruction returns a value, delete the return value
  // register and replace it by a stack slot.
  if (CI.getType() != Type::VoidTy)
    DemoteRegToStack(CI);
}


//---------------------------------------------------------------------------- 
// class FindParallelCalls
//
// Find all CallInst instructions that have at least one other CallInst
// that is independent.  These are the instructions that can produce
// useful parallelism.
//---------------------------------------------------------------------------- 

class FindParallelCalls : public InstVisitor<FindParallelCalls> {
  typedef hash_set<CallInst*>           DependentsSet;
  typedef DependentsSet::iterator       Dependents_iterator;
  typedef DependentsSet::const_iterator Dependents_const_iterator;

  PgmDependenceGraph& depGraph;         // dependence graph for the function
  hash_set<Instruction*> stmtsVisited;  // flags for DFS walk of depGraph
  hash_map<CallInst*, bool > completed; // flags marking if a CI is done
  hash_map<CallInst*, DependentsSet> dependents; // dependent CIs for each CI

  void VisitOutEdges(Instruction*   I,
                     CallInst*      root,
                     DependentsSet& depsOfRoot);

  FindParallelCalls(const FindParallelCalls &); // DO NOT IMPLEMENT
  void operator=(const FindParallelCalls&);     // DO NOT IMPLEMENT
public:
  std::vector<CallInst*> parallelCalls;

public:
  /*ctor*/      FindParallelCalls       (Function& F, PgmDependenceGraph& DG);
  void          visitCallInst           (CallInst& CI);
};


FindParallelCalls::FindParallelCalls(Function& F,
                                     PgmDependenceGraph& DG)
  : depGraph(DG)
{
  // Find all CallInsts reachable from each CallInst using a recursive DFS
  visit(F);

  // Now we've found all CallInsts reachable from each CallInst.
  // Find those CallInsts that are parallel with at least one other CallInst
  // by counting total inEdges and outEdges.
  // 
  unsigned long totalNumCalls = completed.size();

  if (totalNumCalls == 1)
    { // Check first for the special case of a single call instruction not
      // in any loop.  It is not parallel, even if it has no dependences
      // (this is why it is a special case).
      //
      // FIXME:
      // THIS CASE IS NOT HANDLED RIGHT NOW, I.E., THERE IS NO
      // PARALLELISM FOR CALLS IN DIFFERENT ITERATIONS OF A LOOP.
      // 
      return;
    }

  hash_map<CallInst*, unsigned long> numDeps;
  for (hash_map<CallInst*, DependentsSet>::iterator II = dependents.begin(),
         IE = dependents.end(); II != IE; ++II)
    {
      CallInst* fromCI = II->first;
      numDeps[fromCI] += II->second.size();
      for (Dependents_iterator DI = II->second.begin(), DE = II->second.end();
           DI != DE; ++DI)
        numDeps[*DI]++;                 // *DI can be reached from II->first
    }

  for (hash_map<CallInst*, DependentsSet>::iterator
         II = dependents.begin(), IE = dependents.end(); II != IE; ++II)

    // FIXME: Remove "- 1" when considering parallelism in loops
    if (numDeps[II->first] < totalNumCalls - 1)
      parallelCalls.push_back(II->first);
}


void FindParallelCalls::VisitOutEdges(Instruction* I,
                                      CallInst* root,
                                      DependentsSet& depsOfRoot)
{
  assert(stmtsVisited.find(I) == stmtsVisited.end() && "Stmt visited twice?");
  stmtsVisited.insert(I);

  if (CallInst* CI = dyn_cast<CallInst>(I))

    // FIXME: Ignoring parallelism in a loop.  Here we're actually *ignoring*
    // a self-dependence in order to get the count comparison right above.
    // When we include loop parallelism, self-dependences should be included.
    // 
    if (CI != root)

      { // CallInst root has a path to CallInst I and any calls reachable from I
        depsOfRoot.insert(CI);
        if (completed[CI])
          { // We have already visited I so we know all nodes it can reach!
            DependentsSet& depsOfI = dependents[CI];
            depsOfRoot.insert(depsOfI.begin(), depsOfI.end());
            return;
          }
      }

  // If we reach here, we need to visit all children of I
  for (PgmDependenceGraph::iterator DI = depGraph.outDepBegin(*I);
       ! DI.fini(); ++DI)
    {
      Instruction* sink = &DI->getSink()->getInstr();
      if (stmtsVisited.find(sink) == stmtsVisited.end())
        VisitOutEdges(sink, root, depsOfRoot);
    }
}


void FindParallelCalls::visitCallInst(CallInst& CI)
{
  if (completed[&CI])
    return;
  stmtsVisited.clear();                      // clear flags to do a fresh DFS

  // Visit all children of CI using a recursive walk through dep graph
  DependentsSet& depsOfRoot = dependents[&CI];
  for (PgmDependenceGraph::iterator DI = depGraph.outDepBegin(CI);
       ! DI.fini(); ++DI)
    {
      Instruction* sink = &DI->getSink()->getInstr();
      if (stmtsVisited.find(sink) == stmtsVisited.end())
        VisitOutEdges(sink, &CI, depsOfRoot);
    }

  completed[&CI] = true;
}


//---------------------------------------------------------------------------- 
// class Parallelize
//
// (1) Find candidate parallel functions: any function F s.t.
//       there is a call C1 to the function F that is followed or preceded
//       by at least one other call C2 that is independent of this one
//       (i.e., there is no dependence path from C1 to C2 or C2 to C1)
// (2) Label such a function F as a cilk function.
// (3) Convert every call to F to a spawn
// (4) For every function X, insert sync statements so that
//        every spawn is postdominated by a sync before any statements
//        with a data dependence to/from the call site for the spawn
// 
//---------------------------------------------------------------------------- 

namespace {
  class Parallelize: public Pass
  {
  public:
    /// Driver functions to transform a program
    ///
    bool run(Module& M);

    /// getAnalysisUsage - Modifies extensively so preserve nothing.
    /// Uses the DependenceGraph and the Top-down DS Graph (only to find
    /// all functions called via an indirect call).
    ///
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TDDataStructures>();
      AU.addRequired<MemoryDepAnalysis>();  // force this not to be released
      AU.addRequired<PgmDependenceGraph>(); // because it is needed by this
    }
  };

  RegisterOpt<Parallelize> X("parallel", "Parallelize program using Cilk");
}


static Function* FindMain(Module& M)
{
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI)
    if (FI->getName() == std::string("main"))
      return FI;
  return NULL;
}


bool Parallelize::run(Module& M)
{
  hash_set<Function*> parallelFunctions;
  hash_set<Function*> safeParallelFunctions;
  hash_set<const GlobalValue*> indirectlyCalled;

  // If there is no main (i.e., for an incomplete program), we can do nothing.
  // If there is a main, mark main as a parallel function.
  // 
  Function* mainFunc = FindMain(M);
  if (!mainFunc)
    return false;

  // (1) Find candidate parallel functions and mark them as Cilk functions
  // 
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI)
    if (! FI->isExternal())
      {
        Function* F = FI;
        DSGraph& tdg = getAnalysis<TDDataStructures>().getDSGraph(*F);

        // All the hard analysis work gets done here!
        // 
        FindParallelCalls finder(*F,
                                getAnalysis<PgmDependenceGraph>().getGraph(*F));
                        /* getAnalysis<MemoryDepAnalysis>().getGraph(*F)); */

        // Now we know which call instructions are useful to parallelize.
        // Remember those callee functions.
        // 
        for (std::vector<CallInst*>::iterator
               CII = finder.parallelCalls.begin(),
               CIE = finder.parallelCalls.end(); CII != CIE; ++CII)
          {
            // Check if this is a direct call...
            if ((*CII)->getCalledFunction() != NULL)
              { // direct call: if this is to a non-external function,
                // mark it as a parallelizable function
                if (! (*CII)->getCalledFunction()->isExternal())
                  parallelFunctions.insert((*CII)->getCalledFunction());
              }
            else
              { // Indirect call: mark all potential callees as bad
                std::vector<GlobalValue*> callees =
                  tdg.getNodeForValue((*CII)->getCalledValue())
                  .getNode()->getGlobals();
                indirectlyCalled.insert(callees.begin(), callees.end());
              }
          }
      }

  // Remove all indirectly called functions from the list of Cilk functions.
  // 
  for (hash_set<Function*>::iterator PFI = parallelFunctions.begin(),
         PFE = parallelFunctions.end(); PFI != PFE; ++PFI)
    if (indirectlyCalled.count(*PFI) == 0)
      safeParallelFunctions.insert(*PFI);

#undef CAN_USE_BIND1ST_ON_REFERENCE_TYPE_ARGS
#ifdef CAN_USE_BIND1ST_ON_REFERENCE_TYPE_ARGS
  // Use this indecipherable STLese because erase invalidates iterators.
  // Otherwise we have to copy sets as above.
  hash_set<Function*>::iterator extrasBegin = 
    std::remove_if(parallelFunctions.begin(), parallelFunctions.end(),
                   compose1(std::bind2nd(std::greater<int>(), 0),
                            bind_obj(&indirectlyCalled,
                                     &hash_set<const GlobalValue*>::count)));
  parallelFunctions.erase(extrasBegin, parallelFunctions.end());
#endif

  // If there are no parallel functions, we can just give up.
  if (safeParallelFunctions.empty())
    return false;

  // Add main as a parallel function since Cilk requires this.
  safeParallelFunctions.insert(mainFunc);

  // (2,3) Transform each Cilk function and all its calls simply by
  //     adding a unique suffix to the function name.
  //     This should identify both functions and calls to such functions
  //     to the code generator.
  // (4) Also, insert calls to sync at appropriate points.
  // 
  Cilkifier cilkifier(M);
  for (hash_set<Function*>::iterator CFI = safeParallelFunctions.begin(),
         CFE = safeParallelFunctions.end(); CFI != CFE; ++CFI)
    {
      cilkifier.TransformFunc(*CFI, safeParallelFunctions,
                             getAnalysis<PgmDependenceGraph>().getGraph(**CFI));
      /* getAnalysis<MemoryDepAnalysis>().getGraph(**CFI)); */
    }

  return true;
}

