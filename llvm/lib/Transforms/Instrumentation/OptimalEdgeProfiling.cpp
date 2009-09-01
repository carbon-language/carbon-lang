//===- OptimalEdgeProfiling.cpp - Insert counters for opt. edge profiling -===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass instruments the specified program with counters for edge profiling.
// Edge profiling can give a reasonable approximation of the hot paths through a
// program, and is used for a wide variety of program transformations.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "insert-optimal-edge-profiling"
#include "ProfilingUtils.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/Statistic.h"
#include "MaximumSpanningTree.h"
#include <set>
using namespace llvm;

STATISTIC(NumEdgesInserted, "The # of edges inserted.");

namespace {
  class VISIBILITY_HIDDEN OptimalEdgeProfiler : public ModulePass {
    bool runOnModule(Module &M);
    ProfileInfo *PI;
  public:
    static char ID; // Pass identification, replacement for typeid
    OptimalEdgeProfiler() : ModulePass(&ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(ProfileEstimatorPassID);
      AU.addRequired<ProfileInfo>();
    }

    virtual const char *getPassName() const {
      return "Optimal Edge Profiler";
    }
  };
}

char OptimalEdgeProfiler::ID = 0;
static RegisterPass<OptimalEdgeProfiler>
X("insert-optimal-edge-profiling", 
  "Insert optimal instrumentation for edge profiling");

ModulePass *llvm::createOptimalEdgeProfilerPass() {
  return new OptimalEdgeProfiler();
}

inline static void printEdgeCounter(ProfileInfo::Edge e,
                                    BasicBlock* b,
                                    unsigned i) {
  DEBUG(errs() << "--Edge Counter for " << (e) << " in " \
               << ((b)?(b)->getNameStr():"0") << " (# " << (i) << ")\n");
}

bool OptimalEdgeProfiler::runOnModule(Module &M) {
  Function *Main = M.getFunction("main");
  if (Main == 0) {
    errs() << "WARNING: cannot insert edge profiling into a module"
           << " with no main function!\n";
    return false;  // No main, no instrumentation!
  }

  std::set<BasicBlock*> BlocksToInstrument;
  unsigned NumEdges = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration()) continue;
    // Reserve space for (0,entry) edge.
    ++NumEdges;
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      // Keep track of which blocks need to be instrumented.  We don't want to
      // instrument blocks that are added as the result of breaking critical
      // edges!
      BlocksToInstrument.insert(BB);
      if (BB->getTerminator()->getNumSuccessors() == 0) {
        // Reserve space for (BB,0) edge.
        ++NumEdges;
      } else {
        NumEdges += BB->getTerminator()->getNumSuccessors();
      }
    }
  }

  const Type *Int32 = Type::getInt32Ty(M.getContext());
  const ArrayType *ATy = ArrayType::get(Int32, NumEdges);
  GlobalVariable *Counters =
    new GlobalVariable(M, ATy, false, GlobalValue::InternalLinkage,
                       Constant::getNullValue(ATy), "OptEdgeProfCounters");
  NumEdgesInserted = 0;

  std::vector<Constant*> Initializer(NumEdges);
  Constant* zeroc = ConstantInt::get(Int32, 0);
  Constant* minusonec = ConstantInt::get(Int32, ProfileInfo::MissingValue);

  // Instrument all of the edges not in MST...
  unsigned i = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration()) continue;
    DEBUG(errs()<<"Working on "<<F->getNameStr()<<"\n");

    PI = &getAnalysisID<ProfileInfo>(ProfileEstimatorPassID,*F);
    MaximumSpanningTree MST = MaximumSpanningTree(&(*F),PI,true);

    // Create counter for (0,entry) edge.
    BasicBlock *entry = &(F->getEntryBlock());
    ProfileInfo::Edge edge = ProfileInfo::getEdge(0,entry);
    if (std::binary_search(MST.begin(),MST.end(),edge)) {
      printEdgeCounter(edge,entry,i);
      IncrementCounterInBlock(entry, i, Counters); NumEdgesInserted++;
      Initializer[i++] = (zeroc);
    } else{
      Initializer[i++] = (minusonec);
    }

    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      if (!BlocksToInstrument.count(BB)) continue; // Don't count new blocks
      // Okay, we have to add a counter of each outgoing edge not in MST. If
      // the outgoing edge is not critical don't split it, just insert the
      // counter in the source or destination of the edge.
      TerminatorInst *TI = BB->getTerminator();
      if (TI->getNumSuccessors() == 0) {
        // Create counter for (BB,0), edge.
        ProfileInfo::Edge edge = ProfileInfo::getEdge(BB,0);
        if (std::binary_search(MST.begin(),MST.end(),edge)) {
          printEdgeCounter(edge,BB,i);
          IncrementCounterInBlock(BB, i, Counters); NumEdgesInserted++;
          Initializer[i++] = (zeroc);
        } else{
          Initializer[i++] = (minusonec);
        }
      }
      for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s) {
        BasicBlock *Succ = TI->getSuccessor(s);
        ProfileInfo::Edge edge = ProfileInfo::getEdge(BB,Succ);
        if (std::binary_search(MST.begin(),MST.end(),edge)) {

          // If the edge is critical, split it.
          SplitCriticalEdge(TI,s,this);
          Succ = TI->getSuccessor(s);

          // Okay, we are guaranteed that the edge is no longer critical.  If we
          // only have a single successor, insert the counter in this block,
          // otherwise insert it in the successor block.
          if (TI->getNumSuccessors() == 1) {
            // Insert counter at the start of the block
            printEdgeCounter(edge,BB,i);
            IncrementCounterInBlock(BB, i, Counters); NumEdgesInserted++;
          } else {
            // Insert counter at the start of the block
            printEdgeCounter(edge,Succ,i);
            IncrementCounterInBlock(Succ, i, Counters); NumEdgesInserted++;
          }
          Initializer[i++] = (zeroc);
        } else {
          Initializer[i++] = (minusonec);
        }
      }
    }
  }

  // check if indeed all counters have been used
  assert(i==NumEdges && "the number of edges in counting array is wrong");

  // assign initialiser to array
  Constant *init = ConstantArray::get(ATy, Initializer);
  Counters->setInitializer(init);

  // Add the initialization call to main.
  InsertProfilingInitCall(Main, "llvm_start_opt_edge_profiling", Counters);
  return true;
}

