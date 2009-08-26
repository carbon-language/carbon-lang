//===- ProfileEstimatorPass.cpp - LLVM Pass to estimate profile info ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a concrete implementation of profiling information that
// estimates the profiling information in a very crude and unimaginative way.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "profile-estimator"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
using namespace llvm;

static cl::opt<double>
LoopWeight(
    "profile-estimator-loop-weight", cl::init(10),
    cl::value_desc("loop-weight"),
    cl::desc("Number of loop executions used for profile-estimator")
);

namespace {
  class VISIBILITY_HIDDEN ProfileEstimatorPass :
      public FunctionPass, public ProfileInfo {
    double ExecCount;
    LoopInfo *LI;
    std::set<BasicBlock*>  BBisVisited;
    std::map<Loop*,double> LoopExitWeights;
  public:
    static char ID; // Class identification, replacement for typeinfo
    explicit ProfileEstimatorPass(const double execcount = 0)
      : FunctionPass(&ID), ExecCount(execcount) {
      if (execcount == 0) ExecCount = LoopWeight;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<LoopInfo>();
    }

    virtual const char *getPassName() const {
      return "Profiling information estimator";
    }

    /// run - Estimate the profile information from the specified file.
    virtual bool runOnFunction(Function &F);

    virtual void recurseBasicBlock(BasicBlock *BB);

    void inline printEdgeWeight(Edge);
  };
}  // End of anonymous namespace

char ProfileEstimatorPass::ID = 0;
static RegisterPass<ProfileEstimatorPass>
X("profile-estimator", "Estimate profiling information", false, true);

static RegisterAnalysisGroup<ProfileInfo> Y(X);

namespace llvm {
  const PassInfo *ProfileEstimatorPassID = &X;

  FunctionPass *createProfileEstimatorPass() {
    return new ProfileEstimatorPass();
  }

  /// createProfileEstimatorPass - This function returns a Pass that estimates
  /// profiling information using the given loop execution count.
  Pass *createProfileEstimatorPass(const unsigned execcount) {
    return new ProfileEstimatorPass(execcount);
  }
}

static double ignoreMissing(double w) {
  if (w == ProfileInfo::MissingValue) return 0;
  return w;
}

static void inline printEdgeError(BasicBlock *V1, BasicBlock *V2) {
  DEBUG(errs() << "-- Edge (" <<(V1)->getName() << "," << (V2)->getName() \
               << ") is not calculated, returning\n");
}

void inline ProfileEstimatorPass::printEdgeWeight(Edge E) {
  DEBUG(errs() << "-- Weight of Edge " << E << ":"
               << format("%g", getEdgeWeight(E)) << "\n");
}

// recurseBasicBlock() - This calculates the ProfileInfo estimation for a
// single block and then recurses into the successors.
void ProfileEstimatorPass::recurseBasicBlock(BasicBlock *BB) {

  // Break the recursion if this BasicBlock was already visited.
  if (BBisVisited.find(BB) != BBisVisited.end()) return;

  // Check if incoming edges are calculated already, if BB is header allow
  // backedges that are uncalculated for now.
  bool  BBisHeader = LI->isLoopHeader(BB);
  Loop* BBLoop     = LI->getLoopFor(BB);

  double BBWeight = 0;
  std::set<BasicBlock*> ProcessedPreds;
  for ( pred_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
        bbi != bbe; ++bbi ) {
    if (ProcessedPreds.insert(*bbi).second) {
      Edge edge = getEdge(*bbi,BB);
      BBWeight += ignoreMissing(getEdgeWeight(edge));
    }
    if (BBisHeader && BBLoop == LI->getLoopFor(*bbi)) {
      printEdgeError(*bbi,BB);
      continue;
    }
    if (BBisVisited.find(*bbi) == BBisVisited.end()) {
      printEdgeError(*bbi,BB);
      return;
    }
  }
  if (getExecutionCount(BB) != MissingValue) {
    BBWeight = getExecutionCount(BB);
  }

  // Fetch all necessary information for current block.
  SmallVector<Edge, 8> ExitEdges;
  SmallVector<Edge, 8> Edges;
  if (BBLoop) {
    BBLoop->getExitEdges(ExitEdges);
  }

  // If block is an loop header, first subtract all weights from edges that
  // exit this loop, then distribute remaining weight on to the edges exiting
  // this loop. Finally the weight of the block is increased, to simulate
  // several executions of this loop.
  if (BBisHeader) {
    double incoming = BBWeight;
    // Subtract the flow leaving the loop.
    for (SmallVector<Edge, 8>::iterator ei = ExitEdges.begin(),
         ee = ExitEdges.end(); ei != ee; ++ei) {
      double w = getEdgeWeight(*ei);
      if (w == MissingValue) {
        Edges.push_back(*ei);
      } else {
        incoming -= w;
      }
    }
    // Distribute remaining weight onto the exit edges.
    for (SmallVector<Edge, 8>::iterator ei = Edges.begin(), ee = Edges.end();
         ei != ee; ++ei) {
      EdgeInformation[BB->getParent()][*ei] += incoming/Edges.size();
      printEdgeWeight(*ei);
    }
    // Increase flow into the loop.
    BBWeight *= (ExecCount+1);
  }

  // Remove from current flow of block all the successor edges that already
  // have some flow on them.
  Edges.clear();
  std::set<BasicBlock*> ProcessedSuccs;

  // Otherwise consider weight of outgoing edges and store them for
  // distribution of remaining weight.
  for ( succ_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
        bbi != bbe; ++bbi ) {
    if (ProcessedSuccs.insert(*bbi).second) {
      Edge edge = getEdge(BB,*bbi);
      double w = getEdgeWeight(edge);
      if (w != MissingValue) {
        BBWeight -= getEdgeWeight(edge);
      } else {
        Edges.push_back(edge);
      }
    }
  }

  // Distribute remaining flow onto the outgoing edges.
  for (SmallVector<Edge, 8>::iterator ei = Edges.begin(), ee = Edges.end();
       ei != ee; ++ei) {
    EdgeInformation[BB->getParent()][*ei] += BBWeight/Edges.size();
    printEdgeWeight(*ei);
  }

  // Mark this Block visited and recurse into successors.
  BBisVisited.insert(BB);
  for ( succ_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
        bbi != bbe;
        ++bbi ) {
    recurseBasicBlock(*bbi);
  }
}

bool ProfileEstimatorPass::runOnFunction(Function &F) {
  if (F.isDeclaration()) return false;

  LI = &getAnalysis<LoopInfo>();
  FunctionInformation.erase(&F);
  BlockInformation[&F].clear();
  EdgeInformation[&F].clear();
  BBisVisited.clear();

  DEBUG(errs() << "Working on function " << F.getNameStr() << "\n");

  // Since the entry block is the first one and has no predecessors, the edge
  // (0,entry) is inserted with the starting weight of 1.
  BasicBlock *entry = &F.getEntryBlock();
  BlockInformation[&F][entry] = 1;

  Edge edge = getEdge(0,entry);
  EdgeInformation[&F][edge] = 1; printEdgeWeight(edge);
  recurseBasicBlock(entry);

  // In case something went wrong, clear all results, not profiling info is
  // available.
  if (BBisVisited.size() != F.size()) {
    DEBUG(errs() << "-- could not estimate profile, using default profile\n");
    FunctionInformation.erase(&F);
    BlockInformation[&F].clear();
    for (Function::iterator BB = F.begin(), BBE = F.end(); BB != BBE; ++BB) {
      for (pred_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
           bbi != bbe; ++bbi) {
        Edge e = getEdge(*bbi,BB);
        EdgeInformation[&F][e] = 1; 
        printEdgeWeight(e);
      }
      for (succ_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
           bbi != bbe; ++bbi) {
        Edge e = getEdge(BB,*bbi);
        EdgeInformation[&F][e] = 1;
        printEdgeWeight(e);
      }
    }
  }

  return false;
}
