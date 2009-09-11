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
    std::set<BasicBlock*>  BBToVisit;
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

static void inline printEdgeError(ProfileInfo::Edge e, const char *M) {
  DEBUG(errs() << "-- Edge " << e << " is not calculated, " << M << "\n");
}

void inline ProfileEstimatorPass::printEdgeWeight(Edge E) {
  DEBUG(errs() << "-- Weight of Edge " << E << ":"
               << format("%g", getEdgeWeight(E)) << "\n");
}

// recurseBasicBlock() - This calculates the ProfileInfo estimation for a
// single block and then recurses into the successors.
// The algorithm preserves the flow condition, meaning that the sum of the
// weight of the incoming edges must be equal the block weight which must in
// turn be equal to the sume of the weights of the outgoing edges.
// Since the flow of an block is deterimined from the current state of the
// flow, once an edge has a flow assigned this flow is never changed again,
// otherwise it would be possible to violate the flow condition in another
// block.
void ProfileEstimatorPass::recurseBasicBlock(BasicBlock *BB) {

  // Break the recursion if this BasicBlock was already visited.
  if (BBToVisit.find(BB) == BBToVisit.end()) return;

  // Read the LoopInfo for this block.
  bool  BBisHeader = LI->isLoopHeader(BB);
  Loop* BBLoop     = LI->getLoopFor(BB);

  // To get the block weight, read all incoming edges.
  double BBWeight = 0;
  std::set<BasicBlock*> ProcessedPreds;
  for ( pred_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
        bbi != bbe; ++bbi ) {
    // If this block was not considered already, add weight.
    Edge edge = getEdge(*bbi,BB);
    double w = getEdgeWeight(edge);
    if (ProcessedPreds.insert(*bbi).second) {
      BBWeight += ignoreMissing(w);
    }
    // If this block is a loop header and the predecessor is contained in this
    // loop, thus the edge is a backedge, continue and do not check if the
    // value is valid.
    if (BBisHeader && BBLoop->contains(*bbi)) {
      printEdgeError(edge, "but is backedge, continueing");
      continue;
    }
    // If the edges value is missing (and this is no loop header, and this is
    // no backedge) return, this block is currently non estimatable.
    if (w == MissingValue) {
      printEdgeError(edge, "returning");
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

  // If this is a loop header, consider the following:
  // Exactly the flow that is entering this block, must exit this block too. So
  // do the following: 
  // *) get all the exit edges, read the flow that is already leaving this
  // loop, remember the edges that do not have any flow on them right now.
  // (The edges that have already flow on them are most likely exiting edges of
  // other loops, do not touch those flows because the previously caclulated
  // loopheaders would not be exact anymore.)
  // *) In case there is not a single exiting edge left, create one at the loop
  // latch to prevent the flow from building up in the loop.
  // *) Take the flow that is not leaving the loop already and distribute it on
  // the remaining exiting edges.
  // (This ensures that all flow that enters the loop also leaves it.)
  // *) Increase the flow into the loop by increasing the weight of this block.
  // There is at least one incoming backedge that will bring us this flow later
  // on. (So that the flow condition in this node is valid again.)
  if (BBisHeader) {
    double incoming = BBWeight;
    // Subtract the flow leaving the loop.
    std::set<Edge> ProcessedExits;
    for (SmallVector<Edge, 8>::iterator ei = ExitEdges.begin(),
         ee = ExitEdges.end(); ei != ee; ++ei) {
      if (ProcessedExits.insert(*ei).second) {
        double w = getEdgeWeight(*ei);
        if (w == MissingValue) {
          Edges.push_back(*ei);
        } else {
          incoming -= w;
        }
      }
    }
    // If no exit edges, create one:
    if (Edges.size() == 0) {
      BasicBlock *Latch = BBLoop->getLoopLatch();
      if (Latch) {
        Edge edge = getEdge(Latch,0);
        EdgeInformation[BB->getParent()][edge] = BBWeight;
        printEdgeWeight(edge);
        edge = getEdge(Latch, BB);
        EdgeInformation[BB->getParent()][edge] = BBWeight * ExecCount;
        printEdgeWeight(edge);
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

  BlockInformation[BB->getParent()][BB] = BBWeight;
  // Up until now we considered only the loop exiting edges, now we have a
  // definite block weight and must ditribute this onto the outgoing edges.
  // Since there may be already flow attached to some of the edges, read this
  // flow first and remember the edges that have still now flow attached.
  Edges.clear();
  std::set<BasicBlock*> ProcessedSuccs;

  succ_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
  // Also check for (BB,0) edges that may already contain some flow. (But only
  // in case there are no successors.)
  if (bbi == bbe) {
    Edge edge = getEdge(BB,0);
    EdgeInformation[BB->getParent()][edge] = BBWeight;
    printEdgeWeight(edge);
  }
  for ( ; bbi != bbe; ++bbi ) {
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

  // Finally we know what flow is still not leaving the block, distribute this
  // flow onto the empty edges.
  for (SmallVector<Edge, 8>::iterator ei = Edges.begin(), ee = Edges.end();
       ei != ee; ++ei) {
    EdgeInformation[BB->getParent()][*ei] += BBWeight/Edges.size();
    printEdgeWeight(*ei);
  }

  // This block is visited, mark this before the recursion.
  BBToVisit.erase(BB);

  // Recurse into successors.
  for (succ_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
       bbi != bbe; ++bbi) {
    recurseBasicBlock(*bbi);
  }
}

bool ProfileEstimatorPass::runOnFunction(Function &F) {
  if (F.isDeclaration()) return false;

  // Fetch LoopInfo and clear ProfileInfo for this function.
  LI = &getAnalysis<LoopInfo>();
  FunctionInformation.erase(&F);
  BlockInformation[&F].clear();
  EdgeInformation[&F].clear();

  // Mark all blocks as to visit.
  for (Function::iterator bi = F.begin(), be = F.end(); bi != be; ++bi)
    BBToVisit.insert(bi);

  DEBUG(errs() << "Working on function " << F.getNameStr() << "\n");

  // Since the entry block is the first one and has no predecessors, the edge
  // (0,entry) is inserted with the starting weight of 1.
  BasicBlock *entry = &F.getEntryBlock();
  BlockInformation[&F][entry] = 1;
  Edge edge = getEdge(0,entry);
  EdgeInformation[&F][edge] = 1;
  printEdgeWeight(edge);

  // Since recurseBasicBlock() maybe returns with a block which was not fully
  // estimated, use recurseBasicBlock() until everything is calculated. 
  recurseBasicBlock(entry);
  while (BBToVisit.size() > 0) {
    // Remember number of open blocks, this is later used to check if progress
    // was made.
    unsigned size = BBToVisit.size();

    // Try to calculate all blocks in turn.
    for (std::set<BasicBlock*>::iterator bi = BBToVisit.begin(),
         be = BBToVisit.end(); bi != be; ++bi) {
      recurseBasicBlock(*bi);
      // If at least one block was finished, break because iterator may be
      // invalid.
      if (BBToVisit.size() < size) break;
    }

    // If there was not a single block resovled, make some assumptions.
    if (BBToVisit.size() == size) {
      BasicBlock *BB = *(BBToVisit.begin());
      // Since this BB was not calculated because of missing incoming edges,
      // set these edges to zero.
      for (pred_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
           bbi != bbe; ++bbi) {
        Edge e = getEdge(*bbi,BB);
        double w = getEdgeWeight(e);
        if (w == MissingValue) {
          EdgeInformation[&F][e] = 0;
          DEBUG(errs() << "Assuming edge weight: ");
          printEdgeWeight(e);
        }
      }
    }
  }

  return false;
}
