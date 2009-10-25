//===- ProfileInfo.cpp - Profile Info Interface ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the abstract ProfileInfo interface, and the default
// "no profile" implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include <set>
using namespace llvm;

// Register the ProfileInfo interface, providing a nice name to refer to.
static RegisterAnalysisGroup<ProfileInfo> Z("Profile Information");
char ProfileInfo::ID = 0;

ProfileInfo::~ProfileInfo() {}

const double ProfileInfo::MissingValue = -1;

double ProfileInfo::getExecutionCount(const BasicBlock *BB) {
  std::map<const Function*, BlockCounts>::iterator J =
    BlockInformation.find(BB->getParent());
  if (J != BlockInformation.end()) {
    BlockCounts::iterator I = J->second.find(BB);
    if (I != J->second.end())
      return I->second;
  }

  pred_const_iterator PI = pred_begin(BB), PE = pred_end(BB);

  // Are there zero predecessors of this block?
  if (PI == PE) {
    // If this is the entry block, look for the Null -> Entry edge.
    if (BB == &BB->getParent()->getEntryBlock())
      return getEdgeWeight(getEdge(0, BB));
    else
      return 0;   // Otherwise, this is a dead block.
  }

  // Otherwise, if there are predecessors, the execution count of this block is
  // the sum of the edge frequencies from the incoming edges.
  std::set<const BasicBlock*> ProcessedPreds;
  double Count = 0;
  for (; PI != PE; ++PI)
    if (ProcessedPreds.insert(*PI).second) {
      double w = getEdgeWeight(getEdge(*PI, BB));
      if (w == MissingValue) {
        Count = MissingValue;
        break;
      }
      Count += w;
    }

  if (Count != MissingValue) BlockInformation[BB->getParent()][BB] = Count;
  return Count;
}

double ProfileInfo::getExecutionCount(const Function *F) {
  std::map<const Function*, double>::iterator J =
    FunctionInformation.find(F);
  if (J != FunctionInformation.end())
    return J->second;

  // isDeclaration() is checked here and not at start of function to allow
  // functions without a body still to have a execution count.
  if (F->isDeclaration()) return MissingValue;

  double Count = getExecutionCount(&F->getEntryBlock());
  if (Count != MissingValue) FunctionInformation[F] = Count;
  return Count;
}

/// Replaces all occurences of RmBB in the ProfilingInfo with DestBB.
/// This checks all edges of the function the blocks reside in and replaces the
/// occurences of RmBB with DestBB.
void ProfileInfo::replaceAllUses(const BasicBlock *RmBB, 
                                 const BasicBlock *DestBB) {
  DEBUG(errs() << "Replacing " << RmBB->getNameStr()
               << " with " << DestBB->getNameStr() << "\n");
  const Function *F = DestBB->getParent();
  std::map<const Function*, EdgeWeights>::iterator J =
    EdgeInformation.find(F);
  if (J == EdgeInformation.end()) return;

  for (EdgeWeights::iterator I = J->second.begin(), E = J->second.end();
       I != E; ++I) {
    Edge e = I->first;
    Edge newedge; bool foundedge = false;
    if (e.first == RmBB) {
      newedge = getEdge(DestBB, e.second);
      foundedge = true;
    }
    if (e.second == RmBB) {
      newedge = getEdge(e.first, DestBB);
      foundedge = true;
    }
    if (foundedge) {
      double w = getEdgeWeight(e);
      EdgeInformation[F][newedge] = w;
      DEBUG(errs() << "Replacing " << e << " with " << newedge  << "\n");
      J->second.erase(e);
    }
  }
}

/// Splits an edge in the ProfileInfo and redirects flow over NewBB.
/// Since its possible that there is more than one edge in the CFG from FristBB
/// to SecondBB its necessary to redirect the flow proporionally.
void ProfileInfo::splitEdge(const BasicBlock *FirstBB,
                            const BasicBlock *SecondBB,
                            const BasicBlock *NewBB,
                            bool MergeIdenticalEdges) {
  const Function *F = FirstBB->getParent();
  std::map<const Function*, EdgeWeights>::iterator J =
    EdgeInformation.find(F);
  if (J == EdgeInformation.end()) return;

  // Generate edges and read current weight.
  Edge e  = getEdge(FirstBB, SecondBB);
  Edge n1 = getEdge(FirstBB, NewBB);
  Edge n2 = getEdge(NewBB, SecondBB);
  EdgeWeights &ECs = J->second;
  double w = ECs[e];

  int succ_count = 0;
  if (!MergeIdenticalEdges) {
    // First count the edges from FristBB to SecondBB, if there is more than
    // one, only slice out a proporional part for NewBB.
    for(succ_const_iterator BBI = succ_begin(FirstBB), BBE = succ_end(FirstBB);
        BBI != BBE; ++BBI) {
      if (*BBI == SecondBB) succ_count++;  
    }
    // When the NewBB is completely new, increment the count by one so that
    // the counts are properly distributed.
    if (getExecutionCount(NewBB) == ProfileInfo::MissingValue) succ_count++;
  } else {
    // When the edges are merged anyway, then redirect all flow.
    succ_count = 1;
  }

  // We know now how many edges there are from FirstBB to SecondBB, reroute a
  // proportional part of the edge weight over NewBB.
  double neww = w / succ_count;
  ECs[n1] += neww;
  ECs[n2] += neww;
  BlockInformation[F][NewBB] += neww;
  if (succ_count == 1) {
    ECs.erase(e);
  } else {
    ECs[e] -= neww;
  }
}

raw_ostream& llvm::operator<<(raw_ostream &O, ProfileInfo::Edge E) {
  O << "(";
  O << (E.first ? E.first->getNameStr() : "0");
  O << ",";
  O << (E.second ? E.second->getNameStr() : "0");
  return O << ")";
}

//===----------------------------------------------------------------------===//
//  NoProfile ProfileInfo implementation
//

namespace {
  struct NoProfileInfo : public ImmutablePass, public ProfileInfo {
    static char ID; // Class identification, replacement for typeinfo
    NoProfileInfo() : ImmutablePass(&ID) {}
  };
}  // End of anonymous namespace

char NoProfileInfo::ID = 0;
// Register this pass...
static RegisterPass<NoProfileInfo>
X("no-profile", "No Profile Information", false, true);

// Declare that we implement the ProfileInfo interface
static RegisterAnalysisGroup<ProfileInfo, true> Y(X);

ImmutablePass *llvm::createNoProfileInfoPass() { return new NoProfileInfo(); }
