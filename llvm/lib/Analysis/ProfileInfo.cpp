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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
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
  struct VISIBILITY_HIDDEN NoProfileInfo 
    : public ImmutablePass, public ProfileInfo {
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
