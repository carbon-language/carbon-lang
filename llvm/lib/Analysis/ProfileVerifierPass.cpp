//===- ProfileVerifierPass.cpp - LLVM Pass to estimate profile info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that checks profiling information for 
// plausibility.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "profile-verifier"
#include "llvm/Pass.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include <set>
using namespace llvm;

static bool DisableAssertions = false;
static cl::opt<bool,true>
ProfileVerifierDisableAssertions("profile-verifier-noassert",
    cl::location(DisableAssertions), cl::desc("Disable assertions"));
bool PrintedDebugTree = false;

namespace {
  class VISIBILITY_HIDDEN ProfileVerifierPass : public FunctionPass {
    ProfileInfo *PI;
    std::set<const BasicBlock*> BBisVisited;
#ifndef NDEBUG
    std::set<const BasicBlock*> BBisPrinted;
    void debugEntry(const BasicBlock* BB, double w, double inw, int inc,
                    double outw, int outc, double d);
    void printDebugInfo(const BasicBlock *BB);
#endif
  public:
    static char ID; // Class identification, replacement for typeinfo

    explicit ProfileVerifierPass () : FunctionPass(&ID) {
      DisableAssertions = ProfileVerifierDisableAssertions;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<ProfileInfo>();
    }

    const char *getPassName() const {
      return "Profiling information verifier";
    }

    /// run - Verify the profile information.
    bool runOnFunction(Function &F);
    void recurseBasicBlock(const BasicBlock *BB);
  };
}  // End of anonymous namespace

char ProfileVerifierPass::ID = 0;
static RegisterPass<ProfileVerifierPass>
X("profile-verifier", "Verify profiling information", false, true);

namespace llvm {
  FunctionPass *createProfileVerifierPass() {
    return new ProfileVerifierPass(); 
  }
}

#ifndef NDEBUG
void ProfileVerifierPass::printDebugInfo(const BasicBlock *BB) {

  if (BBisPrinted.find(BB) != BBisPrinted.end()) return;

  double BBWeight = PI->getExecutionCount(BB);
  if (BBWeight == ProfileInfo::MissingValue) { BBWeight = 0; }
  double inWeight = 0;
  int inCount = 0;
  std::set<const BasicBlock*> ProcessedPreds;
  for ( pred_const_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
        bbi != bbe; ++bbi ) {
    if (ProcessedPreds.insert(*bbi).second) {
      double EdgeWeight = PI->getEdgeWeight(PI->getEdge(*bbi,BB));
      if (EdgeWeight == ProfileInfo::MissingValue) { EdgeWeight = 0; }
      DEBUG(errs()<<"calculated in-edge ("<<(*bbi)->getNameStr()<<","<<BB->getNameStr()
          <<"): "<<EdgeWeight<<"\n");
      inWeight += EdgeWeight;
      inCount++;
    }
  }
  double outWeight = 0;
  int outCount = 0;
  std::set<const BasicBlock*> ProcessedSuccs;
  for ( succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
        bbi != bbe; ++bbi ) {
    if (ProcessedSuccs.insert(*bbi).second) {
      double EdgeWeight = PI->getEdgeWeight(PI->getEdge(BB,*bbi));
      if (EdgeWeight == ProfileInfo::MissingValue) { EdgeWeight = 0; }
      DEBUG(errs()<<"calculated out-edge ("<<BB->getNameStr()<<","<<(*bbi)->getNameStr()
          <<"): "<<EdgeWeight<<"\n");
      outWeight += EdgeWeight;
      outCount++;
    }
  }
  DEBUG(errs()<<"Block "<<BB->getNameStr()<<" in "<<BB->getParent()->getNameStr()
      <<",BBWeight="<<BBWeight<<",inWeight="<<inWeight<<",inCount="<<inCount
      <<",outWeight="<<outWeight<<",outCount"<<outCount<<"\n");

  // mark as visited and recurse into subnodes
  BBisPrinted.insert(BB);
  for ( succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB); 
        bbi != bbe; ++bbi ) {
    printDebugInfo(*bbi);
  }
}

void ProfileVerifierPass::debugEntry (const BasicBlock* BB, double w, 
                                      double inw,  int inc, double outw, int
                                      outc, double d) {
  DEBUG(errs()<<"TROUBLE: Block "<<BB->getNameStr()<<" in "<<BB->getParent()->getNameStr()
      <<",BBWeight="<<w<<",inWeight="<<inw<<",inCount="<<inc<<",outWeight="
      <<outw<<",outCount"<<outc<<"\n");
  DEBUG(errs()<<"DELTA:"<<d<<"\n");
  if (!PrintedDebugTree) {
    PrintedDebugTree = true;
    printDebugInfo(&(BB->getParent()->getEntryBlock()));
  }
}
#endif

// compare with relative error
static bool dcmp(double A, double B) { 
  double maxRelativeError = 0.0000001;
  if (A == B)
    return true;
  double relativeError;
  if (fabs(B) > fabs(A)) 
    relativeError = fabs((A - B) / B);
  else 
    relativeError = fabs((A - B) / A);
  if (relativeError <= maxRelativeError) return true; 
  return false; 
}

#define CHECK(C,M) \
if (C) { \
  if (DisableAssertions) { errs()<<(M)<<"\n"; } else { assert((!(C)) && (M)); } \
}

#define CHECKDEBUG(C,M,D) \
if (C) { \
  DEBUG(debugEntry(BB, BBWeight, inWeight,  inCount, \
                                 outWeight, outCount, (D))); \
  if (DisableAssertions) { errs()<<(M)<<"\n"; } else { assert((!(C)) && (M)); } \
}

void ProfileVerifierPass::recurseBasicBlock(const BasicBlock *BB) {

  if (BBisVisited.find(BB) != BBisVisited.end()) return;

  double inWeight = 0;
  int inCount = 0;
  std::set<const BasicBlock*> ProcessedPreds;
  for ( pred_const_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
        bbi != bbe; ++bbi ) {
    if (ProcessedPreds.insert(*bbi).second) {
      double EdgeWeight = PI->getEdgeWeight(PI->getEdge(*bbi,BB));
      CHECK(EdgeWeight == ProfileInfo::MissingValue,
            "ASSERT:Edge has missing value");
      inWeight += EdgeWeight; inCount++;
    }
  }

  double outWeight = 0;
  int outCount = 0;
  std::set<const BasicBlock*> ProcessedSuccs;
  for ( succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
        bbi != bbe; ++bbi ) {
    if (ProcessedSuccs.insert(*bbi).second) {
      double EdgeWeight = PI->getEdgeWeight(PI->getEdge(BB,*bbi));
      CHECK(EdgeWeight == ProfileInfo::MissingValue,
            "ASSERT:Edge has missing value");
      outWeight += EdgeWeight; outCount++;
    }
  }

  double BBWeight = PI->getExecutionCount(BB);
  CHECKDEBUG(BBWeight == ProfileInfo::MissingValue,
             "ASSERT:BasicBlock has missing value",-1);

  if (inCount > 0) {
    CHECKDEBUG(!dcmp(inWeight,BBWeight),
        "ASSERT:inWeight and BBWeight do not match",inWeight-BBWeight);
  }
  if (outCount > 0) {
    CHECKDEBUG(!dcmp(outWeight,BBWeight),
        "ASSERT:outWeight and BBWeight do not match",outWeight-BBWeight);
  }

  // mark as visited and recurse into subnodes
  BBisVisited.insert(BB);
  for ( succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB); 
        bbi != bbe; ++bbi ) {
    recurseBasicBlock(*bbi);
  }
}

bool ProfileVerifierPass::runOnFunction(Function &F) {
  PI = &getAnalysis<ProfileInfo>();

  if (PI->getExecutionCount(&F) == ProfileInfo::MissingValue) {
    DEBUG(errs()<<"Function "<<F.getNameStr()<<" has no profile\n");
    return false;
  }

  PrintedDebugTree = false;
  BBisVisited.clear();

  const BasicBlock *entry = &F.getEntryBlock();
  recurseBasicBlock(entry);

  if (!DisableAssertions)
    assert((PI->getExecutionCount(&F)==PI->getExecutionCount(entry)) &&
           "Function count and entry block count do not match");
  return false;
}
