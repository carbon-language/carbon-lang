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

static cl::opt<bool,true>
ProfileVerifierDisableAssertions("profile-verifier-noassert",
     cl::desc("Disable assertions"));

namespace {
  class VISIBILITY_HIDDEN ProfileVerifierPass : public FunctionPass {

    struct DetailedBlockInfo {
      const BasicBlock *BB;
      double            BBWeight;
      double            inWeight;
      int               inCount;
      double            outWeight;
      int               outCount;
    };

    ProfileInfo *PI;
    std::set<const BasicBlock*> BBisVisited;
    bool DisableAssertions;

    // When debugging is enabled, the verifier prints a whole slew of debug
    // information, otherwise its just the assert. These are all the helper
    // functions.
    bool PrintedDebugTree;
    std::set<const BasicBlock*> BBisPrinted;
    void debugEntry(DetailedBlockInfo*);
    void printDebugInfo(const BasicBlock *BB);

  public:
    static char ID; // Class identification, replacement for typeinfo

    explicit ProfileVerifierPass (bool da = false) : FunctionPass(&ID), 
                                                     DisableAssertions(da) {
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

    double ReadOrAssert(ProfileInfo::Edge);
    void   CheckValue(bool, const char*, DetailedBlockInfo*);
  };
}  // End of anonymous namespace

char ProfileVerifierPass::ID = 0;
static RegisterPass<ProfileVerifierPass>
X("profile-verifier", "Verify profiling information", false, true);

namespace llvm {
  FunctionPass *createProfileVerifierPass() {
    return new ProfileVerifierPass(ProfileVerifierDisableAssertions); 
  }
}

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
      errs()<<"calculated in-edge ("<<(*bbi)->getNameStr()<<","<<BB->getNameStr()
            <<"): "<<EdgeWeight<<"\n";
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
      errs()<<"calculated out-edge ("<<BB->getNameStr()<<","<<(*bbi)->getNameStr()
            <<"): "<<EdgeWeight<<"\n";
      outWeight += EdgeWeight;
      outCount++;
    }
  }
  errs()<<"Block "<<BB->getNameStr()<<" in "<<BB->getParent()->getNameStr()
        <<",BBWeight="<<BBWeight<<",inWeight="<<inWeight<<",inCount="<<inCount
        <<",outWeight="<<outWeight<<",outCount"<<outCount<<"\n";

  // mark as visited and recurse into subnodes
  BBisPrinted.insert(BB);
  for ( succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB); 
        bbi != bbe; ++bbi ) {
    printDebugInfo(*bbi);
  }
}

void ProfileVerifierPass::debugEntry (DetailedBlockInfo *DI) {
  errs() << "TROUBLE: Block " << DI->BB->getNameStr() << " in "
         << DI->BB->getParent()->getNameStr()  << ":";
  errs() << "BBWeight="  << DI->BBWeight   << ",";
  errs() << "inWeight="  << DI->inWeight   << ",";
  errs() << "inCount="   << DI->inCount    << ",";
  errs() << "outWeight=" << DI->outWeight  << ",";
  errs() << "outCount="  << DI->outCount   << ",";
  if (!PrintedDebugTree) {
    PrintedDebugTree = true;
    printDebugInfo(&(DI->BB->getParent()->getEntryBlock()));
  }
}

// compare with relative error
static bool Equals(double A, double B) { 
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

double ProfileVerifierPass::ReadOrAssert(ProfileInfo::Edge E) {
  const char *Message = "ASSERT:Edge has missing value";
  double EdgeWeight = PI->getEdgeWeight(E);
  if (EdgeWeight == ProfileInfo::MissingValue) {
    if (DisableAssertions) {
      errs() << Message << "\n";
      return 0;
    } else { 
      assert(0 && Message);
    }
  } else {
    return EdgeWeight;
  }
}

void ProfileVerifierPass::CheckValue(bool Error, const char *Message, DetailedBlockInfo *DI) {
  if (Error) {
    DEBUG(debugEntry(DI));
    if (DisableAssertions) {
      errs() << Message << "\n";
    } else { 
      assert(0 && Message);
    }
  }
  return;
}

void ProfileVerifierPass::recurseBasicBlock(const BasicBlock *BB) {

  if (BBisVisited.find(BB) != BBisVisited.end()) return;

  DetailedBlockInfo DI;
  DI.BB = BB;
  DI.outCount = DI.inCount = DI.inWeight = DI.outWeight = 0;
  std::set<const BasicBlock*> ProcessedPreds;
  for ( pred_const_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
        bbi != bbe; ++bbi ) {
    if (ProcessedPreds.insert(*bbi).second) {
      DI.inWeight += ReadOrAssert(PI->getEdge(*bbi,BB));
      DI.inCount++;
    }
  }

  std::set<const BasicBlock*> ProcessedSuccs;
  for ( succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
        bbi != bbe; ++bbi ) {
    if (ProcessedSuccs.insert(*bbi).second) {
      DI.outWeight += ReadOrAssert(PI->getEdge(BB,*bbi));
      DI.outCount++;
    }
  }

  DI.BBWeight = PI->getExecutionCount(BB);
  CheckValue(DI.BBWeight == ProfileInfo::MissingValue, "ASSERT:BasicBlock has missing value", &DI);

  if (DI.inCount > 0) {
    CheckValue(!Equals(DI.inWeight,DI.BBWeight), "ASSERT:inWeight and BBWeight do not match", &DI);
  }
  if (DI.outCount > 0) {
    CheckValue(!Equals(DI.outWeight,DI.BBWeight), "ASSERT:outWeight and BBWeight do not match", &DI);
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
