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
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include <set>
using namespace llvm;

static cl::opt<bool,false>
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
    std::set<const Function*>   FisVisited;
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

    explicit ProfileVerifierPass () : FunctionPass(&ID) {
      DisableAssertions = ProfileVerifierDisableAssertions;
    }
    explicit ProfileVerifierPass (bool da) : FunctionPass(&ID), 
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
    void recurseBasicBlock(const BasicBlock*);

    bool   exitReachable(const Function*);
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
      ProfileInfo::Edge E = PI->getEdge(*bbi,BB);
      double EdgeWeight = PI->getEdgeWeight(E);
      if (EdgeWeight == ProfileInfo::MissingValue) { EdgeWeight = 0; }
      errs() << "calculated in-edge " << E << ": " << EdgeWeight << "\n";
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
      ProfileInfo::Edge E = PI->getEdge(BB,*bbi);
      double EdgeWeight = PI->getEdgeWeight(E);
      if (EdgeWeight == ProfileInfo::MissingValue) { EdgeWeight = 0; }
      errs() << "calculated out-edge " << E << ": " << EdgeWeight << "\n";
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
  errs() << "outCount="  << DI->outCount   << "\n";
  if (!PrintedDebugTree) {
    PrintedDebugTree = true;
    printDebugInfo(&(DI->BB->getParent()->getEntryBlock()));
  }
}

// This compares A and B but considering maybe small differences.
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

// This checks if the function "exit" is reachable from an given function
// via calls, this is necessary to check if a profile is valid despite the
// counts not fitting exactly.
bool ProfileVerifierPass::exitReachable(const Function *F) {
  if (!F) return false;

  if (FisVisited.count(F)) return false;

  Function *Exit = F->getParent()->getFunction("exit");
  if (Exit == F) {
    return true;
  }

  FisVisited.insert(F);
  bool exits = false;
  for (const_inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    if (const CallInst *CI = dyn_cast<CallInst>(&*I)) {
      exits |= exitReachable(CI->getCalledFunction());
      if (exits) break;
    }
  }
  return exits;
}

#define ASSERTMESSAGE(M) \
    errs() << (M) << "\n"; \
    if (!DisableAssertions) assert(0 && (M));

double ProfileVerifierPass::ReadOrAssert(ProfileInfo::Edge E) {
  double EdgeWeight = PI->getEdgeWeight(E);
  if (EdgeWeight == ProfileInfo::MissingValue) {
    errs() << "Edge " << E << " in Function " 
           << ProfileInfo::getFunction(E)->getNameStr() << ": ";
    ASSERTMESSAGE("ASSERT:Edge has missing value");
    return 0;
  } else {
    return EdgeWeight;
  }
}

void ProfileVerifierPass::CheckValue(bool Error, const char *Message,
                                     DetailedBlockInfo *DI) {
  if (Error) {
    DEBUG(debugEntry(DI));
    errs() << "Block " << DI->BB->getNameStr() << " in Function " 
           << DI->BB->getParent()->getNameStr() << ": ";
    ASSERTMESSAGE(Message);
  }
  return;
}

// This calculates the Information for a block and then recurses into the
// successors.
void ProfileVerifierPass::recurseBasicBlock(const BasicBlock *BB) {

  // Break the recursion by remembering all visited blocks.
  if (BBisVisited.find(BB) != BBisVisited.end()) return;

  // Use a data structure to store all the information, this can then be handed
  // to debug printers.
  DetailedBlockInfo DI;
  DI.BB = BB;
  DI.outCount = DI.inCount = DI.inWeight = DI.outWeight = 0;

  // Read predecessors.
  std::set<const BasicBlock*> ProcessedPreds;
  pred_const_iterator bpi = pred_begin(BB), bpe = pred_end(BB);
  // If there are none, check for (0,BB) edge.
  if (bpi == bpe) {
    DI.inWeight += ReadOrAssert(PI->getEdge(0,BB));
    DI.inCount++;
  }
  for (;bpi != bpe; ++bpi) {
    if (ProcessedPreds.insert(*bpi).second) {
      DI.inWeight += ReadOrAssert(PI->getEdge(*bpi,BB));
      DI.inCount++;
    }
  }

  // Read successors.
  std::set<const BasicBlock*> ProcessedSuccs;
  succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB);
  // If there is an (0,BB) edge, consider it too. (This is done not only when
  // there are no successors, but every time; not every function contains
  // return blocks with no successors (think loop latch as return block)).
  double w = PI->getEdgeWeight(PI->getEdge(BB,0));
  if (w != ProfileInfo::MissingValue) {
    DI.outWeight += w;
    DI.outCount++;
  }
  for (;bbi != bbe; ++bbi) {
    if (ProcessedSuccs.insert(*bbi).second) {
      DI.outWeight += ReadOrAssert(PI->getEdge(BB,*bbi));
      DI.outCount++;
    }
  }

  // Read block weight.
  DI.BBWeight = PI->getExecutionCount(BB);
  CheckValue(DI.BBWeight == ProfileInfo::MissingValue,
             "ASSERT:BasicBlock has missing value", &DI);

  // Check if this block is a setjmp target.
  bool isSetJmpTarget = false;
  if (DI.outWeight > DI.inWeight) {
    for (BasicBlock::const_iterator i = BB->begin(), ie = BB->end();
         i != ie; ++i) {
      if (const CallInst *CI = dyn_cast<CallInst>(&*i)) {
        Function *F = CI->getCalledFunction();
        if (F && (F->getNameStr() == "_setjmp")) {
          isSetJmpTarget = true; break;
        }
      }
    }
  }
  // Check if this block is eventually reaching exit.
  bool isExitReachable = false;
  if (DI.inWeight > DI.outWeight) {
    for (BasicBlock::const_iterator i = BB->begin(), ie = BB->end();
         i != ie; ++i) {
      if (const CallInst *CI = dyn_cast<CallInst>(&*i)) {
        FisVisited.clear();
        isExitReachable |= exitReachable(CI->getCalledFunction());
        if (isExitReachable) break;
      }
    }
  }

  if (DI.inCount > 0 && DI.outCount == 0) {
     // If this is a block with no successors.
    if (!isSetJmpTarget) {
      CheckValue(!Equals(DI.inWeight,DI.BBWeight), 
                 "ASSERT:inWeight and BBWeight do not match", &DI);
    }
  } else if (DI.inCount == 0 && DI.outCount > 0) {
    // If this is a block with no predecessors.
    if (!isExitReachable)
      CheckValue(!Equals(DI.BBWeight,DI.outWeight), 
                 "ASSERT:BBWeight and outWeight do not match", &DI);
  } else {
    // If this block has successors and predecessors.
    if (DI.inWeight > DI.outWeight && !isExitReachable)
      CheckValue(!Equals(DI.inWeight,DI.outWeight), 
                 "ASSERT:inWeight and outWeight do not match", &DI);
    if (DI.inWeight < DI.outWeight && !isSetJmpTarget)
      CheckValue(!Equals(DI.inWeight,DI.outWeight), 
                 "ASSERT:inWeight and outWeight do not match", &DI);
  }


  // Mark this block as visited, rescurse into successors.
  BBisVisited.insert(BB);
  for ( succ_const_iterator bbi = succ_begin(BB), bbe = succ_end(BB); 
        bbi != bbe; ++bbi ) {
    recurseBasicBlock(*bbi);
  }
}

bool ProfileVerifierPass::runOnFunction(Function &F) {
  PI = &getAnalysis<ProfileInfo>();

  // Prepare global variables.
  PrintedDebugTree = false;
  BBisVisited.clear();

  // Fetch entry block and recurse into it.
  const BasicBlock *entry = &F.getEntryBlock();
  recurseBasicBlock(entry);

  if (!DisableAssertions)
    assert((PI->getExecutionCount(&F)==PI->getExecutionCount(entry)) &&
           "Function count and entry block count do not match");
  return false;
}
