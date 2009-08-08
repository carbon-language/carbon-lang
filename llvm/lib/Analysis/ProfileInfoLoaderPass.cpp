//===- ProfileInfoLoaderPass.cpp - LLVM Pass to load profile info ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a concrete implementation of profiling information that
// loads the information from a profile dump file.
//
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/InstrTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Analysis/ProfileInfoLoader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"
using namespace llvm;

static cl::opt<std::string>
ProfileInfoFilename("profile-info-file", cl::init("llvmprof.out"),
                    cl::value_desc("filename"),
                    cl::desc("Profile file loaded by -profile-loader"));

namespace {
  class VISIBILITY_HIDDEN LoaderPass : public ModulePass, public ProfileInfo {
    std::string Filename;
  public:
    static char ID; // Class identification, replacement for typeinfo
    explicit LoaderPass(const std::string &filename = "")
      : ModulePass(&ID), Filename(filename) {
      if (filename.empty()) Filename = ProfileInfoFilename;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    virtual const char *getPassName() const {
      return "Profiling information loader";
    }

    /// run - Load the profile information from the specified file.
    virtual bool runOnModule(Module &M);
  };
}  // End of anonymous namespace

char LoaderPass::ID = 0;
static RegisterPass<LoaderPass>
X("profile-loader", "Load profile information from llvmprof.out", false, true);

static RegisterAnalysisGroup<ProfileInfo> Y(X);

ModulePass *llvm::createProfileLoaderPass() { return new LoaderPass(); }

/// createProfileLoaderPass - This function returns a Pass that loads the
/// profiling information for the module from the specified filename, making it
/// available to the optimizers.
Pass *llvm::createProfileLoaderPass(const std::string &Filename) {
  return new LoaderPass(Filename);
}

bool LoaderPass::runOnModule(Module &M) {
  ProfileInfoLoader PIL("profile-loader", Filename, M);

  EdgeInformation.clear();
  std::vector<unsigned> ECs = PIL.getRawEdgeCounts();
  if (ECs.size() > 0) {
    unsigned ei = 0;
    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
      if (F->isDeclaration()) continue;
      if (ei < ECs.size())
        EdgeInformation[F][ProfileInfo::getEdge(0, &F->getEntryBlock())] +=
          ECs[ei++];
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
        // Okay, we have to add a counter of each outgoing edge.  If the
        // outgoing edge is not critical don't split it, just insert the counter
        // in the source or destination of the edge.
        TerminatorInst *TI = BB->getTerminator();
        for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s) {
          if (ei < ECs.size())
            EdgeInformation[F][ProfileInfo::getEdge(BB, TI->getSuccessor(s))] +=
              ECs[ei++];
        }
      }
    }
    if (ei != ECs.size()) {
      cerr << "WARNING: profile information is inconsistent with "
           << "the current program!\n";
    }
  }

  BlockInformation.clear();
  std::vector<unsigned> BCs = PIL.getRawBlockCounts();
  if (BCs.size() > 0) {
    unsigned bi = 0;
    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
      if (F->isDeclaration()) continue;
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
        if (bi < BCs.size())
          BlockInformation[F][BB] = BCs[bi++];
    }
    if (bi != BCs.size()) {
      cerr << "WARNING: profile information is inconsistent with "
           << "the current program!\n";
    }
  }

  FunctionInformation.clear();
  std::vector<unsigned> FCs = PIL.getRawFunctionCounts();
  if (FCs.size() > 0) {
    unsigned fi = 0;
    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
      if (F->isDeclaration()) continue;
      if (fi < FCs.size())
        FunctionInformation[F] = FCs[fi++];
    }
    if (fi != FCs.size()) {
      cerr << "WARNING: profile information is inconsistent with "
           << "the current program!\n";
    }
  }

  return false;
}
