//===- ProfileInfoLoaderPass.cpp - LLVM Pass to load profile info ---------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a concrete implementation of profiling information that
// loads the information from a profile dump file.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Analysis/ProfileInfoLoader.h"
#include "Support/CommandLine.h"
using namespace llvm;

namespace {
  cl::opt<std::string>
  ProfileInfoFilename("profile-info-file", cl::init("llvmprof.out"),
                      cl::desc(""));

  class LoaderPass : public Pass, public ProfileInfo {
    std::string Filename;
    std::map<BasicBlock*, unsigned> ExecutionCounts;
  public:
    LoaderPass(const std::string &filename = "")
      : Filename(filename) {
      if (filename.empty()) Filename = ProfileInfoFilename;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    virtual const char *getPassName() const {
      return "Profiling information loader";
    }

    /// run - Load the profile information from the specified file.
    virtual bool run(Module &M);

    virtual unsigned getExecutionCount(BasicBlock *BB) {
      std::map<BasicBlock*, unsigned>::iterator I = ExecutionCounts.find(BB);
      return I != ExecutionCounts.end() ? I->second : 0;
    }
  };
 
  RegisterOpt<LoaderPass>
  X("profile-loader", "Load profile information from llvmprof.out");

  RegisterAnalysisGroup<ProfileInfo, LoaderPass> Y;
}  // End of anonymous namespace


/// createProfileLoaderPass - This function returns a Pass that loads the
/// profiling information for the module from the specified filename, making it
/// available to the optimizers.
Pass *llvm::createProfileLoaderPass(const std::string &Filename) {
  return new LoaderPass(Filename);
}

bool LoaderPass::run(Module &M) {
  ProfileInfoLoader PIL("opt", Filename, M);
  if (PIL.hasAccurateBlockCounts()) {
    std::vector<std::pair<BasicBlock*, unsigned> > Counts;
    PIL.getBlockCounts(Counts);
    ExecutionCounts.insert(Counts.begin(), Counts.end());
  }
  return false;
}
