//===- ProfileInfo.cpp - Profile Info Interface ---------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the abstract ProfileInfo interface, and the default
// "no profile" implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Pass.h"
using namespace llvm;

// Register the AliasAnalysis interface, providing a nice name to refer to.
namespace {
  RegisterAnalysisGroup<ProfileInfo> Z("Profile Information");
}

ProfileInfo::~ProfileInfo() {}


//===----------------------------------------------------------------------===//
//  NoProfile ProfileInfo implementation
//

namespace {
  struct NoProfileInfo : public ImmutablePass, public ProfileInfo {
    unsigned getExecutionCount(BasicBlock *BB) { return 0; }
  };
 
  // Register this pass...
  RegisterOpt<NoProfileInfo>
  X("no-profile", "No Profile Information");

  // Declare that we implement the AliasAnalysis interface
  RegisterAnalysisGroup<ProfileInfo, NoProfileInfo, true> Y;
}  // End of anonymous namespace
