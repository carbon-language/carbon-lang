//===- llvm/Analysis/ProfileInfo.h - Profile Info Interface -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the generic ProfileInfo interface, which is used as the
// common interface used by all clients of profiling information, and
// implemented either by making static guestimations, or by actually reading in
// profiling information gathered by running the program.
//
// Note that to be useful, all profile-based optimizations should preserve
// ProfileInfo, which requires that they notify it when changes to the CFG are
// made.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PROFILEINFO_H
#define LLVM_ANALYSIS_PROFILEINFO_H

#include <string>

namespace llvm {
  class BasicBlock;
  class Pass;

  /// createProfileLoaderPass - This function returns a Pass that loads the
  /// profiling information for the module from the specified filename, making
  /// it available to the optimizers.
  Pass *createProfileLoaderPass(const std::string &Filename);

  struct ProfileInfo {
    virtual ~ProfileInfo();  // We want to be subclassed
    
    //===------------------------------------------------------------------===//
    /// Profile Information Queries
    ///
    virtual unsigned getExecutionCount(BasicBlock *BB) = 0;
    
    //===------------------------------------------------------------------===//
    /// Analysis Update Methods
    ///

  };

} // End llvm namespace

#endif
