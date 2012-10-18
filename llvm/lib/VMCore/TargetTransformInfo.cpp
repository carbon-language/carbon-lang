//===- llvm/VMCore/TargetTransformInfo.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetTransformInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

/// Default ctor.
///
/// @note This has to exist, because this is a pass, but it should never be
/// used.
TargetTransformInfo::TargetTransformInfo() : ImmutablePass(ID) {
  report_fatal_error("Bad TargetTransformInfo ctor used.  "
                     "Tool did not specify a TargetTransformInfo to use?");
}

INITIALIZE_PASS(TargetTransformInfo, "TargetTransformInfo",
                "Target Transform Info", false, true)
char TargetTransformInfo::ID = 0;

