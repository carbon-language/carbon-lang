//===- BasicTargetTransformInfo.cpp - Basic target-independent TTI impl ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the implementation of a basic TargetTransformInfo pass
/// predicated on the target abstractions present in the target independent
/// code generator. It uses these (primarily TargetLowering) to model as much
/// of the TTI query interface as possible. It is included by most targets so
/// that they can specialize only a small subset of the query space.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/Support/CommandLine.h"
#include <utility>
using namespace llvm;

cl::opt<unsigned>
    llvm::PartialUnrollingThreshold("partial-unrolling-threshold", cl::init(0),
                                    cl::desc("Threshold for partial unrolling"),
                                    cl::Hidden);

#define DEBUG_TYPE "basictti"

namespace {
class BasicTTIImpl : public BasicTTIImplBase<BasicTTIImpl> {
  typedef BasicTTIImplBase<BasicTTIImpl> BaseT;

public:
  explicit BasicTTIImpl(const TargetMachine *TM = nullptr) : BaseT(TM) {}

  // Provide value semantics. MSVC requires that we spell all of these out.
  BasicTTIImpl(const BasicTTIImpl &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)) {}
  BasicTTIImpl(BasicTTIImpl &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))) {}
  BasicTTIImpl &operator=(const BasicTTIImpl &RHS) {
    BaseT::operator=(static_cast<const BaseT &>(RHS));
    return *this;
  }
  BasicTTIImpl &operator=(BasicTTIImpl &&RHS) {
    BaseT::operator=(std::move(static_cast<BaseT &>(RHS)));
    return *this;
  }
};
}

ImmutablePass *
llvm::createBasicTargetTransformInfoPass(const TargetMachine *TM) {
  return new TargetTransformInfoWrapperPass(BasicTTIImpl(TM));
}


//===----------------------------------------------------------------------===//
//
// Calls used by the vectorizers.
//
//===----------------------------------------------------------------------===//

