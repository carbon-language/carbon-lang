//===-- XCoreTargetTransformInfo.h - XCore specific TTI ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// XCore target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XCORE_XCORETARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_XCORE_XCORETARGETTRANSFORMINFO_H

#include "XCore.h"
#include "XCoreTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {

class XCoreTTIImpl : public BasicTTIImplBase<XCoreTTIImpl> {
  typedef BasicTTIImplBase<XCoreTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;

public:
  explicit XCoreTTIImpl(const XCoreTargetMachine *TM = nullptr) : BaseT(TM) {}

  // Provide value semantics. MSVC requires that we spell all of these out.
  XCoreTTIImpl(const XCoreTTIImpl &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)) {}
  XCoreTTIImpl(XCoreTTIImpl &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))) {}
  XCoreTTIImpl &operator=(const XCoreTTIImpl &RHS) {
    BaseT::operator=(static_cast<const BaseT &>(RHS));
    return *this;
  }
  XCoreTTIImpl &operator=(XCoreTTIImpl &&RHS) {
    BaseT::operator=(std::move(static_cast<BaseT &>(RHS)));
    return *this;
  }

  unsigned getNumberOfRegisters(bool Vector) {
    if (Vector) {
      return 0;
    }
    return 12;
  }
};

} // end namespace llvm

#endif
