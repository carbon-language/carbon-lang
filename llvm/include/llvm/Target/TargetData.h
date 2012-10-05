//===-- llvm/Target/TargetData.h - Data size & alignment info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the wrapper for DataLayout to provide compatibility
// with the old TargetData class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETDATA_H
#define LLVM_TARGET_TARGETDATA_H

#include "llvm/DataLayout.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

/// TargetData - This class is just a wrapper to help with the transition to the
/// new DataLayout class.
class TargetData : public DataLayout {
public:
  /// Default ctor.
  ///
  /// @note This has to exist, because this is a pass, but it should never be
  /// used.
  TargetData() : DataLayout() {}

  /// Constructs a TargetData from a specification string.
  /// See DataLayout::init().
  explicit TargetData(StringRef TargetDescription)
    : DataLayout(TargetDescription) {}

  /// Initialize target data from properties stored in the module.
  explicit TargetData(const Module *M) : DataLayout(M) {}

  TargetData(const TargetData &TD) : DataLayout(TD) {}

  template <typename UIntTy>
  static UIntTy RoundUpAlignment(UIntTy Val, unsigned Alignment) {
    return DataLayout::RoundUpAlignment(Val, Alignment);
  }
};

} // End llvm namespace

#endif
