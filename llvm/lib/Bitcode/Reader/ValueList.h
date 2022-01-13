//===-- Bitcode/Reader/ValueList.h - Number values --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class gives values and types Unique ID's.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_BITCODE_READER_VALUELIST_H
#define LLVM_LIB_BITCODE_READER_VALUELIST_H

#include "llvm/IR/ValueHandle.h"
#include <cassert>
#include <utility>
#include <vector>

namespace llvm {

class Constant;
class LLVMContext;
class Type;
class Value;

class BitcodeReaderValueList {
  std::vector<WeakTrackingVH> ValuePtrs;

  /// As we resolve forward-referenced constants, we add information about them
  /// to this vector.  This allows us to resolve them in bulk instead of
  /// resolving each reference at a time.  See the code in
  /// ResolveConstantForwardRefs for more information about this.
  ///
  /// The key of this vector is the placeholder constant, the value is the slot
  /// number that holds the resolved value.
  using ResolveConstantsTy = std::vector<std::pair<Constant *, unsigned>>;
  ResolveConstantsTy ResolveConstants;
  LLVMContext &Context;

  /// Maximum number of valid references. Forward references exceeding the
  /// maximum must be invalid.
  unsigned RefsUpperBound;

public:
  BitcodeReaderValueList(LLVMContext &C, size_t RefsUpperBound)
      : Context(C),
        RefsUpperBound(std::min((size_t)std::numeric_limits<unsigned>::max(),
                                RefsUpperBound)) {}

  ~BitcodeReaderValueList() {
    assert(ResolveConstants.empty() && "Constants not resolved?");
  }

  // vector compatibility methods
  unsigned size() const { return ValuePtrs.size(); }
  void resize(unsigned N) {
    ValuePtrs.resize(N);
  }
  void push_back(Value *V) { ValuePtrs.emplace_back(V); }

  void clear() {
    assert(ResolveConstants.empty() && "Constants not resolved?");
    ValuePtrs.clear();
  }

  Value *operator[](unsigned i) const {
    assert(i < ValuePtrs.size());
    return ValuePtrs[i];
  }

  Value *back() const { return ValuePtrs.back(); }
  void pop_back() {
    ValuePtrs.pop_back();
  }
  bool empty() const { return ValuePtrs.empty(); }

  void shrinkTo(unsigned N) {
    assert(N <= size() && "Invalid shrinkTo request!");
    ValuePtrs.resize(N);
  }

  Constant *getConstantFwdRef(unsigned Idx, Type *Ty);
  Value *getValueFwdRef(unsigned Idx, Type *Ty);

  void assignValue(Value *V, unsigned Idx);

  /// Once all constants are read, this method bulk resolves any forward
  /// references.
  void resolveConstantForwardRefs();
};

} // end namespace llvm

#endif // LLVM_LIB_BITCODE_READER_VALUELIST_H
