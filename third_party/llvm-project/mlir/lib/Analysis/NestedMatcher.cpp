//===- NestedMatcher.cpp - NestedMatcher Impl  ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

llvm::BumpPtrAllocator *&NestedMatch::allocator() {
  thread_local llvm::BumpPtrAllocator *allocator = nullptr;
  return allocator;
}

NestedMatch NestedMatch::build(Operation *operation,
                               ArrayRef<NestedMatch> nestedMatches) {
  auto *result = allocator()->Allocate<NestedMatch>();
  auto *children = allocator()->Allocate<NestedMatch>(nestedMatches.size());
  std::uninitialized_copy(nestedMatches.begin(), nestedMatches.end(), children);
  new (result) NestedMatch();
  result->matchedOperation = operation;
  result->matchedChildren =
      ArrayRef<NestedMatch>(children, nestedMatches.size());
  return *result;
}

llvm::BumpPtrAllocator *&NestedPattern::allocator() {
  thread_local llvm::BumpPtrAllocator *allocator = nullptr;
  return allocator;
}

void NestedPattern::copyNestedToThis(ArrayRef<NestedPattern> nested) {
  if (nested.empty())
    return;

  auto *newNested = allocator()->Allocate<NestedPattern>(nested.size());
  std::uninitialized_copy(nested.begin(), nested.end(), newNested);
  nestedPatterns = ArrayRef<NestedPattern>(newNested, nested.size());
}

void NestedPattern::freeNested() {
  for (const auto &p : nestedPatterns)
    p.~NestedPattern();
}

NestedPattern::NestedPattern(ArrayRef<NestedPattern> nested,
                             FilterFunctionType filter)
    : nestedPatterns(), filter(std::move(filter)), skip(nullptr) {
  copyNestedToThis(nested);
}

NestedPattern::NestedPattern(const NestedPattern &other)
    : nestedPatterns(), filter(other.filter), skip(other.skip) {
  copyNestedToThis(other.nestedPatterns);
}

NestedPattern &NestedPattern::operator=(const NestedPattern &other) {
  freeNested();
  filter = other.filter;
  skip = other.skip;
  copyNestedToThis(other.nestedPatterns);
  return *this;
}

unsigned NestedPattern::getDepth() const {
  if (nestedPatterns.empty()) {
    return 1;
  }
  unsigned depth = 0;
  for (auto &c : nestedPatterns) {
    depth = std::max(depth, c.getDepth());
  }
  return depth + 1;
}

/// Matches a single operation in the following way:
///   1. checks the kind of operation against the matcher, if different then
///      there is no match;
///   2. calls the customizable filter function to refine the single operation
///      match with extra semantic constraints;
///   3. if all is good, recursively matches the nested patterns;
///   4. if all nested match then the single operation matches too and is
///      appended to the list of matches;
///   5. TODO: Optionally applies actions (lambda), in which case we will want
///      to traverse in post-order DFS to avoid invalidating iterators.
void NestedPattern::matchOne(Operation *op,
                             SmallVectorImpl<NestedMatch> *matches) {
  if (skip == op) {
    return;
  }
  // Local custom filter function
  if (!filter(*op)) {
    return;
  }

  if (nestedPatterns.empty()) {
    SmallVector<NestedMatch, 8> nestedMatches;
    matches->push_back(NestedMatch::build(op, nestedMatches));
    return;
  }
  // Take a copy of each nested pattern so we can match it.
  for (auto nestedPattern : nestedPatterns) {
    SmallVector<NestedMatch, 8> nestedMatches;
    // Skip elem in the walk immediately following. Without this we would
    // essentially need to reimplement walk here.
    nestedPattern.skip = op;
    nestedPattern.match(op, &nestedMatches);
    // If we could not match even one of the specified nestedPattern, early exit
    // as this whole branch is not a match.
    if (nestedMatches.empty()) {
      return;
    }
    matches->push_back(NestedMatch::build(op, nestedMatches));
  }
}

static bool isAffineForOp(Operation &op) { return isa<AffineForOp>(op); }

static bool isAffineIfOp(Operation &op) { return isa<AffineIfOp>(op); }

namespace mlir {
namespace matcher {

NestedPattern Op(FilterFunctionType filter) {
  return NestedPattern({}, std::move(filter));
}

NestedPattern If(const NestedPattern &child) {
  return NestedPattern(child, isAffineIfOp);
}
NestedPattern If(const FilterFunctionType &filter, const NestedPattern &child) {
  return NestedPattern(child, [filter](Operation &op) {
    return isAffineIfOp(op) && filter(op);
  });
}
NestedPattern If(ArrayRef<NestedPattern> nested) {
  return NestedPattern(nested, isAffineIfOp);
}
NestedPattern If(const FilterFunctionType &filter,
                 ArrayRef<NestedPattern> nested) {
  return NestedPattern(nested, [filter](Operation &op) {
    return isAffineIfOp(op) && filter(op);
  });
}

NestedPattern For(const NestedPattern &child) {
  return NestedPattern(child, isAffineForOp);
}
NestedPattern For(const FilterFunctionType &filter,
                  const NestedPattern &child) {
  return NestedPattern(
      child, [=](Operation &op) { return isAffineForOp(op) && filter(op); });
}
NestedPattern For(ArrayRef<NestedPattern> nested) {
  return NestedPattern(nested, isAffineForOp);
}
NestedPattern For(const FilterFunctionType &filter,
                  ArrayRef<NestedPattern> nested) {
  return NestedPattern(
      nested, [=](Operation &op) { return isAffineForOp(op) && filter(op); });
}

bool isLoadOrStore(Operation &op) {
  return isa<AffineLoadOp, AffineStoreOp>(op);
}

} // namespace matcher
} // namespace mlir
