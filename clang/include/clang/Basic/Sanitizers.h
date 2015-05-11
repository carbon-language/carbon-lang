//===--- Sanitizers.h - C Language Family Language Options ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::SanitizerKind enum.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_SANITIZERS_H
#define LLVM_CLANG_BASIC_SANITIZERS_H

#include <stdint.h>

namespace clang {

typedef uint64_t SanitizerMask;

namespace SanitizerKind {

// Assign ordinals to possible values of -fsanitize= flag, which we will use as
// bit positions.
enum SanitizerOrdinal : uint64_t {
#define SANITIZER(NAME, ID) SO_##ID,
#define SANITIZER_GROUP(NAME, ID, ALIAS) SO_##ID##Group,
#include "clang/Basic/Sanitizers.def"
  SO_Count
};

// Define the set of sanitizer kinds, as well as the set of sanitizers each
// sanitizer group expands into.
#define SANITIZER(NAME, ID) \
  const SanitizerMask ID = 1ULL << SO_##ID;
#define SANITIZER_GROUP(NAME, ID, ALIAS) \
  const SanitizerMask ID = ALIAS; \
  const SanitizerMask ID##Group = 1ULL << SO_##ID##Group;
#include "clang/Basic/Sanitizers.def"

}

struct SanitizerSet {
  SanitizerSet();

  /// \brief Check if a certain (single) sanitizer is enabled.
  bool has(SanitizerMask K) const;

  /// \brief Enable or disable a certain (single) sanitizer.
  void set(SanitizerMask K, bool Value);

  /// \brief Disable all sanitizers.
  void clear();

  /// \brief Returns true if at least one sanitizer is enabled.
  bool empty() const;

  /// \brief Bitmask of enabled sanitizers.
  SanitizerMask Mask;
};

}  // end namespace clang

#endif
