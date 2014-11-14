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

namespace clang {

enum class SanitizerKind {
#define SANITIZER(NAME, ID) ID,
#include "clang/Basic/Sanitizers.def"
  Unknown
};

class SanitizerSet {
  /// \brief Bitmask of enabled sanitizers.
  unsigned Kinds;
public:
  SanitizerSet();

  /// \brief Check if a certain sanitizer is enabled.
  bool has(SanitizerKind K) const;

  /// \brief Enable or disable a certain sanitizer.
  void set(SanitizerKind K, bool Value);

  /// \brief Disable all sanitizers.
  void clear();

  /// \brief Returns true if at least one sanitizer is enabled.
  bool empty() const;
};

}  // end namespace clang

#endif
