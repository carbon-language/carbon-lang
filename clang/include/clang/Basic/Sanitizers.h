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

}  // end namespace clang

#endif
