//===-- Core/Refactoring.h - Stand-in for Tooling/Refactoring.h -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file is meant to be used instead of clang/Tooling/Refactoring.h
/// until such time as clang::tooling::Replacements is re-implemented as a
/// vector instead of a set.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_REPLACEMENTS_VEC_H
#define CLANG_MODERNIZE_REPLACEMENTS_VEC_H

#include "clang/Tooling/Refactoring.h"

// FIXME: Remove this file when clang::tooling::Replacements becomes a vector
// instead of a set.

namespace clang {
namespace tooling {
typedef std::vector<clang::tooling::Replacement> ReplacementsVec;
}
}

#endif // CLANG_MODERNIZE_REPLACEMENTS_VEC_H
