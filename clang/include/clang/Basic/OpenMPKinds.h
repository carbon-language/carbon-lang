//===--- OpenMPKinds.h - OpenMP enums ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines some OpenMP-specific enums and functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OPENMPKINDS_H
#define LLVM_CLANG_BASIC_OPENMPKINDS_H

#include "llvm/ADT/StringRef.h"

namespace clang {

/// \brief OpenMP directives.
enum OpenMPDirectiveKind {
  OMPD_unknown = 0,
#define OPENMP_DIRECTIVE(Name) \
  OMPD_##Name,
#include "clang/Basic/OpenMPKinds.def"
  NUM_OPENMP_DIRECTIVES
};

OpenMPDirectiveKind getOpenMPDirectiveKind(llvm::StringRef Str);
const char *getOpenMPDirectiveName(OpenMPDirectiveKind Kind);

}

#endif

