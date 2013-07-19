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

/// \brief OpenMP clauses.
enum OpenMPClauseKind {
  OMPC_unknown = 0,
#define OPENMP_CLAUSE(Name, Class) \
  OMPC_##Name,
#include "clang/Basic/OpenMPKinds.def"
  OMPC_threadprivate,
  NUM_OPENMP_CLAUSES
};

/// \brief OpenMP attributes for 'default' clause.
enum OpenMPDefaultClauseKind {
  OMPC_DEFAULT_unknown = 0,
#define OPENMP_DEFAULT_KIND(Name) \
  OMPC_DEFAULT_##Name,
#include "clang/Basic/OpenMPKinds.def"
  NUM_OPENMP_DEFAULT_KINDS
};

OpenMPDirectiveKind getOpenMPDirectiveKind(llvm::StringRef Str);
const char *getOpenMPDirectiveName(OpenMPDirectiveKind Kind);

OpenMPClauseKind getOpenMPClauseKind(llvm::StringRef Str);
const char *getOpenMPClauseName(OpenMPClauseKind Kind);

unsigned getOpenMPSimpleClauseType(OpenMPClauseKind Kind, llvm::StringRef Str);
const char *getOpenMPSimpleClauseTypeName(OpenMPClauseKind Kind, unsigned Type);

bool isAllowedClauseForDirective(OpenMPDirectiveKind DKind,
                                 OpenMPClauseKind CKind);

}

#endif

