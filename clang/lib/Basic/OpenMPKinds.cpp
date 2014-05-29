//===--- OpenMPKinds.cpp - Token Kinds Support ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements the OpenMP enum and support functions.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace clang;

OpenMPDirectiveKind clang::getOpenMPDirectiveKind(StringRef Str) {
  return llvm::StringSwitch<OpenMPDirectiveKind>(Str)
#define OPENMP_DIRECTIVE(Name) \
           .Case(#Name, OMPD_##Name)
#include "clang/Basic/OpenMPKinds.def"
           .Default(OMPD_unknown);
}

const char *clang::getOpenMPDirectiveName(OpenMPDirectiveKind Kind) {
  assert(Kind <= OMPD_unknown);
  switch (Kind) {
  case OMPD_unknown:
    return "unknown";
#define OPENMP_DIRECTIVE(Name) \
  case OMPD_##Name : return #Name;
#include "clang/Basic/OpenMPKinds.def"
    break;
  }
  llvm_unreachable("Invalid OpenMP directive kind");
}

OpenMPClauseKind clang::getOpenMPClauseKind(StringRef Str) {
  return llvm::StringSwitch<OpenMPClauseKind>(Str)
#define OPENMP_CLAUSE(Name, Class) \
           .Case(#Name, OMPC_##Name)
#include "clang/Basic/OpenMPKinds.def"
           .Default(OMPC_unknown);
}

const char *clang::getOpenMPClauseName(OpenMPClauseKind Kind) {
  assert(Kind <= OMPC_unknown);
  switch (Kind) {
  case OMPC_unknown:
    return "unknown";
#define OPENMP_CLAUSE(Name, Class) \
  case OMPC_##Name : return #Name;
#include "clang/Basic/OpenMPKinds.def"
  case OMPC_threadprivate:
    return "threadprivate or thread local";
  }
  llvm_unreachable("Invalid OpenMP clause kind");
}

unsigned clang::getOpenMPSimpleClauseType(OpenMPClauseKind Kind,
                                          StringRef Str) {
  switch (Kind) {
  case OMPC_default:
    return llvm::StringSwitch<OpenMPDefaultClauseKind>(Str)
#define OPENMP_DEFAULT_KIND(Name) \
             .Case(#Name, OMPC_DEFAULT_##Name)
#include "clang/Basic/OpenMPKinds.def"
             .Default(OMPC_DEFAULT_unknown);
  case OMPC_proc_bind:
    return llvm::StringSwitch<OpenMPProcBindClauseKind>(Str)
#define OPENMP_PROC_BIND_KIND(Name) \
             .Case(#Name, OMPC_PROC_BIND_##Name)
#include "clang/Basic/OpenMPKinds.def"
             .Default(OMPC_PROC_BIND_unknown);
  case OMPC_unknown:
  case OMPC_threadprivate:
  case OMPC_if:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_collapse:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_shared:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
    break;
  }
  llvm_unreachable("Invalid OpenMP simple clause kind");
}

const char *clang::getOpenMPSimpleClauseTypeName(OpenMPClauseKind Kind,
                                                 unsigned Type) {
  switch (Kind) {
  case OMPC_default:
    switch (Type) {
    case OMPC_DEFAULT_unknown:
      return "unknown";
#define OPENMP_DEFAULT_KIND(Name) \
    case OMPC_DEFAULT_##Name : return #Name;
#include "clang/Basic/OpenMPKinds.def"
    }
    llvm_unreachable("Invalid OpenMP 'default' clause type");
  case OMPC_proc_bind:
    switch (Type) {
    case OMPC_PROC_BIND_unknown:
      return "unknown";
#define OPENMP_PROC_BIND_KIND(Name) \
    case OMPC_PROC_BIND_##Name : return #Name;
#include "clang/Basic/OpenMPKinds.def"
    }
    llvm_unreachable("Invalid OpenMP 'proc_bind' clause type");
  case OMPC_unknown:
  case OMPC_threadprivate:
  case OMPC_if:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_collapse:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_shared:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
    break;
  }
  llvm_unreachable("Invalid OpenMP simple clause kind");
}

bool clang::isAllowedClauseForDirective(OpenMPDirectiveKind DKind,
                                        OpenMPClauseKind CKind) {
  assert(DKind <= OMPD_unknown);
  assert(CKind <= OMPC_unknown);
  switch (DKind) {
  case OMPD_parallel:
    switch (CKind) {
#define OPENMP_PARALLEL_CLAUSE(Name) \
    case OMPC_##Name: return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_simd:
    switch (CKind) {
#define OPENMP_SIMD_CLAUSE(Name) \
    case OMPC_##Name: return true;
#include "clang/Basic/OpenMPKinds.def"
    default:
      break;
    }
    break;
  case OMPD_unknown:
  case OMPD_threadprivate:
  case OMPD_task:
    break;
  }
  return false;
}
