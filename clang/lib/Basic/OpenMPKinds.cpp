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
  assert(Kind < NUM_OPENMP_DIRECTIVES);
  switch (Kind) {
  case OMPD_unknown:
    return "unknown";
#define OPENMP_DIRECTIVE(Name) \
  case OMPD_##Name : return #Name;
#include "clang/Basic/OpenMPKinds.def"
  default:
    break;
  }
  llvm_unreachable("Invalid OpenMP directive kind");
}

