//===- DiagnosticNames.cpp - Defines a table of all builtin diagnostics ----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DiagnosticNames.h"
#include "clang/Basic/AllDiagnostics.h"

using namespace clang;

const diagtool::DiagnosticRecord diagtool::BuiltinDiagnostics[] = {
#define DIAG_NAME_INDEX(ENUM) { #ENUM, diag::ENUM, STR_SIZE(#ENUM, uint8_t) },
#include "clang/Basic/DiagnosticIndexName.inc"
#undef DIAG_NAME_INDEX
  { 0, 0, 0 }
};

const size_t diagtool::BuiltinDiagnosticsCount =
  sizeof(diagtool::BuiltinDiagnostics) /
  sizeof(diagtool::BuiltinDiagnostics[0]) - 1;

