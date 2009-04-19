//===--- Warnings.cpp - C-Language Front-end ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Command line warning options handler.
//
//===----------------------------------------------------------------------===//
//
// This file is responsible for handling all warning options. This includes
// a number of -Wfoo options and their variants, which are driven by TableGen-
// generated data, and the special cases -pedantic, -pedantic-errors, -w and
// -Werror.
//
// Each warning option controls any number of actual warnings.
// Given a warning option 'foo', the following are valid:
//    -Wfoo, -Wno-foo, -Werror=foo
//
#include "clang-cc.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Lex/LexDiagnostic.h"
#include "llvm/Support/CommandLine.h"
#include <cstdio>
#include <utility>
#include <algorithm>
using namespace clang;

// This gets all -W options, including -Werror, -W[no-]system-headers, etc.  The
// driver has stripped off -Wa,foo etc.  The driver has also translated -W to
// -Wextra, so we don't need to worry about it.
static llvm::cl::list<std::string>
OptWarnings("W", llvm::cl::Prefix, llvm::cl::ValueOptional);

static llvm::cl::opt<bool> OptPedantic("pedantic");
static llvm::cl::opt<bool> OptPedanticErrors("pedantic-errors");
static llvm::cl::opt<bool> OptNoWarnings("w");

bool clang::ProcessWarningOptions(Diagnostic &Diags) {
  Diags.setSuppressSystemWarnings(true);  // Default to -Wno-system-headers
  Diags.setIgnoreAllWarnings(OptNoWarnings);

  // If -pedantic or -pedantic-errors was specified, then we want to map all
  // extension diagnostics onto WARNING or ERROR unless the user has futz'd
  // around with them explicitly.
  if (OptPedanticErrors)
    Diags.setExtensionHandlingBehavior(Diagnostic::Ext_Error);
  else if (OptPedantic)
    Diags.setExtensionHandlingBehavior(Diagnostic::Ext_Warn);
  else
    Diags.setExtensionHandlingBehavior(Diagnostic::Ext_Ignore);
  
  // FIXME: -Wfatal-errors / -Wfatal-errors=foo

  for (unsigned i = 0, e = OptWarnings.size(); i != e; ++i) {
    const std::string &Opt = OptWarnings[i];
    const char *OptStart = &Opt[0];
    const char *OptEnd = OptStart+Opt.size();
    assert(*OptEnd == 0 && "Expect null termination for lower-bound search");
    
    // Check to see if this warning starts with "no-", if so, this is a negative
    // form of the option.
    bool isPositive = true;
    if (OptEnd-OptStart > 3 && memcmp(OptStart, "no-", 3) == 0) {
      isPositive = false;
      OptStart += 3;
    }

    // Figure out how this option affects the warning.  If -Wfoo, map the
    // diagnostic to a warning, if -Wno-foo, map it to ignore.
    diag::Mapping Mapping = isPositive ? diag::MAP_WARNING : diag::MAP_IGNORE;

    // -Wsystem-headers is a special case, not driven by the option table.  It
    // cannot be controlled with -Werror.
    if (OptEnd-OptStart == 14 && memcmp(OptStart, "system-headers", 14) == 0) {
      Diags.setSuppressSystemWarnings(!isPositive);
      continue;
    }
    
    // -Werror/-Wno-error is a special case, not controlled by the option table.
    // It also has the "specifier" form of -Werror=foo and -Werror-foo.
    if (OptEnd-OptStart >= 5 && memcmp(OptStart, "error", 5) == 0) {
      const char *Specifier = 0;
      if (OptEnd-OptStart != 5) {  // Specifier must be present.
        if ((OptStart[5] != '=' && OptStart[5] != '-') ||
            OptEnd-OptStart == 6) {
          fprintf(stderr, "warning: unknown -Werror warning specifier: -W%s\n",
                  Opt.c_str());
          continue;
        }
        Specifier = OptStart+6;
      }
      
      if (Specifier == 0) {
        Diags.setWarningsAsErrors(true);
        continue;
      }
      
      // -Werror=foo maps foo to Error, -Wno-error=foo maps it to Warning.
      Mapping = isPositive ? diag::MAP_ERROR : diag::MAP_WARNING_NO_WERROR;
      OptStart = Specifier;
    }
    
    if (Diags.setDiagnosticGroupMapping(OptStart, Mapping))
      fprintf(stderr, "warning: unknown warning option: -W%s\n", Opt.c_str());
  }
  
  return false;
}
