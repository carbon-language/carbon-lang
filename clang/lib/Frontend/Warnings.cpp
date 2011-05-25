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
// generated data, and the special cases -pedantic, -pedantic-errors, -w,
// -Werror and -Wfatal-errors.
//
// Each warning option controls any number of actual warnings.
// Given a warning option 'foo', the following are valid:
//    -Wfoo, -Wno-foo, -Werror=foo, -Wfatal-errors=foo
//
#include "clang/Frontend/Utils.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Frontend/DiagnosticOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include <cstring>
#include <utility>
#include <algorithm>
using namespace clang;

void clang::ProcessWarningOptions(Diagnostic &Diags,
                                  const DiagnosticOptions &Opts) {
  Diags.setSuppressSystemWarnings(true);  // Default to -Wno-system-headers
  Diags.setIgnoreAllWarnings(Opts.IgnoreWarnings);
  Diags.setShowOverloads(
    static_cast<Diagnostic::OverloadsShown>(Opts.ShowOverloads));
  
  // Handle -ferror-limit
  if (Opts.ErrorLimit)
    Diags.setErrorLimit(Opts.ErrorLimit);
  if (Opts.TemplateBacktraceLimit)
    Diags.setTemplateBacktraceLimit(Opts.TemplateBacktraceLimit);

  // If -pedantic or -pedantic-errors was specified, then we want to map all
  // extension diagnostics onto WARNING or ERROR unless the user has futz'd
  // around with them explicitly.
  if (Opts.PedanticErrors)
    Diags.setExtensionHandlingBehavior(Diagnostic::Ext_Error);
  else if (Opts.Pedantic)
    Diags.setExtensionHandlingBehavior(Diagnostic::Ext_Warn);
  else
    Diags.setExtensionHandlingBehavior(Diagnostic::Ext_Ignore);

  for (unsigned i = 0, e = Opts.Warnings.size(); i != e; ++i) {
    llvm::StringRef Opt = Opts.Warnings[i];

    // Check to see if this warning starts with "no-", if so, this is a negative
    // form of the option.
    bool isPositive = true;
    if (Opt.startswith("no-")) {
      isPositive = false;
      Opt = Opt.substr(3);
    }

    // Figure out how this option affects the warning.  If -Wfoo, map the
    // diagnostic to a warning, if -Wno-foo, map it to ignore.
    diag::Mapping Mapping = isPositive ? diag::MAP_WARNING : diag::MAP_IGNORE;

    // -Wsystem-headers is a special case, not driven by the option table.  It
    // cannot be controlled with -Werror.
    if (Opt == "system-headers") {
      Diags.setSuppressSystemWarnings(!isPositive);
      continue;
    }

    // -Werror/-Wno-error is a special case, not controlled by the option table.
    // It also has the "specifier" form of -Werror=foo and -Werror-foo.
    if (Opt.startswith("error")) {
      llvm::StringRef Specifier;
      if (Opt.size() > 5) {  // Specifier must be present.
        if ((Opt[5] != '=' && Opt[5] != '-') || Opt.size() == 6) {
          Diags.Report(diag::warn_unknown_warning_specifier)
            << "-Werror" << ("-W" + Opt.str());
          continue;
        }
        Specifier = Opt.substr(6);
      }

      if (Specifier.empty()) {
        Diags.setWarningsAsErrors(isPositive);
        continue;
      }

      // -Werror=foo maps foo to Error, -Wno-error=foo maps it to Warning.
      Mapping = isPositive ? diag::MAP_ERROR : diag::MAP_WARNING_NO_WERROR;
      Opt = Specifier;
    }

    // -Wfatal-errors is yet another special case.
    if (Opt.startswith("fatal-errors")) {
      llvm::StringRef Specifier;
      if (Opt.size() != 12) {
        if ((Opt[12] != '=' && Opt[12] != '-') || Opt.size() == 13) {
          Diags.Report(diag::warn_unknown_warning_specifier)
            << "-Wfatal-errors" << ("-W" + Opt.str());
          continue;
        }
        Specifier = Opt.substr(13);
      }

      if (Specifier.empty()) {
        Diags.setErrorsAsFatal(isPositive);
        continue;
      }

      // -Wfatal-errors=foo maps foo to Fatal, -Wno-fatal-errors=foo
      // maps it to Error.
      Mapping = isPositive ? diag::MAP_FATAL : diag::MAP_ERROR_NO_WFATAL;
      Opt = Specifier;
    }

    if (Diags.setDiagnosticGroupMapping(Opt, Mapping))
      Diags.Report(diag::warn_unknown_warning_option) << ("-W" + Opt.str());
  }
}
