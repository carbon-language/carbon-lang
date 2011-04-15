//===--- DiagnosticOptions.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_DIAGNOSTICOPTIONS_H
#define LLVM_CLANG_FRONTEND_DIAGNOSTICOPTIONS_H

#include "clang/Basic/Diagnostic.h"

#include <string>
#include <vector>

namespace clang {

/// DiagnosticOptions - Options for controlling the compiler diagnostics
/// engine.
class DiagnosticOptions {
public:
  unsigned IgnoreWarnings : 1;   /// -w
  unsigned NoRewriteMacros : 1;  /// -Wno-rewrite-macros
  unsigned Pedantic : 1;         /// -pedantic
  unsigned PedanticErrors : 1;   /// -pedantic-errors
  unsigned ShowColumn : 1;       /// Show column number on diagnostics.
  unsigned ShowLocation : 1;     /// Show source location information.
  unsigned ShowCarets : 1;       /// Show carets in diagnostics.
  unsigned ShowFixits : 1;       /// Show fixit information.
  unsigned ShowSourceRanges : 1; /// Show source ranges in numeric form.
  unsigned ShowParseableFixits : 1; /// Show machine parseable fix-its.
  unsigned ShowNames : 1;        /// Show the diagnostic name
  unsigned ShowOptionNames : 1;  /// Show the option name for mappable
                                 /// diagnostics.
  unsigned ShowNoteIncludeStack : 1; /// Show include stacks for notes.
  unsigned ShowCategories : 2;   /// Show categories: 0 -> none, 1 -> Number,
                                 /// 2 -> Full Name.
  unsigned ShowColors : 1;       /// Show diagnostics with ANSI color sequences.
  unsigned ShowOverloads : 1;    /// Overload candidates to show.  Values from
                                 /// Diagnostic::OverloadsShown
  unsigned VerifyDiagnostics: 1; /// Check that diagnostics match the expected
                                 /// diagnostics, indicated by markers in the
                                 /// input source file.

  unsigned ErrorLimit;           /// Limit # errors emitted.
  unsigned MacroBacktraceLimit;  /// Limit depth of macro instantiation 
                                 /// backtrace.
  unsigned TemplateBacktraceLimit; /// Limit depth of instantiation backtrace.

  /// The distance between tab stops.
  unsigned TabStop;
  enum { DefaultTabStop = 8, MaxTabStop = 100, 
         DefaultMacroBacktraceLimit = 6,
         DefaultTemplateBacktraceLimit = 10 };

  /// Column limit for formatting message diagnostics, or 0 if unused.
  unsigned MessageLength;

  /// If non-empty, a file to log extended build information to, for development
  /// testing and analysis.
  std::string DumpBuildInformation;

  /// The file to log diagnostic output to.
  std::string DiagnosticLogFile;

  /// The list of -W... options used to alter the diagnostic mappings, with the
  /// prefixes removed.
  std::vector<std::string> Warnings;

public:
  DiagnosticOptions() {
    IgnoreWarnings = 0;
    TabStop = DefaultTabStop;
    MessageLength = 0;
    NoRewriteMacros = 0;
    Pedantic = 0;
    PedanticErrors = 0;
    ShowCarets = 1;
    ShowColors = 0;
    ShowOverloads = Diagnostic::Ovl_All;
    ShowColumn = 1;
    ShowFixits = 1;
    ShowLocation = 1;
    ShowNames = 0;
    ShowOptionNames = 0;
    ShowCategories = 0;
    ShowSourceRanges = 0;
    ShowParseableFixits = 0;
    VerifyDiagnostics = 0;
    ErrorLimit = 0;
    TemplateBacktraceLimit = DefaultTemplateBacktraceLimit;
    MacroBacktraceLimit = DefaultMacroBacktraceLimit;
  }
};

}  // end namespace clang

#endif
