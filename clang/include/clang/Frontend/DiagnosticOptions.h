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

#include <string>
#include <vector>

namespace clang {

/// DiagnosticOptions - Options for controlling the compiler diagnostics
/// engine.
class DiagnosticOptions {
public:
  unsigned ShowColumn : 1;       /// Show column number on diagnostics.
  unsigned ShowLocation : 1;     /// Show source location information.
  unsigned ShowCarets : 1;       /// Show carets in diagnostics.
  unsigned ShowFixits : 1;       /// Show fixit information.
  unsigned ShowSourceRanges : 1; /// Show source ranges in numeric form.
  unsigned ShowOptionNames : 1;  /// Show the diagnostic name for mappable
                                 /// diagnostics.
  unsigned ShowColors : 1;       /// Show diagnostics with ANSI color sequences.

  /// Column limit for formatting message diagnostics, or 0 if unused.
  unsigned MessageLength;

  /// If non-empty, a file to log extended build information to, for development
  /// testing and analysis.
  std::string DumpBuildInformation;

public:
  DiagnosticOptions() {
    ShowColumn = 1;
    ShowLocation = 1;
    ShowCarets = 1;
    ShowFixits = 1;
    ShowSourceRanges = 0;
    ShowOptionNames = 0;
    ShowColors = 0;
    MessageLength = 0;
  }
};

}  // end namespace clang

#endif
