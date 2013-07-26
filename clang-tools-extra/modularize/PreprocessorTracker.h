//===- PreprocessorTracker.h - Tracks preprocessor activities -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//
///
/// \file
/// \brief Track preprocessor activities for modularize.
///
//===--------------------------------------------------------------------===//

#ifndef MODULARIZE_PREPROCESSOR_TRACKER_H
#define MODULARIZE_PREPROCESSOR_TRACKER_H

#include "clang/Lex/Preprocessor.h"

namespace Modularize {

// Preprocessor tracker for modularize.
//
// This class stores information about all the headers processed in the
// course of running modularize.
class PreprocessorTracker {
public:
  virtual ~PreprocessorTracker();

  // Handle entering a preprocessing session.
  // (Called after a Preprocessor object is created, but before preprocessing.)
  virtual void handlePreprocessorEntry(clang::Preprocessor &PP,
                                       llvm::StringRef RootHeaderFile) = 0;
  // Handle exiting a preprocessing session.
  // (Called after preprocessing is complete, but before the Preprocessor
  // object is destroyed.)
  virtual void handlePreprocessorExit() = 0;

  // Report on inconsistent macro instances.
  // Returns true if any mismatches.
  virtual bool reportInconsistentMacros(llvm::raw_ostream &OS) = 0;

  // Report on inconsistent conditional directive instances.
  // Returns true if any mismatches.
  virtual bool reportInconsistentConditionals(llvm::raw_ostream &OS) = 0;

  // Create instance of PreprocessorTracker.
  static PreprocessorTracker *create();
};

} // end namespace Modularize

#endif
