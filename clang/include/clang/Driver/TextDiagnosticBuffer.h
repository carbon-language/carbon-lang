//===--- TextDiagnosticBuffer.h - Buffer Text Diagnostics -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a concrete diagnostic client, which buffers the diagnostic messages.
//
//===----------------------------------------------------------------------===//

#ifndef DRIVER_TEXT_DIAGNOSTIC_BUFFER_H_
#define DRIVER_TEXT_DIAGNOSTIC_BUFFER_H_

#include "clang/Driver/TextDiagnostics.h"
#include <vector>

namespace clang {

class Preprocessor;
class SourceManager;

class TextDiagnosticBuffer : public TextDiagnostics {
public:
  typedef std::vector<std::pair<SourceLocation, std::string> > DiagList;
  typedef DiagList::iterator iterator;
  typedef DiagList::const_iterator const_iterator;
private:
  DiagList Errors, Warnings;
public:
  TextDiagnosticBuffer() {}

  const_iterator err_begin() const  { return Errors.begin(); }
  const_iterator err_end() const    { return Errors.end(); }

  const_iterator warn_begin() const { return Warnings.begin(); }
  const_iterator warn_end() const   { return Warnings.end(); }

  virtual void HandleDiagnostic(Diagnostic &Diags,
                                Diagnostic::Level DiagLevel,
                                FullSourceLoc Pos,
                                diag::kind ID,
                                const std::string *Strs,
                                unsigned NumStrs,
                                const SourceRange *Ranges, 
                                unsigned NumRanges);
};

} // end namspace clang

#endif
