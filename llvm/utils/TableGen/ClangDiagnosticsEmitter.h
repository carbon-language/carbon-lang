//===- ClangDiagnosticsEmitter.h - Generate Clang diagnostics tables -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emit Clang diagnostics tables.
//
//===----------------------------------------------------------------------===//

#ifndef CLANGDIAGS_EMITTER_H
#define CLANGDIAGS_EMITTER_H

#include "TableGenBackend.h"

namespace llvm {

/// ClangDiagsDefsEmitter - The top-level class emits .def files containing
///  declarations of Clang diagnostics.
///
class ClangDiagsDefsEmitter : public TableGenBackend {
  RecordKeeper &Records;
  const std::string& Component;
public:
  explicit ClangDiagsDefsEmitter(RecordKeeper &R, const std::string& component)
    : Records(R), Component(component) {}

  // run - Output the .def file contents
  void run(raw_ostream &OS);
};

class ClangDiagGroupsEmitter : public TableGenBackend {
    RecordKeeper &Records;
public:
  explicit ClangDiagGroupsEmitter(RecordKeeper &R) : Records(R) {}
    
  void run(raw_ostream &OS);
};

  
} // End llvm namespace

#endif
