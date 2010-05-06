//===- ClangASTNodesEmitter.h - Generate Clang AST node tables -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emit Clang AST node tables
//
//===----------------------------------------------------------------------===//

#ifndef CLANGAST_EMITTER_H
#define CLANGAST_EMITTER_H

#include "TableGenBackend.h"

namespace llvm {

/// ClangStmtNodesEmitter - The top-level class emits .def files containing
///  declarations of Clang statements.
///
class ClangStmtNodesEmitter : public TableGenBackend {
  RecordKeeper &Records;
public:
  explicit ClangStmtNodesEmitter(RecordKeeper &R)
    : Records(R) {}

  // run - Output the .def file contents
  void run(raw_ostream &OS);
};

} // End llvm namespace

#endif
