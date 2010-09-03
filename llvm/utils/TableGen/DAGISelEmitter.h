//===- DAGISelEmitter.h - Generate an instruction selector ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a DAG instruction selector.
//
//===----------------------------------------------------------------------===//

#ifndef DAGISEL_EMITTER_H
#define DAGISEL_EMITTER_H

#include "TableGenBackend.h"
#include "CodeGenDAGPatterns.h"
#include <set>

namespace llvm {

/// DAGISelEmitter - The top-level class which coordinates construction
/// and emission of the instruction selector.
///
class DAGISelEmitter : public TableGenBackend {
  RecordKeeper &Records;
  CodeGenDAGPatterns CGP;
public:
  explicit DAGISelEmitter(RecordKeeper &R) : Records(R), CGP(R) {}

  // run - Output the isel, returning true on failure.
  void run(raw_ostream &OS);
};

} // End llvm namespace

#endif
