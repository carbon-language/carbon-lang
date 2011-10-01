//===- OptParserEmitter.h - Table Driven Command Line Parsing ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_TABLEGEN_OPTPARSEREMITTER_H
#define UTILS_TABLEGEN_OPTPARSEREMITTER_H

#include "llvm/TableGen/TableGenBackend.h"

namespace llvm {
  /// OptParserEmitter - This tablegen backend takes an input .td file
  /// describing a list of options and emits a data structure for parsing and
  /// working with those options when given an input command line.
  class OptParserEmitter : public TableGenBackend {
    RecordKeeper &Records;
    bool GenDefs;

  public:
    OptParserEmitter(RecordKeeper &R, bool _GenDefs)
      : Records(R), GenDefs(_GenDefs) {}

    /// run - Output the option parsing information.
    ///
    /// \param GenHeader - Generate the header describing the option IDs.x
    void run(raw_ostream &OS);
  };
}

#endif
