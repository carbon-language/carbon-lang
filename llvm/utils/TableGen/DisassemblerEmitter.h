//===- DisassemblerEmitter.h - Disassembler Generator -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef DISASSEMBLEREMITTER_H
#define DISASSEMBLEREMITTER_H

#include "llvm/TableGen/TableGenBackend.h"

namespace llvm {

  class DisassemblerEmitter : public TableGenBackend {
    RecordKeeper &Records;
  public:
    DisassemblerEmitter(RecordKeeper &R) : Records(R) {}

    /// run - Output the disassembler.
    void run(raw_ostream &o);
  };

} // end llvm namespace

#endif
