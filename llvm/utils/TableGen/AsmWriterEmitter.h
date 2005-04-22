//===- AsmWriterEmitter.h - Generate an assembly writer ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting an assembly printer for the
// code generator.
//
//===----------------------------------------------------------------------===//

#ifndef ASMWRITER_EMITTER_H
#define ASMWRITER_EMITTER_H

#include "TableGenBackend.h"

namespace llvm {

  class AsmWriterEmitter : public TableGenBackend {
    RecordKeeper &Records;
  public:
    AsmWriterEmitter(RecordKeeper &R) : Records(R) {}

    // run - Output the asmwriter, returning true on failure.
    void run(std::ostream &o);
  };
}
#endif
