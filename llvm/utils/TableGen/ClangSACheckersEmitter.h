//===- ClangSACheckersEmitter.h - Generate Clang SA checkers tables -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits Clang Static Analyzer checkers tables.
//
//===----------------------------------------------------------------------===//

#ifndef CLANGSACHECKERS_EMITTER_H
#define CLANGSACHECKERS_EMITTER_H

#include "llvm/TableGen/TableGenBackend.h"

namespace llvm {

class ClangSACheckersEmitter : public TableGenBackend {
    RecordKeeper &Records;
public:
  explicit ClangSACheckersEmitter(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS);
};

} // End llvm namespace

#endif
