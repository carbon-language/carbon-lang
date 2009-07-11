//===- AsmMatcherEmitter.h - Generate an assembly matcher -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a target specifier matcher for converting parsed
// assembly operands in the MCInst structures.
//
//===----------------------------------------------------------------------===//

#ifndef ASMMATCHER_EMITTER_H
#define ASMMATCHER_EMITTER_H

#include "TableGenBackend.h"
#include <map>
#include <vector>
#include <cassert>

namespace llvm {
  class AsmMatcherEmitter : public TableGenBackend {
    RecordKeeper &Records;
  public:
    AsmMatcherEmitter(RecordKeeper &R) : Records(R) {}

    // run - Output the matcher, returning true on failure.
    void run(raw_ostream &o);
  };
}
#endif
