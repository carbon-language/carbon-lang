//===- LLVMCConfigurationEmitter.cpp - Generate LLVMCC config -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting LLVMCC configuration code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMCCCONF_EMITTER_H
#define LLVMCCCONF_EMITTER_H

#include "TableGenBackend.h"

namespace llvm {
  class LLVMCCConfigurationEmitter : public TableGenBackend {
    RecordKeeper &Records;
  public:
    explicit LLVMCCConfigurationEmitter(RecordKeeper &R) : Records(R) {}

    // run - Output the asmwriter, returning true on failure.
    void run(std::ostream &o);
  };
}

#endif //LLVMCCCONF_EMITTER_H
