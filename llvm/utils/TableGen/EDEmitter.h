//===- EDEmitter.h - Generate instruction descriptions for ED ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of each
// instruction in a format that the semantic disassembler can use to tokenize
// and parse instructions.
//
//===----------------------------------------------------------------------===//

#ifndef SEMANTIC_INFO_EMITTER_H
#define SEMANTIC_INFO_EMITTER_H

#include "TableGenBackend.h"

namespace llvm {
  
  class EDEmitter : public TableGenBackend {
    RecordKeeper &Records;
  public:
    EDEmitter(RecordKeeper &R);
    
    // run - Output the instruction table.
    void run(raw_ostream &o);
  };
  
} // End llvm namespace

#endif
