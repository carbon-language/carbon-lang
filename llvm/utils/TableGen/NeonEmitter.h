//===- NeonEmitter.h - Generate arm_neon.h for use with clang ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting arm_neon.h, which includes
// a declaration and definition of each function specified by the ARM NEON 
// compiler interface.  See ARM document DUI0348B.
//
//===----------------------------------------------------------------------===//

#ifndef NEON_EMITTER_H
#define NEON_EMITTER_H

#include "TableGenBackend.h"

namespace llvm {
  
  class NeonEmitter : public TableGenBackend {
    RecordKeeper &Records;
  public:
    NeonEmitter(RecordKeeper &R) : Records(R) {}
    
    // run - Emit arm_neon.h.inc
    void run(raw_ostream &o);

    // runHeader - Emit all the __builtin prototypes used in arm_neon.h
    void runHeader(raw_ostream &o);
  };
  
} // End llvm namespace

#endif
