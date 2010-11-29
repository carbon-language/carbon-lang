//===------------ ARMDecoderEmitter.h - Decoder Generator -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the ARM Disassembler.
// It contains the tablegen backend declaration ARMDecoderEmitter.
//
//===----------------------------------------------------------------------===//

#ifndef ARMDECODEREMITTER_H
#define ARMDECODEREMITTER_H

#include "TableGenBackend.h"

#include "llvm/Support/DataTypes.h"

namespace llvm {

class ARMDecoderEmitter : public TableGenBackend {
  RecordKeeper &Records;
public:
  ARMDecoderEmitter(RecordKeeper &R) : Records(R) {
    initBackend();
  }
    
  ~ARMDecoderEmitter() {
    shutdownBackend();
  }

  // run - Output the code emitter
  void run(raw_ostream &o);
    
private:
  // Helper class for ARMDecoderEmitter.
  class ARMDEBackend;

  ARMDEBackend *Backend;
    
  void initBackend();
  void shutdownBackend();
};

} // end llvm namespace

#endif
