//===- RISCDisassemblerEmitter.h - Disassembler Generator -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FIXME: document
//
//===----------------------------------------------------------------------===//

#ifndef RISCDISASSEMBLEREMITTER_H
#define RISCDISASSEMBLEREMITTER_H

#include "TableGenBackend.h"

#include <inttypes.h>

namespace llvm {
    
class RISCDisassemblerEmitter : public TableGenBackend {
  RecordKeeper &Records;
public:
  RISCDisassemblerEmitter(RecordKeeper &R) : Records(R) {
    initBackend();
  }
    
  ~RISCDisassemblerEmitter() {
    shutdownBackend();
  }
	
  // run - Output the code emitter
  void run(raw_ostream &o);
    
private:
  class RISCDEBackend;
    
  RISCDEBackend *Backend;
    
  void initBackend();
  void shutdownBackend();
};

} // end llvm namespace

#endif
