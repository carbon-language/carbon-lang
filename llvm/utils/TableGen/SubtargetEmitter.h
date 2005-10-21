//===- SubtargetEmitter.h - Generate subtarget enumerations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits subtarget enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef SUBTARGET_EMITTER_H
#define SUBTARGET_EMITTER_H

#include "TableGenBackend.h"

namespace llvm {

class SubtargetEmitter : public TableGenBackend {
  RecordKeeper &Records;
public:
  SubtargetEmitter(RecordKeeper &R) : Records(R) {}

  // run - Output the subtarget enumerations, returning true on failure.
  void run(std::ostream &o);

};


} // End llvm namespace

#endif



