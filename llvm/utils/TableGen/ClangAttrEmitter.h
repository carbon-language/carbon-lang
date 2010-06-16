//===- ClangAttrEmitter.h - Generate Clang attribute handling =-*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emit Clang attribute processing code
//
//===----------------------------------------------------------------------===//

#ifndef CLANGATTR_EMITTER_H
#define CLANGATTR_EMITTER_H

#include "TableGenBackend.h"

namespace llvm {

/// ClangAttrClassEmitter - class emits the class defintions for attributes for
///   clang.
class ClangAttrClassEmitter : public TableGenBackend {
  RecordKeeper &Records;
 
 public:
  explicit ClangAttrClassEmitter(RecordKeeper &R)
    : Records(R)
    {}

  void run(raw_ostream &OS);
};

/// ClangAttrListEmitter - class emits the enumeration list for attributes for
///   clang.
class ClangAttrListEmitter : public TableGenBackend {
  RecordKeeper &Records;

 public:
  explicit ClangAttrListEmitter(RecordKeeper &R)
    : Records(R)
    {}

  void run(raw_ostream &OS);
};

}

#endif
