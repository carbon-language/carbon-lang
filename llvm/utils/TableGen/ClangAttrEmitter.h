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

/// ClangAttrImplEmitter - class emits the class method defintions for
///   attributes for clang.
class ClangAttrImplEmitter : public TableGenBackend {
  RecordKeeper &Records;
 
 public:
  explicit ClangAttrImplEmitter(RecordKeeper &R)
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

/// ClangAttrPCHReadEmitter - class emits the code to read an attribute from
///   a clang precompiled header.
class ClangAttrPCHReadEmitter : public TableGenBackend {
  RecordKeeper &Records;

public:
  explicit ClangAttrPCHReadEmitter(RecordKeeper &R)
    : Records(R)
    {}

  void run(raw_ostream &OS);
};

/// ClangAttrPCHWriteEmitter - class emits the code to read an attribute from
///   a clang precompiled header.
class ClangAttrPCHWriteEmitter : public TableGenBackend {
  RecordKeeper &Records;

public:
  explicit ClangAttrPCHWriteEmitter(RecordKeeper &R)
    : Records(R)
    {}

  void run(raw_ostream &OS);
};

/// ClangAttrSpellingListEmitter - class emits the list of spellings for attributes for
///   clang.
class ClangAttrSpellingListEmitter : public TableGenBackend {
  RecordKeeper &Records;

 public:
  explicit ClangAttrSpellingListEmitter(RecordKeeper &R)
    : Records(R)
    {}

  void run(raw_ostream &OS);
};

/// ClangAttrLateParsedListEmitter emits the LateParsed property for attributes
/// for clang.
class ClangAttrLateParsedListEmitter : public TableGenBackend {
  RecordKeeper &Records;

 public:
  explicit ClangAttrLateParsedListEmitter(RecordKeeper &R)
    : Records(R)
    {}

  void run(raw_ostream &OS);
};

}

#endif
