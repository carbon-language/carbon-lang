//===- ClangASTNodesEmitter.h - Generate Clang AST node tables -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emit Clang AST node tables
//
//===----------------------------------------------------------------------===//

#ifndef CLANGAST_EMITTER_H
#define CLANGAST_EMITTER_H

#include "TableGenBackend.h"
#include "Record.h"
#include <string>
#include <cctype>
#include <map>

namespace llvm {

/// ClangASTNodesEmitter - The top-level class emits .inc files containing
///  declarations of Clang statements.
///
class ClangASTNodesEmitter : public TableGenBackend {
  // A map from a node to each of its derived nodes.
  typedef std::multimap<Record*, Record*> ChildMap;
  typedef ChildMap::const_iterator ChildIterator;

  RecordKeeper &Records;
  Record Root;
  const std::string &BaseSuffix;

  // Create a macro-ized version of a name
  static std::string macroName(std::string S) {
    for (unsigned i = 0; i < S.size(); ++i)
      S[i] = std::toupper(S[i]);

    return S;
  }

  // Return the name to be printed in the base field. Normally this is
  // the record's name plus the base suffix, but if it is the root node and
  // the suffix is non-empty, it's just the suffix.
  std::string baseName(Record &R) {
    if (&R == &Root && !BaseSuffix.empty())
      return BaseSuffix;
    
    return R.getName() + BaseSuffix;
  }

  std::pair<Record *, Record *> EmitNode (const ChildMap &Tree, raw_ostream& OS,
                                          Record *Base);
public:
  explicit ClangASTNodesEmitter(RecordKeeper &R, const std::string &N,
                                const std::string &S)
    : Records(R), Root(N, SMLoc()), BaseSuffix(S)
    {}

  // run - Output the .inc file contents
  void run(raw_ostream &OS);
};

/// ClangDeclContextEmitter - Emits an addendum to a .inc file to enumerate the
/// clang declaration contexts.
///
class ClangDeclContextEmitter : public TableGenBackend {
  RecordKeeper &Records;

public:
  explicit ClangDeclContextEmitter(RecordKeeper &R)
    : Records(R)
  {}

  // run - Output the .inc file contents
  void run(raw_ostream &OS);
};

} // End llvm namespace

#endif
