//===--- Analyzer.h - Analysis for indexing information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Analyzer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_ANALYZER_H
#define LLVM_CLANG_INDEX_ANALYZER_H

namespace clang {
  class Decl;
  class ObjCMessageExpr;

namespace idx {
  class Program;
  class IndexProvider;
  class TULocationHandler;

/// \brief Provides indexing information, like finding all references of an
/// Entity across translation units.
class Analyzer {
  Program &Prog;
  IndexProvider &Idxer;

  Analyzer(const Analyzer&); // do not implement
  Analyzer &operator=(const Analyzer &); // do not implement

public:
  explicit Analyzer(Program &prog, IndexProvider &idxer)
    : Prog(prog), Idxer(idxer) { }

  /// \brief Find all TULocations for declarations of the given Decl and pass
  /// them to Handler.
  void FindDeclarations(Decl *D, TULocationHandler &Handler);

  /// \brief Find all TULocations for references of the given Decl and pass
  /// them to Handler.
  void FindReferences(Decl *D, TULocationHandler &Handler);

  /// \brief Find methods that may respond to the given message and pass them
  /// to Handler.
  void FindObjCMethods(ObjCMessageExpr *MsgE, TULocationHandler &Handler);
};

} // namespace idx

} // namespace clang

#endif
