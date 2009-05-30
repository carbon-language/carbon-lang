//===--- PrettyPrinter.h - Classes for aiding with AST printing -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PrinterHelper interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_PRETTY_PRINTER_H
#define LLVM_CLANG_AST_PRETTY_PRINTER_H

namespace llvm {
  class raw_ostream;
}

namespace clang {

class Stmt;
class TagDecl;

class PrinterHelper {
public:
  virtual ~PrinterHelper();
  virtual bool handledStmt(Stmt* E, llvm::raw_ostream& OS) = 0;
};

/// \brief Describes how types, statements, expressions, and
/// declarations should be printed.
struct PrintingPolicy {
  /// \brief Create a default printing policy for C.
  PrintingPolicy() 
    : Indentation(2), CPlusPlus(false), SuppressTypeSpecifiers(false),
      SuppressTagKind(false), Dump(false), OwnedTag(0) { }

  /// \brief The number of spaces to use to indent each line.
  unsigned Indentation : 8;

  /// \brief Whether we're printing C++ code (otherwise, we're
  /// printing C code).
  bool CPlusPlus : 1;

  /// \brief Whether we should suppress printing of the actual type
  /// specifiers within the type that we are printing.
  ///
  /// This flag is only used when we are printing declarators beyond
  /// the first declarator within a declaration group. For example, given:
  ///
  /// \code
  /// const int *x, *y;
  /// \endcode
  ///
  /// SuppressTypeSpecifiers will be false when printing the
  /// declaration for "x", so that we will print "int *x"; it will be
  /// \c true when we print "y", so that we suppress printing the
  /// "const int" type specifier and instead only print the "*y".
  bool SuppressTypeSpecifiers : 1;

  /// \brief If we are printing a tag type, suppresses printing of the
  /// kind of tag, e.g., "struct", "union", "enum".
  bool SuppressTagKind : 1;

  /// \brief True when we are "dumping" rather than "pretty-printing",
  /// where dumping involves printing the internal details of the AST
  /// and pretty-printing involves printing something similar to
  /// source code.
  bool Dump : 1;

  /// \brief If we are printing a type where the tag type (e.g., a
  /// class or enum type) was declared or defined within the type
  /// itself, OwnedTag will point at the declaration node owned by
  /// this type.
  ///
  /// Owned tags occur when a tag type is defined as part of the
  /// declaration specifiers of another declarator, e.g.,
  ///
  /// \code
  /// typedef struct { int x, y; } Point;
  /// \endcode
  ///
  /// Here, the anonymous struct definition is owned by the type of
  /// Point. The actual representation uses a DeclGroup to store both
  /// the RecordDecl and the TypedefDecl.
  TagDecl *OwnedTag;
};

} // end namespace clang

#endif
