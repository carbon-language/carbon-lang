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

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/LLVM.h"

namespace clang {

class Stmt;
class TagDecl;
class LangOptions;

class PrinterHelper {
public:
  virtual ~PrinterHelper();
  virtual bool handledStmt(Stmt* E, raw_ostream& OS) = 0;
};

/// \brief Describes how types, statements, expressions, and
/// declarations should be printed.
struct PrintingPolicy {
  /// \brief Create a default printing policy for C.
  PrintingPolicy(const LangOptions &LO)
    : Indentation(2), LangOpts(LO), SuppressSpecifiers(false),
      SuppressTagKeyword(false), SuppressTag(false), SuppressScope(false),
      SuppressInitializers(false),
      Dump(false), ConstantArraySizeAsWritten(false),
      AnonymousTagLocations(true), SuppressStrongLifetime(false),
      Bool(LO.Bool) { }

  /// \brief The number of spaces to use to indent each line.
  unsigned Indentation : 8;

  /// \brief What language we're printing.
  LangOptions LangOpts;

  /// \brief Whether we should suppress printing of the actual specifiers for
  /// the given type or declaration.
  ///
  /// This flag is only used when we are printing declarators beyond
  /// the first declarator within a declaration group. For example, given:
  ///
  /// \code
  /// const int *x, *y;
  /// \endcode
  ///
  /// SuppressSpecifiers will be false when printing the
  /// declaration for "x", so that we will print "int *x"; it will be
  /// \c true when we print "y", so that we suppress printing the
  /// "const int" type specifier and instead only print the "*y".
  bool SuppressSpecifiers : 1;

  /// \brief Whether type printing should skip printing the tag keyword.
  ///
  /// This is used when printing the inner type of elaborated types,
  /// (as the tag keyword is part of the elaborated type):
  ///
  /// \code
  /// struct Geometry::Point;
  /// \endcode
  bool SuppressTagKeyword : 1;

  /// \brief Whether type printing should skip printing the actual tag type.
  ///
  /// This is used when the caller needs to print a tag definition in front
  /// of the type, as in constructs like the following:
  ///
  /// \code
  /// typedef struct { int x, y; } Point;
  /// \endcode
  bool SuppressTag : 1;

  /// \brief Suppresses printing of scope specifiers.
  bool SuppressScope : 1;

  /// \brief Suppress printing of variable initializers.
  ///
  /// This flag is used when printing the loop variable in a for-range
  /// statement. For example, given:
  ///
  /// \code
  /// for (auto x : coll)
  /// \endcode
  ///
  /// SuppressInitializers will be true when printing "auto x", so that the
  /// internal initializer constructed for x will not be printed.
  bool SuppressInitializers : 1;

  /// \brief True when we are "dumping" rather than "pretty-printing",
  /// where dumping involves printing the internal details of the AST
  /// and pretty-printing involves printing something similar to
  /// source code.
  bool Dump : 1;

  /// \brief Whether we should print the sizes of constant array expressions
  /// as written in the sources.
  ///
  /// This flag is determines whether arrays types declared as
  ///
  /// \code
  /// int a[4+10*10];
  /// char a[] = "A string";
  /// \endcode
  ///
  /// will be printed as written or as follows:
  ///
  /// \code
  /// int a[104];
  /// char a[9] = "A string";
  /// \endcode
  bool ConstantArraySizeAsWritten : 1;
  
  /// \brief When printing an anonymous tag name, also print the location of
  /// that entity (e.g., "enum <anonymous at t.h:10:5>"). Otherwise, just 
  /// prints "<anonymous>" for the name.
  bool AnonymousTagLocations : 1;
  
  /// \brief When true, suppress printing of the __strong lifetime qualifier in
  /// ARC.
  unsigned SuppressStrongLifetime : 1;
  
  /// \brief Whether we can use 'bool' rather than '_Bool', even if the language
  /// doesn't actually have 'bool' (because, e.g., it is defined as a macro).
  unsigned Bool : 1;
};

} // end namespace clang

#endif
