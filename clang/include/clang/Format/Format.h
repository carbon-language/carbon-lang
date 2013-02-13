//===--- Format.h - Format C++ code -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Various functions to configurably format source code.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FORMAT_FORMAT_H
#define LLVM_CLANG_FORMAT_FORMAT_H

#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Refactoring.h"

namespace clang {

class Lexer;
class SourceManager;
class DiagnosticConsumer;

namespace format {

/// \brief The \c FormatStyle is used to configure the formatting to follow
/// specific guidelines.
struct FormatStyle {
  /// \brief The column limit.
  unsigned ColumnLimit;

  /// \brief The penalty for each character outside of the column limit.
  unsigned PenaltyExcessCharacter;

  /// \brief The maximum number of consecutive empty lines to keep.
  unsigned MaxEmptyLinesToKeep;

  /// \brief Set whether & and * bind to the type as opposed to the variable.
  bool PointerBindsToType;

  /// \brief If \c true, analyze the formatted file for the most common binding.
  bool DerivePointerBinding;

  /// \brief The extra indent or outdent of access modifiers (e.g.: public:).
  int AccessModifierOffset;

  enum LanguageStandard {
    LS_Cpp03,
    LS_Cpp11,
    LS_Auto
  };

  /// \brief Format compatible with this standard, e.g. use \c A<A<int> >
  /// instead of \c A<A<int>> for LS_Cpp03.
  LanguageStandard Standard;

  /// \brief If \c true, analyze the formatted file for C++03 compatibility.
  bool DeriveBackwardsCompatibility;

  /// \brief Indent case labels one level from the switch statement.
  ///
  /// When false, use the same indentation level as for the switch statement.
  /// Switch statement body is always indented one level more than case labels.
  bool IndentCaseLabels;

  /// \brief The number of spaces to before trailing line comments.
  unsigned SpacesBeforeTrailingComments;

  /// \brief If false, a function call's or function definition's parameters
  /// will either all be on the same line or will have one line each.
  bool BinPackParameters;

  /// \brief Allow putting all parameters of a function declaration onto
  /// the next line even if \c BinPackParameters is \c false.
  bool AllowAllParametersOfDeclarationOnNextLine;

  /// \brief Penalty for putting the return type of a function onto its own
  /// line.
  unsigned PenaltyReturnTypeOnItsOwnLine;

  /// \brief If the constructor initializers don't fit on a line, put each
  /// initializer on its own line.
  bool ConstructorInitializerAllOnOneLineOrOnePerLine;

  /// \brief If true, "if (a) return;" can be put on a single line.
  bool AllowShortIfStatementsOnASingleLine;

  /// \brief Add a space in front of an Objective-C protocol list, i.e. use
  /// Foo <Protocol> instead of Foo<Protocol>.
  bool ObjCSpaceBeforeProtocolList;
};

/// \brief Returns a format style complying with the LLVM coding standards:
/// http://llvm.org/docs/CodingStandards.html.
FormatStyle getLLVMStyle();

/// \brief Returns a format style complying with Google's C++ style guide:
/// http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml.
FormatStyle getGoogleStyle();

/// \brief Returns a format style complying with Chromium's style guide:
/// http://www.chromium.org/developers/coding-style.
FormatStyle getChromiumStyle();

/// \brief Reformats the given \p Ranges in the token stream coming out of
/// \c Lex.
///
/// Each range is extended on either end to its next bigger logic unit, i.e.
/// everything that might influence its formatting or might be influenced by its
/// formatting.
///
/// \param DiagClient A custom DiagnosticConsumer. Can be 0, in this case
/// diagnostic is output to llvm::errs().
///
/// Returns the \c Replacements necessary to make all \p Ranges comply with
/// \p Style.
tooling::Replacements reformat(const FormatStyle &Style, Lexer &Lex,
                               SourceManager &SourceMgr,
                               std::vector<CharSourceRange> Ranges,
                               DiagnosticConsumer *DiagClient = 0);

/// \brief Returns the \c LangOpts that the formatter expects you to set.
LangOptions getFormattingLangOpts();

} // end namespace format
} // end namespace clang

#endif // LLVM_CLANG_FORMAT_FORMAT_H
