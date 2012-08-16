//===--- CommentCommandTraits.h - Comment command properties ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the class that provides information about comment
//  commands.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CLANG_AST_COMMENT_COMMAND_TRAITS_H
#define LLVM_CLANG_AST_COMMENT_COMMAND_TRAITS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

namespace clang {
namespace comments {

/// This class provides informaiton about commands that can be used
/// in comments.
class CommandTraits {
public:
  CommandTraits() { }

  /// \brief Check if a given command is a verbatim-like block command.
  ///
  /// A verbatim-like block command eats every character (except line starting
  /// decorations) until matching end command is seen or comment end is hit.
  ///
  /// \param StartName name of the command that starts the verbatim block.
  /// \param [out] EndName name of the command that ends the verbatim block.
  ///
  /// \returns true if a given command is a verbatim block command.
  bool isVerbatimBlockCommand(StringRef StartName, StringRef &EndName) const;

  /// \brief Register a new verbatim block command.
  void addVerbatimBlockCommand(StringRef StartName, StringRef EndName);

  /// \brief Check if a given command is a verbatim line command.
  ///
  /// A verbatim-like line command eats everything until a newline is seen or
  /// comment end is hit.
  bool isVerbatimLineCommand(StringRef Name) const;

  /// \brief Check if a given command is a command that contains a declaration
  /// for the entity being documented.
  ///
  /// For example:
  /// \code
  ///   \fn void f(int a);
  /// \endcode
  bool isDeclarationCommand(StringRef Name) const;

  /// \brief Register a new verbatim line command.
  void addVerbatimLineCommand(StringRef Name);

  /// \brief Check if a given command is a block command (of any kind).
  bool isBlockCommand(StringRef Name) const;

  /// \brief Check if a given command is introducing documentation for
  /// a function parameter (\\param or an alias).
  bool isParamCommand(StringRef Name) const;

  /// \brief Check if a given command is introducing documentation for
  /// a template parameter (\\tparam or an alias).
  bool isTParamCommand(StringRef Name) const;

  /// \brief Check if a given command is introducing a brief documentation
  /// paragraph (\\brief or an alias).
  bool isBriefCommand(StringRef Name) const;

  /// \brief Check if a given command is \\brief or an alias.
  bool isReturnsCommand(StringRef Name) const;

  /// \returns the number of word-like arguments for a given block command,
  /// except for \\param and \\tparam commands -- these have special argument
  /// parsers.
  unsigned getBlockCommandNumArgs(StringRef Name) const;

  /// \brief Check if a given command is a inline command (of any kind).
  bool isInlineCommand(StringRef Name) const;

private:
  struct VerbatimBlockCommand {
    StringRef StartName;
    StringRef EndName;
  };

  typedef SmallVector<VerbatimBlockCommand, 4> VerbatimBlockCommandVector;

  /// Registered additional verbatim-like block commands.
  VerbatimBlockCommandVector VerbatimBlockCommands;

  struct VerbatimLineCommand {
    StringRef Name;
  };

  typedef SmallVector<VerbatimLineCommand, 4> VerbatimLineCommandVector;

  /// Registered verbatim-like line commands.
  VerbatimLineCommandVector VerbatimLineCommands;
};

inline bool CommandTraits::isBlockCommand(StringRef Name) const {
  return isBriefCommand(Name) || isReturnsCommand(Name) ||
      isParamCommand(Name) || isTParamCommand(Name) ||
      llvm::StringSwitch<bool>(Name)
      .Case("author", true)
      .Case("authors", true)
      .Case("pre", true)
      .Case("post", true)
      .Default(false);
}

inline bool CommandTraits::isParamCommand(StringRef Name) const {
  return Name == "param";
}

inline bool CommandTraits::isTParamCommand(StringRef Name) const {
  return Name == "tparam" || // Doxygen
         Name == "templatefield"; // HeaderDoc
}

inline bool CommandTraits::isBriefCommand(StringRef Name) const {
  return Name == "brief" || Name == "short";
}

inline bool CommandTraits::isReturnsCommand(StringRef Name) const {
  return Name == "returns" || Name == "return" || Name == "result";
}

inline unsigned CommandTraits::getBlockCommandNumArgs(StringRef Name) const {
  return 0;
}

inline bool CommandTraits::isInlineCommand(StringRef Name) const {
  return llvm::StringSwitch<bool>(Name)
      .Case("b", true)
      .Cases("c", "p", true)
      .Cases("a", "e", "em", true)
      .Default(false);
}

} // end namespace comments
} // end namespace clang

#endif

