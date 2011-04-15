//===--- LangStandard.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_LANGSTANDARD_H
#define LLVM_CLANG_FRONTEND_LANGSTANDARD_H

#include "llvm/ADT/StringRef.h"

namespace clang {

namespace frontend {

enum LangFeatures {
  BCPLComment = (1 << 0),
  C99 = (1 << 1),
  C1X = (1 << 2),
  CPlusPlus = (1 << 3),
  CPlusPlus0x = (1 << 4),
  Digraphs = (1 << 5),
  GNUMode = (1 << 6),
  HexFloat = (1 << 7),
  ImplicitInt = (1 << 8)
};

}

/// LangStandard - Information about the properties of a particular language
/// standard.
struct LangStandard {
  enum Kind {
#define LANGSTANDARD(id, name, desc, features) \
    lang_##id,
#include "clang/Frontend/LangStandards.def"
    lang_unspecified
  };

  const char *ShortName;
  const char *Description;
  unsigned Flags;

public:
  /// getName - Get the name of this standard.
  const char *getName() const { return ShortName; }

  /// getDescription - Get the description of this standard.
  const char *getDescription() const { return Description; }

  /// hasBCPLComments - Language supports '//' comments.
  bool hasBCPLComments() const { return Flags & frontend::BCPLComment; }

  /// isC99 - Language is a superset of C99.
  bool isC99() const { return Flags & frontend::C99; }

  /// isC1X - Language is a superset of C1X.
  bool isC1X() const { return Flags & frontend::C1X; }

  /// isCPlusPlus - Language is a C++ variant.
  bool isCPlusPlus() const { return Flags & frontend::CPlusPlus; }

  /// isCPlusPlus0x - Language is a C++0x variant.
  bool isCPlusPlus0x() const { return Flags & frontend::CPlusPlus0x; }

  /// hasDigraphs - Language supports digraphs.
  bool hasDigraphs() const { return Flags & frontend::Digraphs; }

  /// isGNUMode - Language includes GNU extensions.
  bool isGNUMode() const { return Flags & frontend::GNUMode; }

  /// hasHexFloats - Language supports hexadecimal float constants.
  bool hasHexFloats() const { return Flags & frontend::HexFloat; }

  /// hasImplicitInt - Language allows variables to be typed as int implicitly.
  bool hasImplicitInt() const { return Flags & frontend::ImplicitInt; }

  static const LangStandard &getLangStandardForKind(Kind K);
  static const LangStandard *getLangStandardForName(llvm::StringRef Name);
};

}  // end namespace clang

#endif
