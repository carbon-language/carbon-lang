//======- AttributeCommonInfo.h - Base info about Attributes-----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeCommonInfo type, which is the base for a
// ParsedAttr and is used by Attr as a way to share info between the two.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ATTRIBUTECOMMONINFO_H
#define LLVM_CLANG_BASIC_ATTRIBUTECOMMONINFO_H
#include "clang/Basic/SourceLocation.h"

namespace clang {
class IdentifierInfo;
class ASTRecordWriter;

class AttributeCommonInfo {
public:
  /// The style used to specify an attribute.
  enum Syntax {
    /// __attribute__((...))
    AS_GNU,

    /// [[...]]
    AS_CXX11,

    /// [[...]]
    AS_C2x,

    /// __declspec(...)
    AS_Declspec,

    /// [uuid("...")] class Foo
    AS_Microsoft,

    /// __ptr16, alignas(...), etc.
    AS_Keyword,

    /// #pragma ...
    AS_Pragma,

    // Note TableGen depends on the order above.  Do not add or change the order
    // without adding related code to TableGen/ClangAttrEmitter.cpp.
    /// Context-sensitive version of a keyword attribute.
    AS_ContextSensitiveKeyword,
  };
  enum Kind {
#define PARSED_ATTR(NAME) AT_##NAME,
#include "clang/Sema/AttrParsedAttrList.inc"
#undef PARSED_ATTR
    NoSemaHandlerAttribute,
    IgnoredAttribute,
    UnknownAttribute,
  };

private:
  const IdentifierInfo *AttrName = nullptr;
  const IdentifierInfo *ScopeName = nullptr;
  SourceRange AttrRange;
  const SourceLocation ScopeLoc;
  // Corresponds to the Kind enum.
  unsigned AttrKind : 16;
  /// Corresponds to the Syntax enum.
  unsigned SyntaxUsed : 3;
  unsigned SpellingIndex : 4;

protected:
  static constexpr unsigned SpellingNotCalculated = 0xf;

public:
  AttributeCommonInfo(SourceRange AttrRange)
      : AttrRange(AttrRange), ScopeLoc(), AttrKind(0), SyntaxUsed(0),
        SpellingIndex(SpellingNotCalculated) {}

  AttributeCommonInfo(SourceLocation AttrLoc)
      : AttrRange(AttrLoc), ScopeLoc(), AttrKind(0), SyntaxUsed(0),
        SpellingIndex(SpellingNotCalculated) {}

  AttributeCommonInfo(const IdentifierInfo *AttrName,
                      const IdentifierInfo *ScopeName, SourceRange AttrRange,
                      SourceLocation ScopeLoc, Syntax SyntaxUsed)
      : AttrName(AttrName), ScopeName(ScopeName), AttrRange(AttrRange),
        ScopeLoc(ScopeLoc),
        AttrKind(getParsedKind(AttrName, ScopeName, SyntaxUsed)),
        SyntaxUsed(SyntaxUsed), SpellingIndex(SpellingNotCalculated) {}

  AttributeCommonInfo(const IdentifierInfo *AttrName,
                      const IdentifierInfo *ScopeName, SourceRange AttrRange,
                      SourceLocation ScopeLoc, Kind AttrKind, Syntax SyntaxUsed)
      : AttrName(AttrName), ScopeName(ScopeName), AttrRange(AttrRange),
        ScopeLoc(ScopeLoc), AttrKind(AttrKind), SyntaxUsed(SyntaxUsed),
        SpellingIndex(SpellingNotCalculated) {}

  AttributeCommonInfo(const IdentifierInfo *AttrName,
                      const IdentifierInfo *ScopeName, SourceRange AttrRange,
                      SourceLocation ScopeLoc, Kind AttrKind, Syntax SyntaxUsed,
                      unsigned Spelling)
      : AttrName(AttrName), ScopeName(ScopeName), AttrRange(AttrRange),
        ScopeLoc(ScopeLoc), AttrKind(AttrKind), SyntaxUsed(SyntaxUsed),
        SpellingIndex(Spelling) {}

  AttributeCommonInfo(const IdentifierInfo *AttrName, SourceRange AttrRange,
                      Syntax SyntaxUsed)
      : AttrName(AttrName), ScopeName(nullptr), AttrRange(AttrRange),
        ScopeLoc(), AttrKind(getParsedKind(AttrName, ScopeName, SyntaxUsed)),
        SyntaxUsed(SyntaxUsed), SpellingIndex(SpellingNotCalculated) {}

  AttributeCommonInfo(SourceRange AttrRange, Kind K, Syntax SyntaxUsed)
      : AttrName(nullptr), ScopeName(nullptr), AttrRange(AttrRange), ScopeLoc(),
        AttrKind(K), SyntaxUsed(SyntaxUsed),
        SpellingIndex(SpellingNotCalculated) {}

  AttributeCommonInfo(SourceRange AttrRange, Kind K, Syntax SyntaxUsed,
                      unsigned Spelling)
      : AttrName(nullptr), ScopeName(nullptr), AttrRange(AttrRange), ScopeLoc(),
        AttrKind(K), SyntaxUsed(SyntaxUsed), SpellingIndex(Spelling) {}

  AttributeCommonInfo(AttributeCommonInfo &&) = default;
  AttributeCommonInfo(const AttributeCommonInfo &) = default;

  Kind getParsedKind() const { return Kind(AttrKind); }
  Syntax getSyntax() const { return Syntax(SyntaxUsed); }
  const IdentifierInfo *getAttrName() const { return AttrName; }
  SourceLocation getLoc() const { return AttrRange.getBegin(); }
  SourceRange getRange() const { return AttrRange; }
  void setRange(SourceRange R) { AttrRange = R; }

  bool hasScope() const { return ScopeName; }
  const IdentifierInfo *getScopeName() const { return ScopeName; }
  SourceLocation getScopeLoc() const { return ScopeLoc; }

  /// Gets the normalized full name, which consists of both scope and name and
  /// with surrounding underscores removed as appropriate (e.g.
  /// __gnu__::__attr__ will be normalized to gnu::attr).
  std::string getNormalizedFullName() const;

  bool isDeclspecAttribute() const { return SyntaxUsed == AS_Declspec; }
  bool isMicrosoftAttribute() const { return SyntaxUsed == AS_Microsoft; }

  bool isGNUScope() const;

  bool isAlignasAttribute() const {
    // FIXME: Use a better mechanism to determine this.
    return getParsedKind() == AT_Aligned && isKeywordAttribute();
  }

  bool isCXX11Attribute() const {
    return SyntaxUsed == AS_CXX11 || isAlignasAttribute();
  }

  bool isC2xAttribute() const { return SyntaxUsed == AS_C2x; }

  /// The attribute is spelled [[]] in either C or C++ mode, including standard
  /// attributes spelled with a keyword, like alignas.
  bool isStandardAttributeSyntax() const {
    return isCXX11Attribute() || isC2xAttribute();
  }

  bool isKeywordAttribute() const {
    return SyntaxUsed == AS_Keyword || SyntaxUsed == AS_ContextSensitiveKeyword;
  }

  bool isContextSensitiveKeywordAttribute() const {
    return SyntaxUsed == AS_ContextSensitiveKeyword;
  }

  unsigned getAttributeSpellingListIndex() const {
    assert((isAttributeSpellingListCalculated() || AttrName) &&
           "Spelling cannot be found");
    return isAttributeSpellingListCalculated()
               ? SpellingIndex
               : calculateAttributeSpellingListIndex();
  }
  void setAttributeSpellingListIndex(unsigned V) { SpellingIndex = V; }

  static Kind getParsedKind(const IdentifierInfo *Name,
                            const IdentifierInfo *Scope, Syntax SyntaxUsed);

private:
  /// Get an index into the attribute spelling list
  /// defined in Attr.td. This index is used by an attribute
  /// to pretty print itself.
  unsigned calculateAttributeSpellingListIndex() const;

  friend class clang::ASTRecordWriter;
  // Used exclusively by ASTDeclWriter to get the raw spelling list state.
  unsigned getAttributeSpellingListIndexRaw() const { return SpellingIndex; }

protected:
  bool isAttributeSpellingListCalculated() const {
    return SpellingIndex != SpellingNotCalculated;
  }
};
} // namespace clang

#endif // LLVM_CLANG_BASIC_ATTRIBUTECOMMONINFO_H
