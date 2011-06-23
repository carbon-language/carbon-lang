//===--- Token.h - Token interface ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Token interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOKEN_H
#define LLVM_CLANG_TOKEN_H

#include "clang/Basic/TemplateKinds.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/OperatorKinds.h"
#include <cstdlib>

namespace clang {

class IdentifierInfo;

/// Token - This structure provides full information about a lexed token.
/// It is not intended to be space efficient, it is intended to return as much
/// information as possible about each returned token.  This is expected to be
/// compressed into a smaller form if memory footprint is important.
///
/// The parser can create a special "annotation token" representing a stream of
/// tokens that were parsed and semantically resolved, e.g.: "foo::MyClass<int>"
/// can be represented by a single typename annotation token that carries
/// information about the SourceRange of the tokens and the type object.
class Token {
  /// The location of the token.
  SourceLocation Loc;

  // Conceptually these next two fields could be in a union.  However, this
  // causes gcc 4.2 to pessimize LexTokenInternal, a very performance critical
  // routine. Keeping as separate members with casts until a more beautiful fix
  // presents itself.

  /// UintData - This holds either the length of the token text, when
  /// a normal token, or the end of the SourceRange when an annotation
  /// token.
  unsigned UintData;

  /// PtrData - This is a union of four different pointer types, which depends
  /// on what type of token this is:
  ///  Identifiers, keywords, etc:
  ///    This is an IdentifierInfo*, which contains the uniqued identifier
  ///    spelling.
  ///  Literals:  isLiteral() returns true.
  ///    This is a pointer to the start of the token in a text buffer, which
  ///    may be dirty (have trigraphs / escaped newlines).
  ///  Annotations (resolved type names, C++ scopes, etc): isAnnotation().
  ///    This is a pointer to sema-specific data for the annotation token.
  ///  Other:
  ///    This is null.
  void *PtrData;

  /// Kind - The actual flavor of token this is.
  ///
  unsigned short Kind; // DON'T make Kind a 'tok::TokenKind';
                       // MSVC will treat it as a signed char and
                       // TokenKinds > 127 won't be handled correctly.

  /// Flags - Bits we track about this token, members of the TokenFlags enum.
  unsigned char Flags;
public:

  // Various flags set per token:
  enum TokenFlags {
    StartOfLine   = 0x01,  // At start of line or only after whitespace.
    LeadingSpace  = 0x02,  // Whitespace exists before this token.
    DisableExpand = 0x04,  // This identifier may never be macro expanded.
    NeedsCleaning = 0x08,   // Contained an escaped newline or trigraph.
    LeadingEmptyMacro = 0x10 // Empty macro exists before this token.
  };

  tok::TokenKind getKind() const { return (tok::TokenKind)Kind; }
  void setKind(tok::TokenKind K) { Kind = K; }

  /// is/isNot - Predicates to check if this token is a specific kind, as in
  /// "if (Tok.is(tok::l_brace)) {...}".
  bool is(tok::TokenKind K) const { return Kind == (unsigned) K; }
  bool isNot(tok::TokenKind K) const { return Kind != (unsigned) K; }

  /// isAnyIdentifier - Return true if this is a raw identifier (when lexing
  /// in raw mode) or a non-keyword identifier (when lexing in non-raw mode).
  bool isAnyIdentifier() const {
    return is(tok::identifier) || is(tok::raw_identifier);
  }

  /// isLiteral - Return true if this is a "literal", like a numeric
  /// constant, string, etc.
  bool isLiteral() const {
    return is(tok::numeric_constant) || is(tok::char_constant) ||
           is(tok::string_literal) || is(tok::wide_string_literal) ||
           is(tok::angle_string_literal);
  }

  bool isAnnotation() const {
#define ANNOTATION(NAME) \
    if (is(tok::annot_##NAME)) \
      return true;
#include "clang/Basic/TokenKinds.def"
    return false;
  }

  /// getLocation - Return a source location identifier for the specified
  /// offset in the current file.
  SourceLocation getLocation() const { return Loc; }
  unsigned getLength() const {
    assert(!isAnnotation() && "Annotation tokens have no length field");
    return UintData;
  }

  void setLocation(SourceLocation L) { Loc = L; }
  void setLength(unsigned Len) {
    assert(!isAnnotation() && "Annotation tokens have no length field");
    UintData = Len;
  }

  SourceLocation getAnnotationEndLoc() const {
    assert(isAnnotation() && "Used AnnotEndLocID on non-annotation token");
    return SourceLocation::getFromRawEncoding(UintData);
  }
  void setAnnotationEndLoc(SourceLocation L) {
    assert(isAnnotation() && "Used AnnotEndLocID on non-annotation token");
    UintData = L.getRawEncoding();
  }

  SourceLocation getLastLoc() const {
    return isAnnotation() ? getAnnotationEndLoc() : getLocation();
  }

  /// getAnnotationRange - SourceRange of the group of tokens that this
  /// annotation token represents.
  SourceRange getAnnotationRange() const {
    return SourceRange(getLocation(), getAnnotationEndLoc());
  }
  void setAnnotationRange(SourceRange R) {
    setLocation(R.getBegin());
    setAnnotationEndLoc(R.getEnd());
  }

  const char *getName() const {
    return tok::getTokenName( (tok::TokenKind) Kind);
  }

  /// startToken - Reset all flags to cleared.
  ///
  void startToken() {
    Kind = tok::unknown;
    Flags = 0;
    PtrData = 0;
    UintData = 0;
    Loc = SourceLocation();
  }

  IdentifierInfo *getIdentifierInfo() const {
    assert(isNot(tok::raw_identifier) &&
           "getIdentifierInfo() on a tok::raw_identifier token!");
    assert(!isAnnotation() &&
           "getIdentifierInfo() on an annotation token!");
    if (isLiteral()) return 0;
    return (IdentifierInfo*) PtrData;
  }
  void setIdentifierInfo(IdentifierInfo *II) {
    PtrData = (void*) II;
  }

  /// getRawIdentifierData - For a raw identifier token (i.e., an identifier
  /// lexed in raw mode), returns a pointer to the start of it in the text
  /// buffer if known, null otherwise.
  const char *getRawIdentifierData() const {
    assert(is(tok::raw_identifier));
    return reinterpret_cast<const char*>(PtrData);
  }
  void setRawIdentifierData(const char *Ptr) {
    assert(is(tok::raw_identifier));
    PtrData = const_cast<char*>(Ptr);
  }

  /// getLiteralData - For a literal token (numeric constant, string, etc), this
  /// returns a pointer to the start of it in the text buffer if known, null
  /// otherwise.
  const char *getLiteralData() const {
    assert(isLiteral() && "Cannot get literal data of non-literal");
    return reinterpret_cast<const char*>(PtrData);
  }
  void setLiteralData(const char *Ptr) {
    assert(isLiteral() && "Cannot set literal data of non-literal");
    PtrData = const_cast<char*>(Ptr);
  }

  void *getAnnotationValue() const {
    assert(isAnnotation() && "Used AnnotVal on non-annotation token");
    return PtrData;
  }
  void setAnnotationValue(void *val) {
    assert(isAnnotation() && "Used AnnotVal on non-annotation token");
    PtrData = val;
  }

  /// setFlag - Set the specified flag.
  void setFlag(TokenFlags Flag) {
    Flags |= Flag;
  }

  /// clearFlag - Unset the specified flag.
  void clearFlag(TokenFlags Flag) {
    Flags &= ~Flag;
  }

  /// getFlags - Return the internal represtation of the flags.
  ///  Only intended for low-level operations such as writing tokens to
  //   disk.
  unsigned getFlags() const {
    return Flags;
  }

  /// setFlagValue - Set a flag to either true or false.
  void setFlagValue(TokenFlags Flag, bool Val) {
    if (Val)
      setFlag(Flag);
    else
      clearFlag(Flag);
  }

  /// isAtStartOfLine - Return true if this token is at the start of a line.
  ///
  bool isAtStartOfLine() const { return (Flags & StartOfLine) ? true : false; }

  /// hasLeadingSpace - Return true if this token has whitespace before it.
  ///
  bool hasLeadingSpace() const { return (Flags & LeadingSpace) ? true : false; }

  /// isExpandDisabled - Return true if this identifier token should never
  /// be expanded in the future, due to C99 6.10.3.4p2.
  bool isExpandDisabled() const {
    return (Flags & DisableExpand) ? true : false;
  }

  /// isObjCAtKeyword - Return true if we have an ObjC keyword identifier.
  bool isObjCAtKeyword(tok::ObjCKeywordKind objcKey) const;

  /// getObjCKeywordID - Return the ObjC keyword kind.
  tok::ObjCKeywordKind getObjCKeywordID() const;

  /// needsCleaning - Return true if this token has trigraphs or escaped
  /// newlines in it.
  ///
  bool needsCleaning() const { return (Flags & NeedsCleaning) ? true : false; }

  /// \brief Return true if this token has an empty macro before it.
  ///
  bool hasLeadingEmptyMacro() const {
    return (Flags & LeadingEmptyMacro) ? true : false;
  }

};

/// PPConditionalInfo - Information about the conditional stack (#if directives)
/// currently active.
struct PPConditionalInfo {
  /// IfLoc - Location where the conditional started.
  ///
  SourceLocation IfLoc;

  /// WasSkipping - True if this was contained in a skipping directive, e.g.
  /// in a "#if 0" block.
  bool WasSkipping;

  /// FoundNonSkip - True if we have emitted tokens already, and now we're in
  /// an #else block or something.  Only useful in Skipping blocks.
  bool FoundNonSkip;

  /// FoundElse - True if we've seen a #else in this block.  If so,
  /// #elif/#else directives are not allowed.
  bool FoundElse;
};

}  // end namespace clang

namespace llvm {
  template <>
  struct isPodLike<clang::Token> { static const bool value = true; };
}  // end namespace llvm

#endif
