//===--- LexerToken.h - Token interface -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LexerToken interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEXERTOKEN_H
#define LLVM_CLANG_LEXERTOKEN_H

#include "clang/Basic/TokenKinds.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

class IdentifierInfo;

/// LexerToken - This structure provides full information about a lexed token.
/// It is not intended to be space efficient, it is intended to return as much
/// information as possible about each returned token.  This is expected to be
/// compressed into a smaller form if memory footprint is important.
class LexerToken {
  /// The location and length of the token text itself.
  SourceLocation Loc;
  unsigned Length;
  
  /// IdentifierInfo - If this was an identifier, this points to the uniqued
  /// information about this identifier.
  IdentifierInfo *IdentInfo;

  /// Kind - The actual flavor of token this is.
  ///
  tok::TokenKind Kind : 8;
  
  /// Flags - Bits we track about this token, members of the TokenFlags enum.
  unsigned Flags : 8;
public:
    
  // Various flags set per token:
  enum TokenFlags {
    StartOfLine   = 0x01,  // At start of line or only after whitespace.
    LeadingSpace  = 0x02,  // Whitespace exists before this token.
    DisableExpand = 0x04,  // This identifier may never be macro expanded.
    NeedsCleaning = 0x08   // Contained an escaped newline or trigraph.
  };

  tok::TokenKind getKind() const { return Kind; }
  void setKind(tok::TokenKind K) { Kind = K; }

  /// getLocation - Return a source location identifier for the specified
  /// offset in the current file.
  SourceLocation getLocation() const { return Loc; }
  unsigned getLength() const { return Length; }

  void setLocation(SourceLocation L) { Loc = L; }
  void setLength(unsigned Len) { Length = Len; }
  
  const char *getName() const { return getTokenName(Kind); }
  
  /// startToken - Reset all flags to cleared.
  ///
  void startToken() {
    Flags = 0;
    IdentInfo = 0;
    Loc = SourceLocation();
  }
  
  IdentifierInfo *getIdentifierInfo() const { return IdentInfo; }
  void setIdentifierInfo(IdentifierInfo *II) {
    IdentInfo = II;
  }

  /// setFlag - Set the specified flag.
  void setFlag(TokenFlags Flag) {
    Flags |= Flag;
  }
  
  /// clearFlag - Unset the specified flag.
  void clearFlag(TokenFlags Flag) {
    Flags &= ~Flag;
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
  bool isAtStartOfLine() const { return Flags & StartOfLine; }
  
  /// hasLeadingSpace - Return true if this token has whitespace before it.
  ///
  bool hasLeadingSpace() const { return Flags & LeadingSpace; }
  
  /// isExpandDisabled - Return true if this identifier token should never
  /// be expanded in the future, due to C99 6.10.3.4p2.
  bool isExpandDisabled() const { return Flags & DisableExpand; }
    
  /// needsCleaning - Return true if this token has trigraphs or escaped
  /// newlines in it.
  ///
  bool needsCleaning() const { return Flags & NeedsCleaning; }
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

#endif
