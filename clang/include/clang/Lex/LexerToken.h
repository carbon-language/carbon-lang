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

namespace llvm {
namespace clang {

class IdentifierInfo;

/// LexerToken - This structure provides full information about a lexed token.
/// it is not intended to be space efficient, it is intended to return as much
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
    NeedsCleaning = 0x04   // Contained an escaped newline or trigraph.
    //#define STRINGIFY_ARG   (1 << 2) /* If macro argument to be stringified.
    //#define PASTE_LEFT      (1 << 3) /* If on LHS of a ## operator.
  };

  tok::TokenKind getKind() const { return Kind; }
  void SetKind(tok::TokenKind K) { Kind = K; }

  /// getLocation - Return a source location identifier for the specified
  /// offset in the current file.
  SourceLocation getLocation() const { return Loc; }
  unsigned getLength() const { return Length; }

  void SetLocation(SourceLocation L) { Loc = L; }
  void SetLength(unsigned Len) { Length = Len; }
  
  /// StartToken - Reset all flags to cleared.
  ///
  void StartToken() {
    Flags = 0;
    IdentInfo = 0;
    Loc = SourceLocation();
  }
  
  IdentifierInfo *getIdentifierInfo() const { return IdentInfo; }
  void SetIdentifierInfo(IdentifierInfo *II) {
    IdentInfo = II;
  }

  /// SetFlag - Set the specified flag.
  void SetFlag(TokenFlags Flag) {
    Flags |= Flag;
  }
  
  /// ClearFlag - Unset the specified flag.
  void ClearFlag(TokenFlags Flag) {
    Flags &= ~Flag;
  }

  /// SetFlagValue - Set a flag to either true or false.
  void SetFlagValue(TokenFlags Flag, bool Val) {
    if (Val) 
      SetFlag(Flag);
    else
      ClearFlag(Flag);
  }
  
  /// isAtStartOfLine - Return true if this token is at the start of a line.
  ///
  bool isAtStartOfLine() const { return Flags & StartOfLine; }
  
  /// hasLeadingSpace - Return true if this token has whitespace before it.
  ///
  bool hasLeadingSpace() const { return Flags & LeadingSpace; }
  
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
}  // end namespace llvm

#endif