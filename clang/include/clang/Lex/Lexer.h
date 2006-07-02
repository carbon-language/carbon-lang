//===--- Lexer.h - C Language Family Lexer ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Lexer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEXER_H
#define LLVM_CLANG_LEXER_H

#include "clang/Lex/LexerToken.h"
#include <string>
#include <vector>

namespace llvm {
namespace clang {
class Diagnostic;
class Preprocessor;
class SourceBuffer;

struct LangOptions {
  unsigned Trigraphs    : 1;  // Trigraphs in source files.
  unsigned BCPLComment  : 1;  // BCPL-style // comments.
  unsigned DollarIdents : 1;  // '$' allowed in identifiers.
  unsigned Digraphs     : 1;  // When added to C?  C99?
  unsigned HexFloats    : 1;  // C99 Hexadecimal float constants.
  unsigned C99          : 1;  // C99 Support
  unsigned CPlusPlus    : 1;  // C++ Support
  unsigned CPPMinMax    : 1;  // C++ <?=, >?= tokens.
  unsigned NoExtensions : 1;  // All extensions are disabled, strict mode.
  
  unsigned ObjC1        : 1;  // Objective C 1 support enabled.
  unsigned ObjC2        : 1;  // Objective C 2 support enabled (implies ObjC1).
  
  LangOptions() {
    Trigraphs = BCPLComment = DollarIdents = Digraphs = ObjC1 = ObjC2 = 0;
    C99 = CPlusPlus = CPPMinMax = NoExtensions = 0;
  }
};



/// Lexer - This provides a simple interface that turns a text buffer into a
/// stream of tokens.  This provides no support for file reading or buffering,
/// or buffering/seeking of tokens, only forward lexing is supported.  It relies
/// on the specified Preprocessor object to handle preprocessor directives, etc.
class Lexer {
  const char *BufferPtr;         // Current pointer into the buffer.
  const char * const BufferStart;// Start of the buffer.
  const char * const BufferEnd;  // End of the buffer.
  const SourceBuffer *InputFile; // The file we are reading from.
  unsigned CurFileID;            // FileID for the current input file.
  Preprocessor &PP;              // Preprocessor object controlling lexing.
  LangOptions Features;          // Features enabled by this language (cache).
  
  // Context-specific lexing flags.
  bool IsAtStartOfLine;          // True if sitting at start of line.
  bool ParsingPreprocessorDirective; // True if parsing #XXX
  bool ParsingFilename;          // True after #include: turn <xx> into string.
  
  // Context that changes as the file is lexed.
    
  /// ConditionalStack - Information about the set of #if/#ifdef/#ifndef blocks
  /// we are currently in.
  std::vector<PPConditionalInfo> ConditionalStack;
  
  friend class Preprocessor;
public:
    
  /// Lexer constructor - Create a new lexer object for the specified buffer
  /// with the specified preprocessor managing the lexing process.  This lexer
  /// assumes that the specified SourceBuffer and Preprocessor objects will
  /// outlive it, but doesn't take ownership of either pointer.
  Lexer(const SourceBuffer *InBuffer, unsigned CurFileID, Preprocessor &PP,
        const char *BufStart = 0, const char *BufEnd = 0);
  
  /// getFeatures - Return the language features currently enabled.  NOTE: this
  /// lexer modifies features as a file is parsed!
  const LangOptions &getFeatures() const { return Features; }

  /// getCurFileID - Return the FileID for the file we are lexing out of.  This
  /// implicitly encodes the include path to get to the file.
  unsigned getCurFileID() const { return CurFileID; }
  
  /// Lex - Return the next token in the file.  If this is the end of file, it
  /// return the tok::eof token.  Return true if an error occurred and
  /// compilation should terminate, false if normal.  This implicitly involves
  /// the preprocessor.
  void Lex(LexerToken &Result) {
    // Start a new token.
    Result.StartToken();
    
    // NOTE, any changes here should also change code after calls to 
    // Preprocessor::HandleDirective
    if (IsAtStartOfLine) {
      Result.SetFlag(LexerToken::StartOfLine);
      IsAtStartOfLine = false;
    }
   
    // Get a token.
    LexTokenInternal(Result);
  }
  
  /// ReadToEndOfLine - Read the rest of the current preprocessor line as an
  /// uninterpreted string.  This switches the lexer out of directive mode.
  std::string ReadToEndOfLine();
  
 
  /// Diag - Forwarding function for diagnostics.  This translate a source
  /// position in the current buffer into a SourceLocation object for rendering.
  void Diag(const char *Loc, unsigned DiagID,
            const std::string &Msg = "") const;

  /// getSourceLocation - Return a source location identifier for the specified
  /// offset in the current file.
  SourceLocation getSourceLocation(const char *Loc) const;
  
  //===--------------------------------------------------------------------===//
  // Internal implementation interfaces.
private:

  /// LexTokenInternal - Internal interface to lex a preprocessing token. Called
  /// by Lex.
  ///
  void LexTokenInternal(LexerToken &Result);

  /// FormTokenWithChars - When we lex a token, we have identified a span
  /// starting at BufferPtr, going to TokEnd that forms the token.  This method
  /// takes that range and assigns it to the token as its location and size.  In
  /// addition, since tokens cannot overlap, this also updates BufferPtr to be
  /// TokEnd.
  void FormTokenWithChars(LexerToken &Result, const char *TokEnd) {
    Result.SetLocation(getSourceLocation(BufferPtr));
    Result.SetLength(TokEnd-BufferPtr);
    BufferPtr = TokEnd;
  }
  
  
  //===--------------------------------------------------------------------===//
  // Lexer character reading interfaces.
  
  // This lexer is built on two interfaces for reading characters, both of which
  // automatically provide phase 1/2 translation.  getAndAdvanceChar is used
  // when we know that we will be reading a character from the input buffer and
  // that this character will be part of the result token. This occurs in (f.e.)
  // string processing, because we know we need to read until we find the
  // closing '"' character.
  //
  // The second interface is the combination of PeekCharAndSize with
  // ConsumeChar.  PeekCharAndSize reads a phase 1/2 translated character,
  // returning it and its size.  If the lexer decides that this character is
  // part of the current token, it calls ConsumeChar on it.  This two stage
  // approach allows us to emit diagnostics for characters (e.g. warnings about
  // trigraphs), knowing that they only are emitted if the character is
  // consumed.
  
  
  /// getAndAdvanceChar - Read a single 'character' from the specified buffer,
  /// advance over it, and return it.  This is tricky in several cases.  Here we
  /// just handle the trivial case and fall-back to the non-inlined
  /// getCharAndSizeSlow method to handle the hard case.
  inline char getAndAdvanceChar(const char *&Ptr, LexerToken &Tok) {
    // If this is not a trigraph and not a UCN or escaped newline, return
    // quickly.
    if (Ptr[0] != '?' && Ptr[0] != '\\') return *Ptr++;
    
    unsigned Size = 0;
    char C = getCharAndSizeSlow(Ptr, Size, &Tok);
    Ptr += Size;
    return C;
  }
  
  /// ConsumeChar - When a character (identified by PeekCharAndSize) is consumed
  /// and added to a given token, check to see if there are diagnostics that
  /// need to be emitted or flags that need to be set on the token.  If so, do
  /// it.
  const char *ConsumeChar(const char *Ptr, unsigned Size, LexerToken &Tok) {
    // Normal case, we consumed exactly one token.  Just return it.
    if (Size == 1)
      return Ptr+Size;

    // Otherwise, re-lex the character with a current token, allowing
    // diagnostics to be emitted and flags to be set.
    Size = 0;
    getCharAndSizeSlow(Ptr, Size, &Tok);
    return Ptr+Size;
  }
  
  /// getCharAndSize - Peek a single 'character' from the specified buffer,
  /// get its size, and return it.  This is tricky in several cases.  Here we
  /// just handle the trivial case and fall-back to the non-inlined
  /// getCharAndSizeSlow method to handle the hard case.
  inline char getCharAndSize(const char *Ptr, unsigned &Size) {
    // If this is not a trigraph and not a UCN or escaped newline, return
    // quickly.
    if (Ptr[0] != '?' && Ptr[0] != '\\') {
      Size = 1;
      return *Ptr;
    }
    
    Size = 0;
    return getCharAndSizeSlow(Ptr, Size);
  }
  
  /// getCharAndSizeSlow - Handle the slow/uncommon case of the getCharAndSize
  /// method.
  char getCharAndSizeSlow(const char *Ptr, unsigned &Size, LexerToken *Tok = 0);
  
  /// getCharAndSizeNoWarn - Like the getCharAndSize method, but does not ever
  /// emit a warning.
  static inline char getCharAndSizeNoWarn(const char *Ptr, unsigned &Size,
                                          const LangOptions &Features) {
    // If this is not a trigraph and not a UCN or escaped newline, return
    // quickly.
    if (Ptr[0] != '?' && Ptr[0] != '\\') {
      Size = 1;
      return *Ptr;
    }
    
    Size = 0;
    return getCharAndSizeSlowNoWarn(Ptr, Size, Features);
  }
  
  /// getCharAndSizeSlowNoWarn - Same as getCharAndSizeSlow, but never emits a
  /// diagnostic.
  static char getCharAndSizeSlowNoWarn(const char *Ptr, unsigned &Size,
                                       const LangOptions &Features);
  
  //===--------------------------------------------------------------------===//
  // #if directive handling.
  
  /// pushConditionalLevel - When we enter a #if directive, this keeps track of
  /// what we are currently in for diagnostic emission (e.g. #if with missing
  /// #endif).
  void pushConditionalLevel(SourceLocation DirectiveStart, bool WasSkipping,
                            bool FoundNonSkip, bool FoundElse) {
    PPConditionalInfo CI;
    CI.IfLoc = DirectiveStart;
    CI.WasSkipping = WasSkipping;
    CI.FoundNonSkip = FoundNonSkip;
    CI.FoundElse = FoundElse;
    ConditionalStack.push_back(CI);
  }
  void pushConditionalLevel(const PPConditionalInfo &CI) {
    ConditionalStack.push_back(CI);
  }    
  
  /// popConditionalLevel - Remove an entry off the top of the conditional
  /// stack, returning information about it.  If the conditional stack is empty,
  /// this returns true and does not fill in the arguments.
  bool popConditionalLevel(PPConditionalInfo &CI) {
    if (ConditionalStack.empty()) return true;
    CI = ConditionalStack.back();
    ConditionalStack.pop_back();
    return false;
  }
  
  /// peekConditionalLevel - Return the top of the conditional stack.  This
  /// requires that there be a conditional active.
  PPConditionalInfo &peekConditionalLevel() {
    assert(!ConditionalStack.empty() && "No conditionals active!");
    return ConditionalStack.back();
  }
  
  unsigned getConditionalStackDepth() const { return ConditionalStack.size(); }
  
  //===--------------------------------------------------------------------===//
  // Other lexer functions.
  
  // Helper functions to lex the remainder of a token of the specific type.
  void LexIdentifier         (LexerToken &Result, const char *CurPtr);
  void LexNumericConstant    (LexerToken &Result, const char *CurPtr);
  void LexStringLiteral      (LexerToken &Result, const char *CurPtr);
  void LexAngledStringLiteral(LexerToken &Result, const char *CurPtr);
  void LexCharConstant       (LexerToken &Result, const char *CurPtr);
  void LexEndOfFile          (LexerToken &Result, const char *CurPtr);
  
  void SkipWhitespace        (LexerToken &Result, const char *CurPtr);
  void SkipBCPLComment       (LexerToken &Result, const char *CurPtr);
  void SkipBlockComment      (LexerToken &Result, const char *CurPtr);
  
  
  /// LexIncludeFilename - After the preprocessor has parsed a #include, lex and
  /// (potentially) macro expand the filename.  If the sequence parsed is not
  /// lexically legal, emit a diagnostic and return a result EOM token.  Return
  /// the spelled and checked filename.
  std::string LexIncludeFilename(LexerToken &Result);
};


}  // end namespace clang
}  // end namespace llvm

#endif
