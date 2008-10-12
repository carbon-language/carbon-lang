//===--- Lexer.h - C Language Family Lexer ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Lexer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEXER_H
#define LLVM_CLANG_LEXER_H

#include "clang/Lex/Token.h"
#include "clang/Lex/MultipleIncludeOpt.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <vector>
#include <cassert>

namespace clang {
class Diagnostic;
class SourceManager;
class Preprocessor;

/// Lexer - This provides a simple interface that turns a text buffer into a
/// stream of tokens.  This provides no support for file reading or buffering,
/// or buffering/seeking of tokens, only forward lexing is supported.  It relies
/// on the specified Preprocessor object to handle preprocessor directives, etc.
class Lexer {
  //===--------------------------------------------------------------------===//
  // Constant configuration values for this lexer.
  const char *BufferStart;       // Start of the buffer.
  const char *BufferEnd;         // End of the buffer.
  SourceLocation FileLoc;        // Location for start of file.
  Preprocessor *PP;              // Preprocessor object controlling lexing.
  LangOptions Features;          // Features enabled by this language (cache).
  bool Is_PragmaLexer;           // True if lexer for _Pragma handling.
  
  //===--------------------------------------------------------------------===//
  // Context-specific lexing flags set by the preprocessor.
  //
  
  /// ParsingPreprocessorDirective - This is true when parsing #XXX.  This turns
  /// '\n' into a tok::eom token.
  bool ParsingPreprocessorDirective;
  
  /// ParsingFilename - True after #include: this turns <xx> into a
  /// tok::angle_string_literal token.
  bool ParsingFilename;
  
  /// LexingRawMode - True if in raw mode:  This flag disables interpretation of
  /// tokens and is a far faster mode to lex in than non-raw-mode.  This flag:
  ///  1. If EOF of the current lexer is found, the include stack isn't popped.
  ///  2. Identifier information is not looked up for identifier tokens.  As an
  ///     effect of this, implicit macro expansion is naturally disabled.
  ///  3. "#" tokens at the start of a line are treated as normal tokens, not
  ///     implicitly transformed by the lexer.
  ///  4. All diagnostic messages are disabled.
  ///  5. No callbacks are made into the preprocessor.
  ///
  /// Note that in raw mode that the PP pointer may be null.
  bool LexingRawMode;
  
  /// ExtendedTokenMode - The lexer can optionally keep comments and whitespace
  /// and return them as tokens.  This is used for -C and -CC modes, and
  /// whitespace preservation can be useful for some clients that want to lex
  /// the file in raw mode and get every character from the file.
  ///
  /// When this is set to 2 it returns comments and whitespace.  When set to 1
  /// it returns comments, when it is set to 0 it returns normal tokens only.
  unsigned char ExtendedTokenMode;
  
  //===--------------------------------------------------------------------===//
  // Context that changes as the file is lexed.
  // NOTE: any state that mutates when in raw mode must have save/restore code
  // in Lexer::isNextPPTokenLParen.

  // BufferPtr - Current pointer into the buffer.  This is the next character
  // to be lexed.
  const char *BufferPtr;

  // IsAtStartOfLine - True if the next lexed token should get the "start of
  // line" flag set on it.
  bool IsAtStartOfLine;

  /// MIOpt - This is a state machine that detects the #ifndef-wrapping a file 
  /// idiom for the multiple-include optimization.
  MultipleIncludeOpt MIOpt;
  
  /// ConditionalStack - Information about the set of #if/#ifdef/#ifndef blocks
  /// we are currently in.
  std::vector<PPConditionalInfo> ConditionalStack;
  
  Lexer(const Lexer&);          // DO NOT IMPLEMENT
  void operator=(const Lexer&); // DO NOT IMPLEMENT
  friend class Preprocessor;
public:
    
  /// Lexer constructor - Create a new lexer object for the specified buffer
  /// with the specified preprocessor managing the lexing process.  This lexer
  /// assumes that the associated file buffer and Preprocessor objects will
  /// outlive it, so it doesn't take ownership of either of them.
  Lexer(SourceLocation FileLoc, Preprocessor &PP,
        const char *BufStart = 0, const char *BufEnd = 0);
  
  /// Lexer constructor - Create a new raw lexer object.  This object is only
  /// suitable for calls to 'LexRawToken'.  This lexer assumes that the text
  /// range will outlive it, so it doesn't take ownership of it.
  Lexer(SourceLocation FileLoc, const LangOptions &Features,
        const char *BufStart, const char *BufEnd,
        const llvm::MemoryBuffer *FromFile = 0);
  
  /// getFeatures - Return the language features currently enabled.  NOTE: this
  /// lexer modifies features as a file is parsed!
  const LangOptions &getFeatures() const { return Features; }

  /// getFileLoc - Return the File Location for the file we are lexing out of.
  /// The physical location encodes the location where the characters come from,
  /// the virtual location encodes where we should *claim* the characters came
  /// from.  Currently this is only used by _Pragma handling.
  SourceLocation getFileLoc() const { return FileLoc; }
  
  /// Lex - Return the next token in the file.  If this is the end of file, it
  /// return the tok::eof token.  Return true if an error occurred and
  /// compilation should terminate, false if normal.  This implicitly involves
  /// the preprocessor.
  void Lex(Token &Result) {
    // Start a new token.
    Result.startToken();
    
    // NOTE, any changes here should also change code after calls to 
    // Preprocessor::HandleDirective
    if (IsAtStartOfLine) {
      Result.setFlag(Token::StartOfLine);
      IsAtStartOfLine = false;
    }
   
    // Get a token.  Note that this may delete the current lexer if the end of
    // file is reached.
    LexTokenInternal(Result);
  }
  
  /// LexFromRawLexer - Lex a token from a designated raw lexer (one with no
  /// associated preprocessor object.  Return true if the 'next character to
  /// read' pointer points and the end of the lexer buffer, false otherwise.
  bool LexFromRawLexer(Token &Result) {
    assert(LexingRawMode && "Not already in raw mode!");
    Lex(Result);
    // Note that lexing to the end of the buffer doesn't implicitly delete the
    // lexer when in raw mode.
    return BufferPtr == BufferEnd; 
  }

  /// isKeepWhitespaceMode - Return true if the lexer should return tokens for
  /// every character in the file, including whitespace and comments.  This
  /// should only be used in raw mode, as the preprocessor is not prepared to
  /// deal with the excess tokens.
  bool isKeepWhitespaceMode() const {
    return ExtendedTokenMode > 1;
  }

  /// SetKeepWhitespaceMode - This method lets clients enable or disable
  /// whitespace retention mode.
  void SetKeepWhitespaceMode(bool Val) {
    assert((!Val || LexingRawMode) &&
           "Can only enable whitespace retention in raw mode");
    ExtendedTokenMode = Val ? 2 : 0;
  }

  /// inKeepCommentMode - Return true if the lexer should return comments as
  /// tokens.
  bool inKeepCommentMode() const {
    return ExtendedTokenMode > 0;
  }
  
  /// SetCommentRetentionMode - Change the comment retention mode of the lexer
  /// to the specified mode.  This is really only useful when lexing in raw
  /// mode, because otherwise the lexer needs to manage this.
  void SetCommentRetentionState(bool Mode) { 
    assert(!isKeepWhitespaceMode() &&
           "Can't play with comment retention state when retaining whitespace");
    ExtendedTokenMode = Mode ? 1 : 0;
  }
  
  
  /// ReadToEndOfLine - Read the rest of the current preprocessor line as an
  /// uninterpreted string.  This switches the lexer out of directive mode.
  std::string ReadToEndOfLine();
  
 
  /// Diag - Forwarding function for diagnostics.  This translate a source
  /// position in the current buffer into a SourceLocation object for rendering.
  void Diag(const char *Loc, unsigned DiagID,
            const std::string &Msg = std::string()) const;
  void Diag(SourceLocation Loc, unsigned DiagID,
            const std::string &Msg = std::string()) const;

  /// getSourceLocation - Return a source location identifier for the specified
  /// offset in the current file.
  SourceLocation getSourceLocation(const char *Loc) const;
  
  /// Stringify - Convert the specified string into a C string by escaping '\'
  /// and " characters.  This does not add surrounding ""'s to the string.
  /// If Charify is true, this escapes the ' character instead of ".
  static std::string Stringify(const std::string &Str, bool Charify = false);
  
  /// Stringify - Convert the specified string into a C string by escaping '\'
  /// and " characters.  This does not add surrounding ""'s to the string.
  static void Stringify(llvm::SmallVectorImpl<char> &Str);
  
  /// MeasureTokenLength - Relex the token at the specified location and return
  /// its length in bytes in the input file.  If the token needs cleaning (e.g.
  /// includes a trigraph or an escaped newline) then this count includes bytes
  /// that are part of that.
  static unsigned MeasureTokenLength(SourceLocation Loc,
                                     const SourceManager &SM);
  
  //===--------------------------------------------------------------------===//
  // Internal implementation interfaces.
private:

  /// LexTokenInternal - Internal interface to lex a preprocessing token. Called
  /// by Lex.
  ///
  void LexTokenInternal(Token &Result);

  /// FormTokenWithChars - When we lex a token, we have identified a span
  /// starting at BufferPtr, going to TokEnd that forms the token.  This method
  /// takes that range and assigns it to the token as its location and size.  In
  /// addition, since tokens cannot overlap, this also updates BufferPtr to be
  /// TokEnd.
  void FormTokenWithChars(Token &Result, const char *TokEnd, 
                          tok::TokenKind Kind) {
    Result.setLocation(getSourceLocation(BufferPtr));
    Result.setLength(TokEnd-BufferPtr);
    Result.setKind(Kind);
    BufferPtr = TokEnd;
  }
  
  /// isNextPPTokenLParen - Return 1 if the next unexpanded token will return a
  /// tok::l_paren token, 0 if it is something else and 2 if there are no more
  /// tokens in the buffer controlled by this lexer.
  unsigned isNextPPTokenLParen();

  //===--------------------------------------------------------------------===//
  // Lexer character reading interfaces.
public:
  
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
  
  /// isObviouslySimpleCharacter - Return true if the specified character is
  /// obviously the same in translation phase 1 and translation phase 3.  This
  /// can return false for characters that end up being the same, but it will
  /// never return true for something that needs to be mapped.
  static bool isObviouslySimpleCharacter(char C) {
    return C != '?' && C != '\\';
  }
  
  /// getAndAdvanceChar - Read a single 'character' from the specified buffer,
  /// advance over it, and return it.  This is tricky in several cases.  Here we
  /// just handle the trivial case and fall-back to the non-inlined
  /// getCharAndSizeSlow method to handle the hard case.
  inline char getAndAdvanceChar(const char *&Ptr, Token &Tok) {
    // If this is not a trigraph and not a UCN or escaped newline, return
    // quickly.
    if (isObviouslySimpleCharacter(Ptr[0])) return *Ptr++;
    
    unsigned Size = 0;
    char C = getCharAndSizeSlow(Ptr, Size, &Tok);
    Ptr += Size;
    return C;
  }
  
private:
  /// ConsumeChar - When a character (identified by PeekCharAndSize) is consumed
  /// and added to a given token, check to see if there are diagnostics that
  /// need to be emitted or flags that need to be set on the token.  If so, do
  /// it.
  const char *ConsumeChar(const char *Ptr, unsigned Size, Token &Tok) {
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
    if (isObviouslySimpleCharacter(Ptr[0])) {
      Size = 1;
      return *Ptr;
    }
    
    Size = 0;
    return getCharAndSizeSlow(Ptr, Size);
  }
  
  /// getCharAndSizeSlow - Handle the slow/uncommon case of the getCharAndSize
  /// method.
  char getCharAndSizeSlow(const char *Ptr, unsigned &Size, Token *Tok = 0);
  
  /// getCharAndSizeNoWarn - Like the getCharAndSize method, but does not ever
  /// emit a warning.
  static inline char getCharAndSizeNoWarn(const char *Ptr, unsigned &Size,
                                          const LangOptions &Features) {
    // If this is not a trigraph and not a UCN or escaped newline, return
    // quickly.
    if (isObviouslySimpleCharacter(Ptr[0])) {
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
  void LexIdentifier         (Token &Result, const char *CurPtr);
  void LexNumericConstant    (Token &Result, const char *CurPtr);
  void LexStringLiteral      (Token &Result, const char *CurPtr,bool Wide);
  void LexAngledStringLiteral(Token &Result, const char *CurPtr);
  void LexCharConstant       (Token &Result, const char *CurPtr);
  bool LexEndOfFile          (Token &Result, const char *CurPtr);
  
  bool SkipWhitespace        (Token &Result, const char *CurPtr);
  bool SkipBCPLComment       (Token &Result, const char *CurPtr);
  bool SkipBlockComment      (Token &Result, const char *CurPtr);
  bool SaveBCPLComment       (Token &Result, const char *CurPtr);
  
  /// LexIncludeFilename - After the preprocessor has parsed a #include, lex and
  /// (potentially) macro expand the filename.  If the sequence parsed is not
  /// lexically legal, emit a diagnostic and return a result EOM token.
  void LexIncludeFilename(Token &Result);
};


}  // end namespace clang

#endif
