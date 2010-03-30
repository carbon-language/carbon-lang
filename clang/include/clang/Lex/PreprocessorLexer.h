//===--- PreprocessorLexer.h - C Language Family Lexer ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PreprocessorLexer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PreprocessorLexer_H
#define LLVM_CLANG_PreprocessorLexer_H

#include "clang/Lex/MultipleIncludeOpt.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace clang {

class FileEntry;
class Preprocessor;

class PreprocessorLexer {
protected:
  Preprocessor *PP;              // Preprocessor object controlling lexing.

  /// The SourceManager FileID corresponding to the file being lexed.
  const FileID FID;

  //===--------------------------------------------------------------------===//
  // Context-specific lexing flags set by the preprocessor.
  //===--------------------------------------------------------------------===//

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

  /// MIOpt - This is a state machine that detects the #ifndef-wrapping a file
  /// idiom for the multiple-include optimization.
  MultipleIncludeOpt MIOpt;

  /// ConditionalStack - Information about the set of #if/#ifdef/#ifndef blocks
  /// we are currently in.
  llvm::SmallVector<PPConditionalInfo, 4> ConditionalStack;

  PreprocessorLexer(const PreprocessorLexer&);          // DO NOT IMPLEMENT
  void operator=(const PreprocessorLexer&); // DO NOT IMPLEMENT
  friend class Preprocessor;

  PreprocessorLexer(Preprocessor *pp, FileID fid)
    : PP(pp), FID(fid), ParsingPreprocessorDirective(false),
      ParsingFilename(false), LexingRawMode(false) {}

  PreprocessorLexer()
    : PP(0),
      ParsingPreprocessorDirective(false),
      ParsingFilename(false),
      LexingRawMode(false) {}

  virtual ~PreprocessorLexer() {}

  virtual void IndirectLex(Token& Result) = 0;

  /// getSourceLocation - Return the source location for the next observable
  ///  location.
  virtual SourceLocation getSourceLocation() = 0;

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

public:

  //===--------------------------------------------------------------------===//
  // Misc. lexing methods.

  /// LexIncludeFilename - After the preprocessor has parsed a #include, lex and
  /// (potentially) macro expand the filename.  If the sequence parsed is not
  /// lexically legal, emit a diagnostic and return a result EOM token.
  void LexIncludeFilename(Token &Result);

  /// setParsingPreprocessorDirective - Inform the lexer whether or not
  ///  we are currently lexing a preprocessor directive.
  void setParsingPreprocessorDirective(bool f) {
    ParsingPreprocessorDirective = f;
  }

  /// isLexingRawMode - Return true if this lexer is in raw mode or not.
  bool isLexingRawMode() const { return LexingRawMode; }

  /// getPP - Return the preprocessor object for this lexer.
  Preprocessor *getPP() const { return PP; }

  FileID getFileID() const {
    assert(PP &&
      "PreprocessorLexer::getFileID() should only be used with a Preprocessor");
    return FID;
  }

  /// getFileEntry - Return the FileEntry corresponding to this FileID.  Like
  /// getFileID(), this only works for lexers with attached preprocessors.
  const FileEntry *getFileEntry() const;
};

}  // end namespace clang

#endif
