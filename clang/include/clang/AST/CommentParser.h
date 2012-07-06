//===--- CommentParser.h - Doxygen comment parser ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Doxygen comment parser.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_COMMENT_PARSER_H
#define LLVM_CLANG_AST_COMMENT_PARSER_H

#include "clang/AST/CommentLexer.h"
#include "clang/AST/Comment.h"
#include "clang/AST/CommentSema.h"
#include "llvm/Support/Allocator.h"

namespace clang {
namespace comments {

/// Doxygen comment parser.
class Parser {
  Lexer &L;

  Sema &S;

  llvm::BumpPtrAllocator &Allocator;

  template<typename T>
  ArrayRef<T> copyArray(ArrayRef<T> Source) {
    size_t Size = Source.size();
    if (Size != 0) {
      T *Mem = new (Allocator) T[Size];
      std::copy(Source.begin(), Source.end(), Mem);
      return llvm::makeArrayRef(Mem, Size);
    } else
      return llvm::makeArrayRef(static_cast<T *>(NULL), 0);
  }

  /// Current lookahead token.  We can safely assume that all tokens are from
  /// a single source file.
  Token Tok;

  /// A stack of additional lookahead tokens.
  SmallVector<Token, 8> MoreLATokens;

  SourceLocation consumeToken() {
    SourceLocation Loc = Tok.getLocation();
    if (MoreLATokens.empty())
      L.lex(Tok);
    else {
      Tok = MoreLATokens.back();
      MoreLATokens.pop_back();
    }
    return Loc;
  }

  void putBack(const Token &OldTok) {
    MoreLATokens.push_back(Tok);
    Tok = OldTok;
  }

  void putBack(ArrayRef<Token> Toks) {
    if (Toks.empty())
      return;

    MoreLATokens.push_back(Tok);
    for (const Token *I = &Toks.back(),
         *B = &Toks.front() + 1;
         I != B; --I) {
      MoreLATokens.push_back(*I);
    }

    Tok = Toks[0];
  }

public:
  Parser(Lexer &L, Sema &S, llvm::BumpPtrAllocator &Allocator);

  /// Parse arguments for \\param command.
  ParamCommandComment *parseParamCommandArgs(
                                    ParamCommandComment *PC,
                                    TextTokenRetokenizer &Retokenizer);

  BlockCommandComment *parseBlockCommandArgs(
                                    BlockCommandComment *BC,
                                    TextTokenRetokenizer &Retokenizer,
                                    unsigned NumArgs);

  BlockCommandComment *parseBlockCommand();
  InlineCommandComment *parseInlineCommand();

  HTMLOpenTagComment *parseHTMLOpenTag();
  HTMLCloseTagComment *parseHTMLCloseTag();

  BlockContentComment *parseParagraphOrBlockCommand();

  VerbatimBlockComment *parseVerbatimBlock();
  VerbatimLineComment *parseVerbatimLine();
  BlockContentComment *parseBlockContent();
  FullComment *parseFullComment();
};

} // end namespace comments
} // end namespace clang

#endif

