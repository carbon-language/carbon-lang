//===--- PreprocessorLexer.cpp - C Language Family Lexer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the PreprocessorLexer and Token interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/PreprocessorLexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;

PreprocessorLexer::~PreprocessorLexer() {}

void PreprocessorLexer::Diag(SourceLocation Loc, unsigned DiagID,
                             const std::string &Msg) const {
  if (LexingRawMode && Diagnostic::isBuiltinNoteWarningOrExtension(DiagID))
    return;
  PP->Diag(Loc, DiagID, Msg);
}

/// LexIncludeFilename - After the preprocessor has parsed a #include, lex and
/// (potentially) macro expand the filename.
void PreprocessorLexer::LexIncludeFilename(Token &FilenameTok) {
  assert(ParsingPreprocessorDirective &&
         ParsingFilename == false &&
         "Must be in a preprocessing directive!");

  // We are now parsing a filename!
  ParsingFilename = true;
  
  // Lex the filename.
  IndirectLex(FilenameTok);

  // We should have obtained the filename now.
  ParsingFilename = false;
  
  // No filename?
  if (FilenameTok.is(tok::eom))
    Diag(FilenameTok.getLocation(), diag::err_pp_expects_filename);
}
