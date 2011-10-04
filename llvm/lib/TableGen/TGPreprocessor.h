//===- TGPreprocessor.h - Preprocessor for TableGen Files -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents the Preprocessor for tablegen files.
//
//===----------------------------------------------------------------------===//

#ifndef TGPREPROCESSOR_H
#define TGPREPROCESSOR_H

#include <vector>

namespace llvm {
class MemoryBuffer;
class SourceMgr;
class tool_output_file;

class TGPPLexer;
class TGPPRange;
class TGPPRecord;

typedef std::vector<TGPPRecord> TGPPRecords;

class TGPreprocessor {
  SourceMgr &SrcMgr;
  tool_output_file &Out;

  TGPPLexer *Lexer;
  TGPPRecords *CurRecords;

  bool ParseBlock(bool TopLevel);
  bool ParseForLoop();
  bool ParseRange(TGPPRange *Range);

public:
  TGPreprocessor(SourceMgr &SM, tool_output_file &O)
    : SrcMgr(SM), Out(O), Lexer(NULL), CurRecords(NULL) {
  }

  /// PreprocessFile - Main entrypoint for preprocess a tblgen file.  These
  /// preprocess routines return true on error, or false on success.
  bool PreprocessFile();
};
} // namespace llvm

#endif /* TGPREPROCESSOR_H */
