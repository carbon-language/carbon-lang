//===- TGLexer.h - Lexer for TableGen Files ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents the Lexer for tablegen files.
//
//===----------------------------------------------------------------------===//

#ifndef TGLEXER_H
#define TGLEXER_H

#include <vector>
#include <string>
#include <iosfwd>

namespace llvm {
class MemoryBuffer;

class TGLexer {
  const char *CurPtr;
  unsigned CurLineNo;
  MemoryBuffer *CurBuf;

  /// IncludeRec / IncludeStack - This captures the current set of include
  /// directives we are nested within.
  struct IncludeRec {
    MemoryBuffer *Buffer;
    const char *CurPtr;
    unsigned LineNo;
    IncludeRec(MemoryBuffer *buffer, const char *curPtr, unsigned lineNo)
      : Buffer(buffer), CurPtr(curPtr), LineNo(lineNo) {}
  };
  std::vector<IncludeRec> IncludeStack;
  
  // IncludeDirectories - This is the list of directories we should search for
  // include files in.
  std::vector<std::string> IncludeDirectories;
  
  std::string TheError;
public:
  TGLexer(MemoryBuffer *StartBuf);
  ~TGLexer();
  
  void setIncludeDirs(const std::vector<std::string> &Dirs) {
    IncludeDirectories = Dirs;
  }
  
  int LexToken();

  const std::string getError() const { return TheError; }
  
  std::ostream &err();
  void PrintIncludeStack(std::ostream &OS);
private:
  int getNextChar();
  void SkipBCPLComment();
  bool SkipCComment();
  int LexIdentifier();
  bool LexInclude();
  int LexString();
  int LexVarName();
  int LexNumber();
  int LexBracket();
  int LexExclaim();
};
  
} // end namespace llvm

#endif
