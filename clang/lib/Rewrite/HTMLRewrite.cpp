//== HTMLRewrite.cpp - Translate source code into prettified HTML --*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the HTMLRewriter clas, which is used to translate the
//  text of a source file into prettified HTML.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/Rewriter.h"
#include "clang/Rewrite/HTMLRewrite.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include <sstream>

using namespace clang;

void html::EscapeText(Rewriter& R, unsigned FileID, bool EscapeSpaces) {
  
  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* C = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();
  
  assert (C <= FileEnd);
  
  for (unsigned FilePos = 0; C != FileEnd ; ++C, ++FilePos) {
    
    SourceLocation Loc = SourceLocation::getFileLoc(FileID, FilePos);
  
    switch (*C) {
      default: break;
        
      case ' ':
        if (EscapeSpaces) R.ReplaceText(Loc, 1, "&#32;", 5);
        break;

      case '<': R.ReplaceText(Loc, 1, "&lt;", 4); break;
      case '>': R.ReplaceText(Loc, 1, "&gt;", 4); break;
      case '&': R.ReplaceText(Loc, 1, "&amp;", 5); break;
    }
  }
}

static void AddLineNumber(Rewriter& R, unsigned LineNo,
                          SourceLocation B, SourceLocation E) {
    
  // Surround the line with a span tag.
  
  R.InsertTextBefore(E, "</span>", 7);
  R.InsertTextBefore(B, "<span style=lines>", 18);
  
  // Insert a span tag for the line number.

  std::ostringstream os;
  os << "<span style=nums>" << LineNo << "</span>";
  R.InsertTextBefore(B, os.str().c_str(), os.str().size());
}

void html::AddLineNumbers(Rewriter& R, unsigned FileID) {

  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* FileBeg = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();
  const char* C = FileBeg;
  
  assert (C <= FileEnd);
  
  unsigned LineNo = 0;
  unsigned FilePos = 0;
  
  while (C != FileEnd) {    
    
    ++LineNo;
    unsigned LineStartPos = FilePos;
    unsigned LineEndPos = FileEnd - FileBeg;
    
    assert (FilePos <= LineEndPos);
    assert (C < FileEnd);
    
    // Scan until the newline (or end-of-file).
    
    for ( ; C != FileEnd ; ++C, ++FilePos)
      if (*C == '\n') {
        LineEndPos = FilePos;
        break;
      }
    
    AddLineNumber(R, LineNo,
                  SourceLocation::getFileLoc(FileID, LineStartPos),
                  SourceLocation::getFileLoc(FileID, LineEndPos));
    
    if (C != FileEnd) {
      ++C;
      ++FilePos;
    }      
  }    
}
