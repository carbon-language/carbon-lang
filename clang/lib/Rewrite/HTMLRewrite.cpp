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

//===----------------------------------------------------------------------===//
// Basic operations.
//===----------------------------------------------------------------------===//

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


static void TagOpen(Rewriter& R, const char* TagStr,
                    const char* Attr, const char* Content,
                    SourceLocation L, bool InsertBefore) {
  
  std::ostringstream os;
  os << '<' << TagStr;
  if (Attr) os << ' ' << Attr;
  os << '>';
  if (Content) os << Content;
  
  if (InsertBefore)
    R.InsertTextBefore(L, os.str().c_str(), os.str().size());
  else
    R.InsertTextAfter(L, os.str().c_str(), os.str().size());
}

static void TagClose(Rewriter& R, const char* TagStr, SourceLocation L,
                     bool Newline, bool InsertBefore) {
  
  std::ostringstream os;
  os << "</" << TagStr << ">";
  if (Newline) os << '\n';
  
  if (InsertBefore)
    R.InsertTextBefore(L, os.str().c_str(), os.str().size());
  else
    R.InsertTextAfter(L, os.str().c_str(), os.str().size());
}

void html::InsertTag(Rewriter& R, html::Tags tag,
                     SourceLocation B, SourceLocation E,
                     const char* Attr, const char* Content, bool Newline,
                     bool OpenInsertBefore, bool CloseInsertBefore) {
  
  const char* TagStr = 0;
  
  switch (tag) {
    default: break;      
    case BODY: TagStr = "body"; break;
    case DIV:  TagStr = "div";  break;
    case HEAD: TagStr = "head"; break;
    case HTML: TagStr = "html"; break;
    case PRE:  TagStr = "pre";  break;
    case SPAN: TagStr = "span"; break;
    case STYLE: TagStr = "style"; break;
  }
  
  assert (TagStr && "Tag not supported.");

  // Generate the opening tag.  We also generate the closing
  // tag of the start and end SourceLocations are the same.

  if (OpenInsertBefore) {    
    TagClose(R, TagStr, E, Newline, CloseInsertBefore);
    TagOpen(R, TagStr, Attr, Content, B, true);
  }
  else {
    TagOpen(R, TagStr, Attr, Content, B, false);
    TagClose(R, TagStr, E, Newline, true);
  }
}

//===----------------------------------------------------------------------===//
// High-level operations.
//===----------------------------------------------------------------------===//

static void AddLineNumber(Rewriter& R, unsigned LineNo,
                          SourceLocation B, SourceLocation E) {
  
  // Add two "div" tags: one to contain the line number, and the other
  // to contain the content of the line.

  std::ostringstream os;
  os << LineNo;
  html::InsertTag(R, html::SPAN, B, E, "style=lines");  
  html::InsertTag(R, html::SPAN, B, B, "style=nums", os.str().c_str());  
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
