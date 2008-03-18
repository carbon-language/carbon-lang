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


static void TagOpen(std::ostringstream& os, const char* TagStr,
                    const char* Attr, const char* Content) {
  
  os << '<' << TagStr;
  if (Attr) os << ' ' << Attr;
  os << '>';
  if (Content) os << Content;
}

static void TagClose(std::ostringstream& os, const char* TagStr) {
  os << "</" << TagStr << ">";
}

void html::InsertTag(Rewriter& R, html::Tags tag,
                     SourceLocation B, SourceLocation E,
                     const char* Attributes,
                     const char* Content, bool Newline,
                     bool OpenInsertBefore, bool CloseInsertAfter) {
  
  const char* TagStr = 0;
  
  switch (tag) {
    default: break;      
    case BODY: TagStr = "body"; break;
    case DIV:  TagStr = "div";  break;
    case HEAD: TagStr = "head"; break;
    case HTML: TagStr = "html"; break;
    case PRE:  TagStr = "pre";  break;
    case SPAN: TagStr = "span"; break;
  }
  
  assert (TagStr && "Tag not supported.");

  // Generate the opening tag.  We also generate the closing
  // tag of the start and end SourceLocations are the same.

  { 
    std::ostringstream os;
    TagOpen(os, TagStr, Attributes, Content);
    if (B == E)  {
      TagClose(os, TagStr);
      if (Newline) os << '\n';
    }
    
    if (OpenInsertBefore)    
      R.InsertTextBefore(B, os.str().c_str(), os.str().size());
    else
      R.InsertTextAfter(B, os.str().c_str(), os.str().size());
  }
  
  // Generate the closing tag if the start and end SourceLocations
  // are different.
  
  if (B != E) {
    std::ostringstream os;
    TagClose(os, TagStr);
    if (Newline) os << '\n';
    
    if (CloseInsertAfter)    
      R.InsertTextAfter(E, os.str().c_str(), os.str().size());
    else
      R.InsertTextBefore(E, os.str().c_str(), os.str().size());
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
  html::InsertTag(R, html::SPAN, B, E, "class=Line");  
  html::InsertTag(R, html::SPAN, B, B, "class=Num", os.str().c_str());  
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
