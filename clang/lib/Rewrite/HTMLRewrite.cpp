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

void html::EscapeText(Rewriter& R, unsigned FileID,
                      bool EscapeSpaces, bool ReplaceTabs) {
  
  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* C = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();
  
  assert (C <= FileEnd);
  
  for (unsigned FilePos = 0; C != FileEnd ; ++C, ++FilePos) {
      
    switch (*C) {
      default: break;
        
      case ' ':
        if (EscapeSpaces) {
          SourceLocation Loc = SourceLocation::getFileLoc(FileID, FilePos);
          R.ReplaceText(Loc, 1, "&nbsp;", 6);
        }
        break;

      case '\t': {
        if (!ReplaceTabs)
          break;
        
        SourceLocation Loc = SourceLocation::getFileLoc(FileID, FilePos);
        
        if (EscapeSpaces)
          R.ReplaceText(Loc, 1, "&nbsp;&nbsp;&nbsp;&nbsp;", 6*4);
        else
          R.ReplaceText(Loc, 1, "    ", 4);
        
        break;
      }
        
      case '<': {
        SourceLocation Loc = SourceLocation::getFileLoc(FileID, FilePos);
        R.ReplaceText(Loc, 1, "&lt;", 4);
        break;
      }
        
      case '>': {
        SourceLocation Loc = SourceLocation::getFileLoc(FileID, FilePos);
        R.ReplaceText(Loc, 1, "&gt;", 4);
        break;
      }
        
      case '&': {
        SourceLocation Loc = SourceLocation::getFileLoc(FileID, FilePos);
        R.ReplaceText(Loc, 1, "&amp;", 5);
        break;
      }
    }
  }
}

std::string html::EscapeText(const std::string& s, bool EscapeSpaces,
                             bool ReplaceTabs) {
  
  unsigned len = s.size();
  std::ostringstream os;
  
  for (unsigned i = 0 ; i < len; ++i) {
    
    char c = s[i];
    
    switch (c) {
      default:
        os << c; break;
        
      case ' ':
        if (EscapeSpaces) os << "&nbsp;";
        else os << ' ';
        break;
        
        case '\t':
          if (ReplaceTabs)
            for (unsigned i = 0; i < 4; ++i) os << "&nbsp;";
          else os << c;
        
          break;
        
        case '<': os << "&lt;"; break;
        case '>': os << "&gt;"; break;
        case '&': os << "&amp;"; break;
    }
  }
  
  return os.str();
}

static void AddLineNumber(Rewriter& R, unsigned LineNo,
                          SourceLocation B, SourceLocation E) {
    
  std::ostringstream os;
  os << "<tr><td class=\"num\" id=\"LN" << LineNo << "\">" 
     << LineNo << "</td><td class=\"line\">";

  if (B == E) { // Handle empty lines.
    os << " </td></tr>";
    R.InsertStrBefore(B, os.str());
  }
  else {
    R.InsertStrBefore(B, os.str());
    R.InsertCStrBefore(E, "</td></tr>");
  }
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
    
    while (C != FileEnd) {
      char c = *C;
      ++C;
      
      if (c == '\n') {
        LineEndPos = FilePos++;
        break;
      }
      
      ++FilePos;
    }
    
    AddLineNumber(R, LineNo,
                  SourceLocation::getFileLoc(FileID, LineStartPos),
                  SourceLocation::getFileLoc(FileID, LineEndPos));    
  }
  
  // Add one big div tag that surrounds all of the code.
  
  R.InsertCStrBefore(SourceLocation::getFileLoc(FileID, 0),
                     "<table class=\"code\">\n");
  
  R.InsertCStrAfter(SourceLocation::getFileLoc(FileID, FileEnd - FileBeg),
                    "</table>");
}

void html::AddHeaderFooterInternalBuiltinCSS(Rewriter& R, unsigned FileID) {

  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* FileStart = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();

  SourceLocation StartLoc = SourceLocation::getFileLoc(FileID, 0);
  SourceLocation EndLoc = SourceLocation::getFileLoc(FileID, FileEnd-FileStart);

  // Generate header

  {
    std::ostringstream os;
    
    os << "<html>\n<head>\n"
       << "<style type=\"text/css\">\n"
       << " body { color:#000000; background-color:#ffffff }\n"
       << " body { font-family:Helvetica, sans-serif; font-size:10pt }\n"
       << " h1 { font-size:12pt }\n"
       << " .code { border-spacing:0px; width:100%; }\n"
       << " .code { font-family: \"Andale Mono\", monospace; font-size:10pt }\n"
       << " .code { line-height: 1.2em }\n"
       << " .num { width:2.5em; padding-right:2ex; background-color:#eeeeee }\n"
       << " .num { text-align:right; font-size: smaller }\n"
       << " .num { color:#444444 }\n"
       << " .line { padding-left: 1ex; border-left: 3px solid #ccc }\n"
       << " .line { white-space: pre }\n"
       << " .msg { background-color:#fff8b4; color:#000000 }\n"
       << " .msg { -webkit-box-shadow:1px 1px 7px #000 }\n"
       << " .msg { -webkit-border-radius:5px }\n"
       << " .msg { font-family:Helvetica, sans-serif; font-size: smaller }\n"
       << " .msg { font-weight: bold }\n"
       << " .msg { float:left }\n"
       << " .msg { padding:0.5em 1ex 0.5em 1ex }\n"
       << " .msg { margin-top:10px; margin-bottom:10px }\n"
       << " .mrange { background-color:#dfddf3 }\n"
       << " .mrange { border-bottom:1px solid #6F9DBE }\n"
       << " .PathIndex { font-weight: bold }\n"
       << "</style>\n</head>\n<body>";
    
    R.InsertStrBefore(StartLoc, os.str());
  }
  
  // Generate footer
  
  {
    std::ostringstream os;
    
    os << "</body></html>\n";
    R.InsertStrAfter(EndLoc, os.str());
  }
}
  
  
