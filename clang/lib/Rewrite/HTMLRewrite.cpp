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

#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/Rewrite/HTMLRewrite.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <sstream>
using namespace clang;


/// HighlightRange - Highlight a range in the source code with the specified
/// start/end tags.  B/E must be in the same file.  This ensures that
/// start/end tags are placed at the start/end of each line if the range is
/// multiline.
void html::HighlightRange(Rewriter &R, SourceLocation B, SourceLocation E,
                          const char *StartTag, const char *EndTag) {
  SourceManager &SM = R.getSourceMgr();
  B = SM.getLogicalLoc(B);
  E = SM.getLogicalLoc(E);
  unsigned FileID = SM.getCanonicalFileID(B);
  assert(SM.getCanonicalFileID(E) == FileID && "B/E not in the same file!");

  unsigned BOffset = SM.getFullFilePos(B);
  unsigned EOffset = SM.getFullFilePos(E);
  
  // Include the whole end token in the range.
  EOffset += Lexer::MeasureTokenLength(E, R.getSourceMgr());
  
  HighlightRange(R.getEditBuffer(FileID), BOffset, EOffset,
                 SM.getBufferData(FileID).first, StartTag, EndTag);
}

/// HighlightRange - This is the same as the above method, but takes
/// decomposed file locations.
void html::HighlightRange(RewriteBuffer &RB, unsigned B, unsigned E,
                          const char *BufferStart,
                          const char *StartTag, const char *EndTag) {
  // Insert the tag at the absolute start/end of the range.
  RB.InsertTextAfter(B, StartTag, strlen(StartTag));
  RB.InsertTextBefore(E, EndTag, strlen(EndTag));
  
  // Scan the range to see if there is a \r or \n.  If so, and if the line is
  // not blank, insert tags on that line as well.
  bool HadOpenTag = true;
  
  unsigned LastNonWhiteSpace = B;
  for (unsigned i = B; i != E; ++i) {
    switch (BufferStart[i]) {
    case '\r':
    case '\n':
      // Okay, we found a newline in the range.  If we have an open tag, we need
      // to insert a close tag at the first non-whitespace before the newline.
      if (HadOpenTag)
        RB.InsertTextBefore(LastNonWhiteSpace+1, EndTag, strlen(EndTag));
        
      // Instead of inserting an open tag immediately after the newline, we
      // wait until we see a non-whitespace character.  This prevents us from
      // inserting tags around blank lines, and also allows the open tag to
      // be put *after* whitespace on a non-blank line.
      HadOpenTag = false;
      break;
    case '\0':
    case ' ':
    case '\t':
    case '\f':
    case '\v':
      // Ignore whitespace.
      break;
    
    default:
      // If there is no tag open, do it now.
      if (!HadOpenTag) {
        RB.InsertTextAfter(i, StartTag, strlen(StartTag));
        HadOpenTag = true;
      }
        
      // Remember this character.
      LastNonWhiteSpace = i;
      break;
    }
  }
}

void html::EscapeText(Rewriter& R, unsigned FileID,
                      bool EscapeSpaces, bool ReplaceTabs) {
  
  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* C = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();
  
  assert (C <= FileEnd);
  
  RewriteBuffer &RB = R.getEditBuffer(FileID);

  unsigned ColNo = 0;
  for (unsigned FilePos = 0; C != FileEnd ; ++C, ++FilePos) {
    switch (*C) {
    default: ++ColNo; break;
    case '\n':
    case '\r':
      ColNo = 0;
      break;
      
    case ' ':
      if (EscapeSpaces)
        RB.ReplaceText(FilePos, 1, "&nbsp;", 6);
      ++ColNo;
      break;
    case '\f':
      RB.ReplaceText(FilePos, 1, "<hr>", 4);
      ColNo = 0;
      break;
        
    case '\t': {
      if (!ReplaceTabs)
        break;
      unsigned NumSpaces = 8-(ColNo&7);
      if (EscapeSpaces)
        RB.ReplaceText(FilePos, 1, "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                       "&nbsp;&nbsp;&nbsp;", 6*NumSpaces);
      else
        RB.ReplaceText(FilePos, 1, "        ", NumSpaces);
      ColNo += NumSpaces;
      break;
    }
    case '<':
      RB.ReplaceText(FilePos, 1, "&lt;", 4);
      ++ColNo;
      break;
      
    case '>':
      RB.ReplaceText(FilePos, 1, "&gt;", 4);
      ++ColNo;
      break;
      
    case '&':
      RB.ReplaceText(FilePos, 1, "&amp;", 5);
      ++ColNo;
      break;
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
        if (ReplaceTabs) {
          if (EscapeSpaces)
            for (unsigned i = 0; i < 4; ++i)
              os << "&nbsp;";
          else
            for (unsigned i = 0; i < 4; ++i)
              os << " ";
        }
        else 
          os << c;
      
        break;
      
      case '<': os << "&lt;"; break;
      case '>': os << "&gt;"; break;
      case '&': os << "&amp;"; break;
    }
  }
  
  return os.str();
}

static void AddLineNumber(RewriteBuffer &RB, unsigned LineNo,
                          unsigned B, unsigned E) {
  llvm::SmallString<100> Str;
  Str += "<tr><td class=\"num\" id=\"LN";
  Str.append_uint(LineNo);
  Str += "\">";
  Str.append_uint(LineNo);
  Str += "</td><td class=\"line\">";
  
  if (B == E) { // Handle empty lines.
    Str += " </td></tr>";
    RB.InsertTextBefore(B, &Str[0], Str.size());
  } else {
    RB.InsertTextBefore(B, &Str[0], Str.size());
    RB.InsertTextBefore(E, "</td></tr>", strlen("</td></tr>"));
  }
}

void html::AddLineNumbers(Rewriter& R, unsigned FileID) {

  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* FileBeg = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();
  const char* C = FileBeg;
  RewriteBuffer &RB = R.getEditBuffer(FileID);
  
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
    
    AddLineNumber(RB, LineNo, LineStartPos, LineEndPos);
  }
  
  // Add one big table tag that surrounds all of the code.
  RB.InsertTextBefore(0, "<table class=\"code\">\n",
                      strlen("<table class=\"code\">\n"));
  
  RB.InsertTextAfter(FileEnd - FileBeg, "</table>", strlen("</table>"));
}

void html::AddHeaderFooterInternalBuiltinCSS(Rewriter& R, unsigned FileID, 
                                             const char *title) {

  const llvm::MemoryBuffer *Buf = R.getSourceMgr().getBuffer(FileID);
  const char* FileStart = Buf->getBufferStart();
  const char* FileEnd = Buf->getBufferEnd();

  SourceLocation StartLoc = SourceLocation::getFileLoc(FileID, 0);
  SourceLocation EndLoc = SourceLocation::getFileLoc(FileID, FileEnd-FileStart);

  std::ostringstream os;
  os << "<!doctype html>\n" // Use HTML 5 doctype
        "<html>\n<head>\n";
  
  if (title)
    os << "<title>" << html::EscapeText(title) << "</title>\n";
  
  os << "<style type=\"text/css\">\n"
      " body { color:#000000; background-color:#ffffff }\n"
      " body { font-family:Helvetica, sans-serif; font-size:10pt }\n"
      " h1 { font-size:14pt }\n"
      " .code { border-collapse:collapse; width:100%; }\n"
      " .code { font-family: \"Andale Mono\", monospace; font-size:10pt }\n"
      " .code { line-height: 1.2em }\n"
      " .comment { color: green; font-style: oblique }\n"
      " .keyword { color: blue }\n"
      " .directive { color: darkmagenta }\n"
      // Macro expansions.
      " .expansion { display: none; }\n"
      " .macro:hover .expansion { display: block; border: 2px solid #FF0000; "
          "padding: 2px; background-color:#FFF0F0; font-weight: normal; "
          "  -webkit-border-radius:5px;  -webkit-box-shadow:1px 1px 7px #000; "
          "position: absolute; top: -1em; left:10em; z-index: 1 } \n"
      " .macro { color: darkmagenta; background-color:LemonChiffon;"
             // Macros are position: relative to provide base for expansions.
             " position: relative }\n"
      " .num { width:2.5em; padding-right:2ex; background-color:#eeeeee }\n"
      " .num { text-align:right; font-size: smaller }\n"
      " .num { color:#444444 }\n"
      " .line { padding-left: 1ex; border-left: 3px solid #ccc }\n"
      " .line { white-space: pre }\n"
      " .msg { background-color:#fff8b4; color:#000000 }\n"
      " .msg { -webkit-box-shadow:1px 1px 7px #000 }\n"
      " .msg { -webkit-border-radius:5px }\n"
      " .msg { font-family:Helvetica, sans-serif; font-size: smaller }\n"
      " .msg { font-weight: bold }\n"
      " .msg { float:left }\n"
      " .msg { padding:0.5em 1ex 0.5em 1ex }\n"
      " .msg { margin-top:10px; margin-bottom:10px }\n"
      " .msg { max-width:60em; word-wrap: break-word; white-space: pre-wrap;}\n"
      " .mrange { background-color:#dfddf3 }\n"
      " .mrange { border-bottom:1px solid #6F9DBE }\n"
      " .PathIndex { font-weight: bold }\n"
      " table.simpletable {\n"
      "   padding: 5px;\n"
      "   font-size:12pt;\n"
      "   margin:20px;\n"
      "   border-collapse: collapse; border-spacing: 0px;\n"
      " }\n"
      " td.rowname {\n"
      "   text-align:right; font-weight:bold; color:#444444;\n"
      "   padding-right:2ex; }\n"
      "</style>\n</head>\n<body>";

  // Generate header
  R.InsertStrBefore(StartLoc, os.str());
  // Generate footer
  
  R.InsertCStrAfter(EndLoc, "</body></html>\n");
}

/// SyntaxHighlight - Relex the specified FileID and annotate the HTML with
/// information about keywords, macro expansions etc.  This uses the macro
/// table state from the end of the file, so it won't be perfectly perfect,
/// but it will be reasonably close.
void html::SyntaxHighlight(Rewriter &R, unsigned FileID, Preprocessor &PP) {
  RewriteBuffer &RB = R.getEditBuffer(FileID);

  const SourceManager &SourceMgr = PP.getSourceManager();
  std::pair<const char*, const char*> File = SourceMgr.getBufferData(FileID);
  const char *BufferStart = File.first;
  
  Lexer L(SourceLocation::getFileLoc(FileID, 0), PP.getLangOptions(),
          File.first, File.second);
  
  // Inform the preprocessor that we want to retain comments as tokens, so we 
  // can highlight them.
  L.SetCommentRetentionState(true);
 
  // Lex all the tokens in raw mode, to avoid entering #includes or expanding
  // macros.
  Token Tok;
  L.LexRawToken(Tok);
  
  while (Tok.isNot(tok::eof)) {
    // Since we are lexing unexpanded tokens, all tokens are from the main
    // FileID.
    unsigned TokOffs = SourceMgr.getFullFilePos(Tok.getLocation());
    unsigned TokLen = Tok.getLength();
    switch (Tok.getKind()) {
    default: break;
    case tok::identifier: {
      // Fill in Result.IdentifierInfo, looking up the identifier in the
      // identifier table.
      IdentifierInfo *II = PP.LookUpIdentifierInfo(Tok, BufferStart+TokOffs);
        
      // If this is a pp-identifier, for a keyword, highlight it as such.
      if (II->getTokenID() != tok::identifier)
        HighlightRange(RB, TokOffs, TokOffs+TokLen, BufferStart,
                       "<span class='keyword'>", "</span>");
      break;
    }
    case tok::comment:
      HighlightRange(RB, TokOffs, TokOffs+TokLen, BufferStart,
                     "<span class='comment'>", "</span>");
      break;
    case tok::hash: {
      // If this is a preprocessor directive, all tokens to end of line are too.
      if (!Tok.isAtStartOfLine())
        break;
        
      // Eat all of the tokens until we get to the next one at the start of
      // line.
      unsigned TokEnd = TokOffs+TokLen;
      L.LexRawToken(Tok);
      while (!Tok.isAtStartOfLine() && Tok.isNot(tok::eof)) {
        TokEnd = SourceMgr.getFullFilePos(Tok.getLocation())+Tok.getLength();
        L.LexRawToken(Tok);
      }
      
      // Find end of line.  This is a hack.
      HighlightRange(RB, TokOffs, TokEnd, BufferStart,
                     "<span class='directive'>", "</span>");
      
      // Don't skip the next token.
      continue;
    }
    }
    
    L.LexRawToken(Tok);
  }
}

/// HighlightMacros - This uses the macro table state from the end of the
/// file, to reexpand macros and insert (into the HTML) information about the
/// macro expansions.  This won't be perfectly perfect, but it will be
/// reasonably close.
void html::HighlightMacros(Rewriter &R, unsigned FileID, Preprocessor& PP) {
  
  RewriteBuffer &RB = R.getEditBuffer(FileID);
  
  // Inform the preprocessor that we don't want comments.
  PP.SetCommentRetentionState(false, false);
  
  // Start parsing the specified input file.
  PP.EnterMainSourceFile();
  
  // Lex all the tokens.
  const SourceManager &SourceMgr = PP.getSourceManager();
  Token Tok;
  PP.Lex(Tok);
  while (Tok.isNot(tok::eof)) {
    // Ignore non-macro tokens.
    if (!Tok.getLocation().isMacroID()) {
      PP.Lex(Tok);
      continue;
    }
    
    // Ignore tokens whose logical location was not the main file.
    SourceLocation LLoc = SourceMgr.getLogicalLoc(Tok.getLocation());
    std::pair<unsigned, unsigned> LLocInfo = 
      SourceMgr.getDecomposedFileLoc(LLoc);
    
    if (LLocInfo.first != FileID) {
      PP.Lex(Tok);
      continue;
    }
    
    // Okay, we have the first token of a macro expansion: highlight the
    // instantiation.
  
    // Get the size of current macro call itself.
    // FIXME: This should highlight the args of a function-like
    // macro, using a heuristic.
    unsigned TokLen = Lexer::MeasureTokenLength(LLoc, SourceMgr);
    
    unsigned TokOffs = LLocInfo.second;
    // Highlight the macro invocation itself.
    RB.InsertTextAfter(TokOffs, "<span class='macro'>",
                       strlen("<span class='macro'>"));
    RB.InsertTextBefore(TokOffs+TokLen, "</span>", strlen("</span>"));
    
    std::string Expansion = PP.getSpelling(Tok);
    unsigned LineLen = Expansion.size();
    
    // Okay, eat this token, getting the next one.
    PP.Lex(Tok);
    
    // Skip all the rest of the tokens that are part of this macro
    // instantiation.  It would be really nice to pop up a window with all the
    // spelling of the tokens or something.
    while (!Tok.is(tok::eof) &&
           SourceMgr.getLogicalLoc(Tok.getLocation()) == LLoc) {
      // Insert a newline if the macro expansion is getting large.
      if (LineLen > 60) {
        Expansion += "<br>";
        LineLen = 0;
      }
      
      LineLen -= Expansion.size();
      // Escape any special characters in the token text.
      Expansion += ' ' + EscapeText(PP.getSpelling(Tok));
      LineLen += Expansion.size();
      PP.Lex(Tok);
    }
    
    // Insert the information about the expansion inside the macro span.
    Expansion = "<span class='expansion'>" + Expansion + "</span>";
    RB.InsertTextBefore(TokOffs+TokLen, Expansion.c_str(), Expansion.size());
  }
}

void html::HighlightMacros(Rewriter &R, unsigned FileID,
                           PreprocessorFactory &PPF) {
  
  llvm::OwningPtr<Preprocessor> PP(PPF.CreatePreprocessor());
  HighlightMacros(R, FileID, *PP);
}
