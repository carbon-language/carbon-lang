//===--- HTMLDiagnostics.cpp - HTML Diagnostics for Paths ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the HTMLDiagnostics object.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/PathDiagnosticClients.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/Rewrite/HTMLRewrite.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
#include <fstream>

using namespace clang;

//===----------------------------------------------------------------------===//
// Boilerplate.
//===----------------------------------------------------------------------===//

namespace {

class VISIBILITY_HIDDEN HTMLDiagnostics : public PathDiagnosticClient {
  llvm::sys::Path Directory, FilePrefix;
  bool createdDir, noDir;
  Preprocessor* PP;
  PreprocessorFactory* PPF;
  std::vector<const PathDiagnostic*> BatchedDiags;  
public:
  HTMLDiagnostics(const std::string& prefix, Preprocessor* pp,
                  PreprocessorFactory* ppf);

  virtual ~HTMLDiagnostics();
  
  virtual void HandlePathDiagnostic(const PathDiagnostic* D);
  
  void HandlePiece(Rewriter& R, unsigned BugFileID,
                   const PathDiagnosticPiece& P, unsigned num, unsigned max);
  
  void HighlightRange(Rewriter& R, unsigned BugFileID, SourceRange Range);

  void ReportDiag(const PathDiagnostic& D);
};
  
} // end anonymous namespace

HTMLDiagnostics::HTMLDiagnostics(const std::string& prefix, Preprocessor* pp,
                                 PreprocessorFactory* ppf)
  : Directory(prefix), FilePrefix(prefix), createdDir(false), noDir(false),
    PP(pp), PPF(ppf) {
  
  // All html files begin with "report" 
  FilePrefix.appendComponent("report");
}

PathDiagnosticClient*
clang::CreateHTMLDiagnosticClient(const std::string& prefix, Preprocessor* PP,
                                  PreprocessorFactory* PPF) {
  
  return new HTMLDiagnostics(prefix, PP, PPF);
}

//===----------------------------------------------------------------------===//
// Report processing.
//===----------------------------------------------------------------------===//

void HTMLDiagnostics::HandlePathDiagnostic(const PathDiagnostic* D) {
  if (!D)
    return;
  
  if (D->empty()) {
    delete D;
    return;
  }
  
  BatchedDiags.push_back(D);
}

HTMLDiagnostics::~HTMLDiagnostics() {
  
  while (!BatchedDiags.empty()) {
    const PathDiagnostic* D = BatchedDiags.back();
    BatchedDiags.pop_back();
    ReportDiag(*D);
    delete D;
  }  
}

void HTMLDiagnostics::ReportDiag(const PathDiagnostic& D) {
  
  // Create the HTML directory if it is missing.
  
  if (!createdDir) {
    createdDir = true;
    std::string ErrorMsg;
    Directory.createDirectoryOnDisk(true, &ErrorMsg);
  
    if (!Directory.isDirectory()) {
      llvm::cerr << "warning: could not create directory '"
                 << Directory.toString() << "'\n"
                 << "reason: " << ErrorMsg << '\n'; 
      
      noDir = true;
      
      return;
    }
  }
  
  if (noDir)
    return;
  
  SourceManager& SMgr = D.begin()->getLocation().getManager();

  unsigned FileID = 0;
  bool FileIDInitialized = false;
  
  // Verify that the entire path is from the same FileID.
  for (PathDiagnostic::const_iterator I=D.begin(), E=D.end(); I != E; ++I) {
    
    FullSourceLoc L = I->getLocation().getLogicalLoc();
    
    if (!L.isFileID())
      return; // FIXME: Emit a warning?
    
    if (!FileIDInitialized) {
      FileID = L.getCanonicalFileID();
      FileIDInitialized = true;
    }
    else if (L.getCanonicalFileID() != FileID)
      return; // FIXME: Emit a warning?
    
    // Check the source ranges.
    for (PathDiagnosticPiece::range_iterator RI=I->ranges_begin(),
                                             RE=I->ranges_end(); RI!=RE; ++RI) {
      
      SourceLocation L = SMgr.getLogicalLoc(RI->getBegin());

      if (!L.isFileID())
        return; // FIXME: Emit a warning?      
      
      if (SMgr.getCanonicalFileID(L) != FileID)
        return; // FIXME: Emit a warning?
      
      L = SMgr.getLogicalLoc(RI->getEnd());
      
      if (!L.isFileID())
        return; // FIXME: Emit a warning?      
      
      if (SMgr.getCanonicalFileID(L) != FileID)
        return; // FIXME: Emit a warning?
    }
  }
  
  if (!FileIDInitialized)
    return; // FIXME: Emit a warning?
  
  // Create a new rewriter to generate HTML.
  Rewriter R(SMgr);
  
  // Process the path.
  
  unsigned n = D.size();
  unsigned max = n;
  
  for (PathDiagnostic::const_reverse_iterator I=D.rbegin(), E=D.rend();
        I!=E; ++I, --n) {
    
    HandlePiece(R, FileID, *I, n, max);
  }
  
  // Add line numbers, header, footer, etc.
  
  // unsigned FileID = R.getSourceMgr().getMainFileID();
  html::EscapeText(R, FileID);
  html::AddLineNumbers(R, FileID);
  
  // If we have a preprocessor, relex the file and syntax highlight.
  // We might not have a preprocessor if we come from a deserialized AST file,
  // for example.
  
  if (PP) html::SyntaxHighlight(R, FileID, *PP);

  // FIXME: We eventually want to use PPF to create a fresh Preprocessor,
  //  once we have worked out the bugs.
  //
  // if (PPF) html::HighlightMacros(R, FileID, *PPF);
  //
  if (PP) html::HighlightMacros(R, FileID, *PP);
  
  // Get the full directory name of the analyzed file.

  const FileEntry* Entry = SMgr.getFileEntryForID(FileID);
  
  // This is a cludge; basically we want to append either the full
  // working directory if we have no directory information.  This is
  // a work in progress.

  std::string DirName = "";
  
  if (!llvm::sys::Path(Entry->getName()).isAbsolute()) {
    llvm::sys::Path P = llvm::sys::Path::GetCurrentDirectory();
    DirName = P.toString() + "/";
  }
    
  // Add the name of the file as an <h1> tag.  
  
  {
    std::string s;
    llvm::raw_string_ostream os(s);
    
    os << "<!-- REPORTHEADER -->\n"
       << "<h3>Bug Summary</h3>\n<table class=\"simpletable\">\n"
          "<tr><td class=\"rowname\">File:</td><td>"
       << html::EscapeText(DirName)
       << html::EscapeText(Entry->getName())
       << "</td></tr>\n<tr><td class=\"rowname\">Location:</td><td>"
          "<a href=\"#EndPath\">line "      
       << (*D.rbegin()).getLocation().getLogicalLineNumber()
       << ", column "
       << (*D.rbegin()).getLocation().getLogicalColumnNumber()
       << "</a></td></tr>\n"
          "<tr><td class=\"rowname\">Description:</td><td>"
       << D.getDescription() << "</td></tr>\n";
    
    // Output any other meta data.
    
    for (PathDiagnostic::meta_iterator I=D.meta_begin(), E=D.meta_end();
         I!=E; ++I) {
      os << "<tr><td></td><td>" << html::EscapeText(*I) << "</td></tr>\n";
    }
    
    os << "</table>\n<!-- REPORTSUMMARYEXTRA -->\n"
          "<h3>Annotated Source Code</h3>\n";    
    
    R.InsertStrBefore(SourceLocation::getFileLoc(FileID, 0), os.str());
  }
  
  // Embed meta-data tags.
  
  const std::string& BugDesc = D.getDescription();
  
  if (!BugDesc.empty()) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "\n<!-- BUGDESC " << BugDesc << " -->\n";
    R.InsertStrBefore(SourceLocation::getFileLoc(FileID, 0), os.str());
  }
  
  const std::string& BugCategory = D.getCategory();
  
  if (!BugCategory.empty()) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "\n<!-- BUGCATEGORY " << BugCategory << " -->\n";
    R.InsertStrBefore(SourceLocation::getFileLoc(FileID, 0), os.str());
  }
  
  {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "\n<!-- BUGFILE " << DirName << Entry->getName() << " -->\n";
    R.InsertStrBefore(SourceLocation::getFileLoc(FileID, 0), os.str());
  }
  
  {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "\n<!-- BUGLINE " << D.back()->getLocation().getLogicalLineNumber()
       << " -->\n";
    R.InsertStrBefore(SourceLocation::getFileLoc(FileID, 0), os.str());
  }
  
  {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "\n<!-- BUGPATHLENGTH " << D.size() << " -->\n";
    R.InsertStrBefore(SourceLocation::getFileLoc(FileID, 0), os.str());
  }

  // Add CSS, header, and footer.
  
  html::AddHeaderFooterInternalBuiltinCSS(R, FileID, Entry->getName());
  
  // Get the rewrite buffer.
  const RewriteBuffer *Buf = R.getRewriteBufferFor(FileID);
  
  if (!Buf) {
    llvm::cerr << "warning: no diagnostics generated for main file.\n";
    return;
  }

  // Create the stream to write out the HTML.
  std::ofstream os;
  
  {
    // Create a path for the target HTML file.
    llvm::sys::Path F(FilePrefix);
    F.makeUnique(false, NULL);
  
    // Rename the file with an HTML extension.
    llvm::sys::Path H(F);
    H.appendSuffix("html");
    F.renamePathOnDisk(H, NULL);
    
    os.open(H.toString().c_str());
    
    if (!os) {
      llvm::cerr << "warning: could not create file '" << F.toString() << "'\n";
      return;
    }
  }
  
  // Emit the HTML to disk.

  for (RewriteBuffer::iterator I = Buf->begin(), E = Buf->end(); I!=E; ++I)
      os << *I;
}

void HTMLDiagnostics::HandlePiece(Rewriter& R, unsigned BugFileID,
                                  const PathDiagnosticPiece& P,
                                  unsigned num, unsigned max) {
  
  // For now, just draw a box above the line in question, and emit the
  // warning.
  
  FullSourceLoc Pos = P.getLocation();
  
  if (!Pos.isValid())
    return;  
  
  SourceManager& SM = R.getSourceMgr();
  FullSourceLoc LPos = Pos.getLogicalLoc();
  unsigned FileID = SM.getCanonicalFileID(LPos.getLocation());

  assert (&LPos.getManager() == &SM && "SourceManagers are different!");
  
  if (LPos.getCanonicalFileID() != BugFileID)
    return;
  
  const llvm::MemoryBuffer *Buf = SM.getBuffer(FileID);
  const char* FileStart = Buf->getBufferStart();  
  
  // Compute the column number.  Rewind from the current position to the start
  // of the line.
  
  unsigned ColNo = LPos.getColumnNumber();
  const char *TokLogicalPtr = LPos.getCharacterData();
  const char *LineStart = TokLogicalPtr-ColNo;

  // Only compute LineEnd if we display below a line.
  const char *LineEnd = TokLogicalPtr;
  
  if (P.getDisplayHint() == PathDiagnosticPiece::Below) {
    const char* FileEnd = Buf->getBufferEnd();

    while (*LineEnd != '\n' && LineEnd != FileEnd)
      ++LineEnd;
  }
  
  // Compute the margin offset by counting tabs and non-tabs.
  
  unsigned PosNo = 0;
  
  for (const char* c = LineStart; c != TokLogicalPtr; ++c)
    PosNo += *c == '\t' ? 8 : 1;
  
  // Create the html for the message.
  {
    // Get the string and determining its maximum substring.
    const std::string& Msg = P.getString();
    unsigned max_token = 0;
    unsigned cnt = 0;
    unsigned len = Msg.size();
    
    for (std::string::const_iterator I=Msg.begin(), E=Msg.end(); I!=E; ++I)
      switch (*I) {
        default:
          ++cnt;
          continue;          
        case ' ':
        case '\t':
        case '\n':
          if (cnt > max_token) max_token = cnt;
          cnt = 0;
      }
    
    if (cnt > max_token) max_token = cnt;
    
    // Next, determine the approximate size of the message bubble in em.
    unsigned em;
    const unsigned max_line = 120;
    
    if (max_token >= max_line)
      em = max_token / 2;
    else {
      unsigned characters = max_line;
      unsigned lines = len / max_line;
      
      if (lines > 0) {
        for (; characters > max_token; --characters)
          if (len / characters > lines) {
            ++characters;
            break;
          }
      }
      
      em = characters / 2;
    }
    
    // Now generate the message bubble.    
    std::string s;
    llvm::raw_string_ostream os(s);
    
    os << "\n<tr><td class=\"num\"></td><td class=\"line\"><div id=\"";
    
    if (num == max)
      os << "EndPath";
    else
      os << "Path" << num;
    
    os << "\" class=\"msg\" style=\"margin-left:" << PosNo << "ex";
    if (em < max_line/2) os << "; max-width:" << em << "em";
    os << "\">";
    
    if (max > 1)
      os << "<span class=\"PathIndex\">[" << num << "]</span> ";
    
    os << html::EscapeText(Msg) << "</div></td></tr>";

    // Insert the new html.
    unsigned DisplayPos = 0;
    
    switch (P.getDisplayHint()) {
      case PathDiagnosticPiece::Above:
        DisplayPos = LineStart - FileStart;
        break;
      case PathDiagnosticPiece::Below:
        DisplayPos = LineEnd - FileStart;
        break;
      default:
        assert (false && "Unhandled hint.");
    }
      
    R.InsertStrBefore(SourceLocation::getFileLoc(FileID, DisplayPos), os.str());
  }
  
  // Now highlight the ranges.
  
  for (const SourceRange *I = P.ranges_begin(), *E = P.ranges_end();
        I != E; ++I)
    HighlightRange(R, FileID, *I);
}

void HTMLDiagnostics::HighlightRange(Rewriter& R, unsigned BugFileID,
                                     SourceRange Range) {
  
  SourceManager& SM = R.getSourceMgr();
  
  SourceLocation LogicalStart = SM.getLogicalLoc(Range.getBegin());
  unsigned StartLineNo = SM.getLineNumber(LogicalStart);
  
  SourceLocation LogicalEnd = SM.getLogicalLoc(Range.getEnd());
  unsigned EndLineNo = SM.getLineNumber(LogicalEnd);
  
  if (EndLineNo < StartLineNo)
    return;
  
  if (SM.getCanonicalFileID(LogicalStart) != BugFileID ||
      SM.getCanonicalFileID(LogicalEnd) != BugFileID)
    return;
    
  // Compute the column number of the end.
  unsigned EndColNo = SM.getColumnNumber(LogicalEnd);
  unsigned OldEndColNo = EndColNo;

  if (EndColNo) {
    // Add in the length of the token, so that we cover multi-char tokens.
    EndColNo += Lexer::MeasureTokenLength(Range.getEnd(), SM) - 1;
  }
  
  // Highlight the range.  Make the span tag the outermost tag for the
  // selected range.
    
  SourceLocation E = LogicalEnd.getFileLocWithOffset(EndColNo - OldEndColNo);
  
  html::HighlightRange(R, LogicalStart, E,
                       "<span class=\"mrange\">", "</span>");
}
