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

#include "HTMLDiagnostics.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/Rewrite/HTMLRewrite.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Path.h"
#include <fstream>
#include <sstream>

using namespace clang;

//===----------------------------------------------------------------------===//
// Boilerplate.
//===----------------------------------------------------------------------===//

namespace {

class VISIBILITY_HIDDEN HTMLDiagnostics : public PathDiagnosticClient {
  llvm::sys::Path Directory, FilePrefix;
  bool createdDir, noDir;
public:
  HTMLDiagnostics(const std::string& prefix);

  virtual ~HTMLDiagnostics() {}
  
  virtual void HandlePathDiagnostic(const PathDiagnostic& D);
  
  void HandlePiece(Rewriter& R, const PathDiagnosticPiece& P,
                   unsigned num, unsigned max);
  
  void HighlightRange(Rewriter& R, SourceRange Range, unsigned MainFileID);
};
  
} // end anonymous namespace

HTMLDiagnostics::HTMLDiagnostics(const std::string& prefix)
  : Directory(prefix), FilePrefix(prefix), createdDir(false), noDir(false) {
  
  // All html files begin with "report" 
  FilePrefix.appendComponent("report");
}

PathDiagnosticClient*
clang::CreateHTMLDiagnosticClient(const std::string& prefix) {
  
  return new HTMLDiagnostics(prefix);
}

//===----------------------------------------------------------------------===//
// Report processing.
//===----------------------------------------------------------------------===//

void HTMLDiagnostics::HandlePathDiagnostic(const PathDiagnostic& D) {

  if (D.empty())
    return;
  
  // Create the HTML directory if it is missing.
  
  if (!createdDir) {
    createdDir = true;
    Directory.createDirectoryOnDisk(true, NULL);
  
    if (!Directory.isDirectory()) {
      llvm::cerr << "warning: could not create directory '"
                  << FilePrefix.toString() << "'\n";
      
      noDir = true;
      
      return;
    }
  }
  
  if (noDir)
    return;
  
  // Create a new rewriter to generate HTML.
  SourceManager& SMgr = D.begin()->getLocation().getManager();
  Rewriter R(SMgr);
  
  // Process the path.
  
  unsigned n = D.size();
  unsigned max = n;
  
  for (PathDiagnostic::const_reverse_iterator I=D.rbegin(), E=D.rend();
        I!=E; ++I, --n) {
    
    HandlePiece(R, *I, n, max);
  }
  
  // Add line numbers, header, footer, etc.
  
  unsigned FileID = R.getSourceMgr().getMainFileID();
  html::EscapeText(R, FileID);
  html::AddLineNumbers(R, FileID);
  
  // Add the name of the file.
  
  {
    std::ostringstream os;
    
    os << "<h1>";

    const FileEntry* Entry = SMgr.getFileEntryForID(FileID);
    const char* dname = Entry->getDir()->getName();
    
    if (strcmp(dname,".") == 0)
      os << html::EscapeText(llvm::sys::Path::GetCurrentDirectory().toString());
    else
      os << html::EscapeText(dname);
    
    os << "/" << html::EscapeText(Entry->getName()) << "</h1>\n";

    R.InsertStrBefore(SourceLocation::getFileLoc(FileID, 0), os.str());
  }
  
  // Add the bug description.
  
  const std::string& BugDesc = D.getDescription();
  
  if (!BugDesc.empty()) {
    std::ostringstream os;
    os << "\n<!-- BUGDESC " << BugDesc << " -->\n";
    R.InsertStrBefore(SourceLocation::getFileLoc(FileID, 0), os.str());
  }  

  // Add CSS, header, and footer.
  
  html::AddHeaderFooterInternalBuiltinCSS(R, FileID);
  
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

void HTMLDiagnostics::HandlePiece(Rewriter& R,
                                  const PathDiagnosticPiece& P,
                                  unsigned num, unsigned max) {
  
  // For now, just draw a box above the line in question, and emit the
  // warning.
  
  FullSourceLoc Pos = P.getLocation();
  
  if (!Pos.isValid())
    return;  
  
  SourceManager& SM = R.getSourceMgr();
  FullSourceLoc LPos = Pos.getLogicalLoc();
  unsigned FileID = LPos.getLocation().getFileID();
  
  assert (&LPos.getManager() == &SM && "SourceManagers are different!");
  
  unsigned MainFileID = SM.getMainFileID();
  
  if (FileID != MainFileID)
    return;
  
  // Compute the column number.  Rewind from the current position to the start
  // of the line.
  
  unsigned ColNo = LPos.getColumnNumber();
  const char *TokLogicalPtr = LPos.getCharacterData();
  const char *LineStart = TokLogicalPtr-ColNo;
  
  // Compute the margin offset by counting tabs and non-tabs.
  
  unsigned PosNo = 0;
  
  for (const char* c = LineStart; c != TokLogicalPtr; ++c)
    PosNo += *c == '\t' ? 4 : 1;
  
  // Create the html for the message.
  
  std::ostringstream os;
  
  os << "\n<tr><td class=\"num\"></td><td class=\"line\">"
     << "<div id=\"";
  
  if (num == max)
    os << "EndPath";
  else
    os << "Path" << num;
  
  os << "\" class=\"msg\" style=\"margin-left:"
     << PosNo << "ex\">";
  
  os << html::EscapeText(P.getString()) << "</div></td></tr>";
  
  // Insert the new html.
  
  const llvm::MemoryBuffer *Buf = SM.getBuffer(FileID);
  const char* FileStart = Buf->getBufferStart();
  
  R.InsertStrBefore(SourceLocation::getFileLoc(FileID, LineStart - FileStart),
                    os.str());
  
  // Now highlight the ranges.
  
  for (const SourceRange *I = P.ranges_begin(), *E = P.ranges_end();
        I != E; ++I)
    HighlightRange(R, *I, MainFileID);
}

void HTMLDiagnostics::HighlightRange(Rewriter& R, SourceRange Range,
                                     unsigned MainFileID) {
  
  SourceManager& SourceMgr = R.getSourceMgr();
  
  SourceLocation LogicalStart = SourceMgr.getLogicalLoc(Range.getBegin());
  unsigned StartLineNo = SourceMgr.getLineNumber(LogicalStart);
  
  SourceLocation LogicalEnd = SourceMgr.getLogicalLoc(Range.getEnd());
  unsigned EndLineNo = SourceMgr.getLineNumber(LogicalEnd);
  
  if (EndLineNo < StartLineNo)
    return;
  
  if (LogicalStart.getFileID() != MainFileID ||
      LogicalEnd.getFileID() != MainFileID)
    return;
    
  // Compute the column number of the end.
  unsigned EndColNo = SourceMgr.getColumnNumber(LogicalEnd);
  unsigned OldEndColNo = EndColNo;

  if (EndColNo) {
    // Add in the length of the token, so that we cover multi-char tokens.
    EndColNo += Lexer::MeasureTokenLength(Range.getEnd(), SourceMgr);
  }
  
  // Highlight the range.  Make the span tag the outermost tag for the
  // selected range.
  
  SourceLocation E =
    LogicalEnd.getFileLocWithOffset(OldEndColNo > EndColNo
                                    ? -(OldEndColNo - EndColNo)
                                    : EndColNo - OldEndColNo);
  
  R.InsertCStrBefore(LogicalStart, "<span class=\"mrange\">");
  R.InsertCStrAfter(E, "</span>");
}
