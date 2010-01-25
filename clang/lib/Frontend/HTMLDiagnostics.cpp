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

#include "clang/Frontend/PathDiagnosticClients.h"
#include "clang/Checker/PathDiagnostic.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/Rewrite/HTMLRewrite.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Boilerplate.
//===----------------------------------------------------------------------===//

namespace {

class HTMLDiagnostics : public PathDiagnosticClient {
  llvm::sys::Path Directory, FilePrefix;
  bool createdDir, noDir;
  const Preprocessor &PP;
  std::vector<const PathDiagnostic*> BatchedDiags;
public:
  HTMLDiagnostics(const std::string& prefix, const Preprocessor &pp);
  
  virtual ~HTMLDiagnostics() { FlushDiagnostics(NULL); }
  
  virtual void FlushDiagnostics(llvm::SmallVectorImpl<std::string> *FilesMade);

  virtual void HandlePathDiagnostic(const PathDiagnostic* D);
  
  virtual llvm::StringRef getName() const {
    return "HTMLDiagnostics";
  }

  unsigned ProcessMacroPiece(llvm::raw_ostream& os,
                             const PathDiagnosticMacroPiece& P,
                             unsigned num);

  void HandlePiece(Rewriter& R, FileID BugFileID,
                   const PathDiagnosticPiece& P, unsigned num, unsigned max);

  void HighlightRange(Rewriter& R, FileID BugFileID, SourceRange Range,
                      const char *HighlightStart = "<span class=\"mrange\">",
                      const char *HighlightEnd = "</span>");

  void ReportDiag(const PathDiagnostic& D,
                  llvm::SmallVectorImpl<std::string> *FilesMade);
};

} // end anonymous namespace

HTMLDiagnostics::HTMLDiagnostics(const std::string& prefix,
                                 const Preprocessor &pp)
  : Directory(prefix), FilePrefix(prefix), createdDir(false), noDir(false),
    PP(pp) {
  // All html files begin with "report"
  FilePrefix.appendComponent("report");
}

PathDiagnosticClient*
clang::CreateHTMLDiagnosticClient(const std::string& prefix,
                                  const Preprocessor &PP) {
  return new HTMLDiagnostics(prefix, PP);
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

  const_cast<PathDiagnostic*>(D)->flattenLocations();
  BatchedDiags.push_back(D);
}

void
HTMLDiagnostics::FlushDiagnostics(llvm::SmallVectorImpl<std::string> *FilesMade)
{
  while (!BatchedDiags.empty()) {
    const PathDiagnostic* D = BatchedDiags.back();
    BatchedDiags.pop_back();
    ReportDiag(*D, FilesMade);
    delete D;
  }
  
  BatchedDiags.clear();
}

void HTMLDiagnostics::ReportDiag(const PathDiagnostic& D,
                                 llvm::SmallVectorImpl<std::string> *FilesMade){
  // Create the HTML directory if it is missing.
  if (!createdDir) {
    createdDir = true;
    std::string ErrorMsg;
    Directory.createDirectoryOnDisk(true, &ErrorMsg);

    if (!Directory.isDirectory()) {
      llvm::errs() << "warning: could not create directory '"
                   << Directory.str() << "'\n"
                   << "reason: " << ErrorMsg << '\n';

      noDir = true;

      return;
    }
  }

  if (noDir)
    return;

  const SourceManager &SMgr = D.begin()->getLocation().getManager();
  FileID FID;

  // Verify that the entire path is from the same FileID.
  for (PathDiagnostic::const_iterator I = D.begin(), E = D.end(); I != E; ++I) {
    FullSourceLoc L = I->getLocation().asLocation().getInstantiationLoc();

    if (FID.isInvalid()) {
      FID = SMgr.getFileID(L);
    } else if (SMgr.getFileID(L) != FID)
      return; // FIXME: Emit a warning?

    // Check the source ranges.
    for (PathDiagnosticPiece::range_iterator RI=I->ranges_begin(),
                                             RE=I->ranges_end(); RI!=RE; ++RI) {

      SourceLocation L = SMgr.getInstantiationLoc(RI->getBegin());

      if (!L.isFileID() || SMgr.getFileID(L) != FID)
        return; // FIXME: Emit a warning?

      L = SMgr.getInstantiationLoc(RI->getEnd());

      if (!L.isFileID() || SMgr.getFileID(L) != FID)
        return; // FIXME: Emit a warning?
    }
  }

  if (FID.isInvalid())
    return; // FIXME: Emit a warning?

  // Create a new rewriter to generate HTML.
  Rewriter R(const_cast<SourceManager&>(SMgr), PP.getLangOptions());

  // Process the path.
  unsigned n = D.size();
  unsigned max = n;

  for (PathDiagnostic::const_reverse_iterator I=D.rbegin(), E=D.rend();
        I!=E; ++I, --n)
    HandlePiece(R, FID, *I, n, max);

  // Add line numbers, header, footer, etc.

  // unsigned FID = R.getSourceMgr().getMainFileID();
  html::EscapeText(R, FID);
  html::AddLineNumbers(R, FID);

  // If we have a preprocessor, relex the file and syntax highlight.
  // We might not have a preprocessor if we come from a deserialized AST file,
  // for example.

  html::SyntaxHighlight(R, FID, PP);
  html::HighlightMacros(R, FID, PP);

  // Get the full directory name of the analyzed file.

  const FileEntry* Entry = SMgr.getFileEntryForID(FID);

  // This is a cludge; basically we want to append either the full
  // working directory if we have no directory information.  This is
  // a work in progress.

  std::string DirName = "";

  if (!llvm::sys::Path(Entry->getName()).isAbsolute()) {
    llvm::sys::Path P = llvm::sys::Path::GetCurrentDirectory();
    DirName = P.str() + "/";
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
      << (*D.rbegin()).getLocation().asLocation().getInstantiationLineNumber()
      << ", column "
      << (*D.rbegin()).getLocation().asLocation().getInstantiationColumnNumber()
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

    R.InsertTextBefore(SMgr.getLocForStartOfFile(FID), os.str());
  }

  // Embed meta-data tags.
  {
    std::string s;
    llvm::raw_string_ostream os(s);

    const std::string& BugDesc = D.getDescription();
    if (!BugDesc.empty())
      os << "\n<!-- BUGDESC " << BugDesc << " -->\n";

    const std::string& BugType = D.getBugType();
    if (!BugType.empty())
      os << "\n<!-- BUGTYPE " << BugType << " -->\n";

    const std::string& BugCategory = D.getCategory();
    if (!BugCategory.empty())
      os << "\n<!-- BUGCATEGORY " << BugCategory << " -->\n";

    os << "\n<!-- BUGFILE " << DirName << Entry->getName() << " -->\n";

    os << "\n<!-- BUGLINE "
       << D.back()->getLocation().asLocation().getInstantiationLineNumber()
       << " -->\n";

    os << "\n<!-- BUGPATHLENGTH " << D.size() << " -->\n";

    // Mark the end of the tags.
    os << "\n<!-- BUGMETAEND -->\n";

    // Insert the text.
    R.InsertTextBefore(SMgr.getLocForStartOfFile(FID), os.str());
  }

  // Add CSS, header, and footer.

  html::AddHeaderFooterInternalBuiltinCSS(R, FID, Entry->getName());

  // Get the rewrite buffer.
  const RewriteBuffer *Buf = R.getRewriteBufferFor(FID);

  if (!Buf) {
    llvm::errs() << "warning: no diagnostics generated for main file.\n";
    return;
  }

  // Create a path for the target HTML file.
  llvm::sys::Path F(FilePrefix);
  F.makeUnique(false, NULL);

  // Rename the file with an HTML extension.
  llvm::sys::Path H(F);
  H.appendSuffix("html");
  F.renamePathOnDisk(H, NULL);

  std::string ErrorMsg;
  llvm::raw_fd_ostream os(H.c_str(), ErrorMsg);

  if (!ErrorMsg.empty()) {
    (llvm::errs() << "warning: could not create file '" << F.str() 
                  << "'\n").flush();
    return;
  }

  if (FilesMade)
    FilesMade->push_back(H.getLast());

  // Emit the HTML to disk.
  for (RewriteBuffer::iterator I = Buf->begin(), E = Buf->end(); I!=E; ++I)
      os << *I;
}

void HTMLDiagnostics::HandlePiece(Rewriter& R, FileID BugFileID,
                                  const PathDiagnosticPiece& P,
                                  unsigned num, unsigned max) {

  // For now, just draw a box above the line in question, and emit the
  // warning.
  FullSourceLoc Pos = P.getLocation().asLocation();

  if (!Pos.isValid())
    return;

  SourceManager &SM = R.getSourceMgr();
  assert(&Pos.getManager() == &SM && "SourceManagers are different!");
  std::pair<FileID, unsigned> LPosInfo = SM.getDecomposedInstantiationLoc(Pos);

  if (LPosInfo.first != BugFileID)
    return;

  const llvm::MemoryBuffer *Buf = SM.getBuffer(LPosInfo.first);
  const char* FileStart = Buf->getBufferStart();

  // Compute the column number.  Rewind from the current position to the start
  // of the line.
  unsigned ColNo = SM.getColumnNumber(LPosInfo.first, LPosInfo.second);
  const char *TokInstantiationPtr =Pos.getInstantiationLoc().getCharacterData();
  const char *LineStart = TokInstantiationPtr-ColNo;

  // Compute LineEnd.
  const char *LineEnd = TokInstantiationPtr;
  const char* FileEnd = Buf->getBufferEnd();
  while (*LineEnd != '\n' && LineEnd != FileEnd)
    ++LineEnd;

  // Compute the margin offset by counting tabs and non-tabs.
  unsigned PosNo = 0;
  for (const char* c = LineStart; c != TokInstantiationPtr; ++c)
    PosNo += *c == '\t' ? 8 : 1;

  // Create the html for the message.

  const char *Kind = 0;
  switch (P.getKind()) {
  case PathDiagnosticPiece::Event:  Kind = "Event"; break;
  case PathDiagnosticPiece::ControlFlow: Kind = "Control"; break;
    // Setting Kind to "Control" is intentional.
  case PathDiagnosticPiece::Macro: Kind = "Control"; break;
  }

  std::string sbuf;
  llvm::raw_string_ostream os(sbuf);

  os << "\n<tr><td class=\"num\"></td><td class=\"line\"><div id=\"";

  if (num == max)
    os << "EndPath";
  else
    os << "Path" << num;

  os << "\" class=\"msg";
  if (Kind)
    os << " msg" << Kind;
  os << "\" style=\"margin-left:" << PosNo << "ex";

  // Output a maximum size.
  if (!isa<PathDiagnosticMacroPiece>(P)) {
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

    if (cnt > max_token)
      max_token = cnt;

    // Determine the approximate size of the message bubble in em.
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

    if (em < max_line/2)
      os << "; max-width:" << em << "em";
  }
  else
    os << "; max-width:100em";

  os << "\">";

  if (max > 1) {
    os << "<table class=\"msgT\"><tr><td valign=\"top\">";
    os << "<div class=\"PathIndex";
    if (Kind) os << " PathIndex" << Kind;
    os << "\">" << num << "</div>";
    os << "</td><td>";
  }

  if (const PathDiagnosticMacroPiece *MP =
        dyn_cast<PathDiagnosticMacroPiece>(&P)) {

    os << "Within the expansion of the macro '";

    // Get the name of the macro by relexing it.
    {
      FullSourceLoc L = MP->getLocation().asLocation().getInstantiationLoc();
      assert(L.isFileID());
      std::pair<const char*, const char*> BufferInfo = L.getBufferData();
      const char* MacroName = L.getDecomposedLoc().second + BufferInfo.first;
      Lexer rawLexer(L, PP.getLangOptions(), BufferInfo.first,
                     MacroName, BufferInfo.second);

      Token TheTok;
      rawLexer.LexFromRawLexer(TheTok);
      for (unsigned i = 0, n = TheTok.getLength(); i < n; ++i)
        os << MacroName[i];
    }

    os << "':\n";

    if (max > 1)
      os << "</td></tr></table>";

    // Within a macro piece.  Write out each event.
    ProcessMacroPiece(os, *MP, 0);
  }
  else {
    os << html::EscapeText(P.getString());

    if (max > 1)
      os << "</td></tr></table>";
  }

  os << "</div></td></tr>";

  // Insert the new html.
  unsigned DisplayPos = LineEnd - FileStart;
  SourceLocation Loc =
    SM.getLocForStartOfFile(LPosInfo.first).getFileLocWithOffset(DisplayPos);

  R.InsertTextBefore(Loc, os.str());

  // Now highlight the ranges.
  for (const SourceRange *I = P.ranges_begin(), *E = P.ranges_end();
        I != E; ++I)
    HighlightRange(R, LPosInfo.first, *I);

#if 0
  // If there is a code insertion hint, insert that code.
  // FIXME: This code is disabled because it seems to mangle the HTML
  // output. I'm leaving it here because it's generally the right idea,
  // but needs some help from someone more familiar with the rewriter.
  for (const CodeModificationHint *Hint = P.code_modifications_begin(),
                               *HintEnd = P.code_modifications_end();
       Hint != HintEnd; ++Hint) {
    if (Hint->RemoveRange.isValid()) {
      HighlightRange(R, LPosInfo.first, Hint->RemoveRange,
                     "<span class=\"CodeRemovalHint\">", "</span>");
    }
    if (Hint->InsertionLoc.isValid()) {
      std::string EscapedCode = html::EscapeText(Hint->CodeToInsert, true);
      EscapedCode = "<span class=\"CodeInsertionHint\">" + EscapedCode
        + "</span>";
      R.InsertTextBefore(Hint->InsertionLoc, EscapedCode);
    }
  }
#endif
}

static void EmitAlphaCounter(llvm::raw_ostream& os, unsigned n) {
  llvm::SmallVector<char, 10> buf;

  do {
    unsigned x = n % ('z' - 'a');
    buf.push_back('a' + x);
    n = n / ('z' - 'a');
  } while (n);

  assert(!buf.empty());

  for (llvm::SmallVectorImpl<char>::reverse_iterator I=buf.rbegin(),
       E=buf.rend(); I!=E; ++I)
    os << *I;
}

unsigned HTMLDiagnostics::ProcessMacroPiece(llvm::raw_ostream& os,
                                            const PathDiagnosticMacroPiece& P,
                                            unsigned num) {

  for (PathDiagnosticMacroPiece::const_iterator I=P.begin(), E=P.end();
        I!=E; ++I) {

    if (const PathDiagnosticMacroPiece *MP =
          dyn_cast<PathDiagnosticMacroPiece>(*I)) {
      num = ProcessMacroPiece(os, *MP, num);
      continue;
    }

    if (PathDiagnosticEventPiece *EP = dyn_cast<PathDiagnosticEventPiece>(*I)) {
      os << "<div class=\"msg msgEvent\" style=\"width:94%; "
            "margin-left:5px\">"
            "<table class=\"msgT\"><tr>"
            "<td valign=\"top\"><div class=\"PathIndex PathIndexEvent\">";
      EmitAlphaCounter(os, num++);
      os << "</div></td><td valign=\"top\">"
         << html::EscapeText(EP->getString())
         << "</td></tr></table></div>\n";
    }
  }

  return num;
}

void HTMLDiagnostics::HighlightRange(Rewriter& R, FileID BugFileID,
                                     SourceRange Range,
                                     const char *HighlightStart,
                                     const char *HighlightEnd) {
  SourceManager &SM = R.getSourceMgr();
  const LangOptions &LangOpts = R.getLangOpts();

  SourceLocation InstantiationStart = SM.getInstantiationLoc(Range.getBegin());
  unsigned StartLineNo = SM.getInstantiationLineNumber(InstantiationStart);

  SourceLocation InstantiationEnd = SM.getInstantiationLoc(Range.getEnd());
  unsigned EndLineNo = SM.getInstantiationLineNumber(InstantiationEnd);

  if (EndLineNo < StartLineNo)
    return;

  if (SM.getFileID(InstantiationStart) != BugFileID ||
      SM.getFileID(InstantiationEnd) != BugFileID)
    return;

  // Compute the column number of the end.
  unsigned EndColNo = SM.getInstantiationColumnNumber(InstantiationEnd);
  unsigned OldEndColNo = EndColNo;

  if (EndColNo) {
    // Add in the length of the token, so that we cover multi-char tokens.
    EndColNo += Lexer::MeasureTokenLength(Range.getEnd(), SM, LangOpts)-1;
  }

  // Highlight the range.  Make the span tag the outermost tag for the
  // selected range.

  SourceLocation E =
    InstantiationEnd.getFileLocWithOffset(EndColNo - OldEndColNo);

  html::HighlightRange(R, InstantiationStart, E, HighlightStart, HighlightEnd);
}
