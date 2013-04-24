//===-- clang-format/ClangFormat.cpp - Clang format tool ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a clang-format tool that automatically formats
/// (fragments of) C++ code.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

static cl::list<unsigned>
Offsets("offset", cl::desc("Format a range starting at this file offset. Can "
                           "only be used with one input file."));
static cl::list<unsigned>
Lengths("length", cl::desc("Format a range of this length. "
                           "When it's not specified, end of file is used. "
                           "Can only be used with one input file."));
static cl::opt<std::string> Style(
    "style",
    cl::desc("Coding style, currently supports: LLVM, Google, Chromium."),
    cl::init("LLVM"));
static cl::opt<bool> Inplace("i",
                             cl::desc("Inplace edit <file>s, if specified."));

static cl::opt<bool> OutputXML(
    "output-replacements-xml", cl::desc("Output replacements as XML."));

static cl::list<std::string> FileNames(cl::Positional,
                                       cl::desc("[<file> ...]"));

namespace clang {
namespace format {

static FileID createInMemoryFile(StringRef FileName, const MemoryBuffer *Source,
                                 SourceManager &Sources, FileManager &Files) {
  const FileEntry *Entry = Files.getVirtualFile(FileName == "-" ? "<stdin>" :
                                                    FileName,
                                                Source->getBufferSize(), 0);
  Sources.overrideFileContents(Entry, Source, true);
  return Sources.createFileID(Entry, SourceLocation(), SrcMgr::C_User);
}

static FormatStyle getStyle() {
  FormatStyle TheStyle = getGoogleStyle();
  if (Style == "LLVM")
    TheStyle = getLLVMStyle();
  if (Style == "Chromium")
    TheStyle = getChromiumStyle();
  return TheStyle;
}

// Returns true on error.
static bool format(std::string FileName) {
  FileManager Files((FileSystemOptions()));
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
      new DiagnosticOptions);
  SourceManager Sources(Diagnostics, Files);
  OwningPtr<MemoryBuffer> Code;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(FileName, Code)) {
    llvm::errs() << ec.message() << "\n";
    return true;
  }
  FileID ID = createInMemoryFile(FileName, Code.get(), Sources, Files);
  Lexer Lex(ID, Sources.getBuffer(ID), Sources, getFormattingLangOpts());
  if (Offsets.empty())
    Offsets.push_back(0);
  if (Offsets.size() != Lengths.size() &&
      !(Offsets.size() == 1 && Lengths.empty())) {
    llvm::errs()
        << "error: number of -offset and -length arguments must match.\n";
    return true;
  }
  std::vector<CharSourceRange> Ranges;
  for (unsigned i = 0, e = Offsets.size(); i != e; ++i) {
    if (Offsets[i] >= Code->getBufferSize()) {
      llvm::errs() << "error: offset " << Offsets[i]
                   << " is outside the file\n";
      return true;
    }
    SourceLocation Start =
        Sources.getLocForStartOfFile(ID).getLocWithOffset(Offsets[i]);
    SourceLocation End;
    if (i < Lengths.size()) {
      if (Offsets[i] + Lengths[i] > Code->getBufferSize()) {
        llvm::errs() << "error: invalid length " << Lengths[i]
                     << ", offset + length (" << Offsets[i] + Lengths[i]
                     << ") is outside the file.\n";
        return true;
      }
      End = Start.getLocWithOffset(Lengths[i]);
    } else {
      End = Sources.getLocForEndOfFile(ID);
    }
    Ranges.push_back(CharSourceRange::getCharRange(Start, End));
  }
  tooling::Replacements Replaces = reformat(getStyle(), Lex, Sources, Ranges);
  if (OutputXML) {
    llvm::outs()
        << "<?xml version='1.0'?>\n<replacements xml:space='preserve'>\n";
    for (tooling::Replacements::const_iterator I = Replaces.begin(),
                                               E = Replaces.end();
         I != E; ++I) {
      llvm::outs() << "<replacement "
                   << "offset='" << I->getOffset() << "' "
                   << "length='" << I->getLength() << "'>"
                   << I->getReplacementText() << "</replacement>\n";
    }
    llvm::outs() << "</replacements>\n";
  } else {
    Rewriter Rewrite(Sources, LangOptions());
    tooling::applyAllReplacements(Replaces, Rewrite);
    if (Inplace) {
      if (Replaces.size() == 0)
        return false; // Nothing changed, don't touch the file.

      std::string ErrorInfo;
      llvm::raw_fd_ostream FileStream(FileName.c_str(), ErrorInfo,
                                      llvm::raw_fd_ostream::F_Binary);
      if (!ErrorInfo.empty()) {
        llvm::errs() << "Error while writing file: " << ErrorInfo << "\n";
        return true;
      }
      Rewrite.getEditBuffer(ID).write(FileStream);
      FileStream.flush();
    } else {
      Rewrite.getEditBuffer(ID).write(outs());
    }
  }
  return false;
}

}  // namespace format
}  // namespace clang

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to format C/C++/Obj-C code.\n\n"
      "If no arguments are specified, it formats the code from standard input\n"
      "and writes the result to the standard output.\n"
      "If <file>s are given, it reformats the files. If -i is specified \n"
      "together with <file>s, the files are edited in-place. Otherwise, the \n"
      "result is written to the standard output.\n");

  if (Help)
    cl::PrintHelpMessage();

  bool Error = false;
  switch (FileNames.size()) {
  case 0:
    Error = clang::format::format("-");
    break;
  case 1:
    Error = clang::format::format(FileNames[0]);
    break;
  default:
    if (!Offsets.empty() || !Lengths.empty()) {
      llvm::errs() << "error: \"-offset\" and \"-length\" can only be used for "
                      "single file.\n";
      return 1;
    }
    for (unsigned i = 0; i < FileNames.size(); ++i)
      Error |= clang::format::format(FileNames[i]);
    break;
  }
  return Error ? 1 : 0;
}
