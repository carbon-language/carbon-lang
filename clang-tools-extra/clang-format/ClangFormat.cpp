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

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

static cl::opt<int> Offset(
    "offset", cl::desc("Format a range starting at this file offset."),
    cl::init(0));
static cl::opt<int> Length(
    "length", cl::desc("Format a range of this length, -1 for end of file."),
    cl::init(-1));
static cl::opt<std::string> Style(
    "style",
    cl::desc("Coding style, currently supports: LLVM, Google, Chromium."),
    cl::init("LLVM"));
static cl::opt<bool> Inplace("i",
                             cl::desc("Inplace edit <file>, if specified."));

// FIXME: Remove this when styles are configurable through files.
static cl::opt<bool> InvertPointerBinding(
    "invert-pointer-binding", cl::desc("Inverts the side to which */& bind"),
    cl::init(false));

static cl::opt<std::string> FileName(cl::Positional, cl::desc("[<file>]"),
                                     cl::init("-"));

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
  if (InvertPointerBinding) {
    TheStyle.PointerAndReferenceBindToType =
        !TheStyle.PointerAndReferenceBindToType;
  }
  return TheStyle;
}

static void format() {
  FileManager Files((FileSystemOptions()));
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
      new DiagnosticOptions);
  SourceManager Sources(Diagnostics, Files);
  OwningPtr<MemoryBuffer> Code;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(FileName, Code)) {
    llvm::errs() << ec.message() << "\n";
    return;
  }
  FileID ID = createInMemoryFile(FileName, Code.get(), Sources, Files);
  Lexer Lex(ID, Sources.getBuffer(ID), Sources, getFormattingLangOpts());
  SourceLocation Start =
      Sources.getLocForStartOfFile(ID).getLocWithOffset(Offset);
  SourceLocation End = Sources.getLocForEndOfFile(ID);
  if (Length != -1)
    End = Start.getLocWithOffset(Length);
  std::vector<CharSourceRange> Ranges(
      1, CharSourceRange::getCharRange(Start, End));
  tooling::Replacements Replaces = reformat(getStyle(), Lex, Sources, Ranges);
  Rewriter Rewrite(Sources, LangOptions());
  tooling::applyAllReplacements(Replaces, Rewrite);
  if (Inplace) {
    std::string ErrorInfo;
    llvm::raw_fd_ostream FileStream(FileName.c_str(), ErrorInfo,
                                    llvm::raw_fd_ostream::F_Binary);
    if (!ErrorInfo.empty()) {
      llvm::errs() << "Error while writing file: " << ErrorInfo << "\n";
      return;
    }
    Rewrite.getEditBuffer(ID).write(FileStream);
    FileStream.flush();
  } else {
    Rewrite.getEditBuffer(ID).write(outs());
  }
}

}  // namespace format
}  // namespace clang

int main(int argc, const char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to format C/C++/Obj-C code.\n\n"
      "Currently supports LLVM and Google style guides.\n"
      "If no arguments are specified, it formats the code from standard input\n"
      "and writes the result to the standard output.\n"
      "If <file> is given, it reformats the file. If -i is specified together\n"
      "with <file>, the file is edited in-place. Otherwise, the result is\n"
      "written to the standard output.\n");
  if (Help)
    cl::PrintHelpMessage();
  clang::format::format();
  return 0;
}
