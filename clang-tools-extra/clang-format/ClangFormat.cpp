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
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Format/Format.h"
#include "llvm/Support/FileSystem.h"

using namespace llvm;

static cl::opt<int> Offset(
    "offset", cl::desc("Format a range starting at this file offset."),
    cl::init(0));
static cl::opt<int> Length(
    "length", cl::desc("Format a range of this length, -1 for end of file."),
    cl::init(-1));
static cl::opt<std::string> Style(
    "style", cl::desc("Coding style, currently supports: LLVM, Google."),
    cl::init("LLVM"));

namespace clang {
namespace format {

static FileID createInMemoryFile(const MemoryBuffer *Source,
                                 SourceManager &Sources,
                                 FileManager &Files) {
  const FileEntry *Entry =
      Files.getVirtualFile("<stdio>", Source->getBufferSize(), 0);
  Sources.overrideFileContents(Entry, Source, true);
  return Sources.createFileID(Entry, SourceLocation(), SrcMgr::C_User);
}

static void format() {
  FileManager Files((FileSystemOptions()));
  DiagnosticsEngine Diagnostics(IntrusiveRefCntPtr<DiagnosticIDs>(
      new DiagnosticIDs), new DiagnosticOptions);
  SourceManager Sources(Diagnostics, Files);
  OwningPtr<MemoryBuffer> Code;
  if (error_code ec = MemoryBuffer::getSTDIN(Code)) {
    llvm::errs() << ec.message() << "\n";
    return;
  }
  FileID ID = createInMemoryFile(Code.get(), Sources, Files);
  LangOptions LangOpts;
  LangOpts.CPlusPlus = 1;
  Lexer Lex(ID, Sources.getBuffer(ID), Sources, LangOpts);
  SourceLocation Start =
      Sources.getLocForStartOfFile(ID).getLocWithOffset(Offset);
  SourceLocation End = Sources.getLocForEndOfFile(ID);
  if (Length != -1)
    End = Start.getLocWithOffset(Length);
  std::vector<CharSourceRange> Ranges(
      1, CharSourceRange::getCharRange(Start, End));
  FormatStyle FStyle = Style == "LLVM" ? getLLVMStyle() : getGoogleStyle();
  tooling::Replacements Replaces = reformat(FStyle, Lex, Sources, Ranges);
  Rewriter Rewrite(Sources, LangOptions());
  tooling::applyAllReplacements(Replaces, Rewrite);
  Rewrite.getEditBuffer(ID).write(outs());
}

}  // namespace format
}  // namespace clang

int main(int argc, const char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  clang::format::format();
  return 0;
}
