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
#include "clang/Basic/Version.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory ClangFormatCategory("Clang-format options");

static cl::list<unsigned>
    Offsets("offset",
            cl::desc("Format a range starting at this byte offset.\n"
                     "Multiple ranges can be formatted by specifying\n"
                     "several -offset and -length pairs.\n"
                     "Can only be used with one input file."),
            cl::cat(ClangFormatCategory));
static cl::list<unsigned>
    Lengths("length",
            cl::desc("Format a range of this length (in bytes).\n"
                     "Multiple ranges can be formatted by specifying\n"
                     "several -offset and -length pairs.\n"
                     "When only a single -offset is specified without\n"
                     "-length, clang-format will format up to the end\n"
                     "of the file.\n"
                     "Can only be used with one input file."),
            cl::cat(ClangFormatCategory));
static cl::list<std::string>
LineRanges("lines", cl::desc("<start line>:<end line> - format a range of\n"
                             "lines (both 1-based).\n"
                             "Multiple ranges can be formatted by specifying\n"
                             "several -lines arguments.\n"
                             "Can't be used with -offset and -length.\n"
                             "Can only be used with one input file."),
           cl::cat(ClangFormatCategory));
static cl::opt<std::string>
    Style("style",
          cl::desc(clang::format::StyleOptionHelpDescription),
          cl::init("file"), cl::cat(ClangFormatCategory));
static cl::opt<std::string>
FallbackStyle("fallback-style",
              cl::desc("The name of the predefined style used as a\n"
                       "fallback in case clang-format is invoked with\n"
                       "-style=file, but can not find the .clang-format\n"
                       "file to use."),
              cl::init("LLVM"), cl::cat(ClangFormatCategory));

static cl::opt<std::string>
AssumeFilename("assume-filename",
               cl::desc("When reading from stdin, clang-format assumes this\n"
                        "filename to look for a style config file (with\n"
                        "-style=file)."),
               cl::cat(ClangFormatCategory));

static cl::opt<bool> Inplace("i",
                             cl::desc("Inplace edit <file>s, if specified."),
                             cl::cat(ClangFormatCategory));

static cl::opt<bool> OutputXML("output-replacements-xml",
                               cl::desc("Output replacements as XML."),
                               cl::cat(ClangFormatCategory));
static cl::opt<bool>
    DumpConfig("dump-config",
               cl::desc("Dump configuration options to stdout and exit.\n"
                        "Can be used with -style option."),
               cl::cat(ClangFormatCategory));
static cl::opt<unsigned>
    Cursor("cursor",
           cl::desc("The position of the cursor when invoking\n"
                    "clang-format from an editor integration"),
           cl::init(0), cl::cat(ClangFormatCategory));

static cl::list<std::string> FileNames(cl::Positional, cl::desc("[<file> ...]"),
                                       cl::cat(ClangFormatCategory));

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

// Parses <start line>:<end line> input to a pair of line numbers.
// Returns true on error.
static bool parseLineRange(StringRef Input, unsigned &FromLine,
                           unsigned &ToLine) {
  std::pair<StringRef, StringRef> LineRange = Input.split(':');
  return LineRange.first.getAsInteger(0, FromLine) ||
         LineRange.second.getAsInteger(0, ToLine);
}

static bool fillRanges(SourceManager &Sources, FileID ID,
                       const MemoryBuffer *Code,
                       std::vector<CharSourceRange> &Ranges) {
  if (!LineRanges.empty()) {
    if (!Offsets.empty() || !Lengths.empty()) {
      llvm::errs() << "error: cannot use -lines with -offset/-length\n";
      return true;
    }

    for (unsigned i = 0, e = LineRanges.size(); i < e; ++i) {
      unsigned FromLine, ToLine;
      if (parseLineRange(LineRanges[i], FromLine, ToLine)) {
        llvm::errs() << "error: invalid <start line>:<end line> pair\n";
        return true;
      }
      if (FromLine > ToLine) {
        llvm::errs() << "error: start line should be less than end line\n";
        return true;
      }
      SourceLocation Start = Sources.translateLineCol(ID, FromLine, 1);
      SourceLocation End = Sources.translateLineCol(ID, ToLine, UINT_MAX);
      if (Start.isInvalid() || End.isInvalid())
        return true;
      Ranges.push_back(CharSourceRange::getCharRange(Start, End));
    }
    return false;
  }

  if (Offsets.empty())
    Offsets.push_back(0);
  if (Offsets.size() != Lengths.size() &&
      !(Offsets.size() == 1 && Lengths.empty())) {
    llvm::errs()
        << "error: number of -offset and -length arguments must match.\n";
    return true;
  }
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
  return false;
}

static void outputReplacementXML(StringRef Text) {
  size_t From = 0;
  size_t Index;
  while ((Index = Text.find_first_of("\n\r", From)) != StringRef::npos) {
    llvm::outs() << Text.substr(From, Index - From);
    switch (Text[Index]) {
    case '\n':
      llvm::outs() << "&#10;";
      break;
    case '\r':
      llvm::outs() << "&#13;";
      break;
    default:
      llvm_unreachable("Unexpected character encountered!");
    }
    From = Index + 1;
  }
  llvm::outs() << Text.substr(From);
}

// Returns true on error.
static bool format(StringRef FileName) {
  FileManager Files((FileSystemOptions()));
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
      new DiagnosticOptions);
  SourceManager Sources(Diagnostics, Files);
  std::unique_ptr<MemoryBuffer> Code;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(FileName, Code)) {
    llvm::errs() << ec.message() << "\n";
    return true;
  }
  if (Code->getBufferSize() == 0)
    return false; // Empty files are formatted correctly.
  FileID ID = createInMemoryFile(FileName, Code.get(), Sources, Files);
  std::vector<CharSourceRange> Ranges;
  if (fillRanges(Sources, ID, Code.get(), Ranges))
    return true;

  FormatStyle FormatStyle = getStyle(
      Style, (FileName == "-") ? AssumeFilename : FileName, FallbackStyle);
  Lexer Lex(ID, Sources.getBuffer(ID), Sources,
            getFormattingLangOpts(FormatStyle.Standard));
  tooling::Replacements Replaces = reformat(FormatStyle, Lex, Sources, Ranges);
  if (OutputXML) {
    llvm::outs()
        << "<?xml version='1.0'?>\n<replacements xml:space='preserve'>\n";
    for (tooling::Replacements::const_iterator I = Replaces.begin(),
                                               E = Replaces.end();
         I != E; ++I) {
      llvm::outs() << "<replacement "
                   << "offset='" << I->getOffset() << "' "
                   << "length='" << I->getLength() << "'>";
      outputReplacementXML(I->getReplacementText());
      llvm::outs() << "</replacement>\n";
    }
    llvm::outs() << "</replacements>\n";
  } else {
    Rewriter Rewrite(Sources, LangOptions());
    tooling::applyAllReplacements(Replaces, Rewrite);
    if (Inplace) {
      if (Rewrite.overwriteChangedFiles())
        return true;
    } else {
      if (Cursor.getNumOccurrences() != 0)
        outs() << "{ \"Cursor\": " << tooling::shiftedCodePosition(
                                          Replaces, Cursor) << " }\n";
      Rewrite.getEditBuffer(ID).write(outs());
    }
  }
  return false;
}

}  // namespace format
}  // namespace clang

static void PrintVersion() {
  raw_ostream &OS = outs();
  OS << clang::getClangToolFullVersion("clang-format") << '\n';
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();

  // Hide unrelated options.
  StringMap<cl::Option*> Options;
  cl::getRegisteredOptions(Options);
  for (StringMap<cl::Option *>::iterator I = Options.begin(), E = Options.end();
       I != E; ++I) {
    if (I->second->Category != &ClangFormatCategory && I->first() != "help" &&
        I->first() != "version")
      I->second->setHiddenFlag(cl::ReallyHidden);
  }

  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to format C/C++/Obj-C code.\n\n"
      "If no arguments are specified, it formats the code from standard input\n"
      "and writes the result to the standard output.\n"
      "If <file>s are given, it reformats the files. If -i is specified\n"
      "together with <file>s, the files are edited in-place. Otherwise, the\n"
      "result is written to the standard output.\n");

  if (Help)
    cl::PrintHelpMessage();

  if (DumpConfig) {
    std::string Config =
        clang::format::configurationAsText(clang::format::getStyle(
            Style, FileNames.empty() ? AssumeFilename : FileNames[0],
            FallbackStyle));
    llvm::outs() << Config << "\n";
    return 0;
  }

  bool Error = false;
  switch (FileNames.size()) {
  case 0:
    Error = clang::format::format("-");
    break;
  case 1:
    Error = clang::format::format(FileNames[0]);
    break;
  default:
    if (!Offsets.empty() || !Lengths.empty() || !LineRanges.empty()) {
      llvm::errs() << "error: -offset, -length and -lines can only be used for "
                      "single file.\n";
      return 1;
    }
    for (unsigned i = 0; i < FileNames.size(); ++i)
      Error |= clang::format::format(FileNames[i]);
    break;
  }
  return Error ? 1 : 0;
}
