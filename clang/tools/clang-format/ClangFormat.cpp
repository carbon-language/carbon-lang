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
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"
#include "llvm/ADT/StringMap.h"

using namespace llvm;

// Default style to use when no style specified or specified style not found.
static const char *DefaultStyle = "LLVM";

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
cl::OptionCategory ClangFormatCategory("Clang-format options");

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
static cl::opt<std::string>
    Style("style",
          cl::desc("Coding style, currently supports:\n"
                   "  LLVM, Google, Chromium, Mozilla.\n"
                   "Use -style=file to load style configuration from\n"
                   ".clang-format file located in one of the parent\n"
                   "directories of the source file (or current\n"
                   "directory for stdin).\n"
                   "Use -style=\"{key: value, ...}\" to set specific\n"
                   "parameters, e.g.:\n"
                   "  -style=\"{BasedOnStyle: llvm, IndentWidth: 8}\""),
          cl::init(DefaultStyle), cl::cat(ClangFormatCategory));
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
           cl::desc("The position of the cursor when invoking clang-format from"
                    " an editor integration"),
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

FormatStyle getStyle(StringRef StyleName, StringRef FileName) {
  FormatStyle Style;
  getPredefinedStyle(DefaultStyle, &Style);

  if (StyleName.startswith("{")) {
    // Parse YAML/JSON style from the command line.
    if (error_code ec = parseConfiguration(StyleName, &Style)) {
      llvm::errs() << "Error parsing -style: " << ec.message()
                   << ", using " << DefaultStyle << " style\n";
    }
    return Style;
  }

  if (!StyleName.equals_lower("file")) {
    if (!getPredefinedStyle(StyleName, &Style))
      llvm::errs() << "Invalid value for -style, using " << DefaultStyle
                   << " style\n";
    return Style;
  }

  SmallString<128> Path(FileName);
  llvm::sys::fs::make_absolute(Path);
  for (StringRef Directory = llvm::sys::path::parent_path(Path);
       !Directory.empty();
       Directory = llvm::sys::path::parent_path(Directory)) {
    SmallString<128> ConfigFile(Directory);
    llvm::sys::path::append(ConfigFile, ".clang-format");
    DEBUG(llvm::dbgs() << "Trying " << ConfigFile << "...\n");
    bool IsFile = false;
    // Ignore errors from is_regular_file: we only need to know if we can read
    // the file or not.
    llvm::sys::fs::is_regular_file(Twine(ConfigFile), IsFile);
    if (IsFile) {
      OwningPtr<MemoryBuffer> Text;
      if (error_code ec = MemoryBuffer::getFile(ConfigFile, Text)) {
        llvm::errs() << ec.message() << "\n";
        continue;
      }
      if (error_code ec = parseConfiguration(Text->getBuffer(), &Style)) {
        llvm::errs() << "Error reading " << ConfigFile << ": " << ec.message()
                     << "\n";
        continue;
      }
      DEBUG(llvm::dbgs() << "Using configuration file " << ConfigFile << "\n");
      return Style;
    }
  }
  llvm::errs() << "Can't find usable .clang-format, using " << DefaultStyle
               << " style\n";
  return Style;
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
  tooling::Replacements Replaces =
      reformat(getStyle(Style, FileName), Lex, Sources, Ranges);
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

  if (DumpConfig) {
    std::string Config = clang::format::configurationAsText(
        clang::format::getStyle(Style, FileNames.empty() ? "-" : FileNames[0]));
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
