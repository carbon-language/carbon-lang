//===-- ClangChangeNamespace.cpp - Standalone change namespace ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This tool can be used to change the surrounding namespaces of class/function
// definitions.
//
// Example: test.cc
//    namespace na {
//    class X {};
//    namespace nb {
//    class Y { X x; };
//    } // namespace nb
//    } // namespace na
// To move the definition of class Y from namespace "na::nb" to "x::y", run:
//    clang-change-namespace --old_namespace "na::nb" \
//      --new_namespace "x::y" --file_pattern "test.cc" test.cc --
// Output:
//    namespace na {
//    class X {};
//    } // namespace na
//    namespace x {
//    namespace y {
//    class Y { na::X x; };
//    } // namespace y
//    } // namespace x

#include "ChangeNamespace.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/YAMLTraits.h"

using namespace clang;
using namespace llvm;

namespace {

cl::OptionCategory ChangeNamespaceCategory("Change namespace.");

cl::opt<std::string> OldNamespace("old_namespace", cl::Required,
                                  cl::desc("Old namespace."),
                                  cl::cat(ChangeNamespaceCategory));

cl::opt<std::string> NewNamespace("new_namespace", cl::Required,
                                  cl::desc("New namespace."),
                                  cl::cat(ChangeNamespaceCategory));

cl::opt<std::string> FilePattern(
    "file_pattern", cl::Required,
    cl::desc("Only rename namespaces in files that match the given pattern."),
    cl::cat(ChangeNamespaceCategory));

cl::opt<bool> Inplace("i", cl::desc("Inplace edit <file>s, if specified."),
                      cl::cat(ChangeNamespaceCategory));

cl::opt<bool>
    DumpYAML("dump_result",
         cl::desc("Dump new file contents in YAML, if specified."),
         cl::cat(ChangeNamespaceCategory));

cl::opt<std::string> Style("style",
                           cl::desc("The style name used for reformatting."),
                           cl::init("LLVM"), cl::cat(ChangeNamespaceCategory));

cl::opt<std::string> AllowedFile(
    "allowed_file",
    cl::desc("A file containing regexes of symbol names that are not expected "
             "to be updated when changing namespaces around them."),
    cl::init(""), cl::cat(ChangeNamespaceCategory));

llvm::ErrorOr<std::vector<std::string>> GetAllowedSymbolPatterns() {
  std::vector<std::string> Patterns;
  if (AllowedFile.empty())
    return Patterns;

  llvm::SmallVector<StringRef, 8> Lines;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File =
      llvm::MemoryBuffer::getFile(AllowedFile);
  if (!File)
    return File.getError();
  llvm::StringRef Content = File.get()->getBuffer();
  Content.split(Lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (auto Line : Lines)
    Patterns.push_back(std::string(Line.trim()));
  return Patterns;
}

} // anonymous namespace

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  auto ExpectedParser =
      tooling::CommonOptionsParser::create(argc, argv, ChangeNamespaceCategory);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  tooling::CommonOptionsParser &OptionsParser = ExpectedParser.get();
  const auto &Files = OptionsParser.getSourcePathList();
  tooling::RefactoringTool Tool(OptionsParser.getCompilations(), Files);
  llvm::ErrorOr<std::vector<std::string>> AllowedPatterns =
      GetAllowedSymbolPatterns();
  if (!AllowedPatterns) {
    llvm::errs() << "Failed to open allow file " << AllowedFile << ". "
                 << AllowedPatterns.getError().message() << "\n";
    return 1;
  }
  change_namespace::ChangeNamespaceTool NamespaceTool(
      OldNamespace, NewNamespace, FilePattern, *AllowedPatterns,
      &Tool.getReplacements(), Style);
  ast_matchers::MatchFinder Finder;
  NamespaceTool.registerMatchers(&Finder);
  std::unique_ptr<tooling::FrontendActionFactory> Factory =
      tooling::newFrontendActionFactory(&Finder);

  if (int Result = Tool.run(Factory.get()))
    return Result;
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  clang::TextDiagnosticPrinter DiagnosticPrinter(errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  auto &FileMgr = Tool.getFiles();
  SourceManager Sources(Diagnostics, FileMgr);
  Rewriter Rewrite(Sources, DefaultLangOptions);

  if (!formatAndApplyAllReplacements(Tool.getReplacements(), Rewrite, Style)) {
    llvm::errs() << "Failed applying all replacements.\n";
    return 1;
  }
  if (Inplace)
    return Rewrite.overwriteChangedFiles();

  std::set<llvm::StringRef> ChangedFiles;
  for (const auto &it : Tool.getReplacements())
    ChangedFiles.insert(it.first);

  if (DumpYAML) {
    auto WriteToYAML = [&](llvm::raw_ostream &OS) {
      OS << "[\n";
      for (auto I = ChangedFiles.begin(), E = ChangedFiles.end(); I != E; ++I) {
        OS << "  {\n";
        OS << "    \"FilePath\": \"" << *I << "\",\n";
        const auto Entry = FileMgr.getFile(*I);
        auto ID = Sources.getOrCreateFileID(*Entry, SrcMgr::C_User);
        std::string Content;
        llvm::raw_string_ostream ContentStream(Content);
        Rewrite.getEditBuffer(ID).write(ContentStream);
        OS << "    \"SourceText\": \""
           << llvm::yaml::escape(ContentStream.str()) << "\"\n";
        OS << "  }";
        if (I != std::prev(E))
          OS << ",\n";
      }
      OS << "\n]\n";
    };
    WriteToYAML(llvm::outs());
    return 0;
  }

  for (const auto &File : ChangedFiles) {
    const auto Entry = FileMgr.getFile(File);

    auto ID = Sources.getOrCreateFileID(*Entry, SrcMgr::C_User);
    outs() << "============== " << File << " ==============\n";
    Rewrite.getEditBuffer(ID).write(llvm::outs());
    outs() << "\n============================================\n";
  }

  return 0;
}
