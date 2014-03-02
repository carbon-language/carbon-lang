//===--- tools/clang-check/ClangCheck.cpp - Clang check tool --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a clang-check tool that runs clang based on the info
//  stored in a compilation database.
//
//  This tool uses the Clang Tooling infrastructure, see
//    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
//  for details on setting it up with LLVM source tree.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"

using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp(
    "\tFor example, to run clang-check on all files in a subtree of the\n"
    "\tsource tree, use:\n"
    "\n"
    "\t  find path/in/subtree -name '*.cpp'|xargs clang-check\n"
    "\n"
    "\tor using a specific build path:\n"
    "\n"
    "\t  find path/in/subtree -name '*.cpp'|xargs clang-check -p build/path\n"
    "\n"
    "\tNote, that path/in/subtree and current directory should follow the\n"
    "\trules described above.\n"
    "\n"
);

static cl::OptionCategory ClangCheckCategory("clang-check options");
static OwningPtr<opt::OptTable> Options(createDriverOptTable());
static cl::opt<bool>
ASTDump("ast-dump", cl::desc(Options->getOptionHelpText(options::OPT_ast_dump)),
        cl::cat(ClangCheckCategory));
static cl::opt<bool>
ASTList("ast-list", cl::desc(Options->getOptionHelpText(options::OPT_ast_list)),
        cl::cat(ClangCheckCategory));
static cl::opt<bool>
ASTPrint("ast-print",
         cl::desc(Options->getOptionHelpText(options::OPT_ast_print)),
         cl::cat(ClangCheckCategory));
static cl::opt<std::string> ASTDumpFilter(
    "ast-dump-filter",
    cl::desc(Options->getOptionHelpText(options::OPT_ast_dump_filter)),
    cl::cat(ClangCheckCategory));
static cl::opt<bool>
Analyze("analyze", cl::desc(Options->getOptionHelpText(options::OPT_analyze)),
        cl::cat(ClangCheckCategory));

static cl::opt<bool>
Fixit("fixit", cl::desc(Options->getOptionHelpText(options::OPT_fixit)),
      cl::cat(ClangCheckCategory));
static cl::opt<bool> FixWhatYouCan(
    "fix-what-you-can",
    cl::desc(Options->getOptionHelpText(options::OPT_fix_what_you_can)),
    cl::cat(ClangCheckCategory));

static cl::list<std::string> ArgsAfter(
    "extra-arg",
    cl::desc("Additional argument to append to the compiler command line"),
    cl::cat(ClangCheckCategory));
static cl::list<std::string> ArgsBefore(
    "extra-arg-before",
    cl::desc("Additional argument to prepend to the compiler command line"),
    cl::cat(ClangCheckCategory));

namespace {

// FIXME: Move FixItRewriteInPlace from lib/Rewrite/Frontend/FrontendActions.cpp
// into a header file and reuse that.
class FixItOptions : public clang::FixItOptions {
public:
  FixItOptions() {
    FixWhatYouCan = ::FixWhatYouCan;
  }

  std::string RewriteFilename(const std::string& filename, int &fd) {
    assert(llvm::sys::path::is_absolute(filename) &&
           "clang-fixit expects absolute paths only.");

    // We don't need to do permission checking here since clang will diagnose
    // any I/O errors itself.

    fd = -1;  // No file descriptor for file.

    return filename;
  }
};

/// \brief Subclasses \c clang::FixItRewriter to not count fixed errors/warnings
/// in the final error counts.
///
/// This has the side-effect that clang-check -fixit exits with code 0 on
/// successfully fixing all errors.
class FixItRewriter : public clang::FixItRewriter {
public:
  FixItRewriter(clang::DiagnosticsEngine& Diags,
                clang::SourceManager& SourceMgr,
                const clang::LangOptions& LangOpts,
                clang::FixItOptions* FixItOpts)
      : clang::FixItRewriter(Diags, SourceMgr, LangOpts, FixItOpts) {
  }

  virtual bool IncludeInDiagnosticCounts() const { return false; }
};

/// \brief Subclasses \c clang::FixItAction so that we can install the custom
/// \c FixItRewriter.
class FixItAction : public clang::FixItAction {
public:
  virtual bool BeginSourceFileAction(clang::CompilerInstance& CI,
                                     StringRef Filename) {
    FixItOpts.reset(new FixItOptions);
    Rewriter.reset(new FixItRewriter(CI.getDiagnostics(), CI.getSourceManager(),
                                     CI.getLangOpts(), FixItOpts.get()));
    return true;
  }
};

class InsertAdjuster: public clang::tooling::ArgumentsAdjuster {
public:
  enum Position { BEGIN, END };

  InsertAdjuster(const CommandLineArguments &Extra, Position Pos)
    : Extra(Extra), Pos(Pos) {
  }

  InsertAdjuster(const char *Extra, Position Pos)
    : Extra(1, std::string(Extra)), Pos(Pos) {
  }

  virtual CommandLineArguments
  Adjust(const CommandLineArguments &Args) override {
    CommandLineArguments Return(Args);

    CommandLineArguments::iterator I;
    if (Pos == END) {
      I = Return.end();
    } else {
      I = Return.begin();
      ++I; // To leave the program name in place
    }

    Return.insert(I, Extra.begin(), Extra.end());
    return Return;
  }

private:
  const CommandLineArguments Extra;
  const Position Pos;
};

} // namespace

// Anonymous namespace here causes problems with gcc <= 4.4 on MacOS 10.6.
// "Non-global symbol: ... can't be a weak_definition"
namespace clang_check {
class ClangCheckActionFactory {
public:
  clang::ASTConsumer *newASTConsumer() {
    if (ASTList)
      return clang::CreateASTDeclNodeLister();
    if (ASTDump)
      return clang::CreateASTDumper(ASTDumpFilter);
    if (ASTPrint)
      return clang::CreateASTPrinter(&llvm::outs(), ASTDumpFilter);
    return new clang::ASTConsumer();
  }
};
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  CommonOptionsParser OptionsParser(argc, argv, ClangCheckCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Clear adjusters because -fsyntax-only is inserted by the default chain.
  Tool.clearArgumentsAdjusters();
  Tool.appendArgumentsAdjuster(new ClangStripOutputAdjuster());
  if (ArgsAfter.size() > 0) {
    Tool.appendArgumentsAdjuster(new InsertAdjuster(ArgsAfter,
          InsertAdjuster::END));
  }
  if (ArgsBefore.size() > 0) {
    Tool.appendArgumentsAdjuster(new InsertAdjuster(ArgsBefore,
          InsertAdjuster::BEGIN));
  }

  // Running the analyzer requires --analyze. Other modes can work with the
  // -fsyntax-only option.
  Tool.appendArgumentsAdjuster(new InsertAdjuster(
        Analyze ? "--analyze" : "-fsyntax-only", InsertAdjuster::BEGIN));

  clang_check::ClangCheckActionFactory CheckFactory;
  FrontendActionFactory *FrontendFactory;

  // Choose the correct factory based on the selected mode.
  if (Analyze)
    FrontendFactory = newFrontendActionFactory<clang::ento::AnalysisAction>();
  else if (Fixit)
    FrontendFactory = newFrontendActionFactory<FixItAction>();
  else
    FrontendFactory = newFrontendActionFactory(&CheckFactory);

  return Tool.run(FrontendFactory);
}
