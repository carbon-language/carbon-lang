//===--- tools/extra/clang-tidy/ClangTidy.cpp - Clang tidy tool -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file This file implements a clang-tidy tool.
///
///  This tool uses the Clang Tooling infrastructure, see
///    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

#include "ClangTidy.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyModuleRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include <algorithm>
#include <vector>

using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

namespace clang {
namespace tidy {

namespace {
static const char *AnalyzerCheckerNamePrefix = "clang-analyzer-";

static StringRef StaticAnalyzerCheckers[] = {
#define GET_CHECKERS
#define CHECKER(FULLNAME, CLASS, DESCFILE, HELPTEXT, GROUPINDEX, HIDDEN)       \
  FULLNAME,
#include "../../../lib/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER
#undef GET_CHECKERS
};

} // namespace

ClangTidyASTConsumerFactory::ClangTidyASTConsumerFactory(
    StringRef EnableChecksRegex, StringRef DisableChecksRegex,
    ClangTidyContext &Context)
    : Filter(EnableChecksRegex, DisableChecksRegex), Context(Context),
      CheckFactories(new ClangTidyCheckFactories) {
  for (ClangTidyModuleRegistry::iterator I = ClangTidyModuleRegistry::begin(),
                                         E = ClangTidyModuleRegistry::end();
       I != E; ++I) {
    OwningPtr<ClangTidyModule> Module(I->instantiate());
    Module->addCheckFactories(*CheckFactories);
  }

  CheckFactories->createChecks(Filter, Checks);

  for (SmallVectorImpl<ClangTidyCheck *>::iterator I = Checks.begin(),
                                                   E = Checks.end();
       I != E; ++I) {
    (*I)->setContext(&Context);
    (*I)->registerMatchers(&Finder);
  }
}

ClangTidyASTConsumerFactory::~ClangTidyASTConsumerFactory() {
  for (SmallVectorImpl<ClangTidyCheck *>::iterator I = Checks.begin(),
                                                   E = Checks.end();
       I != E; ++I)
    delete *I;
}

clang::ASTConsumer *ClangTidyASTConsumerFactory::CreateASTConsumer(
    clang::CompilerInstance &Compiler, StringRef File) {
  // FIXME: Move this to a separate method, so that CreateASTConsumer doesn't
  // modify Compiler.
  Context.setSourceManager(&Compiler.getSourceManager());
  for (SmallVectorImpl<ClangTidyCheck *>::iterator I = Checks.begin(),
                                                   E = Checks.end();
       I != E; ++I)
    (*I)->registerPPCallbacks(Compiler);

  AnalyzerOptionsRef Options = Compiler.getAnalyzerOpts();
  Options->CheckersControlList = getCheckersControlList();
  Options->AnalysisStoreOpt = RegionStoreModel;
  Options->AnalysisDiagOpt = PD_TEXT;
  Options->AnalyzeNestedBlocks = true;
  Options->eagerlyAssumeBinOpBifurcation = true;
  ASTConsumer *Consumers[] = {
    Finder.newASTConsumer(),
    ento::CreateAnalysisConsumer(Compiler.getPreprocessor(),
                                 Compiler.getFrontendOpts().OutputFile, Options,
                                 Compiler.getFrontendOpts().Plugins)
  };
  return new MultiplexConsumer(Consumers);
}

std::vector<std::string> ClangTidyASTConsumerFactory::getCheckNames() {
  std::vector<std::string> CheckNames;
  for (ClangTidyCheckFactories::FactoryMap::const_iterator
           I = CheckFactories->begin(),
           E = CheckFactories->end();
       I != E; ++I) {
    if (Filter.IsCheckEnabled(I->first))
      CheckNames.push_back(I->first);
  }

  CheckersList AnalyzerChecks = getCheckersControlList();
  for (CheckersList::const_iterator I = AnalyzerChecks.begin(),
                                    E = AnalyzerChecks.end();
       I != E; ++I)
    CheckNames.push_back(AnalyzerCheckerNamePrefix + I->first);

  std::sort(CheckNames.begin(), CheckNames.end());
  return CheckNames;
}

ClangTidyASTConsumerFactory::CheckersList
ClangTidyASTConsumerFactory::getCheckersControlList() {
  CheckersList List;
  ArrayRef<StringRef> Checkers(StaticAnalyzerCheckers);

  bool AnalyzerChecksEnabled = false;
  for (unsigned i = 0; i < Checkers.size(); ++i) {
    std::string Checker((AnalyzerCheckerNamePrefix + Checkers[i]).str());
    AnalyzerChecksEnabled |=
        Filter.IsCheckEnabled(Checker) && !Checkers[i].startswith("debug");
  }

  if (AnalyzerChecksEnabled) {
    // Run our regex against all possible static analyzer checkers.  Note that
    // debug checkers print values / run programs to visualize the CFG and are
    // thus not applicable to clang-tidy in general.
    //
    // Always add all core checkers if any other static analyzer checks are
    // enabled. This is currently necessary, as other path sensitive checks
    // rely on the core checkers.
    for (unsigned i = 0; i < Checkers.size(); ++i) {
      std::string Checker((AnalyzerCheckerNamePrefix + Checkers[i]).str());

      if (Checkers[i].startswith("core") ||
          (!Checkers[i].startswith("debug") && Filter.IsCheckEnabled(Checker)))
        List.push_back(std::make_pair(Checkers[i], true));
    }
  }
  return List;
}

ChecksFilter::ChecksFilter(StringRef EnableChecksRegex,
                           StringRef DisableChecksRegex)
    : EnableChecks(EnableChecksRegex), DisableChecks(DisableChecksRegex) {}

bool ChecksFilter::IsCheckEnabled(StringRef Name) {
  return EnableChecks.match(Name) && !DisableChecks.match(Name);
}

ClangTidyMessage::ClangTidyMessage(StringRef Message) : Message(Message) {}

ClangTidyMessage::ClangTidyMessage(StringRef Message,
                                   const SourceManager &Sources,
                                   SourceLocation Loc)
    : Message(Message) {
  FilePath = Sources.getFilename(Loc);
  FileOffset = Sources.getFileOffset(Loc);
}

ClangTidyError::ClangTidyError(const ClangTidyMessage &Message)
    : Message(Message) {}

DiagnosticBuilder ClangTidyContext::Diag(SourceLocation Loc,
                                         StringRef Message) {
  return DiagEngine->Report(
      Loc, DiagEngine->getCustomDiagID(DiagnosticsEngine::Warning, Message));
}

void ClangTidyContext::setDiagnosticsEngine(DiagnosticsEngine *Engine) {
  DiagEngine = Engine;
}

void ClangTidyContext::setSourceManager(SourceManager *SourceMgr) {
  DiagEngine->setSourceManager(SourceMgr);
}

/// \brief Store a \c ClangTidyError.
void ClangTidyContext::storeError(const ClangTidyError &Error) {
  Errors->push_back(Error);
}

void ClangTidyCheck::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  Context->setSourceManager(Result.SourceManager);
  check(Result);
}

std::vector<std::string> getCheckNames(StringRef EnableChecksRegex,
                                       StringRef DisableChecksRegex) {
  SmallVector<ClangTidyError, 8> Errors;
  clang::tidy::ClangTidyContext Context(&Errors);
  ClangTidyASTConsumerFactory Factory(EnableChecksRegex, DisableChecksRegex,
                                      Context);
  return Factory.getCheckNames();
}

void runClangTidy(StringRef EnableChecksRegex, StringRef DisableChecksRegex,
                  const tooling::CompilationDatabase &Compilations,
                  ArrayRef<std::string> Ranges,
                  SmallVectorImpl<ClangTidyError> *Errors) {
  // FIXME: Ranges are currently full files. Support selecting specific
  // (line-)ranges.
  ClangTool Tool(Compilations, Ranges);
  clang::tidy::ClangTidyContext Context(Errors);
  ClangTidyDiagnosticConsumer DiagConsumer(Context);

  Tool.setDiagnosticConsumer(&DiagConsumer);

  class ActionFactory : public FrontendActionFactory {
  public:
    ActionFactory(ClangTidyASTConsumerFactory *ConsumerFactory)
        : ConsumerFactory(ConsumerFactory) {}
    FrontendAction *create() LLVM_OVERRIDE {
      return new Action(ConsumerFactory);
    }

  private:
    class Action : public ASTFrontendAction {
    public:
      Action(ClangTidyASTConsumerFactory *Factory) : Factory(Factory) {}
      ASTConsumer *CreateASTConsumer(CompilerInstance &Compiler,
                                     StringRef File) LLVM_OVERRIDE {
        return Factory->CreateASTConsumer(Compiler, File);
      }

    private:
      ClangTidyASTConsumerFactory *Factory;
    };

    ClangTidyASTConsumerFactory *ConsumerFactory;
  };

  Tool.run(new ActionFactory(new ClangTidyASTConsumerFactory(
      EnableChecksRegex, DisableChecksRegex, Context)));
}

static void reportDiagnostic(const ClangTidyMessage &Message,
                             SourceManager &SourceMgr,
                             DiagnosticsEngine::Level Level,
                             DiagnosticsEngine &Diags) {
  SourceLocation Loc;
  if (!Message.FilePath.empty()) {
    const FileEntry *File =
        SourceMgr.getFileManager().getFile(Message.FilePath);
    FileID ID = SourceMgr.createFileID(File, SourceLocation(), SrcMgr::C_User);
    Loc = SourceMgr.getLocForStartOfFile(ID);
    Loc = Loc.getLocWithOffset(Message.FileOffset);
  }
  Diags.Report(Loc, Diags.getCustomDiagID(Level, Message.Message));
}

void handleErrors(SmallVectorImpl<ClangTidyError> &Errors, bool Fix) {
  FileManager Files((FileSystemOptions()));
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticConsumer *DiagPrinter =
      new TextDiagnosticPrinter(llvm::outs(), &*DiagOpts);
  DiagnosticsEngine Diags(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
                          &*DiagOpts, DiagPrinter);
  DiagPrinter->BeginSourceFile(LangOptions());
  SourceManager SourceMgr(Diags, Files);
  Rewriter Rewrite(SourceMgr, LangOptions());
  for (SmallVectorImpl<ClangTidyError>::iterator I = Errors.begin(),
                                                 E = Errors.end();
       I != E; ++I) {
    reportDiagnostic(I->Message, SourceMgr, DiagnosticsEngine::Warning, Diags);
    for (unsigned i = 0, e = I->Notes.size(); i != e; ++i) {
      reportDiagnostic(I->Notes[i], SourceMgr, DiagnosticsEngine::Note, Diags);
    }
    tooling::applyAllReplacements(I->Fix, Rewrite);
  }
  // FIXME: Run clang-format on changes.
  if (Fix)
    Rewrite.overwriteChangedFiles();
}

} // namespace tidy
} // namespace clang
