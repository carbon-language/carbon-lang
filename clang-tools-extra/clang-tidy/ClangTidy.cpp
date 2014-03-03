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
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
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
static const char *AnalyzerCheckNamePrefix = "clang-analyzer-";

static StringRef StaticAnalyzerChecks[] = {
#define GET_CHECKERS
#define CHECKER(FULLNAME, CLASS, DESCFILE, HELPTEXT, GROUPINDEX, HIDDEN)       \
  FULLNAME,
#include "../../../lib/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER
#undef GET_CHECKERS
};

class AnalyzerDiagnosticConsumer : public ento::PathDiagnosticConsumer {
public:
  AnalyzerDiagnosticConsumer(ClangTidyContext &Context) : Context(Context) {}

  void FlushDiagnosticsImpl(std::vector<const ento::PathDiagnostic *> &Diags,
                            FilesMade *filesMade) override {
    for (std::vector<const ento::PathDiagnostic *>::iterator I = Diags.begin(),
                                                             E = Diags.end();
         I != E; ++I) {
      const ento::PathDiagnostic *PD = *I;
      SmallString<64> CheckName(AnalyzerCheckNamePrefix);
      CheckName += PD->getCheckName();
      addRanges(Context.diag(CheckName, PD->getLocation().asLocation(),
                             PD->getShortDescription()),
                PD->path.back()->getRanges());

      ento::PathPieces FlatPath =
          PD->path.flatten(/*ShouldFlattenMacros=*/true);
      for (ento::PathPieces::const_iterator PI = FlatPath.begin(),
                                            PE = FlatPath.end();
           PI != PE; ++PI) {
        addRanges(Context.diag(CheckName, (*PI)->getLocation().asLocation(),
                               (*PI)->getString(), DiagnosticIDs::Note),
                  (*PI)->getRanges());
      }
    }
  }

  StringRef getName() const override { return "ClangTidyDiags"; }
  bool supportsLogicalOpControlFlow() const override { return true; }
  bool supportsCrossFileDiagnostics() const override { return true; }

private:
  ClangTidyContext &Context;

  // FIXME: Convert to operator<<(DiagnosticBuilder&, ArrayRef<SourceRange>).
  static const DiagnosticBuilder &addRanges(const DiagnosticBuilder &DB,
                                            ArrayRef<SourceRange> Ranges) {
    for (ArrayRef<SourceRange>::iterator I = Ranges.begin(), E = Ranges.end();
         I != E; ++I)
      DB << *I;
    return DB;
  }
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

  SmallVector<ASTConsumer *, 2> Consumers;
  if (!CheckFactories->empty())
    Consumers.push_back(Finder.newASTConsumer());

  AnalyzerOptionsRef Options = Compiler.getAnalyzerOpts();
  Options->CheckersControlList = getCheckersControlList();
  if (!Options->CheckersControlList.empty()) {
    Options->AnalysisStoreOpt = RegionStoreModel;
    Options->AnalysisDiagOpt = PD_NONE;
    Options->AnalyzeNestedBlocks = true;
    Options->eagerlyAssumeBinOpBifurcation = true;
    ento::AnalysisASTConsumer *AnalysisConsumer = ento::CreateAnalysisConsumer(
        Compiler.getPreprocessor(), Compiler.getFrontendOpts().OutputFile,
        Options, Compiler.getFrontendOpts().Plugins);
    AnalysisConsumer->AddDiagnosticConsumer(
        new AnalyzerDiagnosticConsumer(Context));
    Consumers.push_back(AnalysisConsumer);
  }
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
    CheckNames.push_back(AnalyzerCheckNamePrefix + I->first);

  std::sort(CheckNames.begin(), CheckNames.end());
  return CheckNames;
}

ClangTidyASTConsumerFactory::CheckersList
ClangTidyASTConsumerFactory::getCheckersControlList() {
  CheckersList List;
  ArrayRef<StringRef> Checks(StaticAnalyzerChecks);

  bool AnalyzerChecksEnabled = false;
  for (unsigned i = 0; i < Checks.size(); ++i) {
    std::string Checker((AnalyzerCheckNamePrefix + Checks[i]).str());
    AnalyzerChecksEnabled |=
        Filter.IsCheckEnabled(Checker) && !Checks[i].startswith("debug");
  }

  if (AnalyzerChecksEnabled) {
    // Run our regex against all possible static analyzer checkers.  Note that
    // debug checkers print values / run programs to visualize the CFG and are
    // thus not applicable to clang-tidy in general.
    //
    // Always add all core checkers if any other static analyzer checks are
    // enabled. This is currently necessary, as other path sensitive checks
    // rely on the core checkers.
    for (unsigned i = 0; i < Checks.size(); ++i) {
      std::string Checker((AnalyzerCheckNamePrefix + Checks[i]).str());

      if (Checks[i].startswith("core") ||
          (!Checks[i].startswith("debug") && Filter.IsCheckEnabled(Checker)))
        List.push_back(std::make_pair(Checks[i], true));
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

DiagnosticBuilder ClangTidyCheck::diag(SourceLocation Loc, StringRef Message,
                                       DiagnosticIDs::Level Level) {
  return Context->diag(CheckName, Loc, Message, Level);
}

void ClangTidyCheck::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  Context->setSourceManager(Result.SourceManager);
  check(Result);
}

void ClangTidyCheck::setName(StringRef Name) {
  assert(CheckName.empty());
  CheckName = Name.str();
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
    FrontendAction *create() override {
      return new Action(ConsumerFactory);
    }

  private:
    class Action : public ASTFrontendAction {
    public:
      Action(ClangTidyASTConsumerFactory *Factory) : Factory(Factory) {}
      ASTConsumer *CreateASTConsumer(CompilerInstance &Compiler,
                                     StringRef File) override {
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

static SourceLocation getLocation(SourceManager &SourceMgr, StringRef FilePath,
                                  unsigned Offset) {
  if (FilePath.empty())
    return SourceLocation();

  const FileEntry *File = SourceMgr.getFileManager().getFile(FilePath);
  FileID ID = SourceMgr.createFileID(File, SourceLocation(), SrcMgr::C_User);
  return SourceMgr.getLocForStartOfFile(ID).getLocWithOffset(Offset);
}

static void reportDiagnostic(const ClangTidyMessage &Message,
                             SourceManager &SourceMgr,
                             DiagnosticsEngine::Level Level,
                             DiagnosticsEngine &Diags,
                             tooling::Replacements *Fixes = NULL) {
  SourceLocation Loc =
      getLocation(SourceMgr, Message.FilePath, Message.FileOffset);
  DiagnosticBuilder Diag = Diags.Report(Loc, Diags.getCustomDiagID(Level, "%0"))
                           << Message.Message;
  if (Fixes != NULL) {
    for (tooling::Replacements::const_iterator I = Fixes->begin(),
                                               E = Fixes->end();
         I != E; ++I) {
      SourceLocation FixLoc =
          getLocation(SourceMgr, I->getFilePath(), I->getOffset());
      Diag << FixItHint::CreateReplacement(
                  SourceRange(FixLoc, FixLoc.getLocWithOffset(I->getLength())),
                  I->getReplacementText());
    }
  }
}

void handleErrors(SmallVectorImpl<ClangTidyError> &Errors, bool Fix) {
  FileManager Files((FileSystemOptions()));
  LangOptions LangOpts; // FIXME: use langopts from each original file
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagOpts->ShowColors = llvm::sys::Process::StandardOutHasColors();
  DiagnosticConsumer *DiagPrinter =
      new TextDiagnosticPrinter(llvm::outs(), &*DiagOpts);
  DiagnosticsEngine Diags(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
                          &*DiagOpts, DiagPrinter);
  DiagPrinter->BeginSourceFile(LangOpts);
  SourceManager SourceMgr(Diags, Files);
  Rewriter Rewrite(SourceMgr, LangOpts);
  for (SmallVectorImpl<ClangTidyError>::iterator I = Errors.begin(),
                                                 E = Errors.end();
       I != E; ++I) {
    reportDiagnostic(I->Message, SourceMgr, DiagnosticsEngine::Warning, Diags,
                     &I->Fix);
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
