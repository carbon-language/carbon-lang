#include "Core/Transform.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

llvm::cl::OptionCategory TransformsOptionsCategory("Transforms' options");

namespace {

using namespace tooling;
using namespace ast_matchers;

/// \brief Custom FrontendActionFactory to produce FrontendActions that simply
/// forward (Begin|End)SourceFileAction calls to a given Transform.
class ActionFactory : public clang::tooling::FrontendActionFactory {
public:
  ActionFactory(MatchFinder &Finder, Transform &Owner)
  : Finder(Finder), Owner(Owner) {}

  virtual FrontendAction *create() LLVM_OVERRIDE {
    return new FactoryAdaptor(Finder, Owner);
  }

private:
  class FactoryAdaptor : public ASTFrontendAction {
  public:
    FactoryAdaptor(MatchFinder &Finder, Transform &Owner)
        : Finder(Finder), Owner(Owner) {}

    ASTConsumer *CreateASTConsumer(CompilerInstance &, StringRef) {
      return Finder.newASTConsumer();
    }

    virtual bool BeginSourceFileAction(CompilerInstance &CI,
                                       StringRef Filename) LLVM_OVERRIDE {
      if (!ASTFrontendAction::BeginSourceFileAction(CI, Filename))
        return false;

      return Owner.handleBeginSource(CI, Filename);
    }

    virtual void EndSourceFileAction() LLVM_OVERRIDE {
      Owner.handleEndSource();
      return ASTFrontendAction::EndSourceFileAction();
    }

  private:
    MatchFinder &Finder;
    Transform &Owner;
  };

  MatchFinder &Finder;
  Transform &Owner;
};
} // namespace

/// \brief Class for creating Rewriter objects and housing Rewriter
/// dependencies.
///
/// A Rewriter depends on a SourceManager which in turn depends on a
/// FileManager and a DiagnosticsEngine. Transform uses this class to create a
/// new Rewriter and SourceManager for every translation unit it transforms. A
/// DiagnosticsEngine doesn't need to be re-created so it's constructed once. A
/// SourceManager and Rewriter and (re)created as required.
///
/// FIXME: The DiagnosticsEngine should really come from somewhere more global.
/// It shouldn't be re-created once for every transform.
///
/// NOTE: SourceManagers cannot be shared. Therefore the one used to parse the
/// translation unit cannot be used to create a Rewriter. This is why both a
/// SourceManager and Rewriter need to be created for each translation unit.
class RewriterManager {
public:
  RewriterManager()
      : DiagOpts(new DiagnosticOptions()),
        DiagnosticPrinter(llvm::errs(), DiagOpts.getPtr()),
        Diagnostics(
            llvm::IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
            DiagOpts.getPtr(), &DiagnosticPrinter, false) {}

  void prepare(FileManager &Files) {
    Sources.reset(new SourceManager(Diagnostics, Files));
    Rewrite.reset(new Rewriter(*Sources, DefaultLangOptions));
  }

  void applyOverrides(const SourceOverrides &Overrides) {
    Overrides.applyOverrides(*Sources);
  }

  Rewriter &getRewriter() { return *Rewrite; }

private:
  LangOptions DefaultLangOptions;
  llvm::IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  TextDiagnosticPrinter DiagnosticPrinter;
  DiagnosticsEngine Diagnostics;
  llvm::OwningPtr<SourceManager> Sources;
  llvm::OwningPtr<Rewriter> Rewrite;
};

/// \brief Flatten the Rewriter buffers of \p Rewrite and store results as
/// file content overrides in \p Overrides.
void collectResults(clang::Rewriter &Rewrite, SourceOverrides &Overrides) {
  for (Rewriter::buffer_iterator I = Rewrite.buffer_begin(),
                                 E = Rewrite.buffer_end();
       I != E; ++I) {
    const FileEntry *Entry = Rewrite.getSourceMgr().getFileEntryForID(I->first);
    assert(Entry != 0 && "Expected a FileEntry");
    assert(Entry->getName() != 0 &&
           "Unexpected NULL return from FileEntry::getName()");

    std::string ResultBuf;

    // Get a copy of the rewritten buffer from the Rewriter.
    llvm::raw_string_ostream StringStream(ResultBuf);
    I->second.write(StringStream);

    // Cause results to be written to ResultBuf.
    StringStream.str();

    // FIXME: Use move semantics to avoid copies of the buffer contents if
    // benchmarking shows the copies are expensive, especially for large source
    // files.

    if (Overrides.MainFileName == Entry->getName()) {
      Overrides.MainFileOverride = ResultBuf;
      continue;
    }

    // Header overrides are treated differently. Eventually, raw replacements
    // will be stored as well for later output to disk. Applying replacements
    // in memory will always be necessary as the source goes down the transform
    // pipeline.

    HeaderOverrides &Headers = Overrides.Headers;
    HeaderOverrides::iterator HeaderI = Headers.find(Entry->getName());
    if (HeaderI == Headers.end())
      HeaderI = Headers.insert(HeaderOverrides::value_type(
          Entry->getName(), Entry->getName())).first;

    HeaderI->second.FileOverride = ResultBuf;
  }
}

Transform::Transform(llvm::StringRef Name, const TransformOptions &Options)
    : Name(Name), GlobalOptions(Options), Overrides(0),
      RewriterOwner(new RewriterManager) {
  Reset();
}

Transform::~Transform() {}

bool Transform::isFileModifiable(const SourceManager &SM,
                                 const SourceLocation &Loc) const {
  if (SM.isFromMainFile(Loc))
    return true;

  if (!GlobalOptions.EnableHeaderModifications)
    return false;

  const FileEntry *FE = SM.getFileEntryForID(SM.getFileID(Loc));
  if (!FE)
    return false;
  
  return GlobalOptions.ModifiableHeaders.isFileIncluded(FE->getName());
}

bool Transform::handleBeginSource(CompilerInstance &CI, StringRef Filename) {
  assert(Overrides != 0 && "Subclass transform didn't provide InputState");

  CurrentSource = Filename.str();

  RewriterOwner->prepare(CI.getFileManager());
  FileOverrides::const_iterator I = Overrides->find(CurrentSource);
  if (I != Overrides->end()) {
    I->second.applyOverrides(CI.getSourceManager());
    RewriterOwner->applyOverrides(I->second);
  }

  Replace.clear();

  if (Options().EnableTiming) {
    Timings.push_back(std::make_pair(Filename.str(), llvm::TimeRecord()));
    Timings.back().second -= llvm::TimeRecord::getCurrentTime(true);
  }
  return true;
}

void Transform::handleEndSource() {
  if (!getReplacements().empty()) {
    // FIXME: applyAllReplacements will indicate if it couldn't apply all
    // replacements. Handle that case.
    applyAllReplacements(getReplacements(), RewriterOwner->getRewriter());

    FileOverrides::iterator I = Overrides->find(CurrentSource);
    if (I == Overrides->end())
      I = Overrides
        ->insert(FileOverrides::value_type(CurrentSource, CurrentSource)).first;

    collectResults(RewriterOwner->getRewriter(), I->second);
  }

  if (Options().EnableTiming)
    Timings.back().second += llvm::TimeRecord::getCurrentTime(false);
}

void Transform::addTiming(llvm::StringRef Label, llvm::TimeRecord Duration) {
  Timings.push_back(std::make_pair(Label.str(), Duration));
}

FrontendActionFactory *
Transform::createActionFactory(MatchFinder &Finder) {
  return new ActionFactory(Finder, /*Owner=*/ *this);
}
