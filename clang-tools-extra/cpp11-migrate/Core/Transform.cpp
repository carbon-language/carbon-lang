#include "Core/Transform.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

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

RewriterContainer::RewriterContainer(clang::FileManager &Files,
                                     const FileOverrides &InputStates)
    : DiagOpts(new clang::DiagnosticOptions()),
      DiagnosticPrinter(llvm::errs(), DiagOpts.getPtr()),
      Diagnostics(llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
                      new clang::DiagnosticIDs()),
                  DiagOpts.getPtr(), &DiagnosticPrinter, false),
      Sources(Diagnostics, Files), Rewrite(Sources, DefaultLangOptions) {
  for (FileOverrides::const_iterator I = InputStates.begin(),
                                       E = InputStates.end();
       I != E; ++I)
    I->second.applyOverrides(Sources, Files);
}

void collectResults(clang::Rewriter &Rewrite,
                    const FileOverrides &InputStates,
                    FileOverrides &Results) {
  // Copy the contents of InputStates to be modified.
  Results = InputStates;

  for (Rewriter::buffer_iterator I = Rewrite.buffer_begin(),
                                 E = Rewrite.buffer_end();
       I != E; ++I) {
    const FileEntry *Entry = Rewrite.getSourceMgr().getFileEntryForID(I->first);
    assert(Entry != 0 && "Expected a FileEntry");
    assert(Entry->getName() != 0 &&
           "Unexpected NULL return from FileEntry::getName()");

    FileOverrides::iterator OverrideI = Results.find(Entry->getName());
    if (OverrideI == Results.end()) {
      OverrideI = Results.insert(FileOverrides::value_type(
          Entry->getName(), Entry->getName())).first;
    }

    std::string ResultBuf;

    // Get a copy of the rewritten buffer from the Rewriter.
    llvm::raw_string_ostream StringStream(ResultBuf);
    I->second.write(StringStream);

    // Cause results to be written to ResultBuf.
    StringStream.str();

    // FIXME: Use move semantics to avoid copies of the buffer contents if
    // benchmarking shows the copies are expensive, especially for large source
    // files.
    OverrideI->second.MainFileOverride = ResultBuf;
  }
}

bool Transform::handleBeginSource(CompilerInstance &CI, StringRef Filename) {
  assert(InputState != 0 && "Subclass transform didn't provide InputState");

  FileOverrides::const_iterator I = InputState->find(Filename.str());
  if (I != InputState->end()) {
    I->second.applyOverrides(CI.getSourceManager(), CI.getFileManager());
  }

  if (Options().EnableTiming) {
    Timings.push_back(std::make_pair(Filename.str(), llvm::TimeRecord()));
    Timings.back().second -= llvm::TimeRecord::getCurrentTime(true);
  }
  return true;
}

void Transform::handleEndSource() {
  if (!Options().EnableTiming)
    return;

  Timings.back().second += llvm::TimeRecord::getCurrentTime(false);
}

void Transform::addTiming(llvm::StringRef Label, llvm::TimeRecord Duration) {
  Timings.push_back(std::make_pair(Label.str(), Duration));
}

FrontendActionFactory *
Transform::createActionFactory(MatchFinder &Finder) {
  return new ActionFactory(Finder, /*Owner=*/ *this);
}
