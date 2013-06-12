#include "Core/Transform.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {

using namespace tooling;
using namespace ast_matchers;

/// \brief Custom FrontendActionFactory to produce FrontendActions that handle
/// overriding source file contents before parsing.
///
/// The nested class FactoryAdaptor overrides BeginSourceFileAction to override
/// source file contents before parsing happens. Both Begin and
/// EndSourceFileAction call corresponding callbacks provided by
/// SourceFileCallbacks.
class ActionFactory : public clang::tooling::FrontendActionFactory {
public:
  ActionFactory(MatchFinder &Finder, const FileContentsByPath &Overrides,
                SourceFileCallbacks &Callbacks)
  : Finder(Finder), Overrides(Overrides), Callbacks(Callbacks) {}

  virtual FrontendAction *create() LLVM_OVERRIDE {
    return new FactoryAdaptor(Finder, Overrides, Callbacks);
  }

private:
  class FactoryAdaptor : public ASTFrontendAction {
  public:
    FactoryAdaptor(MatchFinder &Finder, const FileContentsByPath &Overrides,
                  SourceFileCallbacks &Callbacks)
        : Finder(Finder), Overrides(Overrides), Callbacks(Callbacks) {}

    ASTConsumer *CreateASTConsumer(CompilerInstance &, StringRef) {
      return Finder.newASTConsumer();
    }

    virtual bool BeginSourceFileAction(CompilerInstance &CI,
                                       StringRef Filename) LLVM_OVERRIDE {
      if (!ASTFrontendAction::BeginSourceFileAction(CI, Filename))
        return false;

      FileContentsByPath::const_iterator I = Overrides.find(Filename.str());
      if (I != Overrides.end())
        // If an override exists, use it.
        CI.getSourceManager()
            .overrideFileContents(CI.getFileManager().getFile(I->first),
                                  llvm::MemoryBuffer::getMemBuffer(I->second));

      return Callbacks.handleBeginSource(CI, Filename);
    }

    virtual void EndSourceFileAction() LLVM_OVERRIDE {
      Callbacks.handleEndSource();
      return ASTFrontendAction::EndSourceFileAction();
    }

  private:
    MatchFinder &Finder;
    const FileContentsByPath &Overrides;
    SourceFileCallbacks &Callbacks;
  };

  MatchFinder &Finder;
  const FileContentsByPath &Overrides;
  SourceFileCallbacks &Callbacks;
};

} // namespace

RewriterContainer::RewriterContainer(clang::FileManager &Files,
                                     const FileContentsByPath &InputStates)
    : DiagOpts(new clang::DiagnosticOptions()),
      DiagnosticPrinter(llvm::errs(), DiagOpts.getPtr()),
      Diagnostics(llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
                      new clang::DiagnosticIDs()),
                  DiagOpts.getPtr(), &DiagnosticPrinter, false),
      Sources(Diagnostics, Files), Rewrite(Sources, DefaultLangOptions) {

  // Overwrite source manager's file contents with data from InputStates
  for (FileContentsByPath::const_iterator I = InputStates.begin(),
                                          E = InputStates.end();
       I != E; ++I) {
    Sources.overrideFileContents(Files.getFile(I->first),
                                 llvm::MemoryBuffer::getMemBuffer(I->second));
  }
}

void collectResults(clang::Rewriter &Rewrite,
                    const FileContentsByPath &InputStates,
                    FileContentsByPath &Results) {
  // Copy the contents of InputStates to be modified.
  Results = InputStates;

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
    Results[Entry->getName()] = ResultBuf;
  }
}

bool Transform::handleBeginSource(CompilerInstance &CI, StringRef Filename) {
  if (!Options().EnableTiming)
    return true;

  Timings.push_back(std::make_pair(Filename.str(), llvm::TimeRecord()));
  Timings.back().second -= llvm::TimeRecord::getCurrentTime(true);
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
Transform::createActionFactory(MatchFinder &Finder,
                               const FileContentsByPath &InputStates) {
  return new ActionFactory(Finder, InputStates, *this);
}
