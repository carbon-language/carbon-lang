#include "SyntaxCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

using namespace clang;
using namespace tooling;

class SyntaxCheck : public SyntaxOnlyAction {
public:
  SyntaxCheck(const FileOverrides &Overrides) : Overrides(Overrides) {}

  virtual bool BeginSourceFileAction(CompilerInstance &CI, StringRef Filename) {
    if (!SyntaxOnlyAction::BeginSourceFileAction(CI, Filename))
      return false;

    FileOverrides::const_iterator I = Overrides.find(Filename);
    if (I != Overrides.end())
      I->second.applyOverrides(CI.getSourceManager());

    return true;
  }

private:
  const FileOverrides &Overrides;
};

class SyntaxCheckFactory : public FrontendActionFactory {
public:
  SyntaxCheckFactory(const FileOverrides &Overrides)
      : Overrides(Overrides) {}

  virtual FrontendAction *create() { return new SyntaxCheck(Overrides); }

private:
  const FileOverrides &Overrides;
};

class SyntaxArgumentsAdjuster : public ArgumentsAdjuster {
  CommandLineArguments Adjust(const CommandLineArguments &Args) {
    CommandLineArguments AdjustedArgs = Args;
    AdjustedArgs.push_back("-fsyntax-only");
    AdjustedArgs.push_back("-std=c++11");
    return AdjustedArgs;
  }
};

bool doSyntaxCheck(const CompilationDatabase &Database,
                   const std::vector<std::string> &SourcePaths,
                   const FileOverrides &Overrides) {
  ClangTool SyntaxTool(Database, SourcePaths);

  // Ensure C++11 support is enabled.
  // FIXME: This isn't necessary anymore since the Migrator requires C++11
  // to be enabled in the CompilationDatabase. Remove later.
  SyntaxTool.setArgumentsAdjuster(new SyntaxArgumentsAdjuster);

  return SyntaxTool.run(new SyntaxCheckFactory(Overrides)) == 0;
}
