#include "TestRunner.h"

using namespace llvm;

TestRunner::TestRunner(StringRef TestName, std::vector<std::string> TestArgs,
                       StringRef ReducedFilepath, SmallString<128> TmpDirectory)
    : TestName(TestName), TestArgs(TestArgs), ReducedFilepath(ReducedFilepath),
      TmpDirectory(TmpDirectory) {}

/// Runs the interestingness test, passes file to be tested as first argument
/// and other specified test arguments after that.
int TestRunner::run(StringRef Filename) {
  std::vector<StringRef> ProgramArgs;
  ProgramArgs.push_back(TestName);
  ProgramArgs.push_back(Filename);

  for (unsigned I = 0, E = TestArgs.size(); I != E; ++I)
    ProgramArgs.push_back(TestArgs[I].c_str());

  StringRef SR = "";
  Optional<StringRef> Redirects[3] = {SR, SR, SR}; // STDIN, STDOUT, STDERR
  int Result = sys::ExecuteAndWait(TestName, ProgramArgs, None, Redirects);

  if (Result < 0) {
    Error E = make_error<StringError>("Error running interesting-ness test\n",
                                      inconvertibleErrorCode());
    outs() << toString(std::move(E));
    exit(1);
  }

  return !Result;
}
