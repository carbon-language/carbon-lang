//===- bugpoint.cpp - The LLVM Bugpoint utility ---------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This program is an automated compiler debugger tool.  It is used to narrow
// down miscompilations and crash problems to a specific pass in the compiler,
// and the specific Module or Function input that is causing the problem.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "llvm/Support/PassNameParser.h"
#include "Support/CommandLine.h"
#include "Config/unistd.h"
#include <sys/resource.h>

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
               cl::desc("<input llvm ll/bc files>"));

// The AnalysesList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo*, bool, PassNameParser>
PassList(cl::desc("Passes available:"), cl::ZeroOrMore);

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              " LLVM automatic testcase reducer. See\nhttp://"
                              "llvm.cs.uiuc.edu/docs/CommandGuide/bugpoint.html"
                              " for more information.\n");

  BugDriver D(argv[0]);
  if (D.addSources(InputFilenames)) return 1;
  D.addPasses(PassList.begin(), PassList.end());

  // Bugpoint has the ability of generating a plethora of core files, so to
  // avoid filling up the disk, set the max core file size to 0.
  struct rlimit rlim;
  rlim.rlim_cur = rlim.rlim_max = 0;
  int res = setrlimit(RLIMIT_CORE, &rlim);
  if (res < 0) {
    // setrlimit() may have failed, but we're not going to let that stop us
    perror("setrlimit: RLIMIT_CORE");
  }

  return D.run();
}
