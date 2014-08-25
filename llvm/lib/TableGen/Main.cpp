//===- Main.cpp - Top-Level TableGen implementation -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TableGen is a tool which can be used to build up a description of something,
// then invoke one or more "tablegen backends" to emit information about the
// description in some predefined format.  In practice, this is used by the LLVM
// code generators to automate generation of a code generator through a
// high-level description of the target.
//
//===----------------------------------------------------------------------===//

#include "TGParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <cstdio>
#include <system_error>
using namespace llvm;

namespace {
  cl::opt<std::string>
  OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
                 cl::init("-"));

  cl::opt<std::string>
  DependFilename("d",
                 cl::desc("Dependency filename"),
                 cl::value_desc("filename"),
                 cl::init(""));

  cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

  cl::list<std::string>
  IncludeDirs("I", cl::desc("Directory of include files"),
              cl::value_desc("directory"), cl::Prefix);
}

/// \brief Create a dependency file for `-d` option.
///
/// This functionality is really only for the benefit of the build system.
/// It is similar to GCC's `-M*` family of options.
static int createDependencyFile(const TGParser &Parser, const char *argv0) {
  if (OutputFilename == "-") {
    errs() << argv0 << ": the option -d must be used together with -o\n";
    return 1;
  }
  std::error_code EC;
  tool_output_file DepOut(DependFilename, EC, sys::fs::F_Text);
  if (EC) {
    errs() << argv0 << ": error opening " << DependFilename << ":"
           << EC.message() << "\n";
    return 1;
  }
  DepOut.os() << OutputFilename << ":";
  const TGLexer::DependenciesMapTy &Dependencies = Parser.getDependencies();
  for (TGLexer::DependenciesMapTy::const_iterator I = Dependencies.begin(),
                                                  E = Dependencies.end();
       I != E; ++I) {
    DepOut.os() << " " << I->first;
  }
  DepOut.os() << "\n";
  DepOut.keep();
  return 0;
}

namespace llvm {

int TableGenMain(char *argv0, TableGenMainFn *MainFn) {
  RecordKeeper Records;

  // Parse the input file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (std::error_code EC = FileOrErr.getError()) {
    errs() << "Could not open input file '" << InputFilename
           << "': " << EC.message() << "\n";
    return 1;
  }

  // Tell SrcMgr about this buffer, which is what TGParser will pick up.
  SrcMgr.AddNewSourceBuffer(std::move(*FileOrErr), SMLoc());

  // Record the location of the include directory so that the lexer can find
  // it later.
  SrcMgr.setIncludeDirs(IncludeDirs);

  TGParser Parser(SrcMgr, Records);

  if (Parser.ParseFile())
    return 1;

  std::error_code EC;
  tool_output_file Out(OutputFilename, EC, sys::fs::F_Text);
  if (EC) {
    errs() << argv0 << ": error opening " << OutputFilename << ":"
           << EC.message() << "\n";
    return 1;
  }
  if (!DependFilename.empty()) {
    if (int Ret = createDependencyFile(Parser, argv0))
      return Ret;
  }

  if (MainFn(Out.os(), Records))
    return 1;

  if (ErrorsPrinted > 0) {
    errs() << argv0 << ": " << ErrorsPrinted << " errors.\n";
    return 1;
  }

  // Declare success.
  Out.keep();
  return 0;
}

}
