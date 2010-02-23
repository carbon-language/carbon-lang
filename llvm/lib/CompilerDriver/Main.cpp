//===--- Main.cpp - The LLVM Compiler Driver --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  llvmc::Main function - driver entry point.
//
//===----------------------------------------------------------------------===//

#include "llvm/CompilerDriver/BuiltinOptions.h"
#include "llvm/CompilerDriver/CompilationGraph.h"
#include "llvm/CompilerDriver/Error.h"
#include "llvm/CompilerDriver/Plugin.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace cl = llvm::cl;
namespace sys = llvm::sys;
using namespace llvmc;

namespace {

  std::stringstream* GlobalTimeLog;

  sys::Path getTempDir() {
    sys::Path tempDir;

    // The --temp-dir option.
    if (!TempDirname.empty()) {
      tempDir = TempDirname;
    }
    // GCC 4.5-style -save-temps handling.
    else if (SaveTemps == SaveTempsEnum::Unset) {
      tempDir = sys::Path::GetTemporaryDirectory();
      return tempDir;
    }
    else if (SaveTemps == SaveTempsEnum::Obj && !OutputFilename.empty()) {
      tempDir = OutputFilename;
      tempDir = tempDir.getDirname();
    }
    else {
      // SaveTemps == Cwd --> use current dir (leave tempDir empty).
      return tempDir;
    }

    if (!tempDir.exists()) {
      std::string ErrMsg;
      if (tempDir.createDirectoryOnDisk(true, &ErrMsg))
        throw std::runtime_error(ErrMsg);
    }

    return tempDir;
  }

  /// BuildTargets - A small wrapper for CompilationGraph::Build.
  int BuildTargets(CompilationGraph& graph, const LanguageMap& langMap) {
    int ret;
    const sys::Path& tempDir = getTempDir();
    bool toDelete = (SaveTemps == SaveTempsEnum::Unset);

    try {
      ret = graph.Build(tempDir, langMap);
    }
    catch(...) {
      if (toDelete)
        tempDir.eraseFromDisk(true);
      throw;
    }

    if (toDelete)
      tempDir.eraseFromDisk(true);
    return ret;
  }
}

namespace llvmc {

// Used to implement -time option. External linkage is intentional.
void AppendToGlobalTimeLog(const std::string& cmd, double time) {
  *GlobalTimeLog << "# " << cmd << ' ' << time << '\n';
}

// Sometimes plugins want to condition on the value in argv[0].
const char* ProgramName;

int Main(int argc, char** argv) {
  try {
    LanguageMap langMap;
    CompilationGraph graph;

    ProgramName = argv[0];

    cl::ParseCommandLineOptions
      (argc, argv, "LLVM Compiler Driver (Work In Progress)",
       /* ReadResponseFiles = */ false);

    PluginLoader Plugins;
    Plugins.RunInitialization(langMap, graph);

    if (CheckGraph) {
      int ret = graph.Check();
      if (!ret)
        llvm::errs() << "check-graph: no errors found.\n";

      return ret;
    }

    if (ViewGraph) {
      graph.viewGraph();
      if (!WriteGraph)
        return 0;
    }

    if (WriteGraph) {
      graph.writeGraph(OutputFilename.empty()
                       ? std::string("compilation-graph.dot")
                       : OutputFilename);
      return 0;
    }

    if (Time) {
      GlobalTimeLog = new std::stringstream;
      GlobalTimeLog->precision(2);
    }

    int ret = BuildTargets(graph, langMap);

    if (Time) {
      llvm::errs() << GlobalTimeLog->str();
      delete GlobalTimeLog;
    }

    return ret;
  }
  catch(llvmc::error_code& ec) {
    return ec.code();
  }
  catch(const std::exception& ex) {
    llvm::errs() << argv[0] << ": " << ex.what() << '\n';
  }
  catch(...) {
    llvm::errs() << argv[0] << ": unknown error!\n";
  }
  return 1;
}

} // end namespace llvmc
