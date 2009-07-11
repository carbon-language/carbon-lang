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

#include <stdexcept>
#include <string>

namespace cl = llvm::cl;
namespace sys = llvm::sys;
using namespace llvmc;

namespace {

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

// Sometimes plugins want to condition on the value in argv[0].
const char* ProgramName;

int Main(int argc, char** argv) {
  try {
    LanguageMap langMap;
    CompilationGraph graph;

    ProgramName = argv[0];

    cl::ParseCommandLineOptions
      (argc, argv, "LLVM Compiler Driver (Work In Progress)", true);

    PluginLoader Plugins;
    Plugins.PopulateLanguageMap(langMap);
    Plugins.PopulateCompilationGraph(graph);

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

    if (InputFilenames.empty()) {
      throw std::runtime_error("no input files");
    }

    return BuildTargets(graph, langMap);
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
