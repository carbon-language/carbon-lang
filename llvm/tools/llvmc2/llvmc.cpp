//===--- llvmc.cpp - The LLVM Compiler Driver -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This tool provides a single point of access to the LLVM
//  compilation tools.  It has many options. To discover the options
//  supported please refer to the tools' manual page or run the tool
//  with the --help option.
//
//===----------------------------------------------------------------------===//

#include "CompilationGraph.h"
#include "Error.h"
#include "Plugin.h"

#include "llvm/System/Path.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PluginLoader.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace cl = llvm::cl;
namespace sys = llvm::sys;
using namespace llvmc;

// Built-in command-line options.
// External linkage here is intentional.

cl::list<std::string> InputFilenames(cl::Positional, cl::desc("<input file>"),
                                     cl::ZeroOrMore);
cl::opt<std::string> OutputFilename("o", cl::desc("Output file name"),
                                    cl::value_desc("file"));
cl::list<std::string> Languages("x",
          cl::desc("Specify the language of the following input files"),
          cl::ZeroOrMore);
cl::opt<bool> DryRun("dry-run",
                     cl::desc("Only pretend to run commands"));
cl::opt<bool> VerboseMode("v",
                          cl::desc("Enable verbose mode"));
cl::opt<bool> WriteGraph("write-graph",
                         cl::desc("Write compilation-graph.dot file"),
                         cl::Hidden);
cl::opt<bool> ViewGraph("view-graph",
                         cl::desc("Show compilation graph in GhostView"),
                         cl::Hidden);
cl::opt<bool> SaveTemps("save-temps",
                         cl::desc("Keep temporary files"),
                         cl::Hidden);

namespace {
  /// BuildTargets - A small wrapper for CompilationGraph::Build.
  int BuildTargets(CompilationGraph& graph, const LanguageMap& langMap) {
    int ret;
    const sys::Path& tempDir = SaveTemps
      ? sys::Path("")
      : sys::Path(sys::Path::GetTemporaryDirectory());

    try {
      ret = graph.Build(tempDir, langMap);
    }
    catch(...) {
      tempDir.eraseFromDisk(true);
      throw;
    }

    if (!SaveTemps)
      tempDir.eraseFromDisk(true);
    return ret;
  }
}

int main(int argc, char** argv) {
  try {
    LanguageMap langMap;
    CompilationGraph graph;

    cl::ParseCommandLineOptions
      (argc, argv, "LLVM Compiler Driver (Work In Progress)", true);

    PopulateLanguageMap(langMap);
    PopulateCompilationGraph(graph);

    if (WriteGraph) {
      graph.writeGraph();
      if (!ViewGraph)
        return 0;
    }

    if (ViewGraph) {
      graph.viewGraph();
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
    std::cerr << argv[0] << ": " << ex.what() << '\n';
  }
  catch(...) {
    std::cerr << argv[0] << ": unknown error!\n";
  }
  return 1;
}
