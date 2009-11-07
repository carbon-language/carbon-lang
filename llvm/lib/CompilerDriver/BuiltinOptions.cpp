//===--- BuiltinOptions.cpp - The LLVM Compiler Driver ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Definitions of all global command-line option variables.
//
//===----------------------------------------------------------------------===//

#include "llvm/CompilerDriver/BuiltinOptions.h"

#ifdef ENABLE_LLVMC_DYNAMIC_PLUGINS
#include "llvm/Support/PluginLoader.h"
#endif

namespace cl = llvm::cl;

// External linkage here is intentional.

cl::list<std::string> InputFilenames(cl::Positional, cl::desc("<input file>"),
                                     cl::ZeroOrMore);
cl::opt<std::string> OutputFilename("o", cl::desc("Output file name"),
                                    cl::value_desc("file"), cl::Prefix);
cl::opt<std::string> TempDirname("temp-dir", cl::desc("Temp dir name"),
                                 cl::value_desc("<directory>"), cl::Prefix);
cl::list<std::string> Languages("x",
          cl::desc("Specify the language of the following input files"),
          cl::ZeroOrMore);

cl::opt<bool> DryRun("dry-run",
                     cl::desc("Only pretend to run commands"));
cl::opt<bool> Time("time", cl::desc("Time individual commands"));
cl::opt<bool> VerboseMode("v",
                          cl::desc("Enable verbose mode"));

cl::opt<bool> CheckGraph("check-graph",
                         cl::desc("Check the compilation graph for errors"),
                         cl::Hidden);
cl::opt<bool> WriteGraph("write-graph",
                         cl::desc("Write compilation-graph.dot file"),
                         cl::Hidden);
cl::opt<bool> ViewGraph("view-graph",
                         cl::desc("Show compilation graph in GhostView"),
                         cl::Hidden);

cl::opt<SaveTempsEnum::Values> SaveTemps
("save-temps", cl::desc("Keep temporary files"),
 cl::init(SaveTempsEnum::Unset),
 cl::values(clEnumValN(SaveTempsEnum::Obj, "obj",
                       "Save files in the directory specified with -o"),
            clEnumValN(SaveTempsEnum::Cwd, "cwd",
                       "Use current working directory"),
            clEnumValN(SaveTempsEnum::Obj, "", "Same as 'cwd'"),
            clEnumValEnd),
 cl::ValueOptional);
