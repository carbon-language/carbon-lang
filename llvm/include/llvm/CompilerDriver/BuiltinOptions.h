//===--- BuiltinOptions.h - The LLVM Compiler Driver ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Declarations of all global command-line option variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INCLUDE_COMPILER_DRIVER_BUILTIN_OPTIONS_H
#define LLVM_INCLUDE_COMPILER_DRIVER_BUILTIN_OPTIONS_H

#include "llvm/Support/CommandLine.h"

#include <string>

namespace SaveTempsEnum { enum Values { Cwd, Obj, Unset }; }

extern llvm::cl::list<std::string> InputFilenames;
extern llvm::cl::opt<std::string> OutputFilename;
extern llvm::cl::opt<std::string> TempDirname;
extern llvm::cl::list<std::string> Languages;
extern llvm::cl::opt<bool> DryRun;
extern llvm::cl::opt<bool> Time;
extern llvm::cl::opt<bool> VerboseMode;
extern llvm::cl::opt<bool> CheckGraph;
extern llvm::cl::opt<bool> ViewGraph;
extern llvm::cl::opt<bool> WriteGraph;
extern llvm::cl::opt<SaveTempsEnum::Values> SaveTemps;

#endif // LLVM_INCLUDE_COMPILER_DRIVER_BUILTIN_OPTIONS_H
