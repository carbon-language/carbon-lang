//===- gccld.h - Utility functions header file ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains function prototypes for the functions in util.cpp.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Linker.h"

#include <string>
#include <set>
#include <ostream>

namespace llvm {

int
GenerateBytecode (Module *M,
                  int StripLevel,
                  bool Internalize,
                  std::ostream *Out);

int
GenerateAssembly (const std::string &OutputFilename,
                  const std::string &InputFilename,
                  const sys::Path &llc,
                  std::string& ErrMsg,
                  bool Verbose=false);

int
GenerateCFile (const std::string &OutputFile,
               const std::string &InputFile,
               const sys::Path &llc,
               std::string& ErrMsg,
               bool Verbose=false);
int
GenerateNative (const std::string &OutputFilename,
                const std::string &InputFilename,
                const std::vector<std::string> &LibPaths,
                const std::vector<std::string> &Libraries,
                const sys::Path &gcc,
                char ** const envp,
                bool Shared,
                bool ExportAllAsDynamic,
                const std::vector<std::string> &RPath,
                const std::string &SOName,
                std::string& ErrMsg,
                bool Verbose=false);

} // End llvm namespace
