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
GenerateBytecode (Module * M,
                  bool Strip,
                  bool Internalize,
                  std::ostream * Out);

int
GenerateAssembly (const std::string & OutputFilename,
                  const std::string & InputFilename,
                  const std::string & llc,
                  char ** const envp);

int GenerateCFile(const std::string &OutputFile, const std::string &InputFile,
                  const std::string &llc, char ** const envp);
int
GenerateNative (const std::string & OutputFilename,
                const std::string & InputFilename,
                const std::vector<std::string> & Libraries,
                const std::vector<std::string> & LibPaths,
                const std::string & gcc,
                char ** const envp);

} // End llvm namespace
