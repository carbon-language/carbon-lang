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

#include <string>
#include <set>
#include <ostream>

namespace llvm {

void
GetAllDefinedSymbols (Module *M, std::set<std::string> &DefinedSymbols);

void
GetAllUndefinedSymbols(Module *M, std::set<std::string> &UndefinedSymbols);

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

std::auto_ptr<Module>
LoadObject (const std::string & FN, std::string &OutErrorMessage);

std::string FindLib(const std::string &Filename,
                    const std::vector<std::string> &Paths,
                    bool SharedObjectOnly = false);
  
void LinkLibraries (const char * progname, Module* HeadModule,
                    const std::vector<std::string> & Libraries,
                    const std::vector<std::string> & LibPaths,
                    bool Verbose, bool Native);
bool
LinkFiles (const char * progname,
           Module * HeadModule,
           const std::vector<std::string> & Files,
           bool Verbose);

} // End llvm namespace
