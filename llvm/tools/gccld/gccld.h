//===- util.h - Utility functions header file -----------------------------===//
//
// This file contains function prototypes for the functions in util.cpp.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"

#include <string>
#include <set>
#include <ostream>

int
PrintAndReturn (const char *progname,
                const std::string &Message,
                const std::string &Extra = "");

void
GetAllDefinedSymbols (Module *M, std::set<std::string> &DefinedSymbols);

void
GetAllUndefinedSymbols(Module *M, std::set<std::string> &UndefinedSymbols);

char **
CopyEnv (char ** const envp);

void
RemoveEnv (const char * name, char ** const envp);

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
int
GenerateNative (const std::string & OutputFilename,
                const std::string & InputFilename,
                const std::vector<std::string> & Libraries,
                const std::vector<std::string> & LibPaths,
                const std::string & gcc,
                char ** const envp);

std::auto_ptr<Module>
LoadObject (const std::string & FN, std::string &OutErrorMessage);

bool
LinkLibraries (const char * progname,
               Module * HeadModule,
               const std::vector<std::string> & Libraries,
               const std::vector<std::string> & LibPaths,
               bool Verbose,
               bool Native);
bool
LinkFiles (const char * progname,
           Module * HeadModule,
           const std::vector<std::string> & Files,
           bool Verbose);

