//===- util.h - Utility functions header file -----------------------------===//
//
// This file contains function prototypes for the functions in util.cpp.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"

#include <string>
#include <set>

extern int
PrintAndReturn (const char *progname,
                const std::string &Message,
                const std::string &Extra = "");

extern bool
IsArchive (const std::string &filename);

extern void
GetAllDefinedSymbols (Module *M, std::set<std::string> &DefinedSymbols);

extern void
GetAllUndefinedSymbols(Module *M, std::set<std::string> &UndefinedSymbols);

extern char **
copy_env (char ** const envp);

extern void
remove_env (const char * name, char ** const envp);

