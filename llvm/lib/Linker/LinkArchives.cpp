//===- lib/Linker/LinkArchives.cpp - Link LLVM objects and libraries ------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains routines to handle linking together LLVM bytecode files,
// and to handle annoying things like static libraries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Archive.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Config/config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/System/Signals.h"
#include "llvm/Support/SystemUtils.h"
#include <algorithm>
#include <fstream>
#include <memory>
#include <set>
using namespace llvm;

/// FindLib - Try to convert Filename into the name of a file that we can open,
/// if it does not already name a file we can open, by first trying to open
/// Filename, then libFilename.[suffix] for each of a set of several common
/// library suffixes, in each of the directories in Paths and the directory
/// named by the value of the environment variable LLVM_LIB_SEARCH_PATH. Returns
/// an empty string if no matching file can be found.
///
std::string llvm::FindLib(const std::string &Filename,
                          const std::vector<std::string> &Paths,
                          bool SharedObjectOnly) {
  // Determine if the pathname can be found as it stands.
  if (FileOpenable(Filename))
    return Filename;

  // If that doesn't work, convert the name into a library name.
  std::string LibName = "lib" + Filename;

  // Iterate over the directories in Paths to see if we can find the library
  // there.
  for (unsigned Index = 0; Index != Paths.size(); ++Index) {
    std::string Directory = Paths[Index] + "/";

    if (!SharedObjectOnly && FileOpenable(Directory + LibName + ".bc"))
      return Directory + LibName + ".bc";

    if (FileOpenable(Directory + LibName + SHLIBEXT))
      return Directory + LibName + SHLIBEXT;

    if (!SharedObjectOnly && FileOpenable(Directory + LibName + ".a"))
      return Directory + LibName + ".a";
  }

  // One last hope: Check LLVM_LIB_SEARCH_PATH.
  char *SearchPath = getenv("LLVM_LIB_SEARCH_PATH");
  if (SearchPath == NULL)
    return std::string();

  LibName = std::string(SearchPath) + "/" + LibName;
  if (FileOpenable(LibName))
    return LibName;

  return std::string();
}

/// GetAllDefinedSymbols - Modifies its parameter DefinedSymbols to contain the
/// name of each externally-visible symbol defined in M.
///
void llvm::GetAllDefinedSymbols(Module *M,
                                std::set<std::string> &DefinedSymbols) {
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (I->hasName() && !I->isExternal() && !I->hasInternalLinkage())
      DefinedSymbols.insert(I->getName());
  for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
    if (I->hasName() && !I->isExternal() && !I->hasInternalLinkage())
      DefinedSymbols.insert(I->getName());
}

/// GetAllUndefinedSymbols - calculates the set of undefined symbols that still
/// exist in an LLVM module. This is a bit tricky because there may be two
/// symbols with the same name but different LLVM types that will be resolved to
/// each other but aren't currently (thus we need to treat it as resolved).
///
/// Inputs:
///  M - The module in which to find undefined symbols.
///
/// Outputs:
///  UndefinedSymbols - A set of C++ strings containing the name of all
///                     undefined symbols.
///
void
llvm::GetAllUndefinedSymbols(Module *M,
                             std::set<std::string> &UndefinedSymbols) {
  std::set<std::string> DefinedSymbols;
  UndefinedSymbols.clear();   // Start out empty
  
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (I->hasName()) {
      if (I->isExternal())
        UndefinedSymbols.insert(I->getName());
      else if (!I->hasInternalLinkage())
        DefinedSymbols.insert(I->getName());
    }
  for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
    if (I->hasName()) {
      if (I->isExternal())
        UndefinedSymbols.insert(I->getName());
      else if (!I->hasInternalLinkage())
        DefinedSymbols.insert(I->getName());
    }
  
  // Prune out any defined symbols from the undefined symbols set...
  for (std::set<std::string>::iterator I = UndefinedSymbols.begin();
       I != UndefinedSymbols.end(); )
    if (DefinedSymbols.count(*I))
      UndefinedSymbols.erase(I++);  // This symbol really is defined!
    else
      ++I; // Keep this symbol in the undefined symbols list
}


/// LoadObject - Read in and parse the bytecode file named by FN and return the
/// module it contains (wrapped in an auto_ptr), or 0 and set ErrorMessage if an
/// error occurs.
///
static std::auto_ptr<Module> LoadObject(const std::string &FN,
                                       std::string &ErrorMessage) {
  std::string ParserErrorMessage;
  Module *Result = ParseBytecodeFile(FN, &ParserErrorMessage);
  if (Result) return std::auto_ptr<Module>(Result);
  ErrorMessage = "Bytecode file '" + FN + "' could not be loaded";
  if (ParserErrorMessage.size()) ErrorMessage += ": " + ParserErrorMessage;
  return std::auto_ptr<Module>();
}

/// LinkInArchive - opens an archive library and link in all objects which
/// provide symbols that are currently undefined.
///
/// Inputs:
///  M        - The module in which to link the archives.
///  Filename - The pathname of the archive.
///  Verbose  - Flags whether verbose messages should be printed.
///
/// Outputs:
///  ErrorMessage - A C++ string detailing what error occurred, if any.
///
/// Return Value:
///  TRUE  - An error occurred.
///  FALSE - No errors.
///
bool llvm::LinkInArchive(Module *M,
                          const std::string &Filename,
                          std::string* ErrorMessage,
                          bool Verbose)
{
  // Find all of the symbols currently undefined in the bytecode program.
  // If all the symbols are defined, the program is complete, and there is
  // no reason to link in any archive files.
  std::set<std::string> UndefinedSymbols;
  GetAllUndefinedSymbols(M, UndefinedSymbols);
  if (UndefinedSymbols.empty()) {
    if (Verbose) std::cerr << "  No symbols undefined, don't link library!\n";
    return false;  // No need to link anything in!
  }

  // Open the archive file
  if (Verbose) std::cerr << "  Loading archive file '" << Filename << "'\n";
  Archive* arch = Archive::OpenAndLoadSymbols(sys::Path(Filename));

  // While we are linking in object files, loop.
  while (true) {     
    std::set<ModuleProvider*> Modules;
    // Find the modules we need to link
    arch->findModulesDefiningSymbols(UndefinedSymbols,Modules);

    // If we didn't find any more modules to link this time, we are done.
    if (Modules.empty())
      break;

    // Loop over all the ModuleProviders that we got back from the archive
    for (std::set<ModuleProvider*>::iterator I=Modules.begin(), E=Modules.end();
         I != E; ++I) {
      // Get the module we must link in.
      Module* aModule = (*I)->releaseModule();

      // Link it in
      if (LinkModules(M, aModule, ErrorMessage)) {
        // don't create a memory leak
        delete aModule;
        delete arch;
        return true;   // Couldn't link in the right object file...        
      }
        
      // Since we have linked in this object, throw it away now.
      delete aModule;
    }

    // We have linked in a set of modules determined by the archive to satisfy
    // our missing symbols. Linking in the new modules will have satisfied some
    // symbols but may introduce additional missing symbols. We need to update
    // the list of undefined symbols and try again until the archive doesn't
    // have any modules that satisfy our symbols. 
    GetAllUndefinedSymbols(M, UndefinedSymbols);
  }
  
  return false;
}

/// LinkInFile - opens a bytecode file and links in all objects which
/// provide symbols that are currently undefined.
///
/// Inputs:
///  HeadModule - The module in which to link the bytecode file.
///  Filename   - The pathname of the bytecode file.
///  Verbose    - Flags whether verbose messages should be printed.
///
/// Outputs:
///  ErrorMessage - A C++ string detailing what error occurred, if any.
///
/// Return Value:
///  TRUE  - An error occurred.
///  FALSE - No errors.
///
static bool LinkInFile(Module *HeadModule,
                       const std::string &Filename,
                       std::string &ErrorMessage,
                       bool Verbose)
{
  std::auto_ptr<Module> M(LoadObject(Filename, ErrorMessage));
  if (M.get() == 0) return true;
  bool Result = LinkModules(HeadModule, M.get(), &ErrorMessage);
  if (Verbose) std::cerr << "Linked in bytecode file '" << Filename << "'\n";
  return Result;
}

/// LinkFiles - takes a module and a list of files and links them all together.
/// It locates the file either in the current directory, as its absolute
/// or relative pathname, or as a file somewhere in LLVM_LIB_SEARCH_PATH.
///
/// Inputs:
///  progname   - The name of the program (infamous argv[0]).
///  HeadModule - The module under which all files will be linked.
///  Files      - A vector of C++ strings indicating the LLVM bytecode filenames
///               to be linked.  The names can refer to a mixture of pure LLVM
///               bytecode files and archive (ar) formatted files.
///  Verbose    - Flags whether verbose output should be printed while linking.
///
/// Outputs:
///  HeadModule - The module will have the specified LLVM bytecode files linked
///               in.
///
/// Return value:
///  FALSE - No errors.
///  TRUE  - Some error occurred.
///
bool llvm::LinkFiles(const char *progname, Module *HeadModule,
                     const std::vector<std::string> &Files, bool Verbose) {
  // String in which to receive error messages.
  std::string ErrorMessage;

  // Full pathname of the file
  std::string Pathname;

  // Get the library search path from the environment
  char *SearchPath = getenv("LLVM_LIB_SEARCH_PATH");

  for (unsigned i = 0; i < Files.size(); ++i) {
    // Determine where this file lives.
    if (FileOpenable(Files[i])) {
      Pathname = Files[i];
    } else {
      if (SearchPath == NULL) {
        std::cerr << progname << ": Cannot find linker input file '"
                  << Files[i] << "'\n";
        std::cerr << progname
                  << ": Warning: Your LLVM_LIB_SEARCH_PATH is unset.\n";
        return true;
      }

      Pathname = std::string(SearchPath)+"/"+Files[i];
      if (!FileOpenable(Pathname)) {
        std::cerr << progname << ": Cannot find linker input file '"
                  << Files[i] << "'\n";
        return true;
      }
    }

    // A user may specify an ar archive without -l, perhaps because it
    // is not installed as a library. Detect that and link the library.
    if (IsArchive(Pathname)) {
      if (Verbose)
        std::cerr << "Trying to link archive '" << Pathname << "'\n";

      if (LinkInArchive(HeadModule, Pathname, &ErrorMessage, Verbose)) {
        std::cerr << progname << ": Error linking in archive '" << Pathname 
                  << "': " << ErrorMessage << "\n";
        return true;
      }
    } else if (IsBytecode(Pathname)) {
      if (Verbose)
        std::cerr << "Trying to link bytecode file '" << Pathname << "'\n";

      if (LinkInFile(HeadModule, Pathname, ErrorMessage, Verbose)) {
        std::cerr << progname << ": Error linking in bytecode file '"
                  << Pathname << "': " << ErrorMessage << "\n";
        return true;
      }
    } else {
      std::cerr << progname << ": Warning: invalid file `" << Pathname 
                << "' ignored.\n";
    }
  }

  return false;
}

/// LinkLibraries - takes the specified library files and links them into the
/// main bytecode object file.
///
/// Inputs:
///  progname   - The name of the program (infamous argv[0]).
///  HeadModule - The module into which all necessary libraries will be linked.
///  Libraries  - The list of libraries to link into the module.
///  LibPaths   - The list of library paths in which to find libraries.
///  Verbose    - Flags whether verbose messages should be printed.
///  Native     - Flags whether native code is being generated.
///
/// Outputs:
///  HeadModule - The module will have all necessary libraries linked in.
///
/// Return value:
///  FALSE - No error.
///  TRUE  - Error.
///
void llvm::LinkLibraries(const char *progname, Module *HeadModule,
                         const std::vector<std::string> &Libraries,
                         const std::vector<std::string> &LibPaths,
                         bool Verbose, bool Native) {
  // String in which to receive error messages.
  std::string ErrorMessage;

  for (unsigned i = 0; i < Libraries.size(); ++i) {
    // Determine where this library lives.
    std::string Pathname = FindLib(Libraries[i], LibPaths);
    if (Pathname.empty()) {
      // If the pathname does not exist, then continue to the next one if
      // we're doing a native link and give an error if we're doing a bytecode
      // link.
      if (!Native) {
        std::cerr << progname << ": WARNING: Cannot find library -l"
                  << Libraries[i] << "\n";
        continue;
      }
    }

    // A user may specify an ar archive without -l, perhaps because it
    // is not installed as a library. Detect that and link the library.
    if (IsArchive(Pathname)) {
      if (Verbose)
        std::cerr << "Trying to link archive '" << Pathname << "' (-l"
                  << Libraries[i] << ")\n";

      if (LinkInArchive(HeadModule, Pathname, &ErrorMessage, Verbose)) {
        std::cerr << progname << ": " << ErrorMessage
                  << ": Error linking in archive '" << Pathname << "' (-l"
                  << Libraries[i] << ")\n";
        exit(1);
      }
    } else if (IsBytecode(Pathname)) {
      if (Verbose)
        std::cerr << "Trying to link bytecode file '" << Pathname
                  << "' (-l" << Libraries[i] << ")\n";

      if (LinkInFile(HeadModule, Pathname, ErrorMessage, Verbose)) {
        std::cerr << progname << ": " << ErrorMessage
                  << ": error linking in bytecode file '" << Pathname << "' (-l"
                  << Libraries[i] << ")\n";
        exit(1);
      }
    }
  }
}
