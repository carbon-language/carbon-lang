//===- Linker.cpp - Link together LLVM objects and libraries --------------===//
//
// This file contains routines to handle linking together LLVM bytecode files,
// and to handle annoying things like static libraries.
//
//===----------------------------------------------------------------------===//

#include "gccld.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Linker.h"
#include "Support/CommandLine.h"
#include "Support/FileUtilities.h"
#include "Support/Signals.h"
#include "Support/SystemUtils.h"
#include "Config/stdlib.h"
#include <algorithm>
#include <fstream>
#include <memory>
#include <set>

//
// Function: FileExists ()
//
// Description:
//  Determine if the specified filename exists and is readable.
//
// Inputs:
//  FN - The name of the file.
//
// Outputs:
//  None.
//
// Return Value:
//  TRUE - The file exists and is readable.
//  FALSE - The file does not exist or is unreadable.
//
static inline bool FileExists(const std::string &FN) {
  return access(FN.c_str(), R_OK | F_OK) != -1;
}

//
// Function: IsArchive ()
//
// Description:
//  Determine if the specified file is an ar archive.  It determines this by
//  checking the magic string at the beginning of the file.
//
// Inputs:
//  filename - A C++ string containing the name of the file.
//
// Outputs:
//  None.
//
// Return value:
//  TRUE  - The file is an archive.
//  FALSE - The file is not an archive.
//
static inline bool IsArchive(const std::string &filename)
{
  std::string ArchiveMagic("!<arch>\012");
  char buf[1 + ArchiveMagic.size()];
  std::ifstream f(filename.c_str());
  f.read(buf, ArchiveMagic.size());
  buf[ArchiveMagic.size()] = '\0';
  return ArchiveMagic == buf;
}

//
// Function: FindLib ()
//
// Description:
//  This function locates a particular library.  It will prepend and append
//  various directories, prefixes, and suffixes until it can find the library.
//
// Inputs:
//  Filename  - Name of the file to find.
//  Paths     - List of directories to search.
//
// Outputs:
//  None.
//
// Return value:
//  The name of the file is returned.
//  If the file is not found, an empty string is returned.
//
static std::string
FindLib(const std::string &Filename, const std::vector<std::string> &Paths)
{
  // Determine if the pathname can be found as it stands.
  if (FileExists(Filename))
    return Filename;

  // If that doesn't work, convert the name into a library name.
  std::string LibName = "lib" + Filename;

  // Iterate over the directories in Paths to see if we can find the library
  // there.
  for (unsigned Index = 0; Index != Paths.size(); ++Index) {
    std::string Directory = Paths[Index] + "/";

    if (FileExists(Directory + LibName + ".bc"))
      return Directory + LibName + ".bc";

    if (FileExists(Directory + LibName + ".so"))
      return Directory + LibName + ".so";

    if (FileExists(Directory + LibName + ".a"))
      return Directory + LibName + ".a";
  }

  // One last hope: Check LLVM_LIB_SEARCH_PATH.
  char *SearchPath = getenv("LLVM_LIB_SEARCH_PATH");
  if (SearchPath == NULL)
    return std::string();

  LibName = std::string(SearchPath) + "/" + LibName;
  if (FileExists(LibName))
    return LibName;

  return std::string();
}

//
// Function: GetAllDefinedSymbols ()
//
// Description:
//  Find all of the defined symbols in the specified module.
//
// Inputs:
//  M - The module in which to find defined symbols.
//
// Outputs:
//  DefinedSymbols - A set of C++ strings that will contain the name of all
//                   defined symbols.
//
// Return value:
//  None.
//
void
GetAllDefinedSymbols(Module *M, std::set<std::string> &DefinedSymbols)
{
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (I->hasName() && !I->isExternal() && !I->hasInternalLinkage())
      DefinedSymbols.insert(I->getName());
  for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
    if (I->hasName() && !I->isExternal() && !I->hasInternalLinkage())
      DefinedSymbols.insert(I->getName());
}

//
// Function: GetAllUndefinedSymbols ()
//
// Description:
//  This calculates the set of undefined symbols that still exist in an LLVM
//  module.  This is a bit tricky because there may be two symbols with the
//  same name but different LLVM types that will be resolved to each other but
//  aren't currently (thus we need to treat it as resolved).
//
// Inputs:
//  M - The module in which to find undefined symbols.
//
// Outputs:
//  UndefinedSymbols - A set of C++ strings containing the name of all
//                     undefined symbols.
//
// Return value:
//  None.
//
void
GetAllUndefinedSymbols(Module *M, std::set<std::string> &UndefinedSymbols)
{
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


//
// Function: LoadObject ()
//
// Description:
//  Read the specified bytecode object file.
//
// Inputs:
//  FN - The name of the file to load.
//
// Outputs:
//  OutErrorMessage - The error message to give back to the caller.
//
// Return Value:
//  A pointer to a module represening the bytecode file is returned.
//  If an error occurs, the pointer is 0.
//
std::auto_ptr<Module>
LoadObject(const std::string & FN, std::string &OutErrorMessage) {
  std::string ErrorMessage;
  Module *Result = ParseBytecodeFile(FN, &ErrorMessage);
  if (Result) return std::auto_ptr<Module>(Result);
  OutErrorMessage = "Bytecode file '" + FN + "' corrupt!";
  if (ErrorMessage.size()) OutErrorMessage += ": " + ErrorMessage;
  return std::auto_ptr<Module>();
}

//
// Function: LinkInArchive ()
//
// Description:
//  This function will open an archive library and link in all objects which
//  provide symbols that are currently undefined.
//
// Inputs:
//  M        - The module in which to link the archives.
//  Filename - The pathname of the archive.
//  Verbose  - Flags whether verbose messages should be printed.
//
// Outputs:
//  ErrorMessage - A C++ string detailing what error occurred, if any.
//
// Return Value:
//  TRUE  - An error occurred.
//  FALSE - No errors.
//
static bool LinkInArchive(Module *M,
                          const std::string &Filename,
                          std::string &ErrorMessage,
                          bool Verbose)
{
  //
  // Find all of the symbols currently undefined in the bytecode program.
  // If all the symbols are defined, the program is complete, and there is
  // no reason to link in any archive files.
  //
  std::set<std::string> UndefinedSymbols;
  GetAllUndefinedSymbols(M, UndefinedSymbols);
  if (UndefinedSymbols.empty()) {
    if (Verbose) std::cerr << "  No symbols undefined, don't link library!\n";
    return false;  // No need to link anything in!
  }

  //
  // Load in the archive objects.
  //
  if (Verbose) std::cerr << "  Loading '" << Filename << "'\n";
  std::vector<Module*> Objects;
  if (ReadArchiveFile(Filename, Objects, &ErrorMessage))
    return true;

  //
  // Figure out which symbols are defined by all of the modules in the archive.
  //
  std::vector<std::set<std::string> > DefinedSymbols;
  DefinedSymbols.resize(Objects.size());
  for (unsigned i = 0; i != Objects.size(); ++i) {
    GetAllDefinedSymbols(Objects[i], DefinedSymbols[i]);
  }

  // While we are linking in object files, loop.
  bool Linked = true;
  while (Linked) {     
    Linked = false;

    for (unsigned i = 0; i != Objects.size(); ++i) {
      // Consider whether we need to link in this module...  we only need to
      // link it in if it defines some symbol which is so far undefined.
      //
      const std::set<std::string> &DefSymbols = DefinedSymbols[i];

      bool ObjectRequired = false;
      for (std::set<std::string>::iterator I = UndefinedSymbols.begin(),
             E = UndefinedSymbols.end(); I != E; ++I)
        if (DefSymbols.count(*I)) {
          if (Verbose)
            std::cerr << "  Found object providing symbol '" << *I << "'...\n";
          ObjectRequired = true;
          break;
        }
      
      // We DO need to link this object into the program...
      if (ObjectRequired) {
        if (LinkModules(M, Objects[i], &ErrorMessage))
          return true;   // Couldn't link in the right object file...        
        
        // Since we have linked in this object, delete it from the list of
        // objects to consider in this archive file.
        std::swap(Objects[i], Objects.back());
        std::swap(DefinedSymbols[i], DefinedSymbols.back());
        Objects.pop_back();
        DefinedSymbols.pop_back();
        --i;   // Do not skip an entry
        
        // The undefined symbols set should have shrunk.
        GetAllUndefinedSymbols(M, UndefinedSymbols);
        Linked = true;  // We have linked something in!
      }
    }
  }
  
  return false;
}

//
// Function: LinkInFile ()
//
// Description:
//  This function will open an archive library and link in all objects which
//  provide symbols that are currently undefined.
//
// Inputs:
//  HeadModule - The module in which to link the archives.
//  Filename   - The pathname of the archive.
//  Verbose    - Flags whether verbose messages should be printed.
//
// Outputs:
//  ErrorMessage - A C++ string detailing what error occurred, if any.
//
// Return Value:
//  TRUE  - An error occurred.
//  FALSE - No errors.
//
static bool LinkInFile(Module *HeadModule,
                       const std::string &Filename,
                       std::string &ErrorMessage,
                       bool Verbose)
{
  std::auto_ptr<Module> M(LoadObject(Filename, ErrorMessage));
  if (M.get() == 0) return true;
  if (Verbose) std::cerr << "Linking in '" << Filename << "'\n";
  return LinkModules(HeadModule, M.get(), &ErrorMessage);
}

//
// Function: LinkFiles ()
//
// Description:
//  This function takes a module and a list of files and links them all
//  together.  It locates the file either in the current directory, as it's
//  absolute or relative pathname, or as a file somewhere in
//  LLVM_LIB_SEARCH_PATH.
//
// Inputs:
//  progname   - The name of the program (infamous argv[0]).
//  HeadModule - The module under which all files will be linked.
//  Files      - A vector of C++ strings indicating the LLVM bytecode filenames
//               to be linked.  The names can refer to a mixture of pure LLVM
//               bytecode files and archive (ar) formatted files.
//  Verbose    - Flags whether verbose output should be printed while linking.
//
// Outputs:
//  HeadModule - The module will have the specified LLVM bytecode files linked
//               in.
//
// Return value:
//  FALSE - No errors.
//  TRUE  - Some error occurred.
//
bool LinkFiles(const char *progname,
               Module *HeadModule,
               const std::vector<std::string> &Files,
               bool Verbose)
{
  // String in which to receive error messages.
  std::string ErrorMessage;

  // Full pathname of the file
  std::string Pathname;

  // Get the library search path from the environment
  char *SearchPath = getenv("LLVM_LIB_SEARCH_PATH");

  for (unsigned i = 1; i < Files.size(); ++i) {
    // Determine where this file lives.
    if (FileExists(Files[i])) {
      Pathname = Files[i];
    } else {
      if (SearchPath == NULL) {
        std::cerr << "Cannot find linker input file '" << Files[i] << "'";
        return true;
      }

      Pathname = std::string(SearchPath)+"/"+Files[i];
      if (!FileExists(Pathname)) {
        std::cerr << "Cannot find linker input file '" << Files[i] << "'";
        return true;
      }
    }

    // A user may specify an ar archive without -l, perhaps because it
    // is not installed as a library. Detect that and link the library.
    if (IsArchive(Pathname)) {
      if (Verbose)
        std::cerr << "Linking archive '" << Files[i] << "'\n";

      if (LinkInArchive(HeadModule, Pathname, ErrorMessage, Verbose)) {
        PrintAndReturn(progname, ErrorMessage,
                       ": Error linking in '" + Files[i] + "'");
        return true;
      }
    } else {
      if (Verbose)
        std::cerr << "Linking file '" << Files[i] << "'\n";

      if (LinkInFile(HeadModule, Pathname, ErrorMessage, Verbose)) {
        PrintAndReturn(progname, ErrorMessage,
                       ": error linking in '" + Files[i] + "'");
        return true;
      }
    }
  }

  return false;
}

//
// Function: LinkLibraries ()
//
// Description:
//  This function takes the specified library files and links them into the
//  main bytecode object file.
//
// Inputs:
//  progname   - The name of the program (infamous argv[0]).
//  HeadModule - The module into which all necessary libraries will be linked.
//  Libraries  - The list of libraries to link into the module.
//  LibPaths   - The list of library paths in which to find libraries.
//  Verbose    - Flags whether verbose messages should be printed.
//  Native     - Flags whether native code is being generated.
//
// Outputs:
//  HeadModule - The module will have all necessary libraries linked in.
//
// Return value:
//  FALSE - No error.
//  TRUE  - Error.
//
bool LinkLibraries(const char *progname,
                   Module *HeadModule,
                   const std::vector<std::string> &Libraries,
                   const std::vector<std::string> &LibPaths,
                   bool Verbose,
                   bool Native)
{
  // String in which to receive error messages.
  std::string ErrorMessage;

  for (unsigned i = 1; i < Libraries.size(); ++i) {
    // Determine where this library lives.
    std::string Pathname = FindLib(Libraries[i], LibPaths);
    if (Pathname.empty()) {
      // If the pathname does not exist, then continue to the next one if
      // we're doing a native link and give an error if we're doing a bytecode
      // link.
      if (!Native) {
        PrintAndReturn(progname, "Cannot find " + Libraries[i]);
        return true;
      }
    }

    // A user may specify an ar archive without -l, perhaps because it
    // is not installed as a library. Detect that and link the library.
    if (IsArchive(Pathname)) {
      if (Verbose)
        std::cerr << "Linking archive '" << Libraries[i] << "'\n";

      if (LinkInArchive(HeadModule, Pathname, ErrorMessage, Verbose)) {
        PrintAndReturn(progname, ErrorMessage,
                       ": Error linking in '" + Libraries[i] + "'");
        return true;
      }
    } else {
      if (Verbose)
        std::cerr << "Linking file '" << Libraries[i] << "'\n";

      if (LinkInFile(HeadModule, Pathname, ErrorMessage, Verbose)) {
        PrintAndReturn(progname, ErrorMessage,
                       ": error linking in '" + Libraries[i] + "'");
        return true;
      }
    }
  }

  return false;
}
