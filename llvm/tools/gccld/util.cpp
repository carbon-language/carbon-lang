//===- util.cpp - Utility functions ---------------------------------------===//
//
// This file contains utility functions for gccld.  It essentially holds
// anything from the original gccld.cpp source that was either incidental
// or not inlined.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "Config/string.h"

#include <fstream>
#include <string>
#include <set>

//
// Function: PrintAndReturn ()
//
// Description:
//  Prints a message (usually error message) to standard error (stderr) and
//  returns a value usable for an exit status.
//
// Inputs:
//  progname - The name of the program (i.e. argv[0]).
//  Message  - The message to print to standard error.
//  Extra    - Extra information to print between the program name and thei
//             message.  It is optional.
//
// Outputs:
//  None.
//
// Return value:
//  Returns a value that can be used as the exit status (i.e. for exit()).
//
int
PrintAndReturn (const char *progname,
                const std::string &Message,
                const std::string &Extra = "")
{
  std::cerr << progname << Extra << ": " << Message << "\n";
  return 1;
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
bool
IsArchive (const std::string &filename)
{
  std::string ArchiveMagic("!<arch>\012");
  char buf[1 + ArchiveMagic.size()];

  std::ifstream f(filename.c_str());
  f.read(buf, ArchiveMagic.size());
  buf[ArchiveMagic.size()] = '\0';
  return ArchiveMagic == buf;
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
GetAllDefinedSymbols (Module *M, std::set<std::string> &DefinedSymbols)
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
//
// Function: copy_env()
//
// Description:
//	This function takes an array of environment variables and makes a
//	copy of it.  This copy can then be manipulated any way the caller likes
//  without affecting the process's real environment.
//
// Inputs:
//  envp - An array of C strings containing an environment.
//
// Outputs:
//  None.
//
// Return value:
//  NULL - An error occurred.
//
//  Otherwise, a pointer to a new array of C strings is returned.  Every string
//  in the array is a duplicate of the one in the original array (i.e. we do
//  not copy the char *'s from one array to another).
//
char **
copy_env (char ** const envp)
{
  // The new environment list
  char ** newenv;

  // The number of entries in the old environment list
  int entries;

  //
  // Count the number of entries in the old list;
  //
  for (entries = 0; envp[entries] != NULL; entries++)
  {
    ;
  }

  //
  // Add one more entry for the NULL pointer that ends the list.
  //
  ++entries;

  //
  // If there are no entries at all, just return NULL.
  //
  if (entries == 0)
  {
    return NULL;
  }

  //
  // Allocate a new environment list.
  //
  if ((newenv = new (char *) [entries]) == NULL)
  {
    return NULL;
  }

  //
  // Make a copy of the list.  Don't forget the NULL that ends the list.
  //
  entries = 0;
  while (envp[entries] != NULL)
  {
    newenv[entries] = new char[strlen (envp[entries]) + 1];
    strcpy (newenv[entries], envp[entries]);
    ++entries;
  }
  newenv[entries] = NULL;

  return newenv;
}


//
// Function: remove_env()
//
// Description:
//	Remove the specified environment variable from the environment array.
//
// Inputs:
//	name - The name of the variable to remove.  It cannot be NULL.
//	envp - The array of environment variables.  It cannot be NULL.
//
// Outputs:
//	envp - The pointer to the specified variable name is removed.
//
// Return value:
//	None.
//
// Notes:
//  This is mainly done because functions to remove items from the environment
//  are not available across all platforms.  In particular, Solaris does not
//  seem to have an unsetenv() function or a setenv() function (or they are
//  undocumented if they do exist).
//
void
remove_env (const char * name, char ** const envp)
{
  // Pointer for scanning arrays
  register char * p;

  // Index for selecting elements of the environment array
  register int index;

  for (index=0; envp[index] != NULL; index++)
  {
    //
    // Find the first equals sign in the array and make it an EOS character.
    //
    p = strchr (envp[index], '=');
    if (p == NULL)
    {
      continue;
    }
    else
    {
      *p = '\0';
    }

    //
    // Compare the two strings.  If they are equal, zap this string.
    // Otherwise, restore it.
    //
    if (!strcmp (name, envp[index]))
    {
      *envp[index] = '\0';
    }
    else
    {
      *p = '=';
    }
  }

  return;
}

