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
#include "llvm/ADT/SetOperations.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Archive.h"
#include "llvm/Config/config.h"
#include <memory>
#include <set>
using namespace llvm;

/// GetAllDefinedSymbols - Modifies its parameter DefinedSymbols to contain the
/// name of each externally-visible symbol defined in M.
///
static void
GetAllDefinedSymbols(Module *M, std::set<std::string> &DefinedSymbols) {
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (I->hasName() && !I->isExternal() && !I->hasInternalLinkage())
      DefinedSymbols.insert(I->getName());
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
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
static void
GetAllUndefinedSymbols(Module *M, std::set<std::string> &UndefinedSymbols) {
  std::set<std::string> DefinedSymbols;
  UndefinedSymbols.clear();

  // If the program doesn't define a main, try pulling one in from a .a file.
  // This is needed for programs where the main function is defined in an
  // archive, such f2c'd programs.
  Function *Main = M->getMainFunction();
  if (Main == 0 || Main->isExternal())
    UndefinedSymbols.insert("main");

  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (I->hasName()) {
      if (I->isExternal())
        UndefinedSymbols.insert(I->getName());
      else if (!I->hasInternalLinkage())
        DefinedSymbols.insert(I->getName());
    }
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
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

/// LinkInArchive - opens an archive library and link in all objects which
/// provide symbols that are currently undefined.
///
/// Inputs:
///  Filename - The pathname of the archive.
///
/// Return Value:
///  TRUE  - An error occurred.
///  FALSE - No errors.
bool
Linker::LinkInArchive(const sys::Path &Filename) {

  // Make sure this is an archive file we're dealing with
  if (!Filename.isArchive())
    return error("File '" + Filename.toString() + "' is not an archive.");

  // Open the archive file
  verbose("Linking archive file '" + Filename.toString() + "'");

  // Find all of the symbols currently undefined in the bytecode program.
  // If all the symbols are defined, the program is complete, and there is
  // no reason to link in any archive files.
  std::set<std::string> UndefinedSymbols;
  GetAllUndefinedSymbols(Composite, UndefinedSymbols);

  if (UndefinedSymbols.empty()) {
    verbose("No symbols undefined, skipping library '" +
            Filename.toString() + "'");
    return false;  // No need to link anything in!
  }

  std::string ErrMsg;
  std::auto_ptr<Archive> AutoArch (
    Archive::OpenAndLoadSymbols(Filename,&ErrMsg));

  Archive* arch = AutoArch.get();

  if (!arch)
    return error("Cannot read archive '" + Filename.toString() +
                 "': " + ErrMsg);

  // Save a set of symbols that are not defined by the archive. Since we're
  // entering a loop, there's no point searching for these multiple times. This
  // variable is used to "set_subtract" from the set of undefined symbols.
  std::set<std::string> NotDefinedByArchive;

  // While we are linking in object files, loop.
  while (true) {

    // Find the modules we need to link into the target module
    std::set<ModuleProvider*> Modules;
    if (!arch->findModulesDefiningSymbols(UndefinedSymbols, Modules, &ErrMsg))
      return error("Cannot find symbols in '" + Filename.toString() + 
                   "': " + ErrMsg);

    // If we didn't find any more modules to link this time, we are done
    // searching this archive.
    if (Modules.empty())
      break;

    // Any symbols remaining in UndefinedSymbols after
    // findModulesDefiningSymbols are ones that the archive does not define. So
    // we add them to the NotDefinedByArchive variable now.
    NotDefinedByArchive.insert(UndefinedSymbols.begin(),
        UndefinedSymbols.end());

    // Loop over all the ModuleProviders that we got back from the archive
    for (std::set<ModuleProvider*>::iterator I=Modules.begin(), E=Modules.end();
         I != E; ++I) {

      // Get the module we must link in.
      std::auto_ptr<Module> AutoModule( (*I)->releaseModule() );
      Module* aModule = AutoModule.get();

      verbose("  Linking in module: " + aModule->getModuleIdentifier());

      // Link it in
      if (LinkInModule(aModule))
        return error("Cannot link in module '" +
                     aModule->getModuleIdentifier() + "': " + Error);
    }

    // Get the undefined symbols from the aggregate module. This recomputes the
    // symbols we still need after the new modules have been linked in.
    GetAllUndefinedSymbols(Composite, UndefinedSymbols);

    // At this point we have two sets of undefined symbols: UndefinedSymbols
    // which holds the undefined symbols from all the modules, and
    // NotDefinedByArchive which holds symbols we know the archive doesn't
    // define. There's no point searching for symbols that we won't find in the
    // archive so we subtract these sets.
    set_subtract(UndefinedSymbols, NotDefinedByArchive);

    // If there's no symbols left, no point in continuing to search the
    // archive.
    if (UndefinedSymbols.empty())
      break;
  }

  return false;
}
