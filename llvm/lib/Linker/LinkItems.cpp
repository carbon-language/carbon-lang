//===- lib/Linker/LinkItems.cpp - Link LLVM objects and libraries ---------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains routines to handle linking together LLVM bytecode files,
// and to handle annoying things like static libraries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Module.h"

using namespace llvm;

// LinkItems - preserve link order for an arbitrary set of linkage items.
bool
Linker::LinkInItems(const ItemList& Items) {
  // For each linkage item ...
  for (ItemList::const_iterator I = Items.begin(), E = Items.end(); 
       I != E; ++I) {
    if (I->second) {
      // Link in the library suggested.
      if (LinkInLibrary(I->first))
        return true;
    } else {
      if (LinkInFile(sys::Path(I->first)))
        return true;
    }
  }

  // At this point we have processed all the link items provided to us. Since
  // we have an aggregated module at this point, the dependent libraries in
  // that module should also be aggregated with duplicates eliminated. This is
  // now the time to process the dependent libraries to resolve any remaining
  // symbols.
  const Module::LibraryListType& DepLibs = Composite->getLibraries();
  for (Module::LibraryListType::const_iterator I = DepLibs.begin(), 
      E = DepLibs.end(); I != E; ++I) {
    if(LinkInLibrary(*I))
      return true;
  }

  return false;
}


/// LinkInLibrary - links one library into the HeadModule.
///
bool Linker::LinkInLibrary(const std::string& Lib) {
  // Determine where this library lives.
  sys::Path Pathname = FindLib(Lib);
  if (Pathname.isEmpty())
    return warning("Cannot find library '" + Lib + "'");

  // If its an archive, try to link it in
  if (Pathname.isArchive()) {
    if (LinkInArchive(Pathname))
      return error("Cannot link archive '" + Pathname.toString() + "'");
  } else if (Pathname.isBytecodeFile()) {
    // LLVM ".so" file.
    if (LinkInFile(Pathname))
      return error("Cannot link file '" + Pathname.toString() + "'");

  } else if (Pathname.isDynamicLibrary()) {
    return warning("Library '" + Lib + "' is a native dynamic library.");
  } else {
    return warning("Supposed library '" + Lib + "' isn't a library.");
  }
  return false;
}

/// LinkLibraries - takes the specified library files and links them into the
/// main bytecode object file.
///
/// Inputs:
///  Libraries  - The list of libraries to link into the module.
///
/// Return value:
///  FALSE - No error.
///  TRUE  - Error.
///
bool Linker::LinkInLibraries(const std::vector<std::string> &Libraries) {

  // Process the set of libraries we've been provided.
  for (unsigned i = 0; i < Libraries.size(); ++i)
    if (LinkInLibrary(Libraries[i]))
      return true;

  // At this point we have processed all the libraries provided to us. Since
  // we have an aggregated module at this point, the dependent libraries in
  // that module should also be aggregated with duplicates eliminated. This is
  // now the time to process the dependent libraries to resolve any remaining
  // symbols.
  const Module::LibraryListType& DepLibs = Composite->getLibraries();
  for (Module::LibraryListType::const_iterator I = DepLibs.begin(), 
      E = DepLibs.end(); I != E; ++I)
    if (LinkInLibrary(*I)) 
      return true;

  return false;
}
