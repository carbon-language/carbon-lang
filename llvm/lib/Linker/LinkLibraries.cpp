//===- lib/Linker/LinkLibraries.cpp - Link LLVM libraries -----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains routines to handle finding libraries and linking them in. 
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Module.h"

using namespace llvm;

/// LinkInLibrary - links one library into the HeadModule
bool 
Linker::LinkInLibrary(const std::string& Lib) 
{
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
bool 
Linker::LinkInLibraries(const std::vector<std::string> &Libraries) {

  // Process the set of libraries we've been provided
  for (unsigned i = 0; i < Libraries.size(); ++i) {
    if (LinkInLibrary(Libraries[i]))
      return true;
  }

  // At this point we have processed all the libraries provided to us. Since
  // we have an aggregated module at this point, the dependent libraries in
  // that module should also be aggregated with duplicates eliminated. This is
  // now the time to process the dependent libraries to resolve any remaining
  // symbols.
  const Module::LibraryListType& DepLibs = Composite->getLibraries();
  for (Module::LibraryListType::const_iterator I = DepLibs.begin(), 
      E = DepLibs.end(); I != E; ++I) {
    if (LinkInLibrary(*I)) 
      return true;
  }
  return false;
}
