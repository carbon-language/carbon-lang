//===- lib/Linker/LinkFiles.cpp - Link LLVM bytecode files  ---------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains routines to handle linking together LLVM bytecode files.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Module.h"

using namespace llvm;

/// LinkInFile - opens a bytecode file and links in all objects which
/// provide symbols that are currently undefined.
///
/// Inputs:
///  File - The pathname of the bytecode file.
///
/// Outputs:
///  ErrorMessage - A C++ string detailing what error occurred, if any.
///
/// Return Value:
///  TRUE  - An error occurred.
///  FALSE - No errors.
///
bool 
Linker::LinkInFile(const sys::Path &File)
{
  // Make sure we can at least read the file
  if (!File.readable())
    return error("Cannot find linker input '" + File.toString() + "'");

  // A user may specify an ar archive without -l, perhaps because it
  // is not installed as a library. Detect that and link the library.
  if (File.isArchive()) {
    if (LinkInArchive(File))
      return error("Cannot link archive '" + File.toString() + "'");
  } else if (File.isBytecodeFile()) {
    verbose("Linking bytecode file '" + File.toString() + "'");

    std::auto_ptr<Module> M(LoadObject(File));
    if (M.get() == 0) 
      return error("Cannot load file '" + File.toString() + "'" + Error);
    if (LinkInModule(M.get()))
      return error("Cannot link file '" + File.toString() + "'" + Error);

    verbose("Linked in file '" + File.toString() + "'");
  } else {
    return warning("File of unknown type '" + File.toString() + "' ignored.");
  }
  return false;
}

/// LinkFiles - takes a module and a list of files and links them all together.
/// It locates the file either in the current directory, as its absolute
/// or relative pathname, or as a file somewhere in LLVM_LIB_SEARCH_PATH.
///
/// Inputs:
///  Files      - A vector of sys::Path indicating the LLVM bytecode filenames
///               to be linked.  The names can refer to a mixture of pure LLVM
///               bytecode files and archive (ar) formatted files.
///
/// Return value:
///  FALSE - No errors.
///  TRUE  - Some error occurred.
///
bool 
Linker::LinkInFiles(const std::vector<sys::Path> &Files)
{
  for (unsigned i = 0; i < Files.size(); ++i) {
    if (LinkInFile(Files[i]))
      return true;
  }
  return false;
}
