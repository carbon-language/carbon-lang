//===- llvm/Linker.h - Module Linker Interface ------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the interface to the module/file/archive linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKER_H
#define LLVM_LINKER_H

#include "llvm/Support/CommandLine.h"
#include <string>
#include <vector>
#include <set>

namespace llvm {

class Module;

/// This type is used to pass the linkage items (libraries and files) to
/// the LinkItems function. It is composed of string/bool pairs. The string
/// provides the name of the file or library (as with the -l option). The bool
/// should be true for libraries, false for files, signifying "isLibrary".
/// @brief A list of string/bool pairs
typedef std::vector<std::pair<std::string,bool> > LinkItemList;

/// This function can be used to link a set of linkage items into a module. A
/// linkage item is one of the three things identified by the LinkItemKind
/// enumeration. This function allows linking to preserve the order of 
/// specification associated with a command line, or for other purposes. Each
/// item will be linked in turn as it occurs in \p Items. Note that library
/// path items will only be in effect after they have been processed.
/// @returns The aggregated/linked Module.
/// @throws nothing
Module* LinkItems (
  const char * progname,   ///< Name of the program being linked (for output)
  const LinkItemList& Items, // Set of libraries/files to link in
  const std::vector<std::string>& LibPaths, // Paths to search for libraries
  bool Verbose, ///< Link verbosely, indicating each action
  bool Native ///< Linking is for a native executable
);

/// This function provides some utility for tools that need to build the list
/// of link items from a triplet of command line options: Files, Libraries, and
/// LibraryPaths. The command line ordering is preserved by this function even
/// though the options are split into three separate cl::list<std::string>. The
/// resulting \p OutList is suitable for use with LinkItems.
/// @see LinkItems
/// @throws nothing
void BuildLinkItems(
  LinkItemList& OutList,
  const cl::list<std::string>& Files, ///< List of files to put in list
  const cl::list<std::string>& Libs   ///< List of libraries to put in list
);

/// This is the heart of the linker. The \p Src module is linked into the \p
/// Dest module. If an error occurs, true is returned, otherwise false. If \p
/// ErrorMsg is not null and an error occurs, \p *ErrorMsg will be set to a
/// readable string that indicates the nature of the error.  Note that this can
/// destroy the Src module in arbitrary ways.
///
/// @returns true if there's an error
/// @brief Link two modules together
bool LinkModules(
  Module* Dest,          ///< Module into which \p Src is linked
  Module* Src,     ///< Module linked into \p Dest
  std::string* ErrorMsg  ///< Optional error message string
);

/// This function links the bytecode \p Files into the \p HeadModule. Note that
/// this does not do any linking of unresolved symbols. The \p Files are all
/// completely linked into \p HeadModule regardless of unresolved symbols. This
/// function just loads each bytecode file and calls LinkModules on them. 
/// @returns true if an error occurs, false otherwise
bool LinkFiles (
  const char * progname, ///< Name of the program being linked (for output)
  Module * HeadModule,   ///< Main (resulting) module to be linked into
  const std::vector<std::string> & Files, ///< Files to link in
  bool Verbose ///< Link verbosely, indicating each action
);

/// This function links one archive, \p Filename,  that contains bytecode into
/// \p HeadModule.  If an error occurs, true is returned, otherwise false. If
/// \p ErrorMsg is not null and an error occurs, \p *ErrorMsg will be set to a
/// readable string that indicates the nature of the error.
/// @returns true if there's an error
/// @brief Link in one archive.
bool LinkInArchive( 
  Module* HeadModule,          ///< Main (resulting) module to be linked into
  const std::string& Filename, ///< Filename of the archive to link
  std::string* ErrorMsg,       ///< Error message if an error occurs.
  bool Verbose                 ///< Link verbosely, indicating each action
);

/// This function provides the ability to handle the -L and -l options on a 
/// linker's command line. It will link into \p HeadModule any modules found in
/// the \p Libraries (which might be found in the \p LibPaths). 
/// @brief Link libraries into a module
void LinkLibraries (
  const char * progname,   ///< Name of the program being linked (for output)
  Module* HeadModule,      ///< Main (resulting) module to be linked into
  const std::vector<std::string> & Libraries, ///< Set of libraries to link in
  const std::vector<std::string> & LibPaths,  ///< Set of library paths
  bool Verbose, ///< Link verbosely, indicating each action
  bool Native ///< Linking is for a native executable
);

/// This function looks at Module \p M and returns a set of strings, 
/// \p DefinedSymbols, that is the publicly visible defined symbols in 
/// module \p M.
void GetAllDefinedSymbols (Module *M, std::set<std::string> &DefinedSymbols);

/// This function looks at Module \p M and returns a set of strings, 
/// \p UnefinedSymbols, that is the publicly visible undefined symbols in 
/// module \p M.
void GetAllUndefinedSymbols(Module *M, std::set<std::string> &UndefinedSymbols);

/// This function looks through a set of \p Paths to find a library with the
/// name \p Filename. If \p SharedObjectOnly is true, it only finds a match
/// if the file is a shared library.
std::string FindLib(const std::string &Filename,
                    const std::vector<std::string> &Paths,
                    bool SharedObjectOnly = false);
  
} // End llvm namespace

#endif
