//===--- extra/modularize/ModularizeUtilities.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a class for loading and validating a module map or
// header list by checking that all headers in the corresponding directories
// are accounted for.
//
//===----------------------------------------------------------------------===//

#include "ModularizeUtilities.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace Modularize;

// ModularizeUtilities class implementation.

// Constructor.
ModularizeUtilities::ModularizeUtilities(std::vector<std::string> &InputPaths,
                                         llvm::StringRef Prefix)
    : InputFilePaths(InputPaths),
      HeaderPrefix(Prefix) {}

// Create instance of ModularizeUtilities, to simplify setting up
// subordinate objects.
ModularizeUtilities *ModularizeUtilities::createModularizeUtilities(
    std::vector<std::string> &InputPaths, llvm::StringRef Prefix) {

  return new ModularizeUtilities(InputPaths, Prefix);
}

// Load all header lists and dependencies.
std::error_code ModularizeUtilities::loadAllHeaderListsAndDependencies() {
  typedef std::vector<std::string>::iterator Iter;
  for (Iter I = InputFilePaths.begin(), E = InputFilePaths.end(); I != E; ++I) {
    if (std::error_code EC = loadSingleHeaderListsAndDependencies(*I)) {
      errs() << "modularize: error: Unable to get header list '" << *I
        << "': " << EC.message() << '\n';
      return EC;
    }
  }
  return std::error_code();
}
  
// Load single header list and dependencies.
std::error_code ModularizeUtilities::loadSingleHeaderListsAndDependencies(
    llvm::StringRef InputPath) {

  // By default, use the path component of the list file name.
  SmallString<256> HeaderDirectory(InputPath);
  llvm::sys::path::remove_filename(HeaderDirectory);
  SmallString<256> CurrentDirectory;
  llvm::sys::fs::current_path(CurrentDirectory);

  // Get the prefix if we have one.
  if (HeaderPrefix.size() != 0)
    HeaderDirectory = HeaderPrefix;

  // Read the header list file into a buffer.
  ErrorOr<std::unique_ptr<MemoryBuffer>> listBuffer =
    MemoryBuffer::getFile(InputPath);
  if (std::error_code EC = listBuffer.getError())
    return EC;

  // Parse the header list into strings.
  SmallVector<StringRef, 32> Strings;
  listBuffer.get()->getBuffer().split(Strings, "\n", -1, false);

  // Collect the header file names from the string list.
  for (SmallVectorImpl<StringRef>::iterator I = Strings.begin(),
    E = Strings.end();
    I != E; ++I) {
    StringRef Line = I->trim();
    // Ignore comments and empty lines.
    if (Line.empty() || (Line[0] == '#'))
      continue;
    std::pair<StringRef, StringRef> TargetAndDependents = Line.split(':');
    SmallString<256> HeaderFileName;
    // Prepend header file name prefix if it's not absolute.
    if (llvm::sys::path::is_absolute(TargetAndDependents.first))
      llvm::sys::path::native(TargetAndDependents.first, HeaderFileName);
    else {
      if (HeaderDirectory.size() != 0)
        HeaderFileName = HeaderDirectory;
      else
        HeaderFileName = CurrentDirectory;
      llvm::sys::path::append(HeaderFileName, TargetAndDependents.first);
      llvm::sys::path::native(HeaderFileName);
    }
    // Handle optional dependencies.
    DependentsVector Dependents;
    SmallVector<StringRef, 4> DependentsList;
    TargetAndDependents.second.split(DependentsList, " ", -1, false);
    int Count = DependentsList.size();
    for (int Index = 0; Index < Count; ++Index) {
      SmallString<256> Dependent;
      if (llvm::sys::path::is_absolute(DependentsList[Index]))
        Dependent = DependentsList[Index];
      else {
        if (HeaderDirectory.size() != 0)
          Dependent = HeaderDirectory;
        else
          Dependent = CurrentDirectory;
        llvm::sys::path::append(Dependent, DependentsList[Index]);
      }
      llvm::sys::path::native(Dependent);
      Dependents.push_back(Dependent.str());
    }
    // Save the resulting header file path and dependencies.
    HeaderFileNames.push_back(HeaderFileName.str());
    Dependencies[HeaderFileName.str()] = Dependents;
  }
  return std::error_code();
}
