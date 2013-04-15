//===-- Core/IncludeExcludeInfo.cpp - IncludeExclude class impl -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implemention of the IncludeExcludeInfo class
/// to handle the include and exclude command line options.
///
//===----------------------------------------------------------------------===//

#include "IncludeExcludeInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;

namespace {
/// \brief Helper function to determine whether a file has the same path
/// prefix as \a Path.
///
/// \a Path must be an absolute path.
bool fileHasPathPrefix(StringRef File, StringRef Path) {
  // Converts File to its absolute path.
  SmallString<64> AbsoluteFile = File;
  sys::fs::make_absolute(AbsoluteFile);

  // Convert path strings to sys::path to iterate over each of its directories.
  sys::path::const_iterator FileI = sys::path::begin(AbsoluteFile),
                            FileE = sys::path::end(AbsoluteFile),
                            PathI = sys::path::begin(Path),
                            PathE = sys::path::end(Path);
  while (FileI != FileE && PathI != PathE) {
    // If the strings aren't equal then the two paths aren't contained within
    // each other.
    if (!FileI->equals(*PathI))
      return false;
    ++FileI;
    ++PathI;
  }
  return true;
}

/// \brief Helper function to parse a string of comma seperated paths into
/// the vector.
void parseCLInput(StringRef Line, std::vector<std::string> &List) {
  SmallVector<StringRef, 32> Tokens;
  Line.split(Tokens, ",", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (SmallVectorImpl<StringRef>::iterator I = Tokens.begin(),
                                            E = Tokens.end();
       I != E; ++I) {
    // Convert each path to its absolute path.
    SmallString<64> AbsolutePath = *I;
    sys::fs::make_absolute(AbsolutePath);
    List.push_back(std::string(AbsolutePath.str()));
  }
}
} // end anonymous namespace

IncludeExcludeInfo::IncludeExcludeInfo(StringRef Include, StringRef Exclude) {
  parseCLInput(Include, IncludeList);
  parseCLInput(Exclude, ExcludeList);
}

bool IncludeExcludeInfo::isFileIncluded(StringRef FilePath) {
  bool InIncludeList = false;

  for (std::vector<std::string>::iterator I = IncludeList.begin(),
                                          E = IncludeList.end();
       I != E; ++I)
    if ((InIncludeList = fileHasPathPrefix(FilePath, *I)))
      break;
  // If file is not in the list of included paths then it is not necessary
  // to check the excluded path list.
  if (!InIncludeList)
    return false;

  for (std::vector<std::string>::iterator I = ExcludeList.begin(),
                                          E = ExcludeList.end();
       I != E; ++I)
    if (fileHasPathPrefix(FilePath, *I))
      return false;

  // If the file is in the included list but not in the excluded list, then
  // it is safe to transform.
  return true;
}
