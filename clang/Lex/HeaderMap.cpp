//===--- HeaderMap.cpp - A file that acts like dir of symlinks ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HeaderMap interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/HeaderMap.h"
using namespace clang;

const HeaderMap *HeaderMap::Create(const FileEntry *FE, std::string &ErrorInfo){
  // FIXME: woot!
  return 0; 
}

/// LookupFile - Check to see if the specified relative filename is located in
/// this HeaderMap.  If so, open it and return its FileEntry.
const FileEntry *HeaderMap::LookupFile(const char *FilenameStart,
                                       const char *FilenameEnd,
                                       FileManager &FM) const {
  // FIXME: this needs work.
  return 0;
}
