//===- llvm/System/Darwin/Path.cpp - Linux Path Implementation --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Darwin specific implementation of the Path class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Darwin specific code 
//===          and must not be generic UNIX code (see ../Unix/Path.cpp)
//===----------------------------------------------------------------------===//

// Include the generic unix implementation
#include "../Unix/Path.cpp"

namespace llvm {
using namespace sys;

bool 
Path::isValid() const {
  if (path.empty()) 
    return false;
  if (path.length() >= MAXPATHLEN)
    return false;
  return true;
}

Path
Path::GetTemporaryDirectory() {
  char pathname[MAXPATHLEN];
  strcpy(pathname,"/tmp/llvm_XXXXXX");
  if (0 == mkdtemp(pathname))
    ThrowErrno(std::string(pathname) + ": Can't create temporary directory");
  Path result;
  result.setDirectory(pathname);
  assert(result.isValid() && "mkdtemp didn't create a valid pathname!");
  return result;
}

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
