//===- llvm/System/AIX/Path.cpp - AIX Path Implementation -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the AIX-specific implementation of the Path class.
//
//===----------------------------------------------------------------------===//

// Include the generic unix implementation
#include "../Unix/Path.cpp"
#include <sys/stat.h>

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only AIX specific code 
//===          and must not be generic UNIX code (see ../Unix/Path.cpp)
//===----------------------------------------------------------------------===//

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
  strcpy(pathname, "/tmp/llvm_XXXXXX");
  // AIX does not have a mkdtemp(), so we emulate it as follows:
  // mktemp() returns a valid name for a _file_, not a directory, but does not
  // create it.  We assume that it is a valid name for a directory.
  char *TmpName = mktemp(pathname);
  if (!mkdir(TmpName, S_IRWXU))
    ThrowErrno(std::string(TmpName) + ": Can't create temporary directory");
  Path result;
  result.setDirectory(TmpName);
  assert(result.isValid() && "mkdtemp didn't create a valid pathname!");
  return result;
}

std::string
Path::GetDLLSuffix() {
  return "so";
}

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
