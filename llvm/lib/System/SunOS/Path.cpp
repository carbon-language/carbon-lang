//===- llvm/System/SunOS/Path.cpp - SunOS Path Implementation ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the SunOS specific implementation of the Path class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only SunOS specific code 
//===          and must not be generic UNIX code (see ../Unix/Path.cpp)
//===----------------------------------------------------------------------===//

// Include the generic Unix implementation
#include "../Unix/Path.cpp"

namespace llvm {
using namespace sys;

bool 
Path::is_valid() const {
  if (path.empty()) 
    return false;
  char pathname[MAXPATHLEN];
  if (0 == realpath(path.c_str(), pathname))
    if (errno != EACCES && errno != EIO && errno != ENOENT && errno != ENOTDIR)
      return false;
  return true;
}

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
