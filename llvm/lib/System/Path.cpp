//===-- Path.cpp - Implement OS Path Concept --------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system Path concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/Path.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code. 
//===----------------------------------------------------------------------===//

bool
Path::is_file() const {
  return (is_valid() && path[path.length()-1] != '/');
}

bool
Path::is_directory() const {
  return (is_valid() && path[path.length()-1] == '/');
}

}

// Include the truly platform-specific parts of this class.
#include "platform/Path.cpp"

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
