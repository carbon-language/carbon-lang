//===- Win32/Path.cpp - Win32 Path Implementation ---------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Path class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code 
//===          and must not be generic UNIX code (see ../Unix/Path.cpp)
//===----------------------------------------------------------------------===//

// Include the generic Unix implementation
#include "../Unix/Path.cpp"

namespace llvm {
using namespace sys;


std::string
Path::GetDLLSuffix() {
  return "dll";
}

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
