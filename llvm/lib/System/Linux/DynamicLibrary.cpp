//===- Linux/DynamicLibrary.cpp - Linux Dynamic Library ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Linux version of DynamicLibrary
//
//===----------------------------------------------------------------------===//

// Include the generic unix implementation
#include "../Unix/DynamicLibrary.cpp"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Linux specific code 
//===          and must not be generic UNIX code (see ../Unix/Memory.cpp)
//===----------------------------------------------------------------------===//

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
