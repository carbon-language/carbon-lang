//===- AIX/DynamicLibrary.cpp - AIX Dynamic Library -------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the AIX version of DynamicLibrary
//
//===----------------------------------------------------------------------===//

// Include the generic unix implementation
#include "../Unix/DynamicLibrary.cpp"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only AIX specific code 
//===          and must not be generic UNIX code (see ../Unix/Memory.cpp)
//===----------------------------------------------------------------------===//

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
