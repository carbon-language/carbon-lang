//===- Cygwin/Signals.cpp - Cygwin Signals Implementation -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Cygwin specific implementation of the Signals class.
//
//===----------------------------------------------------------------------===//

// Include the generic unix implementation
#include "Unix/Signals.cpp"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Cygwin specific code 
//===          and must not be generic UNIX code (see ../Unix/Signals.cpp)
//===----------------------------------------------------------------------===//

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
