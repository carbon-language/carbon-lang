//===- MappedFile.cpp - MappedFile Support ----------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the mapped file concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/MappedFile.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code. 
//===----------------------------------------------------------------------===//

}

// Include the platform-specific parts of this class.
#include "platform/MappedFile.cpp"

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
