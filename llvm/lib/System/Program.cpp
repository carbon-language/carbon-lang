//===-- Program.cpp - Implement OS Program Concept --------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system Program concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/Program.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code. 
//===----------------------------------------------------------------------===//

}

// Include the platform-specific parts of this class.
#include "platform/Program.cpp"

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
