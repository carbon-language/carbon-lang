//===- SysConfig.cpp - System Configuration Support ---------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencerand is distributed under the University 
// of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the operating system independent SysConfig abstraction.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/SysConfig.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code. 
//===----------------------------------------------------------------------===//

}

// Include the platform-specific parts of this class.
#include "platform/SysConfig.cpp"

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
