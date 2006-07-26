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
#include "llvm/Config/config.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/MappedFile.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Win32/MappedFile.inc"
#endif

DEFINING_FILE_FOR(SystemMappedFile)
