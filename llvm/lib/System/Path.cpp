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
#include <cassert>

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code. 
//===----------------------------------------------------------------------===//

LLVMFileType 
sys::IdentifyFileType(const char*magic, unsigned length) {
  assert(magic && "Invalid magic number string");
  assert(length >=4 && "Invalid magic number length");
  switch (magic[0]) {
    case 'l':
      if (magic[1] == 'l' && magic[2] == 'v') {
        if (magic[3] == 'c')
          return CompressedBytecodeFileType;
        else if (magic[3] == 'm')
          return BytecodeFileType;
      }
      break;

    case '!':
      if (length >= 8) {
        if (memcmp(magic,"!<arch>\n",8) == 0)
          return ArchiveFileType;
      }
      break;

    default:
      break;
  }
  return UnknownFileType;
}

}

// Include the truly platform-specific parts of this class.
#include "platform/Path.cpp"

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
