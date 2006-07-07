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
#include "llvm/Config/config.h"
#include <cassert>
#include <ostream>
using namespace llvm;
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

std::ostream& llvm::operator<<(std::ostream &strm, const sys::Path &aPath) {
  strm << aPath.toString();
  return strm;
}

Path
Path::GetLLVMConfigDir() {
  Path result;
#ifdef LLVM_ETCDIR
  if (result.set(LLVM_ETCDIR))
    return result;
#endif
  return GetLLVMDefaultConfigDir();
}

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

bool
Path::isArchive() const {
  if (canRead())
    return hasMagicNumber("!<arch>\012");
  return false;
}

bool
Path::isDynamicLibrary() const {
  if (canRead())
    return hasMagicNumber("\177ELF");
  return false;
}

Path
Path::FindLibrary(std::string& name) {
  std::vector<sys::Path> LibPaths;
  GetSystemLibraryPaths(LibPaths);
  for (unsigned i = 0; i < LibPaths.size(); ++i) {
    sys::Path FullPath(LibPaths[i]);
    FullPath.appendComponent("lib" + name + LTDL_SHLIB_EXT);
    if (FullPath.isDynamicLibrary())
      return FullPath;
    FullPath.eraseSuffix();
    FullPath.appendSuffix("a");
    if (FullPath.isArchive())
      return FullPath;
  }
  return sys::Path();
}

std::string Path::GetDLLSuffix() {
  return LTDL_SHLIB_EXT;
}

// Include the truly platform-specific parts of this class.
#if defined(LLVM_ON_UNIX)
#include "Unix/Path.inc"
#endif
#if defined(LLVM_ON_WIN32)
#include "Win32/Path.inc"
#endif
