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
      if (magic[1] == 'l' && magic[2] == 'v')
        if (magic[3] == 'c')
          return CompressedBytecode_FileType;
        else if (magic[3] == 'm')
          return Bytecode_FileType;
      break;
    case '!':
      if (length >= 8)
        if (memcmp(magic,"!<arch>\n",8) == 0)
          return Archive_FileType;
      break;
      
    case '\177':
      if (magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F')
        return ELF_FileType;
      break;

    case 0xCE:
    case 0xCF:
      // This is complicated by an overlap with Java class files. 
      // See the Mach-O section in /usr/share/file/magic for details.
      if (magic[1] == char(0xFA) && magic[2] == char(0xED) && 
          magic[3] == char(0xFE))
        if (length >= 15)
          if (magic[15] == 1 || magic[15] == 3 || magic[15] == 6 || 
              magic[15] == 9)
            return Mach_O_FileType;
      break;

    case 0xF0: // PowerPC Windows
    case 0x83: // Alpha 32-bit
    case 0x84: // Alpha 64-bit
    case 0x66: // MPS R4000 Windows
    case 0x50: // mc68K
    case 0x4c: // 80386 Windows
      if (magic[1] == 0x01)
        return COFF_FileType;

    case 0x90: // PA-RISC Windows
    case 0x68: // mc68K Windows
      if (magic[1] == 0x02)
        return COFF_FileType;
      break;

    default:
      break;
  }
  return Unknown_FileType;
}

bool
Path::isArchive() const {
  if (canRead())
    return hasMagicNumber("!<arch>\012");
  return false;
}

bool
Path::isDynamicLibrary() const {
  if (canRead()) {
    std::string Magic;
    if (getMagicNumber(Magic, 64))
      switch (IdentifyFileType(Magic.c_str(), Magic.length())) {
        default: return false;
        case ELF_FileType:
        case Mach_O_FileType:
        case COFF_FileType:  return true;
      }
  }
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

DEFINING_FILE_FOR(SystemPath)
