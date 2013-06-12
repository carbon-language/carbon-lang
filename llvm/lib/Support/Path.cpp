//===-- Path.cpp - Implement OS Path Concept --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system Path concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Path.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/PathV1.h"
#include <cassert>
#include <cstring>
#include <ostream>
using namespace llvm;
using namespace sys;
namespace {
using support::ulittle32_t;
}

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

bool Path::operator==(const Path &that) const {
  return path == that.path;
}

bool Path::operator<(const Path& that) const {
  return path < that.path;
}

bool
Path::isArchive() const {
  fs::file_magic type;
  if (fs::identify_magic(str(), type))
    return false;
  return type == fs::file_magic::archive;
}

bool
Path::isDynamicLibrary() const {
  fs::file_magic type;
  if (fs::identify_magic(str(), type))
    return false;
  switch (type) {
    default: return false;
    case fs::file_magic::macho_fixed_virtual_memory_shared_lib:
    case fs::file_magic::macho_dynamically_linked_shared_lib:
    case fs::file_magic::macho_dynamically_linked_shared_lib_stub:
    case fs::file_magic::elf_shared_object:
    case fs::file_magic::pecoff_executable:  return true;
  }
}

bool
Path::isObjectFile() const {
  fs::file_magic type;
  if (fs::identify_magic(str(), type) || type == fs::file_magic::unknown)
    return false;
  return true;
}

void
Path::appendSuffix(StringRef suffix) {
  if (!suffix.empty()) {
    path.append(".");
    path.append(suffix);
  }
}

// Include the truly platform-specific parts of this class.
#if defined(LLVM_ON_UNIX)
#include "Unix/Path.inc"
#endif
#if defined(LLVM_ON_WIN32)
#include "Windows/Path.inc"
#endif
