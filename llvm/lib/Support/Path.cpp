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

StringRef Path::GetDLLSuffix() {
  return &(LTDL_SHLIB_EXT[1]);
}

void
Path::appendSuffix(StringRef suffix) {
  if (!suffix.empty()) {
    path.append(".");
    path.append(suffix);
  }
}

bool
Path::isBitcodeFile() const {
  fs::file_magic type;
  if (fs::identify_magic(str(), type))
    return false;
  return type == fs::file_magic::bitcode;
}

bool Path::hasMagicNumber(StringRef Magic) const {
  std::string actualMagic;
  if (getMagicNumber(actualMagic, static_cast<unsigned>(Magic.size())))
    return Magic == actualMagic;
  return false;
}

static StringRef getDirnameCharSep(StringRef path, const char *Sep) {
  assert(Sep[0] != '\0' && Sep[1] == '\0' &&
         "Sep must be a 1-character string literal.");
  if (path.empty())
    return ".";

  // If the path is all slashes, return a single slash.
  // Otherwise, remove all trailing slashes.

  signed pos = static_cast<signed>(path.size()) - 1;

  while (pos >= 0 && path[pos] == Sep[0])
    --pos;

  if (pos < 0)
    return path[0] == Sep[0] ? Sep : ".";

  // Any slashes left?
  signed i = 0;

  while (i < pos && path[i] != Sep[0])
    ++i;

  if (i == pos) // No slashes?  Return "."
    return ".";

  // There is at least one slash left.  Remove all trailing non-slashes.
  while (pos >= 0 && path[pos] != Sep[0])
    --pos;

  // Remove any trailing slashes.
  while (pos >= 0 && path[pos] == Sep[0])
    --pos;

  if (pos < 0)
    return path[0] == Sep[0] ? Sep : ".";

  return path.substr(0, pos+1);
}

// Include the truly platform-specific parts of this class.
#if defined(LLVM_ON_UNIX)
#include "Unix/Path.inc"
#endif
#if defined(LLVM_ON_WIN32)
#include "Windows/Path.inc"
#endif
