//===- Win32/MappedFile.cpp - Win32 MappedFile Implementation ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the MappedFile
// concept.
//
//===----------------------------------------------------------------------===//

#include "Win32.h"

void MappedFile::initialize() {
}

void MappedFile::terminate() {
}

void MappedFile::unmap() {
}

void* MappedFile::map() {
  static char junk[4096];
  return junk;
}

size_t MappedFile::size() {
  return 4096;
}

void MappedFile::size(size_t new_size) {
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
