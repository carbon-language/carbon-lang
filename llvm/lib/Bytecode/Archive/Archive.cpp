//===-- Archive.cpp - Generic LLVM archive functions ------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Builds up standard unix archive files (.a) containing LLVM bytecode.
//
//===----------------------------------------------------------------------===//

#include "ArchiveInternals.h"

using namespace llvm;

Archive::Archive() {
}

Archive::~Archive() {
}

// vim: sw=2 ai
