//===- Support/FileUtilities.cpp - File System Utilities ------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a family of utility functions which are useful for doing
// various things with files.
//
//===----------------------------------------------------------------------===//

#include "FDHandle.h"
#include <unistd.h>

using namespace llvm;

//===----------------------------------------------------------------------===//
// FDHandle class implementation
//

FDHandle::~FDHandle() throw() {
  if (FD != -1) 
    ::close(FD);
}

FDHandle &FDHandle::operator=(int fd) throw() {
  if (FD != -1) 
    ::close(FD);
  FD = fd;
  return *this;
}
