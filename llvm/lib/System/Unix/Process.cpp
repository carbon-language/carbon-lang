//===- Unix/Process.cpp - Unix Process Implementation --------- -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the generic Unix implementation of the Process class.
//
//===----------------------------------------------------------------------===//

#include <unistd.h>

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

namespace llvm {
using namespace sys;

unsigned 
Process::GetPageSize() {
  // NOTE: The getpagesize function doesn't exist in POSIX 1003.1 and is 
  // "deprecated" in SUSv2. Platforms including this implementation should
  // consider sysconf(_SC_PAGE_SIZE) if its available. 
  static const int page_size = getpagesize();
  return static_cast<unsigned>(page_size);
}

}
// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
