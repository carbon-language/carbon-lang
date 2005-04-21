//===- Unix/SUS/Process.cpp - Linux Process Implementation ---- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Linux specific implementation of the Process class.
//
//===----------------------------------------------------------------------===//

#include <unistd.h>

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only code specific to the
//===          SUS (Single Unix Specification).
//===----------------------------------------------------------------------===//

namespace llvm {
using namespace sys;

unsigned
Process::GetPageSize() {
  static const long page_size = sysconf(_SC_PAGE_SIZE);
  return static_cast<unsigned>(page_size);
}

}
// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
