//===- SysConfig.cpp - Generic UNIX System Configuration --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some functions for managing system configuration on Unix
// systems.
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include <sys/resource.h>

namespace llvm {

// Some LLVM programs such as bugpoint produce core files as a normal part of
// their operation. To prevent the disk from filling up, this configuration item
// does what's necessary to prevent their generation.
void sys::PreventCoreFiles() {
  struct rlimit rlim;
  rlim.rlim_cur = rlim.rlim_max = 0;
  int res = setrlimit(RLIMIT_CORE, &rlim);
  if (res != 0)
    ThrowErrno("Can't prevent core file generation");
}

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
