//===- llvm/System/Unix/Unix.h - Common Unix Include File -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines things specific to Unix implementations.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on all UNIX variants.
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"     // Get autoconf configuration settings
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <sys/types.h>
#include <sys/param.h>
#include <assert.h>
#include <string>

inline void ThrowErrno(const std::string& prefix) {
    char buffer[MAXPATHLEN];
    throw prefix + ": " + strerror(errno);
}
