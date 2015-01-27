//===- llvm-pdbdump.h - Common includes for llvm-pdbdump --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Common defines and header includes for all llvm-pdbdump.  The definitions
// here configure the necessary #defines and include system headers in the
// proper order for using DIA.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_LLVMPDBDUMP_H
#define LLVM_TOOLS_LLVMPDBDUMP_LLVMPDBDUMP_H

#define NTDDI_VERSION NTDDI_VISTA
#define _WIN32_WINNT _WIN32_WINNT_VISTA
#define WINVER _WIN32_WINNT_VISTA
#ifndef NOMINMAX
#define NOMINMAX
#endif

// atlbase.h has to come before windows.h
#include <atlbase.h>
#include <windows.h>

// DIA headers must come after windows headers.
#include <cvconst.h>
#include <dia2.h>

#endif