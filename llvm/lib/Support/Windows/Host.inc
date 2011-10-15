//===- llvm/Support/Win32/Host.inc -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 Host support.
//
//===----------------------------------------------------------------------===//

#include "Windows.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

std::string sys::getHostTriple() {
  // Get the execution environment, not the native environment.
  SYSTEM_INFO info;
  ::GetSystemInfo(&info);

  Twine ret;
  switch (info.wProcessorArchitecture) {
  // If we don't know what the processor architecture is, or it is not one
  // we currently support, then we should fall back on something reasonable.
  case PROCESSOR_ARCHITECTURE_IA64:
  default: return LLVM_HOSTTRIPLE;

  case PROCESSOR_ARCHITECTURE_INTEL:
    // We need to figure out what kind of x86 it is (possible values are
    // i386 through i986).
    ret = Twine("i").concat(Twine(info.wProcessorLevel)).concat("86");
    break;
  case PROCESSOR_ARCHITECTURE_AMD64:
    ret = "amd64";
	break;
  case PROCESSOR_ARCHITECTURE_MIPS:
    ret = "mips";
    break;
  case PROCESSOR_ARCHITECTURE_ARM:
    ret = "arm";
    break;
  case PROCESSOR_ARCHITECTURE_PPC:
    ret = "ppc";
    break;
  case PROCESSOR_ARCHITECTURE_ALPHA:
    ret = "alpha";
    break;
  }

  // Since we're on Windows, we're always on pc-win32.
  return ret.concat("-pc-win32").str();
}
