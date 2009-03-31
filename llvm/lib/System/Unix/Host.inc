 //===- llvm/System/Unix/Host.inc -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the UNIX Host support.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include <llvm/Config/config.h>
#include "Unix.h"
#include <sys/utsname.h>
#include <string>

using namespace llvm;

static std::string getOSVersion() {
  struct utsname info;

  if (uname(&info))
    return "";

  return info.release;
}

std::string sys::getHostTriple() {
  // FIXME: Derive more directly instead of relying on the autoconf
  // generated variable.

  std::string Triple = LLVM_HOSTTRIPLE;

  // Force i<N>86 to i386.
  if (Triple[0] == 'i' && isdigit(Triple[1]) && 
      Triple[2] == '8' && Triple[3] == '6')
    Triple[1] = '3';

  // On darwin, we want to update the version to match that of the
  // host.    
  std::string::size_type DarwinDashIdx = Triple.find("-darwin");
  if (DarwinDashIdx != std::string::npos) {
    Triple.resize(DarwinDashIdx + strlen("-darwin"));
    
    // Only add the major part of the os version.
    std::string Version = getOSVersion();
    Triple += Version.substr(0, Version.find('.'));
  }

  return Triple;
}
