 //===- llvm/Support/Unix/Host.inc -------------------------------*- C++ -*-===//
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

#include "llvm/Config/config.h"
#include "llvm/ADT/StringRef.h"
#include "Unix.h"
#include <sys/utsname.h>
#include <cctype>
#include <string>
#include <cstdlib> // ::getenv

using namespace llvm;

static std::string getOSVersion() {
  struct utsname info;

#ifdef __APPLE__
  // Recognize UNAME_RELEASE environment variable to match Darwin uname.
  const char *UnameOverride = ::getenv("UNAME_RELEASE");
  if (UnameOverride && UnameOverride[0] != '\0')
    return UnameOverride;
#endif // __APPLE__

  if (uname(&info))
    return "";

  return info.release;
}

std::string sys::getHostTriple() {
  // FIXME: Derive directly instead of relying on the autoconf generated
  // variable.

  StringRef HostTripleString(LLVM_HOSTTRIPLE);
  std::pair<StringRef, StringRef> ArchSplit = HostTripleString.split('-');

  // Normalize the arch, since the host triple may not actually match the host.
  std::string Arch = ArchSplit.first;

  std::string Triple(Arch);
  Triple += '-';
  Triple += ArchSplit.second;

  // Force i<N>86 to i386.
  if (Triple[0] == 'i' && isdigit(Triple[1]) &&
      Triple[2] == '8' && Triple[3] == '6')
    Triple[1] = '3';

  // On darwin, we want to update the version to match that of the
  // host.
  std::string::size_type DarwinDashIdx = Triple.find("-darwin");
  if (DarwinDashIdx != std::string::npos) {
    Triple.resize(DarwinDashIdx + strlen("-darwin"));
    Triple += getOSVersion();
  }

  return Triple;
}
