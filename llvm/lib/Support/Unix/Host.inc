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

  if (uname(&info))
    return "";

  return info.release;
}

std::string getTriple(StringRef &TripleString) {
  std::pair<StringRef, StringRef> ArchSplit = TripleString.split('-');
  std::string Arch = ArchSplit.first;
  std::string Triple(Arch);
  Triple += '-';
  Triple += ArchSplit.second;

  // Force i<N>86 to i386.
  if (Triple[0] == 'i' && isdigit(Triple[1]) &&
      Triple[2] == '8' && Triple[3] == '6')
    Triple[1] = '3';

  // On darwin, we want to update the version to match that of the
  // target.
  std::string::size_type DarwinDashIdx = Triple.find("-darwin");
  if (DarwinDashIdx != std::string::npos) {
    Triple.resize(DarwinDashIdx + strlen("-darwin"));
    Triple += getOSVersion();
  }

  return Triple;
}

std::string sys::getDefaultTargetTriple() {
  StringRef TargetTripleString(LLVM_DEFAULT_TARGET_TRIPLE);
  return getTriple(TargetTripleString);
}

std::string sys::getHostTriple() {
  StringRef HostTripleString(LLVM_HOST_TRIPLE);
  return getTriple(HostTripleString);
}
