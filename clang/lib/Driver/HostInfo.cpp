//===--- HostInfo.cpp - Host specific information -----------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/HostInfo.h"
 
using namespace clang::driver;

HostInfo::HostInfo(const char *_Arch, const char *_Platform,
                   const char *_OS) 
  : Arch(_Arch), Platform(_Platform), OS(_OS) 
{
  
}

HostInfo::~HostInfo() {
}

// Darwin Host Info

DarwinHostInfo::DarwinHostInfo(const char *Arch, const char *Platform,
                               const char *OS) 
  : HostInfo(Arch, Platform, OS) {

  // FIXME: How to deal with errors?
  
  // We can only call 4.2.1 for now.
  GCCVersion[0] = 4;
  GCCVersion[1] = 2;
  GCCVersion[2] = 1;
}

bool DarwinHostInfo::useDriverDriver() const { 
  return true;
}

ToolChain *DarwinHostInfo::getToolChain(const ArgList &Args, 
                                        const char *ArchName) const {
  return 0;
}

// Unknown Host Info

UnknownHostInfo::UnknownHostInfo(const char *Arch, const char *Platform,
                               const char *OS) 
  : HostInfo(Arch, Platform, OS) {
}

bool UnknownHostInfo::useDriverDriver() const { 
  return false;
}

ToolChain *UnknownHostInfo::getToolChain(const ArgList &Args, 
                                         const char *ArchName) const {
  return 0;
}
