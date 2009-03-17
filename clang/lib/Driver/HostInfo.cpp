//===--- HostInfo.cpp - Host specific information -----------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/HostInfo.h"

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"

#include <cassert>
 
using namespace clang::driver;

HostInfo::HostInfo(const Driver &D, const char *_Arch, const char *_Platform,
                   const char *_OS) 
  : TheDriver(D), Arch(_Arch), Platform(_Platform), OS(_OS) 
{
  
}

HostInfo::~HostInfo() {
}

namespace VISIBILITY_HIDDEN {

// Darwin Host Info

/// DarwinHostInfo - Darwin host information implementation.
class DarwinHostInfo : public HostInfo {
  /// Darwin version of host.
  unsigned DarwinVersion[3];

  /// GCC version to use on this host.
  unsigned GCCVersion[3];

  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  DarwinHostInfo(const Driver &D, const char *Arch, 
                 const char *Platform, const char *OS);

  virtual bool useDriverDriver() const;

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

DarwinHostInfo::DarwinHostInfo(const Driver &D, const char *_Arch, 
                               const char *_Platform, const char *_OS) 
  : HostInfo(D, _Arch, _Platform, _OS) {
  
  assert((getArchName() == "i386" || getArchName() == "x86_64" || 
          getArchName() == "ppc" || getArchName() == "ppc64") &&
         "Unknown Darwin arch.");

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
  if (!ArchName) {
    ArchName = getArchName().c_str();

    // If no arch name is specified, infer it from the host and
    // -m32/-m64.
    if (Arg *A = Args.getLastArg(options::OPT_m32, options::OPT_m64)) {
      if (getArchName() == "i386" || getArchName() == "x86_64") {
        ArchName = 
          (A->getOption().getId() == options::OPT_m32) ? "i386" : "x86_64";
      } else if (getArchName() == "ppc" || getArchName() == "ppc64") {
        ArchName = 
          (A->getOption().getId() == options::OPT_m32) ? "ppc" : "ppc64";
      }
    } 
  }

  ToolChain *&TC = ToolChains[ArchName];
  if (!TC) {
    TC = 0;
#if 0
    if (ArchName == "i386")
      TC = new Darwin_X86_ToolChain(ArchName);
    else if (ArchName == "x86_64")
      TC = new Darwin_X86_ToolChain(ArchName);
    else
      TC = new Darwin_GCC_ToolChain(ArchName);
#endif
  }

  return TC;
}

// Unknown Host Info

/// UnknownHostInfo - Generic host information to use for unknown
/// hosts.
class UnknownHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  UnknownHostInfo(const Driver &D, const char *Arch, 
                  const char *Platform, const char *OS);

  virtual bool useDriverDriver() const;

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

UnknownHostInfo::UnknownHostInfo(const Driver &D, const char *Arch, 
                                 const char *Platform, const char *OS) 
  : HostInfo(D, Arch, Platform, OS) {
}

bool UnknownHostInfo::useDriverDriver() const { 
  return false;
}

ToolChain *UnknownHostInfo::getToolChain(const ArgList &Args, 
                                         const char *ArchName) const {
  assert(!ArchName && 
         "Unexpected arch name on platform without driver driver support.");
  
  // Automatically handle some instances of -m32/-m64 we know about.
  ArchName = getArchName().c_str();
  if (Arg *A = Args.getLastArg(options::OPT_m32, options::OPT_m64)) {
    if (getArchName() == "i386" || getArchName() == "x86_64") {
      ArchName = 
        (A->getOption().getId() == options::OPT_m32) ? "i386" : "x86_64";
    } else if (getArchName() == "ppc" || getArchName() == "ppc64") {
      ArchName = 
        (A->getOption().getId() == options::OPT_m32) ? "ppc" : "ppc64";
    }
  } 
  
  ToolChain *&TC = ToolChains[ArchName];
  if (!TC)
    TC = 0; //new Generic_GCC_ToolChain(ArchName);

  return 0;
}

}

const HostInfo *clang::driver::createDarwinHostInfo(const Driver &D,
                                                    const char *Arch, 
                                                    const char *Platform, 
                                                    const char *OS) {
  return new DarwinHostInfo(D, Arch, Platform, OS);
}

const HostInfo *clang::driver::createUnknownHostInfo(const Driver &D,
                                                     const char *Arch, 
                                                     const char *Platform, 
                                                     const char *OS) {
  return new UnknownHostInfo(D, Arch, Platform, OS);
}
