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
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"

#include "ToolChains.h"

#include <cassert>
 
using namespace clang::driver;

HostInfo::HostInfo(const Driver &D, const char *_Arch, const char *_Platform,
                   const char *_OS) 
  : TheDriver(D), Arch(_Arch), Platform(_Platform), OS(_OS) 
{
  
}

HostInfo::~HostInfo() {
}

namespace {

// Darwin Host Info

/// DarwinHostInfo - Darwin host information implementation.
class DarwinHostInfo : public HostInfo {
  /// Darwin version of host.
  unsigned DarwinVersion[3];

  /// GCC version to use on this host.
  unsigned GCCVersion[3];

  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain *> ToolChains;

public:
  DarwinHostInfo(const Driver &D, const char *Arch, 
                 const char *Platform, const char *OS);
  ~DarwinHostInfo();

  virtual bool useDriverDriver() const;

  virtual types::ID lookupTypeForExtension(const char *Ext) const {
    types::ID Ty = types::lookupTypeForExtension(Ext);

    // Darwin always preprocesses assembly files (unless -x is used
    // explicitly).
    if (Ty == types::TY_PP_Asm)
      return types::TY_Asm;

    return Ty;
  }

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

/// GetReleaseVersion - Parse (([0-9]+)(.([0-9]+)(.([0-9]+)?))?)? and
/// return the grouped values as integers. Numbers which are not
/// provided are set to 0.
///
/// \return True if the entire string was parsed (9.2), or all groups
/// were parsed (10.3.5extrastuff).
static bool GetReleaseVersion(const char *Str, unsigned &Major, 
                              unsigned &Minor, unsigned &Micro) {
  Major = Minor = Micro = 0;
  if (*Str == '\0') 
    return true;

  char *End;
  Major = (unsigned) strtol(Str, &End, 10);
  if (*Str != '\0' && *End == '\0')
    return true;
  if (*End != '.')
    return false;
  
  Str = End+1;
  Minor = (unsigned) strtol(Str, &End, 10);
  if (*Str != '\0' && *End == '\0')
    return true;
  if (*End != '.')
    return false;

  Str = End+1;
  Micro = (unsigned) strtol(Str, &End, 10);
  return true;
}

DarwinHostInfo::DarwinHostInfo(const Driver &D, const char *_Arch, 
                               const char *_Platform, const char *_OS) 
  : HostInfo(D, _Arch, _Platform, _OS) {
  
  assert((getArchName() == "i386" || getArchName() == "x86_64" || 
          getArchName() == "ppc" || getArchName() == "ppc64") &&
         "Unknown Darwin arch.");

  assert(memcmp(&getOSName()[0], "darwin", 6) == 0 &&
         "Unknown Darwin platform.");
  const char *Release = &getOSName()[6];
  if (!GetReleaseVersion(Release, DarwinVersion[0], DarwinVersion[1], 
                         DarwinVersion[2])) {
    D.Diag(clang::diag::err_drv_invalid_darwin_version)
      << Release;
  }
  
  // We can only call 4.2.1 for now.
  GCCVersion[0] = 4;
  GCCVersion[1] = 2;
  GCCVersion[2] = 1;
}

DarwinHostInfo::~DarwinHostInfo() {
  for (llvm::StringMap<ToolChain*>::iterator
         it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it)
    delete it->second;
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
    if (strcmp(ArchName, "i386") == 0 || strcmp(ArchName, "x86_64") == 0)
      TC = new toolchains::Darwin_X86(*this, ArchName, 
                                      getPlatformName().c_str(), 
                                      getOSName().c_str());
    else
      TC = new toolchains::Darwin_GCC(*this, ArchName, 
                                      getPlatformName().c_str(), 
                                      getOSName().c_str());
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
  ~UnknownHostInfo();

  virtual bool useDriverDriver() const;

  virtual types::ID lookupTypeForExtension(const char *Ext) const {
    return types::lookupTypeForExtension(Ext);
  }

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

UnknownHostInfo::UnknownHostInfo(const Driver &D, const char *Arch, 
                                 const char *Platform, const char *OS) 
  : HostInfo(D, Arch, Platform, OS) {
}

UnknownHostInfo::~UnknownHostInfo() {
  for (llvm::StringMap<ToolChain*>::iterator
         it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it)
    delete it->second;
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
    TC = new toolchains::Generic_GCC(*this, ArchName, 
                                     getPlatformName().c_str(), 
                                     getOSName().c_str());

  return TC;
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
