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

HostInfo::HostInfo(const Driver &D, const llvm::Triple &_Triple)
  : TheDriver(D), Triple(_Triple)
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
  DarwinHostInfo(const Driver &D, const llvm::Triple &Triple);
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

DarwinHostInfo::DarwinHostInfo(const Driver &D, const llvm::Triple& Triple)
  : HostInfo(D, Triple) {
  
  assert((getArchName() == "i386" || getArchName() == "x86_64" || 
          getArchName() == "powerpc" || getArchName() == "powerpc64" ||
          getArchName() == "arm") &&
         "Unknown Darwin arch.");

  assert(memcmp(&getOSName()[0], "darwin", 6) == 0 &&
         "Unknown Darwin platform.");
  bool HadExtra;
  if (!Driver::GetReleaseVersion(&getOSName()[6], 
                                 DarwinVersion[0], DarwinVersion[1], 
                                 DarwinVersion[2], HadExtra)) {
    D.Diag(clang::diag::err_drv_invalid_darwin_version)
      << getOSName();
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
  std::string Arch;
  if (!ArchName) {
    // If we aren't looking for a specific arch, infer the default architecture
    // based on -arch and -m32/-m64 command line options.
    if (Arg *A = Args.getLastArg(options::OPT_arch)) {
      // The gcc driver behavior with multiple -arch flags wasn't consistent for
      // things which rely on a default architecture. We just use the last -arch
      // to find the default tool chain.
      Arch = A->getValue(Args);
      
      // Normalize arch name; we shouldn't be doing this here.
      //
      // FIXME: This should be unnecessary once everything moves over to using
      // the ID based Triple interface.
      if (Arch == "ppc")
        Arch = "powerpc";
      else if (Arch == "ppc64")
        Arch = "powerpc64";
    } else {
      // Otherwise default to the arch of the host.
      Arch = getArchName();
    }
    ArchName = Arch.c_str();
    
    // Honor -m32 and -m64 when finding the default tool chain.
    if (Arg *A = Args.getLastArg(options::OPT_m32, options::OPT_m64)) {
      if (Arch == "i386" || Arch == "x86_64") {
        ArchName = (A->getOption().getId() == options::OPT_m32) ? "i386" : 
          "x86_64";
      } else if (Arch == "powerpc" || Arch == "powerpc64") {
        ArchName = (A->getOption().getId() == options::OPT_m32) ? "powerpc" : 
          "powerpc64";
      }
    } 
  } else {
    // Normalize arch name; we shouldn't be doing this here.
    //
    // FIXME: This should be unnecessary once everything moves over to using the
    // ID based Triple interface.
    if (strcmp(ArchName, "ppc") == 0)
      ArchName = "powerpc";
    else if (strcmp(ArchName, "ppc64") == 0)
      ArchName = "powerpc64";
  }

  ToolChain *&TC = ToolChains[ArchName];
  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArchName(ArchName);
                          
    if (strcmp(ArchName, "i386") == 0 || strcmp(ArchName, "x86_64") == 0)
      TC = new toolchains::Darwin_X86(*this, TCTriple,
                                      DarwinVersion,
                                      GCCVersion);
    else
      TC = new toolchains::Darwin_GCC(*this, TCTriple);
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
  UnknownHostInfo(const Driver &D, const llvm::Triple& Triple);
  ~UnknownHostInfo();

  virtual bool useDriverDriver() const;

  virtual types::ID lookupTypeForExtension(const char *Ext) const {
    return types::lookupTypeForExtension(Ext);
  }

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

UnknownHostInfo::UnknownHostInfo(const Driver &D, const llvm::Triple& Triple) 
  : HostInfo(D, Triple) {
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
  std::string Arch = getArchName();
  ArchName = Arch.c_str();
  if (Arg *A = Args.getLastArg(options::OPT_m32, options::OPT_m64)) {
    if (Triple.getArch() == llvm::Triple::x86 ||
        Triple.getArch() == llvm::Triple::x86_64) {
      ArchName =
        (A->getOption().getId() == options::OPT_m32) ? "i386" : "x86_64";
    } else if (Triple.getArch() == llvm::Triple::ppc ||
               Triple.getArch() == llvm::Triple::ppc64) {
      ArchName =
        (A->getOption().getId() == options::OPT_m32) ? "powerpc" : "powerpc64";
    }
  } 
  
  ToolChain *&TC = ToolChains[ArchName];
  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArchName(ArchName);

    TC = new toolchains::Generic_GCC(*this, TCTriple);
  }

  return TC;
}

// OpenBSD Host Info

/// OpenBSDHostInfo -  OpenBSD host information implementation.
class OpenBSDHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  OpenBSDHostInfo(const Driver &D, const llvm::Triple& Triple) 
    : HostInfo(D, Triple) {}
  ~OpenBSDHostInfo();

  virtual bool useDriverDriver() const;

  virtual types::ID lookupTypeForExtension(const char *Ext) const {
    return types::lookupTypeForExtension(Ext);
  }

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

OpenBSDHostInfo::~OpenBSDHostInfo() {
  for (llvm::StringMap<ToolChain*>::iterator
         it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it)
    delete it->second;
}

bool OpenBSDHostInfo::useDriverDriver() const { 
  return false;
}

ToolChain *OpenBSDHostInfo::getToolChain(const ArgList &Args, 
                                         const char *ArchName) const {
  assert(!ArchName && 
         "Unexpected arch name on platform without driver driver support.");
  
  std::string Arch = getArchName();
  ArchName = Arch.c_str();
  
  ToolChain *&TC = ToolChains[ArchName];
  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArchName(ArchName);

    TC = new toolchains::OpenBSD(*this, TCTriple);
  }

  return TC;
}

// AuroraUX Host Info

/// AuroraUXHostInfo - AuroraUX host information implementation.
class AuroraUXHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  AuroraUXHostInfo(const Driver &D, const llvm::Triple& Triple)
    : HostInfo(D, Triple) {}
  ~AuroraUXHostInfo();

  virtual bool useDriverDriver() const;

  virtual types::ID lookupTypeForExtension(const char *Ext) const {
    return types::lookupTypeForExtension(Ext);
  }

  virtual ToolChain *getToolChain(const ArgList &Args,
				  const char *ArchName) const;
};

AuroraUXHostInfo::~AuroraUXHostInfo() {
  for (llvm::StringMap<ToolChain*>::iterator
	 it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it)
    delete it->second;
}

bool AuroraUXHostInfo::useDriverDriver() const {
  return false;
}

ToolChain *AuroraUXHostInfo::getToolChain(const ArgList &Args,
					  const char *ArchName) const {
  assert(!ArchName &&
	 "Unexpected arch name on platform without driver driver support.");

  ToolChain *&TC = ToolChains[getArchName()];

  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArchName(getArchName());

    TC = new toolchains::AuroraUX(*this, TCTriple);
  }

  return TC;
}

// FreeBSD Host Info

/// FreeBSDHostInfo -  FreeBSD host information implementation.
class FreeBSDHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  FreeBSDHostInfo(const Driver &D, const llvm::Triple& Triple) 
    : HostInfo(D, Triple) {}
  ~FreeBSDHostInfo();

  virtual bool useDriverDriver() const;

  virtual types::ID lookupTypeForExtension(const char *Ext) const {
    return types::lookupTypeForExtension(Ext);
  }

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

FreeBSDHostInfo::~FreeBSDHostInfo() {
  for (llvm::StringMap<ToolChain*>::iterator
         it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it)
    delete it->second;
}

bool FreeBSDHostInfo::useDriverDriver() const { 
  return false;
}

ToolChain *FreeBSDHostInfo::getToolChain(const ArgList &Args, 
                                         const char *ArchName) const {
  bool Lib32 = false;

  assert(!ArchName && 
         "Unexpected arch name on platform without driver driver support.");
  
  // On x86_64 we need to be able to compile 32-bits binaries as well.
  // Compiling 64-bit binaries on i386 is not supported. We don't have a
  // lib64.
  std::string Arch = getArchName();
  ArchName = Arch.c_str();
  if (Args.hasArg(options::OPT_m32) && getArchName() == "x86_64") {
    ArchName = "i386";
    Lib32 = true;
  } 
  
  ToolChain *&TC = ToolChains[ArchName];
  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArchName(ArchName);

    TC = new toolchains::FreeBSD(*this, TCTriple, Lib32);
  }

  return TC;
}

// DragonFly Host Info

/// DragonFlyHostInfo -  DragonFly host information implementation.
class DragonFlyHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  DragonFlyHostInfo(const Driver &D, const llvm::Triple& Triple)
    : HostInfo(D, Triple) {}
  ~DragonFlyHostInfo();

  virtual bool useDriverDriver() const;

  virtual types::ID lookupTypeForExtension(const char *Ext) const {
    return types::lookupTypeForExtension(Ext);
  }

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

DragonFlyHostInfo::~DragonFlyHostInfo() {
  for (llvm::StringMap<ToolChain*>::iterator
         it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it)
    delete it->second;
}

bool DragonFlyHostInfo::useDriverDriver() const { 
  return false;
}

ToolChain *DragonFlyHostInfo::getToolChain(const ArgList &Args, 
                                           const char *ArchName) const {
  assert(!ArchName && 
         "Unexpected arch name on platform without driver driver support.");

  ToolChain *&TC = ToolChains[getArchName()];

  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArchName(getArchName());

    TC = new toolchains::DragonFly(*this, TCTriple);
  }

  return TC;
}

// Linux Host Info

/// LinuxHostInfo -  Linux host information implementation.
class LinuxHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  LinuxHostInfo(const Driver &D, const llvm::Triple& Triple)
    : HostInfo(D, Triple) {}
  ~LinuxHostInfo();

  virtual bool useDriverDriver() const;

  virtual types::ID lookupTypeForExtension(const char *Ext) const {
    return types::lookupTypeForExtension(Ext);
  }

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

LinuxHostInfo::~LinuxHostInfo() {
  for (llvm::StringMap<ToolChain*>::iterator
         it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it)
    delete it->second;
}

bool LinuxHostInfo::useDriverDriver() const { 
  return false;
}

ToolChain *LinuxHostInfo::getToolChain(const ArgList &Args, 
                                       const char *ArchName) const {

  assert(!ArchName && 
         "Unexpected arch name on platform without driver driver support.");

  // Automatically handle some instances of -m32/-m64 we know about.
  std::string Arch = getArchName();
  ArchName = Arch.c_str();
  if (Arg *A = Args.getLastArg(options::OPT_m32, options::OPT_m64)) {
    if (Triple.getArch() == llvm::Triple::x86 ||
        Triple.getArch() == llvm::Triple::x86_64) {
      ArchName =
        (A->getOption().getId() == options::OPT_m32) ? "i386" : "x86_64";
    } else if (Triple.getArch() == llvm::Triple::ppc ||
               Triple.getArch() == llvm::Triple::ppc64) {
      ArchName =
        (A->getOption().getId() == options::OPT_m32) ? "powerpc" : "powerpc64";
    }
  }

  ToolChain *&TC = ToolChains[ArchName];

  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArchName(ArchName);

    TC = new toolchains::Linux(*this, TCTriple);
  }

  return TC;
}

}

const HostInfo *
clang::driver::createAuroraUXHostInfo(const Driver &D,
				      const llvm::Triple& Triple){
  return new AuroraUXHostInfo(D, Triple);
}

const HostInfo *
clang::driver::createDarwinHostInfo(const Driver &D,
                                    const llvm::Triple& Triple){
  return new DarwinHostInfo(D, Triple);
}

const HostInfo *
clang::driver::createOpenBSDHostInfo(const Driver &D, 
                                     const llvm::Triple& Triple) {
  return new OpenBSDHostInfo(D, Triple);
}

const HostInfo *
clang::driver::createFreeBSDHostInfo(const Driver &D, 
                                     const llvm::Triple& Triple) {
  return new FreeBSDHostInfo(D, Triple);
}

const HostInfo *
clang::driver::createDragonFlyHostInfo(const Driver &D,
                                       const llvm::Triple& Triple) {
  return new DragonFlyHostInfo(D, Triple);
}

const HostInfo *
clang::driver::createLinuxHostInfo(const Driver &D,
                                   const llvm::Triple& Triple) {
  return new LinuxHostInfo(D, Triple);
}

const HostInfo *
clang::driver::createUnknownHostInfo(const Driver &D,
                                     const llvm::Triple& Triple) {
  return new UnknownHostInfo(D, Triple);
}
