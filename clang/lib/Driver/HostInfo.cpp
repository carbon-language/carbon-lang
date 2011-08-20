//===--- HostInfo.cpp - Host specific information -------------------------===//
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
  : TheDriver(D), Triple(_Triple) {
}

HostInfo::~HostInfo() {
}

namespace {

// Darwin Host Info

/// DarwinHostInfo - Darwin host information implementation.
class DarwinHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::DenseMap<unsigned, ToolChain*> ToolChains;

public:
  DarwinHostInfo(const Driver &D, const llvm::Triple &Triple);
  ~DarwinHostInfo();

  virtual bool useDriverDriver() const;

  virtual ToolChain *CreateToolChain(const ArgList &Args,
                                     const char *ArchName) const;
};

DarwinHostInfo::DarwinHostInfo(const Driver &D, const llvm::Triple& Triple)
  : HostInfo(D, Triple) {
}

DarwinHostInfo::~DarwinHostInfo() {
  for (llvm::DenseMap<unsigned, ToolChain*>::iterator
         it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it)
    delete it->second;
}

bool DarwinHostInfo::useDriverDriver() const {
  return true;
}

ToolChain *DarwinHostInfo::CreateToolChain(const ArgList &Args,
                                           const char *ArchName) const {
  llvm::Triple::ArchType Arch;

  if (!ArchName) {
    // If we aren't looking for a specific arch, infer the default architecture
    // based on -arch and -m32/-m64 command line options.
    if (Arg *A = Args.getLastArg(options::OPT_arch)) {
      // The gcc driver behavior with multiple -arch flags wasn't consistent for
      // things which rely on a default architecture. We just use the last -arch
      // to find the default tool chain (assuming it is valid).
      Arch = llvm::Triple::getArchTypeForDarwinArchName(A->getValue(Args));

      // If it was invalid just use the host, we will reject this command line
      // later.
      if (Arch == llvm::Triple::UnknownArch)
        Arch = getTriple().getArch();
    } else {
      // Otherwise default to the arch of the host.
      Arch = getTriple().getArch();
    }

    // Honor -m32 and -m64 when finding the default tool chain.
    //
    // FIXME: Should this information be in llvm::Triple?
    if (Arg *A = Args.getLastArg(options::OPT_m32, options::OPT_m64)) {
      if (A->getOption().matches(options::OPT_m32)) {
        if (Arch == llvm::Triple::x86_64)
          Arch = llvm::Triple::x86;
        if (Arch == llvm::Triple::ppc64)
          Arch = llvm::Triple::ppc;
      } else {
        if (Arch == llvm::Triple::x86)
          Arch = llvm::Triple::x86_64;
        if (Arch == llvm::Triple::ppc)
          Arch = llvm::Triple::ppc64;
      }
    }
  } else
    Arch = llvm::Triple::getArchTypeForDarwinArchName(ArchName);

  assert(Arch != llvm::Triple::UnknownArch && "Unexpected arch!");
  ToolChain *&TC = ToolChains[Arch];
  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArch(Arch);

    // If we recognized the arch, match it to the toolchains we support.
    if (Arch == llvm::Triple::x86 || Arch == llvm::Triple::x86_64 ||
        Arch == llvm::Triple::arm || Arch == llvm::Triple::thumb) {
      TC = new toolchains::DarwinClang(*this, TCTriple);
    } else
      TC = new toolchains::Darwin_Generic_GCC(*this, TCTriple);
  }

  return TC;
}

// TCE Host Info

/// TCEHostInfo - TCE host information implementation (see http://tce.cs.tut.fi)
class TCEHostInfo : public HostInfo {

public:
  TCEHostInfo(const Driver &D, const llvm::Triple &Triple);
  ~TCEHostInfo() {}

  virtual bool useDriverDriver() const;

  virtual ToolChain *CreateToolChain(const ArgList &Args, 
                                     const char *ArchName) const;
};

TCEHostInfo::TCEHostInfo(const Driver &D, const llvm::Triple& Triple)
  : HostInfo(D, Triple) {
}

bool TCEHostInfo::useDriverDriver() const { 
  return false;
}

ToolChain *TCEHostInfo::CreateToolChain(const ArgList &Args, 
                                        const char *ArchName) const {
  llvm::Triple TCTriple(getTriple());
//  TCTriple.setArchName(ArchName);
  return new toolchains::TCEToolChain(*this, TCTriple);
}


// Unknown Host Info

/// UnknownHostInfo - Generic host information to use for unknown hosts.
class UnknownHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  UnknownHostInfo(const Driver &D, const llvm::Triple& Triple);
  ~UnknownHostInfo();

  virtual bool useDriverDriver() const;

  virtual ToolChain *CreateToolChain(const ArgList &Args,
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

ToolChain *UnknownHostInfo::CreateToolChain(const ArgList &Args,
                                            const char *ArchName) const {
  assert(!ArchName &&
         "Unexpected arch name on platform without driver support.");

  // Automatically handle some instances of -m32/-m64 we know about.
  std::string Arch = getArchName();
  ArchName = Arch.c_str();
  if (Arg *A = Args.getLastArg(options::OPT_m32, options::OPT_m64)) {
    if (Triple.getArch() == llvm::Triple::x86 ||
        Triple.getArch() == llvm::Triple::x86_64) {
      ArchName =
        (A->getOption().matches(options::OPT_m32)) ? "i386" : "x86_64";
    } else if (Triple.getArch() == llvm::Triple::ppc ||
               Triple.getArch() == llvm::Triple::ppc64) {
      ArchName =
        (A->getOption().matches(options::OPT_m32)) ? "powerpc" : "powerpc64";
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

  virtual ToolChain *CreateToolChain(const ArgList &Args,
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

ToolChain *OpenBSDHostInfo::CreateToolChain(const ArgList &Args,
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

  virtual ToolChain *CreateToolChain(const ArgList &Args,
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

ToolChain *AuroraUXHostInfo::CreateToolChain(const ArgList &Args,
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

  virtual ToolChain *CreateToolChain(const ArgList &Args,
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

ToolChain *FreeBSDHostInfo::CreateToolChain(const ArgList &Args,
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
        (A->getOption().matches(options::OPT_m32)) ? "i386" : "x86_64";
    } else if (Triple.getArch() == llvm::Triple::ppc ||
               Triple.getArch() == llvm::Triple::ppc64) {
      ArchName =
        (A->getOption().matches(options::OPT_m32)) ? "powerpc" : "powerpc64";
    }
  }

  ToolChain *&TC = ToolChains[ArchName];
  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArchName(ArchName);

    TC = new toolchains::FreeBSD(*this, TCTriple);
  }

  return TC;
}

// NetBSD Host Info

/// NetBSDHostInfo -  NetBSD host information implementation.
class NetBSDHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  NetBSDHostInfo(const Driver &D, const llvm::Triple& Triple)
    : HostInfo(D, Triple) {}
  ~NetBSDHostInfo();

  virtual bool useDriverDriver() const;

  virtual ToolChain *CreateToolChain(const ArgList &Args,
                                     const char *ArchName) const;
};

NetBSDHostInfo::~NetBSDHostInfo() {
  for (llvm::StringMap<ToolChain*>::iterator
         it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it)
    delete it->second;
}

bool NetBSDHostInfo::useDriverDriver() const {
  return false;
}

ToolChain *NetBSDHostInfo::CreateToolChain(const ArgList &Args,
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
        (A->getOption().matches(options::OPT_m32)) ? "i386" : "x86_64";
    } else if (Triple.getArch() == llvm::Triple::ppc ||
               Triple.getArch() == llvm::Triple::ppc64) {
      ArchName =
        (A->getOption().matches(options::OPT_m32)) ? "powerpc" : "powerpc64";
    }
  }
  llvm::Triple TargetTriple(getTriple());
  TargetTriple.setArchName(ArchName);

  ToolChain *TC;

  // XXX Cache toolchain even if -m32 is used
  if (Arch == ArchName) {
    TC = ToolChains[ArchName];
    if (TC)
      return TC;
  }

  TC = new toolchains::NetBSD(*this, TargetTriple, getTriple());

  return TC;
}

// Minix Host Info

/// MinixHostInfo -  Minix host information implementation.
class MinixHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  MinixHostInfo(const Driver &D, const llvm::Triple& Triple)
    : HostInfo(D, Triple) {}
  ~MinixHostInfo();

  virtual bool useDriverDriver() const;

  virtual ToolChain *CreateToolChain(const ArgList &Args,
                                     const char *ArchName) const;
};

MinixHostInfo::~MinixHostInfo() {
  for (llvm::StringMap<ToolChain*>::iterator
         it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it){
    delete it->second;
  }
}

bool MinixHostInfo::useDriverDriver() const {
  return false;
}

ToolChain *MinixHostInfo::CreateToolChain(const ArgList &Args,
                                            const char *ArchName) const {
  assert(!ArchName &&
         "Unexpected arch name on platform without driver driver support.");

  std::string Arch = getArchName();
  ArchName = Arch.c_str();

  ToolChain *&TC = ToolChains[ArchName];
  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArchName(ArchName);

    TC = new toolchains::Minix(*this, TCTriple);
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

  virtual ToolChain *CreateToolChain(const ArgList &Args,
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

ToolChain *DragonFlyHostInfo::CreateToolChain(const ArgList &Args,
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

  virtual ToolChain *CreateToolChain(const ArgList &Args,
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

ToolChain *LinuxHostInfo::CreateToolChain(const ArgList &Args,
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
        (A->getOption().matches(options::OPT_m32)) ? "i386" : "x86_64";
    } else if (Triple.getArch() == llvm::Triple::ppc ||
               Triple.getArch() == llvm::Triple::ppc64) {
      ArchName =
        (A->getOption().matches(options::OPT_m32)) ? "powerpc" : "powerpc64";
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

// Windows Host Info

/// WindowsHostInfo - Host information to use on Microsoft Windows.
class WindowsHostInfo : public HostInfo {
  /// Cache of tool chains we have created.
  mutable llvm::StringMap<ToolChain*> ToolChains;

public:
  WindowsHostInfo(const Driver &D, const llvm::Triple& Triple);
  ~WindowsHostInfo();

  virtual bool useDriverDriver() const;

  virtual types::ID lookupTypeForExtension(const char *Ext) const {
    return types::lookupTypeForExtension(Ext);
  }

  virtual ToolChain *CreateToolChain(const ArgList &Args,
                                     const char *ArchName) const;
};

WindowsHostInfo::WindowsHostInfo(const Driver &D, const llvm::Triple& Triple)
  : HostInfo(D, Triple) {
}

WindowsHostInfo::~WindowsHostInfo() {
  for (llvm::StringMap<ToolChain*>::iterator
         it = ToolChains.begin(), ie = ToolChains.end(); it != ie; ++it)
    delete it->second;
}

bool WindowsHostInfo::useDriverDriver() const {
  return false;
}

ToolChain *WindowsHostInfo::CreateToolChain(const ArgList &Args,
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
        (A->getOption().matches(options::OPT_m32)) ? "i386" : "x86_64";
    }
  }

  ToolChain *&TC = ToolChains[ArchName];
  if (!TC) {
    llvm::Triple TCTriple(getTriple());
    TCTriple.setArchName(ArchName);

    TC = new toolchains::Windows(*this, TCTriple);
  }

  return TC;
}

// FIXME: This is a placeholder.
class MinGWHostInfo : public UnknownHostInfo {
public:
  MinGWHostInfo(const Driver &D, const llvm::Triple& Triple);
};

MinGWHostInfo::MinGWHostInfo(const Driver &D, const llvm::Triple& Triple)
  : UnknownHostInfo(D, Triple) {}

} // end anon namespace

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
clang::driver::createNetBSDHostInfo(const Driver &D,
                                     const llvm::Triple& Triple) {
  return new NetBSDHostInfo(D, Triple);
}

const HostInfo *
clang::driver::createMinixHostInfo(const Driver &D,
                                     const llvm::Triple& Triple) {
  return new MinixHostInfo(D, Triple);
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
clang::driver::createTCEHostInfo(const Driver &D,
                                   const llvm::Triple& Triple) {
  return new TCEHostInfo(D, Triple);
}

const HostInfo *
clang::driver::createWindowsHostInfo(const Driver &D,
                                     const llvm::Triple& Triple) {
  return new WindowsHostInfo(D, Triple);
}

const HostInfo *
clang::driver::createMinGWHostInfo(const Driver &D,
                                   const llvm::Triple& Triple) {
  return new MinGWHostInfo(D, Triple);
}

const HostInfo *
clang::driver::createUnknownHostInfo(const Driver &D,
                                     const llvm::Triple& Triple) {
  return new UnknownHostInfo(D, Triple);
}
