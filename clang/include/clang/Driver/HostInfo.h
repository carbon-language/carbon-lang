//===--- HostInfo.h - Host specific information -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_HOSTINFO,_H_
#define CLANG_DRIVER_HOSTINFO_H_

#include <string>

namespace clang {
namespace driver {
  class ArgList;
  class ToolChain;

/// HostInfo - Config information about a particular host which may
/// interact with driver behavior.
/// 
/// The host information is used for controlling the parts of the
/// driver which interact with the platform the driver is ostensibly
/// being run from. For testing purposes, the HostInfo used by the
/// driver may differ from the actual host.
class HostInfo {
  std::string Arch, Platform, OS;

protected:
  HostInfo(const char *Arch, const char *Platform, const char *OS);

public:
  virtual ~HostInfo();

  /// useDriverDriver - Whether the driver should act as a driver
  /// driver for this host and support -arch, -Xarch, etc.
  virtual bool useDriverDriver() const = 0;

  /// getToolChain - Construct the toolchain to use for this host.
  ///
  /// \param Args - The argument list, which may be used to alter the
  /// default toolchain, for example in the presence of -m32 or -m64.
  ///
  /// \param ArchName - The architecture to return a toolchain for, or
  /// 0 if unspecified. This will only be non-zero for hosts which
  /// support a driver driver.
  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const = 0;
};

/// DarwinHostInfo - Darwin host information implementation.
class DarwinHostInfo : public HostInfo {
  /// Darwin version of host.
  unsigned DarwinVersion[3];

  /// GCC version to use on this host.
  unsigned GCCVersion[3];

public:
  DarwinHostInfo(const char *Arch, const char *Platform, const char *OS);

  virtual bool useDriverDriver() const;

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

/// UnknownHostInfo - Generic host information to use for unknown
/// hosts.
class UnknownHostInfo : public HostInfo {
public:
  UnknownHostInfo(const char *Arch, const char *Platform, const char *OS);

  virtual bool useDriverDriver() const;

  virtual ToolChain *getToolChain(const ArgList &Args, 
                                  const char *ArchName) const;
};

} // end namespace driver
} // end namespace clang

#endif
