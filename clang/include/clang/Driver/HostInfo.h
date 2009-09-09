//===--- HostInfo.h - Host specific information -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_HOSTINFO_H_
#define CLANG_DRIVER_HOSTINFO_H_

#include "clang/Driver/Types.h"
#include "llvm/ADT/Triple.h"
#include <string>

namespace clang {
namespace driver {
  class ArgList;
  class Driver;
  class ToolChain;

/// HostInfo - Config information about a particular host which may interact
/// with driver behavior.
///
/// The host information is used for controlling the parts of the driver which
/// interact with the platform the driver is ostensibly being run from. For
/// testing purposes, the HostInfo used by the driver may differ from the actual
/// host.
class HostInfo {
protected:
  const Driver &TheDriver;
  const llvm::Triple Triple;

  HostInfo(const Driver &D, const llvm::Triple &_Triple);

public:
  virtual ~HostInfo();

  const Driver &getDriver() const { return TheDriver; }

  const llvm::Triple& getTriple() const { return Triple; }
  std::string getArchName() const { return Triple.getArchName(); }
  std::string getPlatformName() const { return Triple.getVendorName(); }
  std::string getOSName() const { return Triple.getOSName(); }

  /// useDriverDriver - Whether the driver should act as a driver driver for
  /// this host and support -arch, -Xarch, etc.
  virtual bool useDriverDriver() const = 0;

  /// lookupTypeForExtension - Return the default language type to use for the
  /// given extension.
  virtual types::ID lookupTypeForExtension(const char *Ext) const = 0;

  /// CreateToolChain - Construct the toolchain to use for this host (which the
  /// host retains ownership of).
  ///
  /// \param Args - The argument list, which may be used to alter the default
  /// toolchain, for example in the presence of -m32 or -m64.
  ///
  /// \param ArchName - The architecture to return a toolchain for, or 0 if
  /// unspecified. This will only ever be non-zero for hosts which support a
  /// driver driver.

  // FIXME: Pin down exactly what the HostInfo is allowed to use Args
  // for here. Currently this is for -m32 / -m64 defaulting.
  virtual ToolChain *CreateToolChain(const ArgList &Args,
                                     const char *ArchName=0) const = 0;
};

const HostInfo *createAuroraUXHostInfo(const Driver &D,
                                       const llvm::Triple& Triple);
const HostInfo *createDarwinHostInfo(const Driver &D,
                                     const llvm::Triple& Triple);
const HostInfo *createOpenBSDHostInfo(const Driver &D,
                                      const llvm::Triple& Triple);
const HostInfo *createFreeBSDHostInfo(const Driver &D,
                                      const llvm::Triple& Triple);
const HostInfo *createDragonFlyHostInfo(const Driver &D,
                                        const llvm::Triple& Triple);
const HostInfo *createLinuxHostInfo(const Driver &D,
                                    const llvm::Triple& Triple);
const HostInfo *createUnknownHostInfo(const Driver &D,
                                      const llvm::Triple& Triple);

} // end namespace driver
} // end namespace clang

#endif
