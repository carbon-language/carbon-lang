//===- lld/Driver/Driver.h - Linker Driver Emulator -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Interface and factory for creating a specific driver emulator. A Driver is
/// used to transform command line arguments into command line arguments for
/// core. Core arguments are used to generate a LinkerOptions object.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_DRIVER_DRIVER_H
#define LLD_DRIVER_DRIVER_H

#include "lld/Core/LLVM.h"

#include "llvm/Option/ArgList.h"

#include <memory>
#include <string>

namespace lld {
struct LinkerOptions;

/// \brief Base class for all Drivers.
class Driver {
protected:
  Driver(StringRef defaultTargetTriple)
    : _defaultTargetTriple(defaultTargetTriple) {}

  std::string _defaultTargetTriple;

public:
  enum class Flavor {
    invalid,
    ld,
    link,
    ld64,
    core
  };

  virtual ~Driver();

  virtual std::unique_ptr<llvm::opt::DerivedArgList>
    transform(llvm::ArrayRef<const char *const> args) = 0;

  /// \param flavor driver flavor to create.
  /// \param defaultTargetTriple target triple as determined by the program name
  ///        or host. May be overridden by -target.
  /// \returns the created driver.
  static std::unique_ptr<Driver> create(Flavor flavor,
                                        StringRef defaultTargetTriple);
};

LinkerOptions generateOptions(const llvm::opt::ArgList &args);
} // end namespace lld

#endif
