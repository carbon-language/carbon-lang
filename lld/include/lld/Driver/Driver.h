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
/// Interface for Drivers which convert command line arguments into
/// TargetInfo objects, then perform the link.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_DRIVER_DRIVER_H
#define LLD_DRIVER_DRIVER_H

#include "lld/Core/LLVM.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <vector>

namespace lld {
class TargetInfo;
class CoreTargetInfo;
class MachOTargetInfo;
class PECOFFTargetInfo;
class ELFTargetInfo;

/// Base class for all Drivers.
class Driver {
protected:

  /// Performs link using specified options.
  static bool link(const TargetInfo &targetInfo,
                   raw_ostream &diagnostics = llvm::errs());
private:
  Driver() LLVM_DELETED_FUNCTION;
};


/// Driver for "universal" lld tool which can mimic any linker command line
/// parsing once it figures out which command line flavor to use.
class UniversalDriver : public Driver {
public:
  /// Determine flavor and pass control to Driver for that flavor.
  static bool link(int argc, const char *argv[],
                   raw_ostream &diagnostics = llvm::errs());

private:
  UniversalDriver() LLVM_DELETED_FUNCTION;
};


/// Driver for gnu/binutil 'ld' command line options.
class GnuLdDriver : public Driver {
public:
  /// Parses command line arguments same as gnu/binutils ld and performs link.
  /// Returns true iff an error occurred.
  static bool linkELF(int argc, const char *argv[],
                  raw_ostream &diagnostics = llvm::errs());

  /// Uses gnu/binutils style ld command line options to fill in options struct.
  /// Returns true iff there was an error.
  static bool parse(int argc, const char *argv[],
                    std::unique_ptr<ELFTargetInfo> &targetInfo,
                    raw_ostream &diagnostics = llvm::errs());

private:
  static llvm::Triple getDefaultTarget(const char *progName);

  GnuLdDriver() LLVM_DELETED_FUNCTION;
};


/// Driver for darwin/ld64 'ld' command line options.
class DarwinLdDriver : public Driver {
public:
  /// Parses command line arguments same as darwin's ld and performs link.
  /// Returns true iff there was an error.
  static bool linkMachO(int argc, const char *argv[],
                        raw_ostream &diagnostics = llvm::errs());

  /// Uses darwin style ld command line options to update targetInfo object.
  /// Returns true iff there was an error.
  static bool parse(int argc, const char *argv[], MachOTargetInfo &info,
                    raw_ostream &diagnostics = llvm::errs());
private:
  DarwinLdDriver() LLVM_DELETED_FUNCTION;
};


/// Driver for Windows 'link.exe' command line options
class WinLinkDriver : public Driver {
public:
  /// Parses command line arguments same as Windows link.exe and performs link.
  /// Returns true iff there was an error.
  static bool linkPECOFF(int argc, const char *argv[],
                         raw_ostream &diagnostics = llvm::errs());

  /// Uses Windows style link command line options to fill in options struct.
  /// Returns true iff there was an error.
  static bool parse(int argc, const char *argv[], PECOFFTargetInfo &info,
                    raw_ostream &diagnostics = llvm::errs());

private:
  WinLinkDriver() LLVM_DELETED_FUNCTION;
};


/// Driver for lld unit tests
class CoreDriver : public Driver {
public:

  /// Parses command line arguments same as lld-core and performs link.
  /// Returns true iff there was an error.
  static bool link(int argc, const char *argv[],
                   raw_ostream &diagnostics = llvm::errs());

  /// Uses lld-core command line options to fill in options struct.
  /// Returns true iff there was an error.
  static bool parse(int argc, const char *argv[], CoreTargetInfo &info,
                    raw_ostream &diagnostics = llvm::errs());

private:
  CoreDriver() LLVM_DELETED_FUNCTION;
};




} // end namespace lld

#endif
