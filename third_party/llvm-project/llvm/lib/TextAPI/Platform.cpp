//===- llvm/TextAPI/Platform.cpp - Platform ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations of Platform Helper functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/TextAPI/Platform.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"

namespace llvm {
namespace MachO {

PlatformType mapToPlatformType(PlatformType Platform, bool WantSim) {
  switch (Platform) {
  default:
    return Platform;
  case PLATFORM_IOS:
    return WantSim ? PLATFORM_IOSSIMULATOR : PLATFORM_IOS;
  case PLATFORM_TVOS:
    return WantSim ? PLATFORM_TVOSSIMULATOR : PLATFORM_TVOS;
  case PLATFORM_WATCHOS:
    return WantSim ? PLATFORM_WATCHOSSIMULATOR : PLATFORM_WATCHOS;
  }
}

PlatformType mapToPlatformType(const Triple &Target) {
  switch (Target.getOS()) {
  default:
    return PLATFORM_UNKNOWN;
  case Triple::MacOSX:
    return PLATFORM_MACOS;
  case Triple::IOS:
    if (Target.isSimulatorEnvironment())
      return PLATFORM_IOSSIMULATOR;
    if (Target.getEnvironment() == Triple::MacABI)
      return PLATFORM_MACCATALYST;
    return PLATFORM_IOS;
  case Triple::TvOS:
    return Target.isSimulatorEnvironment() ? PLATFORM_TVOSSIMULATOR
                                           : PLATFORM_TVOS;
  case Triple::WatchOS:
    return Target.isSimulatorEnvironment() ? PLATFORM_WATCHOSSIMULATOR
                                           : PLATFORM_WATCHOS;
    // TODO: add bridgeOS & driverKit once in llvm::Triple
  }
}

PlatformSet mapToPlatformSet(ArrayRef<Triple> Targets) {
  PlatformSet Result;
  for (const auto &Target : Targets)
    Result.insert(mapToPlatformType(Target));
  return Result;
}

StringRef getPlatformName(PlatformType Platform) {
  switch (Platform) {
  case PLATFORM_UNKNOWN:
    return "unknown";
  case PLATFORM_MACOS:
    return "macOS";
  case PLATFORM_IOS:
    return "iOS";
  case PLATFORM_TVOS:
    return "tvOS";
  case PLATFORM_WATCHOS:
    return "watchOS";
  case PLATFORM_BRIDGEOS:
    return "bridgeOS";
  case PLATFORM_MACCATALYST:
    return "macCatalyst";
  case PLATFORM_IOSSIMULATOR:
    return "iOS Simulator";
  case PLATFORM_TVOSSIMULATOR:
    return "tvOS Simulator";
  case PLATFORM_WATCHOSSIMULATOR:
    return "watchOS Simulator";
  case PLATFORM_DRIVERKIT:
    return "DriverKit";
  }
  llvm_unreachable("Unknown llvm::MachO::PlatformType enum");
}

PlatformType getPlatformFromName(StringRef Name) {
  return StringSwitch<PlatformType>(Name)
      .Case("macos", PLATFORM_MACOS)
      .Case("ios", PLATFORM_IOS)
      .Case("tvos", PLATFORM_TVOS)
      .Case("watchos", PLATFORM_WATCHOS)
      .Case("bridgeos", PLATFORM_BRIDGEOS)
      .Case("ios-macabi", PLATFORM_MACCATALYST)
      .Case("ios-simulator", PLATFORM_IOSSIMULATOR)
      .Case("tvos-simulator", PLATFORM_TVOSSIMULATOR)
      .Case("watchos-simulator", PLATFORM_WATCHOSSIMULATOR)
      .Case("driverkit", PLATFORM_DRIVERKIT)
      .Default(PLATFORM_UNKNOWN);
}

std::string getOSAndEnvironmentName(PlatformType Platform,
                                    std::string Version) {
  switch (Platform) {
  case PLATFORM_UNKNOWN:
    return "darwin" + Version;
  case PLATFORM_MACOS:
    return "macos" + Version;
  case PLATFORM_IOS:
    return "ios" + Version;
  case PLATFORM_TVOS:
    return "tvos" + Version;
  case PLATFORM_WATCHOS:
    return "watchos" + Version;
  case PLATFORM_BRIDGEOS:
    return "bridgeos" + Version;
  case PLATFORM_MACCATALYST:
    return "ios" + Version + "-macabi";
  case PLATFORM_IOSSIMULATOR:
    return "ios" + Version + "-simulator";
  case PLATFORM_TVOSSIMULATOR:
    return "tvos" + Version + "-simulator";
  case PLATFORM_WATCHOSSIMULATOR:
    return "watchos" + Version + "-simulator";
  case PLATFORM_DRIVERKIT:
    return "driverkit" + Version;
  }
  llvm_unreachable("Unknown llvm::MachO::PlatformType enum");
}

} // end namespace MachO.
} // end namespace llvm.
