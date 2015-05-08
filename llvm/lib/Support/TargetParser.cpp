//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;

namespace {

// List of canonical FPU names (use getFPUSynonym)
// FIXME: TableGen this.
struct {
  const char * Name;
  ARM::FPUKind ID;
} FPUNames[] = {
  { "invalid",              ARM::INVALID_FPU },
  { "vfp",                  ARM::VFP },
  { "vfpv2",                ARM::VFPV2 },
  { "vfpv3",                ARM::VFPV3 },
  { "vfpv3-d16",            ARM::VFPV3_D16 },
  { "vfpv4",                ARM::VFPV4 },
  { "vfpv4-d16",            ARM::VFPV4_D16 },
  { "fpv5-d16",             ARM::FPV5_D16 },
  { "fp-armv8",             ARM::FP_ARMV8 },
  { "neon",                 ARM::NEON },
  { "neon-vfpv4",           ARM::NEON_VFPV4 },
  { "neon-fp-armv8",        ARM::NEON_FP_ARMV8 },
  { "crypto-neon-fp-armv8", ARM::CRYPTO_NEON_FP_ARMV8 },
  { "softvfp",              ARM::SOFTVFP }
};
// List of canonical arch names (use getArchSynonym)
// FIXME: TableGen this.
struct {
  const char *Name;
  ARM::ArchKind ID;
  const char *DefaultCPU;
  ARMBuildAttrs::CPUArch DefaultArch;
} ARCHNames[] = {
  { "invalid",   ARM::INVALID_ARCH, nullptr, ARMBuildAttrs::CPUArch::Pre_v4 },
  { "armv2",     ARM::ARMV2,    "2",       ARMBuildAttrs::CPUArch::v4 },
  { "armv2a",    ARM::ARMV2A,   "2A",      ARMBuildAttrs::CPUArch::v4 },
  { "armv3",     ARM::ARMV3,    "3",       ARMBuildAttrs::CPUArch::v4 },
  { "armv3m",    ARM::ARMV3M,   "3M",      ARMBuildAttrs::CPUArch::v4 },
  { "armv4",     ARM::ARMV4,    "4",       ARMBuildAttrs::CPUArch::v4 },
  { "armv4t",    ARM::ARMV4T,   "4T",      ARMBuildAttrs::CPUArch::v4T },
  { "armv5",     ARM::ARMV5,    "5",       ARMBuildAttrs::CPUArch::v5T },
  { "armv5t",    ARM::ARMV5T,   "5T",      ARMBuildAttrs::CPUArch::v5T },
  { "armv5te",   ARM::ARMV5TE,  "5TE",     ARMBuildAttrs::CPUArch::v5TE },
  { "armv6",     ARM::ARMV6,    "6",       ARMBuildAttrs::CPUArch::v6 },
  { "armv6j",    ARM::ARMV6J,   "6J",      ARMBuildAttrs::CPUArch::v6 },
  { "armv6k",    ARM::ARMV6K,   "6K",      ARMBuildAttrs::CPUArch::v6K },
  { "armv6t2",   ARM::ARMV6T2,  "6T2",     ARMBuildAttrs::CPUArch::v6T2 },
  { "armv6z",    ARM::ARMV6Z,   "6Z",      ARMBuildAttrs::CPUArch::v6KZ },
  { "armv6zk",   ARM::ARMV6ZK,  "6ZK",     ARMBuildAttrs::CPUArch::v6KZ },
  { "armv6-m",   ARM::ARMV6M,   "6-M",     ARMBuildAttrs::CPUArch::v6_M },
  { "armv7",     ARM::ARMV7,    "7",       ARMBuildAttrs::CPUArch::v7 },
  { "armv7-a",   ARM::ARMV7A,   "7-A",     ARMBuildAttrs::CPUArch::v7 },
  { "armv7-r",   ARM::ARMV7R,   "7-R",     ARMBuildAttrs::CPUArch::v7 },
  { "armv7-m",   ARM::ARMV7M,   "7-M",     ARMBuildAttrs::CPUArch::v7 },
  { "armv8-a",   ARM::ARMV8A,   "8-A",     ARMBuildAttrs::CPUArch::v8 },
  { "armv8.1-a", ARM::ARMV8_1A, "8.1-A",   ARMBuildAttrs::CPUArch::v8 },
  { "iwmmxt",    ARM::IWMMXT,   "iwmmxt",  ARMBuildAttrs::CPUArch::v5TE },
  { "iwmmxt2",   ARM::IWMMXT2,  "iwmmxt2", ARMBuildAttrs::CPUArch::v5TE }
};
// List of canonical ARCH names (use getARCHSynonym)
// FIXME: TableGen this.
struct {
  const char *Name;
  ARM::ArchExtKind ID;
} ARCHExtNames[] = {
  { "invalid",  ARM::INVALID_ARCHEXT },
  { "crc",      ARM::CRC },
  { "crypto",   ARM::CRYPTO },
  { "fp",       ARM::FP },
  { "idiv",     ARM::HWDIV },
  { "mp",       ARM::MP },
  { "sec",      ARM::SEC },
  { "virt",     ARM::VIRT }
};

} // namespace

namespace llvm {

// ======================================================= //
// Information by ID
// ======================================================= //

const char *ARMTargetParser::getFPUName(unsigned ID) {
  if (ID >= ARM::LAST_FPU)
    return nullptr;
  return FPUNames[ID].Name;
}

const char *ARMTargetParser::getArchName(unsigned ID) {
  if (ID >= ARM::LAST_ARCH)
    return nullptr;
  return ARCHNames[ID].Name;
}

const char *ARMTargetParser::getArchDefaultCPUName(unsigned ID) {
  if (ID >= ARM::LAST_ARCH)
    return nullptr;
  return ARCHNames[ID].DefaultCPU;
}

unsigned ARMTargetParser::getArchDefaultCPUArch(unsigned ID) {
  if (ID >= ARM::LAST_ARCH)
    return 0;
  return ARCHNames[ID].DefaultArch;
}

const char *ARMTargetParser::getArchExtName(unsigned ID) {
  if (ID >= ARM::LAST_ARCHEXT)
    return nullptr;
  return ARCHExtNames[ID].Name;
}

// ======================================================= //
// Parsers
// ======================================================= //

StringRef ARMTargetParser::getFPUSynonym(StringRef FPU) {
  return StringSwitch<StringRef>(FPU)
    .Cases("fpa", "fpe2", "fpe3", "maverick", "invalid") // Unsupported
    .Case("vfp2", "vfpv2")
    .Case("vfp3", "vfpv3")
    .Case("vfp4", "vfpv4")
    .Case("vfp3-d16", "vfpv3-d16")
    .Case("vfp4-d16", "vfpv4-d16")
    // FIXME: sp-16 is NOT the same as d16
    .Cases("fp4-sp-d16", "fpv4-sp-d16", "vfpv4-d16")
    .Cases("fp4-dp-d16", "fpv4-dp-d16", "vfpv4-d16")
    .Cases("fp5-sp-d16", "fpv5-sp-d16", "fpv5-d16")
    .Cases("fp5-dp-d16", "fpv5-dp-d16", "fpv5-d16")
    // FIXME: Clang uses it, but it's bogus, since neon defaults to vfpv3.
    .Case("neon-vfpv3", "neon")
    .Default(FPU);
}

StringRef ARMTargetParser::getArchSynonym(StringRef Arch) {
  return StringSwitch<StringRef>(Arch)
    .Case("armv5tej", "armv5te")
    .Case("armv6m", "armv6-m")
    .Case("armv7a", "armv7-a")
    .Case("armv7r", "armv7-r")
    .Case("armv7m", "armv7-m")
    .Case("armv8a", "armv8-a")
    .Case("armv8.1a", "armv8.1-a")
    .Default(Arch);
}

unsigned ARMTargetParser::parseFPU(StringRef FPU) {
  StringRef Syn = getFPUSynonym(FPU);
  for (const auto F : FPUNames) {
    if (Syn == F.Name)
      return F.ID;
  }
  return ARM::INVALID_FPU;
}

unsigned ARMTargetParser::parseArch(StringRef Arch) {
  StringRef Syn = getArchSynonym(Arch);
  for (const auto A : ARCHNames) {
    if (Syn == A.Name)
      return A.ID;
  }
  return ARM::INVALID_ARCH;
}

unsigned ARMTargetParser::parseArchExt(StringRef ArchExt) {
  for (const auto A : ARCHExtNames) {
    if (ArchExt == A.Name)
      return A.ID;
  }
  return ARM::INVALID_ARCHEXT;
}

} // namespace llvm
