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
  { "invalid",              ARM::FK_INVALID },
  { "vfp",                  ARM::FK_VFP },
  { "vfpv2",                ARM::FK_VFPV2 },
  { "vfpv3",                ARM::FK_VFPV3 },
  { "vfpv3-d16",            ARM::FK_VFPV3_D16 },
  { "vfpv4",                ARM::FK_VFPV4 },
  { "vfpv4-d16",            ARM::FK_VFPV4_D16 },
  { "fpv5-d16",             ARM::FK_FPV5_D16 },
  { "fp-armv8",             ARM::FK_FP_ARMV8 },
  { "neon",                 ARM::FK_NEON },
  { "neon-vfpv4",           ARM::FK_NEON_VFPV4 },
  { "neon-fp-armv8",        ARM::FK_NEON_FP_ARMV8 },
  { "crypto-neon-fp-armv8", ARM::FK_CRYPTO_NEON_FP_ARMV8 },
  { "softvfp",              ARM::FK_SOFTVFP }
};
// List of canonical arch names (use getArchSynonym)
// FIXME: TableGen this.
struct {
  const char *Name;
  ARM::ArchKind ID;
  const char *DefaultCPU;
  ARMBuildAttrs::CPUArch DefaultArch;
} ARCHNames[] = {
  { "invalid",   ARM::AK_INVALID,  nullptr,   ARMBuildAttrs::CPUArch::Pre_v4 },
  { "armv2",     ARM::AK_ARMV2,    "2",       ARMBuildAttrs::CPUArch::v4 },
  { "armv2a",    ARM::AK_ARMV2A,   "2A",      ARMBuildAttrs::CPUArch::v4 },
  { "armv3",     ARM::AK_ARMV3,    "3",       ARMBuildAttrs::CPUArch::v4 },
  { "armv3m",    ARM::AK_ARMV3M,   "3M",      ARMBuildAttrs::CPUArch::v4 },
  { "armv4",     ARM::AK_ARMV4,    "4",       ARMBuildAttrs::CPUArch::v4 },
  { "armv4t",    ARM::AK_ARMV4T,   "4T",      ARMBuildAttrs::CPUArch::v4T },
  { "armv5",     ARM::AK_ARMV5,    "5",       ARMBuildAttrs::CPUArch::v5T },
  { "armv5t",    ARM::AK_ARMV5T,   "5T",      ARMBuildAttrs::CPUArch::v5T },
  { "armv5te",   ARM::AK_ARMV5TE,  "5TE",     ARMBuildAttrs::CPUArch::v5TE },
  { "armv6",     ARM::AK_ARMV6,    "6",       ARMBuildAttrs::CPUArch::v6 },
  { "armv6j",    ARM::AK_ARMV6J,   "6J",      ARMBuildAttrs::CPUArch::v6 },
  { "armv6k",    ARM::AK_ARMV6K,   "6K",      ARMBuildAttrs::CPUArch::v6K },
  { "armv6t2",   ARM::AK_ARMV6T2,  "6T2",     ARMBuildAttrs::CPUArch::v6T2 },
  { "armv6z",    ARM::AK_ARMV6Z,   "6Z",      ARMBuildAttrs::CPUArch::v6KZ },
  { "armv6zk",   ARM::AK_ARMV6ZK,  "6ZK",     ARMBuildAttrs::CPUArch::v6KZ },
  { "armv6-m",   ARM::AK_ARMV6M,   "6-M",     ARMBuildAttrs::CPUArch::v6_M },
  { "armv7",     ARM::AK_ARMV7,    "7",       ARMBuildAttrs::CPUArch::v7 },
  { "armv7-a",   ARM::AK_ARMV7A,   "7-A",     ARMBuildAttrs::CPUArch::v7 },
  { "armv7-r",   ARM::AK_ARMV7R,   "7-R",     ARMBuildAttrs::CPUArch::v7 },
  { "armv7-m",   ARM::AK_ARMV7M,   "7-M",     ARMBuildAttrs::CPUArch::v7 },
  { "armv8-a",   ARM::AK_ARMV8A,   "8-A",     ARMBuildAttrs::CPUArch::v8 },
  { "armv8.1-a", ARM::AK_ARMV8_1A, "8.1-A",   ARMBuildAttrs::CPUArch::v8 },
  { "iwmmxt",    ARM::AK_IWMMXT,   "iwmmxt",  ARMBuildAttrs::CPUArch::v5TE },
  { "iwmmxt2",   ARM::AK_IWMMXT2,  "iwmmxt2", ARMBuildAttrs::CPUArch::v5TE }
};
// List of canonical ARCH names (use getARCHSynonym)
// FIXME: TableGen this.
struct {
  const char *Name;
  ARM::ArchExtKind ID;
} ARCHExtNames[] = {
  { "invalid",  ARM::AEK_INVALID },
  { "crc",      ARM::AEK_CRC },
  { "crypto",   ARM::AEK_CRYPTO },
  { "fp",       ARM::AEK_FP },
  { "idiv",     ARM::AEK_HWDIV },
  { "mp",       ARM::AEK_MP },
  { "sec",      ARM::AEK_SEC },
  { "virt",     ARM::AEK_VIRT }
};

} // namespace

namespace llvm {

// ======================================================= //
// Information by ID
// ======================================================= //

const char *ARMTargetParser::getFPUName(unsigned ID) {
  if (ID >= ARM::FK_LAST)
    return nullptr;
  return FPUNames[ID].Name;
}

const char *ARMTargetParser::getArchName(unsigned ID) {
  if (ID >= ARM::AK_LAST)
    return nullptr;
  return ARCHNames[ID].Name;
}

const char *ARMTargetParser::getArchDefaultCPUName(unsigned ID) {
  if (ID >= ARM::AK_LAST)
    return nullptr;
  return ARCHNames[ID].DefaultCPU;
}

unsigned ARMTargetParser::getArchDefaultCPUArch(unsigned ID) {
  if (ID >= ARM::AK_LAST)
    return 0;
  return ARCHNames[ID].DefaultArch;
}

const char *ARMTargetParser::getArchExtName(unsigned ID) {
  if (ID >= ARM::AEK_LAST)
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
  return ARM::FK_INVALID;
}

unsigned ARMTargetParser::parseArch(StringRef Arch) {
  StringRef Syn = getArchSynonym(Arch);
  for (const auto A : ARCHNames) {
    if (Syn == A.Name)
      return A.ID;
  }
  return ARM::AK_INVALID;
}

unsigned ARMTargetParser::parseArchExt(StringRef ArchExt) {
  for (const auto A : ARCHExtNames) {
    if (ArchExt == A.Name)
      return A.ID;
  }
  return ARM::AEK_INVALID;
}

} // namespace llvm
