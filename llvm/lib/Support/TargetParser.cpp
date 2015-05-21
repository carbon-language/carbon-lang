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
  // Non-standard Arch names.
  { "iwmmxt",    ARM::AK_IWMMXT,   "iwmmxt",  ARMBuildAttrs::CPUArch::v5TE },
  { "iwmmxt2",   ARM::AK_IWMMXT2,  "iwmmxt2", ARMBuildAttrs::CPUArch::v5TE },
  { "xscale",    ARM::AK_XSCALE,   "xscale",  ARMBuildAttrs::CPUArch::v5TE },
  { "armv5e",    ARM::AK_ARMV5E,   "5E",      ARMBuildAttrs::CPUArch::v5TE },
  { "armv5tej",  ARM::AK_ARMV5TEJ, "5TE",     ARMBuildAttrs::CPUArch::v5TE },
  { "armv6sm",   ARM::AK_ARMV6SM,  "6-M",     ARMBuildAttrs::CPUArch::v6_M },
  { "armv6hl",   ARM::AK_ARMV6HL,  "6-M",     ARMBuildAttrs::CPUArch::v6_M },
  { "armv7e-m",  ARM::AK_ARMV7EM,  "7E-M",    ARMBuildAttrs::CPUArch::v7E_M },
  { "armv7l",    ARM::AK_ARMV7L,   "7-L",     ARMBuildAttrs::CPUArch::v7 },
  { "armv7hl",   ARM::AK_ARMV7HL,  "7H-L",    ARMBuildAttrs::CPUArch::v7 },
  { "armv7s",    ARM::AK_ARMV7S,   "7-S",     ARMBuildAttrs::CPUArch::v7 }
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
// List of CPU names and their arches.
// The same CPU can have multiple arches and can be default on multiple arches.
// When finding the Arch for a CPU, first-found prevails. Sort them accordingly.
// FIXME: TableGen this.
struct {
  const char *Name;
  ARM::ArchKind ArchID;
  bool Default;
} CPUNames[] = {
  { "arm2",          ARM::AK_ARMV2,    true },
  { "arm6",          ARM::AK_ARMV3,    true },
  { "arm7m",         ARM::AK_ARMV3M,   true },
  { "strongarm",     ARM::AK_ARMV4,    true },
  { "arm7tdmi",      ARM::AK_ARMV4T,   true },
  { "arm7tdmi-s",    ARM::AK_ARMV4T,   false },
  { "arm710t",       ARM::AK_ARMV4T,   false },
  { "arm720t",       ARM::AK_ARMV4T,   false },
  { "arm9",          ARM::AK_ARMV4T,   false },
  { "arm9tdmi",      ARM::AK_ARMV4T,   false },
  { "arm920",        ARM::AK_ARMV4T,   false },
  { "arm920t",       ARM::AK_ARMV4T,   false },
  { "arm922t",       ARM::AK_ARMV4T,   false },
  { "arm9312",       ARM::AK_ARMV4T,   false },
  { "arm940t",       ARM::AK_ARMV4T,   false },
  { "ep9312",        ARM::AK_ARMV4T,   false },
  { "arm10tdmi",     ARM::AK_ARMV5,    true },
  { "arm10tdmi",     ARM::AK_ARMV5T,   true },
  { "arm1020t",      ARM::AK_ARMV5T,   false },
  { "xscale",        ARM::AK_XSCALE,   true },
  { "xscale",        ARM::AK_ARMV5TE,  false },
  { "arm9e",         ARM::AK_ARMV5TE,  false },
  { "arm926ej-s",    ARM::AK_ARMV5TE,  false },
  { "arm946ej-s",    ARM::AK_ARMV5TE,  false },
  { "arm966e-s",     ARM::AK_ARMV5TE,  false },
  { "arm968e-s",     ARM::AK_ARMV5TE,  false },
  { "arm1020e",      ARM::AK_ARMV5TE,  false },
  { "arm1022e",      ARM::AK_ARMV5TE,  true },
  { "iwmmxt",        ARM::AK_ARMV5TE,  false },
  { "iwmmxt",        ARM::AK_IWMMXT,   true },
  { "arm1136jf-s",   ARM::AK_ARMV6,    true },
  { "arm1136j-s",    ARM::AK_ARMV6J,   true },
  { "arm1136jz-s",   ARM::AK_ARMV6J,   false },
  { "arm1176j-s",    ARM::AK_ARMV6K,   false },
  { "mpcore",        ARM::AK_ARMV6K,   false },
  { "mpcorenovfp",   ARM::AK_ARMV6K,   false },
  { "arm1176jzf-s",  ARM::AK_ARMV6K,   true },
  { "arm1176jzf-s",  ARM::AK_ARMV6Z,   true },
  { "arm1176jzf-s",  ARM::AK_ARMV6ZK,  true },
  { "arm1156t2-s",   ARM::AK_ARMV6T2,  true },
  { "arm1156t2f-s",  ARM::AK_ARMV6T2,  false },
  { "cortex-m0",     ARM::AK_ARMV6M,   true },
  { "cortex-m0plus", ARM::AK_ARMV6M,   false },
  { "cortex-m1",     ARM::AK_ARMV6M,   false },
  { "sc000",         ARM::AK_ARMV6M,   false },
  { "cortex-a8",     ARM::AK_ARMV7,    true },
  { "cortex-a5",     ARM::AK_ARMV7A,   false },
  { "cortex-a7",     ARM::AK_ARMV7A,   false },
  { "cortex-a8",     ARM::AK_ARMV7A,   true },
  { "cortex-a9",     ARM::AK_ARMV7A,   false },
  { "cortex-a12",    ARM::AK_ARMV7A,   false },
  { "cortex-a15",    ARM::AK_ARMV7A,   false },
  { "cortex-a17",    ARM::AK_ARMV7A,   false },
  { "krait",         ARM::AK_ARMV7A,   false },
  { "cortex-r4",     ARM::AK_ARMV7R,   true },
  { "cortex-r4f",    ARM::AK_ARMV7R,   false },
  { "cortex-r5",     ARM::AK_ARMV7R,   false },
  { "cortex-r7",     ARM::AK_ARMV7R,   false },
  { "sc300",         ARM::AK_ARMV7M,   false },
  { "cortex-m3",     ARM::AK_ARMV7M,   true },
  { "cortex-m4",     ARM::AK_ARMV7M,   false },
  { "cortex-m7",     ARM::AK_ARMV7M,   false },
  { "cortex-a53",    ARM::AK_ARMV8A,   true },
  { "cortex-a57",    ARM::AK_ARMV8A,   false },
  { "cortex-a72",    ARM::AK_ARMV8A,   false },
  { "cyclone",       ARM::AK_ARMV8A,   false },
  { "generic",       ARM::AK_ARMV8_1A, true },
  // Non-standard Arch names.
  { "arm1022e",      ARM::AK_ARMV5E,   true },
  { "arm926ej-s",    ARM::AK_ARMV5TEJ, true },
  { "cortex-m0",     ARM::AK_ARMV6SM,  true },
  { "arm1176jzf-s",  ARM::AK_ARMV6HL,  true },
  { "cortex-a8",     ARM::AK_ARMV7L,   true },
  { "cortex-a8",     ARM::AK_ARMV7HL,  true },
  { "cortex-m4",     ARM::AK_ARMV7EM,  true },
  { "swift",         ARM::AK_ARMV7S,   true },
  // Invalid CPU
  { "invalid",       ARM::AK_INVALID,  true }
};

} // namespace

namespace llvm {

// ======================================================= //
// Information by ID
// ======================================================= //

const char *ARMTargetParser::getFPUName(unsigned FPUKind) {
  if (FPUKind >= ARM::FK_LAST)
    return nullptr;
  return FPUNames[FPUKind].Name;
}

const char *ARMTargetParser::getArchName(unsigned ArchKind) {
  if (ArchKind >= ARM::AK_LAST)
    return nullptr;
  return ARCHNames[ArchKind].Name;
}

const char *ARMTargetParser::getArchDefaultCPUName(unsigned ArchKind) {
  if (ArchKind >= ARM::AK_LAST)
    return nullptr;
  return ARCHNames[ArchKind].DefaultCPU;
}

unsigned ARMTargetParser::getArchDefaultCPUArch(unsigned ArchKind) {
  if (ArchKind >= ARM::AK_LAST)
    return ARMBuildAttrs::CPUArch::Pre_v4;
  return ARCHNames[ArchKind].DefaultArch;
}

const char *ARMTargetParser::getArchExtName(unsigned ArchExtKind) {
  if (ArchExtKind >= ARM::AEK_LAST)
    return nullptr;
  return ARCHExtNames[ArchExtKind].Name;
}

const char *ARMTargetParser::getDefaultCPU(StringRef Arch) {
  unsigned AK = parseArch(Arch);
  if (AK == ARM::AK_INVALID)
    return nullptr;

  // Look for multiple AKs to find the default for pair AK+Name.
  for (const auto CPU : CPUNames) {
    if (CPU.ArchID == AK && CPU.Default)
      return CPU.Name;
  }
  return nullptr;
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
    .Cases("armv6m",   "v6m",   "armv6-m")
    .Cases("armv7a",   "v7a",   "armv7-a")
    .Cases("armv7r",   "v7r",   "armv7-r")
    .Cases("armv7m",   "v7m",   "armv7-m")
    .Cases("armv7em",  "v7em",  "armv7e-m")
    .Cases("armv8",    "v8",    "armv8-a")
    .Cases("armv8a",   "v8a",   "armv8-a")
    .Cases("armv8.1a", "v8.1a", "armv8.1-a")
    .Cases("aarch64",  "arm64", "armv8-a")
    .Default(Arch);
}

// MArch is expected to be of the form (arm|thumb)?(eb)?(v.+)?(eb)?, but
// (iwmmxt|xscale)(eb)? is also permitted. If the former, return
// "v.+", if the latter, return unmodified string. If invalid, return "".
StringRef ARMTargetParser::getCanonicalArchName(StringRef Arch) {
  size_t offset = StringRef::npos;
  StringRef A = Arch;
  StringRef Error = "";

  // Begins with "arm" / "thumb", move past it.
  if (A.startswith("arm"))
    offset = 3;
  else if (A.startswith("thumb"))
    offset = 5;
  else if (A.startswith("aarch64")) {
    offset = 7;
    // AArch64 uses "_be", not "eb" suffix.
    if (A.find("eb") != StringRef::npos)
      return Error;
    if (A.substr(offset,3) == "_be")
      offset += 3;
  }

  // Ex. "armebv7", move past the "eb".
  if (offset != StringRef::npos && A.substr(offset, 2) == "eb")
    offset += 2;
  // Or, if it ends with eb ("armv7eb"), chop it off.
  else if (A.endswith("eb"))
    A = A.substr(0, A.size() - 2);
  // Reached the end or a 'v', canonicalise.
  if (offset != StringRef::npos && (offset == A.size() || A[offset] == 'v'))
    A = A.substr(offset);

  // Empty string mans offset reached the end. Although valid, this arch
  // will not have a match in the table. Return the original string.
  if (A.empty())
    return Arch;

  // If can't find the arch, return an empty StringRef.
  if (parseArch(A) == ARM::AK_INVALID)
    return Error;

  // Arch will either be a 'v' name (v7a) or a marketing name (xscale)
  // or empty, if invalid.
  return A;
}

unsigned ARMTargetParser::parseFPU(StringRef FPU) {
  StringRef Syn = getFPUSynonym(FPU);
  for (const auto F : FPUNames) {
    if (Syn == F.Name)
      return F.ID;
  }
  return ARM::FK_INVALID;
}

// Allows partial match, ex. "v7a" matches "armv7a".
unsigned ARMTargetParser::parseArch(StringRef Arch) {
  StringRef Syn = getArchSynonym(Arch);
  for (const auto A : ARCHNames) {
    if (StringRef(A.Name).endswith(Syn))
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

unsigned ARMTargetParser::parseCPUArch(StringRef CPU) {
  for (const auto C : CPUNames) {
    if (CPU == C.Name)
      return C.ArchID;
  }
  return ARM::AK_INVALID;
}

// ARM, Thumb, AArch64
unsigned ARMTargetParser::parseArchISA(StringRef Arch) {
  return StringSwitch<unsigned>(Arch)
      .StartsWith("aarch64", ARM::IK_AARCH64)
      .StartsWith("arm64",   ARM::IK_AARCH64)
      .StartsWith("thumb",   ARM::IK_THUMB)
      .StartsWith("arm",     ARM::IK_ARM)
      .Default(ARM::EK_INVALID);
}

// Little/Big endian
unsigned ARMTargetParser::parseArchEndian(StringRef Arch) {
  if (Arch.startswith("armeb") ||
      Arch.startswith("thumbeb") ||
      Arch.startswith("aarch64_be"))
    return ARM::EK_BIG;

  if (Arch.startswith("arm") || Arch.startswith("thumb")) {
    if (Arch.endswith("eb"))
      return ARM::EK_BIG;
    else
      return ARM::EK_LITTLE;
  }

  if (Arch.startswith("aarch64"))
    return ARM::EK_LITTLE;

  return ARM::EK_INVALID;
}

} // namespace llvm
