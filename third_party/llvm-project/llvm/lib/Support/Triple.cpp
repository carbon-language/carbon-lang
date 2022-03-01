//===--- Triple.cpp - Target triple helper class --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/ADT/STLArrayExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ARMTargetParser.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/VersionTuple.h"
#include <cassert>
#include <cstring>
using namespace llvm;

StringRef Triple::getArchTypeName(ArchType Kind) {
  switch (Kind) {
  case UnknownArch:    return "unknown";

  case aarch64:        return "aarch64";
  case aarch64_32:     return "aarch64_32";
  case aarch64_be:     return "aarch64_be";
  case amdgcn:         return "amdgcn";
  case amdil64:        return "amdil64";
  case amdil:          return "amdil";
  case arc:            return "arc";
  case arm:            return "arm";
  case armeb:          return "armeb";
  case avr:            return "avr";
  case bpfeb:          return "bpfeb";
  case bpfel:          return "bpfel";
  case csky:           return "csky";
  case hexagon:        return "hexagon";
  case hsail64:        return "hsail64";
  case hsail:          return "hsail";
  case kalimba:        return "kalimba";
  case lanai:          return "lanai";
  case le32:           return "le32";
  case le64:           return "le64";
  case loongarch32:    return "loongarch32";
  case loongarch64:    return "loongarch64";
  case m68k:           return "m68k";
  case mips64:         return "mips64";
  case mips64el:       return "mips64el";
  case mips:           return "mips";
  case mipsel:         return "mipsel";
  case msp430:         return "msp430";
  case nvptx64:        return "nvptx64";
  case nvptx:          return "nvptx";
  case ppc64:          return "powerpc64";
  case ppc64le:        return "powerpc64le";
  case ppc:            return "powerpc";
  case ppcle:          return "powerpcle";
  case r600:           return "r600";
  case renderscript32: return "renderscript32";
  case renderscript64: return "renderscript64";
  case riscv32:        return "riscv32";
  case riscv64:        return "riscv64";
  case shave:          return "shave";
  case sparc:          return "sparc";
  case sparcel:        return "sparcel";
  case sparcv9:        return "sparcv9";
  case spir64:         return "spir64";
  case spir:           return "spir";
  case spirv32:        return "spirv32";
  case spirv64:        return "spirv64";
  case systemz:        return "s390x";
  case tce:            return "tce";
  case tcele:          return "tcele";
  case thumb:          return "thumb";
  case thumbeb:        return "thumbeb";
  case ve:             return "ve";
  case wasm32:         return "wasm32";
  case wasm64:         return "wasm64";
  case x86:            return "i386";
  case x86_64:         return "x86_64";
  case xcore:          return "xcore";
  }

  llvm_unreachable("Invalid ArchType!");
}

StringRef Triple::getArchTypePrefix(ArchType Kind) {
  switch (Kind) {
  default:
    return StringRef();

  case aarch64:
  case aarch64_be:
  case aarch64_32:  return "aarch64";

  case arc:         return "arc";

  case arm:
  case armeb:
  case thumb:
  case thumbeb:     return "arm";

  case avr:         return "avr";

  case ppc64:
  case ppc64le:
  case ppc:
  case ppcle:       return "ppc";

  case m68k:        return "m68k";

  case mips:
  case mipsel:
  case mips64:
  case mips64el:    return "mips";

  case hexagon:     return "hexagon";

  case amdgcn:      return "amdgcn";
  case r600:        return "r600";

  case bpfel:
  case bpfeb:       return "bpf";

  case sparcv9:
  case sparcel:
  case sparc:       return "sparc";

  case systemz:     return "s390";

  case x86:
  case x86_64:      return "x86";

  case xcore:       return "xcore";

  // NVPTX intrinsics are namespaced under nvvm.
  case nvptx:       return "nvvm";
  case nvptx64:     return "nvvm";

  case le32:        return "le32";
  case le64:        return "le64";

  case amdil:
  case amdil64:     return "amdil";

  case hsail:
  case hsail64:     return "hsail";

  case spir:
  case spir64:      return "spir";

  case spirv32:
  case spirv64:     return "spirv";

  case kalimba:     return "kalimba";
  case lanai:       return "lanai";
  case shave:       return "shave";
  case wasm32:
  case wasm64:      return "wasm";

  case riscv32:
  case riscv64:     return "riscv";

  case ve:          return "ve";
  case csky:        return "csky";

  case loongarch32:
  case loongarch64: return "loongarch";
  }
}

StringRef Triple::getVendorTypeName(VendorType Kind) {
  switch (Kind) {
  case UnknownVendor: return "unknown";

  case AMD: return "amd";
  case Apple: return "apple";
  case CSR: return "csr";
  case Freescale: return "fsl";
  case IBM: return "ibm";
  case ImaginationTechnologies: return "img";
  case Mesa: return "mesa";
  case MipsTechnologies: return "mti";
  case Myriad: return "myriad";
  case NVIDIA: return "nvidia";
  case OpenEmbedded: return "oe";
  case PC: return "pc";
  case SCEI: return "scei";
  case SUSE: return "suse";
  }

  llvm_unreachable("Invalid VendorType!");
}

StringRef Triple::getOSTypeName(OSType Kind) {
  switch (Kind) {
  case UnknownOS: return "unknown";

  case AIX: return "aix";
  case AMDHSA: return "amdhsa";
  case AMDPAL: return "amdpal";
  case Ananas: return "ananas";
  case CUDA: return "cuda";
  case CloudABI: return "cloudabi";
  case Contiki: return "contiki";
  case Darwin: return "darwin";
  case DragonFly: return "dragonfly";
  case DriverKit: return "driverkit";
  case ELFIAMCU: return "elfiamcu";
  case Emscripten: return "emscripten";
  case FreeBSD: return "freebsd";
  case Fuchsia: return "fuchsia";
  case Haiku: return "haiku";
  case HermitCore: return "hermit";
  case Hurd: return "hurd";
  case IOS: return "ios";
  case KFreeBSD: return "kfreebsd";
  case Linux: return "linux";
  case Lv2: return "lv2";
  case MacOSX: return "macosx";
  case Mesa3D: return "mesa3d";
  case Minix: return "minix";
  case NVCL: return "nvcl";
  case NaCl: return "nacl";
  case NetBSD: return "netbsd";
  case OpenBSD: return "openbsd";
  case PS4: return "ps4";
  case RTEMS: return "rtems";
  case Solaris: return "solaris";
  case TvOS: return "tvos";
  case WASI: return "wasi";
  case WatchOS: return "watchos";
  case Win32: return "windows";
  case ZOS: return "zos";
  }

  llvm_unreachable("Invalid OSType");
}

StringRef Triple::getEnvironmentTypeName(EnvironmentType Kind) {
  switch (Kind) {
  case UnknownEnvironment: return "unknown";
  case Android: return "android";
  case CODE16: return "code16";
  case CoreCLR: return "coreclr";
  case Cygnus: return "cygnus";
  case EABI: return "eabi";
  case EABIHF: return "eabihf";
  case GNU: return "gnu";
  case GNUABI64: return "gnuabi64";
  case GNUABIN32: return "gnuabin32";
  case GNUEABI: return "gnueabi";
  case GNUEABIHF: return "gnueabihf";
  case GNUX32: return "gnux32";
  case GNUILP32: return "gnu_ilp32";
  case Itanium: return "itanium";
  case MSVC: return "msvc";
  case MacABI: return "macabi";
  case Musl: return "musl";
  case MuslEABI: return "musleabi";
  case MuslEABIHF: return "musleabihf";
  case MuslX32: return "muslx32";
  case Simulator: return "simulator";
  }

  llvm_unreachable("Invalid EnvironmentType!");
}

static Triple::ArchType parseBPFArch(StringRef ArchName) {
  if (ArchName.equals("bpf")) {
    if (sys::IsLittleEndianHost)
      return Triple::bpfel;
    else
      return Triple::bpfeb;
  } else if (ArchName.equals("bpf_be") || ArchName.equals("bpfeb")) {
    return Triple::bpfeb;
  } else if (ArchName.equals("bpf_le") || ArchName.equals("bpfel")) {
    return Triple::bpfel;
  } else {
    return Triple::UnknownArch;
  }
}

Triple::ArchType Triple::getArchTypeForLLVMName(StringRef Name) {
  Triple::ArchType BPFArch(parseBPFArch(Name));
  return StringSwitch<Triple::ArchType>(Name)
    .Case("aarch64", aarch64)
    .Case("aarch64_be", aarch64_be)
    .Case("aarch64_32", aarch64_32)
    .Case("arc", arc)
    .Case("arm64", aarch64) // "arm64" is an alias for "aarch64"
    .Case("arm64_32", aarch64_32)
    .Case("arm", arm)
    .Case("armeb", armeb)
    .Case("avr", avr)
    .StartsWith("bpf", BPFArch)
    .Case("m68k", m68k)
    .Case("mips", mips)
    .Case("mipsel", mipsel)
    .Case("mips64", mips64)
    .Case("mips64el", mips64el)
    .Case("msp430", msp430)
    .Case("ppc64", ppc64)
    .Case("ppc32", ppc)
    .Case("ppc", ppc)
    .Case("ppc32le", ppcle)
    .Case("ppcle", ppcle)
    .Case("ppc64le", ppc64le)
    .Case("r600", r600)
    .Case("amdgcn", amdgcn)
    .Case("riscv32", riscv32)
    .Case("riscv64", riscv64)
    .Case("hexagon", hexagon)
    .Case("sparc", sparc)
    .Case("sparcel", sparcel)
    .Case("sparcv9", sparcv9)
    .Case("systemz", systemz)
    .Case("tce", tce)
    .Case("tcele", tcele)
    .Case("thumb", thumb)
    .Case("thumbeb", thumbeb)
    .Case("x86", x86)
    .Case("x86-64", x86_64)
    .Case("xcore", xcore)
    .Case("nvptx", nvptx)
    .Case("nvptx64", nvptx64)
    .Case("le32", le32)
    .Case("le64", le64)
    .Case("amdil", amdil)
    .Case("amdil64", amdil64)
    .Case("hsail", hsail)
    .Case("hsail64", hsail64)
    .Case("spir", spir)
    .Case("spir64", spir64)
    .Case("spirv32", spirv32)
    .Case("spirv64", spirv64)
    .Case("kalimba", kalimba)
    .Case("lanai", lanai)
    .Case("shave", shave)
    .Case("wasm32", wasm32)
    .Case("wasm64", wasm64)
    .Case("renderscript32", renderscript32)
    .Case("renderscript64", renderscript64)
    .Case("ve", ve)
    .Case("csky", csky)
    .Case("loongarch32", loongarch32)
    .Case("loongarch64", loongarch64)
    .Default(UnknownArch);
}

static Triple::ArchType parseARMArch(StringRef ArchName) {
  ARM::ISAKind ISA = ARM::parseArchISA(ArchName);
  ARM::EndianKind ENDIAN = ARM::parseArchEndian(ArchName);

  Triple::ArchType arch = Triple::UnknownArch;
  switch (ENDIAN) {
  case ARM::EndianKind::LITTLE: {
    switch (ISA) {
    case ARM::ISAKind::ARM:
      arch = Triple::arm;
      break;
    case ARM::ISAKind::THUMB:
      arch = Triple::thumb;
      break;
    case ARM::ISAKind::AARCH64:
      arch = Triple::aarch64;
      break;
    case ARM::ISAKind::INVALID:
      break;
    }
    break;
  }
  case ARM::EndianKind::BIG: {
    switch (ISA) {
    case ARM::ISAKind::ARM:
      arch = Triple::armeb;
      break;
    case ARM::ISAKind::THUMB:
      arch = Triple::thumbeb;
      break;
    case ARM::ISAKind::AARCH64:
      arch = Triple::aarch64_be;
      break;
    case ARM::ISAKind::INVALID:
      break;
    }
    break;
  }
  case ARM::EndianKind::INVALID: {
    break;
  }
  }

  ArchName = ARM::getCanonicalArchName(ArchName);
  if (ArchName.empty())
    return Triple::UnknownArch;

  // Thumb only exists in v4+
  if (ISA == ARM::ISAKind::THUMB &&
      (ArchName.startswith("v2") || ArchName.startswith("v3")))
    return Triple::UnknownArch;

  // Thumb only for v6m
  ARM::ProfileKind Profile = ARM::parseArchProfile(ArchName);
  unsigned Version = ARM::parseArchVersion(ArchName);
  if (Profile == ARM::ProfileKind::M && Version == 6) {
    if (ENDIAN == ARM::EndianKind::BIG)
      return Triple::thumbeb;
    else
      return Triple::thumb;
  }

  return arch;
}

static Triple::ArchType parseArch(StringRef ArchName) {
  auto AT = StringSwitch<Triple::ArchType>(ArchName)
    .Cases("i386", "i486", "i586", "i686", Triple::x86)
    // FIXME: Do we need to support these?
    .Cases("i786", "i886", "i986", Triple::x86)
    .Cases("amd64", "x86_64", "x86_64h", Triple::x86_64)
    .Cases("powerpc", "powerpcspe", "ppc", "ppc32", Triple::ppc)
    .Cases("powerpcle", "ppcle", "ppc32le", Triple::ppcle)
    .Cases("powerpc64", "ppu", "ppc64", Triple::ppc64)
    .Cases("powerpc64le", "ppc64le", Triple::ppc64le)
    .Case("xscale", Triple::arm)
    .Case("xscaleeb", Triple::armeb)
    .Case("aarch64", Triple::aarch64)
    .Case("aarch64_be", Triple::aarch64_be)
    .Case("aarch64_32", Triple::aarch64_32)
    .Case("arc", Triple::arc)
    .Case("arm64", Triple::aarch64)
    .Case("arm64_32", Triple::aarch64_32)
    .Case("arm64e", Triple::aarch64)
    .Case("arm", Triple::arm)
    .Case("armeb", Triple::armeb)
    .Case("thumb", Triple::thumb)
    .Case("thumbeb", Triple::thumbeb)
    .Case("avr", Triple::avr)
    .Case("m68k", Triple::m68k)
    .Case("msp430", Triple::msp430)
    .Cases("mips", "mipseb", "mipsallegrex", "mipsisa32r6",
           "mipsr6", Triple::mips)
    .Cases("mipsel", "mipsallegrexel", "mipsisa32r6el", "mipsr6el",
           Triple::mipsel)
    .Cases("mips64", "mips64eb", "mipsn32", "mipsisa64r6",
           "mips64r6", "mipsn32r6", Triple::mips64)
    .Cases("mips64el", "mipsn32el", "mipsisa64r6el", "mips64r6el",
           "mipsn32r6el", Triple::mips64el)
    .Case("r600", Triple::r600)
    .Case("amdgcn", Triple::amdgcn)
    .Case("riscv32", Triple::riscv32)
    .Case("riscv64", Triple::riscv64)
    .Case("hexagon", Triple::hexagon)
    .Cases("s390x", "systemz", Triple::systemz)
    .Case("sparc", Triple::sparc)
    .Case("sparcel", Triple::sparcel)
    .Cases("sparcv9", "sparc64", Triple::sparcv9)
    .Case("tce", Triple::tce)
    .Case("tcele", Triple::tcele)
    .Case("xcore", Triple::xcore)
    .Case("nvptx", Triple::nvptx)
    .Case("nvptx64", Triple::nvptx64)
    .Case("le32", Triple::le32)
    .Case("le64", Triple::le64)
    .Case("amdil", Triple::amdil)
    .Case("amdil64", Triple::amdil64)
    .Case("hsail", Triple::hsail)
    .Case("hsail64", Triple::hsail64)
    .Case("spir", Triple::spir)
    .Case("spir64", Triple::spir64)
    .Case("spirv32", Triple::spirv32)
    .Case("spirv64", Triple::spirv64)
    .StartsWith("kalimba", Triple::kalimba)
    .Case("lanai", Triple::lanai)
    .Case("renderscript32", Triple::renderscript32)
    .Case("renderscript64", Triple::renderscript64)
    .Case("shave", Triple::shave)
    .Case("ve", Triple::ve)
    .Case("wasm32", Triple::wasm32)
    .Case("wasm64", Triple::wasm64)
    .Case("csky", Triple::csky)
    .Case("loongarch32", Triple::loongarch32)
    .Case("loongarch64", Triple::loongarch64)
    .Default(Triple::UnknownArch);

  // Some architectures require special parsing logic just to compute the
  // ArchType result.
  if (AT == Triple::UnknownArch) {
    if (ArchName.startswith("arm") || ArchName.startswith("thumb") ||
        ArchName.startswith("aarch64"))
      return parseARMArch(ArchName);
    if (ArchName.startswith("bpf"))
      return parseBPFArch(ArchName);
  }

  return AT;
}

static Triple::VendorType parseVendor(StringRef VendorName) {
  return StringSwitch<Triple::VendorType>(VendorName)
    .Case("apple", Triple::Apple)
    .Case("pc", Triple::PC)
    .Case("scei", Triple::SCEI)
    .Case("sie", Triple::SCEI)
    .Case("fsl", Triple::Freescale)
    .Case("ibm", Triple::IBM)
    .Case("img", Triple::ImaginationTechnologies)
    .Case("mti", Triple::MipsTechnologies)
    .Case("nvidia", Triple::NVIDIA)
    .Case("csr", Triple::CSR)
    .Case("myriad", Triple::Myriad)
    .Case("amd", Triple::AMD)
    .Case("mesa", Triple::Mesa)
    .Case("suse", Triple::SUSE)
    .Case("oe", Triple::OpenEmbedded)
    .Default(Triple::UnknownVendor);
}

static Triple::OSType parseOS(StringRef OSName) {
  return StringSwitch<Triple::OSType>(OSName)
    .StartsWith("ananas", Triple::Ananas)
    .StartsWith("cloudabi", Triple::CloudABI)
    .StartsWith("darwin", Triple::Darwin)
    .StartsWith("dragonfly", Triple::DragonFly)
    .StartsWith("freebsd", Triple::FreeBSD)
    .StartsWith("fuchsia", Triple::Fuchsia)
    .StartsWith("ios", Triple::IOS)
    .StartsWith("kfreebsd", Triple::KFreeBSD)
    .StartsWith("linux", Triple::Linux)
    .StartsWith("lv2", Triple::Lv2)
    .StartsWith("macos", Triple::MacOSX)
    .StartsWith("netbsd", Triple::NetBSD)
    .StartsWith("openbsd", Triple::OpenBSD)
    .StartsWith("solaris", Triple::Solaris)
    .StartsWith("win32", Triple::Win32)
    .StartsWith("windows", Triple::Win32)
    .StartsWith("zos", Triple::ZOS)
    .StartsWith("haiku", Triple::Haiku)
    .StartsWith("minix", Triple::Minix)
    .StartsWith("rtems", Triple::RTEMS)
    .StartsWith("nacl", Triple::NaCl)
    .StartsWith("aix", Triple::AIX)
    .StartsWith("cuda", Triple::CUDA)
    .StartsWith("nvcl", Triple::NVCL)
    .StartsWith("amdhsa", Triple::AMDHSA)
    .StartsWith("ps4", Triple::PS4)
    .StartsWith("elfiamcu", Triple::ELFIAMCU)
    .StartsWith("tvos", Triple::TvOS)
    .StartsWith("watchos", Triple::WatchOS)
    .StartsWith("driverkit", Triple::DriverKit)
    .StartsWith("mesa3d", Triple::Mesa3D)
    .StartsWith("contiki", Triple::Contiki)
    .StartsWith("amdpal", Triple::AMDPAL)
    .StartsWith("hermit", Triple::HermitCore)
    .StartsWith("hurd", Triple::Hurd)
    .StartsWith("wasi", Triple::WASI)
    .StartsWith("emscripten", Triple::Emscripten)
    .Default(Triple::UnknownOS);
}

static Triple::EnvironmentType parseEnvironment(StringRef EnvironmentName) {
  return StringSwitch<Triple::EnvironmentType>(EnvironmentName)
      .StartsWith("eabihf", Triple::EABIHF)
      .StartsWith("eabi", Triple::EABI)
      .StartsWith("gnuabin32", Triple::GNUABIN32)
      .StartsWith("gnuabi64", Triple::GNUABI64)
      .StartsWith("gnueabihf", Triple::GNUEABIHF)
      .StartsWith("gnueabi", Triple::GNUEABI)
      .StartsWith("gnux32", Triple::GNUX32)
      .StartsWith("gnu_ilp32", Triple::GNUILP32)
      .StartsWith("code16", Triple::CODE16)
      .StartsWith("gnu", Triple::GNU)
      .StartsWith("android", Triple::Android)
      .StartsWith("musleabihf", Triple::MuslEABIHF)
      .StartsWith("musleabi", Triple::MuslEABI)
      .StartsWith("muslx32", Triple::MuslX32)
      .StartsWith("musl", Triple::Musl)
      .StartsWith("msvc", Triple::MSVC)
      .StartsWith("itanium", Triple::Itanium)
      .StartsWith("cygnus", Triple::Cygnus)
      .StartsWith("coreclr", Triple::CoreCLR)
      .StartsWith("simulator", Triple::Simulator)
      .StartsWith("macabi", Triple::MacABI)
      .Default(Triple::UnknownEnvironment);
}

static Triple::ObjectFormatType parseFormat(StringRef EnvironmentName) {
  return StringSwitch<Triple::ObjectFormatType>(EnvironmentName)
    // "xcoff" must come before "coff" because of the order-dependendent
    // pattern matching.
    .EndsWith("xcoff", Triple::XCOFF)
    .EndsWith("coff", Triple::COFF)
    .EndsWith("elf", Triple::ELF)
    .EndsWith("goff", Triple::GOFF)
    .EndsWith("macho", Triple::MachO)
    .EndsWith("wasm", Triple::Wasm)
    .Default(Triple::UnknownObjectFormat);
}

static Triple::SubArchType parseSubArch(StringRef SubArchName) {
  if (SubArchName.startswith("mips") &&
      (SubArchName.endswith("r6el") || SubArchName.endswith("r6")))
    return Triple::MipsSubArch_r6;

  if (SubArchName == "powerpcspe")
    return Triple::PPCSubArch_spe;

  if (SubArchName == "arm64e")
    return Triple::AArch64SubArch_arm64e;

  StringRef ARMSubArch = ARM::getCanonicalArchName(SubArchName);

  // For now, this is the small part. Early return.
  if (ARMSubArch.empty())
    return StringSwitch<Triple::SubArchType>(SubArchName)
      .EndsWith("kalimba3", Triple::KalimbaSubArch_v3)
      .EndsWith("kalimba4", Triple::KalimbaSubArch_v4)
      .EndsWith("kalimba5", Triple::KalimbaSubArch_v5)
      .Default(Triple::NoSubArch);

  // ARM sub arch.
  switch(ARM::parseArch(ARMSubArch)) {
  case ARM::ArchKind::ARMV4:
    return Triple::NoSubArch;
  case ARM::ArchKind::ARMV4T:
    return Triple::ARMSubArch_v4t;
  case ARM::ArchKind::ARMV5T:
    return Triple::ARMSubArch_v5;
  case ARM::ArchKind::ARMV5TE:
  case ARM::ArchKind::IWMMXT:
  case ARM::ArchKind::IWMMXT2:
  case ARM::ArchKind::XSCALE:
  case ARM::ArchKind::ARMV5TEJ:
    return Triple::ARMSubArch_v5te;
  case ARM::ArchKind::ARMV6:
    return Triple::ARMSubArch_v6;
  case ARM::ArchKind::ARMV6K:
  case ARM::ArchKind::ARMV6KZ:
    return Triple::ARMSubArch_v6k;
  case ARM::ArchKind::ARMV6T2:
    return Triple::ARMSubArch_v6t2;
  case ARM::ArchKind::ARMV6M:
    return Triple::ARMSubArch_v6m;
  case ARM::ArchKind::ARMV7A:
  case ARM::ArchKind::ARMV7R:
    return Triple::ARMSubArch_v7;
  case ARM::ArchKind::ARMV7VE:
    return Triple::ARMSubArch_v7ve;
  case ARM::ArchKind::ARMV7K:
    return Triple::ARMSubArch_v7k;
  case ARM::ArchKind::ARMV7M:
    return Triple::ARMSubArch_v7m;
  case ARM::ArchKind::ARMV7S:
    return Triple::ARMSubArch_v7s;
  case ARM::ArchKind::ARMV7EM:
    return Triple::ARMSubArch_v7em;
  case ARM::ArchKind::ARMV8A:
    return Triple::ARMSubArch_v8;
  case ARM::ArchKind::ARMV8_1A:
    return Triple::ARMSubArch_v8_1a;
  case ARM::ArchKind::ARMV8_2A:
    return Triple::ARMSubArch_v8_2a;
  case ARM::ArchKind::ARMV8_3A:
    return Triple::ARMSubArch_v8_3a;
  case ARM::ArchKind::ARMV8_4A:
    return Triple::ARMSubArch_v8_4a;
  case ARM::ArchKind::ARMV8_5A:
    return Triple::ARMSubArch_v8_5a;
  case ARM::ArchKind::ARMV8_6A:
    return Triple::ARMSubArch_v8_6a;
  case ARM::ArchKind::ARMV8_7A:
    return Triple::ARMSubArch_v8_7a;
  case ARM::ArchKind::ARMV8_8A:
    return Triple::ARMSubArch_v8_8a;
  case ARM::ArchKind::ARMV9A:
    return Triple::ARMSubArch_v9;
  case ARM::ArchKind::ARMV9_1A:
    return Triple::ARMSubArch_v9_1a;
  case ARM::ArchKind::ARMV9_2A:
    return Triple::ARMSubArch_v9_2a;
  case ARM::ArchKind::ARMV9_3A:
    return Triple::ARMSubArch_v9_3a;
  case ARM::ArchKind::ARMV8R:
    return Triple::ARMSubArch_v8r;
  case ARM::ArchKind::ARMV8MBaseline:
    return Triple::ARMSubArch_v8m_baseline;
  case ARM::ArchKind::ARMV8MMainline:
    return Triple::ARMSubArch_v8m_mainline;
  case ARM::ArchKind::ARMV8_1MMainline:
    return Triple::ARMSubArch_v8_1m_mainline;
  default:
    return Triple::NoSubArch;
  }
}

static StringRef getObjectFormatTypeName(Triple::ObjectFormatType Kind) {
  switch (Kind) {
  case Triple::UnknownObjectFormat: return "";
  case Triple::COFF:  return "coff";
  case Triple::ELF:   return "elf";
  case Triple::GOFF:  return "goff";
  case Triple::MachO: return "macho";
  case Triple::Wasm:  return "wasm";
  case Triple::XCOFF: return "xcoff";
  }
  llvm_unreachable("unknown object format type");
}

static Triple::ObjectFormatType getDefaultFormat(const Triple &T) {
  switch (T.getArch()) {
  case Triple::UnknownArch:
  case Triple::aarch64:
  case Triple::aarch64_32:
  case Triple::arm:
  case Triple::thumb:
  case Triple::x86:
  case Triple::x86_64:
    if (T.isOSDarwin())
      return Triple::MachO;
    else if (T.isOSWindows())
      return Triple::COFF;
    return Triple::ELF;

  case Triple::aarch64_be:
  case Triple::amdgcn:
  case Triple::amdil64:
  case Triple::amdil:
  case Triple::arc:
  case Triple::armeb:
  case Triple::avr:
  case Triple::bpfeb:
  case Triple::bpfel:
  case Triple::csky:
  case Triple::hexagon:
  case Triple::hsail64:
  case Triple::hsail:
  case Triple::kalimba:
  case Triple::lanai:
  case Triple::le32:
  case Triple::le64:
  case Triple::loongarch32:
  case Triple::loongarch64:
  case Triple::m68k:
  case Triple::mips64:
  case Triple::mips64el:
  case Triple::mips:
  case Triple::mipsel:
  case Triple::msp430:
  case Triple::nvptx64:
  case Triple::nvptx:
  case Triple::ppc64le:
  case Triple::ppcle:
  case Triple::r600:
  case Triple::renderscript32:
  case Triple::renderscript64:
  case Triple::riscv32:
  case Triple::riscv64:
  case Triple::shave:
  case Triple::sparc:
  case Triple::sparcel:
  case Triple::sparcv9:
  case Triple::spir64:
  case Triple::spir:
  case Triple::tce:
  case Triple::tcele:
  case Triple::thumbeb:
  case Triple::ve:
  case Triple::xcore:
    return Triple::ELF;

  case Triple::ppc64:
  case Triple::ppc:
    if (T.isOSAIX())
      return Triple::XCOFF;
    return Triple::ELF;

  case Triple::systemz:
    if (T.isOSzOS())
      return Triple::GOFF;
    return Triple::ELF;

  case Triple::wasm32:
  case Triple::wasm64:
    return Triple::Wasm;

  case Triple::spirv32:
  case Triple::spirv64:
    // TODO: In future this will be Triple::SPIRV.
    return Triple::UnknownObjectFormat;
  }
  llvm_unreachable("unknown architecture");
}

/// Construct a triple from the string representation provided.
///
/// This stores the string representation and parses the various pieces into
/// enum members.
Triple::Triple(const Twine &Str)
    : Data(Str.str()), Arch(UnknownArch), SubArch(NoSubArch),
      Vendor(UnknownVendor), OS(UnknownOS), Environment(UnknownEnvironment),
      ObjectFormat(UnknownObjectFormat) {
  // Do minimal parsing by hand here.
  SmallVector<StringRef, 4> Components;
  StringRef(Data).split(Components, '-', /*MaxSplit*/ 3);
  if (Components.size() > 0) {
    Arch = parseArch(Components[0]);
    SubArch = parseSubArch(Components[0]);
    if (Components.size() > 1) {
      Vendor = parseVendor(Components[1]);
      if (Components.size() > 2) {
        OS = parseOS(Components[2]);
        if (Components.size() > 3) {
          Environment = parseEnvironment(Components[3]);
          ObjectFormat = parseFormat(Components[3]);
        }
      }
    } else {
      Environment =
          StringSwitch<Triple::EnvironmentType>(Components[0])
              .StartsWith("mipsn32", Triple::GNUABIN32)
              .StartsWith("mips64", Triple::GNUABI64)
              .StartsWith("mipsisa64", Triple::GNUABI64)
              .StartsWith("mipsisa32", Triple::GNU)
              .Cases("mips", "mipsel", "mipsr6", "mipsr6el", Triple::GNU)
              .Default(UnknownEnvironment);
    }
  }
  if (ObjectFormat == UnknownObjectFormat)
    ObjectFormat = getDefaultFormat(*this);
}

/// Construct a triple from string representations of the architecture,
/// vendor, and OS.
///
/// This joins each argument into a canonical string representation and parses
/// them into enum members. It leaves the environment unknown and omits it from
/// the string representation.
Triple::Triple(const Twine &ArchStr, const Twine &VendorStr, const Twine &OSStr)
    : Data((ArchStr + Twine('-') + VendorStr + Twine('-') + OSStr).str()),
      Arch(parseArch(ArchStr.str())),
      SubArch(parseSubArch(ArchStr.str())),
      Vendor(parseVendor(VendorStr.str())),
      OS(parseOS(OSStr.str())),
      Environment(), ObjectFormat(Triple::UnknownObjectFormat) {
  ObjectFormat = getDefaultFormat(*this);
}

/// Construct a triple from string representations of the architecture,
/// vendor, OS, and environment.
///
/// This joins each argument into a canonical string representation and parses
/// them into enum members.
Triple::Triple(const Twine &ArchStr, const Twine &VendorStr, const Twine &OSStr,
               const Twine &EnvironmentStr)
    : Data((ArchStr + Twine('-') + VendorStr + Twine('-') + OSStr + Twine('-') +
            EnvironmentStr).str()),
      Arch(parseArch(ArchStr.str())),
      SubArch(parseSubArch(ArchStr.str())),
      Vendor(parseVendor(VendorStr.str())),
      OS(parseOS(OSStr.str())),
      Environment(parseEnvironment(EnvironmentStr.str())),
      ObjectFormat(parseFormat(EnvironmentStr.str())) {
  if (ObjectFormat == Triple::UnknownObjectFormat)
    ObjectFormat = getDefaultFormat(*this);
}

std::string Triple::normalize(StringRef Str) {
  bool IsMinGW32 = false;
  bool IsCygwin = false;

  // Parse into components.
  SmallVector<StringRef, 4> Components;
  Str.split(Components, '-');

  // If the first component corresponds to a known architecture, preferentially
  // use it for the architecture.  If the second component corresponds to a
  // known vendor, preferentially use it for the vendor, etc.  This avoids silly
  // component movement when a component parses as (eg) both a valid arch and a
  // valid os.
  ArchType Arch = UnknownArch;
  if (Components.size() > 0)
    Arch = parseArch(Components[0]);
  VendorType Vendor = UnknownVendor;
  if (Components.size() > 1)
    Vendor = parseVendor(Components[1]);
  OSType OS = UnknownOS;
  if (Components.size() > 2) {
    OS = parseOS(Components[2]);
    IsCygwin = Components[2].startswith("cygwin");
    IsMinGW32 = Components[2].startswith("mingw");
  }
  EnvironmentType Environment = UnknownEnvironment;
  if (Components.size() > 3)
    Environment = parseEnvironment(Components[3]);
  ObjectFormatType ObjectFormat = UnknownObjectFormat;
  if (Components.size() > 4)
    ObjectFormat = parseFormat(Components[4]);

  // Note which components are already in their final position.  These will not
  // be moved.
  bool Found[4];
  Found[0] = Arch != UnknownArch;
  Found[1] = Vendor != UnknownVendor;
  Found[2] = OS != UnknownOS;
  Found[3] = Environment != UnknownEnvironment;

  // If they are not there already, permute the components into their canonical
  // positions by seeing if they parse as a valid architecture, and if so moving
  // the component to the architecture position etc.
  for (unsigned Pos = 0; Pos != array_lengthof(Found); ++Pos) {
    if (Found[Pos])
      continue; // Already in the canonical position.

    for (unsigned Idx = 0; Idx != Components.size(); ++Idx) {
      // Do not reparse any components that already matched.
      if (Idx < array_lengthof(Found) && Found[Idx])
        continue;

      // Does this component parse as valid for the target position?
      bool Valid = false;
      StringRef Comp = Components[Idx];
      switch (Pos) {
      default: llvm_unreachable("unexpected component type!");
      case 0:
        Arch = parseArch(Comp);
        Valid = Arch != UnknownArch;
        break;
      case 1:
        Vendor = parseVendor(Comp);
        Valid = Vendor != UnknownVendor;
        break;
      case 2:
        OS = parseOS(Comp);
        IsCygwin = Comp.startswith("cygwin");
        IsMinGW32 = Comp.startswith("mingw");
        Valid = OS != UnknownOS || IsCygwin || IsMinGW32;
        break;
      case 3:
        Environment = parseEnvironment(Comp);
        Valid = Environment != UnknownEnvironment;
        if (!Valid) {
          ObjectFormat = parseFormat(Comp);
          Valid = ObjectFormat != UnknownObjectFormat;
        }
        break;
      }
      if (!Valid)
        continue; // Nope, try the next component.

      // Move the component to the target position, pushing any non-fixed
      // components that are in the way to the right.  This tends to give
      // good results in the common cases of a forgotten vendor component
      // or a wrongly positioned environment.
      if (Pos < Idx) {
        // Insert left, pushing the existing components to the right.  For
        // example, a-b-i386 -> i386-a-b when moving i386 to the front.
        StringRef CurrentComponent(""); // The empty component.
        // Replace the component we are moving with an empty component.
        std::swap(CurrentComponent, Components[Idx]);
        // Insert the component being moved at Pos, displacing any existing
        // components to the right.
        for (unsigned i = Pos; !CurrentComponent.empty(); ++i) {
          // Skip over any fixed components.
          while (i < array_lengthof(Found) && Found[i])
            ++i;
          // Place the component at the new position, getting the component
          // that was at this position - it will be moved right.
          std::swap(CurrentComponent, Components[i]);
        }
      } else if (Pos > Idx) {
        // Push right by inserting empty components until the component at Idx
        // reaches the target position Pos.  For example, pc-a -> -pc-a when
        // moving pc to the second position.
        do {
          // Insert one empty component at Idx.
          StringRef CurrentComponent(""); // The empty component.
          for (unsigned i = Idx; i < Components.size();) {
            // Place the component at the new position, getting the component
            // that was at this position - it will be moved right.
            std::swap(CurrentComponent, Components[i]);
            // If it was placed on top of an empty component then we are done.
            if (CurrentComponent.empty())
              break;
            // Advance to the next component, skipping any fixed components.
            while (++i < array_lengthof(Found) && Found[i])
              ;
          }
          // The last component was pushed off the end - append it.
          if (!CurrentComponent.empty())
            Components.push_back(CurrentComponent);

          // Advance Idx to the component's new position.
          while (++Idx < array_lengthof(Found) && Found[Idx])
            ;
        } while (Idx < Pos); // Add more until the final position is reached.
      }
      assert(Pos < Components.size() && Components[Pos] == Comp &&
             "Component moved wrong!");
      Found[Pos] = true;
      break;
    }
  }

  // Replace empty components with "unknown" value.
  for (StringRef &C : Components)
    if (C.empty())
      C = "unknown";

  // Special case logic goes here.  At this point Arch, Vendor and OS have the
  // correct values for the computed components.
  std::string NormalizedEnvironment;
  if (Environment == Triple::Android && Components[3].startswith("androideabi")) {
    StringRef AndroidVersion = Components[3].drop_front(strlen("androideabi"));
    if (AndroidVersion.empty()) {
      Components[3] = "android";
    } else {
      NormalizedEnvironment = Twine("android", AndroidVersion).str();
      Components[3] = NormalizedEnvironment;
    }
  }

  // SUSE uses "gnueabi" to mean "gnueabihf"
  if (Vendor == Triple::SUSE && Environment == llvm::Triple::GNUEABI)
    Components[3] = "gnueabihf";

  if (OS == Triple::Win32) {
    Components.resize(4);
    Components[2] = "windows";
    if (Environment == UnknownEnvironment) {
      if (ObjectFormat == UnknownObjectFormat || ObjectFormat == Triple::COFF)
        Components[3] = "msvc";
      else
        Components[3] = getObjectFormatTypeName(ObjectFormat);
    }
  } else if (IsMinGW32) {
    Components.resize(4);
    Components[2] = "windows";
    Components[3] = "gnu";
  } else if (IsCygwin) {
    Components.resize(4);
    Components[2] = "windows";
    Components[3] = "cygnus";
  }
  if (IsMinGW32 || IsCygwin ||
      (OS == Triple::Win32 && Environment != UnknownEnvironment)) {
    if (ObjectFormat != UnknownObjectFormat && ObjectFormat != Triple::COFF) {
      Components.resize(5);
      Components[4] = getObjectFormatTypeName(ObjectFormat);
    }
  }

  // Stick the corrected components back together to form the normalized string.
  return join(Components, "-");
}

StringRef Triple::getArchName() const {
  return StringRef(Data).split('-').first;           // Isolate first component
}

StringRef Triple::getArchName(ArchType Kind, SubArchType SubArch) const {
  switch (Kind) {
  case Triple::mips:
    if (SubArch == MipsSubArch_r6)
      return "mipsisa32r6";
    break;
  case Triple::mipsel:
    if (SubArch == MipsSubArch_r6)
      return "mipsisa32r6el";
    break;
  case Triple::mips64:
    if (SubArch == MipsSubArch_r6)
      return "mipsisa64r6";
    break;
  case Triple::mips64el:
    if (SubArch == MipsSubArch_r6)
      return "mipsisa64r6el";
    break;
  default:
    break;
  }
  return getArchTypeName(Kind);
}

StringRef Triple::getVendorName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  return Tmp.split('-').first;                       // Isolate second component
}

StringRef Triple::getOSName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  Tmp = Tmp.split('-').second;                       // Strip second component
  return Tmp.split('-').first;                       // Isolate third component
}

StringRef Triple::getEnvironmentName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  Tmp = Tmp.split('-').second;                       // Strip second component
  return Tmp.split('-').second;                      // Strip third component
}

StringRef Triple::getOSAndEnvironmentName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  return Tmp.split('-').second;                      // Strip second component
}

static VersionTuple parseVersionFromName(StringRef Name) {
  VersionTuple Version;
  Version.tryParse(Name);
  return Version.withoutBuild();
}

VersionTuple Triple::getEnvironmentVersion() const {
  StringRef EnvironmentName = getEnvironmentName();
  StringRef EnvironmentTypeName = getEnvironmentTypeName(getEnvironment());
  if (EnvironmentName.startswith(EnvironmentTypeName))
    EnvironmentName = EnvironmentName.substr(EnvironmentTypeName.size());

  return parseVersionFromName(EnvironmentName);
}

VersionTuple Triple::getOSVersion() const {
  StringRef OSName = getOSName();
  // Assume that the OS portion of the triple starts with the canonical name.
  StringRef OSTypeName = getOSTypeName(getOS());
  if (OSName.startswith(OSTypeName))
    OSName = OSName.substr(OSTypeName.size());
  else if (getOS() == MacOSX)
    OSName.consume_front("macos");

  return parseVersionFromName(OSName);
}

bool Triple::getMacOSXVersion(VersionTuple &Version) const {
  Version = getOSVersion();

  switch (getOS()) {
  default: llvm_unreachable("unexpected OS for Darwin triple");
  case Darwin:
    // Default to darwin8, i.e., MacOSX 10.4.
    if (Version.getMajor() == 0)
      Version = VersionTuple(8);
    // Darwin version numbers are skewed from OS X versions.
    if (Version.getMajor() < 4) {
      return false;
    }
    if (Version.getMajor() <= 19) {
      Version = VersionTuple(10, Version.getMajor() - 4);
    } else {
      // darwin20+ corresponds to macOS 11+.
      Version = VersionTuple(11 + Version.getMajor() - 20);
    }
    break;
  case MacOSX:
    // Default to 10.4.
    if (Version.getMajor() == 0) {
      Version = VersionTuple(10, 4);
    } else if (Version.getMajor() < 10) {
      return false;
    }
    break;
  case IOS:
  case TvOS:
  case WatchOS:
    // Ignore the version from the triple.  This is only handled because the
    // the clang driver combines OS X and IOS support into a common Darwin
    // toolchain that wants to know the OS X version number even when targeting
    // IOS.
    Version = VersionTuple(10, 4);
    break;
  case DriverKit:
    llvm_unreachable("OSX version isn't relevant for DriverKit");
  }
  return true;
}

VersionTuple Triple::getiOSVersion() const {
  switch (getOS()) {
  default: llvm_unreachable("unexpected OS for Darwin triple");
  case Darwin:
  case MacOSX:
    // Ignore the version from the triple.  This is only handled because the
    // the clang driver combines OS X and IOS support into a common Darwin
    // toolchain that wants to know the iOS version number even when targeting
    // OS X.
    return VersionTuple(5);
  case IOS:
  case TvOS: {
    VersionTuple Version = getOSVersion();
    // Default to 5.0 (or 7.0 for arm64).
    if (Version.getMajor() == 0)
      return (getArch() == aarch64) ? VersionTuple(7) : VersionTuple(5);
    return Version;
  }
  case WatchOS:
    llvm_unreachable("conflicting triple info");
  case DriverKit:
    llvm_unreachable("DriverKit doesn't have an iOS version");
  }
}

VersionTuple Triple::getWatchOSVersion() const {
  switch (getOS()) {
  default: llvm_unreachable("unexpected OS for Darwin triple");
  case Darwin:
  case MacOSX:
    // Ignore the version from the triple.  This is only handled because the
    // the clang driver combines OS X and IOS support into a common Darwin
    // toolchain that wants to know the iOS version number even when targeting
    // OS X.
    return VersionTuple(2);
  case WatchOS: {
    VersionTuple Version = getOSVersion();
    if (Version.getMajor() == 0)
      return VersionTuple(2);
    return Version;
  }
  case IOS:
    llvm_unreachable("conflicting triple info");
  case DriverKit:
    llvm_unreachable("DriverKit doesn't have a WatchOS version");
  }
}

VersionTuple Triple::getDriverKitVersion() const {
  switch (getOS()) {
  default:
    llvm_unreachable("unexpected OS for Darwin triple");
  case DriverKit:
    VersionTuple Version = getOSVersion();
    if (Version.getMajor() == 0)
      return Version.withMajorReplaced(19);
    return Version;
  }
}

void Triple::setTriple(const Twine &Str) {
  *this = Triple(Str);
}

void Triple::setArch(ArchType Kind, SubArchType SubArch) {
  setArchName(getArchName(Kind, SubArch));
}

void Triple::setVendor(VendorType Kind) {
  setVendorName(getVendorTypeName(Kind));
}

void Triple::setOS(OSType Kind) {
  setOSName(getOSTypeName(Kind));
}

void Triple::setEnvironment(EnvironmentType Kind) {
  if (ObjectFormat == getDefaultFormat(*this))
    return setEnvironmentName(getEnvironmentTypeName(Kind));

  setEnvironmentName((getEnvironmentTypeName(Kind) + Twine("-") +
                      getObjectFormatTypeName(ObjectFormat)).str());
}

void Triple::setObjectFormat(ObjectFormatType Kind) {
  if (Environment == UnknownEnvironment)
    return setEnvironmentName(getObjectFormatTypeName(Kind));

  setEnvironmentName((getEnvironmentTypeName(Environment) + Twine("-") +
                      getObjectFormatTypeName(Kind)).str());
}

void Triple::setArchName(StringRef Str) {
  // Work around a miscompilation bug for Twines in gcc 4.0.3.
  SmallString<64> Triple;
  Triple += Str;
  Triple += "-";
  Triple += getVendorName();
  Triple += "-";
  Triple += getOSAndEnvironmentName();
  setTriple(Triple);
}

void Triple::setVendorName(StringRef Str) {
  setTriple(getArchName() + "-" + Str + "-" + getOSAndEnvironmentName());
}

void Triple::setOSName(StringRef Str) {
  if (hasEnvironment())
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str +
              "-" + getEnvironmentName());
  else
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

void Triple::setEnvironmentName(StringRef Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + getOSName() +
            "-" + Str);
}

void Triple::setOSAndEnvironmentName(StringRef Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

static unsigned getArchPointerBitWidth(llvm::Triple::ArchType Arch) {
  switch (Arch) {
  case llvm::Triple::UnknownArch:
    return 0;

  case llvm::Triple::avr:
  case llvm::Triple::msp430:
    return 16;

  case llvm::Triple::aarch64_32:
  case llvm::Triple::amdil:
  case llvm::Triple::arc:
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::csky:
  case llvm::Triple::hexagon:
  case llvm::Triple::hsail:
  case llvm::Triple::kalimba:
  case llvm::Triple::lanai:
  case llvm::Triple::le32:
  case llvm::Triple::loongarch32:
  case llvm::Triple::m68k:
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::nvptx:
  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::r600:
  case llvm::Triple::renderscript32:
  case llvm::Triple::riscv32:
  case llvm::Triple::shave:
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::spir:
  case llvm::Triple::spirv32:
  case llvm::Triple::tce:
  case llvm::Triple::tcele:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
  case llvm::Triple::wasm32:
  case llvm::Triple::x86:
  case llvm::Triple::xcore:
    return 32;

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::amdgcn:
  case llvm::Triple::amdil64:
  case llvm::Triple::bpfeb:
  case llvm::Triple::bpfel:
  case llvm::Triple::hsail64:
  case llvm::Triple::le64:
  case llvm::Triple::loongarch64:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
  case llvm::Triple::nvptx64:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
  case llvm::Triple::renderscript64:
  case llvm::Triple::riscv64:
  case llvm::Triple::sparcv9:
  case llvm::Triple::spir64:
  case llvm::Triple::spirv64:
  case llvm::Triple::systemz:
  case llvm::Triple::ve:
  case llvm::Triple::wasm64:
  case llvm::Triple::x86_64:
    return 64;
  }
  llvm_unreachable("Invalid architecture value");
}

bool Triple::isArch64Bit() const {
  return getArchPointerBitWidth(getArch()) == 64;
}

bool Triple::isArch32Bit() const {
  return getArchPointerBitWidth(getArch()) == 32;
}

bool Triple::isArch16Bit() const {
  return getArchPointerBitWidth(getArch()) == 16;
}

Triple Triple::get32BitArchVariant() const {
  Triple T(*this);
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::amdgcn:
  case Triple::avr:
  case Triple::bpfeb:
  case Triple::bpfel:
  case Triple::msp430:
  case Triple::systemz:
  case Triple::ve:
    T.setArch(UnknownArch);
    break;

  case Triple::aarch64_32:
  case Triple::amdil:
  case Triple::arc:
  case Triple::arm:
  case Triple::armeb:
  case Triple::csky:
  case Triple::hexagon:
  case Triple::hsail:
  case Triple::kalimba:
  case Triple::lanai:
  case Triple::le32:
  case Triple::loongarch32:
  case Triple::m68k:
  case Triple::mips:
  case Triple::mipsel:
  case Triple::nvptx:
  case Triple::ppc:
  case Triple::ppcle:
  case Triple::r600:
  case Triple::renderscript32:
  case Triple::riscv32:
  case Triple::shave:
  case Triple::sparc:
  case Triple::sparcel:
  case Triple::spir:
  case Triple::spirv32:
  case Triple::tce:
  case Triple::tcele:
  case Triple::thumb:
  case Triple::thumbeb:
  case Triple::wasm32:
  case Triple::x86:
  case Triple::xcore:
    // Already 32-bit.
    break;

  case Triple::aarch64:        T.setArch(Triple::arm);     break;
  case Triple::aarch64_be:     T.setArch(Triple::armeb);   break;
  case Triple::amdil64:        T.setArch(Triple::amdil);   break;
  case Triple::hsail64:        T.setArch(Triple::hsail);   break;
  case Triple::le64:           T.setArch(Triple::le32);    break;
  case Triple::loongarch64:    T.setArch(Triple::loongarch32); break;
  case Triple::mips64:
    T.setArch(Triple::mips, getSubArch());
    break;
  case Triple::mips64el:
    T.setArch(Triple::mipsel, getSubArch());
    break;
  case Triple::nvptx64:        T.setArch(Triple::nvptx);   break;
  case Triple::ppc64:          T.setArch(Triple::ppc);     break;
  case Triple::ppc64le:        T.setArch(Triple::ppcle);   break;
  case Triple::renderscript64: T.setArch(Triple::renderscript32); break;
  case Triple::riscv64:        T.setArch(Triple::riscv32); break;
  case Triple::sparcv9:        T.setArch(Triple::sparc);   break;
  case Triple::spir64:         T.setArch(Triple::spir);    break;
  case Triple::spirv64:        T.setArch(Triple::spirv32); break;
  case Triple::wasm64:         T.setArch(Triple::wasm32);  break;
  case Triple::x86_64:         T.setArch(Triple::x86);     break;
  }
  return T;
}

Triple Triple::get64BitArchVariant() const {
  Triple T(*this);
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::arc:
  case Triple::avr:
  case Triple::csky:
  case Triple::hexagon:
  case Triple::kalimba:
  case Triple::lanai:
  case Triple::m68k:
  case Triple::msp430:
  case Triple::r600:
  case Triple::shave:
  case Triple::sparcel:
  case Triple::tce:
  case Triple::tcele:
  case Triple::xcore:
    T.setArch(UnknownArch);
    break;

  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::amdgcn:
  case Triple::amdil64:
  case Triple::bpfeb:
  case Triple::bpfel:
  case Triple::hsail64:
  case Triple::le64:
  case Triple::loongarch64:
  case Triple::mips64:
  case Triple::mips64el:
  case Triple::nvptx64:
  case Triple::ppc64:
  case Triple::ppc64le:
  case Triple::renderscript64:
  case Triple::riscv64:
  case Triple::sparcv9:
  case Triple::spir64:
  case Triple::spirv64:
  case Triple::systemz:
  case Triple::ve:
  case Triple::wasm64:
  case Triple::x86_64:
    // Already 64-bit.
    break;

  case Triple::aarch64_32:      T.setArch(Triple::aarch64);    break;
  case Triple::amdil:           T.setArch(Triple::amdil64);    break;
  case Triple::arm:             T.setArch(Triple::aarch64);    break;
  case Triple::armeb:           T.setArch(Triple::aarch64_be); break;
  case Triple::hsail:           T.setArch(Triple::hsail64);    break;
  case Triple::le32:            T.setArch(Triple::le64);       break;
  case Triple::loongarch32:     T.setArch(Triple::loongarch64);    break;
  case Triple::mips:
    T.setArch(Triple::mips64, getSubArch());
    break;
  case Triple::mipsel:
    T.setArch(Triple::mips64el, getSubArch());
    break;
  case Triple::nvptx:           T.setArch(Triple::nvptx64);    break;
  case Triple::ppc:             T.setArch(Triple::ppc64);      break;
  case Triple::ppcle:           T.setArch(Triple::ppc64le);    break;
  case Triple::renderscript32:  T.setArch(Triple::renderscript64);     break;
  case Triple::riscv32:         T.setArch(Triple::riscv64);    break;
  case Triple::sparc:           T.setArch(Triple::sparcv9);    break;
  case Triple::spir:            T.setArch(Triple::spir64);     break;
  case Triple::spirv32:         T.setArch(Triple::spirv64);    break;
  case Triple::thumb:           T.setArch(Triple::aarch64);    break;
  case Triple::thumbeb:         T.setArch(Triple::aarch64_be); break;
  case Triple::wasm32:          T.setArch(Triple::wasm64);     break;
  case Triple::x86:             T.setArch(Triple::x86_64);     break;
  }
  return T;
}

Triple Triple::getBigEndianArchVariant() const {
  Triple T(*this);
  // Already big endian.
  if (!isLittleEndian())
    return T;
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::amdgcn:
  case Triple::amdil64:
  case Triple::amdil:
  case Triple::avr:
  case Triple::hexagon:
  case Triple::hsail64:
  case Triple::hsail:
  case Triple::kalimba:
  case Triple::le32:
  case Triple::le64:
  case Triple::loongarch32:
  case Triple::loongarch64:
  case Triple::msp430:
  case Triple::nvptx64:
  case Triple::nvptx:
  case Triple::r600:
  case Triple::renderscript32:
  case Triple::renderscript64:
  case Triple::riscv32:
  case Triple::riscv64:
  case Triple::shave:
  case Triple::spir64:
  case Triple::spir:
  case Triple::spirv32:
  case Triple::spirv64:
  case Triple::wasm32:
  case Triple::wasm64:
  case Triple::x86:
  case Triple::x86_64:
  case Triple::xcore:
  case Triple::ve:
  case Triple::csky:

  // ARM is intentionally unsupported here, changing the architecture would
  // drop any arch suffixes.
  case Triple::arm:
  case Triple::thumb:
    T.setArch(UnknownArch);
    break;

  case Triple::aarch64: T.setArch(Triple::aarch64_be); break;
  case Triple::bpfel:   T.setArch(Triple::bpfeb);      break;
  case Triple::mips64el:
    T.setArch(Triple::mips64, getSubArch());
    break;
  case Triple::mipsel:
    T.setArch(Triple::mips, getSubArch());
    break;
  case Triple::ppcle:   T.setArch(Triple::ppc);        break;
  case Triple::ppc64le: T.setArch(Triple::ppc64);      break;
  case Triple::sparcel: T.setArch(Triple::sparc);      break;
  case Triple::tcele:   T.setArch(Triple::tce);        break;
  default:
    llvm_unreachable("getBigEndianArchVariant: unknown triple.");
  }
  return T;
}

Triple Triple::getLittleEndianArchVariant() const {
  Triple T(*this);
  if (isLittleEndian())
    return T;

  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::lanai:
  case Triple::sparcv9:
  case Triple::systemz:
  case Triple::m68k:

  // ARM is intentionally unsupported here, changing the architecture would
  // drop any arch suffixes.
  case Triple::armeb:
  case Triple::thumbeb:
    T.setArch(UnknownArch);
    break;

  case Triple::aarch64_be: T.setArch(Triple::aarch64);  break;
  case Triple::bpfeb:      T.setArch(Triple::bpfel);    break;
  case Triple::mips64:
    T.setArch(Triple::mips64el, getSubArch());
    break;
  case Triple::mips:
    T.setArch(Triple::mipsel, getSubArch());
    break;
  case Triple::ppc:        T.setArch(Triple::ppcle);    break;
  case Triple::ppc64:      T.setArch(Triple::ppc64le);  break;
  case Triple::sparc:      T.setArch(Triple::sparcel);  break;
  case Triple::tce:        T.setArch(Triple::tcele);    break;
  default:
    llvm_unreachable("getLittleEndianArchVariant: unknown triple.");
  }
  return T;
}

bool Triple::isLittleEndian() const {
  switch (getArch()) {
  case Triple::aarch64:
  case Triple::aarch64_32:
  case Triple::amdgcn:
  case Triple::amdil64:
  case Triple::amdil:
  case Triple::arm:
  case Triple::avr:
  case Triple::bpfel:
  case Triple::csky:
  case Triple::hexagon:
  case Triple::hsail64:
  case Triple::hsail:
  case Triple::kalimba:
  case Triple::le32:
  case Triple::le64:
  case Triple::loongarch32:
  case Triple::loongarch64:
  case Triple::mips64el:
  case Triple::mipsel:
  case Triple::msp430:
  case Triple::nvptx64:
  case Triple::nvptx:
  case Triple::ppcle:
  case Triple::ppc64le:
  case Triple::r600:
  case Triple::renderscript32:
  case Triple::renderscript64:
  case Triple::riscv32:
  case Triple::riscv64:
  case Triple::shave:
  case Triple::sparcel:
  case Triple::spir64:
  case Triple::spir:
  case Triple::spirv32:
  case Triple::spirv64:
  case Triple::tcele:
  case Triple::thumb:
  case Triple::ve:
  case Triple::wasm32:
  case Triple::wasm64:
  case Triple::x86:
  case Triple::x86_64:
  case Triple::xcore:
    return true;
  default:
    return false;
  }
}

bool Triple::isCompatibleWith(const Triple &Other) const {
  // ARM and Thumb triples are compatible, if subarch, vendor and OS match.
  if ((getArch() == Triple::thumb && Other.getArch() == Triple::arm) ||
      (getArch() == Triple::arm && Other.getArch() == Triple::thumb) ||
      (getArch() == Triple::thumbeb && Other.getArch() == Triple::armeb) ||
      (getArch() == Triple::armeb && Other.getArch() == Triple::thumbeb)) {
    if (getVendor() == Triple::Apple)
      return getSubArch() == Other.getSubArch() &&
             getVendor() == Other.getVendor() && getOS() == Other.getOS();
    else
      return getSubArch() == Other.getSubArch() &&
             getVendor() == Other.getVendor() && getOS() == Other.getOS() &&
             getEnvironment() == Other.getEnvironment() &&
             getObjectFormat() == Other.getObjectFormat();
  }

  // If vendor is apple, ignore the version number.
  if (getVendor() == Triple::Apple)
    return getArch() == Other.getArch() && getSubArch() == Other.getSubArch() &&
           getVendor() == Other.getVendor() && getOS() == Other.getOS();

  return *this == Other;
}

std::string Triple::merge(const Triple &Other) const {
  // If vendor is apple, pick the triple with the larger version number.
  if (getVendor() == Triple::Apple)
    if (Other.isOSVersionLT(*this))
      return str();

  return Other.str();
}

bool Triple::isMacOSXVersionLT(unsigned Major, unsigned Minor,
                               unsigned Micro) const {
  assert(isMacOSX() && "Not an OS X triple!");

  // If this is OS X, expect a sane version number.
  if (getOS() == Triple::MacOSX)
    return isOSVersionLT(Major, Minor, Micro);

  // Otherwise, compare to the "Darwin" number.
  if (Major == 10) {
    return isOSVersionLT(Minor + 4, Micro, 0);
  } else {
    assert(Major >= 11 && "Unexpected major version");
    return isOSVersionLT(Major - 11 + 20, Minor, Micro);
  }
}

VersionTuple Triple::getMinimumSupportedOSVersion() const {
  if (getVendor() != Triple::Apple || getArch() != Triple::aarch64)
    return VersionTuple();
  switch (getOS()) {
  case Triple::MacOSX:
    // ARM64 slice is supported starting from macOS 11.0+.
    return VersionTuple(11, 0, 0);
  case Triple::IOS:
    // ARM64 slice is supported starting from Mac Catalyst 14 (macOS 11).
    // ARM64 simulators are supported for iOS 14+.
    if (isMacCatalystEnvironment() || isSimulatorEnvironment())
      return VersionTuple(14, 0, 0);
    // ARM64e slice is supported starting from iOS 14.
    if (isArm64e())
      return VersionTuple(14, 0, 0);
    break;
  case Triple::TvOS:
    // ARM64 simulators are supported for tvOS 14+.
    if (isSimulatorEnvironment())
      return VersionTuple(14, 0, 0);
    break;
  case Triple::WatchOS:
    // ARM64 simulators are supported for watchOS 7+.
    if (isSimulatorEnvironment())
      return VersionTuple(7, 0, 0);
    break;
  case Triple::DriverKit:
    return VersionTuple(20, 0, 0);
  default:
    break;
  }
  return VersionTuple();
}

StringRef Triple::getARMCPUForArch(StringRef MArch) const {
  if (MArch.empty())
    MArch = getArchName();
  MArch = ARM::getCanonicalArchName(MArch);

  // Some defaults are forced.
  switch (getOS()) {
  case llvm::Triple::FreeBSD:
  case llvm::Triple::NetBSD:
  case llvm::Triple::OpenBSD:
    if (!MArch.empty() && MArch == "v6")
      return "arm1176jzf-s";
    if (!MArch.empty() && MArch == "v7")
      return "cortex-a8";
    break;
  case llvm::Triple::Win32:
    // FIXME: this is invalid for WindowsCE
    if (ARM::parseArchVersion(MArch) <= 7)
      return "cortex-a9";
    break;
  case llvm::Triple::IOS:
  case llvm::Triple::MacOSX:
  case llvm::Triple::TvOS:
  case llvm::Triple::WatchOS:
  case llvm::Triple::DriverKit:
    if (MArch == "v7k")
      return "cortex-a7";
    break;
  default:
    break;
  }

  if (MArch.empty())
    return StringRef();

  StringRef CPU = ARM::getDefaultCPU(MArch);
  if (!CPU.empty() && !CPU.equals("invalid"))
    return CPU;

  // If no specific architecture version is requested, return the minimum CPU
  // required by the OS and environment.
  switch (getOS()) {
  case llvm::Triple::NetBSD:
    switch (getEnvironment()) {
    case llvm::Triple::EABI:
    case llvm::Triple::EABIHF:
    case llvm::Triple::GNUEABI:
    case llvm::Triple::GNUEABIHF:
      return "arm926ej-s";
    default:
      return "strongarm";
    }
  case llvm::Triple::NaCl:
  case llvm::Triple::OpenBSD:
    return "cortex-a8";
  default:
    switch (getEnvironment()) {
    case llvm::Triple::EABIHF:
    case llvm::Triple::GNUEABIHF:
    case llvm::Triple::MuslEABIHF:
      return "arm1176jzf-s";
    default:
      return "arm7tdmi";
    }
  }

  llvm_unreachable("invalid arch name");
}

VersionTuple Triple::getCanonicalVersionForOS(OSType OSKind,
                                              const VersionTuple &Version) {
  switch (OSKind) {
  case MacOSX:
    // macOS 10.16 is canonicalized to macOS 11.
    if (Version == VersionTuple(10, 16))
      return VersionTuple(11, 0);
    LLVM_FALLTHROUGH;
  default:
    return Version;
  }
}
