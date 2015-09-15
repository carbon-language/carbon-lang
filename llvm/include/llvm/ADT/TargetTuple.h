//===-- llvm/ADT/TargetTuple.h - Target tuple class -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains the definitions for TargetTuples which describe the
/// target in a unique unambiguous way. This is in contrast to the GNU triples
/// handled by the Triple class which are more of a guideline than a
/// description and whose meaning can be overridden by vendors and distributors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_TARGETTUPLE_H
#define LLVM_ADT_TARGETTUPLE_H

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/Triple.h"

// Some system headers or GCC predefined macros conflict with identifiers in
// this file.  Undefine them here.
#undef NetBSD
#undef mips
#undef sparc

namespace llvm {

/// TargetTuple is currently a proxy for Triple but will become an unambiguous,
/// authoratitive, and mutable counterpart to the GNU triples handled by Triple.
class TargetTuple {
public:
  // FIXME: Don't duplicate Triple::ArchType. It's worth mentioning that these
  //        these values don't have to match Triple::ArchType. For example, it
  //        would be fairly sensible to have a single 'mips' architecture and
  //        distinguish endianness and ABI elsewhere.
  enum ArchType {
    UnknownArch,

    arm,        // ARM (little endian): arm, armv.*, xscale
    armeb,      // ARM (big endian): armeb
    aarch64,    // AArch64 (little endian): aarch64
    aarch64_be, // AArch64 (big endian): aarch64_be
    bpfel,      // eBPF or extended BPF or 64-bit BPF (little endian)
    bpfeb,      // eBPF or extended BPF or 64-bit BPF (big endian)
    hexagon,    // Hexagon: hexagon
    mips,       // MIPS: mips, mipsallegrex
    mipsel,     // MIPSEL: mipsel, mipsallegrexel
    mips64,     // MIPS64: mips64
    mips64el,   // MIPS64EL: mips64el
    msp430,     // MSP430: msp430
    ppc,        // PPC: powerpc
    ppc64,      // PPC64: powerpc64, ppu
    ppc64le,    // PPC64LE: powerpc64le
    r600,       // R600: AMD GPUs HD2XXX - HD6XXX
    amdgcn,     // AMDGCN: AMD GCN GPUs
    sparc,      // Sparc: sparc
    sparcv9,    // Sparcv9: Sparcv9
    sparcel,    // Sparc: (endianness = little). NB: 'Sparcle' is a CPU variant
    systemz,    // SystemZ: s390x
    tce,        // TCE (http://tce.cs.tut.fi/): tce
                // FIXME: thumb/thumbeb will be merged into arm/armeb soon.
    thumb,      // Thumb (little endian): thumb, thumbv.*
    thumbeb,    // Thumb (big endian): thumbeb
    x86,        // X86: i[3-9]86
    x86_64,     // X86-64: amd64, x86_64
    xcore,      // XCore: xcore
    nvptx,      // NVPTX: 32-bit
    nvptx64,    // NVPTX: 64-bit
    le32,       // le32: generic little-endian 32-bit CPU (PNaCl / Emscripten)
    le64,       // le64: generic little-endian 64-bit CPU (PNaCl / Emscripten)
    amdil,      // AMDIL
    amdil64,    // AMDIL with 64-bit pointers
    hsail,      // AMD HSAIL
    hsail64,    // AMD HSAIL with 64-bit pointers
    spir,       // SPIR: standard portable IR for OpenCL 32-bit version
    spir64,     // SPIR: standard portable IR for OpenCL 64-bit version
    kalimba,    // Kalimba: generic kalimba
    shave,      // SHAVE: Movidius vector VLIW processors
    wasm32,     // WebAssembly with 32-bit pointers
    wasm64,     // WebAssembly with 64-bit pointers
    LastArchType = wasm64
  };

  enum SubArchType {
    NoSubArch,

    ARMSubArch_v8_1a,
    ARMSubArch_v8,
    ARMSubArch_v7,
    ARMSubArch_v7em,
    ARMSubArch_v7m,
    ARMSubArch_v7s,
    ARMSubArch_v6,
    ARMSubArch_v6m,
    ARMSubArch_v6k,
    ARMSubArch_v6t2,
    ARMSubArch_v5,
    ARMSubArch_v5te,
    ARMSubArch_v4t,

    KalimbaSubArch_v3,
    KalimbaSubArch_v4,
    KalimbaSubArch_v5
  };

  enum VendorType {
    UnknownVendor,

    Apple,
    PC,
    SCEI,
    BGP,
    BGQ,
    Freescale,
    IBM,
    ImaginationTechnologies,
    MipsTechnologies,
    NVIDIA,
    CSR,
    Myriad,
    LastVendorType = Myriad
  };

  enum OSType {
    UnknownOS,

    CloudABI,
    Darwin,
    DragonFly,
    FreeBSD,
    IOS,
    KFreeBSD,
    Linux,
    Lv2, // PS3
    MacOSX,
    NetBSD,
    OpenBSD,
    Solaris,
    Win32,
    Haiku,
    Minix,
    RTEMS,
    NaCl, // Native Client
    CNK,  // BG/P Compute-Node Kernel
    Bitrig,
    AIX,
    CUDA,   // NVIDIA CUDA
    NVCL,   // NVIDIA OpenCL
    AMDHSA, // AMD HSA Runtime
    PS4,
    LastOSType = PS4
  };

  enum EnvironmentType {
    UnknownEnvironment,

    GNU,
    GNUEABI,
    GNUEABIHF,
    GNUX32,
    CODE16,
    EABI,
    EABIHF,
    Android,

    MSVC,
    Itanium,
    Cygnus,
    AMDOpenCL,
    CoreCLR,
    LastEnvironmentType = CoreCLR
  };

  enum ObjectFormatType {
    UnknownObjectFormat,

    COFF,
    ELF,
    MachO,
  };

public:
  /// @name Constructors
  /// @{

  /// Default constructor leaves all fields unknown.
  TargetTuple() : GnuTT() {}

  /// Convert a GNU Triple to a TargetTuple.
  ///
  /// This conversion assumes that GNU Triple's have a specific defined meaning
  /// which isn't strictly true. A single Triple can potentially have multiple
  /// contradictory meanings depending on compiler options and configure-time
  /// options. Despite this, Triple's do tend to have a 'usual' meaning, or
  /// rather a default behaviour and this function selects it.
  ///
  /// When tool options affect the desired TargetTuple, the tool should obtain
  /// the usual meaning of the GNU Triple using this constructor and then use
  /// the mutator methods to apply the tool options.
  explicit TargetTuple(const Triple &GnuTT) : GnuTT(GnuTT) {}

  /// @}
  /// @name Typed Component Access
  /// @{

  /// Get the parsed architecture type of this triple.
  ArchType getArch() const;

  /// get the parsed subarchitecture type for this triple.
  SubArchType getSubArch() const;

  /// Get the parsed vendor type of this triple.
  VendorType getVendor() const;

  /// Get the parsed operating system type of this triple.
  OSType getOS() const;

  /// Does this triple have the optional environment
  /// (fourth) component?
  bool hasEnvironment() const { return GnuTT.hasEnvironment(); }

  /// Get the parsed environment type of this triple.
  EnvironmentType getEnvironment() const;

  /// Parse the version number from the OS name component of the
  /// triple, if present.
  ///
  /// For example, "fooos1.2.3" would return (1, 2, 3).
  ///
  /// If an entry is not defined, it will be returned as 0.
  void getEnvironmentVersion(unsigned &Major, unsigned &Minor,
                             unsigned &Micro) const;

  /// Get the object format for this triple.
  ObjectFormatType getObjectFormat() const;

  /// Parse the version number from the OS name component of the
  /// triple, if present.
  ///
  /// For example, "fooos1.2.3" would return (1, 2, 3).
  ///
  /// If an entry is not defined, it will be returned as 0.
  void getOSVersion(unsigned &Major, unsigned &Minor, unsigned &Micro) const {
    return GnuTT.getOSVersion(Major, Minor, Micro);
  }

  /// Return just the major version number, this is
  /// specialized because it is a common query.
  unsigned getOSMajorVersion() const { return GnuTT.getOSMajorVersion(); }

  /// Parse the version number as with getOSVersion and then
  /// translate generic "darwin" versions to the corresponding OS X versions.
  /// This may also be called with IOS triples but the OS X version number is
  /// just set to a constant 10.4.0 in that case.  Returns true if successful.
  bool getMacOSXVersion(unsigned &Major, unsigned &Minor,
                        unsigned &Micro) const {
    return GnuTT.getMacOSXVersion(Major, Minor, Micro);
  }

  /// Parse the version number as with getOSVersion.  This
  /// should
  /// only be called with IOS triples.
  void getiOSVersion(unsigned &Major, unsigned &Minor, unsigned &Micro) const {
    return GnuTT.getiOSVersion(Major, Minor, Micro);
  }

  /// @}
  /// @name Direct Component Access
  /// @{

  const std::string &str() const { return GnuTT.str(); }

  const std::string &getTriple() const { return GnuTT.str(); }

  /// Get the architecture (first) component of the
  /// triple.
  StringRef getArchName() const { return GnuTT.getArchName(); }

  /// Get the vendor (second) component of the triple.
  StringRef getVendorName() const { return GnuTT.getVendorName(); }

  /// Get the operating system (third) component of the
  /// triple.
  StringRef getOSName() const { return GnuTT.getOSName(); }

  /// Get the optional environment (fourth)
  /// component of the triple, or "" if empty.
  StringRef getEnvironmentName() const { return GnuTT.getEnvironmentName(); }

  /// Get the operating system and optional
  /// environment components as a single string (separated by a '-'
  /// if the environment component is present).
  StringRef getOSAndEnvironmentName() const {
    return GnuTT.getOSAndEnvironmentName();
  }

  /// @}
  /// @name Convenience Predicates
  /// @{

  /// Test whether the architecture is 64-bit
  ///
  /// Note that this tests for 64-bit pointer width, and nothing else. Note
  /// that we intentionally expose only three predicates, 64-bit, 32-bit, and
  /// 16-bit. The inner details of pointer width for particular architectures
  /// is not summed up in the triple, and so only a coarse grained predicate
  /// system is provided.
  bool isArch64Bit() const { return GnuTT.isArch64Bit(); }

  /// Test whether the architecture is 32-bit
  ///
  /// Note that this tests for 32-bit pointer width, and nothing else.
  bool isArch32Bit() const { return GnuTT.isArch32Bit(); }

  /// Test whether the architecture is 16-bit
  ///
  /// Note that this tests for 16-bit pointer width, and nothing else.
  bool isArch16Bit() const { return GnuTT.isArch16Bit(); }

  /// Helper function for doing comparisons against version
  /// numbers included in the target triple.
  bool isOSVersionLT(unsigned Major, unsigned Minor = 0,
                     unsigned Micro = 0) const {
    return GnuTT.isOSVersionLT(Major, Minor, Micro);
  }

  bool isOSVersionLT(const Triple &Other) const {
    return GnuTT.isOSVersionLT(Other);
  }

  /// Comparison function for checking OS X version
  /// compatibility, which handles supporting skewed version numbering schemes
  /// used by the "darwin" triples.
  unsigned isMacOSXVersionLT(unsigned Major, unsigned Minor = 0,
                             unsigned Micro = 0) const {
    return GnuTT.isMacOSXVersionLT(Major, Minor, Micro);
  }

  /// Is this a Mac OS X triple. For legacy reasons, we support both
  /// "darwin" and "osx" as OS X triples.
  bool isMacOSX() const { return GnuTT.isMacOSX(); }

  /// Is this an iOS triple.
  bool isiOS() const { return GnuTT.isiOS(); }

  /// Is this a "Darwin" OS (OS X or iOS).
  bool isOSDarwin() const { return GnuTT.isOSDarwin(); }

  bool isOSNetBSD() const { return GnuTT.isOSNetBSD(); }

  bool isOSOpenBSD() const { return GnuTT.isOSOpenBSD(); }

  bool isOSFreeBSD() const { return GnuTT.isOSFreeBSD(); }

  bool isOSDragonFly() const { return GnuTT.isOSDragonFly(); }

  bool isOSSolaris() const { return GnuTT.isOSSolaris(); }

  bool isOSBitrig() const { return GnuTT.isOSBitrig(); }

  bool isWindowsMSVCEnvironment() const {
    return GnuTT.isWindowsMSVCEnvironment();
  }

  bool isKnownWindowsMSVCEnvironment() const {
    return GnuTT.isKnownWindowsMSVCEnvironment();
  }

  bool isWindowsCoreCLREnvironment() const {
    return GnuTT.isWindowsCoreCLREnvironment();
  }

  bool isWindowsItaniumEnvironment() const {
    return GnuTT.isWindowsItaniumEnvironment();
  }

  bool isWindowsCygwinEnvironment() const {
    return GnuTT.isWindowsCygwinEnvironment();
  }

  bool isWindowsGNUEnvironment() const {
    return GnuTT.isWindowsGNUEnvironment();
  }

  /// Tests for either Cygwin or MinGW OS
  bool isOSCygMing() const { return GnuTT.isOSCygMing(); }

  /// Is this a "Windows" OS targeting a "MSVCRT.dll" environment.
  bool isOSMSVCRT() const { return GnuTT.isOSMSVCRT(); }

  /// Tests whether the OS is Windows.
  bool isOSWindows() const { return GnuTT.isOSWindows(); }

  /// Tests whether the OS is NaCl (Native Client)
  bool isOSNaCl() const { return GnuTT.isOSNaCl(); }

  /// Tests whether the OS is Linux.
  bool isOSLinux() const { return GnuTT.isOSLinux(); }

  /// Tests whether the OS uses the ELF binary format.
  bool isOSBinFormatELF() const { return GnuTT.isOSBinFormatELF(); }

  /// Tests whether the OS uses the COFF binary format.
  bool isOSBinFormatCOFF() const { return GnuTT.isOSBinFormatCOFF(); }

  /// Tests whether the environment is MachO.
  bool isOSBinFormatMachO() const { return GnuTT.isOSBinFormatMachO(); }

  /// Tests whether the target is the PS4 CPU
  bool isPS4CPU() const { return GnuTT.isPS4CPU(); }

  /// Tests whether the target is the PS4 platform
  bool isPS4() const { return GnuTT.isPS4(); }

  /// @}

  // FIXME: Remove. This function exists to avoid having to migrate everything
  //        at once.
  const Triple &getTargetTriple() const { return GnuTT; }

private:
  Triple GnuTT;
};

} // End llvm namespace

#endif
