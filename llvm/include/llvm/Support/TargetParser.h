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

#ifndef LLVM_SUPPORT_TARGETPARSER_H
#define LLVM_SUPPORT_TARGETPARSER_H

namespace llvm {
  class StringRef;

// Target specific information into their own namespaces. These should be
// generated from TableGen because the information is already there, and there
// is where new information about targets will be added.
// FIXME: To TableGen this we need to make some table generated files available
// even if the back-end is not compiled with LLVM, plus we need to create a new
// back-end to TableGen to create these clean tables.
namespace ARM {
  // FPU names.
  enum FPUKind {
    INVALID_FPU = 0,
    VFP,
    VFPV2,
    VFPV3,
    VFPV3_D16,
    VFPV4,
    VFPV4_D16,
    FPV5_D16,
    FP_ARMV8,
    NEON,
    NEON_VFPV4,
    NEON_FP_ARMV8,
    CRYPTO_NEON_FP_ARMV8,
    SOFTVFP,
    LAST_FPU
  };

  // Arch names.
  enum ArchKind {
    INVALID_ARCH = 0,
    ARMV2,
    ARMV2A,
    ARMV3,
    ARMV3M,
    ARMV4,
    ARMV4T,
    ARMV5,
    ARMV5T,
    ARMV5TE,
    ARMV6,
    ARMV6J,
    ARMV6K,
    ARMV6T2,
    ARMV6Z,
    ARMV6ZK,
    ARMV6M,
    ARMV7,
    ARMV7A,
    ARMV7R,
    ARMV7M,
    ARMV8A,
    ARMV8_1A,
    IWMMXT,
    IWMMXT2,
    LAST_ARCH
  };

  // Arch extension modifiers for CPUs.
  enum ArchExtKind {
    INVALID_ARCHEXT = 0,
    CRC,
    CRYPTO,
    FP,
    HWDIV,
    MP,
    SEC,
    VIRT,
    LAST_ARCHEXT
  };

} // namespace ARM

// Target Parsers, one per architecture.
class ARMTargetParser {
  static StringRef getFPUSynonym(StringRef FPU);
  static StringRef getArchSynonym(StringRef Arch);

public:
  // Information by ID
  static const char * getFPUName(unsigned ID);
  static const char * getArchName(unsigned ID);
  static unsigned getArchDefaultCPUArch(unsigned ID);
  static const char * getArchDefaultCPUName(unsigned ID);
  static const char * getArchExtName(unsigned ID);

  // Parser
  static unsigned parseFPU(StringRef FPU);
  static unsigned parseArch(StringRef Arch);
  static unsigned parseArchExt(StringRef ArchExt);
};

} // namespace llvm

#endif
