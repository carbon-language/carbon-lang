/*
 * kmp_platform.h -- header for determining operating system and architecture
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_PLATFORM_H
#define KMP_PLATFORM_H

/* ---------------------- Operating system recognition ------------------- */

#define KMP_OS_LINUX 0
#define KMP_OS_DRAGONFLY 0
#define KMP_OS_FREEBSD 0
#define KMP_OS_NETBSD 0
#define KMP_OS_OPENBSD 0
#define KMP_OS_DARWIN 0
#define KMP_OS_WINDOWS 0
#define KMP_OS_HURD 0
#define KMP_OS_UNIX 0 /* disjunction of KMP_OS_LINUX, KMP_OS_DARWIN etc. */

#ifdef _WIN32
#undef KMP_OS_WINDOWS
#define KMP_OS_WINDOWS 1
#endif

#if (defined __APPLE__ && defined __MACH__)
#undef KMP_OS_DARWIN
#define KMP_OS_DARWIN 1
#endif

// in some ppc64 linux installations, only the second condition is met
#if (defined __linux)
#undef KMP_OS_LINUX
#define KMP_OS_LINUX 1
#elif (defined __linux__)
#undef KMP_OS_LINUX
#define KMP_OS_LINUX 1
#else
#endif

#if (defined __DragonFly__)
#undef KMP_OS_DRAGONFLY
#define KMP_OS_DRAGONFLY 1
#endif

#if (defined __FreeBSD__)
#undef KMP_OS_FREEBSD
#define KMP_OS_FREEBSD 1
#endif

#if (defined __NetBSD__)
#undef KMP_OS_NETBSD
#define KMP_OS_NETBSD 1
#endif

#if (defined __OpenBSD__)
#undef KMP_OS_OPENBSD
#define KMP_OS_OPENBSD 1
#endif

#if (defined __GNU__)
#undef KMP_OS_HURD
#define KMP_OS_HURD 1
#endif

#if (1 != KMP_OS_LINUX + KMP_OS_DRAGONFLY + KMP_OS_FREEBSD + KMP_OS_NETBSD +   \
              KMP_OS_OPENBSD + KMP_OS_DARWIN + KMP_OS_WINDOWS + KMP_OS_HURD)
#error Unknown OS
#endif

#if KMP_OS_LINUX || KMP_OS_DRAGONFLY || KMP_OS_FREEBSD || KMP_OS_NETBSD ||     \
    KMP_OS_OPENBSD || KMP_OS_DARWIN || KMP_OS_HURD
#undef KMP_OS_UNIX
#define KMP_OS_UNIX 1
#endif

/* ---------------------- Architecture recognition ------------------- */

#define KMP_ARCH_X86 0
#define KMP_ARCH_X86_64 0
#define KMP_ARCH_AARCH64 0
#define KMP_ARCH_PPC64_ELFv1 0
#define KMP_ARCH_PPC64_ELFv2 0
#define KMP_ARCH_PPC64 (KMP_ARCH_PPC64_ELFv2 || KMP_ARCH_PPC64_ELFv1)
#define KMP_ARCH_MIPS 0
#define KMP_ARCH_MIPS64 0
#define KMP_ARCH_RISCV64 0

#if KMP_OS_WINDOWS
#if defined(_M_AMD64) || defined(__x86_64)
#undef KMP_ARCH_X86_64
#define KMP_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#undef KMP_ARCH_AARCH64
#define KMP_ARCH_AARCH64 1
#else
#undef KMP_ARCH_X86
#define KMP_ARCH_X86 1
#endif
#endif

#if KMP_OS_UNIX
#if defined __x86_64
#undef KMP_ARCH_X86_64
#define KMP_ARCH_X86_64 1
#elif defined __i386
#undef KMP_ARCH_X86
#define KMP_ARCH_X86 1
#elif defined __powerpc64__
#if defined(_CALL_ELF) && _CALL_ELF == 2
#undef KMP_ARCH_PPC64_ELFv2
#define KMP_ARCH_PPC64_ELFv2 1
#else
#undef KMP_ARCH_PPC64_ELFv1
#define KMP_ARCH_PPC64_ELFv1 1
#endif
#elif defined __aarch64__
#undef KMP_ARCH_AARCH64
#define KMP_ARCH_AARCH64 1
#elif defined __mips__
#if defined __mips64
#undef KMP_ARCH_MIPS64
#define KMP_ARCH_MIPS64 1
#else
#undef KMP_ARCH_MIPS
#define KMP_ARCH_MIPS 1
#endif
#elif defined __riscv && __riscv_xlen == 64
#undef KMP_ARCH_RISCV64
#define KMP_ARCH_RISCV64 1
#endif
#endif

#if defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7R__) ||                     \
    defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7VE__)
#define KMP_ARCH_ARMV7 1
#endif

#if defined(KMP_ARCH_ARMV7) || defined(__ARM_ARCH_6__) ||                      \
    defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) ||                    \
    defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6T2__) ||                   \
    defined(__ARM_ARCH_6ZK__)
#define KMP_ARCH_ARMV6 1
#endif

#if defined(KMP_ARCH_ARMV6) || defined(__ARM_ARCH_5T__) ||                     \
    defined(__ARM_ARCH_5E__) || defined(__ARM_ARCH_5TE__) ||                   \
    defined(__ARM_ARCH_5TEJ__)
#define KMP_ARCH_ARMV5 1
#endif

#if defined(KMP_ARCH_ARMV5) || defined(__ARM_ARCH_4__) ||                      \
    defined(__ARM_ARCH_4T__)
#define KMP_ARCH_ARMV4 1
#endif

#if defined(KMP_ARCH_ARMV4) || defined(__ARM_ARCH_3__) ||                      \
    defined(__ARM_ARCH_3M__)
#define KMP_ARCH_ARMV3 1
#endif

#if defined(KMP_ARCH_ARMV3) || defined(__ARM_ARCH_2__)
#define KMP_ARCH_ARMV2 1
#endif

#if defined(KMP_ARCH_ARMV2)
#define KMP_ARCH_ARM 1
#endif

#if defined(__MIC__) || defined(__MIC2__)
#define KMP_MIC 1
#if __MIC2__ || __KNC__
#define KMP_MIC1 0
#define KMP_MIC2 1
#else
#define KMP_MIC1 1
#define KMP_MIC2 0
#endif
#else
#define KMP_MIC 0
#define KMP_MIC1 0
#define KMP_MIC2 0
#endif

/* Specify 32 bit architectures here */
#define KMP_32_BIT_ARCH (KMP_ARCH_X86 || KMP_ARCH_ARM || KMP_ARCH_MIPS)

// Platforms which support Intel(R) Many Integrated Core Architecture
#define KMP_MIC_SUPPORTED                                                      \
  ((KMP_ARCH_X86 || KMP_ARCH_X86_64) && (KMP_OS_LINUX || KMP_OS_WINDOWS))

// TODO: Fixme - This is clever, but really fugly
#if (1 != KMP_ARCH_X86 + KMP_ARCH_X86_64 + KMP_ARCH_ARM + KMP_ARCH_PPC64 +     \
              KMP_ARCH_AARCH64 + KMP_ARCH_MIPS + KMP_ARCH_MIPS64 +             \
              KMP_ARCH_RISCV64)
#error Unknown or unsupported architecture
#endif

#endif // KMP_PLATFORM_H
