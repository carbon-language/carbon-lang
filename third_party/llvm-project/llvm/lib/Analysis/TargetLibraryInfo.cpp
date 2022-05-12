//===-- TargetLibraryInfo.cpp - Runtime library information ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetLibraryInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

static cl::opt<TargetLibraryInfoImpl::VectorLibrary> ClVectorLibrary(
    "vector-library", cl::Hidden, cl::desc("Vector functions library"),
    cl::init(TargetLibraryInfoImpl::NoLibrary),
    cl::values(clEnumValN(TargetLibraryInfoImpl::NoLibrary, "none",
                          "No vector functions library"),
               clEnumValN(TargetLibraryInfoImpl::Accelerate, "Accelerate",
                          "Accelerate framework"),
               clEnumValN(TargetLibraryInfoImpl::DarwinLibSystemM,
                          "Darwin_libsystem_m", "Darwin libsystem_m"),
               clEnumValN(TargetLibraryInfoImpl::LIBMVEC_X86, "LIBMVEC-X86",
                          "GLIBC Vector Math library"),
               clEnumValN(TargetLibraryInfoImpl::MASSV, "MASSV",
                          "IBM MASS vector library"),
               clEnumValN(TargetLibraryInfoImpl::SVML, "SVML",
                          "Intel SVML library")));

StringLiteral const TargetLibraryInfoImpl::StandardNames[LibFunc::NumLibFuncs] =
    {
#define TLI_DEFINE_STRING
#include "llvm/Analysis/TargetLibraryInfo.def"
};

static bool hasSinCosPiStret(const Triple &T) {
  // Only Darwin variants have _stret versions of combined trig functions.
  if (!T.isOSDarwin())
    return false;

  // The ABI is rather complicated on x86, so don't do anything special there.
  if (T.getArch() == Triple::x86)
    return false;

  if (T.isMacOSX() && T.isMacOSXVersionLT(10, 9))
    return false;

  if (T.isiOS() && T.isOSVersionLT(7, 0))
    return false;

  return true;
}

static bool hasBcmp(const Triple &TT) {
  // Posix removed support from bcmp() in 2001, but the glibc and several
  // implementations of the libc still have it.
  if (TT.isOSLinux())
    return TT.isGNUEnvironment() || TT.isMusl();
  // Both NetBSD and OpenBSD are planning to remove the function. Windows does
  // not have it.
  return TT.isOSFreeBSD() || TT.isOSSolaris();
}

static bool isCallingConvCCompatible(CallingConv::ID CC, StringRef TT,
                                     FunctionType *FuncTy) {
  switch (CC) {
  default:
    return false;
  case llvm::CallingConv::C:
    return true;
  case llvm::CallingConv::ARM_APCS:
  case llvm::CallingConv::ARM_AAPCS:
  case llvm::CallingConv::ARM_AAPCS_VFP: {

    // The iOS ABI diverges from the standard in some cases, so for now don't
    // try to simplify those calls.
    if (Triple(TT).isiOS())
      return false;

    if (!FuncTy->getReturnType()->isPointerTy() &&
        !FuncTy->getReturnType()->isIntegerTy() &&
        !FuncTy->getReturnType()->isVoidTy())
      return false;

    for (auto *Param : FuncTy->params()) {
      if (!Param->isPointerTy() && !Param->isIntegerTy())
        return false;
    }
    return true;
  }
  }
  return false;
}

bool TargetLibraryInfoImpl::isCallingConvCCompatible(CallBase *CI) {
  return ::isCallingConvCCompatible(CI->getCallingConv(),
                                    CI->getModule()->getTargetTriple(),
                                    CI->getFunctionType());
}

bool TargetLibraryInfoImpl::isCallingConvCCompatible(Function *F) {
  return ::isCallingConvCCompatible(F->getCallingConv(),
                                    F->getParent()->getTargetTriple(),
                                    F->getFunctionType());
}

/// Initialize the set of available library functions based on the specified
/// target triple. This should be carefully written so that a missing target
/// triple gets a sane set of defaults.
static void initialize(TargetLibraryInfoImpl &TLI, const Triple &T,
                       ArrayRef<StringLiteral> StandardNames) {
  // Verify that the StandardNames array is in alphabetical order.
  assert(
      llvm::is_sorted(StandardNames,
                      [](StringRef LHS, StringRef RHS) { return LHS < RHS; }) &&
      "TargetLibraryInfoImpl function names must be sorted");

  // Set IO unlocked variants as unavailable
  // Set them as available per system below
  TLI.setUnavailable(LibFunc_getc_unlocked);
  TLI.setUnavailable(LibFunc_getchar_unlocked);
  TLI.setUnavailable(LibFunc_putc_unlocked);
  TLI.setUnavailable(LibFunc_putchar_unlocked);
  TLI.setUnavailable(LibFunc_fputc_unlocked);
  TLI.setUnavailable(LibFunc_fgetc_unlocked);
  TLI.setUnavailable(LibFunc_fread_unlocked);
  TLI.setUnavailable(LibFunc_fwrite_unlocked);
  TLI.setUnavailable(LibFunc_fputs_unlocked);
  TLI.setUnavailable(LibFunc_fgets_unlocked);

  bool ShouldExtI32Param = false, ShouldExtI32Return = false,
       ShouldSignExtI32Param = false;
  // PowerPC64, Sparc64, SystemZ need signext/zeroext on i32 parameters and
  // returns corresponding to C-level ints and unsigned ints.
  if (T.isPPC64() || T.getArch() == Triple::sparcv9 ||
      T.getArch() == Triple::systemz) {
    ShouldExtI32Param = true;
    ShouldExtI32Return = true;
  }
  // Mips, on the other hand, needs signext on i32 parameters corresponding
  // to both signed and unsigned ints.
  if (T.isMIPS()) {
    ShouldSignExtI32Param = true;
  }
  TLI.setShouldExtI32Param(ShouldExtI32Param);
  TLI.setShouldExtI32Return(ShouldExtI32Return);
  TLI.setShouldSignExtI32Param(ShouldSignExtI32Param);

  // Let's assume by default that the size of int is 32 bits, unless the target
  // is a 16-bit architecture because then it most likely is 16 bits. If that
  // isn't true for a target those defaults should be overridden below.
  TLI.setIntSize(T.isArch16Bit() ? 16 : 32);

  // There is really no runtime library on AMDGPU, apart from
  // __kmpc_alloc/free_shared.
  if (T.isAMDGPU()) {
    TLI.disableAllFunctions();
    TLI.setAvailable(llvm::LibFunc___kmpc_alloc_shared);
    TLI.setAvailable(llvm::LibFunc___kmpc_free_shared);
    return;
  }

  // memset_pattern{4,8,16} is only available on iOS 3.0 and Mac OS X 10.5 and
  // later. All versions of watchOS support it.
  if (T.isMacOSX()) {
    // available IO unlocked variants on Mac OS X
    TLI.setAvailable(LibFunc_getc_unlocked);
    TLI.setAvailable(LibFunc_getchar_unlocked);
    TLI.setAvailable(LibFunc_putc_unlocked);
    TLI.setAvailable(LibFunc_putchar_unlocked);

    if (T.isMacOSXVersionLT(10, 5)) {
      TLI.setUnavailable(LibFunc_memset_pattern4);
      TLI.setUnavailable(LibFunc_memset_pattern8);
      TLI.setUnavailable(LibFunc_memset_pattern16);
    }
  } else if (T.isiOS()) {
    if (T.isOSVersionLT(3, 0)) {
      TLI.setUnavailable(LibFunc_memset_pattern4);
      TLI.setUnavailable(LibFunc_memset_pattern8);
      TLI.setUnavailable(LibFunc_memset_pattern16);
    }
  } else if (!T.isWatchOS()) {
    TLI.setUnavailable(LibFunc_memset_pattern4);
    TLI.setUnavailable(LibFunc_memset_pattern8);
    TLI.setUnavailable(LibFunc_memset_pattern16);
  }

  if (!hasSinCosPiStret(T)) {
    TLI.setUnavailable(LibFunc_sinpi);
    TLI.setUnavailable(LibFunc_sinpif);
    TLI.setUnavailable(LibFunc_cospi);
    TLI.setUnavailable(LibFunc_cospif);
    TLI.setUnavailable(LibFunc_sincospi_stret);
    TLI.setUnavailable(LibFunc_sincospif_stret);
  }

  if (!hasBcmp(T))
    TLI.setUnavailable(LibFunc_bcmp);

  if (T.isMacOSX() && T.getArch() == Triple::x86 &&
      !T.isMacOSXVersionLT(10, 7)) {
    // x86-32 OSX has a scheme where fwrite and fputs (and some other functions
    // we don't care about) have two versions; on recent OSX, the one we want
    // has a $UNIX2003 suffix. The two implementations are identical except
    // for the return value in some edge cases.  However, we don't want to
    // generate code that depends on the old symbols.
    TLI.setAvailableWithName(LibFunc_fwrite, "fwrite$UNIX2003");
    TLI.setAvailableWithName(LibFunc_fputs, "fputs$UNIX2003");
  }

  // iprintf and friends are only available on XCore, TCE, and Emscripten.
  if (T.getArch() != Triple::xcore && T.getArch() != Triple::tce &&
      T.getOS() != Triple::Emscripten) {
    TLI.setUnavailable(LibFunc_iprintf);
    TLI.setUnavailable(LibFunc_siprintf);
    TLI.setUnavailable(LibFunc_fiprintf);
  }

  // __small_printf and friends are only available on Emscripten.
  if (T.getOS() != Triple::Emscripten) {
    TLI.setUnavailable(LibFunc_small_printf);
    TLI.setUnavailable(LibFunc_small_sprintf);
    TLI.setUnavailable(LibFunc_small_fprintf);
  }

  if (T.isOSWindows() && !T.isOSCygMing()) {
    // XXX: The earliest documentation available at the moment is for VS2015/VC19:
    // https://docs.microsoft.com/en-us/cpp/c-runtime-library/floating-point-support?view=vs-2015
    // XXX: In order to use an MSVCRT older than VC19,
    // the specific library version must be explicit in the target triple,
    // e.g., x86_64-pc-windows-msvc18.
    bool hasPartialC99 = true;
    if (T.isKnownWindowsMSVCEnvironment()) {
      VersionTuple Version = T.getEnvironmentVersion();
      hasPartialC99 = (Version.getMajor() == 0 || Version.getMajor() >= 19);
    }

    // Latest targets support C89 math functions, in part.
    bool isARM = (T.getArch() == Triple::aarch64 ||
                  T.getArch() == Triple::arm);
    bool hasPartialFloat = (isARM ||
                            T.getArch() == Triple::x86_64);

    // Win32 does not support float C89 math functions, in general.
    if (!hasPartialFloat) {
      TLI.setUnavailable(LibFunc_acosf);
      TLI.setUnavailable(LibFunc_asinf);
      TLI.setUnavailable(LibFunc_atan2f);
      TLI.setUnavailable(LibFunc_atanf);
      TLI.setUnavailable(LibFunc_ceilf);
      TLI.setUnavailable(LibFunc_cosf);
      TLI.setUnavailable(LibFunc_coshf);
      TLI.setUnavailable(LibFunc_expf);
      TLI.setUnavailable(LibFunc_floorf);
      TLI.setUnavailable(LibFunc_fmodf);
      TLI.setUnavailable(LibFunc_log10f);
      TLI.setUnavailable(LibFunc_logf);
      TLI.setUnavailable(LibFunc_modff);
      TLI.setUnavailable(LibFunc_powf);
      TLI.setUnavailable(LibFunc_remainderf);
      TLI.setUnavailable(LibFunc_sinf);
      TLI.setUnavailable(LibFunc_sinhf);
      TLI.setUnavailable(LibFunc_sqrtf);
      TLI.setUnavailable(LibFunc_tanf);
      TLI.setUnavailable(LibFunc_tanhf);
    }
    if (!isARM)
      TLI.setUnavailable(LibFunc_fabsf);
    TLI.setUnavailable(LibFunc_frexpf);
    TLI.setUnavailable(LibFunc_ldexpf);

    // Win32 does not support long double C89 math functions.
    TLI.setUnavailable(LibFunc_acosl);
    TLI.setUnavailable(LibFunc_asinl);
    TLI.setUnavailable(LibFunc_atan2l);
    TLI.setUnavailable(LibFunc_atanl);
    TLI.setUnavailable(LibFunc_ceill);
    TLI.setUnavailable(LibFunc_cosl);
    TLI.setUnavailable(LibFunc_coshl);
    TLI.setUnavailable(LibFunc_expl);
    TLI.setUnavailable(LibFunc_fabsl);
    TLI.setUnavailable(LibFunc_floorl);
    TLI.setUnavailable(LibFunc_fmodl);
    TLI.setUnavailable(LibFunc_frexpl);
    TLI.setUnavailable(LibFunc_ldexpl);
    TLI.setUnavailable(LibFunc_log10l);
    TLI.setUnavailable(LibFunc_logl);
    TLI.setUnavailable(LibFunc_modfl);
    TLI.setUnavailable(LibFunc_powl);
    TLI.setUnavailable(LibFunc_remainderl);
    TLI.setUnavailable(LibFunc_sinl);
    TLI.setUnavailable(LibFunc_sinhl);
    TLI.setUnavailable(LibFunc_sqrtl);
    TLI.setUnavailable(LibFunc_tanl);
    TLI.setUnavailable(LibFunc_tanhl);

    // Win32 does not fully support C99 math functions.
    if (!hasPartialC99) {
      TLI.setUnavailable(LibFunc_acosh);
      TLI.setUnavailable(LibFunc_acoshf);
      TLI.setUnavailable(LibFunc_asinh);
      TLI.setUnavailable(LibFunc_asinhf);
      TLI.setUnavailable(LibFunc_atanh);
      TLI.setUnavailable(LibFunc_atanhf);
      TLI.setAvailableWithName(LibFunc_cabs, "_cabs");
      TLI.setUnavailable(LibFunc_cabsf);
      TLI.setUnavailable(LibFunc_cbrt);
      TLI.setUnavailable(LibFunc_cbrtf);
      TLI.setAvailableWithName(LibFunc_copysign, "_copysign");
      TLI.setAvailableWithName(LibFunc_copysignf, "_copysignf");
      TLI.setUnavailable(LibFunc_exp2);
      TLI.setUnavailable(LibFunc_exp2f);
      TLI.setUnavailable(LibFunc_expm1);
      TLI.setUnavailable(LibFunc_expm1f);
      TLI.setUnavailable(LibFunc_fmax);
      TLI.setUnavailable(LibFunc_fmaxf);
      TLI.setUnavailable(LibFunc_fmin);
      TLI.setUnavailable(LibFunc_fminf);
      TLI.setUnavailable(LibFunc_log1p);
      TLI.setUnavailable(LibFunc_log1pf);
      TLI.setUnavailable(LibFunc_log2);
      TLI.setUnavailable(LibFunc_log2f);
      TLI.setAvailableWithName(LibFunc_logb, "_logb");
      if (hasPartialFloat)
        TLI.setAvailableWithName(LibFunc_logbf, "_logbf");
      else
        TLI.setUnavailable(LibFunc_logbf);
      TLI.setUnavailable(LibFunc_rint);
      TLI.setUnavailable(LibFunc_rintf);
      TLI.setUnavailable(LibFunc_round);
      TLI.setUnavailable(LibFunc_roundf);
      TLI.setUnavailable(LibFunc_trunc);
      TLI.setUnavailable(LibFunc_truncf);
    }

    // Win32 does not support long double C99 math functions.
    TLI.setUnavailable(LibFunc_acoshl);
    TLI.setUnavailable(LibFunc_asinhl);
    TLI.setUnavailable(LibFunc_atanhl);
    TLI.setUnavailable(LibFunc_cabsl);
    TLI.setUnavailable(LibFunc_cbrtl);
    TLI.setUnavailable(LibFunc_copysignl);
    TLI.setUnavailable(LibFunc_exp2l);
    TLI.setUnavailable(LibFunc_expm1l);
    TLI.setUnavailable(LibFunc_fmaxl);
    TLI.setUnavailable(LibFunc_fminl);
    TLI.setUnavailable(LibFunc_log1pl);
    TLI.setUnavailable(LibFunc_log2l);
    TLI.setUnavailable(LibFunc_logbl);
    TLI.setUnavailable(LibFunc_nearbyintl);
    TLI.setUnavailable(LibFunc_rintl);
    TLI.setUnavailable(LibFunc_roundl);
    TLI.setUnavailable(LibFunc_truncl);

    // Win32 does not support these functions, but
    // they are generally available on POSIX-compliant systems.
    TLI.setUnavailable(LibFunc_access);
    TLI.setUnavailable(LibFunc_chmod);
    TLI.setUnavailable(LibFunc_closedir);
    TLI.setUnavailable(LibFunc_fdopen);
    TLI.setUnavailable(LibFunc_fileno);
    TLI.setUnavailable(LibFunc_fseeko);
    TLI.setUnavailable(LibFunc_fstat);
    TLI.setUnavailable(LibFunc_ftello);
    TLI.setUnavailable(LibFunc_gettimeofday);
    TLI.setUnavailable(LibFunc_memccpy);
    TLI.setUnavailable(LibFunc_mkdir);
    TLI.setUnavailable(LibFunc_open);
    TLI.setUnavailable(LibFunc_opendir);
    TLI.setUnavailable(LibFunc_pclose);
    TLI.setUnavailable(LibFunc_popen);
    TLI.setUnavailable(LibFunc_read);
    TLI.setUnavailable(LibFunc_rmdir);
    TLI.setUnavailable(LibFunc_stat);
    TLI.setUnavailable(LibFunc_strcasecmp);
    TLI.setUnavailable(LibFunc_strncasecmp);
    TLI.setUnavailable(LibFunc_unlink);
    TLI.setUnavailable(LibFunc_utime);
    TLI.setUnavailable(LibFunc_write);
  }

  if (T.isOSWindows() && !T.isWindowsCygwinEnvironment()) {
    // These functions aren't available in either MSVC or MinGW environments.
    TLI.setUnavailable(LibFunc_bcmp);
    TLI.setUnavailable(LibFunc_bcopy);
    TLI.setUnavailable(LibFunc_bzero);
    TLI.setUnavailable(LibFunc_chown);
    TLI.setUnavailable(LibFunc_ctermid);
    TLI.setUnavailable(LibFunc_ffs);
    TLI.setUnavailable(LibFunc_flockfile);
    TLI.setUnavailable(LibFunc_fstatvfs);
    TLI.setUnavailable(LibFunc_ftrylockfile);
    TLI.setUnavailable(LibFunc_funlockfile);
    TLI.setUnavailable(LibFunc_getitimer);
    TLI.setUnavailable(LibFunc_getlogin_r);
    TLI.setUnavailable(LibFunc_getpwnam);
    TLI.setUnavailable(LibFunc_htonl);
    TLI.setUnavailable(LibFunc_htons);
    TLI.setUnavailable(LibFunc_lchown);
    TLI.setUnavailable(LibFunc_lstat);
    TLI.setUnavailable(LibFunc_ntohl);
    TLI.setUnavailable(LibFunc_ntohs);
    TLI.setUnavailable(LibFunc_pread);
    TLI.setUnavailable(LibFunc_pwrite);
    TLI.setUnavailable(LibFunc_readlink);
    TLI.setUnavailable(LibFunc_realpath);
    TLI.setUnavailable(LibFunc_setitimer);
    TLI.setUnavailable(LibFunc_statvfs);
    TLI.setUnavailable(LibFunc_stpcpy);
    TLI.setUnavailable(LibFunc_stpncpy);
    TLI.setUnavailable(LibFunc_times);
    TLI.setUnavailable(LibFunc_uname);
    TLI.setUnavailable(LibFunc_unsetenv);
    TLI.setUnavailable(LibFunc_utimes);
  }

  // Pick just one set of new/delete variants.
  if (T.isOSMSVCRT()) {
    // MSVC, doesn't have the Itanium new/delete.
    TLI.setUnavailable(LibFunc_ZdaPv);
    TLI.setUnavailable(LibFunc_ZdaPvRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_ZdaPvSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZdaPvSt11align_val_tRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_ZdaPvj);
    TLI.setUnavailable(LibFunc_ZdaPvjSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZdaPvm);
    TLI.setUnavailable(LibFunc_ZdaPvmSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZdlPv);
    TLI.setUnavailable(LibFunc_ZdlPvRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_ZdlPvSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZdlPvSt11align_val_tRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_ZdlPvj);
    TLI.setUnavailable(LibFunc_ZdlPvjSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZdlPvm);
    TLI.setUnavailable(LibFunc_ZdlPvmSt11align_val_t);
    TLI.setUnavailable(LibFunc_Znaj);
    TLI.setUnavailable(LibFunc_ZnajRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_ZnajSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZnajSt11align_val_tRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_Znam);
    TLI.setUnavailable(LibFunc_ZnamRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_ZnamSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZnamSt11align_val_tRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_Znwj);
    TLI.setUnavailable(LibFunc_ZnwjRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_ZnwjSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_Znwm);
    TLI.setUnavailable(LibFunc_ZnwmRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_ZnwmSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t);
  } else {
    // Not MSVC, assume it's Itanium.
    TLI.setUnavailable(LibFunc_msvc_new_int);
    TLI.setUnavailable(LibFunc_msvc_new_int_nothrow);
    TLI.setUnavailable(LibFunc_msvc_new_longlong);
    TLI.setUnavailable(LibFunc_msvc_new_longlong_nothrow);
    TLI.setUnavailable(LibFunc_msvc_delete_ptr32);
    TLI.setUnavailable(LibFunc_msvc_delete_ptr32_nothrow);
    TLI.setUnavailable(LibFunc_msvc_delete_ptr32_int);
    TLI.setUnavailable(LibFunc_msvc_delete_ptr64);
    TLI.setUnavailable(LibFunc_msvc_delete_ptr64_nothrow);
    TLI.setUnavailable(LibFunc_msvc_delete_ptr64_longlong);
    TLI.setUnavailable(LibFunc_msvc_new_array_int);
    TLI.setUnavailable(LibFunc_msvc_new_array_int_nothrow);
    TLI.setUnavailable(LibFunc_msvc_new_array_longlong);
    TLI.setUnavailable(LibFunc_msvc_new_array_longlong_nothrow);
    TLI.setUnavailable(LibFunc_msvc_delete_array_ptr32);
    TLI.setUnavailable(LibFunc_msvc_delete_array_ptr32_nothrow);
    TLI.setUnavailable(LibFunc_msvc_delete_array_ptr32_int);
    TLI.setUnavailable(LibFunc_msvc_delete_array_ptr64);
    TLI.setUnavailable(LibFunc_msvc_delete_array_ptr64_nothrow);
    TLI.setUnavailable(LibFunc_msvc_delete_array_ptr64_longlong);
  }

  switch (T.getOS()) {
  case Triple::MacOSX:
    // exp10 and exp10f are not available on OS X until 10.9 and iOS until 7.0
    // and their names are __exp10 and __exp10f. exp10l is not available on
    // OS X or iOS.
    TLI.setUnavailable(LibFunc_exp10l);
    if (T.isMacOSXVersionLT(10, 9)) {
      TLI.setUnavailable(LibFunc_exp10);
      TLI.setUnavailable(LibFunc_exp10f);
    } else {
      TLI.setAvailableWithName(LibFunc_exp10, "__exp10");
      TLI.setAvailableWithName(LibFunc_exp10f, "__exp10f");
    }
    break;
  case Triple::IOS:
  case Triple::TvOS:
  case Triple::WatchOS:
    TLI.setUnavailable(LibFunc_exp10l);
    if (!T.isWatchOS() &&
        (T.isOSVersionLT(7, 0) || (T.isOSVersionLT(9, 0) && T.isX86()))) {
      TLI.setUnavailable(LibFunc_exp10);
      TLI.setUnavailable(LibFunc_exp10f);
    } else {
      TLI.setAvailableWithName(LibFunc_exp10, "__exp10");
      TLI.setAvailableWithName(LibFunc_exp10f, "__exp10f");
    }
    break;
  case Triple::Linux:
    // exp10, exp10f, exp10l is available on Linux (GLIBC) but are extremely
    // buggy prior to glibc version 2.18. Until this version is widely deployed
    // or we have a reasonable detection strategy, we cannot use exp10 reliably
    // on Linux.
    //
    // Fall through to disable all of them.
    LLVM_FALLTHROUGH;
  default:
    TLI.setUnavailable(LibFunc_exp10);
    TLI.setUnavailable(LibFunc_exp10f);
    TLI.setUnavailable(LibFunc_exp10l);
  }

  // ffsl is available on at least Darwin, Mac OS X, iOS, FreeBSD, and
  // Linux (GLIBC):
  // http://developer.apple.com/library/mac/#documentation/Darwin/Reference/ManPages/man3/ffsl.3.html
  // http://svn.freebsd.org/base/head/lib/libc/string/ffsl.c
  // http://www.gnu.org/software/gnulib/manual/html_node/ffsl.html
  switch (T.getOS()) {
  case Triple::Darwin:
  case Triple::MacOSX:
  case Triple::IOS:
  case Triple::TvOS:
  case Triple::WatchOS:
  case Triple::FreeBSD:
  case Triple::Linux:
    break;
  default:
    TLI.setUnavailable(LibFunc_ffsl);
  }

  // ffsll is available on at least FreeBSD and Linux (GLIBC):
  // http://svn.freebsd.org/base/head/lib/libc/string/ffsll.c
  // http://www.gnu.org/software/gnulib/manual/html_node/ffsll.html
  switch (T.getOS()) {
  case Triple::Darwin:
  case Triple::MacOSX:
  case Triple::IOS:
  case Triple::TvOS:
  case Triple::WatchOS:
  case Triple::FreeBSD:
  case Triple::Linux:
    break;
  default:
    TLI.setUnavailable(LibFunc_ffsll);
  }

  // The following functions are available on at least FreeBSD:
  // http://svn.freebsd.org/base/head/lib/libc/string/fls.c
  // http://svn.freebsd.org/base/head/lib/libc/string/flsl.c
  // http://svn.freebsd.org/base/head/lib/libc/string/flsll.c
  if (!T.isOSFreeBSD()) {
    TLI.setUnavailable(LibFunc_fls);
    TLI.setUnavailable(LibFunc_flsl);
    TLI.setUnavailable(LibFunc_flsll);
  }

  // The following functions are only available on GNU/Linux (using glibc).
  // Linux variants without glibc (eg: bionic, musl) may have some subset.
  if (!T.isOSLinux() || !T.isGNUEnvironment()) {
    TLI.setUnavailable(LibFunc_dunder_strdup);
    TLI.setUnavailable(LibFunc_dunder_strtok_r);
    TLI.setUnavailable(LibFunc_dunder_isoc99_scanf);
    TLI.setUnavailable(LibFunc_dunder_isoc99_sscanf);
    TLI.setUnavailable(LibFunc_under_IO_getc);
    TLI.setUnavailable(LibFunc_under_IO_putc);
    // But, Android and musl have memalign.
    if (!T.isAndroid() && !T.isMusl())
      TLI.setUnavailable(LibFunc_memalign);
    TLI.setUnavailable(LibFunc_fopen64);
    TLI.setUnavailable(LibFunc_fseeko64);
    TLI.setUnavailable(LibFunc_fstat64);
    TLI.setUnavailable(LibFunc_fstatvfs64);
    TLI.setUnavailable(LibFunc_ftello64);
    TLI.setUnavailable(LibFunc_lstat64);
    TLI.setUnavailable(LibFunc_open64);
    TLI.setUnavailable(LibFunc_stat64);
    TLI.setUnavailable(LibFunc_statvfs64);
    TLI.setUnavailable(LibFunc_tmpfile64);

    // Relaxed math functions are included in math-finite.h on Linux (GLIBC).
    // Note that math-finite.h is no longer supported by top-of-tree GLIBC,
    // so we keep these functions around just so that they're recognized by
    // the ConstantFolder.
    TLI.setUnavailable(LibFunc_acos_finite);
    TLI.setUnavailable(LibFunc_acosf_finite);
    TLI.setUnavailable(LibFunc_acosl_finite);
    TLI.setUnavailable(LibFunc_acosh_finite);
    TLI.setUnavailable(LibFunc_acoshf_finite);
    TLI.setUnavailable(LibFunc_acoshl_finite);
    TLI.setUnavailable(LibFunc_asin_finite);
    TLI.setUnavailable(LibFunc_asinf_finite);
    TLI.setUnavailable(LibFunc_asinl_finite);
    TLI.setUnavailable(LibFunc_atan2_finite);
    TLI.setUnavailable(LibFunc_atan2f_finite);
    TLI.setUnavailable(LibFunc_atan2l_finite);
    TLI.setUnavailable(LibFunc_atanh_finite);
    TLI.setUnavailable(LibFunc_atanhf_finite);
    TLI.setUnavailable(LibFunc_atanhl_finite);
    TLI.setUnavailable(LibFunc_cosh_finite);
    TLI.setUnavailable(LibFunc_coshf_finite);
    TLI.setUnavailable(LibFunc_coshl_finite);
    TLI.setUnavailable(LibFunc_exp10_finite);
    TLI.setUnavailable(LibFunc_exp10f_finite);
    TLI.setUnavailable(LibFunc_exp10l_finite);
    TLI.setUnavailable(LibFunc_exp2_finite);
    TLI.setUnavailable(LibFunc_exp2f_finite);
    TLI.setUnavailable(LibFunc_exp2l_finite);
    TLI.setUnavailable(LibFunc_exp_finite);
    TLI.setUnavailable(LibFunc_expf_finite);
    TLI.setUnavailable(LibFunc_expl_finite);
    TLI.setUnavailable(LibFunc_log10_finite);
    TLI.setUnavailable(LibFunc_log10f_finite);
    TLI.setUnavailable(LibFunc_log10l_finite);
    TLI.setUnavailable(LibFunc_log2_finite);
    TLI.setUnavailable(LibFunc_log2f_finite);
    TLI.setUnavailable(LibFunc_log2l_finite);
    TLI.setUnavailable(LibFunc_log_finite);
    TLI.setUnavailable(LibFunc_logf_finite);
    TLI.setUnavailable(LibFunc_logl_finite);
    TLI.setUnavailable(LibFunc_pow_finite);
    TLI.setUnavailable(LibFunc_powf_finite);
    TLI.setUnavailable(LibFunc_powl_finite);
    TLI.setUnavailable(LibFunc_sinh_finite);
    TLI.setUnavailable(LibFunc_sinhf_finite);
    TLI.setUnavailable(LibFunc_sinhl_finite);
    TLI.setUnavailable(LibFunc_sqrt_finite);
    TLI.setUnavailable(LibFunc_sqrtf_finite);
    TLI.setUnavailable(LibFunc_sqrtl_finite);
  }

  if ((T.isOSLinux() && T.isGNUEnvironment()) ||
      (T.isAndroid() && !T.isAndroidVersionLT(28))) {
    // available IO unlocked variants on GNU/Linux and Android P or later
    TLI.setAvailable(LibFunc_getc_unlocked);
    TLI.setAvailable(LibFunc_getchar_unlocked);
    TLI.setAvailable(LibFunc_putc_unlocked);
    TLI.setAvailable(LibFunc_putchar_unlocked);
    TLI.setAvailable(LibFunc_fputc_unlocked);
    TLI.setAvailable(LibFunc_fgetc_unlocked);
    TLI.setAvailable(LibFunc_fread_unlocked);
    TLI.setAvailable(LibFunc_fwrite_unlocked);
    TLI.setAvailable(LibFunc_fputs_unlocked);
    TLI.setAvailable(LibFunc_fgets_unlocked);
  }

  if (T.isAndroid() && T.isAndroidVersionLT(21)) {
    TLI.setUnavailable(LibFunc_stpcpy);
    TLI.setUnavailable(LibFunc_stpncpy);
  }

  if (T.isPS4()) {
    // PS4 does have memalign.
    TLI.setAvailable(LibFunc_memalign);

    // PS4 does not have new/delete with "unsigned int" size parameter;
    // it only has the "unsigned long" versions.
    TLI.setUnavailable(LibFunc_ZdaPvj);
    TLI.setUnavailable(LibFunc_ZdaPvjSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZdlPvj);
    TLI.setUnavailable(LibFunc_ZdlPvjSt11align_val_t);
    TLI.setUnavailable(LibFunc_Znaj);
    TLI.setUnavailable(LibFunc_ZnajRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_ZnajSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZnajSt11align_val_tRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_Znwj);
    TLI.setUnavailable(LibFunc_ZnwjRKSt9nothrow_t);
    TLI.setUnavailable(LibFunc_ZnwjSt11align_val_t);
    TLI.setUnavailable(LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t);

    // None of the *_chk functions.
    TLI.setUnavailable(LibFunc_memccpy_chk);
    TLI.setUnavailable(LibFunc_memcpy_chk);
    TLI.setUnavailable(LibFunc_memmove_chk);
    TLI.setUnavailable(LibFunc_mempcpy_chk);
    TLI.setUnavailable(LibFunc_memset_chk);
    TLI.setUnavailable(LibFunc_snprintf_chk);
    TLI.setUnavailable(LibFunc_sprintf_chk);
    TLI.setUnavailable(LibFunc_stpcpy_chk);
    TLI.setUnavailable(LibFunc_stpncpy_chk);
    TLI.setUnavailable(LibFunc_strcat_chk);
    TLI.setUnavailable(LibFunc_strcpy_chk);
    TLI.setUnavailable(LibFunc_strlcat_chk);
    TLI.setUnavailable(LibFunc_strlcpy_chk);
    TLI.setUnavailable(LibFunc_strlen_chk);
    TLI.setUnavailable(LibFunc_strncat_chk);
    TLI.setUnavailable(LibFunc_strncpy_chk);
    TLI.setUnavailable(LibFunc_vsnprintf_chk);
    TLI.setUnavailable(LibFunc_vsprintf_chk);

    // Various Posix system functions.
    TLI.setUnavailable(LibFunc_access);
    TLI.setUnavailable(LibFunc_chmod);
    TLI.setUnavailable(LibFunc_chown);
    TLI.setUnavailable(LibFunc_closedir);
    TLI.setUnavailable(LibFunc_ctermid);
    TLI.setUnavailable(LibFunc_execl);
    TLI.setUnavailable(LibFunc_execle);
    TLI.setUnavailable(LibFunc_execlp);
    TLI.setUnavailable(LibFunc_execv);
    TLI.setUnavailable(LibFunc_execvP);
    TLI.setUnavailable(LibFunc_execve);
    TLI.setUnavailable(LibFunc_execvp);
    TLI.setUnavailable(LibFunc_execvpe);
    TLI.setUnavailable(LibFunc_fork);
    TLI.setUnavailable(LibFunc_fstat);
    TLI.setUnavailable(LibFunc_fstatvfs);
    TLI.setUnavailable(LibFunc_getenv);
    TLI.setUnavailable(LibFunc_getitimer);
    TLI.setUnavailable(LibFunc_getlogin_r);
    TLI.setUnavailable(LibFunc_getpwnam);
    TLI.setUnavailable(LibFunc_gettimeofday);
    TLI.setUnavailable(LibFunc_lchown);
    TLI.setUnavailable(LibFunc_lstat);
    TLI.setUnavailable(LibFunc_mkdir);
    TLI.setUnavailable(LibFunc_open);
    TLI.setUnavailable(LibFunc_opendir);
    TLI.setUnavailable(LibFunc_pclose);
    TLI.setUnavailable(LibFunc_popen);
    TLI.setUnavailable(LibFunc_pread);
    TLI.setUnavailable(LibFunc_pwrite);
    TLI.setUnavailable(LibFunc_read);
    TLI.setUnavailable(LibFunc_readlink);
    TLI.setUnavailable(LibFunc_realpath);
    TLI.setUnavailable(LibFunc_rename);
    TLI.setUnavailable(LibFunc_rmdir);
    TLI.setUnavailable(LibFunc_setitimer);
    TLI.setUnavailable(LibFunc_stat);
    TLI.setUnavailable(LibFunc_statvfs);
    TLI.setUnavailable(LibFunc_system);
    TLI.setUnavailable(LibFunc_times);
    TLI.setUnavailable(LibFunc_tmpfile);
    TLI.setUnavailable(LibFunc_unlink);
    TLI.setUnavailable(LibFunc_uname);
    TLI.setUnavailable(LibFunc_unsetenv);
    TLI.setUnavailable(LibFunc_utime);
    TLI.setUnavailable(LibFunc_utimes);
    TLI.setUnavailable(LibFunc_valloc);
    TLI.setUnavailable(LibFunc_write);

    // Miscellaneous other functions not provided.
    TLI.setUnavailable(LibFunc_atomic_load);
    TLI.setUnavailable(LibFunc_atomic_store);
    TLI.setUnavailable(LibFunc___kmpc_alloc_shared);
    TLI.setUnavailable(LibFunc___kmpc_free_shared);
    TLI.setUnavailable(LibFunc_dunder_strndup);
    TLI.setUnavailable(LibFunc_bcmp);
    TLI.setUnavailable(LibFunc_bcopy);
    TLI.setUnavailable(LibFunc_bzero);
    TLI.setUnavailable(LibFunc_cabs);
    TLI.setUnavailable(LibFunc_cabsf);
    TLI.setUnavailable(LibFunc_cabsl);
    TLI.setUnavailable(LibFunc_ffs);
    TLI.setUnavailable(LibFunc_flockfile);
    TLI.setUnavailable(LibFunc_fseeko);
    TLI.setUnavailable(LibFunc_ftello);
    TLI.setUnavailable(LibFunc_ftrylockfile);
    TLI.setUnavailable(LibFunc_funlockfile);
    TLI.setUnavailable(LibFunc_htonl);
    TLI.setUnavailable(LibFunc_htons);
    TLI.setUnavailable(LibFunc_isascii);
    TLI.setUnavailable(LibFunc_memccpy);
    TLI.setUnavailable(LibFunc_mempcpy);
    TLI.setUnavailable(LibFunc_memrchr);
    TLI.setUnavailable(LibFunc_ntohl);
    TLI.setUnavailable(LibFunc_ntohs);
    TLI.setUnavailable(LibFunc_reallocf);
    TLI.setUnavailable(LibFunc_roundeven);
    TLI.setUnavailable(LibFunc_roundevenf);
    TLI.setUnavailable(LibFunc_roundevenl);
    TLI.setUnavailable(LibFunc_stpcpy);
    TLI.setUnavailable(LibFunc_stpncpy);
    TLI.setUnavailable(LibFunc_strlcat);
    TLI.setUnavailable(LibFunc_strlcpy);
    TLI.setUnavailable(LibFunc_strndup);
    TLI.setUnavailable(LibFunc_strnlen);
    TLI.setUnavailable(LibFunc_toascii);
  }

  // As currently implemented in clang, NVPTX code has no standard library to
  // speak of.  Headers provide a standard-ish library implementation, but many
  // of the signatures are wrong -- for example, many libm functions are not
  // extern "C".
  //
  // libdevice, an IR library provided by nvidia, is linked in by the front-end,
  // but only used functions are provided to llvm.  Moreover, most of the
  // functions in libdevice don't map precisely to standard library functions.
  //
  // FIXME: Having no standard library prevents e.g. many fastmath
  // optimizations, so this situation should be fixed.
  if (T.isNVPTX()) {
    TLI.disableAllFunctions();
    TLI.setAvailable(LibFunc_nvvm_reflect);
    TLI.setAvailable(llvm::LibFunc_malloc);
    TLI.setAvailable(llvm::LibFunc_free);

    // TODO: We could enable the following two according to [0] but we haven't
    //       done an evaluation wrt. the performance implications.
    // [0]
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-global-memory-allocation-and-operations
    //
    //    TLI.setAvailable(llvm::LibFunc_memcpy);
    //    TLI.setAvailable(llvm::LibFunc_memset);

    TLI.setAvailable(llvm::LibFunc___kmpc_alloc_shared);
    TLI.setAvailable(llvm::LibFunc___kmpc_free_shared);
  } else {
    TLI.setUnavailable(LibFunc_nvvm_reflect);
  }

  // These vec_malloc/free routines are only available on AIX.
  if (!T.isOSAIX()) {
    TLI.setUnavailable(LibFunc_vec_calloc);
    TLI.setUnavailable(LibFunc_vec_malloc);
    TLI.setUnavailable(LibFunc_vec_realloc);
    TLI.setUnavailable(LibFunc_vec_free);
  }

  TLI.addVectorizableFunctionsFromVecLib(ClVectorLibrary);
}

TargetLibraryInfoImpl::TargetLibraryInfoImpl() {
  // Default to everything being available.
  memset(AvailableArray, -1, sizeof(AvailableArray));

  initialize(*this, Triple(), StandardNames);
}

TargetLibraryInfoImpl::TargetLibraryInfoImpl(const Triple &T) {
  // Default to everything being available.
  memset(AvailableArray, -1, sizeof(AvailableArray));

  initialize(*this, T, StandardNames);
}

TargetLibraryInfoImpl::TargetLibraryInfoImpl(const TargetLibraryInfoImpl &TLI)
    : CustomNames(TLI.CustomNames), ShouldExtI32Param(TLI.ShouldExtI32Param),
      ShouldExtI32Return(TLI.ShouldExtI32Return),
      ShouldSignExtI32Param(TLI.ShouldSignExtI32Param),
      SizeOfInt(TLI.SizeOfInt) {
  memcpy(AvailableArray, TLI.AvailableArray, sizeof(AvailableArray));
  VectorDescs = TLI.VectorDescs;
  ScalarDescs = TLI.ScalarDescs;
}

TargetLibraryInfoImpl::TargetLibraryInfoImpl(TargetLibraryInfoImpl &&TLI)
    : CustomNames(std::move(TLI.CustomNames)),
      ShouldExtI32Param(TLI.ShouldExtI32Param),
      ShouldExtI32Return(TLI.ShouldExtI32Return),
      ShouldSignExtI32Param(TLI.ShouldSignExtI32Param),
      SizeOfInt(TLI.SizeOfInt) {
  std::move(std::begin(TLI.AvailableArray), std::end(TLI.AvailableArray),
            AvailableArray);
  VectorDescs = TLI.VectorDescs;
  ScalarDescs = TLI.ScalarDescs;
}

TargetLibraryInfoImpl &TargetLibraryInfoImpl::operator=(const TargetLibraryInfoImpl &TLI) {
  CustomNames = TLI.CustomNames;
  ShouldExtI32Param = TLI.ShouldExtI32Param;
  ShouldExtI32Return = TLI.ShouldExtI32Return;
  ShouldSignExtI32Param = TLI.ShouldSignExtI32Param;
  SizeOfInt = TLI.SizeOfInt;
  memcpy(AvailableArray, TLI.AvailableArray, sizeof(AvailableArray));
  return *this;
}

TargetLibraryInfoImpl &TargetLibraryInfoImpl::operator=(TargetLibraryInfoImpl &&TLI) {
  CustomNames = std::move(TLI.CustomNames);
  ShouldExtI32Param = TLI.ShouldExtI32Param;
  ShouldExtI32Return = TLI.ShouldExtI32Return;
  ShouldSignExtI32Param = TLI.ShouldSignExtI32Param;
  SizeOfInt = TLI.SizeOfInt;
  std::move(std::begin(TLI.AvailableArray), std::end(TLI.AvailableArray),
            AvailableArray);
  return *this;
}

static StringRef sanitizeFunctionName(StringRef funcName) {
  // Filter out empty names and names containing null bytes, those can't be in
  // our table.
  if (funcName.empty() || funcName.contains('\0'))
    return StringRef();

  // Check for \01 prefix that is used to mangle __asm declarations and
  // strip it if present.
  return GlobalValue::dropLLVMManglingEscape(funcName);
}

bool TargetLibraryInfoImpl::getLibFunc(StringRef funcName, LibFunc &F) const {
  funcName = sanitizeFunctionName(funcName);
  if (funcName.empty())
    return false;

  const auto *Start = std::begin(StandardNames);
  const auto *End = std::end(StandardNames);
  const auto *I = std::lower_bound(Start, End, funcName);
  if (I != End && *I == funcName) {
    F = (LibFunc)(I - Start);
    return true;
  }
  return false;
}

bool TargetLibraryInfoImpl::isValidProtoForLibFunc(const FunctionType &FTy,
                                                   LibFunc F,
                                                   const Module &M) const {
  // FIXME: There is really no guarantee that sizeof(size_t) is equal to
  // sizeof(int*) for every target. So the assumption used here to derive the
  // SizeTBits based on the size of an integer pointer in address space zero
  // isn't always valid.
  unsigned SizeTBits = M.getDataLayout().getPointerSizeInBits(/*AddrSpace=*/0);
  unsigned NumParams = FTy.getNumParams();

  switch (F) {
  case LibFunc_execl:
  case LibFunc_execlp:
  case LibFunc_execle:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            FTy.getReturnType()->isIntegerTy(32));
  case LibFunc_execv:
  case LibFunc_execvp:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            FTy.getReturnType()->isIntegerTy(32));
  case LibFunc_execvP:
  case LibFunc_execvpe:
  case LibFunc_execve:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy() &&
            FTy.getReturnType()->isIntegerTy(32));
  case LibFunc_strlen_chk:
    --NumParams;
    if (!FTy.getParamType(NumParams)->isIntegerTy(SizeTBits))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc_strlen:
    return NumParams == 1 && FTy.getParamType(0)->isPointerTy() &&
           FTy.getReturnType()->isIntegerTy(SizeTBits);

  case LibFunc_strchr:
  case LibFunc_strrchr:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0) == FTy.getReturnType() &&
            FTy.getParamType(1)->isIntegerTy());

  case LibFunc_strtol:
  case LibFunc_strtod:
  case LibFunc_strtof:
  case LibFunc_strtoul:
  case LibFunc_strtoll:
  case LibFunc_strtold:
  case LibFunc_strtoull:
    return ((NumParams == 2 || NumParams == 3) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_strcat_chk:
    --NumParams;
    if (!FTy.getParamType(NumParams)->isIntegerTy(SizeTBits))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc_strcat:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0) == FTy.getReturnType() &&
            FTy.getParamType(1) == FTy.getReturnType());

  case LibFunc_strncat_chk:
    --NumParams;
    if (!FTy.getParamType(NumParams)->isIntegerTy(SizeTBits))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc_strncat:
    return (NumParams == 3 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0) == FTy.getReturnType() &&
            FTy.getParamType(1) == FTy.getReturnType() &&
            FTy.getParamType(2)->isIntegerTy(SizeTBits));

  case LibFunc_strcpy_chk:
  case LibFunc_stpcpy_chk:
    --NumParams;
    if (!FTy.getParamType(NumParams)->isIntegerTy(SizeTBits))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc_strcpy:
  case LibFunc_stpcpy:
    return (NumParams == 2 && FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(0) == FTy.getParamType(1) &&
            FTy.getParamType(0)->isPointerTy());

  case LibFunc_strlcat_chk:
  case LibFunc_strlcpy_chk:
    --NumParams;
    if (!FTy.getParamType(NumParams)->isIntegerTy(SizeTBits))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc_strlcat:
  case LibFunc_strlcpy:
    return NumParams == 3 && FTy.getReturnType()->isIntegerTy(SizeTBits) &&
           FTy.getParamType(0)->isPointerTy() &&
           FTy.getParamType(1)->isPointerTy() &&
           FTy.getParamType(2)->isIntegerTy(SizeTBits);

  case LibFunc_strncpy_chk:
  case LibFunc_stpncpy_chk:
    --NumParams;
    if (!FTy.getParamType(NumParams)->isIntegerTy(SizeTBits))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc_strncpy:
  case LibFunc_stpncpy:
    return (NumParams == 3 && FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(0) == FTy.getParamType(1) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(2)->isIntegerTy(SizeTBits));

  case LibFunc_strxfrm:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());

  case LibFunc_strcmp:
    return (NumParams == 2 && FTy.getReturnType()->isIntegerTy(32) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(0) == FTy.getParamType(1));

  case LibFunc_strncmp:
    return (NumParams == 3 && FTy.getReturnType()->isIntegerTy(32) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(0) == FTy.getParamType(1) &&
            FTy.getParamType(2)->isIntegerTy(SizeTBits));

  case LibFunc_strspn:
  case LibFunc_strcspn:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(0) == FTy.getParamType(1) &&
            FTy.getReturnType()->isIntegerTy());

  case LibFunc_strcoll:
  case LibFunc_strcasecmp:
  case LibFunc_strncasecmp:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());

  case LibFunc_strstr:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());

  case LibFunc_strpbrk:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(0) == FTy.getParamType(1));

  case LibFunc_strtok:
  case LibFunc_strtok_r:
    return (NumParams >= 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_scanf:
  case LibFunc_setbuf:
  case LibFunc_setvbuf:
    return (NumParams >= 1 && FTy.getParamType(0)->isPointerTy());
  case LibFunc_strdup:
  case LibFunc_strndup:
    return (NumParams >= 1 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy());
  case LibFunc_sscanf:
  case LibFunc_stat:
  case LibFunc_statvfs:
  case LibFunc_siprintf:
  case LibFunc_small_sprintf:
  case LibFunc_sprintf:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            FTy.getReturnType()->isIntegerTy(32));

  case LibFunc_sprintf_chk:
    return NumParams == 4 && FTy.getParamType(0)->isPointerTy() &&
           FTy.getParamType(1)->isIntegerTy(32) &&
           FTy.getParamType(2)->isIntegerTy(SizeTBits) &&
           FTy.getParamType(3)->isPointerTy() &&
           FTy.getReturnType()->isIntegerTy(32);

  case LibFunc_snprintf:
    return NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
           FTy.getParamType(1)->isIntegerTy(SizeTBits) &&
           FTy.getParamType(2)->isPointerTy() &&
           FTy.getReturnType()->isIntegerTy(32);

  case LibFunc_snprintf_chk:
    return NumParams == 5 && FTy.getParamType(0)->isPointerTy() &&
           FTy.getParamType(1)->isIntegerTy(SizeTBits) &&
           FTy.getParamType(2)->isIntegerTy(32) &&
           FTy.getParamType(3)->isIntegerTy(SizeTBits) &&
           FTy.getParamType(4)->isPointerTy() &&
           FTy.getReturnType()->isIntegerTy(32);

  case LibFunc_setitimer:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc_system:
    return (NumParams == 1 && FTy.getParamType(0)->isPointerTy());
  case LibFunc___kmpc_alloc_shared:
  case LibFunc_malloc:
  case LibFunc_vec_malloc:
    return (NumParams == 1 && FTy.getReturnType()->isPointerTy());
  case LibFunc_memcmp:
    return NumParams == 3 && FTy.getReturnType()->isIntegerTy(32) &&
           FTy.getParamType(0)->isPointerTy() &&
           FTy.getParamType(1)->isPointerTy() &&
           FTy.getParamType(2)->isIntegerTy(SizeTBits);

  case LibFunc_memchr:
  case LibFunc_memrchr:
    return (NumParams == 3 && FTy.getReturnType()->isPointerTy() &&
            FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(1)->isIntegerTy(32) &&
            FTy.getParamType(2)->isIntegerTy(SizeTBits));
  case LibFunc_modf:
  case LibFunc_modff:
  case LibFunc_modfl:
    return (NumParams >= 2 && FTy.getParamType(1)->isPointerTy());

  case LibFunc_memcpy_chk:
  case LibFunc_mempcpy_chk:
  case LibFunc_memmove_chk:
    --NumParams;
    if (!FTy.getParamType(NumParams)->isIntegerTy(SizeTBits))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc_memcpy:
  case LibFunc_mempcpy:
  case LibFunc_memmove:
    return (NumParams == 3 && FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isIntegerTy(SizeTBits));

  case LibFunc_memset_chk:
    --NumParams;
    if (!FTy.getParamType(NumParams)->isIntegerTy(SizeTBits))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc_memset:
    return (NumParams == 3 && FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isIntegerTy() &&
            FTy.getParamType(2)->isIntegerTy(SizeTBits));

  case LibFunc_memccpy_chk:
      --NumParams;
    if (!FTy.getParamType(NumParams)->isIntegerTy(SizeTBits))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc_memccpy:
    return (NumParams >= 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_memalign:
    return (FTy.getReturnType()->isPointerTy());
  case LibFunc_realloc:
  case LibFunc_reallocf:
  case LibFunc_vec_realloc:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0) == FTy.getReturnType() &&
            FTy.getParamType(1)->isIntegerTy(SizeTBits));
  case LibFunc_read:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_rewind:
  case LibFunc_rmdir:
  case LibFunc_remove:
  case LibFunc_realpath:
    return (NumParams >= 1 && FTy.getParamType(0)->isPointerTy());
  case LibFunc_rename:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_readlink:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_write:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_aligned_alloc:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy());
  case LibFunc_bcopy:
  case LibFunc_bcmp:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_bzero:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy());
  case LibFunc_calloc:
  case LibFunc_vec_calloc:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0) == FTy.getParamType(1));

  case LibFunc_atof:
  case LibFunc_atoi:
  case LibFunc_atol:
  case LibFunc_atoll:
  case LibFunc_ferror:
  case LibFunc_getenv:
  case LibFunc_getpwnam:
  case LibFunc_iprintf:
  case LibFunc_small_printf:
  case LibFunc_pclose:
  case LibFunc_perror:
  case LibFunc_printf:
  case LibFunc_puts:
  case LibFunc_uname:
  case LibFunc_under_IO_getc:
  case LibFunc_unlink:
  case LibFunc_unsetenv:
    return (NumParams == 1 && FTy.getParamType(0)->isPointerTy());

  case LibFunc_access:
  case LibFunc_chmod:
  case LibFunc_chown:
  case LibFunc_clearerr:
  case LibFunc_closedir:
  case LibFunc_ctermid:
  case LibFunc_fclose:
  case LibFunc_feof:
  case LibFunc_fflush:
  case LibFunc_fgetc:
  case LibFunc_fgetc_unlocked:
  case LibFunc_fileno:
  case LibFunc_flockfile:
  case LibFunc_free:
  case LibFunc_fseek:
  case LibFunc_fseeko64:
  case LibFunc_fseeko:
  case LibFunc_fsetpos:
  case LibFunc_ftell:
  case LibFunc_ftello64:
  case LibFunc_ftello:
  case LibFunc_ftrylockfile:
  case LibFunc_funlockfile:
  case LibFunc_getc:
  case LibFunc_getc_unlocked:
  case LibFunc_getlogin_r:
  case LibFunc_mkdir:
  case LibFunc_mktime:
  case LibFunc_times:
  case LibFunc_vec_free:
    return (NumParams != 0 && FTy.getParamType(0)->isPointerTy());
  case LibFunc___kmpc_free_shared:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isIntegerTy(SizeTBits));

  case LibFunc_fopen:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_fork:
    return (NumParams == 0 && FTy.getReturnType()->isIntegerTy(32));
  case LibFunc_fdopen:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_fputc:
  case LibFunc_fputc_unlocked:
  case LibFunc_fstat:
  case LibFunc_frexp:
  case LibFunc_frexpf:
  case LibFunc_frexpl:
  case LibFunc_fstatvfs:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_fgets:
  case LibFunc_fgets_unlocked:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc_fread:
  case LibFunc_fread_unlocked:
    return (NumParams == 4 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(3)->isPointerTy());
  case LibFunc_fwrite:
  case LibFunc_fwrite_unlocked:
    return (NumParams == 4 && FTy.getReturnType()->isIntegerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isIntegerTy() &&
            FTy.getParamType(2)->isIntegerTy() &&
            FTy.getParamType(3)->isPointerTy());
  case LibFunc_fputs:
  case LibFunc_fputs_unlocked:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_fscanf:
  case LibFunc_fiprintf:
  case LibFunc_small_fprintf:
  case LibFunc_fprintf:
    return (NumParams >= 2 && FTy.getReturnType()->isIntegerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_fgetpos:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_getchar:
  case LibFunc_getchar_unlocked:
    return (NumParams == 0 && FTy.getReturnType()->isIntegerTy());
  case LibFunc_gets:
    return (NumParams == 1 && FTy.getParamType(0)->isPointerTy());
  case LibFunc_getitimer:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_ungetc:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_utime:
  case LibFunc_utimes:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_putc:
  case LibFunc_putc_unlocked:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_pread:
  case LibFunc_pwrite:
    return (NumParams == 4 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_popen:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_vscanf:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_vsscanf:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc_vfscanf:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc_valloc:
    return (FTy.getReturnType()->isPointerTy());
  case LibFunc_vprintf:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy());
  case LibFunc_vfprintf:
  case LibFunc_vsprintf:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_vsprintf_chk:
    return NumParams == 5 && FTy.getParamType(0)->isPointerTy() &&
           FTy.getParamType(1)->isIntegerTy(32) &&
           FTy.getParamType(2)->isIntegerTy(SizeTBits) && FTy.getParamType(3)->isPointerTy();
  case LibFunc_vsnprintf:
    return (NumParams == 4 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc_vsnprintf_chk:
    return NumParams == 6 && FTy.getParamType(0)->isPointerTy() &&
           FTy.getParamType(2)->isIntegerTy(32) &&
           FTy.getParamType(3)->isIntegerTy(SizeTBits) && FTy.getParamType(4)->isPointerTy();
  case LibFunc_open:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy());
  case LibFunc_opendir:
    return (NumParams == 1 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy());
  case LibFunc_tmpfile:
    return (FTy.getReturnType()->isPointerTy());
  case LibFunc_htonl:
  case LibFunc_ntohl:
    return (NumParams == 1 && FTy.getReturnType()->isIntegerTy(32) &&
            FTy.getReturnType() == FTy.getParamType(0));
  case LibFunc_htons:
  case LibFunc_ntohs:
    return (NumParams == 1 && FTy.getReturnType()->isIntegerTy(16) &&
            FTy.getReturnType() == FTy.getParamType(0));
  case LibFunc_lstat:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_lchown:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy());
  case LibFunc_qsort:
    return (NumParams == 4 && FTy.getParamType(3)->isPointerTy());
  case LibFunc_dunder_strdup:
  case LibFunc_dunder_strndup:
    return (NumParams >= 1 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy());
  case LibFunc_dunder_strtok_r:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_under_IO_putc:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_dunder_isoc99_scanf:
    return (NumParams >= 1 && FTy.getParamType(0)->isPointerTy());
  case LibFunc_stat64:
  case LibFunc_lstat64:
  case LibFunc_statvfs64:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_dunder_isoc99_sscanf:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_fopen64:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc_tmpfile64:
    return (FTy.getReturnType()->isPointerTy());
  case LibFunc_fstat64:
  case LibFunc_fstatvfs64:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc_open64:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy());
  case LibFunc_gettimeofday:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());

  // new(unsigned int);
  case LibFunc_Znwj:
  // new(unsigned long);
  case LibFunc_Znwm:
  // new[](unsigned int);
  case LibFunc_Znaj:
  // new[](unsigned long);
  case LibFunc_Znam:
  // new(unsigned int);
  case LibFunc_msvc_new_int:
  // new(unsigned long long);
  case LibFunc_msvc_new_longlong:
  // new[](unsigned int);
  case LibFunc_msvc_new_array_int:
  // new[](unsigned long long);
  case LibFunc_msvc_new_array_longlong:
    return (NumParams == 1 && FTy.getReturnType()->isPointerTy());

  // new(unsigned int, nothrow);
  case LibFunc_ZnwjRKSt9nothrow_t:
  // new(unsigned long, nothrow);
  case LibFunc_ZnwmRKSt9nothrow_t:
  // new[](unsigned int, nothrow);
  case LibFunc_ZnajRKSt9nothrow_t:
  // new[](unsigned long, nothrow);
  case LibFunc_ZnamRKSt9nothrow_t:
  // new(unsigned int, nothrow);
  case LibFunc_msvc_new_int_nothrow:
  // new(unsigned long long, nothrow);
  case LibFunc_msvc_new_longlong_nothrow:
  // new[](unsigned int, nothrow);
  case LibFunc_msvc_new_array_int_nothrow:
  // new[](unsigned long long, nothrow);
  case LibFunc_msvc_new_array_longlong_nothrow:
  // new(unsigned int, align_val_t)
  case LibFunc_ZnwjSt11align_val_t:
  // new(unsigned long, align_val_t)
  case LibFunc_ZnwmSt11align_val_t:
  // new[](unsigned int, align_val_t)
  case LibFunc_ZnajSt11align_val_t:
  // new[](unsigned long, align_val_t)
  case LibFunc_ZnamSt11align_val_t:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy());

  // new(unsigned int, align_val_t, nothrow)
  case LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t:
  // new(unsigned long, align_val_t, nothrow)
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t:
  // new[](unsigned int, align_val_t, nothrow)
  case LibFunc_ZnajSt11align_val_tRKSt9nothrow_t:
  // new[](unsigned long, align_val_t, nothrow)
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t:
    return (NumParams == 3 && FTy.getReturnType()->isPointerTy());

  // void operator delete[](void*);
  case LibFunc_ZdaPv:
  // void operator delete(void*);
  case LibFunc_ZdlPv:
  // void operator delete[](void*);
  case LibFunc_msvc_delete_array_ptr32:
  // void operator delete[](void*);
  case LibFunc_msvc_delete_array_ptr64:
  // void operator delete(void*);
  case LibFunc_msvc_delete_ptr32:
  // void operator delete(void*);
  case LibFunc_msvc_delete_ptr64:
    return (NumParams == 1 && FTy.getParamType(0)->isPointerTy());

  // void operator delete[](void*, nothrow);
  case LibFunc_ZdaPvRKSt9nothrow_t:
  // void operator delete[](void*, unsigned int);
  case LibFunc_ZdaPvj:
  // void operator delete[](void*, unsigned long);
  case LibFunc_ZdaPvm:
  // void operator delete(void*, nothrow);
  case LibFunc_ZdlPvRKSt9nothrow_t:
  // void operator delete(void*, unsigned int);
  case LibFunc_ZdlPvj:
  // void operator delete(void*, unsigned long);
  case LibFunc_ZdlPvm:
  // void operator delete(void*, align_val_t)
  case LibFunc_ZdlPvSt11align_val_t:
  // void operator delete[](void*, align_val_t)
  case LibFunc_ZdaPvSt11align_val_t:
  // void operator delete[](void*, unsigned int);
  case LibFunc_msvc_delete_array_ptr32_int:
  // void operator delete[](void*, nothrow);
  case LibFunc_msvc_delete_array_ptr32_nothrow:
  // void operator delete[](void*, unsigned long long);
  case LibFunc_msvc_delete_array_ptr64_longlong:
  // void operator delete[](void*, nothrow);
  case LibFunc_msvc_delete_array_ptr64_nothrow:
  // void operator delete(void*, unsigned int);
  case LibFunc_msvc_delete_ptr32_int:
  // void operator delete(void*, nothrow);
  case LibFunc_msvc_delete_ptr32_nothrow:
  // void operator delete(void*, unsigned long long);
  case LibFunc_msvc_delete_ptr64_longlong:
  // void operator delete(void*, nothrow);
  case LibFunc_msvc_delete_ptr64_nothrow:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy());

  // void operator delete(void*, align_val_t, nothrow)
  case LibFunc_ZdlPvSt11align_val_tRKSt9nothrow_t:
  // void operator delete[](void*, align_val_t, nothrow)
  case LibFunc_ZdaPvSt11align_val_tRKSt9nothrow_t:
  // void operator delete(void*, unsigned int, align_val_t)
  case LibFunc_ZdlPvjSt11align_val_t:
  // void operator delete(void*, unsigned long, align_val_t)
  case LibFunc_ZdlPvmSt11align_val_t:
  // void operator delete[](void*, unsigned int, align_val_t);
  case LibFunc_ZdaPvjSt11align_val_t:
  // void operator delete[](void*, unsigned long, align_val_t);
  case LibFunc_ZdaPvmSt11align_val_t:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy());

  // void __atomic_load(size_t, void *, void *, int)
  case LibFunc_atomic_load:
  // void __atomic_store(size_t, void *, void *, int)
  case LibFunc_atomic_store:
    return (NumParams == 4 && FTy.getParamType(0)->isIntegerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy() &&
            FTy.getParamType(3)->isIntegerTy());

  case LibFunc_memset_pattern4:
  case LibFunc_memset_pattern8:
  case LibFunc_memset_pattern16:
    return (!FTy.isVarArg() && NumParams == 3 &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isIntegerTy());

  case LibFunc_cxa_guard_abort:
  case LibFunc_cxa_guard_acquire:
  case LibFunc_cxa_guard_release:
  case LibFunc_nvvm_reflect:
    return (NumParams == 1 && FTy.getParamType(0)->isPointerTy());

  case LibFunc_sincospi_stret:
  case LibFunc_sincospif_stret:
    return (NumParams == 1 && FTy.getParamType(0)->isFloatingPointTy());

  case LibFunc_acos:
  case LibFunc_acos_finite:
  case LibFunc_acosf:
  case LibFunc_acosf_finite:
  case LibFunc_acosh:
  case LibFunc_acosh_finite:
  case LibFunc_acoshf:
  case LibFunc_acoshf_finite:
  case LibFunc_acoshl:
  case LibFunc_acoshl_finite:
  case LibFunc_acosl:
  case LibFunc_acosl_finite:
  case LibFunc_asin:
  case LibFunc_asin_finite:
  case LibFunc_asinf:
  case LibFunc_asinf_finite:
  case LibFunc_asinh:
  case LibFunc_asinhf:
  case LibFunc_asinhl:
  case LibFunc_asinl:
  case LibFunc_asinl_finite:
  case LibFunc_atan:
  case LibFunc_atanf:
  case LibFunc_atanh:
  case LibFunc_atanh_finite:
  case LibFunc_atanhf:
  case LibFunc_atanhf_finite:
  case LibFunc_atanhl:
  case LibFunc_atanhl_finite:
  case LibFunc_atanl:
  case LibFunc_cbrt:
  case LibFunc_cbrtf:
  case LibFunc_cbrtl:
  case LibFunc_ceil:
  case LibFunc_ceilf:
  case LibFunc_ceill:
  case LibFunc_cos:
  case LibFunc_cosf:
  case LibFunc_cosh:
  case LibFunc_cosh_finite:
  case LibFunc_coshf:
  case LibFunc_coshf_finite:
  case LibFunc_coshl:
  case LibFunc_coshl_finite:
  case LibFunc_cosl:
  case LibFunc_exp10:
  case LibFunc_exp10_finite:
  case LibFunc_exp10f:
  case LibFunc_exp10f_finite:
  case LibFunc_exp10l:
  case LibFunc_exp10l_finite:
  case LibFunc_exp2:
  case LibFunc_exp2_finite:
  case LibFunc_exp2f:
  case LibFunc_exp2f_finite:
  case LibFunc_exp2l:
  case LibFunc_exp2l_finite:
  case LibFunc_exp:
  case LibFunc_exp_finite:
  case LibFunc_expf:
  case LibFunc_expf_finite:
  case LibFunc_expl:
  case LibFunc_expl_finite:
  case LibFunc_expm1:
  case LibFunc_expm1f:
  case LibFunc_expm1l:
  case LibFunc_fabs:
  case LibFunc_fabsf:
  case LibFunc_fabsl:
  case LibFunc_floor:
  case LibFunc_floorf:
  case LibFunc_floorl:
  case LibFunc_log10:
  case LibFunc_log10_finite:
  case LibFunc_log10f:
  case LibFunc_log10f_finite:
  case LibFunc_log10l:
  case LibFunc_log10l_finite:
  case LibFunc_log1p:
  case LibFunc_log1pf:
  case LibFunc_log1pl:
  case LibFunc_log2:
  case LibFunc_log2_finite:
  case LibFunc_log2f:
  case LibFunc_log2f_finite:
  case LibFunc_log2l:
  case LibFunc_log2l_finite:
  case LibFunc_log:
  case LibFunc_log_finite:
  case LibFunc_logb:
  case LibFunc_logbf:
  case LibFunc_logbl:
  case LibFunc_logf:
  case LibFunc_logf_finite:
  case LibFunc_logl:
  case LibFunc_logl_finite:
  case LibFunc_nearbyint:
  case LibFunc_nearbyintf:
  case LibFunc_nearbyintl:
  case LibFunc_rint:
  case LibFunc_rintf:
  case LibFunc_rintl:
  case LibFunc_round:
  case LibFunc_roundf:
  case LibFunc_roundl:
  case LibFunc_roundeven:
  case LibFunc_roundevenf:
  case LibFunc_roundevenl:
  case LibFunc_sin:
  case LibFunc_sinf:
  case LibFunc_sinh:
  case LibFunc_sinh_finite:
  case LibFunc_sinhf:
  case LibFunc_sinhf_finite:
  case LibFunc_sinhl:
  case LibFunc_sinhl_finite:
  case LibFunc_sinl:
  case LibFunc_sqrt:
  case LibFunc_sqrt_finite:
  case LibFunc_sqrtf:
  case LibFunc_sqrtf_finite:
  case LibFunc_sqrtl:
  case LibFunc_sqrtl_finite:
  case LibFunc_tan:
  case LibFunc_tanf:
  case LibFunc_tanh:
  case LibFunc_tanhf:
  case LibFunc_tanhl:
  case LibFunc_tanl:
  case LibFunc_trunc:
  case LibFunc_truncf:
  case LibFunc_truncl:
    return (NumParams == 1 && FTy.getReturnType()->isFloatingPointTy() &&
            FTy.getReturnType() == FTy.getParamType(0));

  case LibFunc_atan2:
  case LibFunc_atan2_finite:
  case LibFunc_atan2f:
  case LibFunc_atan2f_finite:
  case LibFunc_atan2l:
  case LibFunc_atan2l_finite:
  case LibFunc_fmin:
  case LibFunc_fminf:
  case LibFunc_fminl:
  case LibFunc_fmax:
  case LibFunc_fmaxf:
  case LibFunc_fmaxl:
  case LibFunc_fmod:
  case LibFunc_fmodf:
  case LibFunc_fmodl:
  case LibFunc_remainder:
  case LibFunc_remainderf:
  case LibFunc_remainderl:
  case LibFunc_copysign:
  case LibFunc_copysignf:
  case LibFunc_copysignl:
  case LibFunc_pow:
  case LibFunc_pow_finite:
  case LibFunc_powf:
  case LibFunc_powf_finite:
  case LibFunc_powl:
  case LibFunc_powl_finite:
    return (NumParams == 2 && FTy.getReturnType()->isFloatingPointTy() &&
            FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getReturnType() == FTy.getParamType(1));

  case LibFunc_ldexp:
  case LibFunc_ldexpf:
  case LibFunc_ldexpl:
    return (NumParams == 2 && FTy.getReturnType()->isFloatingPointTy() &&
            FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(1)->isIntegerTy(getIntSize()));

  case LibFunc_ffs:
  case LibFunc_ffsl:
  case LibFunc_ffsll:
  case LibFunc_fls:
  case LibFunc_flsl:
  case LibFunc_flsll:
    return (NumParams == 1 && FTy.getReturnType()->isIntegerTy(32) &&
            FTy.getParamType(0)->isIntegerTy());

  case LibFunc_isdigit:
  case LibFunc_isascii:
  case LibFunc_toascii:
  case LibFunc_putchar:
  case LibFunc_putchar_unlocked:
    return (NumParams == 1 && FTy.getReturnType()->isIntegerTy(32) &&
            FTy.getReturnType() == FTy.getParamType(0));

  case LibFunc_abs:
  case LibFunc_labs:
  case LibFunc_llabs:
    return (NumParams == 1 && FTy.getReturnType()->isIntegerTy() &&
            FTy.getReturnType() == FTy.getParamType(0));

  case LibFunc_cxa_atexit:
    return (NumParams == 3 && FTy.getReturnType()->isIntegerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());

  case LibFunc_sinpi:
  case LibFunc_cospi:
    return (NumParams == 1 && FTy.getReturnType()->isDoubleTy() &&
            FTy.getReturnType() == FTy.getParamType(0));

  case LibFunc_sinpif:
  case LibFunc_cospif:
    return (NumParams == 1 && FTy.getReturnType()->isFloatTy() &&
            FTy.getReturnType() == FTy.getParamType(0));

  case LibFunc_strnlen:
    return (NumParams == 2 && FTy.getReturnType() == FTy.getParamType(1) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isIntegerTy(SizeTBits));

  case LibFunc_posix_memalign:
    return (NumParams == 3 && FTy.getReturnType()->isIntegerTy(32) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isIntegerTy(SizeTBits) &&
            FTy.getParamType(2)->isIntegerTy(SizeTBits));

  case LibFunc_wcslen:
    return (NumParams == 1 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getReturnType()->isIntegerTy());

  case LibFunc_cabs:
  case LibFunc_cabsf:
  case LibFunc_cabsl: {
    Type* RetTy = FTy.getReturnType();
    if (!RetTy->isFloatingPointTy())
      return false;

    // NOTE: These prototypes are target specific and currently support
    // "complex" passed as an array or discrete real & imaginary parameters.
    // Add other calling conventions to enable libcall optimizations.
    if (NumParams == 1)
      return (FTy.getParamType(0)->isArrayTy() &&
              FTy.getParamType(0)->getArrayNumElements() == 2 &&
              FTy.getParamType(0)->getArrayElementType() == RetTy);
    else if (NumParams == 2)
      return (FTy.getParamType(0) == RetTy && FTy.getParamType(1) == RetTy);
    else
      return false;
  }
  case LibFunc::NumLibFuncs:
  case LibFunc::NotLibFunc:
    break;
  }

  llvm_unreachable("Invalid libfunc");
}

bool TargetLibraryInfoImpl::getLibFunc(const Function &FDecl,
                                       LibFunc &F) const {
  // Intrinsics don't overlap w/libcalls; if our module has a large number of
  // intrinsics, this ends up being an interesting compile time win since we
  // avoid string normalization and comparison.
  if (FDecl.isIntrinsic()) return false;

  const Module *M = FDecl.getParent();
  assert(M && "Expecting FDecl to be connected to a Module.");

  return getLibFunc(FDecl.getName(), F) &&
         isValidProtoForLibFunc(*FDecl.getFunctionType(), F, *M);
}

void TargetLibraryInfoImpl::disableAllFunctions() {
  memset(AvailableArray, 0, sizeof(AvailableArray));
}

static bool compareByScalarFnName(const VecDesc &LHS, const VecDesc &RHS) {
  return LHS.ScalarFnName < RHS.ScalarFnName;
}

static bool compareByVectorFnName(const VecDesc &LHS, const VecDesc &RHS) {
  return LHS.VectorFnName < RHS.VectorFnName;
}

static bool compareWithScalarFnName(const VecDesc &LHS, StringRef S) {
  return LHS.ScalarFnName < S;
}

void TargetLibraryInfoImpl::addVectorizableFunctions(ArrayRef<VecDesc> Fns) {
  llvm::append_range(VectorDescs, Fns);
  llvm::sort(VectorDescs, compareByScalarFnName);

  llvm::append_range(ScalarDescs, Fns);
  llvm::sort(ScalarDescs, compareByVectorFnName);
}

void TargetLibraryInfoImpl::addVectorizableFunctionsFromVecLib(
    enum VectorLibrary VecLib) {
  switch (VecLib) {
  case Accelerate: {
    const VecDesc VecFuncs[] = {
    #define TLI_DEFINE_ACCELERATE_VECFUNCS
    #include "llvm/Analysis/VecFuncs.def"
    };
    addVectorizableFunctions(VecFuncs);
    break;
  }
  case DarwinLibSystemM: {
    const VecDesc VecFuncs[] = {
    #define TLI_DEFINE_DARWIN_LIBSYSTEM_M_VECFUNCS
    #include "llvm/Analysis/VecFuncs.def"
    };
    addVectorizableFunctions(VecFuncs);
    break;
  }
  case LIBMVEC_X86: {
    const VecDesc VecFuncs[] = {
    #define TLI_DEFINE_LIBMVEC_X86_VECFUNCS
    #include "llvm/Analysis/VecFuncs.def"
    };
    addVectorizableFunctions(VecFuncs);
    break;
  }
  case MASSV: {
    const VecDesc VecFuncs[] = {
    #define TLI_DEFINE_MASSV_VECFUNCS
    #include "llvm/Analysis/VecFuncs.def"
    };
    addVectorizableFunctions(VecFuncs);
    break;
  }
  case SVML: {
    const VecDesc VecFuncs[] = {
    #define TLI_DEFINE_SVML_VECFUNCS
    #include "llvm/Analysis/VecFuncs.def"
    };
    addVectorizableFunctions(VecFuncs);
    break;
  }
  case NoLibrary:
    break;
  }
}

bool TargetLibraryInfoImpl::isFunctionVectorizable(StringRef funcName) const {
  funcName = sanitizeFunctionName(funcName);
  if (funcName.empty())
    return false;

  std::vector<VecDesc>::const_iterator I =
      llvm::lower_bound(VectorDescs, funcName, compareWithScalarFnName);
  return I != VectorDescs.end() && StringRef(I->ScalarFnName) == funcName;
}

StringRef
TargetLibraryInfoImpl::getVectorizedFunction(StringRef F,
                                             const ElementCount &VF) const {
  F = sanitizeFunctionName(F);
  if (F.empty())
    return F;
  std::vector<VecDesc>::const_iterator I =
      llvm::lower_bound(VectorDescs, F, compareWithScalarFnName);
  while (I != VectorDescs.end() && StringRef(I->ScalarFnName) == F) {
    if (I->VectorizationFactor == VF)
      return I->VectorFnName;
    ++I;
  }
  return StringRef();
}

TargetLibraryInfo TargetLibraryAnalysis::run(const Function &F,
                                             FunctionAnalysisManager &) {
  if (!BaselineInfoImpl)
    BaselineInfoImpl =
        TargetLibraryInfoImpl(Triple(F.getParent()->getTargetTriple()));
  return TargetLibraryInfo(*BaselineInfoImpl, &F);
}

unsigned TargetLibraryInfoImpl::getWCharSize(const Module &M) const {
  if (auto *ShortWChar = cast_or_null<ConstantAsMetadata>(
      M.getModuleFlag("wchar_size")))
    return cast<ConstantInt>(ShortWChar->getValue())->getZExtValue();
  return 0;
}

TargetLibraryInfoWrapperPass::TargetLibraryInfoWrapperPass()
    : ImmutablePass(ID), TLA(TargetLibraryInfoImpl()) {
  initializeTargetLibraryInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

TargetLibraryInfoWrapperPass::TargetLibraryInfoWrapperPass(const Triple &T)
    : ImmutablePass(ID), TLA(TargetLibraryInfoImpl(T)) {
  initializeTargetLibraryInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

TargetLibraryInfoWrapperPass::TargetLibraryInfoWrapperPass(
    const TargetLibraryInfoImpl &TLIImpl)
    : ImmutablePass(ID), TLA(TLIImpl) {
  initializeTargetLibraryInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

AnalysisKey TargetLibraryAnalysis::Key;

// Register the basic pass.
INITIALIZE_PASS(TargetLibraryInfoWrapperPass, "targetlibinfo",
                "Target Library Information", false, true)
char TargetLibraryInfoWrapperPass::ID = 0;

void TargetLibraryInfoWrapperPass::anchor() {}

void TargetLibraryInfoImpl::getWidestVF(StringRef ScalarF,
                                        ElementCount &FixedVF,
                                        ElementCount &ScalableVF) const {
  ScalarF = sanitizeFunctionName(ScalarF);
  // Use '0' here because a type of the form <vscale x 1 x ElTy> is not the
  // same as a scalar.
  ScalableVF = ElementCount::getScalable(0);
  FixedVF = ElementCount::getFixed(1);
  if (ScalarF.empty())
    return;

  std::vector<VecDesc>::const_iterator I =
      llvm::lower_bound(VectorDescs, ScalarF, compareWithScalarFnName);
  while (I != VectorDescs.end() && StringRef(I->ScalarFnName) == ScalarF) {
    ElementCount *VF =
        I->VectorizationFactor.isScalable() ? &ScalableVF : &FixedVF;
    if (ElementCount::isKnownGT(I->VectorizationFactor, *VF))
      *VF = I->VectorizationFactor;
    ++I;
  }
}
