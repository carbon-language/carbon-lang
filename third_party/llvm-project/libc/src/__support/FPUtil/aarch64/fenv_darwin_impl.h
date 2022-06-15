//===- darwin-aarch64 floating point env manipulation functions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FENV_DARWIN_IMPL_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FENV_DARWIN_IMPL_H

#include "src/__support/architectures.h"

#if !defined(LLVM_LIBC_ARCH_AARCH64) || !defined(__APPLE__)
#error "Invalid include"
#endif

#include <arm_acle.h>
#include <fenv.h>
#include <stdint.h>

#include "src/__support/FPUtil/FPBits.h"

namespace __llvm_libc {
namespace fputil {

struct FEnv {
  struct FPState {
    uint64_t StatusWord;
    uint64_t ControlWord;
  };

  static_assert(
      sizeof(fenv_t) == sizeof(FPState),
      "Internal floating point state does not match the public fenv_t type.");

  static constexpr uint32_t TONEAREST = 0x0;
  static constexpr uint32_t UPWARD = 0x1;
  static constexpr uint32_t DOWNWARD = 0x2;
  static constexpr uint32_t TOWARDZERO = 0x3;

  // These will be the exception flags we use for exception values normalized
  // from both status word and control word.
  // We add EX_ prefix to the names since macOS <math.h> defines OVERFLOW and
  // UNDERFLOW macros.
  static constexpr uint32_t EX_INVALID = 0x1;
  static constexpr uint32_t EX_DIVBYZERO = 0x2;
  static constexpr uint32_t EX_OVERFLOW = 0x4;
  static constexpr uint32_t EX_UNDERFLOW = 0x8;
  static constexpr uint32_t EX_INEXACT = 0x10;
  // __APPLE__ ARM64 has an extra flag that is raised when a denormal is flushed
  // to zero.
  static constexpr uint32_t EX_FLUSHTOZERO = 0x20;

  // Zero-th bit is the first bit.
  static constexpr uint32_t ROUNDING_CONTROL_BIT_POSITION = 22;

  // In addition to the 5 floating point exceptions, macOS on arm64 defines
  // another floating point exception: FE_FLUSHTOZERO, which is controlled by
  // __fpcr_flush_to_zero bit in the FPCR register.  This control bit is
  // located in a different place from FE_FLUSHTOZERO status bit relative to
  // the other exceptions.
  static inline uint32_t exception_value_from_status(int status) {
    return (status & FE_INVALID ? EX_INVALID : 0) |
           (status & FE_DIVBYZERO ? EX_DIVBYZERO : 0) |
           (status & FE_OVERFLOW ? EX_OVERFLOW : 0) |
           (status & FE_UNDERFLOW ? EX_UNDERFLOW : 0) |
           (status & FE_INEXACT ? EX_INEXACT : 0) |
           (status & FE_FLUSHTOZERO ? EX_FLUSHTOZERO : 0);
  }

  static inline uint32_t exception_value_from_control(int control) {
    return (control & __fpcr_trap_invalid ? EX_INVALID : 0) |
           (control & __fpcr_trap_divbyzero ? EX_DIVBYZERO : 0) |
           (control & __fpcr_trap_overflow ? EX_OVERFLOW : 0) |
           (control & __fpcr_trap_underflow ? EX_UNDERFLOW : 0) |
           (control & __fpcr_trap_inexact ? EX_INEXACT : 0) |
           (control & __fpcr_flush_to_zero ? EX_FLUSHTOZERO : 0);
  }

  static inline int exception_value_to_status(uint32_t excepts) {
    return (excepts & EX_INVALID ? FE_INVALID : 0) |
           (excepts & EX_DIVBYZERO ? FE_DIVBYZERO : 0) |
           (excepts & EX_OVERFLOW ? FE_OVERFLOW : 0) |
           (excepts & EX_UNDERFLOW ? FE_UNDERFLOW : 0) |
           (excepts & EX_INEXACT ? FE_INEXACT : 0) |
           (excepts & EX_FLUSHTOZERO ? FE_FLUSHTOZERO : 0);
  }

  static inline int exception_value_to_control(uint32_t excepts) {
    return (excepts & EX_INVALID ? __fpcr_trap_invalid : 0) |
           (excepts & EX_DIVBYZERO ? __fpcr_trap_divbyzero : 0) |
           (excepts & EX_OVERFLOW ? __fpcr_trap_overflow : 0) |
           (excepts & EX_UNDERFLOW ? __fpcr_trap_underflow : 0) |
           (excepts & EX_INEXACT ? __fpcr_trap_inexact : 0) |
           (excepts & EX_FLUSHTOZERO ? __fpcr_flush_to_zero : 0);
  }

  static uint32_t get_control_word() { return __arm_rsr("fpcr"); }

  static void set_control_word(uint32_t fpcr) { __arm_wsr("fpcr", fpcr); }

  static uint32_t get_status_word() { return __arm_rsr("fpsr"); }

  static void set_status_word(uint32_t fpsr) { __arm_wsr("fpsr", fpsr); }
};

static inline int enable_except(int excepts) {
  uint32_t new_excepts = FEnv::exception_value_from_status(excepts);
  uint32_t control_word = FEnv::get_control_word();
  uint32_t old_excepts = FEnv::exception_value_from_control(control_word);
  if (new_excepts != old_excepts) {
    control_word |= FEnv::exception_value_to_control(new_excepts);
    FEnv::set_control_word(control_word);
  }
  return FEnv::exception_value_to_status(old_excepts);
}

static inline int disable_except(int excepts) {
  uint32_t disabled_excepts = FEnv::exception_value_from_status(excepts);
  uint32_t control_word = FEnv::get_control_word();
  uint32_t old_excepts = FEnv::exception_value_from_control(control_word);
  control_word &= ~FEnv::exception_value_to_control(disabled_excepts);
  FEnv::set_control_word(control_word);
  return FEnv::exception_value_to_status(old_excepts);
}

static inline int get_except() {
  uint32_t control_word = FEnv::get_control_word();
  uint32_t enabled_excepts = FEnv::exception_value_from_control(control_word);
  return FEnv::exception_value_to_status(enabled_excepts);
}

static inline int clear_except(int excepts) {
  uint32_t status_word = FEnv::get_status_word();
  uint32_t except_value = FEnv::exception_value_from_status(excepts);
  status_word &= ~FEnv::exception_value_to_status(except_value);
  FEnv::set_status_word(status_word);
  return 0;
}

static inline int test_except(int excepts) {
  uint32_t statusWord = FEnv::get_status_word();
  uint32_t ex_value = FEnv::exception_value_from_status(excepts);
  return statusWord & FEnv::exception_value_to_status(ex_value);
}

static inline int set_except(int excepts) {
  uint32_t status_word = FEnv::get_status_word();
  uint32_t new_exceptions = FEnv::exception_value_from_status(excepts);
  status_word |= FEnv::exception_value_to_status(new_exceptions);
  FEnv::set_status_word(status_word);
  return 0;
}

static inline int raise_except(int excepts) {
  float zero = 0.0f;
  float one = 1.0f;
  float large_value = float(FPBits<float>(FPBits<float>::MAX_NORMAL));
  float small_value = float(FPBits<float>(FPBits<float>::MIN_NORMAL));
  auto divfunc = [](float a, float b) {
    __asm__ __volatile__("ldr  s0, %0\n\t"
                         "ldr  s1, %1\n\t"
                         "fdiv s0, s0, s1\n\t"
                         : // No outputs
                         : "m"(a), "m"(b)
                         : "s0", "s1" /* s0 and s1 are clobbered */);
  };

  uint32_t to_raise = FEnv::exception_value_from_status(excepts);
  int result = 0;

  if (to_raise & FEnv::EX_INVALID) {
    divfunc(zero, zero);
    uint32_t status_word = FEnv::get_status_word();
    if (!(FEnv::exception_value_from_status(status_word) & FEnv::EX_INVALID))
      result = -1;
  }

  if (to_raise & FEnv::EX_DIVBYZERO) {
    divfunc(one, zero);
    uint32_t status_word = FEnv::get_status_word();
    if (!(FEnv::exception_value_from_status(status_word) & FEnv::EX_DIVBYZERO))
      result = -1;
  }
  if (to_raise & FEnv::EX_OVERFLOW) {
    divfunc(large_value, small_value);
    uint32_t status_word = FEnv::get_status_word();
    if (!(FEnv::exception_value_from_status(status_word) & FEnv::EX_OVERFLOW))
      result = -1;
  }
  if (to_raise & FEnv::EX_UNDERFLOW) {
    divfunc(small_value, large_value);
    uint32_t status_word = FEnv::get_status_word();
    if (!(FEnv::exception_value_from_status(status_word) & FEnv::EX_UNDERFLOW))
      result = -1;
  }
  if (to_raise & FEnv::EX_INEXACT) {
    float two = 2.0f;
    float three = 3.0f;
    // 2.0 / 3.0 cannot be represented exactly in any radix 2 floating point
    // format.
    divfunc(two, three);
    uint32_t status_word = FEnv::get_status_word();
    if (!(FEnv::exception_value_from_status(status_word) & FEnv::EX_INEXACT))
      result = -1;
  }
  if (to_raise & FEnv::EX_FLUSHTOZERO) {
    // TODO: raise the flush to zero floating point exception.
    result = -1;
  }
  return result;
}

static inline int get_round() {
  uint32_t rounding_mode =
      (FEnv::get_control_word() >> FEnv::ROUNDING_CONTROL_BIT_POSITION) & 0x3;
  switch (rounding_mode) {
  case FEnv::TONEAREST:
    return FE_TONEAREST;
  case FEnv::DOWNWARD:
    return FE_DOWNWARD;
  case FEnv::UPWARD:
    return FE_UPWARD;
  case FEnv::TOWARDZERO:
    return FE_TOWARDZERO;
  default:
    return -1; // Error value.
  }
}

static inline int set_round(int mode) {
  uint16_t bit_value;
  switch (mode) {
  case FE_TONEAREST:
    bit_value = FEnv::TONEAREST;
    break;
  case FE_DOWNWARD:
    bit_value = FEnv::DOWNWARD;
    break;
  case FE_UPWARD:
    bit_value = FEnv::UPWARD;
    break;
  case FE_TOWARDZERO:
    bit_value = FEnv::TOWARDZERO;
    break;
  default:
    return 1; // To indicate failure
  }

  uint32_t control_word = FEnv::get_control_word();
  control_word &= ~(0x3 << FEnv::ROUNDING_CONTROL_BIT_POSITION);
  control_word |= (bit_value << FEnv::ROUNDING_CONTROL_BIT_POSITION);
  FEnv::set_control_word(control_word);

  return 0;
}

static inline int get_env(fenv_t *envp) {
  FEnv::FPState *state = reinterpret_cast<FEnv::FPState *>(envp);
  state->ControlWord = FEnv::get_control_word();
  state->StatusWord = FEnv::get_status_word();
  return 0;
}

static inline int set_env(const fenv_t *envp) {
  if (envp == FE_DFL_ENV) {
    // Default status and control words bits are all zeros so we just
    // write zeros.
    FEnv::set_status_word(0);
    FEnv::set_control_word(0);
    return 0;
  }
  const FEnv::FPState *state = reinterpret_cast<const FEnv::FPState *>(envp);
  FEnv::set_control_word(state->ControlWord);
  FEnv::set_status_word(state->StatusWord);
  return 0;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_FENV_DARWIN_IMPL_H
