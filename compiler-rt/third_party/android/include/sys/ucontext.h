// Copyright (c) 2012, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_SYS_UCONTEXT_H
#define GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_SYS_UCONTEXT_H

#include <sys/cdefs.h>
#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifndef __BIONIC_HAVE_UCONTEXT_T

// Ensure that 'stack_t' is defined.
#include <asm/signal.h>

// This version of the Android C library headers do not provide ucontext_t.
// Provide custom definitions for Google Breakpad.
#if defined(__arm__)

// Ensure that 'struct sigcontext' is defined.
#include <asm/sigcontext.h>
typedef struct sigcontext mcontext_t;

// The ARM kernel uses a 64-bit signal mask.
typedef uint32_t  kernel_sigmask_t[2];

typedef struct ucontext {
  uint32_t uc_flags;
  struct ucontext* uc_link;
  stack_t uc_stack;
  mcontext_t uc_mcontext;
  kernel_sigmask_t uc_sigmask;
  // Other fields are not used by Google Breakpad. Don't define them.
} ucontext_t;

#elif defined(__i386__)

/* 80-bit floating-point register */
struct _libc_fpreg {
  unsigned short significand[4];
  unsigned short exponent;
};

/* Simple floating-point state, see FNSTENV instruction */
struct _libc_fpstate {
  unsigned long cw;
  unsigned long sw;
  unsigned long tag;
  unsigned long ipoff;
  unsigned long cssel;
  unsigned long dataoff;
  unsigned long datasel;
  struct _libc_fpreg _st[8];
  unsigned long status;
};

typedef uint32_t  greg_t;

typedef struct {
  uint32_t gregs[19];
  struct _libc_fpstate* fpregs;
  uint32_t oldmask;
  uint32_t cr2;
} mcontext_t;

enum {
  REG_GS = 0,
  REG_FS,
  REG_ES,
  REG_DS,
  REG_EDI,
  REG_ESI,
  REG_EBP,
  REG_ESP,
  REG_EBX,
  REG_EDX,
  REG_ECX,
  REG_EAX,
  REG_TRAPNO,
  REG_ERR,
  REG_EIP,
  REG_CS,
  REG_EFL,
  REG_UESP,
  REG_SS,
};

// The i386 kernel uses a 64-bit signal mask.
typedef uint32_t kernel_sigmask_t[2];

typedef struct ucontext {
  uint32_t uc_flags;
  struct ucontext* uc_link;
  stack_t uc_stack;
  mcontext_t uc_mcontext;
  kernel_sigmask_t uc_sigmask;
  struct _libc_fpstate __fpregs_mem;
} ucontext_t;

#elif defined(__mips__)

typedef struct {
  uint32_t regmask;
  uint32_t status;
  uint64_t pc;
  uint64_t gregs[32];
  uint64_t fpregs[32];
  uint32_t acx;
  uint32_t fpc_csr;
  uint32_t fpc_eir;
  uint32_t used_math;
  uint32_t dsp;
  uint64_t mdhi;
  uint64_t mdlo;
  uint32_t hi1;
  uint32_t lo1;
  uint32_t hi2;
  uint32_t lo2;
  uint32_t hi3;
  uint32_t lo3;
} mcontext_t;

// The MIPS kernel uses a 128-bit signal mask.
typedef uint32_t kernel_sigmask_t[4];

typedef struct ucontext {
  uint32_t uc_flags;
  struct ucontext* uc_link;
  stack_t uc_stack;
  mcontext_t uc_mcontext;
  kernel_sigmask_t uc_sigmask;
  // Other fields are not used by Google Breakpad. Don't define them.
} ucontext_t;

#else
#  error "Unsupported Android CPU ABI!"
#endif

#endif  // __BIONIC_HAVE_UCONTEXT_T

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_SYS_UCONTEXT_H
