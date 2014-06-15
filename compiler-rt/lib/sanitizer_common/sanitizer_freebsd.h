//===-- sanitizer_freebsd.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer runtime. It contains FreeBSD-specific
// definitions.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_FREEBSD_H
#define SANITIZER_FREEBSD_H

#include "sanitizer_internal_defs.h"

// x86-64 FreeBSD 9.2 and older define 'ucontext_t' incorrectly in
// 32-bit mode.
#if SANITIZER_FREEBSD && (SANITIZER_WORDSIZE == 32)
# include <osreldate.h>
# if __FreeBSD_version <= 902001  // v9.2
#  include <ucontext.h>

namespace __sanitizer {

typedef __int32_t __xregister_t;

typedef struct __xmcontext {
  __xregister_t mc_onstack;
  __xregister_t mc_gs;
  __xregister_t mc_fs;
  __xregister_t mc_es;
  __xregister_t mc_ds;
  __xregister_t mc_edi;
  __xregister_t mc_esi;
  __xregister_t mc_ebp;
  __xregister_t mc_isp;
  __xregister_t mc_ebx;
  __xregister_t mc_edx;
  __xregister_t mc_ecx;
  __xregister_t mc_eax;
  __xregister_t mc_trapno;
  __xregister_t mc_err;
  __xregister_t mc_eip;
  __xregister_t mc_cs;
  __xregister_t mc_eflags;
  __xregister_t mc_esp;
  __xregister_t mc_ss;

  int mc_len;
  int mc_fpformat;
  int mc_ownedfp;
  __xregister_t mc_flags;

  int mc_fpstate[128] __aligned(16);
  __xregister_t mc_fsbase;
  __xregister_t mc_gsbase;
  __xregister_t mc_xfpustate;
  __xregister_t mc_xfpustate_len;

  int mc_spare2[4];
} xmcontext_t;

typedef struct __xucontext {
  sigset_t  uc_sigmask;
  xmcontext_t  uc_mcontext;

  struct __ucontext *uc_link;
  stack_t uc_stack;
  int uc_flags;
  int __spare__[4];
} xucontext_t;

}  // namespace __sanitizer

# endif  // __FreeBSD_version <= 902001
#endif  // SANITIZER_FREEBSD && (SANITIZER_WORDSIZE == 32)

#endif  // SANITIZER_FREEBSD_H
