// RUN: %clangxx_msan -m64 -O0 %s -o %t && %t 2>&1
// RUN: %clangxx_msan -m64 -O3 %s -o %t && %t 2>&1

#include <assert.h>
#include <errno.h>
#include <glob.h>
#include <stdio.h>
#include <string.h>

#include <sanitizer/linux_syscall_hooks.h>
#include <sanitizer/msan_interface.h>

/* Test the presence of __sanitizer_syscall_ in the tool runtime, and general
   sanity of their behaviour. */

int main(int argc, char *argv[]) {
  char buf[1000];
  const int kTen = 10;
  memset(buf, 0, sizeof(buf));
  __msan_unpoison(buf, sizeof(buf));
  __sanitizer_syscall_pre_recvmsg(0, buf, 0);
  __sanitizer_syscall_pre_rt_sigpending(buf, kTen);
  __sanitizer_syscall_pre_getdents(0, buf, kTen);
  __sanitizer_syscall_pre_getdents64(0, buf, kTen);

  __msan_unpoison(buf, sizeof(buf));
  __sanitizer_syscall_post_recvmsg(0, 0, buf, 0);
  __sanitizer_syscall_post_rt_sigpending(-1, buf, kTen);
  __sanitizer_syscall_post_getdents(0, 0, buf, kTen);
  __sanitizer_syscall_post_getdents64(0, 0, buf, kTen);
  assert(__msan_test_shadow(buf, sizeof(buf)) == -1);

  __msan_unpoison(buf, sizeof(buf));
  __sanitizer_syscall_post_recvmsg(kTen, 0, buf, 0);

  // Tell the kernel that the output struct size is 10 bytes, verify that those
  // bytes are unpoisoned, and the next byte is not.
  __msan_poison(buf, kTen + 1);
  __sanitizer_syscall_post_rt_sigpending(0, buf, kTen);
  assert(__msan_test_shadow(buf, sizeof(buf)) == kTen);

  __msan_poison(buf, kTen + 1);
  __sanitizer_syscall_post_getdents(kTen, 0, buf, kTen);
  assert(__msan_test_shadow(buf, sizeof(buf)) == kTen);

  __msan_poison(buf, kTen + 1);
  __sanitizer_syscall_post_getdents64(kTen, 0, buf, kTen);
  assert(__msan_test_shadow(buf, sizeof(buf)) == kTen);
  return 0;
}
