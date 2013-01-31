//===-- test.c ------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Sanity test for Go runtime.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

void __tsan_init(void **thr);
void __tsan_fini();
void __tsan_map_shadow(void *addr, unsigned long size);
void __tsan_go_start(void *thr, void **chthr, void *pc);
void __tsan_go_end(void *thr);
void __tsan_read(void *thr, void *addr, void *pc);
void __tsan_write(void *thr, void *addr, void *pc);
void __tsan_func_enter(void *thr, void *pc);
void __tsan_func_exit(void *thr);
void __tsan_malloc(void *thr, void *p, unsigned long sz, void *pc);
void __tsan_free(void *p);
void __tsan_acquire(void *thr, void *addr);
void __tsan_release(void *thr, void *addr);
void __tsan_release_merge(void *thr, void *addr);

int __tsan_symbolize(void *pc, char **img, char **rtn, char **file, int *l) {
  return 0;
}

char buf[10];

int main(void) {
  void *thr0 = 0;
  __tsan_init(&thr0);
  __tsan_map_shadow(buf, sizeof(buf) + 4096);
  __tsan_func_enter(thr0, &main);
  __tsan_malloc(thr0, buf, 10, 0);
  __tsan_release(thr0, buf);
  __tsan_release_merge(thr0, buf);
  void *thr1 = 0;
  __tsan_go_start(thr0, &thr1, 0);
  __tsan_write(thr1, buf, 0);
  __tsan_acquire(thr1, buf);
  __tsan_go_end(thr1);
  __tsan_read(thr0, buf, 0);
  __tsan_free(buf);
  __tsan_func_exit(thr0);
  __tsan_fini();
  return 0;
}
