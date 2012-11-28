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

void __tsan_init();
void __tsan_fini();
void __tsan_map_shadow(void *addr, unsigned long size);
void __tsan_go_start(int pgoid, int chgoid, void *pc);
void __tsan_go_end(int goid);
void __tsan_read(int goid, void *addr, void *pc);
void __tsan_write(int goid, void *addr, void *pc);
void __tsan_func_enter(int goid, void *pc);
void __tsan_func_exit(int goid);
void __tsan_malloc(int goid, void *p, unsigned long sz, void *pc);
void __tsan_free(void *p);
void __tsan_acquire(int goid, void *addr);
void __tsan_release(int goid, void *addr);
void __tsan_release_merge(int goid, void *addr);

int __tsan_symbolize(void *pc, char **img, char **rtn, char **file, int *l) {
  return 0;
}

char buf[10];

int main(void) {
  __tsan_init();
  __tsan_map_shadow(buf, sizeof(buf) + 4096);
  __tsan_func_enter(0, &main);
  __tsan_malloc(0, buf, 10, 0);
  __tsan_release(0, buf);
  __tsan_release_merge(0, buf);
  __tsan_go_start(0, 1, 0);
  __tsan_write(1, buf, 0);
  __tsan_acquire(1, buf);
  __tsan_go_end(1);
  __tsan_read(0, buf, 0);
  __tsan_free(buf);
  __tsan_func_exit(0);
  __tsan_fini();
  return 0;
}
