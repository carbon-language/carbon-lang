#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

extern "C" {
typedef unsigned long jptr;  // NOLINT
void __tsan_java_init(jptr heap_begin, jptr heap_size);
int  __tsan_java_fini();
void __tsan_java_alloc(jptr ptr, jptr size);
void __tsan_java_free(jptr ptr, jptr size);
void __tsan_java_move(jptr src, jptr dst, jptr size);
void __tsan_java_mutex_lock(jptr addr);
void __tsan_java_mutex_unlock(jptr addr);
void __tsan_java_mutex_read_lock(jptr addr);
void __tsan_java_mutex_read_unlock(jptr addr);
void __tsan_java_mutex_lock_rec(jptr addr, int rec);
int  __tsan_java_mutex_unlock_rec(jptr addr);
}
