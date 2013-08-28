// RUN: %clang_dfsan -m64 %s -o %t && %t
// RUN: %clang_dfsan -mllvm -dfsan-args-abi -m64 %s -o %t && %t

// Tests custom implementations of various libc functions.

#define _GNU_SOURCE
#include <sanitizer/dfsan_interface.h>
#include <assert.h>
#include <link.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

void *ptcb(void *p) {
  assert(p == (void *)1);
  assert(dfsan_get_label((uintptr_t)p) == 0);
  return (void *)2;
}

int dlcb(struct dl_phdr_info *info, size_t size, void *data) {
  assert(data == (void *)3);
  assert(dfsan_get_label((uintptr_t)info) == 0);
  assert(dfsan_get_label(size) == 0);
  assert(dfsan_get_label((uintptr_t)data) == 0);
  return 0;
}

int main(void) {
  int i = 1;
  dfsan_label i_label = dfsan_create_label("i", 0);
  dfsan_set_label(i_label, &i, sizeof(i));

  int j = 2;
  dfsan_label j_label = dfsan_create_label("j", 0);
  dfsan_set_label(j_label, &j, sizeof(j));

  struct stat s;
  s.st_dev = i;
  int rv = stat("/", &s);
  assert(rv == 0);
  assert(dfsan_get_label(s.st_dev) == 0);

  s.st_dev = i;
  rv = stat("/nonexistent", &s);
  assert(rv == -1);
  assert(dfsan_get_label(s.st_dev) == i_label);

  int fd = open("/dev/zero", O_RDONLY);
  s.st_dev = i;
  rv = fstat(fd, &s);
  assert(rv == 0);
  assert(dfsan_get_label(s.st_dev) == 0);

  char str1[] = "str1", str2[] = "str2";
  dfsan_set_label(i_label, &str1[3], 1);
  dfsan_set_label(j_label, &str2[3], 1);

  rv = memcmp(str1, str2, sizeof(str1));
  assert(rv < 0);
  assert(dfsan_get_label(rv) == dfsan_union(i_label, j_label));

  char strc[sizeof(str1)];
  memcpy(strc, str1, sizeof(str1));
  assert(dfsan_get_label(strc[0]) == 0);
  assert(dfsan_get_label(strc[3]) == i_label);

  memset(strc, j, sizeof(strc));
  assert(dfsan_get_label(strc[0]) == j_label);
  assert(dfsan_get_label(strc[1]) == j_label);
  assert(dfsan_get_label(strc[2]) == j_label);
  assert(dfsan_get_label(strc[3]) == j_label);
  assert(dfsan_get_label(strc[4]) == j_label);

  rv = strcmp(str1, str2);
  assert(rv < 0);
  assert(dfsan_get_label(rv) == dfsan_union(i_label, j_label));

  char *strd = strdup(str1);
  assert(dfsan_get_label(strd[0]) == 0);
  assert(dfsan_get_label(strd[3]) == i_label);
  free(strd);

  rv = strncmp(str1, str2, sizeof(str1));
  assert(rv < 0);
  assert(dfsan_get_label(rv) == dfsan_union(i_label, j_label));

  rv = strncmp(str1, str2, 3);
  assert(rv == 0);
  assert(dfsan_get_label(rv) == 0);

  str1[0] = 'S';

  rv = strncasecmp(str1, str2, sizeof(str1));
  assert(rv < 0);
  assert(dfsan_get_label(rv) == dfsan_union(i_label, j_label));

  rv = strncasecmp(str1, str2, 3);
  assert(rv == 0);
  assert(dfsan_get_label(rv) == 0);

  char *crv = strchr(str1, 'r');
  assert(crv == &str1[2]);
  assert(dfsan_get_label((uintptr_t)crv) == 0);

  crv = strchr(str1, '1');
  assert(crv == &str1[3]);
  assert(dfsan_get_label((uintptr_t)crv) == i_label);

  crv = strchr(str1, 'x');
  assert(crv == 0);
  assert(dfsan_get_label((uintptr_t)crv) == i_label);

  // With any luck this sequence of calls will cause calloc to return the same
  // pointer both times.  This is probably the best we can do to test this
  // function.
  crv = calloc(4096, 1);
  assert(dfsan_get_label(crv[0]) == 0);
  free(crv);

  crv = calloc(4096, 1);
  assert(dfsan_get_label(crv[0]) == 0);
  free(crv);

  char buf[16];
  buf[0] = i;
  buf[15] = j;
  rv = read(fd, buf, sizeof(buf));
  assert(rv == sizeof(buf));
  assert(dfsan_get_label(buf[0]) == 0);
  assert(dfsan_get_label(buf[15]) == 0);

  close(fd);
  fd = open("/bin/sh", O_RDONLY);
  buf[0] = i;
  buf[15] = j;
  rv = pread(fd, buf, sizeof(buf), 0);
  assert(rv == sizeof(buf));
  assert(dfsan_get_label(buf[0]) == 0);
  assert(dfsan_get_label(buf[15]) == 0);

  pthread_t pt;
  pthread_create(&pt, 0, ptcb, (void *)1);
  void *cbrv;
  pthread_join(pt, &cbrv);
  assert(cbrv == (void *)2);

  dl_iterate_phdr(dlcb, (void *)3);

  return 0;
}
