//===-- dfsan.cc ----------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DataFlowSanitizer.
//
// This file defines the custom functions listed in done_abilist.txt.
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_linux.h"

#include "dfsan/dfsan.h"

#include <ctype.h>
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

using namespace __dfsan;

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE int
__dfsw_stat(const char *path, struct stat *buf, dfsan_label path_label,
            dfsan_label buf_label, dfsan_label *ret_label) {
  int ret = stat(path, buf);
  if (ret == 0)
    dfsan_set_label(0, buf, sizeof(struct stat));
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_fstat(int fd, struct stat *buf,
                                               dfsan_label fd_label,
                                               dfsan_label buf_label,
                                               dfsan_label *ret_label) {
  int ret = fstat(fd, buf);
  if (ret == 0)
    dfsan_set_label(0, buf, sizeof(struct stat));
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE char *__dfsw_strchr(const char *s, int c,
                                                  dfsan_label s_label,
                                                  dfsan_label c_label,
                                                  dfsan_label *ret_label) {
  for (size_t i = 0;; ++i) {
    if (s[i] == c || s[i] == 0) {
      *ret_label = dfsan_union(dfsan_read_label(s, i+1), c_label);
      return s[i] == 0 ? 0 : const_cast<char *>(s+i);
    }
  }
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_memcmp(const void *s1, const void *s2,
                                                size_t n, dfsan_label s1_label,
                                                dfsan_label s2_label,
                                                dfsan_label n_label,
                                                dfsan_label *ret_label) {
  const char *cs1 = (const char *) s1, *cs2 = (const char *) s2;
  for (size_t i = 0; i != n; ++i) {
    if (cs1[i] != cs2[i]) {
      *ret_label = dfsan_union(dfsan_read_label(cs1, i+1),
                               dfsan_read_label(cs2, i+1));
      return cs1[i] - cs2[i];
    }
  }
  *ret_label = dfsan_union(dfsan_read_label(cs1, n),
                           dfsan_read_label(cs2, n));
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_strcmp(const char *s1, const char *s2,
                                                dfsan_label s1_label,
                                                dfsan_label s2_label,
                                                dfsan_label *ret_label) {
  for (size_t i = 0;; ++i) {
    if (s1[i] != s2[i] || s1[i] == 0 || s2[i] == 0) {
      *ret_label = dfsan_union(dfsan_read_label(s1, i+1),
                               dfsan_read_label(s2, i+1));
      return s1[i] - s2[i];
    }
  }
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE int
__dfsw_strcasecmp(const char *s1, const char *s2, dfsan_label s1_label,
                  dfsan_label s2_label, dfsan_label *ret_label) {
  for (size_t i = 0;; ++i) {
    if (tolower(s1[i]) != tolower(s2[i]) || s1[i] == 0 || s2[i] == 0) {
      *ret_label = dfsan_union(dfsan_read_label(s1, i+1),
                               dfsan_read_label(s2, i+1));
      return s1[i] - s2[i];
    }
  }
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_strncmp(const char *s1, const char *s2,
                                                 size_t n, dfsan_label s1_label,
                                                 dfsan_label s2_label,
                                                 dfsan_label n_label,
                                                 dfsan_label *ret_label) {
  if (n == 0) {
    *ret_label = 0;
    return 0;
  }

  for (size_t i = 0;; ++i) {
    if (s1[i] != s2[i] || s1[i] == 0 || s2[i] == 0 || i == n-1) {
      *ret_label = dfsan_union(dfsan_read_label(s1, i+1),
                               dfsan_read_label(s2, i+1));
      return s1[i] - s2[i];
    }
  }
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE int
__dfsw_strncasecmp(const char *s1, const char *s2, size_t n,
                   dfsan_label s1_label, dfsan_label s2_label,
                   dfsan_label n_label, dfsan_label *ret_label) {
  if (n == 0) {
    *ret_label = 0;
    return 0;
  }

  for (size_t i = 0;; ++i) {
    if (tolower(s1[i]) != tolower(s2[i]) || s1[i] == 0 || s2[i] == 0 ||
        i == n - 1) {
      *ret_label = dfsan_union(dfsan_read_label(s1, i+1),
                               dfsan_read_label(s2, i+1));
      return s1[i] - s2[i];
    }
  }
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE void *__dfsw_calloc(size_t nmemb, size_t size,
                                                  dfsan_label nmemb_label,
                                                  dfsan_label size_label,
                                                  dfsan_label *ret_label) {
  void *p = calloc(nmemb, size);
  dfsan_set_label(0, p, nmemb * size);
  *ret_label = 0;
  return p;
}

SANITIZER_INTERFACE_ATTRIBUTE size_t
__dfsw_strlen(const char *s, dfsan_label s_label, dfsan_label *ret_label) {
  size_t ret = strlen(s);
  *ret_label = dfsan_read_label(s, ret+1);
  return ret;
}


static void *dfsan_memcpy(void *dest, const void *src, size_t n) {
  dfsan_label *sdest = shadow_for(dest), *ssrc = shadow_for((void *)src);
  internal_memcpy((void *)sdest, (void *)ssrc, n * sizeof(dfsan_label));
  return internal_memcpy(dest, src, n);
}

static void dfsan_memset(void *s, int c, dfsan_label c_label, size_t n) {
  internal_memset(s, c, n);
  dfsan_set_label(c_label, s, n);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *__dfsw_memcpy(void *dest, const void *src, size_t n,
                    dfsan_label dest_label, dfsan_label src_label,
                    dfsan_label n_label, dfsan_label *ret_label) {
  *ret_label = 0;
  return dfsan_memcpy(dest, src, n);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *__dfsw_memset(void *s, int c, size_t n,
                    dfsan_label s_label, dfsan_label c_label,
                    dfsan_label n_label, dfsan_label *ret_label) {
  dfsan_memset(s, c, c_label, n);
  *ret_label = 0;
  return s;
}

SANITIZER_INTERFACE_ATTRIBUTE char *
__dfsw_strdup(const char *s, dfsan_label s_label, dfsan_label *ret_label) {
  size_t len = strlen(s);
  void *p = malloc(len+1);
  dfsan_memcpy(p, s, len+1);
  *ret_label = 0;
  return static_cast<char *>(p);
}

SANITIZER_INTERFACE_ATTRIBUTE char *
__dfsw_strncpy(char *s1, const char *s2, size_t n, dfsan_label s1_label,
               dfsan_label s2_label, dfsan_label n_label,
               dfsan_label *ret_label) {
  size_t len = strlen(s2);
  if (len < n) {
    dfsan_memcpy(s1, s2, len+1);
    dfsan_memset(s1+len+1, 0, 0, n-len-1);
  } else {
    dfsan_memcpy(s1, s2, n);
  }

  *ret_label = 0;
  return s1;
}

SANITIZER_INTERFACE_ATTRIBUTE ssize_t
__dfsw_pread(int fd, void *buf, size_t count, off_t offset,
             dfsan_label fd_label, dfsan_label buf_label,
             dfsan_label count_label, dfsan_label offset_label,
             dfsan_label *ret_label) {
  ssize_t ret = pread(fd, buf, count, offset);
  if (ret > 0)
    dfsan_set_label(0, buf, ret);
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE ssize_t
__dfsw_read(int fd, void *buf, size_t count,
             dfsan_label fd_label, dfsan_label buf_label,
             dfsan_label count_label,
             dfsan_label *ret_label) {
  ssize_t ret = read(fd, buf, count);
  if (ret > 0)
    dfsan_set_label(0, buf, ret);
  *ret_label = 0;
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_clock_gettime(clockid_t clk_id,
                                                       struct timespec *tp,
                                                       dfsan_label clk_id_label,
                                                       dfsan_label tp_label,
                                                       dfsan_label *ret_label) {
  int ret = clock_gettime(clk_id, tp);
  if (ret == 0)
    dfsan_set_label(0, tp, sizeof(struct timespec));
  *ret_label = 0;
  return ret;
}

static void unpoison(const void *ptr, uptr size) {
  dfsan_set_label(0, const_cast<void *>(ptr), size);
}

// dlopen() ultimately calls mmap() down inside the loader, which generally
// doesn't participate in dynamic symbol resolution.  Therefore we won't
// intercept its calls to mmap, and we have to hook it here.
SANITIZER_INTERFACE_ATTRIBUTE void *
__dfsw_dlopen(const char *filename, int flag, dfsan_label filename_label,
              dfsan_label flag_label, dfsan_label *ret_label) {
  link_map *map = (link_map *)dlopen(filename, flag);
  if (map)
    ForEachMappedRegion(map, unpoison);
  *ret_label = 0;
  return (void *)map;
}

struct pthread_create_info {
  void *(*start_routine_trampoline)(void *, void *, dfsan_label, dfsan_label *);
  void *start_routine;
  void *arg;
};

static void *pthread_create_cb(void *p) {
  pthread_create_info pci(*(pthread_create_info *)p);
  free(p);
  dfsan_label ret_label;
  return pci.start_routine_trampoline(pci.start_routine, pci.arg, 0,
                                      &ret_label);
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_pthread_create(
    pthread_t *thread, const pthread_attr_t *attr,
    void *(*start_routine_trampoline)(void *, void *, dfsan_label,
                                      dfsan_label *),
    void *start_routine, void *arg, dfsan_label thread_label,
    dfsan_label attr_label, dfsan_label start_routine_label,
    dfsan_label arg_label, dfsan_label *ret_label) {
  pthread_create_info *pci =
      (pthread_create_info *)malloc(sizeof(pthread_create_info));
  pci->start_routine_trampoline = start_routine_trampoline;
  pci->start_routine = start_routine;
  pci->arg = arg;
  int rv = pthread_create(thread, attr, pthread_create_cb, (void *)pci);
  if (rv != 0)
    free(pci);
  *ret_label = 0;
  return rv;
}

struct dl_iterate_phdr_info {
  int (*callback_trampoline)(void *callback, struct dl_phdr_info *info,
                             size_t size, void *data, dfsan_label info_label,
                             dfsan_label size_label, dfsan_label data_label,
                             dfsan_label *ret_label);
  void *callback;
  void *data;
};

int dl_iterate_phdr_cb(struct dl_phdr_info *info, size_t size, void *data) {
  dl_iterate_phdr_info *dipi = (dl_iterate_phdr_info *)data;
  dfsan_set_label(0, *info);
  dfsan_set_label(0, (void *)info->dlpi_name, strlen(info->dlpi_name) + 1);
  dfsan_set_label(0, (void *)info->dlpi_phdr,
                  sizeof(*info->dlpi_phdr) * info->dlpi_phnum);
  dfsan_label ret_label;
  return dipi->callback_trampoline(dipi->callback, info, size, dipi->data, 0, 0,
                                   0, &ret_label);
}

SANITIZER_INTERFACE_ATTRIBUTE int __dfsw_dl_iterate_phdr(
    int (*callback_trampoline)(void *callback, struct dl_phdr_info *info,
                               size_t size, void *data, dfsan_label info_label,
                               dfsan_label size_label, dfsan_label data_label,
                               dfsan_label *ret_label),
    void *callback, void *data, dfsan_label callback_label,
    dfsan_label data_label, dfsan_label *ret_label) {
  dl_iterate_phdr_info dipi = { callback_trampoline, callback, data };
  *ret_label = 0;
  return dl_iterate_phdr(dl_iterate_phdr_cb, &dipi);
}

}
