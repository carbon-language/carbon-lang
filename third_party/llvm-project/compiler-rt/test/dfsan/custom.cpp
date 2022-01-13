// RUN: %clang_dfsan %s -o %t && DFSAN_OPTIONS="strict_data_dependencies=0" %run %t
// RUN: %clang_dfsan -mllvm -dfsan-args-abi %s -o %t && DFSAN_OPTIONS="strict_data_dependencies=0" %run %t
// RUN: %clang_dfsan %s -o %t && DFSAN_OPTIONS="strict_data_dependencies=0" %run %t
// RUN: %clang_dfsan -DSTRICT_DATA_DEPENDENCIES %s -o %t && %run %t
// RUN: %clang_dfsan -DSTRICT_DATA_DEPENDENCIES -mllvm -dfsan-args-abi %s -o %t && %run %t
// RUN: %clang_dfsan -DORIGIN_TRACKING -mllvm -dfsan-track-origins=1 -mllvm -dfsan-combine-pointer-labels-on-load=false -DSTRICT_DATA_DEPENDENCIES %s -o %t && %run %t
// RUN: %clang_dfsan -DORIGIN_TRACKING -mllvm -dfsan-track-origins=1 -mllvm -dfsan-combine-pointer-labels-on-load=false %s -o %t && DFSAN_OPTIONS="strict_data_dependencies=0" %run %t
//
// Tests custom implementations of various glibc functions.
//
// REQUIRES: x86_64-target-arch

#pragma clang diagnostic ignored "-Wformat-extra-args"

#include <sanitizer/dfsan_interface.h>

#include <arpa/inet.h>
#include <assert.h>
#include <fcntl.h>
#include <link.h>
#include <poll.h>
#include <pthread.h>
#include <pwd.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/epoll.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

dfsan_label i_label = 0;
dfsan_label j_label = 0;
dfsan_label k_label = 0;
dfsan_label m_label = 0;
dfsan_label n_label = 0;
dfsan_label i_j_label = 0;

#define ASSERT_ZERO_LABEL(data) \
  assert(0 == dfsan_get_label((long) (data)))

#define ASSERT_READ_ZERO_LABEL(ptr, size) \
  assert(0 == dfsan_read_label(ptr, size))

#define ASSERT_LABEL(data, label) \
  assert(label == dfsan_get_label((long) (data)))

#define ASSERT_READ_LABEL(ptr, size, label) \
  assert(label == dfsan_read_label(ptr, size))

#ifdef ORIGIN_TRACKING
#define ASSERT_ZERO_ORIGIN(data) \
  assert(0 == dfsan_get_origin((long)(data)))
#else
#define ASSERT_ZERO_ORIGIN(data)
#endif

#ifdef ORIGIN_TRACKING
#define ASSERT_ZERO_ORIGINS(ptr, size)                       \
  for (int i = 0; i < size; ++i) {                           \
    assert(0 == dfsan_get_origin((long)(((char *)ptr)[i]))); \
  }
#else
#define ASSERT_ZERO_ORIGINS(ptr, size)
#endif

#ifdef ORIGIN_TRACKING
#define ASSERT_ORIGIN(data, origin) \
  assert(origin == dfsan_get_origin((long)(data)))
#else
#define ASSERT_ORIGIN(data, origin)
#endif

#ifdef ORIGIN_TRACKING
#define ASSERT_ORIGINS(ptr, size, origin)                         \
  for (int i = 0; i < size; ++i) {                                \
    assert(origin == dfsan_get_origin((long)(((char *)ptr)[i]))); \
  }
#else
#define ASSERT_ORIGINS(ptr, size, origin)
#endif

#ifdef ORIGIN_TRACKING
#define ASSERT_INIT_ORIGIN(ptr, origin) \
  assert(origin == dfsan_get_init_origin(ptr))
#else
#define ASSERT_INIT_ORIGIN(ptr, origin)
#endif

#ifdef ORIGIN_TRACKING
#define ASSERT_INIT_ORIGIN_EQ_ORIGIN(ptr, data) \
  assert(dfsan_get_origin((long)(data)) == dfsan_get_init_origin(ptr))
#else
#define ASSERT_INIT_ORIGIN_EQ_ORIGIN(ptr, data)
#endif

#ifdef ORIGIN_TRACKING
#define ASSERT_INIT_ORIGINS(ptr, size, origin)                  \
  for (int i = 0; i < size; ++i) {                              \
    assert(origin == dfsan_get_init_origin(&((char *)ptr)[i])); \
  }
#else
#define ASSERT_INIT_ORIGINS(ptr, size, origin)
#endif

#ifdef ORIGIN_TRACKING
#define ASSERT_EQ_ORIGIN(data1, data2) \
  assert(dfsan_get_origin((long)(data1)) == dfsan_get_origin((long)(data2)))
#else
#define ASSERT_EQ_ORIGIN(data1, data2)
#endif

#ifdef ORIGIN_TRACKING
#define DEFINE_AND_SAVE_ORIGINS(val)    \
  dfsan_origin val##_o[sizeof(val)];    \
  for (int i = 0; i < sizeof(val); ++i) \
    val##_o[i] = dfsan_get_origin((long)(((char *)(&val))[i]));
#else
#define DEFINE_AND_SAVE_ORIGINS(val)
#endif

#ifdef ORIGIN_TRACKING
#define SAVE_ORIGINS(val)               \
  for (int i = 0; i < sizeof(val); ++i) \
    val##_o[i] = dfsan_get_origin((long)(((char *)(&val))[i]));
#else
#define SAVE_ORIGINS(val)
#endif

#ifdef ORIGIN_TRACKING
#define ASSERT_SAVED_ORIGINS(val)       \
  for (int i = 0; i < sizeof(val); ++i) \
    ASSERT_ORIGIN(((char *)(&val))[i], val##_o[i]);
#else
#define ASSERT_SAVED_ORIGINS(val)
#endif

#ifdef ORIGIN_TRACKING
#define DEFINE_AND_SAVE_N_ORIGINS(val, n) \
  dfsan_origin val##_o[n];                \
  for (int i = 0; i < n; ++i)             \
    val##_o[i] = dfsan_get_origin((long)(val[i]));
#else
#define DEFINE_AND_SAVE_N_ORIGINS(val, n)
#endif

#ifdef ORIGIN_TRACKING
#define ASSERT_SAVED_N_ORIGINS(val, n) \
  for (int i = 0; i < n; ++i)          \
    ASSERT_ORIGIN(val[i], val##_o[i]);
#else
#define ASSERT_SAVED_N_ORIGINS(val, n)
#endif

#if !defined(__GLIBC_PREREQ)
#  define __GLIBC_PREREQ(a, b) 0
#endif

void test_stat() {
  int i = 1;
  dfsan_set_label(i_label, &i, sizeof(i));

  struct stat s;
  s.st_dev = i;
  DEFINE_AND_SAVE_ORIGINS(s)
  int ret = stat("/", &s);
  assert(0 == ret);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_ZERO_LABEL(s.st_dev);
  ASSERT_SAVED_ORIGINS(s)

  s.st_dev = i;
  SAVE_ORIGINS(s)
  ret = stat("/nonexistent", &s);
  assert(-1 == ret);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_LABEL(s.st_dev, i_label);
  ASSERT_SAVED_ORIGINS(s)
}

void test_fstat() {
  int i = 1;
  dfsan_set_label(i_label, &i, sizeof(i));

  struct stat s;
  int fd = open("/dev/zero", O_RDONLY);
  s.st_dev = i;
  DEFINE_AND_SAVE_ORIGINS(s)
  int rv = fstat(fd, &s);
  assert(0 == rv);
  ASSERT_ZERO_LABEL(rv);
  ASSERT_ZERO_LABEL(s.st_dev);
  ASSERT_SAVED_ORIGINS(s)
}

void test_memcmp() {
  char str1[] = "str1", str2[] = "str2";
  dfsan_set_label(i_label, &str1[3], 1);
  dfsan_set_label(j_label, &str2[3], 1);

  int rv = memcmp(str1, str2, sizeof(str1));
  assert(rv < 0);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, i_j_label);
  ASSERT_EQ_ORIGIN(rv, str1[3]);
#endif

  rv = memcmp(str1, str2, sizeof(str1) - 2);
  assert(rv == 0);
  ASSERT_ZERO_LABEL(rv);
}

void test_bcmp() {
  char str1[] = "str1", str2[] = "str2";
  dfsan_set_label(i_label, &str1[3], 1);
  dfsan_set_label(j_label, &str2[3], 1);

  int rv = bcmp(str1, str2, sizeof(str1));
  assert(rv != 0);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, i_j_label);
  ASSERT_EQ_ORIGIN(rv, str1[3]);
#endif

  rv = bcmp(str1, str2, sizeof(str1) - 2);
  assert(rv == 0);
  ASSERT_ZERO_LABEL(rv);
}

void test_memcpy() {
  char str1[] = "str1";
  char str2[sizeof(str1)];
  dfsan_set_label(i_label, &str1[3], 1);

  DEFINE_AND_SAVE_ORIGINS(str1)

  char *ptr2 = str2;
  dfsan_set_label(j_label, &ptr2, sizeof(ptr2));

  void *r = memcpy(ptr2, str1, sizeof(str1));
  ASSERT_LABEL(r, j_label);
  ASSERT_EQ_ORIGIN(r, ptr2);
  assert(0 == memcmp(str2, str1, sizeof(str1)));
  ASSERT_ZERO_LABEL(str2[0]);
  ASSERT_LABEL(str2[3], i_label);

  for (int i = 0; i < sizeof(str2); ++i) {
    if (!dfsan_get_label(str2[i]))
      continue;
    ASSERT_INIT_ORIGIN(&(str2[i]), str1_o[i]);
  }
}

void test_memmove() {
  char str[] = "str1xx";
  dfsan_set_label(i_label, &str[3], 1);

  DEFINE_AND_SAVE_ORIGINS(str)

  char *ptr = str + 2;
  dfsan_set_label(j_label, &ptr, sizeof(ptr));

  void *r = memmove(ptr, str, 4);
  ASSERT_LABEL(r, j_label);
  ASSERT_EQ_ORIGIN(r, ptr);
  assert(0 == memcmp(str + 2, "str1", 4));
  ASSERT_ZERO_LABEL(str[4]);
  ASSERT_LABEL(str[5], i_label);

  for (int i = 0; i < 4; ++i) {
    if (!dfsan_get_label(ptr[i]))
      continue;
    ASSERT_INIT_ORIGIN(&(ptr[i]), str_o[i]);
  }
}

void test_memset() {
  char buf[8];
  int j = 'a';
  char *ptr = buf;
  dfsan_set_label(j_label, &j, sizeof(j));
  dfsan_set_label(k_label, &ptr, sizeof(ptr));
  void *ret = memset(ptr, j, sizeof(buf));
  ASSERT_LABEL(ret, k_label);
  ASSERT_EQ_ORIGIN(ret, ptr);
  for (int i = 0; i < 8; ++i) {
    ASSERT_LABEL(buf[i], j_label);
    ASSERT_EQ_ORIGIN(buf[i], j);
    assert(buf[i] == 'a');
  }
}

void test_strcmp() {
  char str1[] = "str1", str2[] = "str2";
  dfsan_set_label(i_label, &str1[3], 1);
  dfsan_set_label(j_label, &str2[3], 1);

  int rv = strcmp(str1, str2);
  assert(rv < 0);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, i_j_label);
  ASSERT_EQ_ORIGIN(rv, str1[3]);
#endif

  rv = strcmp(str1, str1);
  assert(rv == 0);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
  ASSERT_ZERO_ORIGIN(rv);
#else
  ASSERT_LABEL(rv, i_label);
  ASSERT_EQ_ORIGIN(rv, str1[3]);
#endif
}

void test_strcat() {
  char src[] = "world";
  int volatile x = 0; // buffer to ensure src and dst do not share origins
  (void)x;
  char dst[] = "hello \0    ";
  int volatile y = 0; // buffer to ensure dst and p do not share origins
  (void)y;
  char *p = dst;
  dfsan_set_label(k_label, &p, sizeof(p));
  dfsan_set_label(i_label, src, sizeof(src));
  dfsan_set_label(j_label, dst, sizeof(dst));
  dfsan_origin dst_o = dfsan_get_origin((long)dst[0]);
  (void)dst_o;
  char *ret = strcat(p, src);
  ASSERT_LABEL(ret, k_label);
  ASSERT_EQ_ORIGIN(ret, p);
  assert(ret == dst);
  assert(strcmp(src, dst + 6) == 0);
  // Origins are assigned for every 4 contiguous 4-aligned bytes. After
  // appending src to dst, origins of src can overwrite origins of dst if their
  // application adddresses are within [start_aligned_down, end_aligned_up).
  // Other origins are not changed.
  char *start_aligned_down = (char *)(((size_t)(dst + 6)) & ~3UL);
  char *end_aligned_up = (char *)(((size_t)(dst + 11 + 4)) & ~3UL);
  for (int i = 0; i < 12; ++i) {
    if (dst + i < start_aligned_down || dst + i >= end_aligned_up) {
      ASSERT_INIT_ORIGIN(&dst[i], dst_o);
    } else {
      ASSERT_INIT_ORIGIN_EQ_ORIGIN(&dst[i], src[0]);
    }
  }
  for (int i = 0; i < 6; ++i) {
    ASSERT_LABEL(dst[i], j_label);
  }
  for (int i = 6; i < strlen(dst); ++i) {
    ASSERT_LABEL(dst[i], i_label);
    assert(dfsan_get_label(dst[i]) == dfsan_get_label(src[i - 6]));
  }
  ASSERT_LABEL(dst[11], j_label);
}

void test_strlen() {
  char str1[] = "str1";
  dfsan_set_label(i_label, &str1[3], 1);

  int rv = strlen(str1);
  assert(rv == 4);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, i_label);
  ASSERT_EQ_ORIGIN(rv, str1[3]);
#endif
}

void test_strdup() {
  char str1[] = "str1";
  dfsan_set_label(i_label, &str1[3], 1);
  DEFINE_AND_SAVE_ORIGINS(str1)

  char *strd = strdup(str1);
  ASSERT_ZERO_LABEL(strd);
  ASSERT_ZERO_LABEL(strd[0]);
  ASSERT_LABEL(strd[3], i_label);

  for (int i = 0; i < strlen(strd); ++i) {
    if (!dfsan_get_label(strd[i]))
      continue;
    ASSERT_INIT_ORIGIN(&(strd[i]), str1_o[i]);
  }

  free(strd);
}

void test_strncpy() {
  char str1[] = "str1";
  char str2[sizeof(str1)];
  dfsan_set_label(i_label, &str1[3], 1);

  char *strd = strncpy(str2, str1, 5);
  assert(strd == str2);
  assert(strcmp(str1, str2) == 0);
  ASSERT_ZERO_LABEL(strd);
  ASSERT_ZERO_LABEL(strd[0]);
  ASSERT_ZERO_LABEL(strd[1]);
  ASSERT_ZERO_LABEL(strd[2]);
  ASSERT_LABEL(strd[3], i_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&(strd[3]), str1[3]);

  char *p2 = str2;
  dfsan_set_label(j_label, &p2, sizeof(p2));
  strd = strncpy(p2, str1, 3);
  assert(strd == str2);
  assert(strncmp(str1, str2, 3) == 0);
  ASSERT_LABEL(strd, j_label);
  ASSERT_EQ_ORIGIN(strd, p2);
  // When -dfsan-combine-pointer-labels-on-load is on, strd's label propagates
  // to strd[i]'s label. When ORIGIN_TRACKING is defined,
  // -dfsan-combine-pointer-labels-on-load is always off, otherwise the flag
  // is on by default.
#if defined(ORIGIN_TRACKING)
  ASSERT_ZERO_LABEL(strd[0]);
  ASSERT_ZERO_LABEL(strd[1]);
  ASSERT_ZERO_LABEL(strd[2]);
#else
  ASSERT_LABEL(strd[0], j_label);
  ASSERT_LABEL(strd[1], j_label);
  ASSERT_LABEL(strd[2], j_label);
#endif
}

void test_strncmp() {
  char str1[] = "str1", str2[] = "str2";
  dfsan_set_label(i_label, &str1[3], 1);
  dfsan_set_label(j_label, &str2[3], 1);

  int rv = strncmp(str1, str2, sizeof(str1));
  assert(rv < 0);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, dfsan_union(i_label, j_label));
  ASSERT_EQ_ORIGIN(rv, str1[3]);
#endif

  rv = strncmp(str1, str2, 0);
  assert(rv == 0);
  ASSERT_ZERO_LABEL(rv);

  rv = strncmp(str1, str2, 3);
  assert(rv == 0);
  ASSERT_ZERO_LABEL(rv);

  rv = strncmp(str1, str1, 4);
  assert(rv == 0);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, i_label);
  ASSERT_EQ_ORIGIN(rv, str1[3]);
#endif
}

void test_strcasecmp() {
  char str1[] = "str1", str2[] = "str2", str3[] = "Str1";
  dfsan_set_label(i_label, &str1[3], 1);
  dfsan_set_label(j_label, &str2[3], 1);
  dfsan_set_label(j_label, &str3[2], 1);

  int rv = strcasecmp(str1, str2);
  assert(rv < 0);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, dfsan_union(i_label, j_label));
  ASSERT_EQ_ORIGIN(rv, str1[3]);
#endif

  rv = strcasecmp(str1, str3);
  assert(rv == 0);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, dfsan_union(i_label, j_label));
  ASSERT_EQ_ORIGIN(rv, str1[3]);
#endif

  char s1[] = "AbZ";
  char s2[] = "aBy";
  dfsan_set_label(i_label, &s1[2], 1);
  dfsan_set_label(j_label, &s2[2], 1);

  rv = strcasecmp(s1, s2);
  assert(rv > 0); // 'Z' > 'y'
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, dfsan_union(i_label, j_label));
  ASSERT_EQ_ORIGIN(rv, s1[2]);
#endif
}

void test_strncasecmp() {
  char str1[] = "Str1", str2[] = "str2";
  dfsan_set_label(i_label, &str1[3], 1);
  dfsan_set_label(j_label, &str2[3], 1);

  int rv = strncasecmp(str1, str2, sizeof(str1));
  assert(rv < 0);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, dfsan_union(i_label, j_label));
  ASSERT_EQ_ORIGIN(rv, str1[3]);
#endif

  rv = strncasecmp(str1, str2, 3);
  assert(rv == 0);
  ASSERT_ZERO_LABEL(rv);

  char s1[] = "AbZ";
  char s2[] = "aBy";
  dfsan_set_label(i_label, &s1[2], 1);
  dfsan_set_label(j_label, &s2[2], 1);

  rv = strncasecmp(s1, s2, 0);
  assert(rv == 0); // Compare zero chars.
  ASSERT_ZERO_LABEL(rv);

  rv = strncasecmp(s1, s2, 1);
  assert(rv == 0); // 'A' == 'a'
  ASSERT_ZERO_LABEL(rv);

  rv = strncasecmp(s1, s2, 2);
  assert(rv == 0); // 'b' == 'B'
  ASSERT_ZERO_LABEL(rv);

  rv = strncasecmp(s1, s2, 3);
  assert(rv > 0); // 'Z' > 'y'
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, dfsan_union(i_label, j_label));
  ASSERT_EQ_ORIGIN(rv, s1[2]);
#endif
}

void test_strchr() {
  char str1[] = "str1";
  dfsan_set_label(i_label, &str1[3], 1);

  char *p1 = str1;
  char c = 'r';
  dfsan_set_label(k_label, &c, sizeof(c));

  char *crv = strchr(p1, c);
  assert(crv == &str1[2]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(crv);
#else
  ASSERT_LABEL(crv, k_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&crv, c);
#endif

  dfsan_set_label(j_label, &p1, sizeof(p1));
  crv = strchr(p1, 'r');
  assert(crv == &str1[2]);
  ASSERT_LABEL(crv, j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&crv, p1);

  crv = strchr(p1, '1');
  assert(crv == &str1[3]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_LABEL(crv, j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&crv, p1);
#else
  ASSERT_LABEL(crv, i_j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&crv, str1[3]);
#endif

  crv = strchr(p1, 'x');
  assert(!crv);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_LABEL(crv, j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&crv, p1);
#else
  ASSERT_LABEL(crv, i_j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&crv, str1[3]);
#endif

  // `man strchr` says:
  // The terminating null byte is considered part of the string, so that if c
  // is specified as '\0', these functions return a pointer to the terminator.
  crv = strchr(p1, '\0');
  assert(crv == &str1[4]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_LABEL(crv, j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&crv, p1);
#else
  ASSERT_LABEL(crv, i_j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&crv, str1[3]);
#endif
}

void test_recvmmsg() {
  int sockfds[2];
  int ret = socketpair(AF_UNIX, SOCK_DGRAM, 0, sockfds);
  assert(ret != -1);

  // Setup messages to send.
  struct mmsghdr smmsg[2] = {};
  char sbuf0[] = "abcdefghijkl";
  struct iovec siov0[2] = {{&sbuf0[0], 4}, {&sbuf0[4], 4}};
  smmsg[0].msg_hdr.msg_iov = siov0;
  smmsg[0].msg_hdr.msg_iovlen = 2;
  char sbuf1[] = "1234567890";
  struct iovec siov1[1] = {{&sbuf1[0], 7}};
  smmsg[1].msg_hdr.msg_iov = siov1;
  smmsg[1].msg_hdr.msg_iovlen = 1;

  // Send messages.
  int sent_msgs = sendmmsg(sockfds[0], smmsg, 2, 0);
  assert(sent_msgs == 2);

  // Setup receive buffers.
  struct mmsghdr rmmsg[2] = {};
  char rbuf0[128];
  struct iovec riov0[2] = {{&rbuf0[0], 4}, {&rbuf0[4], 4}};
  rmmsg[0].msg_hdr.msg_iov = riov0;
  rmmsg[0].msg_hdr.msg_iovlen = 2;
  char rbuf1[128];
  struct iovec riov1[1] = {{&rbuf1[0], 16}};
  rmmsg[1].msg_hdr.msg_iov = riov1;
  rmmsg[1].msg_hdr.msg_iovlen = 1;
  struct timespec timeout = {1, 1};
  dfsan_set_label(i_label, rbuf0, sizeof(rbuf0));
  dfsan_set_label(i_label, rbuf1, sizeof(rbuf1));
  dfsan_set_label(i_label, &rmmsg[0].msg_len, sizeof(rmmsg[0].msg_len));
  dfsan_set_label(i_label, &rmmsg[1].msg_len, sizeof(rmmsg[1].msg_len));
  dfsan_set_label(i_label, &timeout, sizeof(timeout));

  dfsan_origin msg_len0_o = dfsan_get_origin((long)(rmmsg[0].msg_len));
  dfsan_origin msg_len1_o = dfsan_get_origin((long)(rmmsg[1].msg_len));
#ifndef ORIGIN_TRACKING
  (void)msg_len0_o;
  (void)msg_len1_o;
#endif

  // Receive messages and check labels.
  int received_msgs = recvmmsg(sockfds[1], rmmsg, 2, 0, &timeout);
  assert(received_msgs == sent_msgs);
  assert(rmmsg[0].msg_len == smmsg[0].msg_len);
  assert(rmmsg[1].msg_len == smmsg[1].msg_len);
  assert(memcmp(sbuf0, rbuf0, 8) == 0);
  assert(memcmp(sbuf1, rbuf1, 7) == 0);
  ASSERT_ZERO_LABEL(received_msgs);
  ASSERT_ZERO_LABEL(rmmsg[0].msg_len);
  ASSERT_ZERO_LABEL(rmmsg[1].msg_len);
  ASSERT_READ_ZERO_LABEL(&rbuf0[0], 8);
  ASSERT_READ_LABEL(&rbuf0[8], 1, i_label);
  ASSERT_READ_ZERO_LABEL(&rbuf1[0], 7);
  ASSERT_READ_LABEL(&rbuf1[7], 1, i_label);
  ASSERT_LABEL(timeout.tv_sec, i_label);
  ASSERT_LABEL(timeout.tv_nsec, i_label);

  ASSERT_ORIGIN((long)(rmmsg[0].msg_len), msg_len0_o);
  ASSERT_ORIGIN((long)(rmmsg[1].msg_len), msg_len1_o);

  close(sockfds[0]);
  close(sockfds[1]);
}

void test_recvmsg() {
  int sockfds[2];
  int ret = socketpair(AF_UNIX, SOCK_DGRAM, 0, sockfds);
  assert(ret != -1);

  char sbuf[] = "abcdefghijkl";
  struct iovec siovs[2] = {{&sbuf[0], 4}, {&sbuf[4], 4}};
  struct msghdr smsg = {};
  smsg.msg_iov = siovs;
  smsg.msg_iovlen = 2;

  ssize_t sent = sendmsg(sockfds[0], &smsg, 0);
  assert(sent > 0);

  char rbuf[128];
  struct iovec riovs[2] = {{&rbuf[0], 4}, {&rbuf[4], 4}};
  struct msghdr rmsg = {};
  rmsg.msg_iov = riovs;
  rmsg.msg_iovlen = 2;

  dfsan_set_label(i_label, rbuf, sizeof(rbuf));
  dfsan_set_label(i_label, &rmsg, sizeof(rmsg));

  DEFINE_AND_SAVE_ORIGINS(rmsg)

  ssize_t received = recvmsg(sockfds[1], &rmsg, 0);
  assert(received == sent);
  assert(memcmp(sbuf, rbuf, 8) == 0);
  ASSERT_ZERO_LABEL(received);
  ASSERT_READ_ZERO_LABEL(&rmsg, sizeof(rmsg));
  ASSERT_READ_ZERO_LABEL(&rbuf[0], 8);
  ASSERT_READ_LABEL(&rbuf[8], 1, i_label);

  ASSERT_SAVED_ORIGINS(rmsg)

  close(sockfds[0]);
  close(sockfds[1]);
}

void test_read() {
  char buf[16];
  dfsan_set_label(i_label, buf, 1);
  dfsan_set_label(j_label, buf + 15, 1);

  DEFINE_AND_SAVE_ORIGINS(buf)
  ASSERT_LABEL(buf[0], i_label);
  ASSERT_LABEL(buf[15], j_label);

  int fd = open("/dev/zero", O_RDONLY);
  int rv = read(fd, buf, sizeof(buf));
  assert(rv == sizeof(buf));
  ASSERT_ZERO_LABEL(rv);
  ASSERT_ZERO_LABEL(buf[0]);
  ASSERT_ZERO_LABEL(buf[15]);
  ASSERT_SAVED_ORIGINS(buf)
  close(fd);
}

void test_pread() {
  char buf[16];
  dfsan_set_label(i_label, buf, 1);
  dfsan_set_label(j_label, buf + 15, 1);

  DEFINE_AND_SAVE_ORIGINS(buf)
  ASSERT_LABEL(buf[0], i_label);
  ASSERT_LABEL(buf[15], j_label);

  int fd = open("/bin/sh", O_RDONLY);
  int rv = pread(fd, buf, sizeof(buf), 0);
  assert(rv == sizeof(buf));
  ASSERT_ZERO_LABEL(rv);
  ASSERT_ZERO_LABEL(buf[0]);
  ASSERT_ZERO_LABEL(buf[15]);
  ASSERT_SAVED_ORIGINS(buf)
  close(fd);
}

void test_dlopen() {
  void *map = dlopen(NULL, RTLD_NOW);
  assert(map);
  ASSERT_ZERO_LABEL(map);
  dlclose(map);
  map = dlopen("/nonexistent", RTLD_NOW);
  assert(!map);
  ASSERT_ZERO_LABEL(map);
}

void test_clock_gettime() {
  struct timespec tp;
  dfsan_set_label(j_label, ((char *)&tp) + 3, 1);
  dfsan_origin origin = dfsan_get_origin((long)(((char *)&tp)[3]));
#ifndef ORIGIN_TRACKING
  (void)origin;
#endif
  int t = clock_gettime(CLOCK_REALTIME, &tp);
  assert(t == 0);
  ASSERT_ZERO_LABEL(t);
  ASSERT_ZERO_LABEL(((char *)&tp)[3]);
  ASSERT_ORIGIN(((char *)&tp)[3], origin);
}

void test_ctime_r() {
  char *buf = (char*) malloc(64);
  time_t t = 0;

  DEFINE_AND_SAVE_ORIGINS(buf)
  dfsan_origin t_o = dfsan_get_origin((long)t);

  char *ret = ctime_r(&t, buf);
  ASSERT_ZERO_LABEL(ret);
  assert(buf == ret);
  ASSERT_READ_ZERO_LABEL(buf, strlen(buf) + 1);
  ASSERT_SAVED_ORIGINS(buf)

  dfsan_set_label(i_label, &t, sizeof(t));
  t_o = dfsan_get_origin((long)t);
  ret = ctime_r(&t, buf);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_LABEL(buf, strlen(buf) + 1, i_label);
  for (int i = 0; i < strlen(buf) + 1; ++i)
    ASSERT_ORIGIN(buf[i], t_o);

  t = 0;
  dfsan_set_label(j_label, &buf, sizeof(&buf));
  dfsan_origin buf_ptr_o = dfsan_get_origin((long)buf);
#ifndef ORIGIN_TRACKING
  (void)buf_ptr_o;
#endif
  ret = ctime_r(&t, buf);
  ASSERT_LABEL(ret, j_label);
  ASSERT_ORIGIN(ret, buf_ptr_o);
  ASSERT_READ_ZERO_LABEL(buf, strlen(buf) + 1);
  for (int i = 0; i < strlen(buf) + 1; ++i)
    ASSERT_ORIGIN(buf[i], t_o);
}

static int write_callback_count = 0;
static int last_fd;
static const unsigned char *last_buf;
static size_t last_count;

void write_callback(int fd, const void *buf, size_t count) {
  write_callback_count++;

  last_fd = fd;
  last_buf = (const unsigned char*) buf;
  last_count = count;
}

void test_dfsan_set_write_callback() {
  char buf[] = "Sample chars";
  int buf_len = strlen(buf);

  int fd = open("/dev/null", O_WRONLY);

  dfsan_set_write_callback(write_callback);

  write_callback_count = 0;

  DEFINE_AND_SAVE_ORIGINS(buf)

  // Callback should be invoked on every call to write().
  int res = write(fd, buf, buf_len);
  assert(write_callback_count == 1);
  ASSERT_READ_ZERO_LABEL(&res, sizeof(res));
  ASSERT_READ_ZERO_LABEL(&last_fd, sizeof(last_fd));
  ASSERT_READ_ZERO_LABEL(last_buf, sizeof(last_buf));
  ASSERT_READ_ZERO_LABEL(&last_count, sizeof(last_count));

  for (int i = 0; i < buf_len; ++i)
    ASSERT_ORIGIN(last_buf[i], buf_o[i]);

  ASSERT_ZERO_ORIGINS(&last_count, sizeof(last_count));

  // Add a label to write() arguments.  Check that the labels are readable from
  // the values passed to the callback.
  dfsan_set_label(i_label, &fd, sizeof(fd));
  dfsan_set_label(j_label, &(buf[3]), 1);
  dfsan_set_label(k_label, &buf_len, sizeof(buf_len));

  dfsan_origin fd_o = dfsan_get_origin((long)fd);
  dfsan_origin buf3_o = dfsan_get_origin((long)(buf[3]));
  dfsan_origin buf_len_o = dfsan_get_origin((long)buf_len);
#ifndef ORIGIN_TRACKING
  (void)fd_o;
  (void)buf3_o;
  (void)buf_len_o;
#endif

  res = write(fd, buf, buf_len);
  assert(write_callback_count == 2);
  ASSERT_READ_ZERO_LABEL(&res, sizeof(res));
  ASSERT_READ_LABEL(&last_fd, sizeof(last_fd), i_label);
  ASSERT_READ_LABEL(&last_buf[3], sizeof(last_buf[3]), j_label);
  ASSERT_READ_LABEL(last_buf, sizeof(last_buf), j_label);
  ASSERT_READ_LABEL(&last_count, sizeof(last_count), k_label);
  ASSERT_ZERO_ORIGINS(&res, sizeof(res));
  ASSERT_INIT_ORIGINS(&last_fd, sizeof(last_fd), fd_o);
  ASSERT_INIT_ORIGINS(&last_buf[3], sizeof(last_buf[3]), buf3_o);

  // Origins are assigned for every 4 contiguous 4-aligned bytes. After
  // appending src to dst, origins of src can overwrite origins of dst if their
  // application adddresses are within an aligned range. Other origins are not
  // changed.
  for (int i = 0; i < buf_len; ++i) {
    size_t i_addr = size_t(&last_buf[i]);
    if (((size_t(&last_buf[3]) & ~3UL) > i_addr) ||
        (((size_t(&last_buf[3]) + 4) & ~3UL) <= i_addr))
      ASSERT_ORIGIN(last_buf[i], buf_o[i]);
  }

  ASSERT_INIT_ORIGINS(&last_count, sizeof(last_count), buf_len_o);

  dfsan_set_write_callback(NULL);
}

void test_fgets() {
  char *buf = (char*) malloc(128);
  FILE *f = fopen("/etc/passwd", "r");
  dfsan_set_label(j_label, buf, 1);
  DEFINE_AND_SAVE_N_ORIGINS(buf, 128)

  char *ret = fgets(buf, sizeof(buf), f);
  assert(ret == buf);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_EQ_ORIGIN(ret, buf);
  ASSERT_READ_ZERO_LABEL(buf, 128);
  ASSERT_SAVED_N_ORIGINS(buf, 128)

  dfsan_set_label(j_label, &buf, sizeof(&buf));
  ret = fgets(buf, sizeof(buf), f);
  ASSERT_LABEL(ret, j_label);
  ASSERT_EQ_ORIGIN(ret, buf);
  ASSERT_SAVED_N_ORIGINS(buf, 128)

  fclose(f);
  free(buf);
}

void test_getcwd() {
  char buf[1024];
  char *ptr = buf;
  dfsan_set_label(i_label, buf + 2, 2);
  DEFINE_AND_SAVE_ORIGINS(buf)

  char* ret = getcwd(buf, sizeof(buf));
  assert(ret == buf);
  assert(ret[0] == '/');
  ASSERT_ZERO_LABEL(ret);
  ASSERT_EQ_ORIGIN(ret, buf);
  ASSERT_READ_ZERO_LABEL(buf + 2, 2);
  ASSERT_SAVED_ORIGINS(buf)

  dfsan_set_label(i_label, &ptr, sizeof(ptr));
  ret = getcwd(ptr, sizeof(buf));
  ASSERT_LABEL(ret, i_label);
  ASSERT_EQ_ORIGIN(ret, ptr);
  ASSERT_SAVED_ORIGINS(buf)
}

void test_get_current_dir_name() {
  char* ret = get_current_dir_name();
  assert(ret);
  assert(ret[0] == '/');
  ASSERT_READ_ZERO_LABEL(ret, strlen(ret) + 1);
  ASSERT_ZERO_LABEL(ret);
}

void test_getentropy() {
  char buf[64];
  dfsan_set_label(i_label, buf + 2, 2);
  DEFINE_AND_SAVE_ORIGINS(buf)
#if __GLIBC_PREREQ(2, 25)
  // glibc >= 2.25 has getentropy()
  int ret = getentropy(buf, sizeof(buf));
  ASSERT_ZERO_LABEL(ret);
  if (ret == 0) {
    ASSERT_READ_ZERO_LABEL(buf + 2, 2);
    ASSERT_SAVED_ORIGINS(buf)
  }
#endif
}

void test_gethostname() {
  char buf[1024];
  dfsan_set_label(i_label, buf + 2, 2);
  DEFINE_AND_SAVE_ORIGINS(buf)
  int ret = gethostname(buf, sizeof(buf));
  assert(ret == 0);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_ZERO_LABEL(buf + 2, 2);
  ASSERT_SAVED_ORIGINS(buf)
}

void test_getrlimit() {
  struct rlimit rlim;
  dfsan_set_label(i_label, &rlim, sizeof(rlim));
  DEFINE_AND_SAVE_ORIGINS(rlim);
  int ret = getrlimit(RLIMIT_CPU, &rlim);
  assert(ret == 0);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_ZERO_LABEL(&rlim, sizeof(rlim));
  ASSERT_SAVED_ORIGINS(rlim)
}

void test_getrusage() {
  struct rusage usage;
  dfsan_set_label(i_label, &usage, sizeof(usage));
  DEFINE_AND_SAVE_ORIGINS(usage);
  int ret = getrusage(RUSAGE_SELF, &usage);
  assert(ret == 0);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_ZERO_LABEL(&usage, sizeof(usage));
  ASSERT_SAVED_ORIGINS(usage)
}

void test_strcpy() {
  char src[] = "hello world";
  char dst[sizeof(src) + 2];
  char *p_dst = dst;
  dfsan_set_label(0, src, sizeof(src));
  dfsan_set_label(0, dst, sizeof(dst));
  dfsan_set_label(k_label, &p_dst, sizeof(p_dst));
  dfsan_set_label(i_label, src + 2, 1);
  dfsan_set_label(j_label, src + 3, 1);
  dfsan_set_label(j_label, dst + 4, 1);
  dfsan_set_label(i_label, dst + 12, 1);
  char *ret = strcpy(p_dst, src);
  assert(ret == dst);
  assert(strcmp(src, dst) == 0);
  ASSERT_LABEL(ret, k_label);
  ASSERT_EQ_ORIGIN(ret, p_dst);
  for (int i = 0; i < strlen(src) + 1; ++i) {
    assert(dfsan_get_label(dst[i]) == dfsan_get_label(src[i]));
    if (dfsan_get_label(dst[i]))
      assert(dfsan_get_init_origin(&dst[i]) == dfsan_get_origin(src[i]));
  }
  // Note: if strlen(src) + 1 were used instead to compute the first untouched
  // byte of dest, the label would be I|J. This is because strlen() might
  // return a non-zero label, and because by default pointer labels are not
  // ignored on loads.
  ASSERT_LABEL(dst[12], i_label);
}

void test_strtol() {
  char non_number_buf[] = "ab ";
  char *endptr = NULL;
  long int ret = strtol(non_number_buf, &endptr, 10);
  assert(ret == 0);
  assert(endptr == non_number_buf);
  ASSERT_ZERO_LABEL(ret);

  char buf[] = "1234578910";
  int base = 10;
  dfsan_set_label(k_label, &base, sizeof(base));
  ret = strtol(buf, &endptr, base);
  assert(ret == 1234578910);
  assert(endptr == buf + 10);
  ASSERT_LABEL(ret, k_label);
  ASSERT_EQ_ORIGIN(ret, base);

  dfsan_set_label(i_label, buf + 1, 1);
  dfsan_set_label(j_label, buf + 10, 1);
  ret = strtol(buf, &endptr, 10);
  assert(ret == 1234578910);
  assert(endptr == buf + 10);
  ASSERT_LABEL(ret, i_j_label);
  ASSERT_EQ_ORIGIN(ret, buf[1]);
}

void test_strtoll() {
  char non_number_buf[] = "ab ";
  char *endptr = NULL;
  long long int ret = strtoll(non_number_buf, &endptr, 10);
  assert(ret == 0);
  assert(endptr == non_number_buf);
  ASSERT_ZERO_LABEL(ret);

  char buf[] = "1234578910 ";
  int base = 10;
  dfsan_set_label(k_label, &base, sizeof(base));
  ret = strtoll(buf, &endptr, base);
  assert(ret == 1234578910);
  assert(endptr == buf + 10);
  ASSERT_LABEL(ret, k_label);
  ASSERT_EQ_ORIGIN(ret, base);

  dfsan_set_label(i_label, buf + 1, 1);
  dfsan_set_label(j_label, buf + 2, 1);
  ret = strtoll(buf, &endptr, 10);
  assert(ret == 1234578910);
  assert(endptr == buf + 10);
  ASSERT_LABEL(ret, i_j_label);
  ASSERT_EQ_ORIGIN(ret, buf[1]);
}

void test_strtoul() {
  char non_number_buf[] = "xy ";
  char *endptr = NULL;
  long unsigned int ret = strtoul(non_number_buf, &endptr, 16);
  assert(ret == 0);
  assert(endptr == non_number_buf);
  ASSERT_ZERO_LABEL(ret);

  char buf[] = "ffffffffffffaa";
  int base = 16;
  dfsan_set_label(k_label, &base, sizeof(base));
  ret = strtoul(buf, &endptr, base);
  assert(ret == 72057594037927850);
  assert(endptr == buf + 14);
  ASSERT_LABEL(ret, k_label);
  ASSERT_EQ_ORIGIN(ret, base);

  dfsan_set_label(i_label, buf + 1, 1);
  dfsan_set_label(j_label, buf + 2, 1);
  ret = strtoul(buf, &endptr, 16);
  assert(ret == 72057594037927850);
  assert(endptr == buf + 14);
  ASSERT_LABEL(ret, i_j_label);
  ASSERT_EQ_ORIGIN(ret, buf[1]);
}

void test_strtoull() {
  char non_number_buf[] = "xy ";
  char *endptr = NULL;
  long long unsigned int ret = strtoull(non_number_buf, &endptr, 16);
  assert(ret == 0);
  assert(endptr == non_number_buf);
  ASSERT_ZERO_LABEL(ret);

  char buf[] = "ffffffffffffffaa";
  int base = 16;
  dfsan_set_label(k_label, &base, sizeof(base));
  ret = strtoull(buf, &endptr, base);
  assert(ret == 0xffffffffffffffaa);
  assert(endptr == buf + 16);
  ASSERT_LABEL(ret, k_label);
  ASSERT_EQ_ORIGIN(ret, base);

  dfsan_set_label(i_label, buf + 1, 1);
  dfsan_set_label(j_label, buf + 2, 1);
  ret = strtoull(buf, &endptr, 16);
  assert(ret == 0xffffffffffffffaa);
  assert(endptr == buf + 16);
  ASSERT_LABEL(ret, i_j_label);
  ASSERT_EQ_ORIGIN(ret, buf[1]);
}

void test_strtod() {
  char non_number_buf[] = "ab ";
  char *endptr = NULL;
  double ret = strtod(non_number_buf, &endptr);
  assert(ret == 0);
  assert(endptr == non_number_buf);
  ASSERT_ZERO_LABEL(ret);

  char buf[] = "12345.76 foo";
  dfsan_set_label(i_label, buf + 1, 1);
  dfsan_set_label(j_label, buf + 6, 1);
  ret = strtod(buf, &endptr);
  assert(ret == 12345.76);
  assert(endptr == buf + 8);
  ASSERT_LABEL(ret, i_j_label);
  ASSERT_EQ_ORIGIN(ret, buf[1]);
}

void test_time() {
  time_t t = 0;
  dfsan_set_label(i_label, &t, 1);
  DEFINE_AND_SAVE_ORIGINS(t)
  time_t ret = time(&t);
  assert(ret == t);
  assert(ret > 0);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_ZERO_LABEL(t);
  ASSERT_SAVED_ORIGINS(t)
}

void test_inet_pton() {
  char addr4[] = "127.0.0.1";
  dfsan_set_label(i_label, addr4 + 3, 1);
  struct in_addr in4;
  int ret4 = inet_pton(AF_INET, addr4, &in4);
  assert(ret4 == 1);
  ASSERT_ZERO_LABEL(ret4);
  ASSERT_READ_LABEL(&in4, sizeof(in4), i_label);
  ASSERT_ORIGINS(&in4, sizeof(in4), dfsan_get_origin((long)(addr4[3])))
  assert(in4.s_addr == htonl(0x7f000001));

  char addr6[] = "::1";
  dfsan_set_label(j_label, addr6 + 3, 1);
  struct in6_addr in6;
  int ret6 = inet_pton(AF_INET6, addr6, &in6);
  assert(ret6 == 1);
  ASSERT_ZERO_LABEL(ret6);
  ASSERT_READ_LABEL(((char *) &in6) + sizeof(in6) - 1, 1, j_label);
  ASSERT_ORIGINS(&in6, sizeof(in6), dfsan_get_origin((long)(addr6[3])))
}

void test_localtime_r() {
  time_t t0 = 1384800998;
  struct tm t1;
  dfsan_set_label(i_label, &t0, sizeof(t0));
  dfsan_origin t0_o = dfsan_get_origin((long)t0);
  struct tm *pt1 = &t1;
  dfsan_set_label(j_label, &pt1, sizeof(pt1));
  dfsan_origin pt1_o = dfsan_get_origin((long)pt1);

#ifndef ORIGIN_TRACKING
  (void)t0_o;
  (void)pt1_o;
#endif

  struct tm *ret = localtime_r(&t0, pt1);
  assert(ret == &t1);
  assert(t1.tm_min == 56);
  ASSERT_LABEL(ret, j_label);
  ASSERT_INIT_ORIGIN(&ret, pt1_o);
  ASSERT_READ_LABEL(&ret, sizeof(ret), j_label);
  ASSERT_LABEL(t1.tm_mon, i_label);
  ASSERT_ORIGIN(t1.tm_mon, t0_o);
}

void test_getpwuid_r() {
  struct passwd pwd;
  char buf[1024];
  struct passwd *result;

  dfsan_set_label(i_label, &pwd, 4);
  DEFINE_AND_SAVE_ORIGINS(pwd)
  DEFINE_AND_SAVE_ORIGINS(buf)
  int ret = getpwuid_r(0, &pwd, buf, sizeof(buf), &result);
  assert(ret == 0);
  assert(strcmp(pwd.pw_name, "root") == 0);
  assert(result == &pwd);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_ZERO_LABEL(&pwd, 4);
  ASSERT_SAVED_ORIGINS(pwd)
  ASSERT_SAVED_ORIGINS(buf)
}

void test_epoll_wait() {
  // Set up a pipe to monitor with epoll.
  int pipe_fds[2];
  int ret = pipe(pipe_fds);
  assert(ret != -1);

  // Configure epoll to monitor the pipe.
  int epfd = epoll_create1(0);
  assert(epfd != -1);
  struct epoll_event event;
  event.events = EPOLLIN;
  event.data.fd = pipe_fds[0];
  ret = epoll_ctl(epfd, EPOLL_CTL_ADD, pipe_fds[0], &event);
  assert(ret != -1);

  // Test epoll_wait when no events have occurred.
  event = {};
  dfsan_set_label(i_label, &event, sizeof(event));
  DEFINE_AND_SAVE_ORIGINS(event)
  ret = epoll_wait(epfd, &event, /*maxevents=*/1, /*timeout=*/0);
  assert(ret == 0);
  assert(event.events == 0);
  assert(event.data.fd == 0);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_LABEL(&event, sizeof(event), i_label);
  ASSERT_SAVED_ORIGINS(event)

  // Test epoll_wait when an event occurs.
  write(pipe_fds[1], "x", 1);
  ret = epoll_wait(epfd, &event, /*maxevents=*/1, /*timeout=*/0);
  assert(ret == 1);
  assert(event.events == EPOLLIN);
  assert(event.data.fd == pipe_fds[0]);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_ZERO_LABEL(&event, sizeof(event));
  ASSERT_SAVED_ORIGINS(event)

  // Clean up.
  close(epfd);
  close(pipe_fds[0]);
  close(pipe_fds[1]);
}

void test_poll() {
  struct pollfd fd;
  fd.fd = 0;
  fd.events = POLLIN;
  dfsan_set_label(i_label, &fd.revents, sizeof(fd.revents));
  DEFINE_AND_SAVE_ORIGINS(fd)
  int ret = poll(&fd, 1, 1);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_ZERO_LABEL(fd.revents);
  ASSERT_SAVED_ORIGINS(fd)
  assert(ret >= 0);
}

void test_select() {
  struct timeval t;
  fd_set fds;
  t.tv_sec = 2;
  FD_SET(0, &fds);
  dfsan_set_label(i_label, &fds, sizeof(fds));
  dfsan_set_label(j_label, &t, sizeof(t));
  DEFINE_AND_SAVE_ORIGINS(fds)
  DEFINE_AND_SAVE_ORIGINS(t)
  int ret = select(1, &fds, NULL, NULL, &t);
  assert(ret >= 0);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_ZERO_LABEL(t.tv_sec);
  ASSERT_READ_ZERO_LABEL(&fds, sizeof(fds));
  ASSERT_SAVED_ORIGINS(fds)
  ASSERT_SAVED_ORIGINS(t)
}

void test_sched_getaffinity() {
  cpu_set_t mask;
  dfsan_set_label(j_label, &mask, 1);
  DEFINE_AND_SAVE_ORIGINS(mask)
  int ret = sched_getaffinity(0, sizeof(mask), &mask);
  assert(ret == 0);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_ZERO_LABEL(&mask, sizeof(mask));
  ASSERT_SAVED_ORIGINS(mask)
}

void test_sigemptyset() {
  sigset_t set;
  dfsan_set_label(j_label, &set, 1);
  DEFINE_AND_SAVE_ORIGINS(set)
  int ret = sigemptyset(&set);
  assert(ret == 0);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_ZERO_LABEL(&set, sizeof(set));
  ASSERT_SAVED_ORIGINS(set)
}

static void SignalHandler(int signo) {}

static void SignalAction(int signo, siginfo_t *si, void *uc) {}

void test_sigaction() {
  struct sigaction newact_with_sigaction = {};
  newact_with_sigaction.sa_flags = SA_SIGINFO;
  newact_with_sigaction.sa_sigaction = SignalAction;

  // Set sigaction to be SignalAction, save the last one into origin_act
  struct sigaction origin_act;
  dfsan_set_label(j_label, &origin_act, 1);
  DEFINE_AND_SAVE_ORIGINS(origin_act)
  int ret = sigaction(SIGUSR1, &newact_with_sigaction, &origin_act);
  assert(ret == 0);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_ZERO_LABEL(&origin_act, sizeof(origin_act));
  ASSERT_SAVED_ORIGINS(origin_act)

  struct sigaction newact_with_sighandler = {};
  newact_with_sighandler.sa_handler = SignalHandler;

  // Set sigaction to be SignalHandler, check the last one is SignalAction
  struct sigaction oldact;
  assert(0 == sigaction(SIGUSR1, &newact_with_sighandler, &oldact));
  assert(oldact.sa_sigaction == SignalAction);
  assert(oldact.sa_flags & SA_SIGINFO);

  // Set SIG_IGN or SIG_DFL, and check the previous one is expected.
  newact_with_sighandler.sa_handler = SIG_IGN;
  assert(0 == sigaction(SIGUSR1, &newact_with_sighandler, &oldact));
  assert(oldact.sa_handler == SignalHandler);
  assert((oldact.sa_flags & SA_SIGINFO) == 0);

  newact_with_sighandler.sa_handler = SIG_DFL;
  assert(0 == sigaction(SIGUSR1, &newact_with_sighandler, &oldact));
  assert(oldact.sa_handler == SIG_IGN);
  assert((oldact.sa_flags & SA_SIGINFO) == 0);

  // Restore sigaction to the orginal setting, check the last one is SignalHandler
  assert(0 == sigaction(SIGUSR1, &origin_act, &oldact));
  assert(oldact.sa_handler == SIG_DFL);
  assert((oldact.sa_flags & SA_SIGINFO) == 0);
}

void test_signal() {
  // Set signal to be SignalHandler, save the previous one into
  // old_signal_handler.
  sighandler_t old_signal_handler = signal(SIGHUP, SignalHandler);
  ASSERT_ZERO_LABEL(old_signal_handler);

  // Set SIG_IGN or SIG_DFL, and check the previous one is expected.
  assert(SignalHandler == signal(SIGHUP, SIG_DFL));
  assert(SIG_DFL == signal(SIGHUP, SIG_IGN));

  // Restore signal to old_signal_handler.
  assert(SIG_IGN == signal(SIGHUP, old_signal_handler));
}

void test_sigaltstack() {
  stack_t old_altstack = {};
  dfsan_set_label(j_label, &old_altstack, sizeof(old_altstack));
  DEFINE_AND_SAVE_ORIGINS(old_altstack)
  int ret = sigaltstack(NULL, &old_altstack);
  assert(ret == 0);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_READ_ZERO_LABEL(&old_altstack, sizeof(old_altstack));
  ASSERT_SAVED_ORIGINS(old_altstack)
}

void test_gettimeofday() {
  struct timeval tv;
  struct timezone tz;
  dfsan_set_label(i_label, &tv, sizeof(tv));
  dfsan_set_label(j_label, &tz, sizeof(tz));
  DEFINE_AND_SAVE_ORIGINS(tv)
  DEFINE_AND_SAVE_ORIGINS(tz)
  int ret = gettimeofday(&tv, &tz);
  assert(ret == 0);
  ASSERT_READ_ZERO_LABEL(&tv, sizeof(tv));
  ASSERT_READ_ZERO_LABEL(&tz, sizeof(tz));
  ASSERT_SAVED_ORIGINS(tv)
  ASSERT_SAVED_ORIGINS(tz)
}

void *pthread_create_test_cb(void *p) {
  assert(p == (void *)1);
  ASSERT_ZERO_LABEL(p);
  return (void *)2;
}

void test_pthread_create() {
  pthread_t pt;
  int create_ret = pthread_create(&pt, 0, pthread_create_test_cb, (void *)1);
  assert(create_ret == 0);
  ASSERT_ZERO_LABEL(create_ret);
  void *cbrv;
  dfsan_set_label(i_label, &cbrv, sizeof(cbrv));
  DEFINE_AND_SAVE_ORIGINS(cbrv)
  int joint_ret = pthread_join(pt, &cbrv);
  assert(joint_ret == 0);
  assert(cbrv == (void *)2);
  ASSERT_ZERO_LABEL(joint_ret);
  ASSERT_ZERO_LABEL(cbrv);
  ASSERT_SAVED_ORIGINS(cbrv);
}

// Tested by test_pthread_create().  This empty function is here to appease the
// check-wrappers script.
void test_pthread_join() {}

int dl_iterate_phdr_test_cb(struct dl_phdr_info *info, size_t size,
                            void *data) {
  assert(data == (void *)3);
  ASSERT_ZERO_LABEL(info);
  ASSERT_ZERO_LABEL(size);
  ASSERT_ZERO_LABEL(data);
  return 0;
}

void test_dl_iterate_phdr() {
  dl_iterate_phdr(dl_iterate_phdr_test_cb, (void *)3);
}

// On glibc < 2.27, this symbol is not available.  Mark it weak so we can skip
// testing in this case.
__attribute__((weak)) extern "C" void _dl_get_tls_static_info(size_t *sizep,
                                                              size_t *alignp);

void test__dl_get_tls_static_info() {
  if (!_dl_get_tls_static_info)
    return;
  size_t sizep = 0, alignp = 0;
  dfsan_set_label(i_label, &sizep, sizeof(sizep));
  dfsan_set_label(i_label, &alignp, sizeof(alignp));
  dfsan_origin sizep_o = dfsan_get_origin(sizep);
  dfsan_origin alignp_o = dfsan_get_origin(alignp);
#ifndef ORIGIN_TRACKING
  (void)sizep_o;
  (void)alignp_o;
#endif
  _dl_get_tls_static_info(&sizep, &alignp);
  ASSERT_ZERO_LABEL(sizep);
  ASSERT_ZERO_LABEL(alignp);
  ASSERT_ORIGIN(sizep, sizep_o);
  ASSERT_ORIGIN(alignp, alignp_o);
}

void test_strrchr() {
  char str1[] = "str1str1";

  char *p = str1;
  dfsan_set_label(j_label, &p, sizeof(p));

  char *rv = strrchr(p, 'r');
  assert(rv == &str1[6]);
  ASSERT_LABEL(rv, j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, p);

  char c = 'r';
  dfsan_set_label(k_label, &c, sizeof(c));
  rv = strrchr(str1, c);
  assert(rv == &str1[6]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, k_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, c);
#endif

  dfsan_set_label(i_label, &str1[7], 1);

  rv = strrchr(str1, 'r');
  assert(rv == &str1[6]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, i_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, str1[7]);
#endif
}

void test_strstr() {
  char str1[] = "str1str1";

  char *p1 = str1;
  dfsan_set_label(k_label, &p1, sizeof(p1));
  char *rv = strstr(p1, "1s");
  assert(rv == &str1[3]);
  ASSERT_LABEL(rv, k_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, p1);

  char str2[] = "1s";
  char *p2 = str2;
  dfsan_set_label(m_label, &p2, sizeof(p2));
  rv = strstr(str1, p2);
  assert(rv == &str1[3]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, m_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, p2);
#endif

  dfsan_set_label(n_label, &str2[0], 1);
  rv = strstr(str1, str2);
  assert(rv == &str1[3]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, n_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, str2[0]);
#endif

  dfsan_set_label(i_label, &str1[3], 1);
  dfsan_set_label(j_label, &str1[5], 1);

  rv = strstr(str1, "1s");
  assert(rv == &str1[3]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, i_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, str1[3]);
#endif

  rv = strstr(str1, "2s");
  assert(rv == NULL);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, i_j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, str1[3]);
#endif
}

void test_strpbrk() {
  char s[] = "abcdefg";
  char accept[] = "123fd";

  char *p_s = s;
  char *p_accept = accept;

  dfsan_set_label(n_label, &p_accept, sizeof(p_accept));

  char *rv = strpbrk(p_s, p_accept);
  assert(rv == &s[3]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, n_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, p_accept);
#endif

  dfsan_set_label(m_label, &p_s, sizeof(p_s));

  rv = strpbrk(p_s, p_accept);
  assert(rv == &s[3]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_LABEL(rv, m_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, p_s);
#else
  ASSERT_LABEL(rv, dfsan_union(m_label, n_label));
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, p_s);
#endif

  dfsan_set_label(i_label, &s[5], 1);
  dfsan_set_label(j_label, &accept[1], 1);

  rv = strpbrk(s, accept);
  assert(rv == &s[3]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, accept[1]);
#endif

  char *ps = s;
  dfsan_set_label(j_label, &ps, sizeof(ps));

  rv = strpbrk(ps, "123gf");
  assert(rv == &s[5]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_LABEL(rv, j_label);
#else
  ASSERT_LABEL(rv, i_j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, s[5]);
#endif

  rv = strpbrk(ps, "123");
  assert(rv == NULL);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(rv);
#else
  ASSERT_LABEL(rv, i_j_label);
  ASSERT_INIT_ORIGIN_EQ_ORIGIN(&rv, s[5]);
#endif
}

void test_memchr() {
  char str1[] = "str1";
  dfsan_set_label(i_label, &str1[3], 1);
  dfsan_set_label(j_label, &str1[4], 1);

  char *crv = (char *) memchr(str1, 'r', sizeof(str1));
  assert(crv == &str1[2]);
  ASSERT_ZERO_LABEL(crv);

  char c = 'r';
  dfsan_set_label(k_label, &c, sizeof(c));
  crv = (char *)memchr(str1, c, sizeof(str1));
  assert(crv == &str1[2]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(crv);
#else
  ASSERT_LABEL(crv, k_label);
  ASSERT_EQ_ORIGIN(crv, c);
#endif

  char *ptr = str1;
  dfsan_set_label(k_label, &ptr, sizeof(ptr));
  crv = (char *)memchr(ptr, 'r', sizeof(str1));
  assert(crv == &str1[2]);
  ASSERT_LABEL(crv, k_label);
  ASSERT_EQ_ORIGIN(crv, ptr);

  crv = (char *) memchr(str1, '1', sizeof(str1));
  assert(crv == &str1[3]);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(crv);
#else
  ASSERT_LABEL(crv, i_label);
  ASSERT_EQ_ORIGIN(crv, str1[3]);
#endif

  crv = (char *) memchr(str1, 'x', sizeof(str1));
  assert(!crv);
#ifdef STRICT_DATA_DEPENDENCIES
  ASSERT_ZERO_LABEL(crv);
#else
  ASSERT_LABEL(crv, i_j_label);
  ASSERT_EQ_ORIGIN(crv, str1[3]);
#endif
}

void alarm_handler(int unused) {
  ;
}

void test_nanosleep() {
  struct timespec req, rem;
  req.tv_sec = 1;
  req.tv_nsec = 0;
  dfsan_set_label(i_label, &rem, sizeof(rem));
  DEFINE_AND_SAVE_ORIGINS(rem)

  // non interrupted
  int rv = nanosleep(&req, &rem);
  assert(rv == 0);
  ASSERT_ZERO_LABEL(rv);
  ASSERT_READ_LABEL(&rem, 1, i_label);
  ASSERT_SAVED_ORIGINS(rem)

  // interrupted by an alarm
  signal(SIGALRM, alarm_handler);
  req.tv_sec = 3;
  alarm(1);
  rv = nanosleep(&req, &rem);
  assert(rv == -1);
  ASSERT_ZERO_LABEL(rv);
  ASSERT_READ_ZERO_LABEL(&rem, sizeof(rem));
  ASSERT_SAVED_ORIGINS(rem)
}

void test_socketpair() {
  int fd[2];
  dfsan_origin fd_o[2];

  dfsan_set_label(i_label, fd, sizeof(fd));
  fd_o[0] = dfsan_get_origin((long)(fd[0]));
  fd_o[1] = dfsan_get_origin((long)(fd[1]));
  int rv = socketpair(PF_LOCAL, SOCK_STREAM, 0, fd);
  assert(rv == 0);
  ASSERT_ZERO_LABEL(rv);
  ASSERT_READ_ZERO_LABEL(fd, sizeof(fd));
  ASSERT_ORIGIN(fd[0], fd_o[0]);
  ASSERT_ORIGIN(fd[1], fd_o[1]);
}

void test_getpeername() {
  int sockfds[2];
  int ret = socketpair(AF_UNIX, SOCK_DGRAM, 0, sockfds);
  assert(ret != -1);

  struct sockaddr addr = {};
  socklen_t addrlen = sizeof(addr);
  dfsan_set_label(i_label, &addr, addrlen);
  dfsan_set_label(i_label, &addrlen, sizeof(addrlen));
  DEFINE_AND_SAVE_ORIGINS(addr)
  DEFINE_AND_SAVE_ORIGINS(addrlen)

  ret = getpeername(sockfds[0], &addr, &addrlen);
  assert(ret != -1);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_ZERO_LABEL(addrlen);
  assert(addrlen < sizeof(addr));
  ASSERT_READ_ZERO_LABEL(&addr, addrlen);
  ASSERT_READ_LABEL(((char *)&addr) + addrlen, 1, i_label);
  ASSERT_SAVED_ORIGINS(addr)
  ASSERT_SAVED_ORIGINS(addrlen)

  close(sockfds[0]);
  close(sockfds[1]);
}

void test_getsockname() {
  int sockfd = socket(AF_UNIX, SOCK_DGRAM, 0);
  assert(sockfd != -1);

  struct sockaddr addr = {};
  socklen_t addrlen = sizeof(addr);
  dfsan_set_label(i_label, &addr, addrlen);
  dfsan_set_label(i_label, &addrlen, sizeof(addrlen));
  DEFINE_AND_SAVE_ORIGINS(addr)
  DEFINE_AND_SAVE_ORIGINS(addrlen)
  int ret = getsockname(sockfd, &addr, &addrlen);
  assert(ret != -1);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_ZERO_LABEL(addrlen);
  assert(addrlen < sizeof(addr));
  ASSERT_READ_ZERO_LABEL(&addr, addrlen);
  ASSERT_READ_LABEL(((char *)&addr) + addrlen, 1, i_label);
  ASSERT_SAVED_ORIGINS(addr)
  ASSERT_SAVED_ORIGINS(addrlen)

  close(sockfd);
}

void test_getsockopt() {
  int sockfd = socket(AF_UNIX, SOCK_DGRAM, 0);
  assert(sockfd != -1);

  int optval[2] = {-1, -1};
  socklen_t optlen = sizeof(optval);
  dfsan_set_label(i_label, &optval, sizeof(optval));
  dfsan_set_label(i_label, &optlen, sizeof(optlen));
  DEFINE_AND_SAVE_ORIGINS(optval)
  DEFINE_AND_SAVE_ORIGINS(optlen)
  int ret = getsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &optval, &optlen);
  assert(ret != -1);
  assert(optlen == sizeof(int));
  assert(optval[0] == 0);
  assert(optval[1] == -1);
  ASSERT_ZERO_LABEL(ret);
  ASSERT_ZERO_LABEL(optlen);
  ASSERT_ZERO_LABEL(optval[0]);
  ASSERT_LABEL(optval[1], i_label);
  ASSERT_SAVED_ORIGINS(optval)
  ASSERT_SAVED_ORIGINS(optlen)

  close(sockfd);
}

void test_write() {
  int fd = open("/dev/null", O_WRONLY);

  char buf[] = "a string";
  int len = strlen(buf);

  // The result of a write always unlabeled.
  int res = write(fd, buf, len);
  assert(res > 0);
  ASSERT_ZERO_LABEL(res);

  // Label all arguments to write().
  dfsan_set_label(i_label, &(buf[3]), 1);
  dfsan_set_label(j_label, &fd, sizeof(fd));
  dfsan_set_label(i_label, &len, sizeof(len));

  // The value returned by write() should have no label.
  res = write(fd, buf, len);
  ASSERT_ZERO_LABEL(res);

  close(fd);
}

template <class T>
void test_sprintf_chunk(const char* expected, const char* format, T arg) {
  char buf[512];
  memset(buf, 'a', sizeof(buf));

  char padded_expected[512];
  strcpy(padded_expected, "foo ");
  strcat(padded_expected, expected);
  strcat(padded_expected, " bar");

  char padded_format[512];
  strcpy(padded_format, "foo ");
  strcat(padded_format, format);
  strcat(padded_format, " bar");

  // Non labelled arg.
  assert(sprintf(buf, padded_format,  arg) == strlen(padded_expected));
  assert(strcmp(buf, padded_expected) == 0);
  ASSERT_READ_LABEL(buf, strlen(padded_expected), 0);
  memset(buf, 'a', sizeof(buf));

  // Labelled arg.
  dfsan_set_label(i_label, &arg, sizeof(arg));
  dfsan_origin a_o = dfsan_get_origin((long)(arg));
#ifndef ORIGIN_TRACKING
  (void)a_o;
#endif
  assert(sprintf(buf, padded_format,  arg) == strlen(padded_expected));
  assert(strcmp(buf, padded_expected) == 0);
  ASSERT_READ_LABEL(buf, 4, 0);
  ASSERT_READ_LABEL(buf + 4, strlen(padded_expected) - 8, i_label);
  ASSERT_INIT_ORIGINS(buf + 4, strlen(padded_expected) - 8, a_o);
  ASSERT_READ_LABEL(buf + (strlen(padded_expected) - 4), 4, 0);
}

void test_sprintf() {
  char buf[2048];
  memset(buf, 'a', sizeof(buf));

  // Test formatting (no conversion specifier).
  assert(sprintf(buf, "Hello world!") == 12);
  assert(strcmp(buf, "Hello world!") == 0);
  ASSERT_READ_LABEL(buf, sizeof(buf), 0);

  // Test for extra arguments.
  assert(sprintf(buf, "Hello world!", 42, "hello") == 12);
  assert(strcmp(buf, "Hello world!") == 0);
  ASSERT_READ_LABEL(buf, sizeof(buf), 0);

  // Test formatting & label propagation (multiple conversion specifiers): %s,
  // %d, %n, %f, and %%.
  const char* s = "world";
  int m = 8;
  int d = 27;
  dfsan_set_label(k_label, (void *) (s + 1), 2);
  dfsan_origin s_o = dfsan_get_origin((long)(s[1]));
  dfsan_set_label(i_label, &m, sizeof(m));
  dfsan_origin m_o = dfsan_get_origin((long)m);
  dfsan_set_label(j_label, &d, sizeof(d));
  dfsan_origin d_o = dfsan_get_origin((long)d);
#ifndef ORIGIN_TRACKING
  (void)s_o;
  (void)m_o;
  (void)d_o;
#endif
  int n;
  int r = sprintf(buf, "hello %s, %-d/%d/%d %f %% %n%d", s, 2014, m, d,
                  12345.6781234, &n, 1000);
  assert(r == 42);
  assert(strcmp(buf, "hello world, 2014/8/27 12345.678123 % 1000") == 0);
  ASSERT_READ_LABEL(buf, 7, 0);
  ASSERT_READ_LABEL(buf + 7, 2, k_label);
  ASSERT_INIT_ORIGINS(buf + 7, 2, s_o);
  ASSERT_READ_LABEL(buf + 9, 9, 0);
  ASSERT_READ_LABEL(buf + 18, 1, i_label);
  ASSERT_INIT_ORIGINS(buf + 18, 1, m_o);
  ASSERT_READ_LABEL(buf + 19, 1, 0);
  ASSERT_READ_LABEL(buf + 20, 2, j_label);
  ASSERT_INIT_ORIGINS(buf + 20, 2, d_o);
  ASSERT_READ_LABEL(buf + 22, 15, 0);
  ASSERT_LABEL(r, 0);
  assert(n == 38);

  // Test formatting & label propagation (single conversion specifier, with
  // additional length and precision modifiers).
  test_sprintf_chunk("-559038737", "%d", 0xdeadbeef);
  test_sprintf_chunk("3735928559", "%u", 0xdeadbeef);
  test_sprintf_chunk("12345", "%i", 12345);
  test_sprintf_chunk("751", "%o", 0751);
  test_sprintf_chunk("babe", "%x", 0xbabe);
  test_sprintf_chunk("0000BABE", "%.8X", 0xbabe);
  test_sprintf_chunk("-17", "%hhd", 0xdeadbeef);
  test_sprintf_chunk("-16657", "%hd", 0xdeadbeef);
  test_sprintf_chunk("deadbeefdeadbeef", "%lx", 0xdeadbeefdeadbeef);
  test_sprintf_chunk("0xdeadbeefdeadbeef", "%p",
                 (void *)  0xdeadbeefdeadbeef);
  test_sprintf_chunk("18446744073709551615", "%ju", (intmax_t) -1);
  test_sprintf_chunk("18446744073709551615", "%zu", (size_t) -1);
  test_sprintf_chunk("18446744073709551615", "%tu", (size_t) -1);

  test_sprintf_chunk("0x1.f9acffa7eb6bfp-4", "%a", 0.123456);
  test_sprintf_chunk("0X1.F9ACFFA7EB6BFP-4", "%A", 0.123456);
  test_sprintf_chunk("0.12346", "%.5f", 0.123456);
  test_sprintf_chunk("0.123456", "%g", 0.123456);
  test_sprintf_chunk("1.234560e-01", "%e", 0.123456);
  test_sprintf_chunk("1.234560E-01", "%E", 0.123456);
  test_sprintf_chunk("0.1234567891234560", "%.16Lf",
                     (long double) 0.123456789123456);

  test_sprintf_chunk("z", "%c", 'z');

  // %n, %s, %d, %f, and %% already tested

  // Test formatting with width passed as an argument.
  r = sprintf(buf, "hi %*d my %*s friend %.*f", 3, 1, 6, "dear", 4, 3.14159265359);
  assert(r == 30);
  assert(strcmp(buf, "hi   1 my   dear friend 3.1416") == 0);
}

void test_snprintf() {
  char buf[2048];
  memset(buf, 'a', sizeof(buf));
  dfsan_set_label(0, buf, sizeof(buf));
  const char* s = "world";
  int y = 2014;
  int m = 8;
  int d = 27;
  dfsan_set_label(k_label, (void *) (s + 1), 2);
  dfsan_origin s_o = dfsan_get_origin((long)(s[1]));
  dfsan_set_label(i_label, &y, sizeof(y));
  dfsan_origin y_o = dfsan_get_origin((long)y);
  dfsan_set_label(j_label, &m, sizeof(m));
  dfsan_origin m_o = dfsan_get_origin((long)m);
#ifndef ORIGIN_TRACKING
  (void)s_o;
  (void)y_o;
  (void)m_o;
#endif
  int r = snprintf(buf, 19, "hello %s, %-d/   %d/%d %f", s, y, m, d,
                   12345.6781234);
  // The return value is the number of bytes that would have been written to
  // the final string if enough space had been available.
  assert(r == 38);
  assert(memcmp(buf, "hello world, 2014/", 19) == 0);
  ASSERT_READ_LABEL(buf, 7, 0);
  ASSERT_READ_LABEL(buf + 7, 2, k_label);
  ASSERT_INIT_ORIGINS(buf + 7, 2, s_o);
  ASSERT_READ_LABEL(buf + 9, 4, 0);
  ASSERT_READ_LABEL(buf + 13, 4, i_label);
  ASSERT_INIT_ORIGINS(buf + 13, 4, y_o);
  ASSERT_READ_LABEL(buf + 17, 2, 0);
  ASSERT_LABEL(r, 0);
}

// Tested by a seperate source file.  This empty function is here to appease the
// check-wrappers script.
void test_fork() {}

int main(void) {
  i_label = 1;
  j_label = 2;
  k_label = 4;
  m_label = 8;
  n_label = 16;
  i_j_label = dfsan_union(i_label, j_label);
  assert(i_j_label != i_label);
  assert(i_j_label != j_label);
  assert(i_j_label != k_label);

  test__dl_get_tls_static_info();
  test_bcmp();
  test_clock_gettime();
  test_ctime_r();
  test_dfsan_set_write_callback();
  test_dl_iterate_phdr();
  test_dlopen();
  test_epoll_wait();
  test_fgets();
  test_fork();
  test_fstat();
  test_get_current_dir_name();
  test_getcwd();
  test_getentropy();
  test_gethostname();
  test_getpeername();
  test_getpwuid_r();
  test_getrlimit();
  test_getrusage();
  test_getsockname();
  test_getsockopt();
  test_gettimeofday();
  test_inet_pton();
  test_localtime_r();
  test_memchr();
  test_memcmp();
  test_memcpy();
  test_memmove();
  test_memset();
  test_nanosleep();
  test_poll();
  test_pread();
  test_pthread_create();
  test_pthread_join();
  test_read();
  test_recvmmsg();
  test_recvmsg();
  test_sched_getaffinity();
  test_select();
  test_sigaction();
  test_signal();
  test_sigaltstack();
  test_sigemptyset();
  test_snprintf();
  test_socketpair();
  test_sprintf();
  test_stat();
  test_strcasecmp();
  test_strchr();
  test_strcmp();
  test_strcat();
  test_strcpy();
  test_strdup();
  test_strlen();
  test_strncasecmp();
  test_strncmp();
  test_strncpy();
  test_strpbrk();
  test_strrchr();
  test_strstr();
  test_strtod();
  test_strtol();
  test_strtoll();
  test_strtoul();
  test_strtoull();
  test_time();
  test_write();
}
