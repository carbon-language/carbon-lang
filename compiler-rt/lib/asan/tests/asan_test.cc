//===-- asan_test.cc ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <setjmp.h>
#include <assert.h>

#if defined(__i386__) or defined(__x86_64__)
#include <emmintrin.h>
#endif

#include "asan_test_config.h"
#include "asan_test_utils.h"

#ifndef __APPLE__
#include <malloc.h>
#endif  // __APPLE__

#ifdef __APPLE__
static bool APPLE = true;
#else
static bool APPLE = false;
#endif

#if ASAN_HAS_EXCEPTIONS
# define ASAN_THROW(x) throw (x)
#else
# define ASAN_THROW(x)
#endif

#include <sys/mman.h>

typedef uint8_t   U1;
typedef uint16_t  U2;
typedef uint32_t  U4;
typedef uint64_t  U8;

static const char *progname;
static const int kPageSize = 4096;

// Simple stand-alone pseudorandom number generator.
// Current algorithm is ANSI C linear congruential PRNG.
static inline uint32_t my_rand(uint32_t* state) {
  return (*state = *state * 1103515245 + 12345) >> 16;
}

static uint32_t global_seed = 0;

class ObjdumpOfMyself {
 public:
  explicit ObjdumpOfMyself(const string &binary) {
    is_correct = true;
    string objdump_name = APPLE ? "gobjdump" : "objdump";
    string prog = objdump_name + " -d " + binary;
    // TODO(glider): popen() succeeds even if the file does not exist.
    FILE *pipe = popen(prog.c_str(), "r");
    string objdump;
    if (pipe) {
      const int kBuffSize = 4096;
      char buff[kBuffSize+1];
      int read_bytes;
      while ((read_bytes = fread(buff, 1, kBuffSize, pipe)) > 0) {
        buff[read_bytes] = 0;
        objdump.append(buff);
      }
      pclose(pipe);
    } else {
      is_correct = false;
    }
    // cut the objdump into functions
    string fn, next_fn;
    size_t next_start;
    for (size_t start = fn_start(objdump, 0, &fn);
         start != string::npos;
         start = next_start, fn = next_fn) {
      next_start = fn_start(objdump, start, &next_fn);
      // fprintf(stderr, "start: %d next_start = %d fn: %s\n",
      //        (int)start, (int)next_start, fn.c_str());
      // Mac OS adds the "_" prefix to function names.
      if (fn.find(APPLE ? "_Disasm" : "Disasm") == string::npos) {
        continue;
      }
      string fn_body = objdump.substr(start, next_start - start);
      // fprintf(stderr, "%s:\n%s", fn.c_str(), fn_body.c_str());
      functions_[fn] = fn_body;
    }
  }

  string &GetFuncDisasm(const string &fn) {
    return functions_[fn];
  }

  int CountInsnInFunc(const string &fn, const vector<string> &insns) {
    // Mac OS adds the "_" prefix to function names.
    string fn_ref = APPLE ? "_" + fn : fn;
    const string &disasm = GetFuncDisasm(fn_ref);
    if (disasm.empty()) return -1;
    size_t counter = 0;
    for (size_t i = 0; i < insns.size(); i++) {
      size_t pos = 0;
      while ((pos = disasm.find(insns[i], pos)) != string::npos) {
        counter++;
        pos++;
      }
    }
    return counter;
  }

  bool IsCorrect() { return is_correct; }

 private:
  size_t fn_start(const string &objdump, size_t start_pos, string *fn) {
    size_t pos = objdump.find(">:\n", start_pos);
    if (pos == string::npos)
      return string::npos;
    size_t beg = pos;
    while (beg > 0 && objdump[beg - 1] != '<')
      beg--;
    *fn = objdump.substr(beg, pos - beg);
    return pos + 3;
  }

  map<string, string> functions_;
  bool is_correct;
};

static ObjdumpOfMyself *objdump_of_myself() {
  static ObjdumpOfMyself *o = new ObjdumpOfMyself(progname);
  return o;
}

const size_t kLargeMalloc = 1 << 24;

template<class T>
__attribute__((noinline))
void asan_write(T *a) {
  *a = 0;
}

__attribute__((noinline))
void asan_write_sized_aligned(uint8_t *p, size_t size) {
  EXPECT_EQ(0, ((uintptr_t)p % size));
  if      (size == 1) asan_write((uint8_t*)p);
  else if (size == 2) asan_write((uint16_t*)p);
  else if (size == 4) asan_write((uint32_t*)p);
  else if (size == 8) asan_write((uint64_t*)p);
}

__attribute__((noinline)) void *malloc_fff(size_t size) {
  void *res = malloc/**/(size); break_optimization(0); return res;}
__attribute__((noinline)) void *malloc_eee(size_t size) {
  void *res = malloc_fff(size); break_optimization(0); return res;}
__attribute__((noinline)) void *malloc_ddd(size_t size) {
  void *res = malloc_eee(size); break_optimization(0); return res;}
__attribute__((noinline)) void *malloc_ccc(size_t size) {
  void *res = malloc_ddd(size); break_optimization(0); return res;}
__attribute__((noinline)) void *malloc_bbb(size_t size) {
  void *res = malloc_ccc(size); break_optimization(0); return res;}
__attribute__((noinline)) void *malloc_aaa(size_t size) {
  void *res = malloc_bbb(size); break_optimization(0); return res;}

#ifndef __APPLE__
__attribute__((noinline)) void *memalign_fff(size_t alignment, size_t size) {
  void *res = memalign/**/(alignment, size); break_optimization(0); return res;}
__attribute__((noinline)) void *memalign_eee(size_t alignment, size_t size) {
  void *res = memalign_fff(alignment, size); break_optimization(0); return res;}
__attribute__((noinline)) void *memalign_ddd(size_t alignment, size_t size) {
  void *res = memalign_eee(alignment, size); break_optimization(0); return res;}
__attribute__((noinline)) void *memalign_ccc(size_t alignment, size_t size) {
  void *res = memalign_ddd(alignment, size); break_optimization(0); return res;}
__attribute__((noinline)) void *memalign_bbb(size_t alignment, size_t size) {
  void *res = memalign_ccc(alignment, size); break_optimization(0); return res;}
__attribute__((noinline)) void *memalign_aaa(size_t alignment, size_t size) {
  void *res = memalign_bbb(alignment, size); break_optimization(0); return res;}
#endif  // __APPLE__


__attribute__((noinline))
  void free_ccc(void *p) { free(p); break_optimization(0);}
__attribute__((noinline))
  void free_bbb(void *p) { free_ccc(p); break_optimization(0);}
__attribute__((noinline))
  void free_aaa(void *p) { free_bbb(p); break_optimization(0);}

template<class T>
__attribute__((noinline))
void oob_test(int size, int off) {
  char *p = (char*)malloc_aaa(size);
  // fprintf(stderr, "writing %d byte(s) into [%p,%p) with offset %d\n",
  //        sizeof(T), p, p + size, off);
  asan_write((T*)(p + off));
  free_aaa(p);
}


template<class T>
__attribute__((noinline))
void uaf_test(int size, int off) {
  char *p = (char *)malloc_aaa(size);
  free_aaa(p);
  for (int i = 1; i < 100; i++)
    free_aaa(malloc_aaa(i));
  fprintf(stderr, "writing %ld byte(s) at %p with offset %d\n",
          (long)sizeof(T), p, off);
  asan_write((T*)(p + off));
}

TEST(AddressSanitizer, ADDRESS_SANITIZER_MacroTest) {
  EXPECT_EQ(1, ADDRESS_SANITIZER);
}

TEST(AddressSanitizer, SimpleDeathTest) {
  EXPECT_DEATH(exit(1), "");
}

TEST(AddressSanitizer, VariousMallocsTest) {
  // fprintf(stderr, "malloc:\n");
  int *a = (int*)malloc(100 * sizeof(int));
  a[50] = 0;
  free(a);

  // fprintf(stderr, "realloc:\n");
  int *r = (int*)malloc(10);
  r = (int*)realloc(r, 2000 * sizeof(int));
  r[1000] = 0;
  free(r);

  // fprintf(stderr, "operator new []\n");
  int *b = new int[100];
  b[50] = 0;
  delete [] b;

  // fprintf(stderr, "operator new\n");
  int *c = new int;
  *c = 0;
  delete c;

#ifndef __APPLE__
  // cfree
  cfree(Ident(malloc(1)));

  // fprintf(stderr, "posix_memalign\n");
  int *pm;
  int pm_res = posix_memalign((void**)&pm, kPageSize, kPageSize);
  EXPECT_EQ(0, pm_res);
  free(pm);

  int *ma = (int*)memalign(kPageSize, kPageSize);
  EXPECT_EQ(0, (uintptr_t)ma % kPageSize);
  ma[123] = 0;
  free(ma);
#endif  // __APPLE__
}

TEST(AddressSanitizer, CallocTest) {
  int *a = (int*)calloc(100, sizeof(int));
  EXPECT_EQ(0, a[10]);
  free(a);
}

TEST(AddressSanitizer, VallocTest) {
  void *a = valloc(100);
  EXPECT_EQ(0, (uintptr_t)a % kPageSize);
  free(a);
}

#ifndef __APPLE__
TEST(AddressSanitizer, PvallocTest) {
  char *a = (char*)pvalloc(kPageSize + 100);
  EXPECT_EQ(0, (uintptr_t)a % kPageSize);
  a[kPageSize + 101] = 1;  // we should not report an error here.
  free(a);

  a = (char*)pvalloc(0);  // pvalloc(0) should allocate at least one page.
  EXPECT_EQ(0, (uintptr_t)a % kPageSize);
  a[101] = 1;  // we should not report an error here.
  free(a);
}
#endif  // __APPLE__

void NoOpSignalHandler(int unused) {
  fprintf(stderr, "NoOpSignalHandler (should not happen). Aborting\n");
  abort();
}

void NoOpSigaction(int, siginfo_t *siginfo, void *context) {
  fprintf(stderr, "NoOpSigaction (should not happen). Aborting\n");
  abort();
}

TEST(AddressSanitizer, SignalTest) {
  signal(SIGSEGV, NoOpSignalHandler);
  signal(SIGILL, NoOpSignalHandler);
  // If asan did not intercept sigaction NoOpSigaction will fire.
  char *x = Ident((char*)malloc(5));
  EXPECT_DEATH(x[6]++, "is located 1 bytes to the right");
  free(Ident(x));
}

TEST(AddressSanitizer, SigactionTest) {
  {
    struct sigaction sigact;
    memset(&sigact, 0, sizeof(sigact));
    sigact.sa_sigaction = NoOpSigaction;;
    sigact.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sigact, 0);
  }

  {
    struct sigaction sigact;
    memset(&sigact, 0, sizeof(sigact));
    sigact.sa_sigaction = NoOpSigaction;;
    sigact.sa_flags = SA_SIGINFO;
    sigaction(SIGILL, &sigact, 0);
  }

  // If asan did not intercept sigaction NoOpSigaction will fire.
  char *x = Ident((char*)malloc(5));
  EXPECT_DEATH(x[6]++, "is located 1 bytes to the right");
  free(Ident(x));
}

void *TSDWorker(void *test_key) {
  if (test_key) {
    pthread_setspecific(*(pthread_key_t*)test_key, (void*)0xfeedface);
  }
  return NULL;
}

void TSDDestructor(void *tsd) {
  // Spawning a thread will check that the current thread id is not -1.
  pthread_t th;
  pthread_create(&th, NULL, TSDWorker, NULL);
  pthread_join(th, NULL);
}

// This tests triggers the thread-specific data destruction fiasco which occurs
// if we don't manage the TSD destructors ourselves. We create a new pthread
// key with a non-NULL destructor which is likely to be put after the destructor
// of AsanThread in the list of destructors.
// In this case the TSD for AsanThread will be destroyed before TSDDestructor
// is called for the child thread, and a CHECK will fail when we call
// pthread_create() to spawn the grandchild.
TEST(AddressSanitizer, DISABLED_TSDTest) {
  pthread_t th;
  pthread_key_t test_key;
  pthread_key_create(&test_key, TSDDestructor);
  pthread_create(&th, NULL, TSDWorker, &test_key);
  pthread_join(th, NULL);
  pthread_key_delete(test_key);
}

template<class T>
void OOBTest() {
  char expected_str[100];
  for (int size = sizeof(T); size < 20; size += 5) {
    for (int i = -5; i < 0; i++) {
      const char *str =
          "is located.*%d byte.*to the left";
      sprintf(expected_str, str, abs(i));
      EXPECT_DEATH(oob_test<T>(size, i), expected_str);
    }

    for (int i = 0; i < size - sizeof(T) + 1; i++)
      oob_test<T>(size, i);

    for (int i = size - sizeof(T) + 1; i <= size + 3 * sizeof(T); i++) {
      const char *str =
          "is located.*%d byte.*to the right";
      int off = i >= size ? (i - size) : 0;
      // we don't catch unaligned partially OOB accesses.
      if (i % sizeof(T)) continue;
      sprintf(expected_str, str, off);
      EXPECT_DEATH(oob_test<T>(size, i), expected_str);
    }
  }

  EXPECT_DEATH(oob_test<T>(kLargeMalloc, -1),
          "is located.*1 byte.*to the left");
  EXPECT_DEATH(oob_test<T>(kLargeMalloc, kLargeMalloc),
          "is located.*0 byte.*to the right");
}

// TODO(glider): the following tests are EXTREMELY slow on Darwin:
//   AddressSanitizer.OOB_char (125503 ms)
//   AddressSanitizer.OOB_int (126890 ms)
//   AddressSanitizer.OOBRightTest (315605 ms)
//   AddressSanitizer.SimpleStackTest (366559 ms)

TEST(AddressSanitizer, OOB_char) {
  OOBTest<U1>();
}

TEST(AddressSanitizer, OOB_int) {
  OOBTest<U4>();
}

TEST(AddressSanitizer, OOBRightTest) {
  for (size_t access_size = 1; access_size <= 8; access_size *= 2) {
    for (size_t alloc_size = 1; alloc_size <= 8; alloc_size++) {
      for (size_t offset = 0; offset <= 8; offset += access_size) {
        void *p = malloc(alloc_size);
        // allocated: [p, p + alloc_size)
        // accessed:  [p + offset, p + offset + access_size)
        uint8_t *addr = (uint8_t*)p + offset;
        if (offset + access_size <= alloc_size) {
          asan_write_sized_aligned(addr, access_size);
        } else {
          int outside_bytes = offset > alloc_size ? (offset - alloc_size) : 0;
          const char *str =
              "is located.%d *byte.*to the right";
          char expected_str[100];
          sprintf(expected_str, str, outside_bytes);
          EXPECT_DEATH(asan_write_sized_aligned(addr, access_size),
                       expected_str);
        }
        free(p);
      }
    }
  }
}

TEST(AddressSanitizer, UAF_char) {
  const char *uaf_string = "AddressSanitizer.*heap-use-after-free";
  EXPECT_DEATH(uaf_test<U1>(1, 0), uaf_string);
  EXPECT_DEATH(uaf_test<U1>(10, 0), uaf_string);
  EXPECT_DEATH(uaf_test<U1>(10, 10), uaf_string);
  EXPECT_DEATH(uaf_test<U1>(kLargeMalloc, 0), uaf_string);
  EXPECT_DEATH(uaf_test<U1>(kLargeMalloc, kLargeMalloc / 2), uaf_string);
}

#if ASAN_HAS_BLACKLIST
TEST(AddressSanitizer, IgnoreTest) {
  int *x = Ident(new int);
  delete Ident(x);
  *x = 0;
}
#endif  // ASAN_HAS_BLACKLIST

struct StructWithBitField {
  int bf1:1;
  int bf2:1;
  int bf3:1;
  int bf4:29;
};

TEST(AddressSanitizer, BitFieldPositiveTest) {
  StructWithBitField *x = new StructWithBitField;
  delete Ident(x);
  EXPECT_DEATH(x->bf1 = 0, "use-after-free");
  EXPECT_DEATH(x->bf2 = 0, "use-after-free");
  EXPECT_DEATH(x->bf3 = 0, "use-after-free");
  EXPECT_DEATH(x->bf4 = 0, "use-after-free");
};

struct StructWithBitFields_8_24 {
  int a:8;
  int b:24;
};

TEST(AddressSanitizer, BitFieldNegativeTest) {
  StructWithBitFields_8_24 *x = Ident(new StructWithBitFields_8_24);
  x->a = 0;
  x->b = 0;
  delete Ident(x);
}

TEST(AddressSanitizer, OutOfMemoryTest) {
  size_t size = __WORDSIZE == 64 ? (size_t)(1ULL << 48) : (0xf0000000);
  EXPECT_EQ(0, realloc(0, size));
  EXPECT_EQ(0, realloc(0, ~Ident(0)));
  EXPECT_EQ(0, malloc(size));
  EXPECT_EQ(0, malloc(~Ident(0)));
  EXPECT_EQ(0, calloc(1, size));
  EXPECT_EQ(0, calloc(1, ~Ident(0)));
}

#if ASAN_NEEDS_SEGV
TEST(AddressSanitizer, WildAddressTest) {
  char *c = (char*)0x123;
  EXPECT_DEATH(*c = 0, "AddressSanitizer crashed on unknown address");
}
#endif

static void MallocStress(size_t n) {
  uint32_t seed = my_rand(&global_seed);
  for (size_t iter = 0; iter < 10; iter++) {
    vector<void *> vec;
    for (size_t i = 0; i < n; i++) {
      if ((i % 3) == 0) {
        if (vec.empty()) continue;
        size_t idx = my_rand(&seed) % vec.size();
        void *ptr = vec[idx];
        vec[idx] = vec.back();
        vec.pop_back();
        free_aaa(ptr);
      } else {
        size_t size = my_rand(&seed) % 1000 + 1;
#ifndef __APPLE__
        size_t alignment = 1 << (my_rand(&seed) % 7 + 3);
        char *ptr = (char*)memalign_aaa(alignment, size);
#else
        char *ptr = (char*) malloc_aaa(size);
#endif
        vec.push_back(ptr);
        ptr[0] = 0;
        ptr[size-1] = 0;
        ptr[size/2] = 0;
      }
    }
    for (size_t i = 0; i < vec.size(); i++)
      free_aaa(vec[i]);
  }
}

TEST(AddressSanitizer, MallocStressTest) {
  MallocStress(200000);
}

static void TestLargeMalloc(size_t size) {
  char buff[1024];
  sprintf(buff, "is located 1 bytes to the left of %lu-byte", (long)size);
  EXPECT_DEATH(Ident((char*)malloc(size))[-1] = 0, buff);
}

TEST(AddressSanitizer, LargeMallocTest) {
  for (int i = 113; i < (1 << 28); i = i * 2 + 13) {
    TestLargeMalloc(i);
  }
}

TEST(AddressSanitizer, HugeMallocTest) {
#ifdef __APPLE__
  // It was empirically found out that 1215 megabytes is the maximum amount of
  // memory available to the process under AddressSanitizer on Darwin.
  // (the libSystem malloc() allows allocating up to 2300 megabytes without
  // ASan).
  size_t n_megs = __WORDSIZE == 32 ? 1200 : 4100;
#else
  size_t n_megs = __WORDSIZE == 32 ? 2600 : 4100;
#endif
  TestLargeMalloc(n_megs << 20);
}

TEST(AddressSanitizer, ThreadedMallocStressTest) {
  const int kNumThreads = 4;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    pthread_create(&t[i], 0, (void* (*)(void *x))MallocStress, (void*)100000);
  }
  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(t[i], 0);
  }
}

void *ManyThreadsWorker(void *a) {
  for (int iter = 0; iter < 100; iter++) {
    for (size_t size = 100; size < 2000; size *= 2) {
      free(Ident(malloc(size)));
    }
  }
  return 0;
}

TEST(AddressSanitizer, ManyThreadsTest) {
  const size_t kNumThreads = __WORDSIZE == 32 ? 150 : 1000;
  pthread_t t[kNumThreads];
  for (size_t i = 0; i < kNumThreads; i++) {
    pthread_create(&t[i], 0, (void* (*)(void *x))ManyThreadsWorker, (void*)i);
  }
  for (size_t i = 0; i < kNumThreads; i++) {
    pthread_join(t[i], 0);
  }
}

TEST(AddressSanitizer, ReallocTest) {
  const int kMinElem = 5;
  int *ptr = (int*)malloc(sizeof(int) * kMinElem);
  ptr[3] = 3;
  for (int i = 0; i < 10000; i++) {
    ptr = (int*)realloc(ptr,
        (my_rand(&global_seed) % 1000 + kMinElem) * sizeof(int));
    EXPECT_EQ(3, ptr[3]);
  }
}

void WrongFree() {
  int *x = (int*)malloc(100 * sizeof(int));
  // Use the allocated memory, otherwise Clang will optimize it out.
  Ident(x);
  free(x + 1);
}

TEST(AddressSanitizer, WrongFreeTest) {
  EXPECT_DEATH(WrongFree(), "attempting free.*not malloc");
}

void DoubleFree() {
  int *x = (int*)malloc(100 * sizeof(int));
  fprintf(stderr, "DoubleFree: x=%p\n", x);
  free(x);
  free(x);
  fprintf(stderr, "should have failed in the second free(%p)\n", x);
  abort();
}

TEST(AddressSanitizer, DoubleFreeTest) {
  EXPECT_DEATH(DoubleFree(), "attempting double-free");
}

template<int kSize>
__attribute__((noinline))
void SizedStackTest() {
  char a[kSize];
  char  *A = Ident((char*)&a);
  for (size_t i = 0; i < kSize; i++)
    A[i] = i;
  EXPECT_DEATH(A[-1] = 0, "");
  EXPECT_DEATH(A[-20] = 0, "");
  EXPECT_DEATH(A[-31] = 0, "");
  EXPECT_DEATH(A[kSize] = 0, "");
  EXPECT_DEATH(A[kSize + 1] = 0, "");
  EXPECT_DEATH(A[kSize + 10] = 0, "");
  EXPECT_DEATH(A[kSize + 31] = 0, "");
}

TEST(AddressSanitizer, SimpleStackTest) {
  SizedStackTest<1>();
  SizedStackTest<2>();
  SizedStackTest<3>();
  SizedStackTest<4>();
  SizedStackTest<5>();
  SizedStackTest<6>();
  SizedStackTest<7>();
  SizedStackTest<16>();
  SizedStackTest<25>();
  SizedStackTest<34>();
  SizedStackTest<43>();
  SizedStackTest<51>();
  SizedStackTest<62>();
  SizedStackTest<64>();
  SizedStackTest<128>();
}

TEST(AddressSanitizer, ManyStackObjectsTest) {
  char XXX[10];
  char YYY[20];
  char ZZZ[30];
  Ident(XXX);
  Ident(YYY);
  EXPECT_DEATH(Ident(ZZZ)[-1] = 0, ASAN_PCRE_DOTALL "XXX.*YYY.*ZZZ");
}

__attribute__((noinline))
static void Frame0(int frame, char *a, char *b, char *c) {
  char d[4] = {0};
  char *D = Ident(d);
  switch (frame) {
    case 3: a[5]++; break;
    case 2: b[5]++; break;
    case 1: c[5]++; break;
    case 0: D[5]++; break;
  }
}
__attribute__((noinline)) static void Frame1(int frame, char *a, char *b) {
  char c[4] = {0}; Frame0(frame, a, b, c);
  break_optimization(0);
}
__attribute__((noinline)) static void Frame2(int frame, char *a) {
  char b[4] = {0}; Frame1(frame, a, b);
  break_optimization(0);
}
__attribute__((noinline)) static void Frame3(int frame) {
  char a[4] = {0}; Frame2(frame, a);
  break_optimization(0);
}

TEST(AddressSanitizer, GuiltyStackFrame0Test) {
  EXPECT_DEATH(Frame3(0), "located .*in frame <.*Frame0");
}
TEST(AddressSanitizer, GuiltyStackFrame1Test) {
  EXPECT_DEATH(Frame3(1), "located .*in frame <.*Frame1");
}
TEST(AddressSanitizer, GuiltyStackFrame2Test) {
  EXPECT_DEATH(Frame3(2), "located .*in frame <.*Frame2");
}
TEST(AddressSanitizer, GuiltyStackFrame3Test) {
  EXPECT_DEATH(Frame3(3), "located .*in frame <.*Frame3");
}

__attribute__((noinline))
void LongJmpFunc1(jmp_buf buf) {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  longjmp(buf, 1);
}

__attribute__((noinline))
void UnderscopeLongJmpFunc1(jmp_buf buf) {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  _longjmp(buf, 1);
}

__attribute__((noinline))
void SigLongJmpFunc1(sigjmp_buf buf) {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  siglongjmp(buf, 1);
}


__attribute__((noinline))
void TouchStackFunc() {
  int a[100];  // long array will intersect with redzones from LongJmpFunc1.
  int *A = Ident(a);
  for (int i = 0; i < 100; i++)
    A[i] = i*i;
}

// Test that we handle longjmp and do not report fals positives on stack.
TEST(AddressSanitizer, LongJmpTest) {
  static jmp_buf buf;
  if (!setjmp(buf)) {
    LongJmpFunc1(buf);
  } else {
    TouchStackFunc();
  }
}

TEST(AddressSanitizer, UnderscopeLongJmpTest) {
  static jmp_buf buf;
  if (!_setjmp(buf)) {
    UnderscopeLongJmpFunc1(buf);
  } else {
    TouchStackFunc();
  }
}

TEST(AddressSanitizer, SigLongJmpTest) {
  static sigjmp_buf buf;
  if (!sigsetjmp(buf, 1)) {
    SigLongJmpFunc1(buf);
  } else {
    TouchStackFunc();
  }
}

#ifdef __EXCEPTIONS
__attribute__((noinline))
void ThrowFunc() {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  ASAN_THROW(1);
}

TEST(AddressSanitizer, CxxExceptionTest) {
  if (ASAN_UAR) return;
  // TODO(kcc): this test crashes on 32-bit for some reason...
  if (__WORDSIZE == 32) return;
  try {
    ThrowFunc();
  } catch(...) {}
  TouchStackFunc();
}
#endif

void *ThreadStackReuseFunc1(void *unused) {
  // create three red zones for these two stack objects.
  int a;
  int b;

  int *A = Ident(&a);
  int *B = Ident(&b);
  *A = *B;
  pthread_exit(0);
  return 0;
}

void *ThreadStackReuseFunc2(void *unused) {
  TouchStackFunc();
  return 0;
}

TEST(AddressSanitizer, ThreadStackReuseTest) {
  pthread_t t;
  pthread_create(&t, 0, ThreadStackReuseFunc1, 0);
  pthread_join(t, 0);
  pthread_create(&t, 0, ThreadStackReuseFunc2, 0);
  pthread_join(t, 0);
}

#if defined(__i386__) or defined(__x86_64__)
TEST(AddressSanitizer, Store128Test) {
  char *a = Ident((char*)malloc(Ident(12)));
  char *p = a;
  if (((uintptr_t)a % 16) != 0)
    p = a + 8;
  assert(((uintptr_t)p % 16) == 0);
  __m128i value_wide = _mm_set1_epi16(0x1234);
  EXPECT_DEATH(_mm_store_si128((__m128i*)p, value_wide),
               "WRITE of size 16");
  EXPECT_DEATH(_mm_store_si128((__m128i*)p, value_wide),
               "located 0 bytes to the right of 12-byte");
  free(a);
}
#endif

static string RightOOBErrorMessage(int oob_distance) {
  assert(oob_distance >= 0);
  char expected_str[100];
  sprintf(expected_str, "located %d bytes to the right", oob_distance);
  return string(expected_str);
}

static string LeftOOBErrorMessage(int oob_distance) {
  assert(oob_distance > 0);
  char expected_str[100];
  sprintf(expected_str, "located %d bytes to the left", oob_distance);
  return string(expected_str);
}

template<class T>
void MemSetOOBTestTemplate(size_t length) {
  if (length == 0) return;
  size_t size = Ident(sizeof(T) * length);
  T *array = Ident((T*)malloc(size));
  int element = Ident(42);
  int zero = Ident(0);
  // memset interval inside array
  memset(array, element, size);
  memset(array, element, size - 1);
  memset(array + length - 1, element, sizeof(T));
  memset(array, element, 1);

  // memset 0 bytes
  memset(array - 10, element, zero);
  memset(array - 1, element, zero);
  memset(array, element, zero);
  memset(array + length, 0, zero);
  memset(array + length + 1, 0, zero);

  // try to memset bytes to the right of array
  EXPECT_DEATH(memset(array, 0, size + 1),
               RightOOBErrorMessage(0));
  EXPECT_DEATH(memset((char*)(array + length) - 1, element, 6),
               RightOOBErrorMessage(4));
  EXPECT_DEATH(memset(array + 1, element, size + sizeof(T)),
               RightOOBErrorMessage(2 * sizeof(T) - 1));
  // whole interval is to the right
  EXPECT_DEATH(memset(array + length + 1, 0, 10),
               RightOOBErrorMessage(sizeof(T)));

  // try to memset bytes to the left of array
  EXPECT_DEATH(memset((char*)array - 1, element, size),
               LeftOOBErrorMessage(1));
  EXPECT_DEATH(memset((char*)array - 5, 0, 6),
               LeftOOBErrorMessage(5));
  EXPECT_DEATH(memset(array - 5, element, size + 5 * sizeof(T)),
               LeftOOBErrorMessage(5 * sizeof(T)));
  // whole interval is to the left
  EXPECT_DEATH(memset(array - 2, 0, sizeof(T)),
               LeftOOBErrorMessage(2 * sizeof(T)));

  // try to memset bytes both to the left & to the right
  EXPECT_DEATH(memset((char*)array - 2, element, size + 4),
               LeftOOBErrorMessage(2));

  free(array);
}

TEST(AddressSanitizer, MemSetOOBTest) {
  MemSetOOBTestTemplate<char>(100);
  MemSetOOBTestTemplate<int>(5);
  MemSetOOBTestTemplate<double>(256);
  // We can test arrays of structres/classes here, but what for?
}

// Same test for memcpy and memmove functions
template <class T, class M>
void MemTransferOOBTestTemplate(size_t length) {
  if (length == 0) return;
  size_t size = Ident(sizeof(T) * length);
  T *src = Ident((T*)malloc(size));
  T *dest = Ident((T*)malloc(size));
  int zero = Ident(0);

  // valid transfer of bytes between arrays
  M::transfer(dest, src, size);
  M::transfer(dest + 1, src, size - sizeof(T));
  M::transfer(dest, src + length - 1, sizeof(T));
  M::transfer(dest, src, 1);

  // transfer zero bytes
  M::transfer(dest - 1, src, 0);
  M::transfer(dest + length, src, zero);
  M::transfer(dest, src - 1, zero);
  M::transfer(dest, src, zero);

  // try to change mem to the right of dest
  EXPECT_DEATH(M::transfer(dest + 1, src, size),
               RightOOBErrorMessage(sizeof(T) - 1));
  EXPECT_DEATH(M::transfer((char*)(dest + length) - 1, src, 5),
               RightOOBErrorMessage(3));

  // try to change mem to the left of dest
  EXPECT_DEATH(M::transfer(dest - 2, src, size),
               LeftOOBErrorMessage(2 * sizeof(T)));
  EXPECT_DEATH(M::transfer((char*)dest - 3, src, 4),
               LeftOOBErrorMessage(3));

  // try to access mem to the right of src
  EXPECT_DEATH(M::transfer(dest, src + 2, size),
               RightOOBErrorMessage(2 * sizeof(T) - 1));
  EXPECT_DEATH(M::transfer(dest, (char*)(src + length) - 3, 6),
               RightOOBErrorMessage(2));

  // try to access mem to the left of src
  EXPECT_DEATH(M::transfer(dest, src - 1, size),
               LeftOOBErrorMessage(sizeof(T)));
  EXPECT_DEATH(M::transfer(dest, (char*)src - 6, 7),
               LeftOOBErrorMessage(6));

  // Generally we don't need to test cases where both accessing src and writing
  // to dest address to poisoned memory.

  T *big_src = Ident((T*)malloc(size * 2));
  T *big_dest = Ident((T*)malloc(size * 2));
  // try to change mem to both sides of dest
  EXPECT_DEATH(M::transfer(dest - 1, big_src, size * 2),
               LeftOOBErrorMessage(sizeof(T)));
  // try to access mem to both sides of src
  EXPECT_DEATH(M::transfer(big_dest, src - 2, size * 2),
               LeftOOBErrorMessage(2 * sizeof(T)));

  free(src);
  free(dest);
  free(big_src);
  free(big_dest);
}

class MemCpyWrapper {
 public:
  static void* transfer(void *to, const void *from, size_t size) {
    return memcpy(to, from, size);
  }
};
TEST(AddressSanitizer, MemCpyOOBTest) {
  MemTransferOOBTestTemplate<char, MemCpyWrapper>(100);
  MemTransferOOBTestTemplate<int, MemCpyWrapper>(1024);
}

class MemMoveWrapper {
 public:
  static void* transfer(void *to, const void *from, size_t size) {
    return memmove(to, from, size);
  }
};
TEST(AddressSanitizer, MemMoveOOBTest) {
  MemTransferOOBTestTemplate<char, MemMoveWrapper>(100);
  MemTransferOOBTestTemplate<int, MemMoveWrapper>(1024);
}

// Tests for string functions

// Used for string functions tests
static char global_string[] = "global";
static size_t global_string_length = 6;

// Input to a test is a zero-terminated string str with given length
// Accesses to the bytes to the left and to the right of str
// are presumed to produce OOB errors
void StrLenOOBTestTemplate(char *str, size_t length, bool is_global) {
  // Normal strlen calls
  EXPECT_EQ(strlen(str), length);
  if (length > 0) {
    EXPECT_EQ(strlen(str + 1), length - 1);
    EXPECT_EQ(strlen(str + length), 0);
  }
  // Arg of strlen is not malloced, OOB access
  if (!is_global) {
    // We don't insert RedZones to the left of global variables
    EXPECT_DEATH(Ident(strlen(str - 1)), LeftOOBErrorMessage(1));
    EXPECT_DEATH(Ident(strlen(str - 5)), LeftOOBErrorMessage(5));
  }
  EXPECT_DEATH(Ident(strlen(str + length + 1)), RightOOBErrorMessage(0));
  // Overwrite terminator
  str[length] = 'a';
  // String is not zero-terminated, strlen will lead to OOB access
  EXPECT_DEATH(Ident(strlen(str)), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strlen(str + length)), RightOOBErrorMessage(0));
  // Restore terminator
  str[length] = 0;
}
TEST(AddressSanitizer, StrLenOOBTest) {
  // Check heap-allocated string
  size_t length = Ident(10);
  char *heap_string = Ident((char*)malloc(length + 1));
  char stack_string[10 + 1];
  for (int i = 0; i < length; i++) {
    heap_string[i] = 'a';
    stack_string[i] = 'b';
  }
  heap_string[length] = 0;
  stack_string[length] = 0;
  StrLenOOBTestTemplate(heap_string, length, false);
  // TODO(samsonov): Fix expected messages in StrLenOOBTestTemplate to
  //      make test for stack_string work. Or move it to output tests.
  // StrLenOOBTestTemplate(stack_string, length, false);
  StrLenOOBTestTemplate(global_string, global_string_length, true);
  free(heap_string);
}

#ifndef __APPLE__
TEST(AddressSanitizer, StrNLenOOBTest) {
  size_t size = Ident(123);
  char *str = Ident((char*)malloc(size));
  memset(str, 'z', size);
  // Normal strnlen calls.
  Ident(strnlen(str - 1, 0));
  Ident(strnlen(str, size));
  Ident(strnlen(str + size - 1, 1));
  str[size - 1] = '\0';
  Ident(strnlen(str, 2 * size));
  // Argument points to not allocated memory.
  EXPECT_DEATH(Ident(strnlen(str - 1, 1)), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strnlen(str + size, 1)), RightOOBErrorMessage(0));
  // Overwrite the terminating '\0' and hit unallocated memory.
  str[size - 1] = 'z';
  EXPECT_DEATH(Ident(strnlen(str, size + 1)), RightOOBErrorMessage(0));
  free(str);
}
#endif

TEST(AddressSanitizer, StrDupOOBTest) {
  size_t size = Ident(42);
  char *str = Ident((char*)malloc(size));
  char *new_str;
  memset(str, 'z', size);
  // Normal strdup calls.
  str[size - 1] = '\0';
  new_str = strdup(str);
  free(new_str);
  new_str = strdup(str + size - 1);
  free(new_str);
  // Argument points to not allocated memory.
  EXPECT_DEATH(Ident(strdup(str - 1)), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strdup(str + size)), RightOOBErrorMessage(0));
  // Overwrite the terminating '\0' and hit unallocated memory.
  str[size - 1] = 'z';
  EXPECT_DEATH(Ident(strdup(str)), RightOOBErrorMessage(0));
  free(str);
}

TEST(AddressSanitizer, StrCpyOOBTest) {
  size_t to_size = Ident(30);
  size_t from_size = Ident(6);  // less than to_size
  char *to = Ident((char*)malloc(to_size));
  char *from = Ident((char*)malloc(from_size));
  // Normal strcpy calls.
  strcpy(from, "hello");
  strcpy(to, from);
  strcpy(to + to_size - from_size, from);
  // Length of "from" is too small.
  EXPECT_DEATH(Ident(strcpy(from, "hello2")), RightOOBErrorMessage(0));
  // "to" or "from" points to not allocated memory.
  EXPECT_DEATH(Ident(strcpy(to - 1, from)), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strcpy(to, from - 1)), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strcpy(to, from + from_size)), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strcpy(to + to_size, from)), RightOOBErrorMessage(0));
  // Overwrite the terminating '\0' character and hit unallocated memory.
  from[from_size - 1] = '!';
  EXPECT_DEATH(Ident(strcpy(to, from)), RightOOBErrorMessage(0));
  free(to);
  free(from);
}

TEST(AddressSanitizer, StrNCpyOOBTest) {
  size_t to_size = Ident(20);
  size_t from_size = Ident(6);  // less than to_size
  char *to = Ident((char*)malloc(to_size));
  // From is a zero-terminated string "hello\0" of length 6
  char *from = Ident((char*)malloc(from_size));
  strcpy(from, "hello");
  // copy 0 bytes
  strncpy(to, from, 0);
  strncpy(to - 1, from - 1, 0);
  // normal strncpy calls
  strncpy(to, from, from_size);
  strncpy(to, from, to_size);
  strncpy(to, from + from_size - 1, to_size);
  strncpy(to + to_size - 1, from, 1);
  // One of {to, from} points to not allocated memory
  EXPECT_DEATH(Ident(strncpy(to, from - 1, from_size)),
               LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strncpy(to - 1, from, from_size)),
               LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strncpy(to, from + from_size, 1)),
               RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strncpy(to + to_size, from, 1)),
               RightOOBErrorMessage(0));
  // Length of "to" is too small
  EXPECT_DEATH(Ident(strncpy(to + to_size - from_size + 1, from, from_size)),
               RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strncpy(to + 1, from, to_size)),
               RightOOBErrorMessage(0));
  // Overwrite terminator in from
  from[from_size - 1] = '!';
  // normal strncpy call
  strncpy(to, from, from_size);
  // Length of "from" is too small
  EXPECT_DEATH(Ident(strncpy(to, from, to_size)),
               RightOOBErrorMessage(0));
  free(to);
  free(from);
}

typedef char*(*PointerToStrChr)(const char*, int);
void RunStrChrTest(PointerToStrChr StrChr) {
  size_t size = Ident(100);
  char *str = Ident((char*)malloc(size));
  memset(str, 'z', size);
  str[10] = 'q';
  str[11] = '\0';
  EXPECT_EQ(str, StrChr(str, 'z'));
  EXPECT_EQ(str + 10, StrChr(str, 'q'));
  EXPECT_EQ(NULL, StrChr(str, 'a'));
  // StrChr argument points to not allocated memory.
  EXPECT_DEATH(Ident(StrChr(str - 1, 'z')), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(StrChr(str + size, 'z')), RightOOBErrorMessage(0));
  // Overwrite the terminator and hit not allocated memory.
  str[11] = 'z';
  EXPECT_DEATH(Ident(StrChr(str, 'a')), RightOOBErrorMessage(0));
  free(str);
}
TEST(AddressSanitizer, StrChrAndIndexOOBTest) {
  RunStrChrTest(&strchr);
  RunStrChrTest(&index);
}

TEST(AddressSanitizer, StrCmpAndFriendsLogicTest) {
  // strcmp
  EXPECT_EQ(0, strcmp("", ""));
  EXPECT_EQ(0, strcmp("abcd", "abcd"));
  EXPECT_EQ(-1, strcmp("ab", "ac"));
  EXPECT_EQ(-1, strcmp("abc", "abcd"));
  EXPECT_EQ(1, strcmp("acc", "abc"));
  EXPECT_EQ(1, strcmp("abcd", "abc"));

  // strncmp
  EXPECT_EQ(0, strncmp("a", "b", 0));
  EXPECT_EQ(0, strncmp("abcd", "abcd", 10));
  EXPECT_EQ(0, strncmp("abcd", "abcef", 3));
  EXPECT_EQ(-1, strncmp("abcde", "abcfa", 4));
  EXPECT_EQ(-1, strncmp("a", "b", 5));
  EXPECT_EQ(-1, strncmp("bc", "bcde", 4));
  EXPECT_EQ(1, strncmp("xyz", "xyy", 10));
  EXPECT_EQ(1, strncmp("baa", "aaa", 1));
  EXPECT_EQ(1, strncmp("zyx", "", 2));
}

static inline char* MallocAndMemsetString(size_t size) {
  char *s = Ident((char*)malloc(size));
  memset(s, 'z', size);
  return s;
}

TEST(AddressSanitizer, StrCmpOOBTest) {
  size_t size = Ident(100);
  char *s1 = MallocAndMemsetString(size);
  char *s2 = MallocAndMemsetString(size);
  s1[size - 1] = '\0';
  s2[size - 1] = '\0';
  // Normal strcmp calls
  Ident(strcmp(s1, s2));
  Ident(strcmp(s1, s2 + size - 1));
  Ident(strcmp(s1 + size - 1, s2 + size - 1));
  s1[size - 1] = 'z';
  s2[size - 1] = 'x';
  Ident(strcmp(s1, s2));
  // One of arguments points to not allocated memory.
  EXPECT_DEATH(Ident(strcmp)(s1 - 1, s2), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strcmp)(s1, s2 - 1), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strcmp)(s1 + size, s2), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strcmp)(s1, s2 + size), RightOOBErrorMessage(0));
  // Hit unallocated memory and die.
  s2[size - 1] = 'z';
  EXPECT_DEATH(Ident(strcmp)(s1, s1), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strcmp)(s1 + size - 1, s2), RightOOBErrorMessage(0));
  free(s1);
  free(s2);
}

TEST(AddressSanitizer, StrNCmpOOBTest) {
  size_t size = Ident(100);
  char *s1 = MallocAndMemsetString(size);
  char *s2 = MallocAndMemsetString(size);
  s1[size - 1] = '\0';
  s2[size - 1] = '\0';
  // Normal strncmp calls
  Ident(strncmp(s1, s2, size + 2));
  s1[size - 1] = 'z';
  s2[size - 1] = 'x';
  Ident(strncmp(s1 + size - 2, s2 + size - 2, size));
  s2[size - 1] = 'z';
  Ident(strncmp(s1 - 1, s2 - 1, 0));
  Ident(strncmp(s1 + size - 1, s2 + size - 1, 1));
  // One of arguments points to not allocated memory.
  EXPECT_DEATH(Ident(strncmp)(s1 - 1, s2, 1), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strncmp)(s1, s2 - 1, 1), LeftOOBErrorMessage(1));
  EXPECT_DEATH(Ident(strncmp)(s1 + size, s2, 1), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strncmp)(s1, s2 + size, 1), RightOOBErrorMessage(0));
  // Hit unallocated memory and die.
  EXPECT_DEATH(Ident(strncmp)(s1 + 1, s2 + 1, size), RightOOBErrorMessage(0));
  EXPECT_DEATH(Ident(strncmp)(s1 + size - 1, s2, 2), RightOOBErrorMessage(0));
  free(s1);
  free(s2);
}

static const char *kOverlapErrorMessage = "strcpy-param-overlap";

TEST(AddressSanitizer, StrArgsOverlapTest) {
  size_t size = Ident(100);
  char *str = Ident((char*)malloc(size));

#if 0
  // Check "memcpy". Use Ident() to avoid inlining.
  memset(str, 'z', size);
  Ident(memcpy)(str + 1, str + 11, 10);
  Ident(memcpy)(str, str, 0);
  EXPECT_DEATH(Ident(memcpy)(str, str + 14, 15), kOverlapErrorMessage);
  EXPECT_DEATH(Ident(memcpy)(str + 14, str, 15), kOverlapErrorMessage);
  EXPECT_DEATH(Ident(memcpy)(str + 20, str + 20, 1), kOverlapErrorMessage);
#endif

  // Check "strcpy".
  memset(str, 'z', size);
  str[9] = '\0';
  strcpy(str + 10, str);
  EXPECT_DEATH(strcpy(str + 9, str), kOverlapErrorMessage);
  EXPECT_DEATH(strcpy(str, str + 4), kOverlapErrorMessage);
  strcpy(str, str + 5);

  // Check "strncpy".
  memset(str, 'z', size);
  strncpy(str, str + 10, 10);
  EXPECT_DEATH(strncpy(str, str + 9, 10), kOverlapErrorMessage);
  EXPECT_DEATH(strncpy(str + 9, str, 10), kOverlapErrorMessage);
  str[10] = '\0';
  strncpy(str + 11, str, 20);
  EXPECT_DEATH(strncpy(str + 10, str, 20), kOverlapErrorMessage);

  free(str);
}

// At the moment we instrument memcpy/memove/memset calls at compile time so we
// can't handle OOB error if these functions are called by pointer, see disabled
// MemIntrinsicCallByPointerTest below
typedef void*(*PointerToMemTransfer)(void*, const void*, size_t);
typedef void*(*PointerToMemSet)(void*, int, size_t);

void CallMemSetByPointer(PointerToMemSet MemSet) {
  size_t size = Ident(100);
  char *array = Ident((char*)malloc(size));
  EXPECT_DEATH(MemSet(array, 0, 101), RightOOBErrorMessage(0));
  free(array);
}

void CallMemTransferByPointer(PointerToMemTransfer MemTransfer) {
  size_t size = Ident(100);
  char *src = Ident((char*)malloc(size));
  char *dst = Ident((char*)malloc(size));
  EXPECT_DEATH(MemTransfer(dst, src, 101), RightOOBErrorMessage(0));
  free(src);
  free(dst);
}

TEST(AddressSanitizer, DISABLED_MemIntrinsicCallByPointerTest) {
  CallMemSetByPointer(&memset);
  CallMemTransferByPointer(&memcpy);
  CallMemTransferByPointer(&memmove);
}

// This test case fails
// Clang optimizes memcpy/memset calls which lead to unaligned access
TEST(AddressSanitizer, DISABLED_MemIntrinsicUnalignedAccessTest) {
  int size = Ident(4096);
  char *s = Ident((char*)malloc(size));
  EXPECT_DEATH(memset(s + size - 1, 0, 2), RightOOBErrorMessage(0));
  free(s);
}

// TODO(samsonov): Add a test with malloc(0)
// TODO(samsonov): Add tests for str* and mem* functions.

__attribute__((noinline))
static int LargeFunction(bool do_bad_access) {
  int *x = new int[100];
  x[0]++;
  x[1]++;
  x[2]++;
  x[3]++;
  x[4]++;
  x[5]++;
  x[6]++;
  x[7]++;
  x[8]++;
  x[9]++;

  x[do_bad_access ? 100 : 0]++; int res = __LINE__;

  x[10]++;
  x[11]++;
  x[12]++;
  x[13]++;
  x[14]++;
  x[15]++;
  x[16]++;
  x[17]++;
  x[18]++;
  x[19]++;

  delete x;
  return res;
}

// Test the we have correct debug info for the failing instruction.
// This test requires the in-process symbolizer to be enabled by default.
TEST(AddressSanitizer, DISABLED_LargeFunctionSymbolizeTest) {
  int failing_line = LargeFunction(false);
  char expected_warning[128];
  sprintf(expected_warning, "LargeFunction.*asan_test.cc:%d", failing_line);
  EXPECT_DEATH(LargeFunction(true), expected_warning);
}

// Check that we unwind and symbolize correctly.
TEST(AddressSanitizer, DISABLED_MallocFreeUnwindAndSymbolizeTest) {
  int *a = (int*)malloc_aaa(sizeof(int));
  *a = 1;
  free_aaa(a);
  EXPECT_DEATH(*a = 1, "free_ccc.*free_bbb.*free_aaa.*"
               "malloc_fff.*malloc_eee.*malloc_ddd");
}

void *ThreadedTestAlloc(void *a) {
  int **p = (int**)a;
  *p = new int;
  return 0;
}

void *ThreadedTestFree(void *a) {
  int **p = (int**)a;
  delete *p;
  return 0;
}

void *ThreadedTestUse(void *a) {
  int **p = (int**)a;
  **p = 1;
  return 0;
}

void ThreadedTestSpawn() {
  pthread_t t;
  int *x;
  pthread_create(&t, 0, ThreadedTestAlloc, &x);
  pthread_join(t, 0);
  pthread_create(&t, 0, ThreadedTestFree, &x);
  pthread_join(t, 0);
  pthread_create(&t, 0, ThreadedTestUse, &x);
  pthread_join(t, 0);
}

TEST(AddressSanitizer, ThreadedTest) {
  EXPECT_DEATH(ThreadedTestSpawn(),
               ASAN_PCRE_DOTALL
               "Thread T.*created"
               ".*Thread T.*created"
               ".*Thread T.*created");
}

#if ASAN_NEEDS_SEGV
TEST(AddressSanitizer, ShadowGapTest) {
#if __WORDSIZE == 32
  char *addr = (char*)0x22000000;
#else
  char *addr = (char*)0x0000100000080000;
#endif
  EXPECT_DEATH(*addr = 1, "AddressSanitizer crashed on unknown");
}
#endif  // ASAN_NEEDS_SEGV

extern "C" {
__attribute__((noinline))
static void UseThenFreeThenUse() {
  char *x = Ident((char*)malloc(8));
  *x = 1;
  free_aaa(x);
  *x = 2;
}
}

TEST(AddressSanitizer, UseThenFreeThenUseTest) {
  EXPECT_DEATH(UseThenFreeThenUse(), "freed by thread");
}

TEST(AddressSanitizer, StrDupTest) {
  free(strdup(Ident("123")));
}

TEST(AddressSanitizer, ObjdumpTest) {
  ObjdumpOfMyself *o = objdump_of_myself();
  EXPECT_TRUE(o->IsCorrect());
}

extern "C" {
__attribute__((noinline))
static void DisasmSimple() {
  Ident(0);
}

__attribute__((noinline))
static void DisasmParamWrite(int *a) {
  *a = 1;
}

__attribute__((noinline))
static void DisasmParamInc(int *a) {
  (*a)++;
}

__attribute__((noinline))
static void DisasmParamReadIfWrite(int *a) {
  if (*a)
    *a = 1;
}

__attribute__((noinline))
static int DisasmParamIfReadWrite(int *a, int cond) {
  int res = 0;
  if (cond)
    res = *a;
  *a = 0;
  return res;
}

static int GLOBAL;

__attribute__((noinline))
static void DisasmWriteGlob() {
  GLOBAL = 1;
}
}  // extern "C"

TEST(AddressSanitizer, DisasmTest) {
  int a;
  DisasmSimple();
  DisasmParamWrite(&a);
  DisasmParamInc(&a);
  Ident(DisasmWriteGlob)();
  DisasmParamReadIfWrite(&a);

  a = 7;
  EXPECT_EQ(7, DisasmParamIfReadWrite(&a, Ident(1)));
  EXPECT_EQ(0, a);

  ObjdumpOfMyself *o = objdump_of_myself();
  vector<string> insns;
  insns.push_back("ud2");
  insns.push_back("__asan_report_");
  EXPECT_EQ(0, o->CountInsnInFunc("DisasmSimple", insns));
  EXPECT_EQ(1, o->CountInsnInFunc("DisasmParamWrite", insns));
  EXPECT_EQ(1, o->CountInsnInFunc("DisasmParamInc", insns));
  EXPECT_EQ(0, o->CountInsnInFunc("DisasmWriteGlob", insns));

  // TODO(kcc): implement these (needs just one __asan_report).
  EXPECT_EQ(2, o->CountInsnInFunc("DisasmParamReadIfWrite", insns));
  EXPECT_EQ(2, o->CountInsnInFunc("DisasmParamIfReadWrite", insns));
}

// Currently we create and poison redzone at right of global variables.
char glob5[5];
static char static110[110];
const char ConstGlob[7] = {1, 2, 3, 4, 5, 6, 7};
static const char StaticConstGlob[3] = {9, 8, 7};
extern int GlobalsTest(int x);

TEST(AddressSanitizer, GlobalTest) {
  static char func_static15[15];

  static char fs1[10];
  static char fs2[10];
  static char fs3[10];

  glob5[Ident(0)] = 0;
  glob5[Ident(1)] = 0;
  glob5[Ident(2)] = 0;
  glob5[Ident(3)] = 0;
  glob5[Ident(4)] = 0;

  EXPECT_DEATH(glob5[Ident(5)] = 0,
               "0 bytes to the right of global variable.*glob5.* size 5");
  EXPECT_DEATH(glob5[Ident(5+6)] = 0,
               "6 bytes to the right of global variable.*glob5.* size 5");
  Ident(static110);  // avoid optimizations
  static110[Ident(0)] = 0;
  static110[Ident(109)] = 0;
  EXPECT_DEATH(static110[Ident(110)] = 0,
               "0 bytes to the right of global variable");
  EXPECT_DEATH(static110[Ident(110+7)] = 0,
               "7 bytes to the right of global variable");

  Ident(func_static15);  // avoid optimizations
  func_static15[Ident(0)] = 0;
  EXPECT_DEATH(func_static15[Ident(15)] = 0,
               "0 bytes to the right of global variable");
  EXPECT_DEATH(func_static15[Ident(15 + 9)] = 0,
               "9 bytes to the right of global variable");

  Ident(fs1);
  Ident(fs2);
  Ident(fs3);

  // We don't create left redzones, so this is not 100% guaranteed to fail.
  // But most likely will.
  EXPECT_DEATH(fs2[Ident(-1)] = 0, "is located.*of global variable");

  EXPECT_DEATH(Ident(Ident(ConstGlob)[8]),
               "is located 1 bytes to the right of .*ConstGlob");
  EXPECT_DEATH(Ident(Ident(StaticConstGlob)[5]),
               "is located 2 bytes to the right of .*StaticConstGlob");

  // call stuff from another file.
  GlobalsTest(0);
}

TEST(AddressSanitizer, GlobalStringConstTest) {
  static const char *zoo = "FOOBAR123";
  const char *p = Ident(zoo);
  EXPECT_DEATH(Ident(p[15]), "is ascii string 'FOOBAR123'");
}

int *ReturnsPointerToALocalObject() {
  int a = 0;
  return Ident(&a);
}

TEST(AddressSanitizer, LocalReferenceReturnTest) {
  int *(*f)() = Ident(ReturnsPointerToALocalObject);
  // Call f several times, only the first time should be reported.
  f();
  f();
  f();
  f();
  if (ASAN_UAR) {
    EXPECT_DEATH(*f() = 1, "is located.*in frame .*ReturnsPointerToALocal");
  }
}

template <int kSize>
__attribute__((noinline))
static void FuncWithStack() {
  char x[kSize];
  Ident(x)[0] = 0;
  Ident(x)[kSize-1] = 0;
}

static void LotsOfStackReuse() {
  int LargeStack[10000];
  Ident(LargeStack)[0] = 0;
  for (int i = 0; i < 10000; i++) {
    FuncWithStack<128 * 1>();
    FuncWithStack<128 * 2>();
    FuncWithStack<128 * 4>();
    FuncWithStack<128 * 8>();
    FuncWithStack<128 * 16>();
    FuncWithStack<128 * 32>();
    FuncWithStack<128 * 64>();
    FuncWithStack<128 * 128>();
    FuncWithStack<128 * 256>();
    FuncWithStack<128 * 512>();
    Ident(LargeStack)[0] = 0;
  }
}

TEST(AddressSanitizer, StressStackReuseTest) {
  LotsOfStackReuse();
}

TEST(AddressSanitizer, ThreadedStressStackReuseTest) {
  const int kNumThreads = 20;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    pthread_create(&t[i], 0, (void* (*)(void *x))LotsOfStackReuse, 0);
  }
  for (int i = 0; i < kNumThreads; i++) {
    pthread_join(t[i], 0);
  }
}

#ifdef __EXCEPTIONS
__attribute__((noinline))
static void StackReuseAndException() {
  int large_stack[1000];
  Ident(large_stack);
  ASAN_THROW(1);
}

// TODO(kcc): support exceptions with use-after-return.
TEST(AddressSanitizer, DISABLED_StressStackReuseAndExceptionsTest) {
  for (int i = 0; i < 10000; i++) {
    try {
    StackReuseAndException();
    } catch(...) {
    }
  }
}
#endif

TEST(AddressSanitizer, MlockTest) {
  EXPECT_EQ(0, mlockall(MCL_CURRENT));
  EXPECT_EQ(0, mlock((void*)0x12345, 0x5678));
  EXPECT_EQ(0, munlockall());
  EXPECT_EQ(0, munlock((void*)0x987, 0x654));
}

// ------------------ demo tests; run each one-by-one -------------
// e.g. --gtest_filter=*DemoOOBLeftHigh --gtest_also_run_disabled_tests
TEST(AddressSanitizer, DISABLED_DemoThreadedTest) {
  ThreadedTestSpawn();
}

void *SimpleBugOnSTack(void *x = 0) {
  char a[20];
  Ident(a)[20] = 0;
  return 0;
}

TEST(AddressSanitizer, DISABLED_DemoStackTest) {
  SimpleBugOnSTack();
}

TEST(AddressSanitizer, DISABLED_DemoThreadStackTest) {
  pthread_t t;
  pthread_create(&t, 0, SimpleBugOnSTack, 0);
  pthread_join(t, 0);
}

TEST(AddressSanitizer, DISABLED_DemoUAFLowIn) {
  uaf_test<U1>(10, 0);
}
TEST(AddressSanitizer, DISABLED_DemoUAFLowLeft) {
  uaf_test<U1>(10, -2);
}
TEST(AddressSanitizer, DISABLED_DemoUAFLowRight) {
  uaf_test<U1>(10, 10);
}

TEST(AddressSanitizer, DISABLED_DemoUAFHigh) {
  uaf_test<U1>(kLargeMalloc, 0);
}

TEST(AddressSanitizer, DISABLED_DemoOOBLeftLow) {
  oob_test<U1>(10, -1);
}

TEST(AddressSanitizer, DISABLED_DemoOOBLeftHigh) {
  oob_test<U1>(kLargeMalloc, -1);
}

TEST(AddressSanitizer, DISABLED_DemoOOBRightLow) {
  oob_test<U1>(10, 10);
}

TEST(AddressSanitizer, DISABLED_DemoOOBRightHigh) {
  oob_test<U1>(kLargeMalloc, kLargeMalloc);
}

TEST(AddressSanitizer, DISABLED_DemoOOM) {
  size_t size = __WORDSIZE == 64 ? (size_t)(1ULL << 40) : (0xf0000000);
  printf("%p\n", malloc(size));
}

TEST(AddressSanitizer, DISABLED_DemoDoubleFreeTest) {
  DoubleFree();
}

TEST(AddressSanitizer, DISABLED_DemoNullDerefTest) {
  int *a = 0;
  Ident(a)[10] = 0;
}

TEST(AddressSanitizer, DISABLED_DemoFunctionStaticTest) {
  static char a[100];
  static char b[100];
  static char c[100];
  Ident(a);
  Ident(b);
  Ident(c);
  Ident(a)[5] = 0;
  Ident(b)[105] = 0;
  Ident(a)[5] = 0;
}

TEST(AddressSanitizer, DISABLED_DemoTooMuchMemoryTest) {
  const size_t kAllocSize = (1 << 28) - 1024;
  size_t total_size = 0;
  while (true) {
    char *x = (char*)malloc(kAllocSize);
    memset(x, 0, kAllocSize);
    total_size += kAllocSize;
    fprintf(stderr, "total: %ldM\n", (long)total_size >> 20);
  }
}

#ifdef __APPLE__
#include "asan_mac_test.h"
// TODO(glider): figure out whether we still need these tests. Is it correct
// to intercept CFAllocator?
TEST(AddressSanitizerMac, DISABLED_CFAllocatorDefaultDoubleFree) {
  EXPECT_DEATH(
      CFAllocatorDefaultDoubleFree(),
      "attempting double-free");
}

TEST(AddressSanitizerMac, DISABLED_CFAllocatorSystemDefaultDoubleFree) {
  EXPECT_DEATH(
      CFAllocatorSystemDefaultDoubleFree(),
      "attempting double-free");
}

TEST(AddressSanitizerMac, DISABLED_CFAllocatorMallocDoubleFree) {
  EXPECT_DEATH(CFAllocatorMallocDoubleFree(), "attempting double-free");
}

TEST(AddressSanitizerMac, DISABLED_CFAllocatorMallocZoneDoubleFree) {
  EXPECT_DEATH(CFAllocatorMallocZoneDoubleFree(), "attempting double-free");
}

TEST(AddressSanitizerMac, DISABLED_GCDDispatchAsync) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDDispatchAsync(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, DISABLED_GCDDispatchSync) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDDispatchSync(), "Shadow byte and word");
}


TEST(AddressSanitizerMac, DISABLED_GCDReuseWqthreadsAsync) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDReuseWqthreadsAsync(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, DISABLED_GCDReuseWqthreadsSync) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDReuseWqthreadsSync(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, DISABLED_GCDDispatchAfter) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDDispatchAfter(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, DISABLED_GCDSourceEvent) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDSourceEvent(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, DISABLED_GCDSourceCancel) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDSourceCancel(), "Shadow byte and word");
}

TEST(AddressSanitizerMac, DISABLED_GCDGroupAsync) {
  // Make sure the whole ASan report is printed, i.e. that we don't die
  // on a CHECK.
  EXPECT_DEATH(TestGCDGroupAsync(), "Shadow byte and word");
}

void *MallocIntrospectionLockWorker(void *_) {
  const int kNumPointers = 100;
  int i;
  void *pointers[kNumPointers];
  for (i = 0; i < kNumPointers; i++) {
    pointers[i] = malloc(i + 1);
  }
  for (i = 0; i < kNumPointers; i++) {
    free(pointers[i]);
  }

  return NULL;
}

void *MallocIntrospectionLockForker(void *_) {
  pid_t result = fork();
  if (result == -1) {
    perror("fork");
  }
  assert(result != -1);
  if (result == 0) {
    // Call malloc in the child process to make sure we won't deadlock.
    void *ptr = malloc(42);
    free(ptr);
    exit(0);
  } else {
    // Return in the parent process.
    return NULL;
  }
}

TEST(AddressSanitizerMac, MallocIntrospectionLock) {
  // Incorrect implementation of force_lock and force_unlock in our malloc zone
  // will cause forked processes to deadlock.
  // TODO(glider): need to detect that none of the child processes deadlocked.
  const int kNumWorkers = 5, kNumIterations = 100;
  int i, iter;
  for (iter = 0; iter < kNumIterations; iter++) {
    pthread_t workers[kNumWorkers], forker;
    for (i = 0; i < kNumWorkers; i++) {
      pthread_create(&workers[i], 0, MallocIntrospectionLockWorker, 0);
    }
    pthread_create(&forker, 0, MallocIntrospectionLockForker, 0);
    for (i = 0; i < kNumWorkers; i++) {
      pthread_join(workers[i], 0);
    }
    pthread_join(forker, 0);
  }
}

void *TSDAllocWorker(void *test_key) {
  if (test_key) {
    void *mem = malloc(10);
    pthread_setspecific(*(pthread_key_t*)test_key, mem);
  }
  return NULL;
}

TEST(AddressSanitizerMac, DISABLED_TSDWorkqueueTest) {
  pthread_t th;
  pthread_key_t test_key;
  pthread_key_create(&test_key, CallFreeOnWorkqueue);
  pthread_create(&th, NULL, TSDAllocWorker, &test_key);
  pthread_join(th, NULL);
  pthread_key_delete(test_key);
}
#endif  // __APPLE__

int main(int argc, char **argv) {
  progname = argv[0];
  testing::GTEST_FLAG(death_test_style) = "threadsafe";
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
