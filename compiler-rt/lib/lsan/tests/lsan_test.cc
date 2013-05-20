//=-- lsan_test.cc --------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Tests for leak checking functionality.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_LINUX && defined(__x86_64__)

#include <dlfcn.h>
#include <pthread.h>
#include <string>

#include "gtest/gtest.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/tests/sanitizer_test_utils.h"

#include "lsan.h"
#include "lsan_common.h"

static char **global_argv;

namespace {
uptr kMagic = 0xBABABABABABABABA;
#define HIDE(p) ((void *)((uptr)(p) ^ kMagic))
#define PEEK(p) HIDE(p)

uptr kSmallAllocSize = 10;
// maxsize in primary allocator is always less than this.
uptr kLargeAllocSize = 1 << 25;

::testing::AssertionResult IsLeaked(void *hidden_p, uptr sources) {
  InternalVector<void *> leaked(1);
  __lsan::ReportLeaked(&leaked, sources);
  for (uptr i = 0; i < leaked.size(); i++)
    if (leaked[i] == PEEK(hidden_p))
      return ::testing::AssertionSuccess() << PEEK(hidden_p) << " is leaked";
  return ::testing::AssertionFailure() << PEEK(hidden_p) << " is not leaked";
}

#define EXPECT_LEAKED(p, sources) EXPECT_TRUE(IsLeaked(HIDE((p)), sources))
#define EXPECT_NOT_LEAKED(p, sources) EXPECT_FALSE(IsLeaked(HIDE((p)), sources))

// Tests for various sources of pointers.
// Stacks and registers are tricky: pointers sometimes get stuck in them and
// cause false negatives. This is not considered a bug in LSan, but we don't
// want it to interfere with our tests, so we disable those sources whenever we
// can. We don't disable globals though: if a stale pointer somehow makes it
// into global state and causes a false negative, we want to know.

uptr kAllButStackAndRegisters =
    __lsan::kSourceAllAligned & ~__lsan::kSourceStacks &
    ~__lsan::kSourceRegisters;

void TestSource(void **p, uptr source) {
  uptr baseline = kAllButStackAndRegisters | source;
  *p = malloc(kSmallAllocSize);
  EXPECT_NOT_LEAKED(*p, baseline);
  EXPECT_LEAKED(*p, baseline & ~source);
  // Check again, in case the first EXPECT_NOT_LEAKED was a false negative.
  EXPECT_NOT_LEAKED(*p, baseline);
  free((void *)*p);
  *p = NULL;
}

void *data_var = (void*) 1;

TEST(LeakSanitizer, InitializedGlobals) {
  TestSource(&data_var, __lsan::kSourceGlobals);
}

void *bss_var;

TEST(LeakSanitizer, UninitializedGlobals) {
  TestSource(&bss_var, __lsan::kSourceGlobals);
}

TEST(LeakSanitizer, Stack) {
  void *local_var;
  TestSource(&local_var, __lsan::kSourceStacks);
}

THREADLOCAL void *tl_var;

TEST(LeakSanitizer, StaticTLS) {
  TestSource(&tl_var, __lsan::kSourceTLS);
}

// Dynamically allocated TLS space.
TEST(LeakSanitizer, DynamicTLS) {
  // Compute the path to our loadable DSO.  We assume it's in the same
  // directory.
  char **argv = global_argv;
  const std::string kLoadableSO = "liblsan_tls_loadable-x86_64.so";
  std::string path = argv[0];
  size_t last_slash = path.rfind('/');
  ASSERT_NE(last_slash, std::string::npos);
  path.erase(last_slash + 1);
  path.append(kLoadableSO);
  void *handle = dlopen(path.c_str(), RTLD_LAZY);
  ASSERT_TRUE(handle != NULL) << "dlerror " << dlerror();
  typedef void **(* store_t)(void *p);
  store_t StoreToTLS = (store_t)dlsym(handle, "StoreToTLS");
  ASSERT_EQ(0, dlerror());

  void *p = malloc(kSmallAllocSize);
  void **p_in_tls = StoreToTLS(p);
  EXPECT_NOT_LEAKED(p, kAllButStackAndRegisters);
  EXPECT_LEAKED(p, kAllButStackAndRegisters & ~__lsan::kSourceTLS);
  EXPECT_NOT_LEAKED(p, kAllButStackAndRegisters);
  free(p);
  *p_in_tls = NULL;
}

// From glibc: this many keys are stored in the thread descriptor directly.
const uptr PTHREAD_KEY_2NDLEVEL_SIZE = 32;

// Thread-specific storage that is statically alocated in the thread descriptor.
TEST(LeakSanitizer, PthreadSpecificStatic) {
  pthread_key_t key;
  ASSERT_EQ(0, pthread_key_create(&key, NULL));
  ASSERT_LT(key, PTHREAD_KEY_2NDLEVEL_SIZE);
  void *p = malloc(kSmallAllocSize);
  ASSERT_EQ(0, pthread_setspecific(key, p));
  EXPECT_NOT_LEAKED(p, kAllButStackAndRegisters);
  EXPECT_LEAKED(p, kAllButStackAndRegisters & ~__lsan::kSourceTLS);
  EXPECT_NOT_LEAKED(p, kAllButStackAndRegisters);
  ASSERT_EQ(0, pthread_setspecific(key, 0));
  free(p);
}

// Dynamically allocated thread-specific storage.
TEST(LeakSanitizer, PthreadSpecificDynamic) {
  static const uptr kDummyKeysCount = PTHREAD_KEY_2NDLEVEL_SIZE;
  pthread_key_t dummy_keys[kDummyKeysCount];
  for (uptr i = 0; i < kDummyKeysCount; i++)
    ASSERT_EQ(0, pthread_key_create(&dummy_keys[i], NULL));
  pthread_key_t key;
  ASSERT_EQ(0, pthread_key_create(&key, NULL));
  void *p  = malloc(kSmallAllocSize);
  ASSERT_EQ(0, pthread_setspecific(key, p));
  EXPECT_NOT_LEAKED(p, kAllButStackAndRegisters);
  EXPECT_LEAKED(p, kAllButStackAndRegisters & ~__lsan::kSourceTLS);
  EXPECT_NOT_LEAKED(p, kAllButStackAndRegisters);
  ASSERT_EQ(0, pthread_setspecific(key, NULL));
  CHECK_EQ(0, pthread_key_delete(key));
  for (uptr i = 0; i < kDummyKeysCount; i++)
    CHECK_EQ(0, pthread_key_delete(dummy_keys[i]));
  free(p);
}

// Put pointer far enough on the stack that LSan has space to run in without
// overwriting it.
NOINLINE uptr PutPointerOnStaleStack(void *p) {
  void *locals[2048];
  locals[0] = p;
  break_optimization(&locals[0]);
  // Hide the result, just to suppress the compiler warning.
  return (uptr)HIDE(&locals[0]);
}

// Local variables that have gone out of scope should be ignored by LSan.
TEST(LeakSanitizer, StaleLocalsAreUnreachable) {
  void *p = malloc(kSmallAllocSize);
  void **stale_var = (void **)PEEK(PutPointerOnStaleStack(p));
  p = HIDE(p);
  EXPECT_LEAKED(PEEK(p), __lsan::kSourceAllAligned & ~__lsan::kSourceRegisters);
  p = PEEK(p);
  // Make sure LSan didn't overwrite the pointer at some point.
  EXPECT_EQ(p, *stale_var);
  free(p);
}

void *large_alloc;

// Make sure LargeMmapAllocator's chunks aren't reachable via some internal data
// structure.
TEST(LeakSanitizer, SimpleLargeAllocationLeaked) {
  large_alloc = HIDE(malloc(kLargeAllocSize));
  EXPECT_TRUE(IsLeaked((void *)large_alloc,
      __lsan::kSourceAllAligned & ~__lsan::kSourceRegisters));
  large_alloc = PEEK(large_alloc);
  EXPECT_FALSE(IsLeaked((void *)large_alloc,
      __lsan::kSourceAllAligned & ~__lsan::kSourceRegisters));
  free((void *)large_alloc);
  large_alloc = NULL;
}

// Multi-threaded tests.
struct ThreadArgument {
  void sync_wait(uptr value) {
    while (atomic_load(&sync, memory_order_seq_cst) != value)
      pthread_yield();
  }
  void sync_store(uptr value) {
    atomic_store(&sync, value, memory_order_seq_cst);
  }

  void *hidden_p;
  atomic_uintptr_t sync;
};

void *StackThreadFunc(void *param) {
  ThreadArgument *arg = reinterpret_cast<ThreadArgument *>(param);
  void *p = malloc(kSmallAllocSize);
  // Take p's address to ensure it's not optimized into a register.
  void * volatile *pp = &p;
  arg->hidden_p = HIDE(*pp);
  arg->sync_store(1);
  arg->sync_wait(2);
  free(p);
  return NULL;
}

void *RegistersThreadFunc(void *param) {
  ThreadArgument *arg = reinterpret_cast<ThreadArgument *>(param);
  // To store the pointer, choose a register which is unlikely to be reused by
  // a function call.
#if defined(__i386__)
  register void* p asm("esi");
#elif defined(__x86_64__)
  register void* p asm("r15");
#else
  register void* p;
#endif
  p = malloc(kSmallAllocSize);
  arg->hidden_p = HIDE(p);
  arg->sync_store(1);
  arg->sync_wait(2);
  free(p);
  return NULL;
}

void MultiThreadedTest(uptr source) {
  uptr other_source;
  void *(*func)(void *arg);
  if (source == __lsan::kSourceStacks) {
    func = StackThreadFunc;
    other_source = __lsan::kSourceRegisters;
  } else if (source == __lsan::kSourceRegisters) {
    func = RegistersThreadFunc;
    other_source = __lsan::kSourceStacks;
  } else {
    FAIL();
  }
  uptr baseline = __lsan::kSourceAllAligned & ~other_source;
  ThreadArgument arg;
  arg.sync_store(0);
  pthread_t thread_id;
  ASSERT_EQ(0, pthread_create(&thread_id, NULL, func, &arg));
  arg.sync_wait(1);
  EXPECT_NOT_LEAKED(PEEK(arg.hidden_p), baseline);
  EXPECT_LEAKED(PEEK(arg.hidden_p), baseline & ~source);
  EXPECT_NOT_LEAKED(PEEK(arg.hidden_p), baseline);
  arg.sync_store(2);
  ASSERT_EQ(0, pthread_join(thread_id, NULL));
}

TEST(LeakSanitizer, ThreadStacks) {
  MultiThreadedTest(__lsan::kSourceStacks);
}

TEST(LeakSanitizer, ThreadRegisters) {
  MultiThreadedTest(__lsan::kSourceRegisters);
}

// End of tests for pointer sources.

TEST(LeakSanitizer, UnalignedPointers) {
  // Static so we can disable stack.
  static uptr arr[2];
  char *char_arr = (char *)arr;
  void *p = malloc(kSmallAllocSize);
  memcpy(char_arr + 1, &p, sizeof(uptr));
  EXPECT_LEAKED(p, kAllButStackAndRegisters);
  EXPECT_NOT_LEAKED(p, kAllButStackAndRegisters | __lsan::kSourceUnaligned);
  EXPECT_LEAKED(p, kAllButStackAndRegisters);
  free(p);
}

}  // namespace

int main(int argc, char **argv) {
  global_argv = argv;
  __lsan::Init();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif  // SANITIZER_LINUX && defined(__x86_64__)
