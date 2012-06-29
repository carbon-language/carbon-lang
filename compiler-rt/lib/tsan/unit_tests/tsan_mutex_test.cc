//===-- tsan_mutex_test.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "tsan_mutex.h"
#include "gtest/gtest.h"

namespace __tsan {

template<typename MutexType>
class TestData {
 public:
  explicit TestData(MutexType *mtx)
    : mtx_(mtx) {
    for (int i = 0; i < kSize; i++)
      data_[i] = 0;
  }

  void Write() {
    Lock l(mtx_);
    T v0 = data_[0];
    for (int i = 0; i < kSize; i++) {
      CHECK_EQ(data_[i], v0);
      data_[i]++;
    }
  }

  void Read() {
    ReadLock l(mtx_);
    T v0 = data_[0];
    for (int i = 0; i < kSize; i++) {
      CHECK_EQ(data_[i], v0);
    }
  }

  void Backoff() {
    volatile T data[kSize] = {};
    for (int i = 0; i < kSize; i++) {
      data[i]++;
      CHECK_EQ(data[i], 1);
    }
  }

 private:
  typedef GenericScopedLock<MutexType> Lock;
  static const int kSize = 64;
  typedef u64 T;
  MutexType *mtx_;
  char pad_[kCacheLineSize];
  T data_[kSize];
};

const int kThreads = 8;
const int kWriteRate = 1024;
#if TSAN_DEBUG
const int kIters = 16*1024;
#else
const int kIters = 64*1024;
#endif

template<typename MutexType>
static void *write_mutex_thread(void *param) {
  TestData<MutexType> *data = (TestData<MutexType>*)param;
  for (int i = 0; i < kIters; i++) {
    data->Write();
    data->Backoff();
  }
  return 0;
}

template<typename MutexType>
static void *read_mutex_thread(void *param) {
  TestData<MutexType> *data = (TestData<MutexType>*)param;
  for (int i = 0; i < kIters; i++) {
    if ((i % kWriteRate) == 0)
      data->Write();
    else
      data->Read();
    data->Backoff();
  }
  return 0;
}

TEST(Mutex, Write) {
  Mutex mtx(MutexTypeAnnotations, StatMtxAnnotations);
  TestData<Mutex> data(&mtx);
  pthread_t threads[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&threads[i], 0, write_mutex_thread<Mutex>, &data);
  for (int i = 0; i < kThreads; i++)
    pthread_join(threads[i], 0);
}

TEST(Mutex, ReadWrite) {
  Mutex mtx(MutexTypeAnnotations, StatMtxAnnotations);
  TestData<Mutex> data(&mtx);
  pthread_t threads[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&threads[i], 0, read_mutex_thread<Mutex>, &data);
  for (int i = 0; i < kThreads; i++)
    pthread_join(threads[i], 0);
}

TEST(Mutex, SpinWrite) {
  SpinMutex mtx;
  TestData<SpinMutex> data(&mtx);
  pthread_t threads[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&threads[i], 0, write_mutex_thread<SpinMutex>, &data);
  for (int i = 0; i < kThreads; i++)
    pthread_join(threads[i], 0);
}

}  // namespace __tsan
