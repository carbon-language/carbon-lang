//===-- tsan_sync.h ---------------------------------------------*- C++ -*-===//
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
#ifndef TSAN_SYNC_H
#define TSAN_SYNC_H

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_deadlock_detector_interface.h"
#include "tsan_clock.h"
#include "tsan_defs.h"
#include "tsan_mutex.h"

namespace __tsan {

class SlabCache;

class StackTrace {
 public:
  StackTrace();
  // Initialized the object in "static mode",
  // in this mode it never calls malloc/free but uses the provided buffer.
  StackTrace(uptr *buf, uptr cnt);
  ~StackTrace();
  void Reset();

  void Init(const uptr *pcs, uptr cnt);
  void ObtainCurrent(ThreadState *thr, uptr toppc);
  bool IsEmpty() const;
  uptr Size() const;
  uptr Get(uptr i) const;
  const uptr *Begin() const;
  void CopyFrom(const StackTrace& other);

 private:
  uptr n_;
  uptr *s_;
  const uptr c_;

  StackTrace(const StackTrace&);
  void operator = (const StackTrace&);
};

struct SyncVar {
  explicit SyncVar(uptr addr, u64 uid);

  static const int kInvalidTid = -1;

  Mutex mtx;
  uptr addr;
  const u64 uid;  // Globally unique id.
  SyncClock clock;
  SyncClock read_clock;  // Used for rw mutexes only.
  u32 creation_stack_id;
  int owner_tid;  // Set only by exclusive owners.
  u64 last_lock;
  int recursion;
  bool is_rw;
  bool is_recursive;
  bool is_broken;
  bool is_linker_init;
  SyncVar *next;  // In SyncTab hashtable.
  DDMutex dd;

  uptr GetMemoryConsumption();
  u64 GetId() const {
    // 47 lsb is addr, then 14 bits is low part of uid, then 3 zero bits.
    return GetLsb((u64)addr | (uid << 47), 61);
  }
  bool CheckId(u64 uid) const {
    CHECK_EQ(uid, GetLsb(uid, 14));
    return GetLsb(this->uid, 14) == uid;
  }
  static uptr SplitId(u64 id, u64 *uid) {
    *uid = id >> 47;
    return (uptr)GetLsb(id, 47);
  }
};

class SyncTab {
 public:
  SyncTab();
  ~SyncTab();

  SyncVar* GetOrCreateAndLock(ThreadState *thr, uptr pc,
                              uptr addr, bool write_lock);
  SyncVar* GetIfExistsAndLock(uptr addr, bool write_lock);

  // If the SyncVar does not exist, returns 0.
  SyncVar* GetAndRemove(ThreadState *thr, uptr pc, uptr addr);

  SyncVar* Create(ThreadState *thr, uptr pc, uptr addr);

  uptr GetMemoryConsumption(uptr *nsync);

 private:
  struct Part {
    Mutex mtx;
    SyncVar *val;
    char pad[kCacheLineSize - sizeof(Mutex) - sizeof(SyncVar*)];  // NOLINT
    Part();
  };

  // FIXME: Implement something more sane.
  static const int kPartCount = 1009;
  Part tab_[kPartCount];
  atomic_uint64_t uid_gen_;

  int PartIdx(uptr addr);

  SyncVar* GetAndLock(ThreadState *thr, uptr pc,
                      uptr addr, bool write_lock, bool create);

  SyncTab(const SyncTab&);  // Not implemented.
  void operator = (const SyncTab&);  // Not implemented.
};

}  // namespace __tsan

#endif  // TSAN_SYNC_H
