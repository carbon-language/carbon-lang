//===-- SBQueue.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cinttypes>

#include "SBReproducerPrivate.h"
#include "lldb/API/SBQueue.h"

#include "lldb/API/SBProcess.h"
#include "lldb/API/SBQueueItem.h"
#include "lldb/API/SBThread.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Queue.h"
#include "lldb/Target/QueueItem.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {

class QueueImpl {
public:
  QueueImpl() : m_queue_wp(), m_threads(), m_pending_items() {}

  QueueImpl(const lldb::QueueSP &queue_sp)
      : m_queue_wp(), m_threads(), m_thread_list_fetched(false),
        m_pending_items(), m_pending_items_fetched(false) {
    m_queue_wp = queue_sp;
  }

  QueueImpl(const QueueImpl &rhs) {
    if (&rhs == this)
      return;
    m_queue_wp = rhs.m_queue_wp;
    m_threads = rhs.m_threads;
    m_thread_list_fetched = rhs.m_thread_list_fetched;
    m_pending_items = rhs.m_pending_items;
    m_pending_items_fetched = rhs.m_pending_items_fetched;
  }

  ~QueueImpl() = default;

  bool IsValid() { return m_queue_wp.lock() != nullptr; }

  void Clear() {
    m_queue_wp.reset();
    m_thread_list_fetched = false;
    m_threads.clear();
    m_pending_items_fetched = false;
    m_pending_items.clear();
  }

  void SetQueue(const lldb::QueueSP &queue_sp) {
    Clear();
    m_queue_wp = queue_sp;
  }

  lldb::queue_id_t GetQueueID() const {
    lldb::queue_id_t result = LLDB_INVALID_QUEUE_ID;
    lldb::QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp) {
      result = queue_sp->GetID();
    }
    return result;
  }

  uint32_t GetIndexID() const {
    uint32_t result = LLDB_INVALID_INDEX32;
    lldb::QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp) {
      result = queue_sp->GetIndexID();
    }
    return result;
  }

  const char *GetName() const {
    const char *name = nullptr;
    lldb::QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp.get()) {
      name = queue_sp->GetName();
    }
    return name;
  }

  void FetchThreads() {
    if (!m_thread_list_fetched) {
      lldb::QueueSP queue_sp = m_queue_wp.lock();
      if (queue_sp) {
        Process::StopLocker stop_locker;
        if (stop_locker.TryLock(&queue_sp->GetProcess()->GetRunLock())) {
          const std::vector<ThreadSP> thread_list(queue_sp->GetThreads());
          m_thread_list_fetched = true;
          const uint32_t num_threads = thread_list.size();
          for (uint32_t idx = 0; idx < num_threads; ++idx) {
            ThreadSP thread_sp = thread_list[idx];
            if (thread_sp && thread_sp->IsValid()) {
              m_threads.push_back(thread_sp);
            }
          }
        }
      }
    }
  }

  void FetchItems() {
    if (!m_pending_items_fetched) {
      QueueSP queue_sp = m_queue_wp.lock();
      if (queue_sp) {
        Process::StopLocker stop_locker;
        if (stop_locker.TryLock(&queue_sp->GetProcess()->GetRunLock())) {
          const std::vector<QueueItemSP> queue_items(
              queue_sp->GetPendingItems());
          m_pending_items_fetched = true;
          const uint32_t num_pending_items = queue_items.size();
          for (uint32_t idx = 0; idx < num_pending_items; ++idx) {
            QueueItemSP item = queue_items[idx];
            if (item && item->IsValid()) {
              m_pending_items.push_back(item);
            }
          }
        }
      }
    }
  }

  uint32_t GetNumThreads() {
    uint32_t result = 0;

    FetchThreads();
    if (m_thread_list_fetched) {
      result = m_threads.size();
    }
    return result;
  }

  lldb::SBThread GetThreadAtIndex(uint32_t idx) {
    FetchThreads();

    SBThread sb_thread;
    QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp && idx < m_threads.size()) {
      ProcessSP process_sp = queue_sp->GetProcess();
      if (process_sp) {
        ThreadSP thread_sp = m_threads[idx].lock();
        if (thread_sp) {
          sb_thread.SetThread(thread_sp);
        }
      }
    }
    return sb_thread;
  }

  uint32_t GetNumPendingItems() {
    uint32_t result = 0;

    QueueSP queue_sp = m_queue_wp.lock();
    if (!m_pending_items_fetched && queue_sp) {
      result = queue_sp->GetNumPendingWorkItems();
    } else {
      result = m_pending_items.size();
    }
    return result;
  }

  lldb::SBQueueItem GetPendingItemAtIndex(uint32_t idx) {
    SBQueueItem result;
    FetchItems();
    if (m_pending_items_fetched && idx < m_pending_items.size()) {
      result.SetQueueItem(m_pending_items[idx]);
    }
    return result;
  }

  uint32_t GetNumRunningItems() {
    uint32_t result = 0;
    QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp)
      result = queue_sp->GetNumRunningWorkItems();
    return result;
  }

  lldb::SBProcess GetProcess() {
    SBProcess result;
    QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp) {
      result.SetSP(queue_sp->GetProcess());
    }
    return result;
  }

  lldb::QueueKind GetKind() {
    lldb::QueueKind kind = eQueueKindUnknown;
    QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp)
      kind = queue_sp->GetKind();

    return kind;
  }

private:
  lldb::QueueWP m_queue_wp;
  std::vector<lldb::ThreadWP>
      m_threads; // threads currently executing this queue's items
  bool m_thread_list_fetched =
      false; // have we tried to fetch the threads list already?
  std::vector<lldb::QueueItemSP> m_pending_items; // items currently enqueued
  bool m_pending_items_fetched =
      false; // have we tried to fetch the item list already?
};
}

SBQueue::SBQueue() : m_opaque_sp(new QueueImpl()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBQueue);
}

SBQueue::SBQueue(const QueueSP &queue_sp)
    : m_opaque_sp(new QueueImpl(queue_sp)) {
  LLDB_RECORD_CONSTRUCTOR(SBQueue, (const lldb::QueueSP &), queue_sp);
}

SBQueue::SBQueue(const SBQueue &rhs) {
  LLDB_RECORD_CONSTRUCTOR(SBQueue, (const lldb::SBQueue &), rhs);

  if (&rhs == this)
    return;

  m_opaque_sp = rhs.m_opaque_sp;
}

const lldb::SBQueue &SBQueue::operator=(const lldb::SBQueue &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBQueue &,
                     SBQueue, operator=,(const lldb::SBQueue &), rhs);

  m_opaque_sp = rhs.m_opaque_sp;
  return LLDB_RECORD_RESULT(*this);
}

SBQueue::~SBQueue() = default;

bool SBQueue::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBQueue, IsValid);
  return this->operator bool();
}
SBQueue::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBQueue, operator bool);

  return m_opaque_sp->IsValid();
}

void SBQueue::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBQueue, Clear);

  m_opaque_sp->Clear();
}

void SBQueue::SetQueue(const QueueSP &queue_sp) {
  m_opaque_sp->SetQueue(queue_sp);
}

lldb::queue_id_t SBQueue::GetQueueID() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::queue_id_t, SBQueue, GetQueueID);

  return m_opaque_sp->GetQueueID();
}

uint32_t SBQueue::GetIndexID() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBQueue, GetIndexID);

  uint32_t index_id = m_opaque_sp->GetIndexID();
  return index_id;
}

const char *SBQueue::GetName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBQueue, GetName);

  return m_opaque_sp->GetName();
}

uint32_t SBQueue::GetNumThreads() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBQueue, GetNumThreads);

  return m_opaque_sp->GetNumThreads();
}

SBThread SBQueue::GetThreadAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::SBThread, SBQueue, GetThreadAtIndex, (uint32_t),
                     idx);

  SBThread th = m_opaque_sp->GetThreadAtIndex(idx);
  return LLDB_RECORD_RESULT(th);
}

uint32_t SBQueue::GetNumPendingItems() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBQueue, GetNumPendingItems);

  return m_opaque_sp->GetNumPendingItems();
}

SBQueueItem SBQueue::GetPendingItemAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::SBQueueItem, SBQueue, GetPendingItemAtIndex,
                     (uint32_t), idx);

  return LLDB_RECORD_RESULT(m_opaque_sp->GetPendingItemAtIndex(idx));
}

uint32_t SBQueue::GetNumRunningItems() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBQueue, GetNumRunningItems);

  return m_opaque_sp->GetNumRunningItems();
}

SBProcess SBQueue::GetProcess() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBProcess, SBQueue, GetProcess);

  return LLDB_RECORD_RESULT(m_opaque_sp->GetProcess());
}

lldb::QueueKind SBQueue::GetKind() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::QueueKind, SBQueue, GetKind);

  return m_opaque_sp->GetKind();
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBQueue>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBQueue, ());
  LLDB_REGISTER_CONSTRUCTOR(SBQueue, (const lldb::QueueSP &));
  LLDB_REGISTER_CONSTRUCTOR(SBQueue, (const lldb::SBQueue &));
  LLDB_REGISTER_METHOD(const lldb::SBQueue &,
                       SBQueue, operator=,(const lldb::SBQueue &));
  LLDB_REGISTER_METHOD_CONST(bool, SBQueue, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBQueue, operator bool, ());
  LLDB_REGISTER_METHOD(void, SBQueue, Clear, ());
  LLDB_REGISTER_METHOD_CONST(lldb::queue_id_t, SBQueue, GetQueueID, ());
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBQueue, GetIndexID, ());
  LLDB_REGISTER_METHOD_CONST(const char *, SBQueue, GetName, ());
  LLDB_REGISTER_METHOD(uint32_t, SBQueue, GetNumThreads, ());
  LLDB_REGISTER_METHOD(lldb::SBThread, SBQueue, GetThreadAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(uint32_t, SBQueue, GetNumPendingItems, ());
  LLDB_REGISTER_METHOD(lldb::SBQueueItem, SBQueue, GetPendingItemAtIndex,
                       (uint32_t));
  LLDB_REGISTER_METHOD(uint32_t, SBQueue, GetNumRunningItems, ());
  LLDB_REGISTER_METHOD(lldb::SBProcess, SBQueue, GetProcess, ());
  LLDB_REGISTER_METHOD(lldb::QueueKind, SBQueue, GetKind, ());
}

}
}
