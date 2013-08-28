//===-- sanitizer_stackdepot.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_STACKDEPOT_H
#define SANITIZER_STACKDEPOT_H

#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"

namespace __sanitizer {

// StackDepot efficiently stores huge amounts of stack traces.

// Maps stack trace to an unique id.
u32 StackDepotPut(const uptr *stack, uptr size);
// Retrieves a stored stack trace by the id.
const uptr *StackDepotGet(u32 id, uptr *size);

struct StackDepotStats {
  uptr n_uniq_ids;
  uptr mapped;
};

StackDepotStats *StackDepotGetStats();

struct StackDesc;

// Instantiating this class creates a snapshot of StackDepot which can be
// efficiently queried with StackDepotGet(). You can use it concurrently with
// StackDepot, but the snapshot is only guaranteed to contain those stack traces
// which were stored before it was instantiated.
class StackDepotReverseMap {
 public:
  StackDepotReverseMap();
  const uptr *Get(u32 id, uptr *size);

 private:
  struct IdDescPair {
    u32 id;
    StackDesc *desc;

    static bool IdComparator(const IdDescPair &a, const IdDescPair &b);
  };

  InternalMmapVector<IdDescPair> map_;

  // Disallow evil constructors.
  StackDepotReverseMap(const StackDepotReverseMap&);
  void operator=(const StackDepotReverseMap&);
};
}  // namespace __sanitizer

#endif  // SANITIZER_STACKDEPOT_H
