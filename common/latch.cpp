// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/latch.h"

#include "common/check.h"

namespace Carbon {

auto Latch::Inc() -> void {
  // The increment must be _atomic_ but is _relaxed_.
  //
  // Increments and decrements can happen concurrently on separate threads, so
  // we need to prevent tearing and for there to be a total ordering of stores
  // to this atomic.
  //
  // However we provide no _synchronization_ of the increment with any other
  // operations. Instead, the caller must provide some extrinsic happens-before
  // between its call to `Inc` and its later call to `Dec`. When that call to
  // `Dec` synchronizes-with another call to `Dec`, all relaxed stores are
  // covered by the resulting inter-thread happens-before relationship.
  count_.fetch_add(1, std::memory_order_relaxed);
}

auto Latch::Dec() -> bool {
  // The decrement is both an _acquire_ and _release_ operation.
  //
  // All threads which decrement to a non-zero value need to synchronize-with
  // the thread which decrements to a zero value. This means the decrements to
  // non-zero values need to have _release_ semantics that are _acquired_ by the
  // decrement to zero. Since there is a single decrement operation, it must be
  // both _acquire_ and _release_.
  //
  // Note that this technically provides a stronger guarantee than the contract
  // of `Dec` requires -- *all* decrements synchronize with all decrements whose
  // value they observe, we only need that to be true of the decrement arriving
  // at zero. This could in theory be modeled by conditional fences, but those
  // have their own problems and we don't need to model the more precise
  // semantics for efficiency.
  auto previous = count_.fetch_sub(1, std::memory_order_acq_rel);

  CARBON_CHECK(previous > 0);
  if (previous == 1) {
    // Ensure that our closure is fully destroyed here, releasing any
    // resources, locks, or other synchronization primitives.
    auto on_zero = std::exchange(on_zero_, [] {});
    std::move(on_zero)();
    return true;
  }
  return false;
}

auto Latch::Init(llvm::unique_function<auto()->void> on_zero) -> Handle {
  CARBON_CHECK(count_ == 0);
  on_zero_ = std::move(on_zero);
  return Handle(this);
}

}  // namespace Carbon
