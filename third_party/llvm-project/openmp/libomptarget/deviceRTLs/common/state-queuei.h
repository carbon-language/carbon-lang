//===------- state-queuei.h - OpenMP GPU State Queue ------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of a queue to hand out OpenMP state
// objects to teams of one or more kernels.
//
// Reference:
// Thomas R.W. Scogland and Wu-chun Feng. 2015.
// Design and Evaluation of Scalable Concurrent Queues for Many-Core
// Architectures. International Conference on Performance Engineering.
//
//===----------------------------------------------------------------------===//

#include "state-queue.h"

template <typename ElementType, uint32_t SIZE>
INLINE uint32_t omptarget_nvptx_Queue<ElementType, SIZE>::ENQUEUE_TICKET() {
  return __kmpc_atomic_add((unsigned int *)&tail, 1u);
}

template <typename ElementType, uint32_t SIZE>
INLINE uint32_t omptarget_nvptx_Queue<ElementType, SIZE>::DEQUEUE_TICKET() {
  return __kmpc_atomic_add((unsigned int *)&head, 1u);
}

template <typename ElementType, uint32_t SIZE>
INLINE uint32_t omptarget_nvptx_Queue<ElementType, SIZE>::ID(uint32_t ticket) {
  return (ticket / SIZE) * 2;
}

template <typename ElementType, uint32_t SIZE>
INLINE bool omptarget_nvptx_Queue<ElementType, SIZE>::IsServing(uint32_t slot,
                                                                uint32_t id) {
  return __kmpc_atomic_add((unsigned int *)&ids[slot], 0u) == id;
}

template <typename ElementType, uint32_t SIZE>
INLINE void
omptarget_nvptx_Queue<ElementType, SIZE>::PushElement(uint32_t slot,
                                                      ElementType *element) {
  __kmpc_atomic_exchange((unsigned long long *)&elementQueue[slot],
                         (unsigned long long)element);
}

template <typename ElementType, uint32_t SIZE>
INLINE ElementType *
omptarget_nvptx_Queue<ElementType, SIZE>::PopElement(uint32_t slot) {
  return (ElementType *)__kmpc_atomic_add(
      (unsigned long long *)&elementQueue[slot], (unsigned long long)0);
}

template <typename ElementType, uint32_t SIZE>
INLINE void omptarget_nvptx_Queue<ElementType, SIZE>::DoneServing(uint32_t slot,
                                                                  uint32_t id) {
  __kmpc_atomic_exchange((unsigned int *)&ids[slot], (id + 1) % MAX_ID);
}

template <typename ElementType, uint32_t SIZE>
INLINE void
omptarget_nvptx_Queue<ElementType, SIZE>::Enqueue(ElementType *element) {
  uint32_t ticket = ENQUEUE_TICKET();
  uint32_t slot = ticket % SIZE;
  uint32_t id = ID(ticket) + 1;
  while (!IsServing(slot, id))
    ;
  PushElement(slot, element);
  DoneServing(slot, id);
}

template <typename ElementType, uint32_t SIZE>
INLINE ElementType *omptarget_nvptx_Queue<ElementType, SIZE>::Dequeue() {
  uint32_t ticket = DEQUEUE_TICKET();
  uint32_t slot = ticket % SIZE;
  uint32_t id = ID(ticket);
  while (!IsServing(slot, id))
    ;
  ElementType *element = PopElement(slot);
  // This is to populate the queue because of the lack of GPU constructors.
  if (element == 0)
    element = &elements[slot];
  DoneServing(slot, id);
  return element;
}
