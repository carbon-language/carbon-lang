/*===- InstrProfiling.c - Support library for PGO instrumentation ---------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

__attribute__((visibility("hidden"))) uint64_t __llvm_profile_get_magic(void) {
  /* Magic number to detect file format and endianness.
   *
   * Use 255 at one end, since no UTF-8 file can use that character.  Avoid 0,
   * so that utilities, like strings, don't grab it as a string.  129 is also
   * invalid UTF-8, and high enough to be interesting.
   *
   * Use "lprofr" in the centre to stand for "LLVM Profile Raw", or "lprofR"
   * for 32-bit platforms.
   */
  unsigned char R = sizeof(void *) == sizeof(uint64_t) ? 'r' : 'R';
  return (uint64_t)255 << 56 | (uint64_t)'l' << 48 | (uint64_t)'p' << 40 |
         (uint64_t)'r' << 32 | (uint64_t)'o' << 24 | (uint64_t)'f' << 16 |
         (uint64_t)R << 8 | (uint64_t)129;
}

/* Return the number of bytes needed to add to SizeInBytes to make it
 *   the result a multiple of 8.
 */
__attribute__((visibility("hidden"))) uint8_t
__llvm_profile_get_num_padding_bytes(uint64_t SizeInBytes) {
  return 7 & (sizeof(uint64_t) - SizeInBytes % sizeof(uint64_t));
}

__attribute__((visibility("hidden"))) uint64_t
__llvm_profile_get_version(void) {
  /* This should be bumped any time the output format changes. */
  return 2;
}

__attribute__((visibility("hidden"))) void __llvm_profile_reset_counters(void) {
  uint64_t *I = __llvm_profile_begin_counters();
  uint64_t *E = __llvm_profile_end_counters();

  memset(I, 0, sizeof(uint64_t) * (E - I));

  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const __llvm_profile_data *DI;
  for (DI = DataBegin; DI != DataEnd; ++DI) {
    uint64_t CurrentVSiteCount = 0;
    uint32_t VKI, i;
    if (!DI->ValueCounters)
      continue;

    for (VKI = VK_FIRST; VKI <= VK_LAST; ++VKI)
      CurrentVSiteCount += DI->NumValueSites[VKI];

    for (i = 0; i < CurrentVSiteCount; ++i) {
      __llvm_profile_value_node *CurrentVNode = DI->ValueCounters[i];

      while (CurrentVNode) {
        CurrentVNode->VData.NumTaken = 0;
        CurrentVNode = CurrentVNode->Next;
      }
    }
  }
}

/* Total number of value profile data in bytes. */
static uint64_t TotalValueDataSize = 0;

#ifdef _MIPS_ARCH
__attribute__((visibility("hidden"))) void
__llvm_profile_instrument_target(uint64_t TargetValue, void *Data_,
                                 uint32_t CounterIndex) {
}

#else

/* Allocate an array that holds the pointers to the linked lists of
 * value profile counter nodes. The number of element of the array
 * is the total number of value profile sites instrumented. Returns
 *  0 if allocation fails.
 */

static int allocateValueProfileCounters(__llvm_profile_data *Data) {
  uint64_t NumVSites = 0;
  uint32_t VKI;
  for (VKI = VK_FIRST; VKI <= VK_LAST; ++VKI)
    NumVSites += Data->NumValueSites[VKI];

  __llvm_profile_value_node **Mem = (__llvm_profile_value_node **)calloc(
      NumVSites, sizeof(__llvm_profile_value_node *));
  if (!Mem)
    return 0;
  if (!__sync_bool_compare_and_swap(&Data->ValueCounters, 0, Mem)) {
    free(Mem);
    return 0;
  }
  /*  In the raw format, there will be an value count array preceding
   *  the value profile data. The element type of the array is uint8_t,
   *  and there is one element in array per value site. The element
   *  stores the number of values profiled for the corresponding site.
   */
  uint8_t Padding = __llvm_profile_get_num_padding_bytes(NumVSites);
  __sync_fetch_and_add(&TotalValueDataSize, NumVSites + Padding);
  return 1;
}

__attribute__((visibility("hidden"))) void
__llvm_profile_instrument_target(uint64_t TargetValue, void *Data_,
                                 uint32_t CounterIndex) {

  __llvm_profile_data *Data = (__llvm_profile_data *)Data_;
  if (!Data)
    return;

  if (!Data->ValueCounters) {
    if (!allocateValueProfileCounters(Data))
      return;
  }

  __llvm_profile_value_node *PrevVNode = NULL;
  __llvm_profile_value_node *CurrentVNode = Data->ValueCounters[CounterIndex];

  uint8_t VDataCount = 0;
  while (CurrentVNode) {
    if (TargetValue == CurrentVNode->VData.TargetValue) {
      CurrentVNode->VData.NumTaken++;
      return;
    }
    PrevVNode = CurrentVNode;
    CurrentVNode = CurrentVNode->Next;
    ++VDataCount;
  }

  if (VDataCount >= UCHAR_MAX)
    return;

  CurrentVNode =
      (__llvm_profile_value_node *)calloc(1, sizeof(__llvm_profile_value_node));
  if (!CurrentVNode)
    return;

  CurrentVNode->VData.TargetValue = TargetValue;
  CurrentVNode->VData.NumTaken++;

  uint32_t Success = 0;
  if (!Data->ValueCounters[CounterIndex])
    Success = __sync_bool_compare_and_swap(&(Data->ValueCounters[CounterIndex]),
                                           0, CurrentVNode);
  else if (PrevVNode && !PrevVNode->Next)
    Success = __sync_bool_compare_and_swap(&(PrevVNode->Next), 0, CurrentVNode);

  if (!Success) {
    free(CurrentVNode);
    return;
  }
  __sync_fetch_and_add(&TotalValueDataSize,
                       Success * sizeof(__llvm_profile_value_data));
}
#endif

__attribute__((visibility("hidden"))) uint64_t
__llvm_profile_gather_value_data(uint8_t **VDataArray) {

  if (!VDataArray || 0 == TotalValueDataSize)
    return 0;

  uint64_t NumData = TotalValueDataSize;
  *VDataArray = (uint8_t *)calloc(NumData, sizeof(uint8_t));
  if (!*VDataArray)
    return 0;

  uint8_t *VDataEnd = *VDataArray + NumData;
  uint8_t *PerSiteCountsHead = *VDataArray;
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  __llvm_profile_data *I;
  for (I = (__llvm_profile_data *)DataBegin; I != DataEnd; ++I) {

    uint64_t NumVSites = 0;
    uint32_t VKI, i;

    if (!I->ValueCounters)
      continue;

    for (VKI = VK_FIRST; VKI <= VK_LAST; ++VKI)
      NumVSites += I->NumValueSites[VKI];
    uint8_t Padding = __llvm_profile_get_num_padding_bytes(NumVSites);

    uint8_t *PerSiteCountPtr = PerSiteCountsHead;
    __llvm_profile_value_data *VDataPtr =
        (__llvm_profile_value_data *)(PerSiteCountPtr + NumVSites + Padding);

    for (i = 0; i < NumVSites; ++i) {

      __llvm_profile_value_node *VNode = I->ValueCounters[i];

      uint8_t VDataCount = 0;
      while (VNode && ((uint8_t *)(VDataPtr + 1) <= VDataEnd)) {
        *VDataPtr = VNode->VData;
        VNode = VNode->Next;
        ++VDataPtr;
        if (++VDataCount == UCHAR_MAX)
          break;
      }
      *PerSiteCountPtr = VDataCount;
      ++PerSiteCountPtr;
    }
    I->ValueCounters = (void *)PerSiteCountsHead;
    PerSiteCountsHead = (uint8_t *)VDataPtr;
  }
  return PerSiteCountsHead - *VDataArray;
}
