/*===- InstrProfilingMerge.c - Profile in-process Merging  ---------------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
|*===----------------------------------------------------------------------===*
|* This file defines the API needed for in-process merging of profile data
|* stored in memory buffer.
\*===---------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingUtil.h"

#define INSTR_PROF_VALUE_PROF_DATA
#include "profile/InstrProfData.inc"

COMPILER_RT_VISIBILITY
void (*VPMergeHook)(ValueProfData *, __llvm_profile_data *);

COMPILER_RT_VISIBILITY
uint64_t lprofGetLoadModuleSignature(void) {
  /* A very fast way to compute a module signature.  */
  uint64_t Version = __llvm_profile_get_version();
  uint64_t NumCounters = __llvm_profile_get_num_counters(
      __llvm_profile_begin_counters(), __llvm_profile_end_counters());
  uint64_t NumData = __llvm_profile_get_num_data(__llvm_profile_begin_data(),
                                                 __llvm_profile_end_data());
  uint64_t NamesSize =
      (uint64_t)(__llvm_profile_end_names() - __llvm_profile_begin_names());
  uint64_t NumVnodes =
      (uint64_t)(__llvm_profile_end_vnodes() - __llvm_profile_begin_vnodes());
  const __llvm_profile_data *FirstD = __llvm_profile_begin_data();

  return (NamesSize << 40) + (NumCounters << 30) + (NumData << 20) +
         (NumVnodes << 10) + (NumData > 0 ? FirstD->NameRef : 0) + Version +
         __llvm_profile_get_magic();
}

/* Returns 1 if profile is not structurally compatible.  */
COMPILER_RT_VISIBILITY
int __llvm_profile_check_compatibility(const char *ProfileData,
                                       uint64_t ProfileSize) {
  /* Check profile header only for now  */
  __llvm_profile_header *Header = (__llvm_profile_header *)ProfileData;
  __llvm_profile_data *SrcDataStart, *SrcDataEnd, *SrcData, *DstData;
  SrcDataStart =
      (__llvm_profile_data *)(ProfileData + sizeof(__llvm_profile_header) +
                              Header->BinaryIdsSize);
  SrcDataEnd = SrcDataStart + Header->DataSize;

  if (ProfileSize < sizeof(__llvm_profile_header))
    return 1;

  /* Check the header first.  */
  if (Header->Magic != __llvm_profile_get_magic() ||
      Header->Version != __llvm_profile_get_version() ||
      Header->DataSize !=
          __llvm_profile_get_num_data(__llvm_profile_begin_data(),
                                      __llvm_profile_end_data()) ||
      Header->CountersSize !=
          __llvm_profile_get_num_counters(__llvm_profile_begin_counters(),
                                          __llvm_profile_end_counters()) ||
      Header->NamesSize != (uint64_t)(__llvm_profile_end_names() -
                                      __llvm_profile_begin_names()) ||
      Header->ValueKindLast != IPVK_Last)
    return 1;

  if (ProfileSize <
      sizeof(__llvm_profile_header) + Header->BinaryIdsSize +
          Header->DataSize * sizeof(__llvm_profile_data) + Header->NamesSize +
          Header->CountersSize * __llvm_profile_counter_entry_size())
    return 1;

  for (SrcData = SrcDataStart,
       DstData = (__llvm_profile_data *)__llvm_profile_begin_data();
       SrcData < SrcDataEnd; ++SrcData, ++DstData) {
    if (SrcData->NameRef != DstData->NameRef ||
        SrcData->FuncHash != DstData->FuncHash ||
        SrcData->NumCounters != DstData->NumCounters)
      return 1;
  }

  /* Matched! */
  return 0;
}

static uintptr_t signextIfWin64(void *V) {
#ifdef _WIN64
  return (uintptr_t)(int32_t)(uintptr_t)V;
#else
  return (uintptr_t)V;
#endif
}

COMPILER_RT_VISIBILITY
int __llvm_profile_merge_from_buffer(const char *ProfileData,
                                     uint64_t ProfileSize) {
  if (__llvm_profile_get_version() & VARIANT_MASK_DBG_CORRELATE) {
    PROF_ERR(
        "%s\n",
        "Debug info correlation does not support profile merging at runtime. "
        "Instead, merge raw profiles using the llvm-profdata tool.");
    return 1;
  }

  __llvm_profile_data *SrcDataStart, *SrcDataEnd, *SrcData, *DstData;
  __llvm_profile_header *Header = (__llvm_profile_header *)ProfileData;
  char *SrcCountersStart;
  const char *SrcNameStart;
  const char *SrcValueProfDataStart, *SrcValueProfData;
  uintptr_t CountersDelta = Header->CountersDelta;

  SrcDataStart =
      (__llvm_profile_data *)(ProfileData + sizeof(__llvm_profile_header) +
                              Header->BinaryIdsSize);
  SrcDataEnd = SrcDataStart + Header->DataSize;
  SrcCountersStart = (char *)SrcDataEnd;
  SrcNameStart = SrcCountersStart +
                 Header->CountersSize * __llvm_profile_counter_entry_size();
  SrcValueProfDataStart =
      SrcNameStart + Header->NamesSize +
      __llvm_profile_get_num_padding_bytes(Header->NamesSize);
  if (SrcNameStart < SrcCountersStart)
    return 1;

  for (SrcData = SrcDataStart,
      DstData = (__llvm_profile_data *)__llvm_profile_begin_data(),
      SrcValueProfData = SrcValueProfDataStart;
       SrcData < SrcDataEnd; ++SrcData, ++DstData) {
    // For the in-memory destination, CounterPtr is the distance from the start
    // address of the data to the start address of the counter. On WIN64,
    // CounterPtr is a truncated 32-bit value due to COFF limitation. Sign
    // extend CounterPtr to get the original value.
    char *DstCounters =
        (char *)((uintptr_t)DstData + signextIfWin64(DstData->CounterPtr));
    unsigned NVK = 0;

    // SrcData is a serialized representation of the memory image. We need to
    // compute the in-buffer counter offset from the in-memory address distance.
    // The initial CountersDelta is the in-memory address difference
    // start(__llvm_prf_cnts)-start(__llvm_prf_data), so SrcData->CounterPtr -
    // CountersDelta computes the offset into the in-buffer counter section.
    //
    // On WIN64, CountersDelta is truncated as well, so no need for signext.
    char *SrcCounters =
        SrcCountersStart + ((uintptr_t)SrcData->CounterPtr - CountersDelta);
    // CountersDelta needs to be decreased as we advance to the next data
    // record.
    CountersDelta -= sizeof(*SrcData);
    unsigned NC = SrcData->NumCounters;
    if (NC == 0)
      return 1;
    if (SrcCounters < SrcCountersStart || SrcCounters >= SrcNameStart ||
        (SrcCounters + __llvm_profile_counter_entry_size() * NC) > SrcNameStart)
      return 1;
    for (unsigned I = 0; I < NC; I++) {
      if (__llvm_profile_get_version() & VARIANT_MASK_BYTE_COVERAGE) {
        // A value of zero signifies the function is covered.
        DstCounters[I] &= SrcCounters[I];
      } else {
        ((uint64_t *)DstCounters)[I] += ((uint64_t *)SrcCounters)[I];
      }
    }

    /* Now merge value profile data. */
    if (!VPMergeHook)
      continue;

    for (unsigned I = 0; I <= IPVK_Last; I++)
      NVK += (SrcData->NumValueSites[I] != 0);

    if (!NVK)
      continue;

    if (SrcValueProfData >= ProfileData + ProfileSize)
      return 1;
    VPMergeHook((ValueProfData *)SrcValueProfData, DstData);
    SrcValueProfData =
        SrcValueProfData + ((ValueProfData *)SrcValueProfData)->TotalSize;
  }

  return 0;
}
