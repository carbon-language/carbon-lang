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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define INSTR_PROF_VALUE_PROF_DATA
#define INSTR_PROF_COMMON_API_IMPL
#include "InstrProfData.inc"

#define PROF_OOM(Msg) PROF_ERR(Msg ":%s\n", "Out of memory");
#define PROF_OOM_RETURN(Msg)                                                   \
  {                                                                            \
    PROF_OOM(Msg)                                                              \
    return 0;                                                                  \
  }

#if COMPILER_RT_HAS_ATOMICS != 1
LLVM_LIBRARY_VISIBILITY
uint32_t BoolCmpXchg(void **Ptr, void *OldV, void *NewV) {
  void *R = *Ptr;
  if (R == OldV) {
    *Ptr = NewV;
    return 1;
  }
  return 0;
}
#endif

char *(*GetEnvHook)(const char *) = 0;

LLVM_LIBRARY_VISIBILITY uint64_t __llvm_profile_get_magic(void) {
  return sizeof(void *) == sizeof(uint64_t) ? (INSTR_PROF_RAW_MAGIC_64)
                                            : (INSTR_PROF_RAW_MAGIC_32);
}

/* Return the number of bytes needed to add to SizeInBytes to make it
 *   the result a multiple of 8.
 */
LLVM_LIBRARY_VISIBILITY uint8_t
__llvm_profile_get_num_padding_bytes(uint64_t SizeInBytes) {
  return 7 & (sizeof(uint64_t) - SizeInBytes % sizeof(uint64_t));
}

LLVM_LIBRARY_VISIBILITY uint64_t __llvm_profile_get_version(void) {
  return INSTR_PROF_RAW_VERSION;
}

LLVM_LIBRARY_VISIBILITY void __llvm_profile_reset_counters(void) {
  uint64_t *I = __llvm_profile_begin_counters();
  uint64_t *E = __llvm_profile_end_counters();

  memset(I, 0, sizeof(uint64_t) * (E - I));

  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const __llvm_profile_data *DI;
  for (DI = DataBegin; DI != DataEnd; ++DI) {
    uint64_t CurrentVSiteCount = 0;
    uint32_t VKI, i;
    if (!DI->Values)
      continue;

    ValueProfNode **ValueCounters = (ValueProfNode **)DI->Values;

    for (VKI = IPVK_First; VKI <= IPVK_Last; ++VKI)
      CurrentVSiteCount += DI->NumValueSites[VKI];

    for (i = 0; i < CurrentVSiteCount; ++i) {
      ValueProfNode *CurrentVNode = ValueCounters[i];

      while (CurrentVNode) {
        CurrentVNode->VData.Count = 0;
        CurrentVNode = CurrentVNode->Next;
      }
    }
  }
}

/* This method is only used in value profiler mock testing.  */
LLVM_LIBRARY_VISIBILITY void
__llvm_profile_set_num_value_sites(__llvm_profile_data *Data,
                                   uint32_t ValueKind, uint16_t NumValueSites) {
  *((uint16_t *)&Data->NumValueSites[ValueKind]) = NumValueSites;
}

/* This method is only used in value profiler mock testing.  */
LLVM_LIBRARY_VISIBILITY const __llvm_profile_data *
__llvm_profile_iterate_data(const __llvm_profile_data *Data) {
  return Data + 1;
}

/* This method is only used in value profiler mock testing.  */
LLVM_LIBRARY_VISIBILITY void *
__llvm_get_function_addr(const __llvm_profile_data *Data) {
  return Data->FunctionPointer;
}

/* Allocate an array that holds the pointers to the linked lists of
 * value profile counter nodes. The number of element of the array
 * is the total number of value profile sites instrumented. Returns
 * 0 if allocation fails.
 */

static int allocateValueProfileCounters(__llvm_profile_data *Data) {
  uint64_t NumVSites = 0;
  uint32_t VKI;
  for (VKI = IPVK_First; VKI <= IPVK_Last; ++VKI)
    NumVSites += Data->NumValueSites[VKI];

  ValueProfNode **Mem =
      (ValueProfNode **)calloc(NumVSites, sizeof(ValueProfNode *));
  if (!Mem)
    return 0;
  if (!BOOL_CMPXCHG(&Data->Values, 0, Mem)) {
    free(Mem);
    return 0;
  }
  return 1;
}

static void deallocateValueProfileCounters(__llvm_profile_data *Data) {
  uint64_t NumVSites = 0, I;
  uint32_t VKI;
  if (!Data->Values)
    return;
  for (VKI = IPVK_First; VKI <= IPVK_Last; ++VKI)
    NumVSites += Data->NumValueSites[VKI];
  for (I = 0; I < NumVSites; I++) {
    ValueProfNode *Node = ((ValueProfNode **)Data->Values)[I];
    while (Node) {
      ValueProfNode *Next = Node->Next;
      free(Node);
      Node = Next;
    }
  }
  free(Data->Values);
}

LLVM_LIBRARY_VISIBILITY void
__llvm_profile_instrument_target(uint64_t TargetValue, void *Data,
                                 uint32_t CounterIndex) {

  __llvm_profile_data *PData = (__llvm_profile_data *)Data;
  if (!PData)
    return;

  if (!PData->Values) {
    if (!allocateValueProfileCounters(PData))
      return;
  }

  ValueProfNode **ValueCounters = (ValueProfNode **)PData->Values;
  ValueProfNode *PrevVNode = NULL;
  ValueProfNode *CurrentVNode = ValueCounters[CounterIndex];

  uint8_t VDataCount = 0;
  while (CurrentVNode) {
    if (TargetValue == CurrentVNode->VData.Value) {
      CurrentVNode->VData.Count++;
      return;
    }
    PrevVNode = CurrentVNode;
    CurrentVNode = CurrentVNode->Next;
    ++VDataCount;
  }

  if (VDataCount >= UCHAR_MAX)
    return;

  CurrentVNode = (ValueProfNode *)calloc(1, sizeof(ValueProfNode));
  if (!CurrentVNode)
    return;

  CurrentVNode->VData.Value = TargetValue;
  CurrentVNode->VData.Count++;

  uint32_t Success = 0;
  if (!ValueCounters[CounterIndex])
    Success = BOOL_CMPXCHG(&ValueCounters[CounterIndex], 0, CurrentVNode);
  else if (PrevVNode && !PrevVNode->Next)
    Success = BOOL_CMPXCHG(&(PrevVNode->Next), 0, CurrentVNode);

  if (!Success) {
    free(CurrentVNode);
    return;
  }
}

/* For multi-threaded programs, while the profile is being dumped, other
   threads may still be updating the value profile data and creating new
   value entries. To accommadate this, we need to add extra bytes to the
   data buffer. The size of the extra space is controlled by an environment
   variable. */
static unsigned getVprofExtraBytes() {
  const char *ExtraStr =
      GetEnvHook ? GetEnvHook("LLVM_VALUE_PROF_BUFFER_EXTRA") : 0;
  if (!ExtraStr || !ExtraStr[0])
    return 1024;
  return (unsigned)atoi(ExtraStr);
}

/* Extract the value profile data info from the runtime. */
#define DEF_VALUE_RECORD(R, NS, V)                                             \
  ValueProfRuntimeRecord R;                                                    \
  if (initializeValueProfRuntimeRecord(&R, NS, V))                             \
    PROF_OOM_RETURN("Failed to write value profile data ");

#define DTOR_VALUE_RECORD(R) finalizeValueProfRuntimeRecord(&R);

LLVM_LIBRARY_VISIBILITY uint64_t
__llvm_profile_gather_value_data(uint8_t **VDataArray) {
  size_t S = 0, RealSize = 0, BufferCapacity = 0, Extra = 0;
  __llvm_profile_data *I;
  if (!VDataArray)
    PROF_OOM_RETURN("Failed to write value profile data ");

  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();

  /*
   * Compute the total Size of the buffer to hold ValueProfData
   * structures for functions with value profile data.
   */
  for (I = (__llvm_profile_data *)DataBegin; I != DataEnd; ++I) {

    DEF_VALUE_RECORD(R, I->NumValueSites, I->Values);

    /* Compute the size of ValueProfData from this runtime record.  */
    if (getNumValueKindsRT(&R) != 0)
      S += getValueProfDataSizeRT(&R);

    DTOR_VALUE_RECORD(R);
  }
  /* No value sites or no value profile data is collected. */
  if (!S)
    return 0;

  Extra = getVprofExtraBytes();
  BufferCapacity = S + Extra;
  *VDataArray = calloc(BufferCapacity, sizeof(uint8_t));
  if (!*VDataArray)
    PROF_OOM_RETURN("Failed to write value profile data ");

  ValueProfData *VD = (ValueProfData *)(*VDataArray);
  /*
   * Extract value profile data and write into ValueProfData structure
   * one by one. Note that new value profile data added to any value
   * site (from another thread) after the ValueProfRuntimeRecord is
   * initialized (when the profile data snapshot is taken) won't be
   * collected. This is not a problem as those dropped value will have
   * very low taken count.
   */
  for (I = (__llvm_profile_data *)DataBegin; I != DataEnd; ++I) {
    DEF_VALUE_RECORD(R, I->NumValueSites, I->Values);
    if (getNumValueKindsRT(&R) == 0)
      continue;

    /* Record R has taken a snapshot of the VP data at this point. Newly
       added VP data for this function will be dropped.  */
    /* Check if there is enough space.  */
    if (BufferCapacity - RealSize < getValueProfDataSizeRT(&R)) {
      PROF_ERR("Value profile data is dropped :%s \n",
               "Out of buffer space. Use environment "
               " LLVM_VALUE_PROF_BUFFER_EXTRA to allocate more");
      I->Values = 0;
    }

    serializeValueProfDataFromRT(&R, VD);
    deallocateValueProfileCounters(I);
    I->Values = VD;
    RealSize += VD->TotalSize;
    VD = (ValueProfData *)((char *)VD + VD->TotalSize);
    DTOR_VALUE_RECORD(R);
  }

  return RealSize;
}
