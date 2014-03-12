/*===- PGOProfiling.c - Support library for PGO instrumentation -----------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define I386_FREEBSD (defined(__FreeBSD__) && defined(__i386__))

#if !I386_FREEBSD
#include <inttypes.h>
#endif

#if !defined(_MSC_VER) && !I386_FREEBSD
#include <stdint.h>
#endif

#if defined(_MSC_VER)
typedef unsigned int uint32_t;
typedef unsigned int uint64_t;
#elif I386_FREEBSD
/* System headers define 'size_t' incorrectly on x64 FreeBSD (prior to
 * FreeBSD 10, r232261) when compiled in 32-bit mode.
 */
#define PRIu64 "llu"
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#endif

static FILE *OutputFile = NULL;

/*
 * A list of functions to register counters.
 */
typedef void (*CounterFunc)();

struct CounterFuncNode {
  CounterFunc Func;
  struct CounterFuncNode *Next;
};

static struct CounterFuncNode *CounterFuncHead = NULL;
static struct CounterFuncNode *CounterFuncTail = NULL;

static uint64_t CounterBufSize;
static uint64_t CounterNextIndex;
static uint64_t *CounterData;

struct __attribute__((packed)) ProfileDataHeader {
  char     Magic[4];
  uint32_t Version;
  uint32_t DataStart;
  uint32_t Padding;
  uint64_t MaxFunctionCount;
};

static struct ProfileDataHeader ProfileDataHeader = {
  .Magic = {'L', 'P', 'R', 'F'},
  .Version = 1,
  .DataStart = 0,
  .Padding = 0,
  .MaxFunctionCount = 0
};

static int write32(uint32_t Val) {
  char Buf[4] = {Val >> 0, Val >> 8, Val >> 16, Val >> 24};
  return fwrite(Buf, 1, 4, OutputFile) != 4;
}

static int write64(uint64_t Val) {
  char Buf[8] = {Val >> 0,  Val >> 8,  Val >> 16, Val >> 24,
                 Val >> 32, Val >> 40, Val >> 48, Val >> 56};
  return fwrite(Buf, 1, 8, OutputFile) != 8;
}

static int writeChars(const char *Val, size_t Count) {
  return fwrite(Val, 1, Count, OutputFile) != Count;
}

static int writeIndexEntry(const char *Name) {
  uint32_t Len = strlen(Name);
  if (write32(Len)) return 1;
  if (writeChars(Name, Len)) return 1;
  if (write32(CounterNextIndex * sizeof(uint64_t))) return 1;
  return 0;
}

static int addCounters(uint64_t FunctionHash, uint32_t NumCounters,
                       uint64_t *Counters) {
  uint64_t Needed = sizeof(uint64_t) * (CounterNextIndex + NumCounters + 2);
  if (CounterBufSize < Needed) {
    while (CounterBufSize < Needed)
      CounterBufSize *= 2;
    uint64_t *PrevData = CounterData;
    if (NULL == (CounterData = realloc(CounterData, CounterBufSize))) {
      free(PrevData);
      return 1;
    }
  }
  CounterData[CounterNextIndex++] = FunctionHash;
  CounterData[CounterNextIndex++] = NumCounters;
  if (NumCounters > 0 && Counters[0] > ProfileDataHeader.MaxFunctionCount)
    ProfileDataHeader.MaxFunctionCount = Counters[0];
  for (uint32_t I = 0; I < NumCounters; ++I)
    CounterData[CounterNextIndex++] = Counters[I];
  return 0;
}

static int reserveHeader() {
  return fseek(OutputFile, sizeof(struct ProfileDataHeader), SEEK_SET) != 0;
}

static int writeIndex() {
  if (!CounterData) {
    CounterBufSize = 4096;
    if (NULL == (CounterData = malloc(CounterBufSize)))
      return 1;
  }
  while (CounterFuncHead) {
    struct CounterFuncNode *Node = CounterFuncHead;
    CounterFuncHead = CounterFuncHead->Next;
    Node->Func();
    free(Node);
  }
  return 0;
}

static int writeCounterData() {
  ProfileDataHeader.DataStart = ftell(OutputFile);
  if (fseek(OutputFile,
            sizeof(uint64_t) - ProfileDataHeader.DataStart % sizeof(uint64_t),
            SEEK_CUR) != 0)
    return 1;
  for (uint32_t I = 0; I < CounterNextIndex; ++I)
    if (write64(CounterData[I])) return 1;
  return 0;
}

static int writeHeader() {
  if (fseek(OutputFile, 0, SEEK_SET) != 0) return 1;
  if (writeChars(ProfileDataHeader.Magic, 4)) return 1;
  if (write32(ProfileDataHeader.Version)) return 1;
  if (write32(ProfileDataHeader.DataStart)) return 1;
  if (write32(ProfileDataHeader.Padding)) return 1;
  if (write64(ProfileDataHeader.MaxFunctionCount)) return 1;
  return 0;
}

void llvm_pgo_add_function(const char *MangledName, uint64_t FunctionHash,
                           uint32_t NumCounters, uint64_t *Counters) {
  if (!CounterData) return;
  if (writeIndexEntry(MangledName)) {
    fprintf(stderr, "profile: Failed to write index for %s\n", MangledName);
    return;
  }
  if (addCounters(FunctionHash, NumCounters, Counters)) {
    fprintf(stderr, "profile: Failed to add counters for %s\n", MangledName);
    return;
  }
}

void llvm_pgo_register_counter_function(CounterFunc Func) {
  struct CounterFuncNode *NewNode = malloc(sizeof(struct CounterFuncNode));
  if (!NewNode) {
    fprintf(stderr, "profile: Failed to register counter function: %p\n",
            Func);
    return;
  }
  NewNode->Func = Func;
  NewNode->Next = NULL;

  if (!CounterFuncHead) {
    CounterFuncHead = CounterFuncTail = NewNode;
  } else {
    CounterFuncTail->Next = NewNode;
    CounterFuncTail = NewNode;
  }
}

void llvm_pgo_write_file() {
  const char *OutputName = getenv("LLVM_PROFILE_FILE");
  if (OutputName == NULL || OutputName[0] == '\0')
    OutputName = "default.profdata";
  OutputFile = fopen(OutputName, "w");
  if (!OutputFile) {
    fprintf(stderr, "profile: Failed to open %s for writing: %s\n",
            OutputName, strerror(errno));
    return;
  }

  if (reserveHeader()) goto end;
  if (writeIndex()) goto end;
  if (writeCounterData()) goto end;
  if (writeHeader()) goto end;

end:
  if (CounterData) free(CounterData);
  fclose(OutputFile);
}

void llvm_pgo_init(CounterFunc Func) {
  static int Ran = 0;

  llvm_pgo_register_counter_function(Func);

  if (Ran == 0) {
    Ran = 1;
    atexit(llvm_pgo_write_file);
  }
}
