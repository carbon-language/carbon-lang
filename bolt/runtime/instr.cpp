//===-- instr.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// This file contains code that is linked to the final binary with a function
// that is called at program exit to dump instrumented data collected during
// execution.
//
//===----------------------------------------------------------------------===//
//
// BOLT runtime instrumentation library.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

// All extern declarations here need to be defined by BOLT itself.

// Counters inserted by instrumentation, incremented during runtime when
// points of interest (locations) in the program are reached.
extern uint64_t __bolt_instr_locations[];
// Number of counters.
extern uint32_t __bolt_instr_num_locs;
// String table with function names.
extern char __bolt_instr_strings[];
// Filename to dump data to.
extern char __bolt_instr_filename[];

// A location is a function name plus offset. Function name needs to be
// retrieved from the string table and is stored as an index to this table.
typedef struct _Location {
  uint32_t FunctionName;
  uint32_t Offset;
} Location;

// An edge description defines an instrumented edge in the program, fully
// identified by where the jump is located and its destination.
typedef struct _EdgeDescription {
  Location From;
  Location To;
} EdgeDescription;

extern EdgeDescription __bolt_instr_descriptions[];

// Declare some syscall wrappers we use throughout this code to avoid linking
// against system libc.
static uint64_t
myopen(const char *pathname,
       uint64_t flags,
       uint64_t mode) {
  uint64_t ret;
  __asm__ __volatile__ (
          "movq $2, %%rax\n"
          "syscall"
          : "=a"(ret)
          : "D"(pathname), "S"(flags), "d"(mode)
          : "cc", "rcx", "r11", "memory");
  return ret;
}

static uint64_t mywrite(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
  __asm__ __volatile__ (
          "movq $1, %%rax\n"
          "syscall\n"
          : "=a"(ret)
          : "D"(fd), "S"(buf), "d"(count)
          : "cc", "rcx", "r11", "memory");
  return ret;
}

static int myclose(uint64_t fd) {
  uint64_t ret;
  __asm__ __volatile__ (
          "movq $3, %%rax\n"
          "syscall\n"
          : "=a"(ret)
          : "D"(fd)
          : "cc", "rcx", "r11", "memory");
  return ret;
}

static char *intToStr(char *OutBuf, uint32_t Num, uint32_t Base) {
  const char *Chars = "0123456789abcdef";
  char Buf[20];
  char *Ptr = Buf;
  while (Num) {
    *Ptr++ = *(Chars + (Num % Base));
    Num /= Base;
  }
  if (Ptr == Buf) {
    *OutBuf++ = '0';
    return OutBuf;
  }
  while (Ptr != Buf) {
    *OutBuf++ = *--Ptr;
  }
  return OutBuf;
}

static char *serializeLoc(char *OutBuf, uint32_t FuncStrIndex,
                          uint32_t Offset) {
  *OutBuf++ = '1';
  *OutBuf++ = ' ';
  char *Str = __bolt_instr_strings + FuncStrIndex;
  while (*Str) {
    *OutBuf++ = *Str++;
  }
  *OutBuf++ = ' ';
  OutBuf = intToStr(OutBuf, Offset, 16);
  *OutBuf++ = ' ';
  return OutBuf;
}

extern "C" void __bolt_instr_data_dump() {
  uint64_t FD = myopen(__bolt_instr_filename,
                       /*flags=*/0x241 /*O_WRONLY|O_TRUNC|O_CREAT*/,
                       /*mode=*/0666);

  for (int I = 0, E = __bolt_instr_num_locs; I < E; ++I) {
    char LineBuf[2000];
    char *Ptr = LineBuf;
    uint32_t HitCount = __bolt_instr_locations[I];
    if (!HitCount)
      continue;

    EdgeDescription *Desc = &__bolt_instr_descriptions[I];
    Ptr = serializeLoc(Ptr, Desc->From.FunctionName, Desc->From.Offset);
    Ptr = serializeLoc(Ptr, Desc->To.FunctionName, Desc->To.Offset);
    *Ptr++ = '0';
    *Ptr++ = ' ';
    Ptr = intToStr(Ptr, HitCount, 10);
    *Ptr++ = '\n';
    mywrite(FD, LineBuf, Ptr - LineBuf);
  }
  myclose(FD);
}
