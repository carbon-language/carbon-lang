// RUN: %clang_cc1 -triple x86_64-pc-win32 -emit-llvm -O2 -o - %s | FileCheck %s

// Under Windows 64, int and long are 32-bits.  Make sure pointer math doesn't
// cause any sign extensions.

// CHECK: %sub.ptr.sub = add i64 %param, -8
// CHECK: %0 = inttoptr i64 %sub.ptr.sub to %struct.anon*
// CHECK: %tmp3 = getelementptr inbounds %struct.anon* %0, i64 0, i32 0

#define CR(Record, TYPE, Field) \
  ((TYPE *) ((unsigned char *) (Record) - (unsigned char *) &(((TYPE *) 0)->Field)))

typedef struct _LIST_ENTRY {
  struct _LIST_ENTRY  *ForwardLink;
  struct _LIST_ENTRY  *BackLink;
} LIST_ENTRY;

typedef struct {
  unsigned long long    Signature;
  LIST_ENTRY            Link;
} MEMORY_MAP;

int test(unsigned long long param)
{
  LIST_ENTRY      *Link;
  MEMORY_MAP      *Entry;

  Link = (LIST_ENTRY *) param;

  Entry = CR (Link, MEMORY_MAP, Link);
  return (int) Entry->Signature;
}
