// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s

struct EmptyIntMemb {
  int FlexArrayMemb[0];
};
// CHECK: *** Dumping AST Record Layout
// CHECK: Type: struct EmptyIntMemb
// CHECK: Record: 
// CHECK: Layout: <ASTRecordLayout
// CHECK:     Size:32
// CHECK:     Alignment:32
// CHECK:     FieldOffsets: [0]>

struct EmptyLongLongMemb {
  long long FlexArrayMemb[0];
};
// CHECK: *** Dumping AST Record Layout
// CHECK: Type: struct EmptyLongLongMemb
// CHECK: Record: 
// CHECK: Layout: <ASTRecordLayout
// CHECK:     Size:32
// CHECK:     Alignment:64
// CHECK:     FieldOffsets: [0]>

struct EmptyAligned2LongLongMemb {
  long long __declspec(align(2)) FlexArrayMemb[0];
};

// CHECK: *** Dumping AST Record Layout
// CHECK: Type: struct EmptyAligned2LongLongMemb
// CHECK: Record: 
// CHECK: Layout: <ASTRecordLayout
// CHECK:     Size:32
// CHECK:     Alignment:64
// CHECK:     FieldOffsets: [0]>

struct EmptyAligned8LongLongMemb {
  long long __declspec(align(8)) FlexArrayMemb[0];
};

// CHECK: *** Dumping AST Record Layout
// CHECK: Type: struct EmptyAligned8LongLongMemb
// CHECK: Record: 
// CHECK: Layout: <ASTRecordLayout
// CHECK:     Size:64
// CHECK:     Alignment:64
// CHECK:     FieldOffsets: [0]>

#pragma pack(1)
struct __declspec(align(4)) EmptyPackedAligned4LongLongMemb {
  long long FlexArrayMemb[0];
};
#pragma pack()

// CHECK: *** Dumping AST Record Layout
// CHECK: Type: struct EmptyPackedAligned4LongLongMemb
// CHECK: Record: 
// CHECK: Layout: <ASTRecordLayout
// CHECK:     Size:32
// CHECK:     Alignment:32
// CHECK:     FieldOffsets: [0]>

#pragma pack(1)
struct EmptyPackedAligned8LongLongMemb {
  long long __declspec(align(8)) FlexArrayMemb[0];
};
#pragma pack()

// CHECK: *** Dumping AST Record Layout
// CHECK: Type: struct EmptyPackedAligned8LongLongMemb
// CHECK: Record: 
// CHECK: Layout: <ASTRecordLayout
// CHECK:     Size:64
// CHECK:     Alignment:64
// CHECK:     FieldOffsets: [0]>


int a[
sizeof(struct EmptyIntMemb)+
sizeof(struct EmptyLongLongMemb)+
sizeof(struct EmptyAligned2LongLongMemb)+
sizeof(struct EmptyAligned8LongLongMemb)+
sizeof(struct EmptyPackedAligned4LongLongMemb)+
sizeof(struct EmptyPackedAligned8LongLongMemb)+
0];
