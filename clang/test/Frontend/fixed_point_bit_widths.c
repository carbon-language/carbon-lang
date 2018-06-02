// RUN: %clang -x c -ffixed-point -S -emit-llvm -o - %s | FileCheck %s

int size_SsA = sizeof(signed short _Accum);
int size_SA  = sizeof(signed _Accum);
int size_SlA = sizeof(signed long _Accum);
int align_SsA = __alignof(signed short _Accum);
int align_SA  = __alignof(signed _Accum);
int align_SlA = __alignof(signed long _Accum);

int size_UsA = sizeof(unsigned short _Accum);
int size_UA  = sizeof(unsigned _Accum);
int size_UlA = sizeof(unsigned long _Accum);
int align_UsA = __alignof(unsigned short _Accum);
int align_UA  = __alignof(unsigned _Accum);
int align_UlA = __alignof(unsigned long _Accum);

int size_sA = sizeof(short _Accum);
int size_A  = sizeof(_Accum);
int size_lA = sizeof(long _Accum);
int align_sA = __alignof(short _Accum);
int align_A  = __alignof(_Accum);
int align_lA = __alignof(long _Accum);

// CHECK:      @size_SsA  = dso_local global i{{[0-9]+}} 2
// CHECK-NEXT: @size_SA   = dso_local global i{{[0-9]+}} 4
// CHECK-NEXT: @size_SlA  = dso_local global i{{[0-9]+}} 8
// CHECK-NEXT: @align_SsA = dso_local global i{{[0-9]+}} 2
// CHECK-NEXT: @align_SA  = dso_local global i{{[0-9]+}} 4
// CHECK-NEXT: @align_SlA = dso_local global i{{[0-9]+}} 8

// CHECK-NEXT: @size_UsA  = dso_local global i{{[0-9]+}} 2
// CHECK-NEXT: @size_UA   = dso_local global i{{[0-9]+}} 4
// CHECK-NEXT: @size_UlA  = dso_local global i{{[0-9]+}} 8
// CHECK-NEXT: @align_UsA = dso_local global i{{[0-9]+}} 2
// CHECK-NEXT: @align_UA  = dso_local global i{{[0-9]+}} 4
// CHECK-NEXT: @align_UlA = dso_local global i{{[0-9]+}} 8

// CHECK-NEXT: @size_sA   = dso_local global i{{[0-9]+}} 2
// CHECK-NEXT: @size_A    = dso_local global i{{[0-9]+}} 4
// CHECK-NEXT: @size_lA   = dso_local global i{{[0-9]+}} 8
// CHECK-NEXT: @align_sA  = dso_local global i{{[0-9]+}} 2
// CHECK-NEXT: @align_A   = dso_local global i{{[0-9]+}} 4
// CHECK-NEXT: @align_lA  = dso_local global i{{[0-9]+}} 8
