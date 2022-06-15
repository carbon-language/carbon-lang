// RUN: %clang_cc1 -no-opaque-pointers -std=c++14 -triple x86_64-unknown-unknown -fblocks %s -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK,CHECK-O0
// RUN: %clang_cc1 -no-opaque-pointers -std=c++14 -triple x86_64-unknown-unknown -fblocks -ftrivial-auto-var-init=pattern %s -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-O0,PATTERN,PATTERN-O0
// RUN: %clang_cc1 -no-opaque-pointers -std=c++14 -triple x86_64-unknown-unknown -fblocks -ftrivial-auto-var-init=pattern %s -O1 -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-O1,PATTERN,PATTERN-O1
// RUN: %clang_cc1 -no-opaque-pointers -std=c++14 -triple x86_64-unknown-unknown -fblocks -ftrivial-auto-var-init=zero %s -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-O0,ZERO,ZERO-O0
// RUN: %clang_cc1 -no-opaque-pointers -std=c++14 -triple x86_64-unknown-unknown -fblocks -ftrivial-auto-var-init=zero %s -O1 -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-O1,ZERO,ZERO-O1
// RUN: %clang_cc1 -no-opaque-pointers -std=c++14 -triple i386-unknown-unknown -fblocks -ftrivial-auto-var-init=pattern %s -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-O0,PATTERN,PATTERN-O0

#pragma clang diagnostic ignored "-Winaccessible-base"

#ifdef __x86_64__
char inits[] = {"-86/-21846/-1431655766/i64/-6148914691236517206/-6148914691236517206/i128/-113427455640312821154458202477256070486/i64/-6148914691236517206/AA/"};
#else
char inits[] = {"-1/-1/-1/i32/-1/-1/i32/-1/i32/-1/FF/"};
#define __int128 int;
#endif
// PATTERN: @inits = {{.*}} c"[[I8:[^/]+]]/[[I16:[^/]+]]/[[I32:[^/]+]]/[[ILONGT:[^/]+]]/[[ILONG:[^/]+]]/[[I64:[^/]+]]/[[I128T:[^/]+]]/[[I128:[^/]+]]/[[IPTRT:[^/]+]]/[[IPTR:[^/]+]]/[[IC:[^/]+]]/\00", align 1

template<typename T> void used(T &) noexcept;

#define TEST_UNINIT(NAME, TYPE)                 \
  using type_##NAME = TYPE;                     \
  void test_##NAME##_uninit() {                 \
    type_##NAME uninit;                         \
    used(uninit);                               \
  }

// Value initialization on scalars, aggregate initialization on aggregates.
#define TEST_BRACES(NAME, TYPE)                 \
  using type_##NAME = TYPE;                     \
  void test_##NAME##_braces() {                 \
    type_##NAME braces = {};                    \
    used(braces);                               \
  }

#define TEST_CUSTOM(NAME, TYPE, ...)            \
  using type_##NAME = TYPE;                     \
  void test_##NAME##_custom() {                 \
    type_##NAME custom __VA_ARGS__;             \
    used(custom);                               \
  }

// None of the synthesized globals should contain `undef`.
// PATTERN-NOT: undef
// ZERO-NOT: undef

// PATTERN-O0: @__const.test_empty_uninit.uninit = private unnamed_addr constant %struct.empty { i8 [[I8]] }, align 1
// PATTERN-O1-NOT: @__const.test_empty_uninit.uninit
struct empty {};
// PATTERN-O0: @__const.test_small_uninit.uninit = private unnamed_addr constant %struct.small { i8 [[I8]] }, align 1
// PATTERN-O0: @__const.test_small_custom.custom = private unnamed_addr constant %struct.small { i8 42 }, align 1
// ZERO-O0: @__const.test_small_custom.custom = private unnamed_addr constant %struct.small { i8 42 }, align 1
// PATTERN-O1-NOT: @__const.test_small_uninit.uninit
// PATTERN-O1-NOT: @__const.test_small_custom.custom
// ZERO-O1-NOT: @__const.test_small_custom.custom
struct small { char c; };
// PATTERN-O0: @__const.test_smallinit_uninit.uninit = private unnamed_addr constant %struct.smallinit { i8 [[I8]] }, align 1
// PATTERN-O0: @__const.test_smallinit_braces.braces = private unnamed_addr constant %struct.smallinit { i8 [[I8]] }, align 1
// PATTERN-O0: @__const.test_smallinit_custom.custom = private unnamed_addr constant %struct.smallinit { i8 [[I8]] }, align 1
// PATTERN-O1-NOT: @__const.test_smallinit_uninit.uninit
// PATTERN-O1-NOT: @__const.test_smallinit_braces.braces
// PATTERN-O1-NOT: @__const.test_smallinit_custom.custom
struct smallinit { char c = 42; };
// PATTERN-O0: @__const.test_smallpartinit_uninit.uninit = private unnamed_addr constant %struct.smallpartinit { i8 [[I8]], i8 [[I8]] }, align 1
// PATTERN-O0: @__const.test_smallpartinit_braces.braces = private unnamed_addr constant %struct.smallpartinit { i8 [[I8]], i8 [[I8]] }, align 1
// PATTERN-O0: @__const.test_smallpartinit_custom.custom = private unnamed_addr constant %struct.smallpartinit { i8 [[I8]], i8 [[I8]] }, align 1
// PATTERN-O1-NOT: @__const.test_smallpartinit_uninit.uninit
// PATTERN-O1-NOT: @__const.test_smallpartinit_braces.braces
// PATTERN-O1-NOT: @__const.test_smallpartinit_custom.custom
struct smallpartinit { char c = 42, d; };
// PATTERN-O0: @__const.test_nullinit_uninit.uninit = private unnamed_addr constant %struct.nullinit { i8* inttoptr ([[IPTRT]] [[IPTR]] to i8*) }, align
// PATTERN-O0: @__const.test_nullinit_braces.braces = private unnamed_addr constant %struct.nullinit { i8* inttoptr ([[IPTRT]] [[IPTR]] to i8*) }, align
// PATTERN-O0: @__const.test_nullinit_custom.custom = private unnamed_addr constant %struct.nullinit { i8* inttoptr ([[IPTRT]] [[IPTR]] to i8*) }, align
// PATTERN-O1-NOT: @__const.test_nullinit_uninit.uninit
// PATTERN-O1-NOT: @__const.test_nullinit_braces.braces
// PATTERN-O1-NOT: @__const.test_nullinit_custom.custom
struct nullinit { char* null = nullptr; };
// PATTERN-O0: @__const.test_padded_uninit.uninit = private unnamed_addr constant { i8, [3 x i8], i32 } { i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]", i32 [[I32]] }, align 4
// PATTERN-O0: @__const.test_padded_custom.custom = private unnamed_addr constant { i8, [3 x i8], i32 } { i8 42, [3 x i8] zeroinitializer, i32 13371337 }, align 4
// ZERO-O0: @__const.test_padded_custom.custom = private unnamed_addr constant { i8, [3 x i8], i32 } { i8 42, [3 x i8] zeroinitializer, i32 13371337 }, align 4
// PATTERN-O1-NOT: @__const.test_padded_uninit.uninit
// PATTERN-O1-NOT: @__const.test_padded_custom.custom
// ZERO-O1-NOT: @__const.test_padded_custom.custom
struct padded { char c; int i; };
// PATTERN-O0: @__const.test_paddednullinit_uninit.uninit = private unnamed_addr constant { i8, [3 x i8], i32 } { i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]", i32 [[I32]] }, align 4
// PATTERN-O0: @__const.test_paddednullinit_braces.braces = private unnamed_addr constant { i8, [3 x i8], i32 } { i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]", i32 [[I32]] }, align 4
// PATTERN-O0: @__const.test_paddednullinit_custom.custom = private unnamed_addr constant { i8, [3 x i8], i32 } { i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]", i32 [[I32]] }, align 4
// PATTERN-O1-NOT: @__const.test_paddednullinit_uninit.uninit
// PATTERN-O1-NOT: @__const.test_paddednullinit_braces.braces
// PATTERN-O1-NOT: @__const.test_paddednullinit_custom.custom
struct paddednullinit { char c = 0; int i = 0; };
// PATTERN-O0: @__const.test_paddedpacked_uninit.uninit = private unnamed_addr constant %struct.paddedpacked <{ i8 [[I8]], i32 [[I32]] }>, align 1
// PATTERN: @__const.test_paddedpacked_custom.custom = private unnamed_addr constant %struct.paddedpacked <{ i8 42, i32 13371337 }>, align 1
// ZERO: @__const.test_paddedpacked_custom.custom = private unnamed_addr constant %struct.paddedpacked <{ i8 42, i32 13371337 }>, align 1
struct paddedpacked { char c; int i; } __attribute__((packed));
// PATTERN-O0: @__const.test_paddedpackedarray_uninit.uninit = private unnamed_addr constant %struct.paddedpackedarray { [2 x %struct.paddedpacked] [%struct.paddedpacked <{ i8 [[I8]], i32 [[I32]] }>, %struct.paddedpacked <{ i8 [[I8]], i32 [[I32]] }>] }, align 1
// PATTERN: @__const.test_paddedpackedarray_custom.custom = private unnamed_addr constant %struct.paddedpackedarray { [2 x %struct.paddedpacked] [%struct.paddedpacked <{ i8 42, i32 13371337 }>, %struct.paddedpacked <{ i8 43, i32 13371338 }>] }, align 1
// ZERO: @__const.test_paddedpackedarray_custom.custom = private unnamed_addr constant %struct.paddedpackedarray { [2 x %struct.paddedpacked] [%struct.paddedpacked <{ i8 42, i32 13371337 }>, %struct.paddedpacked <{ i8 43, i32 13371338 }>] }, align 1
struct paddedpackedarray { struct paddedpacked p[2]; };
// PATTERN-O0: @__const.test_unpackedinpacked_uninit.uninit = private unnamed_addr constant <{ { i8, [3 x i8], i32 }, i8 }> <{ { i8, [3 x i8], i32 } { i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]", i32 [[I32]] }, i8 [[I8]] }>, align 1
struct unpackedinpacked { padded a; char b; } __attribute__((packed));
// PATTERN-O0: @__const.test_paddednested_uninit.uninit = private unnamed_addr constant { { i8, [3 x i8], i32 }, { i8, [3 x i8], i32 } } { { i8, [3 x i8], i32 } { i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]", i32 [[I32]] }, { i8, [3 x i8], i32 } { i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]", i32 [[I32]] } }, align 4
// PATTERN: @__const.test_paddednested_custom.custom = private unnamed_addr constant { { i8, [3 x i8], i32 }, { i8, [3 x i8], i32 } } { { i8, [3 x i8], i32 } { i8 42, [3 x i8] zeroinitializer, i32 13371337 }, { i8, [3 x i8], i32 } { i8 43, [3 x i8] zeroinitializer, i32 13371338 } }, align 4
// ZERO: @__const.test_paddednested_custom.custom = private unnamed_addr constant { { i8, [3 x i8], i32 }, { i8, [3 x i8], i32 } } { { i8, [3 x i8], i32 } { i8 42, [3 x i8] zeroinitializer, i32 13371337 }, { i8, [3 x i8], i32 } { i8 43, [3 x i8] zeroinitializer, i32 13371338 } }, align 4
struct paddednested { struct padded p1, p2; };
// PATTERN-O0: @__const.test_paddedpackednested_uninit.uninit = private unnamed_addr constant %struct.paddedpackednested { %struct.paddedpacked <{ i8 [[I8]], i32 [[I32]] }>, %struct.paddedpacked <{ i8 [[I8]], i32 [[I32]] }> }, align 1
// PATTERN: @__const.test_paddedpackednested_custom.custom = private unnamed_addr constant %struct.paddedpackednested { %struct.paddedpacked <{ i8 42, i32 13371337 }>, %struct.paddedpacked <{ i8 43, i32 13371338 }> }, align 1
// ZERO: @__const.test_paddedpackednested_custom.custom = private unnamed_addr constant %struct.paddedpackednested { %struct.paddedpacked <{ i8 42, i32 13371337 }>, %struct.paddedpacked <{ i8 43, i32 13371338 }> }, align 1
struct paddedpackednested { struct paddedpacked p1, p2; };
// PATTERN-O0: @__const.test_bitfield_uninit.uninit = private unnamed_addr constant %struct.bitfield { i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]" }, align 4
// PATTERN-O0: @__const.test_bitfield_custom.custom = private unnamed_addr constant %struct.bitfield { i8 20, [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]" }, align 4
// ZERO-O0: @__const.test_bitfield_custom.custom = private unnamed_addr constant %struct.bitfield { i8 20, [3 x i8] zeroinitializer }, align 4
// PATTERN-O1-NOT: @__const.test_bitfield_uninit.uninit
// PATTERN-O1-NOT: @__const.test_bitfield_custom.custom
// ZERO-O1-NOT: @__const.test_bitfield_custom.custom
struct bitfield { int i : 4; int j : 2; };
// PATTERN-O0: @__const.test_bitfieldaligned_uninit.uninit = private unnamed_addr constant %struct.bitfieldaligned { i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]", i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]" }, align 4
// PATTERN-O0: @__const.test_bitfieldaligned_custom.custom = private unnamed_addr constant %struct.bitfieldaligned { i8 4, [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]", i8 1, [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]" }, align 4
// ZERO-O0: @__const.test_bitfieldaligned_custom.custom = private unnamed_addr constant %struct.bitfieldaligned { i8 4, [3 x i8] zeroinitializer, i8 1, [3 x i8] zeroinitializer }, align 4
// PATTERN-O1-NOT: @__const.test_bitfieldaligned_uninit.uninit
// PATTERN-O1-NOT: @__const.test_bitfieldaligned_custom.custom
// ZERO-O1-NOT: @__const.test_bitfieldaligned_custom.custom
struct bitfieldaligned { int i : 4; int : 0; int j : 2; };
struct big { unsigned a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z; };
// PATTERN-O0: @__const.test_arraytail_uninit.uninit = private unnamed_addr constant %struct.arraytail { i32 [[I32]], [0 x i32] zeroinitializer }, align 4
// PATTERN-O0: @__const.test_arraytail_custom.custom = private unnamed_addr constant %struct.arraytail { i32 57005, [0 x i32] zeroinitializer }, align 4
// ZERO-O0: @__const.test_arraytail_custom.custom = private unnamed_addr constant %struct.arraytail { i32 57005, [0 x i32] zeroinitializer }, align 4
// PATTERN-O1-NOT: @__const.test_arraytail_uninit.uninit
// PATTERN-O1-NOT: @__const.test_arraytail_custom.custom
// ZERO-O1-NOT: @__const.test_arraytail_custom.custom
struct arraytail { int i; int arr[]; };
// PATTERN-O0: @__const.test_int1_uninit.uninit = private unnamed_addr constant [1 x i32] {{\[}}i32 [[I32]]], align 4
// PATTERN-O0: @__const.test_int1_custom.custom = private unnamed_addr constant [1 x i32] [i32 858993459], align 4
// ZERO-O0: @__const.test_int1_custom.custom = private unnamed_addr constant [1 x i32] [i32 858993459], align 4
// PATTERN-O1-NOT: @__const.test_int1_uninit.uninit
// PATTERN-O1-NOT: @__const.test_int1_custom.custom
// ZERO-O1-NOT: @__const.test_int1_custom.custom

// PATTERN-O0: @__const.test_bool4_uninit.uninit = private unnamed_addr constant [4 x i8] c"\[[IC]]\[[IC]]\[[IC]]\[[IC]]", align 1
// PATTERN-O0: @__const.test_bool4_custom.custom = private unnamed_addr constant [4 x i8] c"\01\01\01\01", align 1
// ZERO-O0: @__const.test_bool4_custom.custom = private unnamed_addr constant [4 x i8] c"\01\01\01\01", align 1
// PATTERN-O1-NOT: @__const.test_bool4_uninit.uninit
// PATTERN-O1-NOT: @__const.test_bool4_custom.custom
// ZERO-O1-NOT: @__const.test_bool4_custom.custom

// PATTERN: @__const.test_intptr4_custom.custom = private unnamed_addr constant [4 x i32*] [i32* inttoptr ([[IPTRT]] 572662306 to i32*), i32* inttoptr ([[IPTRT]] 572662306 to i32*), i32* inttoptr ([[IPTRT]] 572662306 to i32*), i32* inttoptr ([[IPTRT]] 572662306 to i32*)], align
// ZERO: @__const.test_intptr4_custom.custom = private unnamed_addr constant [4 x i32*] [i32* inttoptr (i64 572662306 to i32*), i32* inttoptr (i64 572662306 to i32*), i32* inttoptr (i64 572662306 to i32*), i32* inttoptr (i64 572662306 to i32*)], align 16
// PATTERN-O0: @__const.test_tailpad4_uninit.uninit = private unnamed_addr constant [4 x { i16, i8, [1 x i8] }] [{ i16, i8, [1 x i8] } { i16 [[I16]], i8 [[I8]], [1 x i8] c"\[[IC]]" }, { i16, i8, [1 x i8] } { i16 [[I16]], i8 [[I8]], [1 x i8] c"\[[IC]]" }, { i16, i8, [1 x i8] } { i16 [[I16]], i8 [[I8]], [1 x i8] c"\[[IC]]" }, { i16, i8, [1 x i8] } { i16 [[I16]], i8 [[I8]], [1 x i8] c"\[[IC]]" }], align
// PATTERN-O1-NOT: @__const.test_tailpad4_uninit.uninit
// PATTERN:   @__const.test_tailpad4_custom.custom = private unnamed_addr constant [4 x { i16, i8, [1 x i8] }] [{ i16, i8, [1 x i8] } { i16 257, i8 1, [1 x i8] zeroinitializer }, { i16, i8, [1 x i8] } { i16 257, i8 1, [1 x i8] zeroinitializer }, { i16, i8, [1 x i8] } { i16 257, i8 1, [1 x i8] zeroinitializer }, { i16, i8, [1 x i8] } { i16 257, i8 1, [1 x i8] zeroinitializer }], align
// ZERO: @__const.test_tailpad4_custom.custom = private unnamed_addr constant [4 x { i16, i8, [1 x i8] }] [{ i16, i8, [1 x i8] } { i16 257, i8 1, [1 x i8] zeroinitializer }, { i16, i8, [1 x i8] } { i16 257, i8 1, [1 x i8] zeroinitializer }, { i16, i8, [1 x i8] } { i16 257, i8 1, [1 x i8] zeroinitializer }, { i16, i8, [1 x i8] } { i16 257, i8 1, [1 x i8] zeroinitializer }], align 16
struct tailpad { short s; char c; };
// PATTERN-O0: @__const.test_atomicnotlockfree_uninit.uninit = private unnamed_addr constant %struct.notlockfree { [4 x i64] {{\[}}i64 [[I64]], i64 [[I64]], i64 [[I64]], i64 [[I64]]] }, align
// PATTERN-O1-NOT: @__const.test_atomicnotlockfree_uninit.uninit
struct notlockfree { long long a[4]; };
// PATTERN-O0: @__const.test_atomicpadded_uninit.uninit = private unnamed_addr constant { i8, [3 x i8], i32 } { i8 [[I8]], [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]", i32 [[I32]] }, align 8
// PATTERN-O1-NOT: @__const.test_atomicpadded_uninit.uninit
// PATTERN-O0: @__const.test_atomictailpad_uninit.uninit = private unnamed_addr constant { i16, i8, [1 x i8] } { i16 [[I16]], i8 [[I8]], [1 x i8] c"\[[IC]]" }, align 4
// PATTERN-O1-NOT: @__const.test_atomictailpad_uninit.uninit
// PATTERN-O0: @__const.test_complexfloat_uninit.uninit = private unnamed_addr constant { float, float } { float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000 }, align 4
// PATTERN-O1-NOT: @__const.test_complexfloat_uninit.uninit
// PATTERN-O0: @__const.test_complexfloat_braces.braces = private unnamed_addr constant { float, float } { float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000 }, align 4
// PATTERN-O1-NOT: @__const.test_complexfloat_braces.braces
// PATTERN-O0: @__const.test_complexfloat_custom.custom = private unnamed_addr constant { float, float } { float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000 }, align 4
// PATTERN-O1-NOT: @__const.test_complexfloat_custom.custom
// PATTERN-O0: @__const.test_complexdouble_uninit.uninit = private unnamed_addr constant { double, double } { double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF }, align 8
// PATTERN-O1-NOT: @__const.test_complexdouble_uninit.uninit
// PATTERN-O0: @__const.test_complexdouble_braces.braces = private unnamed_addr constant { double, double } { double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF }, align 8
// PATTERN-O1-NOT: @__const.test_complexdouble_braces.braces
// PATTERN-O0: @__const.test_complexdouble_custom.custom = private unnamed_addr constant { double, double } { double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF }, align 8
// PATTERN-O1-NOT: @__const.test_complexdouble_custom.custom
// PATTERN-O0: @__const.test_semivolatile_uninit.uninit = private unnamed_addr constant %struct.semivolatile { i32 [[I32]], i32 [[I32]] }, align 4
// PATTERN-O0: @__const.test_semivolatile_custom.custom = private unnamed_addr constant %struct.semivolatile { i32 1145324612, i32 1145324612 }, align 4
// PATTERN-O1-NOT: @__const.test_semivolatile_custom.custom
struct semivolatile { int i; volatile int vi; };
// PATTERN-O0: @__const.test_semivolatileinit_uninit.uninit = private unnamed_addr constant %struct.semivolatileinit { i32 [[I32]], i32 [[I32]] }, align 4
// PATTERN-O1-NOT: @__const.test_semivolatileinit_uninit.uninit
// PATTERN-O0: @__const.test_semivolatileinit_braces.braces = private unnamed_addr constant %struct.semivolatileinit { i32 [[I32]], i32 [[I32]] }, align 4
// PATTERN-O1-NOT: @__const.test_semivolatileinit_braces.braces
// PATTERN-O0: @__const.test_semivolatileinit_custom.custom = private unnamed_addr constant %struct.semivolatileinit { i32 [[I32]], i32 [[I32]] }, align 4
// PATTERN-O1-NOT: @__const.test_semivolatileinit_custom.custom = private unnamed_addr constant %struct.semivolatileinit { i32 [[I32]], i32 [[I32]] }, align 4
// ZERO-O0: @__const.test_semivolatile_custom.custom = private unnamed_addr constant %struct.semivolatile { i32 1145324612, i32 1145324612 }, align 4
// ZERO-O1-NOT: @__const.test_semivolatile_custom.custom
struct semivolatileinit { int i = 0x11111111; volatile int vi = 0x11111111; };
// PATTERN-O0: @__const.test_base_uninit.uninit = private unnamed_addr constant %struct.base { i32 (...)** inttoptr ([[IPTRT]] [[IPTR]] to i32 (...)**) }, align
// PATTERN-O1-NOT: @__const.test_base_uninit.uninit
// PATTERN-O0: @__const.test_base_braces.braces = private unnamed_addr constant %struct.base { i32 (...)** inttoptr ([[IPTRT]] [[IPTR]] to i32 (...)**) }, align
// PATTERN-O1-NOT: @__const.test_base_braces.braces
struct base { virtual ~base(); };
// PATTERN-O0: @__const.test_derived_uninit.uninit = private unnamed_addr constant %struct.derived { %struct.base { i32 (...)** inttoptr ([[IPTRT]] [[IPTR]] to i32 (...)**) } }, align
// PATTERN-O1-NOT: @__const.test_derived_uninit.uninit
// PATTERN-O0: @__const.test_derived_braces.braces = private unnamed_addr constant %struct.derived { %struct.base { i32 (...)** inttoptr ([[IPTRT]] [[IPTR]] to i32 (...)**) } }, align
// PATTERN-O1-NOT: @__const.test_derived_braces.braces
struct derived : public base {};
// PATTERN-O0: @__const.test_virtualderived_uninit.uninit = private unnamed_addr constant %struct.virtualderived { %struct.base { i32 (...)** inttoptr ([[IPTRT]] [[IPTR]] to i32 (...)**) }, %struct.derived { %struct.base { i32 (...)** inttoptr ([[IPTRT]] [[IPTR]] to i32 (...)**) } } }, align
// PATTERN-O1-NOT: @__const.test_virtualderived_uninit.uninit
// PATTERN-O0: @__const.test_virtualderived_braces.braces = private unnamed_addr constant %struct.virtualderived { %struct.base { i32 (...)** inttoptr ([[IPTRT]] [[IPTR]] to i32 (...)**) }, %struct.derived { %struct.base { i32 (...)** inttoptr ([[IPTRT]] [[IPTR]] to i32 (...)**) } } }, align
// PATTERN-O1-NOT: @__const.test_virtualderived_braces.braces
struct virtualderived : public virtual base, public virtual derived {};
// PATTERN-O0: @__const.test_matching_uninit.uninit = private unnamed_addr constant %union.matching { i32 [[I32]] }, align 4
// PATTERN-O1-NOT: @__const.test_matching_uninit.uninit
// PATTERN-O0: @__const.test_matching_custom.custom = private unnamed_addr constant { float } { float 6.145500e+04 }, align 4
// PATTERN-O1-NOT: @__const.test_matching_custom.custom
// ZERO-O0: @__const.test_matching_custom.custom = private unnamed_addr constant { float } { float 6.145500e+04 }, align 4
// ZERO-O1-NOT: @__const.test_matching_custom.custom
union matching { int i; float f; };
// PATTERN-O0: @__const.test_matchingreverse_uninit.uninit = private unnamed_addr constant %union.matchingreverse { float 0xFFFFFFFFE0000000 }, align 4
// PATTERN-O1-NOT: @__const.test_matchingreverse_uninit.uninit
// PATTERN-O0: @__const.test_matchingreverse_custom.custom = private unnamed_addr constant { i32 } { i32 61455 }, align 4
// PATTERN-O1-NOT: @__const.test_matchingreverse_custom.custom
// ZERO-O0: @__const.test_matchingreverse_custom.custom = private unnamed_addr constant { i32 } { i32 61455 }, align 4
// ZERO-O1-NOT: @__const.test_matchingreverse_custom.custom
union matchingreverse { float f; int i; };
// PATTERN-O0: @__const.test_unmatched_uninit.uninit = private unnamed_addr constant %union.unmatched { i32 [[I32]] }, align 4
// PATTERN-O1-NOT: @__const.test_unmatched_uninit.uninit
// PATTERN-O0: @__const.test_unmatched_custom.custom = private unnamed_addr constant %union.unmatched { i32 1001242351 }, align 4
// PATTERN-O1-NOT: @__const.test_unmatched_custom.custom
// ZERO-O0: @__const.test_unmatched_custom.custom = private unnamed_addr constant %union.unmatched { i32 1001242351 }, align 4
// ZERO-O1-NOT: @__const.test_unmatched_custom.custom
union unmatched { char c; int i; };
// PATTERN-O0: @__const.test_unmatchedreverse_uninit.uninit = private unnamed_addr constant %union.unmatchedreverse { i32 [[I32]] }, align 4
// PATTERN-O1-NOT: @__const.test_unmatchedreverse_uninit.uninit
// PATTERN-O0: @__const.test_unmatchedreverse_custom.custom = private unnamed_addr constant { i8, [3 x i8] } { i8 42, [3 x i8] c"\[[IC]]\[[IC]]\[[IC]]" }, align 4
// PATTERN-O1-NOT: @__const.test_unmatchedreverse_custom.custom
// ZERO-O0: @__const.test_unmatchedreverse_custom.custom = private unnamed_addr constant { i8, [3 x i8] } { i8 42, [3 x i8] zeroinitializer }, align 4
// ZERO-O1-NOT: @__const.test_unmatchedreverse_custom.custom
union unmatchedreverse { int i; char c; };
// PATTERN-O0: @__const.test_unmatchedfp_uninit.uninit = private unnamed_addr constant %union.unmatchedfp { double 0xFFFFFFFFFFFFFFFF }, align
// PATTERN-O1-NOT: @__const.test_unmatchedfp_uninit.uninit
// PATTERN-O0: @__const.test_unmatchedfp_custom.custom = private unnamed_addr constant %union.unmatchedfp { double 0x400921FB54442D18 }, align
// PATTERN-O1-NOT: @__const.test_unmatchedfp_custom.custom
// ZERO-O0: @__const.test_unmatchedfp_custom.custom = private unnamed_addr constant %union.unmatchedfp { double 0x400921FB54442D18 }, align 8
// ZERO-O1-NOT: @__const.test_unmatchedfp_custom.custom
union unmatchedfp { float f; double d; };
enum emptyenum {};
enum smallenum { VALUE };

extern "C" {

TEST_UNINIT(char, char);
// CHECK-LABEL: @test_char_uninit()
// CHECK:       %uninit = alloca i8, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_char_uninit()
// PATTERN: store i8 [[I8]], i8* %uninit, align 1, !annotation [[AUTO_INIT:!.+]]
// ZERO-LABEL: @test_char_uninit()
// ZERO: store i8 0, i8* %uninit, align 1, !annotation [[AUTO_INIT:!.+]]

TEST_BRACES(char, char);
// CHECK-LABEL: @test_char_braces()
// CHECK:       %braces = alloca i8, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i8 0, i8* %braces, align [[ALIGN]]
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(uchar, unsigned char);
// CHECK-LABEL: @test_uchar_uninit()
// CHECK:       %uninit = alloca i8, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_uchar_uninit()
// PATTERN: store i8 [[I8]], i8* %uninit, align 1, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_uchar_uninit()
// ZERO: store i8 0, i8* %uninit, align 1, !annotation [[AUTO_INIT]]

TEST_BRACES(uchar, unsigned char);
// CHECK-LABEL: @test_uchar_braces()
// CHECK:       %braces = alloca i8, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i8 0, i8* %braces, align [[ALIGN]]
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(schar, signed char);
// CHECK-LABEL: @test_schar_uninit()
// CHECK:       %uninit = alloca i8, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_schar_uninit()
// PATTERN: store i8 [[I8]], i8* %uninit, align 1, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_schar_uninit()
// ZERO: store i8 0, i8* %uninit, align 1, !annotation [[AUTO_INIT]]

TEST_BRACES(schar, signed char);
// CHECK-LABEL: @test_schar_braces()
// CHECK:       %braces = alloca i8, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i8 0, i8* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(wchar_t, wchar_t);
// CHECK-LABEL: @test_wchar_t_uninit()
// CHECK:       %uninit = alloca i32, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_wchar_t_uninit()
// PATTERN: store i32 [[I32]], i32* %uninit, align 4, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_wchar_t_uninit()
// ZERO: store i32 0, i32* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(wchar_t, wchar_t);
// CHECK-LABEL: @test_wchar_t_braces()
// CHECK:       %braces = alloca i32, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i32 0, i32* %braces, align [[ALIGN]]
//  CHECK-NOT:  !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(short, short);
// CHECK-LABEL: @test_short_uninit()
// CHECK:       %uninit = alloca i16, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_short_uninit()
// PATTERN: store i16 [[I16]], i16* %uninit, align 2, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_short_uninit()
// ZERO: store i16 0, i16* %uninit, align 2, !annotation [[AUTO_INIT]]

TEST_BRACES(short, short);
// CHECK-LABEL: @test_short_braces()
// CHECK:       %braces = alloca i16, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i16 0, i16* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(ushort, unsigned short);
// CHECK-LABEL: @test_ushort_uninit()
// CHECK:       %uninit = alloca i16, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_ushort_uninit()
// PATTERN: store i16 [[I16]], i16* %uninit, align 2, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_ushort_uninit()
// ZERO: store i16 0, i16* %uninit, align 2, !annotation [[AUTO_INIT]]

TEST_BRACES(ushort, unsigned short);
// CHECK-LABEL: @test_ushort_braces()
// CHECK:       %braces = alloca i16, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i16 0, i16* %braces, align [[ALIGN]]
//CHECK-NOT:    !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(int, int);
// CHECK-LABEL: @test_int_uninit()
// CHECK:       %uninit = alloca i32, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_int_uninit()
// PATTERN: store i32 [[I32]], i32* %uninit, align 4, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_int_uninit()
// ZERO: store i32 0, i32* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(int, int);
// CHECK-LABEL: @test_int_braces()
// CHECK:       %braces = alloca i32, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i32 0, i32* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(unsigned, unsigned);
// CHECK-LABEL: @test_unsigned_uninit()
// CHECK:       %uninit = alloca i32, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_unsigned_uninit()
// PATTERN: store i32 [[I32]], i32* %uninit, align 4, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_unsigned_uninit()
// ZERO: store i32 0, i32* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(unsigned, unsigned);
// CHECK-LABEL: @test_unsigned_braces()
// CHECK:       %braces = alloca i32, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i32 0, i32* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(long, long);
// CHECK-LABEL: @test_long_uninit()
// CHECK:       %uninit = alloca i64, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_long_uninit()
// PATTERN: store [[ILONGT]] [[ILONG]], [[ILONGT]]* %uninit, align {{.+}}, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_long_uninit()
// ZERO: store i64 0, i64* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_BRACES(long, long);
// CHECK-LABEL: @test_long_braces()
// CHECK:       %braces = alloca i64, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i64 0, i64* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(ulong, unsigned long);
// CHECK-LABEL: @test_ulong_uninit()
// CHECK:       %uninit = alloca i64, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_ulong_uninit()
// PATTERN: store [[ILONGT]] [[ILONG]], [[ILONGT]]* %uninit, align {{.+}}, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_ulong_uninit()
// ZERO: store i64 0, i64* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_BRACES(ulong, unsigned long);
// CHECK-LABEL: @test_ulong_braces()
// CHECK:       %braces = alloca i64, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i64 0, i64* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(longlong, long long);
// CHECK-LABEL: @test_longlong_uninit()
// CHECK:       %uninit = alloca i64, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_longlong_uninit()
// PATTERN: store i64 [[I64]], i64* %uninit, align 8, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_longlong_uninit()
// ZERO: store i64 0, i64* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_BRACES(longlong, long long);
// CHECK-LABEL: @test_longlong_braces()
// CHECK:       %braces = alloca i64, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i64 0, i64* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(ulonglong, unsigned long long);
// CHECK-LABEL: @test_ulonglong_uninit()
// CHECK:       %uninit = alloca i64, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_ulonglong_uninit()
// PATTERN: store i64 [[I64]], i64* %uninit, align 8, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_ulonglong_uninit()
// ZERO: store i64 0, i64* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_BRACES(ulonglong, unsigned long long);
// CHECK-LABEL: @test_ulonglong_braces()
// CHECK:       %braces = alloca i64, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i64 0, i64* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(int128, __int128);
// CHECK-LABEL: @test_int128_uninit()
// CHECK:       %uninit = alloca i128, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_int128_uninit()
// PATTERN: store [[I128T]] [[I128]], [[I128T]]* %uninit, align {{.+}}, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_int128_uninit()
// ZERO: store i128 0, i128* %uninit, align 16, !annotation [[AUTO_INIT]]

TEST_BRACES(int128, __int128);
// CHECK-LABEL: @test_int128_braces()
// CHECK:       %braces = alloca i128, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i128 0, i128* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(uint128, unsigned __int128);
// CHECK-LABEL: @test_uint128_uninit()
// CHECK:       %uninit = alloca i128, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_uint128_uninit()
// PATTERN: store [[I128T]] [[I128]], [[I128T]]* %uninit, align {{.+}}, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_uint128_uninit()
// ZERO: store i128 0, i128* %uninit, align 16, !annotation [[AUTO_INIT]]

TEST_BRACES(uint128, unsigned __int128);
// CHECK-LABEL: @test_uint128_braces()
// CHECK:       %braces = alloca i128, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i128 0, i128* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(fp16, __fp16);
// CHECK-LABEL: @test_fp16_uninit()
// CHECK:       %uninit = alloca half, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_fp16_uninit()
// PATTERN: store half 0xHFFFF, half* %uninit, align 2, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_fp16_uninit()
// ZERO: store half 0xH0000, half* %uninit, align 2, !annotation [[AUTO_INIT]]

TEST_BRACES(fp16, __fp16);
// CHECK-LABEL: @test_fp16_braces()
// CHECK:       %braces = alloca half, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store half 0xH0000, half* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(float, float);
// CHECK-LABEL: @test_float_uninit()
// CHECK:       %uninit = alloca float, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_float_uninit()
// PATTERN: store float 0xFFFFFFFFE0000000, float* %uninit, align 4, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_float_uninit()
// ZERO: store float 0.000000e+00, float* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(float, float);
// CHECK-LABEL: @test_float_braces()
// CHECK:       %braces = alloca float, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store float 0.000000e+00, float* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(double, double);
// CHECK-LABEL: @test_double_uninit()
// CHECK:       %uninit = alloca double, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_double_uninit()
// PATTERN: store double 0xFFFFFFFFFFFFFFFF, double* %uninit, align 8, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_double_uninit()
// ZERO: store double 0.000000e+00, double* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_BRACES(double, double);
// CHECK-LABEL: @test_double_braces()
// CHECK:       %braces = alloca double, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store double 0.000000e+00, double* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(longdouble, long double);
// CHECK-LABEL: @test_longdouble_uninit()
// CHECK:       %uninit = alloca x86_fp80, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_longdouble_uninit()
// PATTERN: store x86_fp80 0xKFFFFFFFFFFFFFFFFFFFF, x86_fp80* %uninit, align {{.+}}, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_longdouble_uninit()
// ZERO: store x86_fp80 0xK00000000000000000000, x86_fp80* %uninit, align {{.+}}, !annotation [[AUTO_INIT]]

TEST_BRACES(longdouble, long double);
// CHECK-LABEL: @test_longdouble_braces()
// CHECK:       %braces = alloca x86_fp80, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store x86_fp80 0xK00000000000000000000, x86_fp80* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(intptr, int*);
// CHECK-LABEL: @test_intptr_uninit()
// CHECK:       %uninit = alloca i32*, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_intptr_uninit()
// PATTERN: store i32* inttoptr ([[IPTRT]] [[IPTR]] to i32*), i32** %uninit, align {{.+}}, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_intptr_uninit()
// ZERO: store i32* null, i32** %uninit, align {{.+}}, !annotation [[AUTO_INIT]]

TEST_BRACES(intptr, int*);
// CHECK-LABEL: @test_intptr_braces()
// CHECK:       %braces = alloca i32*, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i32* null, i32** %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(intptrptr, int**);
// CHECK-LABEL: @test_intptrptr_uninit()
// CHECK:       %uninit = alloca i32**, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_intptrptr_uninit()
// PATTERN: store i32** inttoptr ([[IPTRT]] [[IPTR]] to i32**), i32*** %uninit, align {{.+}}, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_intptrptr_uninit()
// ZERO: store i32** null, i32*** %uninit, align {{.+}}, !annotation [[AUTO_INIT]]

TEST_BRACES(intptrptr, int**);
// CHECK-LABEL: @test_intptrptr_braces()
// CHECK:       %braces = alloca i32**, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i32** null, i32*** %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(function, void(*)());
// CHECK-LABEL: @test_function_uninit()
// CHECK:       %uninit = alloca void ()*, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_function_uninit()
// PATTERN: store void ()* inttoptr ([[IPTRT]] [[IPTR]] to void ()*), void ()** %uninit, align {{.+}}, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_function_uninit()
// ZERO: store void ()* null, void ()** %uninit, align {{.+}}, !annotation [[AUTO_INIT]]

TEST_BRACES(function, void(*)());
// CHECK-LABEL: @test_function_braces()
// CHECK:       %braces = alloca void ()*, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store void ()* null, void ()** %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(bool, bool);
// CHECK-LABEL: @test_bool_uninit()
// CHECK:       %uninit = alloca i8, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_bool_uninit()
// PATTERN: store i8 [[I8]], i8* %uninit, align 1, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_bool_uninit()
// ZERO: store i8 0, i8* %uninit, align 1, !annotation [[AUTO_INIT]]

TEST_BRACES(bool, bool);
// CHECK-LABEL: @test_bool_braces()
// CHECK:       %braces = alloca i8, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i8 0, i8* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(empty, empty);
// CHECK-LABEL: @test_empty_uninit()
// CHECK:       %uninit = alloca %struct.empty, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_empty_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_empty_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i8 [[I8]], {{.*}} align 1, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_empty_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i8 0, {{.*}} align 1, !annotation [[AUTO_INIT]]

TEST_BRACES(empty, empty);
// CHECK-LABEL: @test_empty_braces()
// CHECK:       %braces = alloca %struct.empty, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(small, small);
// CHECK-LABEL: @test_small_uninit()
// CHECK:       %uninit = alloca %struct.small, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_small_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_small_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i8 [[I8]], {{.*}} align 1, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_small_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i8 0, {{.*}} align 1, !annotation [[AUTO_INIT]]

TEST_BRACES(small, small);
// CHECK-LABEL: @test_small_braces()
// CHECK:       %braces = alloca %struct.small, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 1, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(small, small, { 42 });
// CHECK-LABEL: @test_small_custom()
// CHECK:       %custom = alloca %struct.small, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(smallinit, smallinit);
// CHECK-LABEL: @test_smallinit_uninit()
// CHECK:       %uninit = alloca %struct.smallinit, align
// CHECK-NEXT:  call void @{{.*}}smallinit{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(smallinit, smallinit);
// CHECK-LABEL: @test_smallinit_braces()
// CHECK:       %braces = alloca %struct.smallinit, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.smallinit, %struct.smallinit* %braces, i32 0, i32 0
// CHECK-NEXT:  store i8 42, i8* %[[C]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(smallinit, smallinit, { 100 });
// CHECK-LABEL: @test_smallinit_custom()
// CHECK:       %custom = alloca %struct.smallinit, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.smallinit, %struct.smallinit* %custom, i32 0, i32 0
// CHECK-NEXT:  store i8 100, i8* %[[C]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(smallpartinit, smallpartinit);
// CHECK-LABEL: @test_smallpartinit_uninit()
// CHECK:       %uninit = alloca %struct.smallpartinit, align
// CHECK-NEXT:  call void @{{.*}}smallpartinit{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_smallpartinit_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_smallpartinit_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i8 [[I8]], {{.*}} align 1, !annotation [[AUTO_INIT]]
// PATTERN-O1: store i8 42, {{.*}} align 1
// ZERO-LABEL: @test_smallpartinit_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i16 0, i16* %uninit, align 2, !annotation [[AUTO_INIT]]

TEST_BRACES(smallpartinit, smallpartinit);
// CHECK-LABEL: @test_smallpartinit_braces()
// CHECK:       %braces = alloca %struct.smallpartinit, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.smallpartinit, %struct.smallpartinit* %braces, i32 0, i32 0
// CHECK-NEXT:  store i8 42, i8* %[[C]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  %[[D:[^ ]*]] = getelementptr inbounds %struct.smallpartinit, %struct.smallpartinit* %braces, i32 0, i32 1
// CHECK-NEXT:  store i8 0, i8* %[[D]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(smallpartinit, smallpartinit, { 100, 42 });
// CHECK-LABEL: @test_smallpartinit_custom()
// CHECK:       %custom = alloca %struct.smallpartinit, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.smallpartinit, %struct.smallpartinit* %custom, i32 0, i32 0
// CHECK-NEXT:  store i8 100, i8* %[[C]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  %[[D:[^ ]*]] = getelementptr inbounds %struct.smallpartinit, %struct.smallpartinit* %custom, i32 0, i32 1
// CHECK-NEXT:  store i8 42, i8* %[[D]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(nullinit, nullinit);
// CHECK-LABEL: @test_nullinit_uninit()
// CHECK:       %uninit = alloca %struct.nullinit, align
// CHECK-NEXT:  call void @{{.*}}nullinit{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(nullinit, nullinit);
// CHECK-LABEL: @test_nullinit_braces()
// CHECK:       %braces = alloca %struct.nullinit, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[N:[^ ]*]] = getelementptr inbounds %struct.nullinit, %struct.nullinit* %braces, i32 0, i32 0
// CHECK-NEXT:  store i8* null, i8** %[[N]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(nullinit, nullinit, { (char*)"derp" });
// CHECK-LABEL: @test_nullinit_custom()
// CHECK:       %custom = alloca %struct.nullinit, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[N:[^ ]*]] = getelementptr inbounds %struct.nullinit, %struct.nullinit* %custom, i32 0, i32 0
// CHECK-NEXT:  store i8* getelementptr inbounds {{.*}}, i8** %[[N]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(padded, padded);
// CHECK-LABEL: @test_padded_uninit()
// CHECK:       %uninit = alloca %struct.padded, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_padded_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_padded_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i64 [[I64]], i64* %uninit, align 8, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_padded_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i64 0, i64* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_BRACES(padded, padded);
// CHECK-LABEL: @test_padded_braces()
// CHECK:       %braces = alloca %struct.padded, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 8, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(padded, padded, { 42, 13371337 });
// CHECK-LABEL: @test_padded_custom()
// CHECK:       %custom = alloca %struct.padded, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(paddednullinit, paddednullinit);
// CHECK-LABEL: @test_paddednullinit_uninit()
// CHECK:       %uninit = alloca %struct.paddednullinit, align
// CHECK-NEXT:  call void @{{.*}}paddednullinit{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_paddednullinit_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_paddednullinit_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i64 [[I64]], i64* %uninit, align 8, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_paddednullinit_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0, {{.*}}, !annotation [[AUTO_INIT]]
// ZERO-O1: store i64 0, i64* %uninit, align 8
// FIXME: !annotation dropped by optimizations
// ZERO-O1-NOT: !annotation
// ZERO: ret


TEST_BRACES(paddednullinit, paddednullinit);
// CHECK-LABEL: @test_paddednullinit_braces()
// CHECK:       %braces = alloca %struct.paddednullinit, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.paddednullinit, %struct.paddednullinit* %braces, i32 0, i32 0
// CHECK-NEXT:  store i8 0, i8* %[[C]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds %struct.paddednullinit, %struct.paddednullinit* %braces, i32 0, i32 1
// CHECK-NEXT:  store i32 0, i32* %[[I]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(paddednullinit, paddednullinit, { 42, 13371337 });
// CHECK-LABEL: @test_paddednullinit_custom()
// CHECK:       %custom = alloca %struct.paddednullinit, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.paddednullinit, %struct.paddednullinit* %custom, i32 0, i32 0
// CHECK-NEXT:  store i8 42, i8* %[[C]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds %struct.paddednullinit, %struct.paddednullinit* %custom, i32 0, i32 1
// CHECK-NEXT:  store i32 13371337, i32* %[[I]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(paddedpacked, paddedpacked);
// CHECK-LABEL: @test_paddedpacked_uninit()
// CHECK:       %uninit = alloca %struct.paddedpacked, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_paddedpacked_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_paddedpacked_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1:  %[[C:[^ ]*]] = getelementptr inbounds {{.*}}%uninit, i64 0, i32 0
// PATTERN-O1:  store i8 [[I8]], i8* %[[C]], align {{.+}}, !annotation [[AUTO_INIT]]
// PATTERN-O1:  %[[I:[^ ]*]] = getelementptr inbounds {{.*}}%uninit, i64 0, i32 1
// PATTERN-O1: store i32 [[I32]], i32* %[[I]], align {{.+}}, !annotation [[AUTO_INIT]]

// ZERO-LABEL: @test_paddedpacked_uninit()
// ZERO: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(paddedpacked, paddedpacked);
// CHECK-LABEL: @test_paddedpacked_braces()
// CHECK:       %braces = alloca %struct.paddedpacked, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 5, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(paddedpacked, paddedpacked, { 42, 13371337 });
// CHECK-LABEL: @test_paddedpacked_custom()
// CHECK:       %custom = alloca %struct.paddedpacked, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy{{.*}}({{.*}}@__const.test_paddedpacked_custom.custom
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(paddedpackedarray, paddedpackedarray);
// CHECK-LABEL: @test_paddedpackedarray_uninit()
// CHECK:       %uninit = alloca %struct.paddedpackedarray, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_paddedpackedarray_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_paddedpackedarray_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: getelementptr
// PATTERN-O1: call void @llvm.memset{{.*}}({{.*}}i8 [[I8]], i64 10
// FIXME: !annotation dropped by optimizations
// PATTERN-O1-NOT: !annotation
// ZERO-LABEL: @test_paddedpackedarray_uninit()
// ZERO: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(paddedpackedarray, paddedpackedarray);
// CHECK-LABEL: @test_paddedpackedarray_braces()
// CHECK:       %braces = alloca %struct.paddedpackedarray, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 10, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(paddedpackedarray, paddedpackedarray, { {{ 42, 13371337 }, { 43, 13371338 }} });
// CHECK-LABEL: @test_paddedpackedarray_custom()
// CHECK:       %custom = alloca %struct.paddedpackedarray, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy{{.*}}({{.*}}@__const.test_paddedpackedarray_custom.custom
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(unpackedinpacked, unpackedinpacked);
// PATTERN-LABEL: @test_unpackedinpacked_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} 9, i1 false), !annotation [[AUTO_INIT]]

TEST_UNINIT(paddednested, paddednested);
// CHECK-LABEL: @test_paddednested_uninit()
// CHECK:       %uninit = alloca %struct.paddednested, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_paddednested_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_paddednested_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: getelementptr
// PATTERN-O1: call void @llvm.memset{{.*}}({{.*}}, i8 [[I8]], i64 16{{.+}})
// FIXME: !annotation dropped by optimizations
// PATTERN-O1-NOT: !annotation
// ZERO-LABEL: @test_paddednested_uninit()
// ZERO: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(paddednested, paddednested);
// CHECK-LABEL: @test_paddednested_braces()
// CHECK:       %braces = alloca %struct.paddednested, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 16, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(paddednested, paddednested, { { 42, 13371337 }, { 43, 13371338 } });
// CHECK-LABEL: @test_paddednested_custom()
// CHECK:       %custom = alloca %struct.paddednested, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy{{.*}}({{.*}}@__const.test_paddednested_custom.custom
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(paddedpackednested, paddedpackednested);
// CHECK-LABEL: @test_paddedpackednested_uninit()
// CHECK:       %uninit = alloca %struct.paddedpackednested, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_paddedpackednested_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_paddedpackednested_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: getelementptr
// PATTERN-O1: call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 1 dereferenceable(10) %0, i8 [[I8]], i64 10, i1 false)
// FIXME: !annotation dropped by optimizations
// PATTERN-O1-NOT: !annotation
// ZERO-LABEL: @test_paddedpackednested_uninit()
// ZERO: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(paddedpackednested, paddedpackednested);
// CHECK-LABEL: @test_paddedpackednested_braces()
// CHECK:       %braces = alloca %struct.paddedpackednested, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 10, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(paddedpackednested, paddedpackednested, { { 42, 13371337 }, { 43, 13371338 } });
// CHECK-LABEL: @test_paddedpackednested_custom()
// CHECK:       %custom = alloca %struct.paddedpackednested, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy{{.*}}({{.*}}@__const.test_paddedpackednested_custom.custom
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(bitfield, bitfield);
// CHECK-LABEL: @test_bitfield_uninit()
// CHECK:       %uninit = alloca %struct.bitfield, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_bitfield_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_bitfield_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i32 [[I32]], i32* %uninit, align 4
// FIXME: !annotation dropped by optimizations
// PATTERN-O1-NOT: !annotation
// ZERO-LABEL: @test_bitfield_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i32 0, i32* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(bitfield, bitfield);
// CHECK-LABEL: @test_bitfield_braces()
// CHECK:       %braces = alloca %struct.bitfield, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(bitfield, bitfield, { 4, 1 });
// CHECK-LABEL: @test_bitfield_custom()
// CHECK:       %custom = alloca %struct.bitfield, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(bitfieldaligned, bitfieldaligned);
// CHECK-LABEL: @test_bitfieldaligned_uninit()
// CHECK:       %uninit = alloca %struct.bitfieldaligned, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_bitfieldaligned_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_bitfieldaligned_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i64 [[IPTR]], i64* %uninit, align 8
// FIXME: !annotation dropped by optimizations
// PATTERN-O1-NOT: !annotation
// ZERO-LABEL: @test_bitfieldaligned_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i64 0, i64* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_BRACES(bitfieldaligned, bitfieldaligned);
// CHECK-LABEL: @test_bitfieldaligned_braces()
// CHECK:       %braces = alloca %struct.bitfieldaligned, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(bitfieldaligned, bitfieldaligned, { 4, 1  });
// CHECK-LABEL: @test_bitfieldaligned_custom()
// CHECK:       %custom = alloca %struct.bitfieldaligned, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(big, big);
// CHECK-LABEL: @test_big_uninit()
// CHECK:       %uninit = alloca %struct.big, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_big_uninit()
// PATTERN: call void @llvm.memset{{.*}}, i8 [[I8]],{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_big_uninit()
// ZERO: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(big, big);
// CHECK-LABEL: @test_big_braces()
// CHECK:       %braces = alloca %struct.big, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 104, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(big, big, { 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA });
// CHECK-LABEL: @test_big_custom()
// CHECK:       %custom = alloca %struct.big, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 -86, i64 104, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(arraytail, arraytail);
// CHECK-LABEL: @test_arraytail_uninit()
// CHECK:       %uninit = alloca %struct.arraytail, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_arraytail_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_arraytail_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i32 [[I32]], {{.*}} align 4, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_arraytail_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i32 0, {{.*}} align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(arraytail, arraytail);
// CHECK-LABEL: @test_arraytail_braces()
// CHECK:       %braces = alloca %struct.arraytail, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(arraytail, arraytail, { 0xdead });
// CHECK-LABEL: @test_arraytail_custom()
// CHECK:       %custom = alloca %struct.arraytail, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(int0, int[0]);
// CHECK-LABEL: @test_int0_uninit()
// CHECK:       %uninit = alloca [0 x i32], align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_int0_uninit()
// PATTERN:       %uninit = alloca [0 x i32], align
// PATTERN-O0-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// ZERO-LABEL: @test_int0_uninit()
// ZERO:       %uninit = alloca [0 x i32], align
// ZERO-O0-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(int0, int[0]);
// CHECK-LABEL: @test_int0_braces()
// CHECK:       %braces = alloca [0 x i32], align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(int1, int[1]);
// CHECK-LABEL: @test_int1_uninit()
// CHECK:       %uninit = alloca [1 x i32], align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_int1_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_int1_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i32 [[I32]], {{.*}} align 4, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_int1_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i32 0, {{.*}} align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(int1, int[1]);
// CHECK-LABEL: @test_int1_braces()
// CHECK:       %braces = alloca [1 x i32], align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(int1, int[1], { 0x33333333 });
// CHECK-LABEL: @test_int1_custom()
// CHECK:       %custom = alloca [1 x i32], align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(int64, int[64]);
// CHECK-LABEL: @test_int64_uninit()
// CHECK:       %uninit = alloca [64 x i32], align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_int64_uninit()
// PATTERN: call void @llvm.memset{{.*}}, i8 [[I8]],{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_int64_uninit()
// ZERO: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(int64, int[64]);
// CHECK-LABEL: @test_int64_braces()
// CHECK:       %braces = alloca [64 x i32], align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 256, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(int64, int[64], = { 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111 });
// CHECK-LABEL: @test_int64_custom()
// CHECK:       %custom = alloca [64 x i32], align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 17, i64 256, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(bool4, bool[4]);
// CHECK-LABEL: @test_bool4_uninit()
// CHECK:       %uninit = alloca [4 x i8], align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_bool4_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_bool4_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i32 [[I32]], i32* %uninit, align 4
// FIXME: !annotation dropped by optimizations
// PATTERN-O1-NOT: !annotation
// ZERO-LABEL: @test_bool4_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i32 0, i32* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(bool4, bool[4]);
// CHECK-LABEL: @test_bool4_braces()
// CHECK:       %braces = alloca [4 x i8], align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(bool4, bool[4], { true, true, true, true });
// CHECK-LABEL: @test_bool4_custom()
// CHECK:       %custom = alloca [4 x i8], align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(intptr4, int*[4]);
// CHECK-LABEL:      @test_intptr4_uninit()
// CHECK:            %uninit = alloca [4 x i32*], align
// CHECK-NEXT:       call void @{{.*}}used{{.*}}%uninit)
// PATTERN-O1-LABEL: @test_intptr4_uninit()
// PATTERN-O1:  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 16  dereferenceable(32) %{{[0-9*]}}, i8 -86, i64 32, i1 false)
// FIXME: !annotation dropped by optimizations
// PATTERN-O1-NOT: !annotation
// ZERO-LABEL:       @test_intptr4_uninit()
// ZERO:             call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(intptr4, int*[4]);
// CHECK-LABEL: @test_intptr4_braces()
// CHECK:       %braces = alloca [4 x i32*], align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 32, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(intptr4, int *[4], = {(int *)0x22222222, (int *)0x22222222, (int *)0x22222222, (int *)0x22222222});
// CHECK-LABEL: @test_intptr4_custom()
// CHECK:       %custom = alloca [4 x i32*], align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(tailpad4, tailpad[4]);
// CHECK-LABEL: @test_tailpad4_uninit()
// CHECK:       %uninit = alloca [4 x %struct.tailpad], align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_tailpad4_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_tailpad4_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: bitcast
// PATTERN-O1: call void @llvm.memset{{.*}}({{.*}}0, i8 [[I8]], i64 16{{.+}})
// FIXME: !annotation dropped by optimizations
// PATTERN-O1-NOT: !annotation
// ZERO-LABEL: @test_tailpad4_uninit()
// ZERO: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(tailpad4, tailpad[4]);
// CHECK-LABEL: @test_tailpad4_braces()
// CHECK:       %braces = alloca [4 x %struct.tailpad], align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 16, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(tailpad4, tailpad[4], { {257, 1}, {257, 1}, {257, 1}, {257, 1} });
// CHECK-LABEL: @test_tailpad4_custom()
// CHECK:       %custom = alloca [4 x %struct.tailpad], align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(tailpad9, tailpad[9]);
// CHECK-LABEL: @test_tailpad9_uninit()
// CHECK:       %uninit = alloca [9 x %struct.tailpad], align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_tailpad9_uninit()
// PATTERN-O0: call void @llvm.memset{{.*}}, i8 [[I8]],{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_tailpad9_uninit()
// ZERO: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(tailpad9, tailpad[9]);
// CHECK-LABEL: @test_tailpad9_braces()
// CHECK:       %braces = alloca [9 x %struct.tailpad], align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 36, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(tailpad9, tailpad[9], { {257, 1}, {257, 1}, {257, 1}, {257, 1}, {257, 1}, {257, 1}, {257, 1}, {257, 1}, {257, 1} });
// CHECK-LABEL: @test_tailpad9_custom()
// CHECK:       %custom = alloca [9 x %struct.tailpad], align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 1, i64 36, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(atomicbool, _Atomic(bool));
// CHECK-LABEL: @test_atomicbool_uninit()
// CHECK:       %uninit = alloca i8, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_atomicbool_uninit()
// PATTERN: store i8 [[I8]], i8* %uninit, align 1, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_atomicbool_uninit()
// ZERO: store i8 0, i8* %uninit, align 1, !annotation [[AUTO_INIT]]

TEST_UNINIT(atomicint, _Atomic(int));
// CHECK-LABEL: @test_atomicint_uninit()
// CHECK:       %uninit = alloca i32, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_atomicint_uninit()
// PATTERN: store i32 [[I32]], i32* %uninit, align 4, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_atomicint_uninit()
// ZERO: store i32 0, i32* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_UNINIT(atomicdouble, _Atomic(double));
// CHECK-LABEL: @test_atomicdouble_uninit()
// CHECK:       %uninit = alloca double, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_atomicdouble_uninit()
// PATTERN: store double 0xFFFFFFFFFFFFFFFF, double* %uninit, align 8, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_atomicdouble_uninit()
// ZERO: store double 0.000000e+00, double* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_UNINIT(atomicnotlockfree, _Atomic(notlockfree));
// CHECK-LABEL: @test_atomicnotlockfree_uninit()
// CHECK:       %uninit = alloca %struct.notlockfree, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_atomicnotlockfree_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_atomicnotlockfree_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: bitcast
// PATTERN-O1: call void @llvm.memset{{.*}}({{.*}}, i8 [[I8]], i64 32{{.*}}
// FIXME: !annotation dropped by optimizations
// PATTERN-O1-NOT: !annotation
// ZERO-LABEL: @test_atomicnotlockfree_uninit()
// ZERO: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]

TEST_UNINIT(atomicpadded, _Atomic(padded));
// CHECK-LABEL: @test_atomicpadded_uninit()
// CHECK:       %uninit = alloca %struct.padded, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_atomicpadded_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_atomicpadded_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store i64 [[IPTR]], i64* %uninit, align 8, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_atomicpadded_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i64 0, i64* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_UNINIT(atomictailpad, _Atomic(tailpad));
// CHECK-LABEL: @test_atomictailpad_uninit()
// CHECK:       %uninit = alloca %struct.tailpad, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_atomictailpad_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_atomictailpad_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_atomictailpad_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i32 0, i32* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_UNINIT(complexfloat, _Complex float);
// CHECK-LABEL: @test_complexfloat_uninit()
// CHECK:       %uninit = alloca { float, float }, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_complexfloat_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_complexfloat_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1:  %[[F1:[^ ]*]] = getelementptr inbounds {{.*}}%uninit, i64 0, i32 0
// PATTERN-O1: store float 0xFFFFFFFFE0000000, float* %[[F1]], align {{.+}}, !annotation [[AUTO_INIT]]

// PATTERN-O1:  %[[F2:[^ ]*]] = getelementptr inbounds {{.*}}%uninit, i64 0, i32 1
// PATTERN-O1: store float 0xFFFFFFFFE0000000, float* %[[F2]], align {{.+}}, !annotation [[AUTO_INIT]]

// ZERO-LABEL: @test_complexfloat_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i64 0, i64* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_BRACES(complexfloat, _Complex float);
// CHECK-LABEL: @test_complexfloat_braces()
// CHECK:       %braces = alloca { float, float }, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[R:[^ ]*]] = getelementptr inbounds { float, float }, { float, float }* %braces, i32 0, i32 0
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds { float, float }, { float, float }* %braces, i32 0, i32 1
// CHECK-NEXT:  store float 0.000000e+00, float* %[[R]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  store float 0.000000e+00, float* %[[I]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(complexfloat, _Complex float, { 3.1415926535897932384626433, 3.1415926535897932384626433 });
// CHECK-LABEL: @test_complexfloat_custom()
// CHECK:       %custom = alloca { float, float }, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[R:[^ ]*]] = getelementptr inbounds { float, float }, { float, float }* %custom, i32 0, i32 0
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds { float, float }, { float, float }* %custom, i32 0, i32 1
// CHECK-NEXT:  store float 0x400921FB60000000, float* %[[R]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  store float 0x400921FB60000000, float* %[[I]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(complexdouble, _Complex double);
// CHECK-LABEL: @test_complexdouble_uninit()
// CHECK:       %uninit = alloca { double, double }, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_complexdouble_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_complexdouble_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_complexdouble_uninit()
// ZERO: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(complexdouble, _Complex double);
// CHECK-LABEL: @test_complexdouble_braces()
// CHECK:       %braces = alloca { double, double }, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[R:[^ ]*]] = getelementptr inbounds { double, double }, { double, double }* %braces, i32 0, i32 0
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds { double, double }, { double, double }* %braces, i32 0, i32 1
// CHECK-NEXT:  store double 0.000000e+00, double* %[[R]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  store double 0.000000e+00, double* %[[I]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(complexdouble, _Complex double, { 3.1415926535897932384626433, 3.1415926535897932384626433 });
// CHECK-LABEL: @test_complexdouble_custom()
// CHECK:       %custom = alloca { double, double }, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[R:[^ ]*]] = getelementptr inbounds { double, double }, { double, double }* %custom, i32 0, i32 0
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds { double, double }, { double, double }* %custom, i32 0, i32 1
// CHECK-NEXT:  store double 0x400921FB54442D18, double* %[[R]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  store double 0x400921FB54442D18, double* %[[I]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(volatileint, volatile int);
// CHECK-LABEL: @test_volatileint_uninit()
// CHECK:       %uninit = alloca i32, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_volatileint_uninit()
// PATTERN: store volatile i32 [[I32]], i32* %uninit, align 4, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_volatileint_uninit()
// ZERO: store volatile i32 0, i32* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(volatileint, volatile int);
// CHECK-LABEL: @test_volatileint_braces()
// CHECK:       %braces = alloca i32, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store volatile i32 0, i32* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(semivolatile, semivolatile);
// CHECK-LABEL: @test_semivolatile_uninit()
// CHECK:       %uninit = alloca %struct.semivolatile, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_semivolatile_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_semivolatile_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_semivolatile_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i64 0, i64* %uninit, align 8, !annotation [[AUTO_INIT]]

TEST_BRACES(semivolatile, semivolatile);
// CHECK-LABEL: @test_semivolatile_braces()
// CHECK:       %braces = alloca %struct.semivolatile, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 8, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(semivolatile, semivolatile, { 0x44444444, 0x44444444 });
// CHECK-LABEL: @test_semivolatile_custom()
// CHECK:       %custom = alloca %struct.semivolatile, align
// CHECK-O0:  bitcast
// CHECK-O0:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-O0:  call void @{{.*}}used{{.*}}%custom)
// CHECK-O1:  store i64 4919131752989213764, i64* %custom, align 8
// CHECK-NOT:   !annotation

TEST_UNINIT(semivolatileinit, semivolatileinit);
// CHECK-LABEL: @test_semivolatileinit_uninit()
// CHECK:       %uninit = alloca %struct.semivolatileinit, align
// CHECK-NEXT:  call void @{{.*}}semivolatileinit{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(semivolatileinit, semivolatileinit);
// CHECK-LABEL: @test_semivolatileinit_braces()
// CHECK:       %braces = alloca %struct.semivolatileinit, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds %struct.semivolatileinit, %struct.semivolatileinit* %braces, i32 0, i32 0
// CHECK-NEXT:  store i32 286331153, i32* %[[I]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  %[[VI:[^ ]*]] = getelementptr inbounds %struct.semivolatileinit, %struct.semivolatileinit* %braces, i32 0, i32 1
// CHECK-NEXT:  store volatile i32 286331153, i32* %[[VI]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(semivolatileinit, semivolatileinit, { 0x44444444, 0x44444444 });
// CHECK-LABEL: @test_semivolatileinit_custom()
// CHECK:       %custom = alloca %struct.semivolatileinit, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds %struct.semivolatileinit, %struct.semivolatileinit* %custom, i32 0, i32 0
// CHECK-NEXT:  store i32 1145324612, i32* %[[I]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  %[[VI:[^ ]*]] = getelementptr inbounds %struct.semivolatileinit, %struct.semivolatileinit* %custom, i32 0, i32 1
// CHECK-NEXT:  store volatile i32 1145324612, i32* %[[VI]], align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(base, base);
// CHECK-LABEL: @test_base_uninit()
// CHECK:       %uninit = alloca %struct.base, align
// CHECK-NEXT:  call void @{{.*}}base{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_base_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_base_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_base_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV4base, i64 0, inrange i32 0, i64 2) to i32 (...)**), {{.*}}, align 8
// ZERO-O1-NOT: !annotation

TEST_BRACES(base, base);
// CHECK-LABEL: @test_base_braces()
// CHECK:       %braces = alloca %struct.base, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 8, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}base{{.*}}%braces)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(derived, derived);
// CHECK-LABEL: @test_derived_uninit()
// CHECK:       %uninit = alloca %struct.derived, align
// CHECK-NEXT:  call void @{{.*}}derived{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_derived_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_derived_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_derived_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i64 0, {{.*}} align 8, !annotation [[AUTO_INIT]]
// ZERO-O1: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV7derived, i64 0, inrange i32 0, i64 2) to i32 (...)**), {{.*}} align 8

TEST_BRACES(derived, derived);
// CHECK-LABEL: @test_derived_braces()
// CHECK:       %braces = alloca %struct.derived, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 8, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}derived{{.*}}%braces)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(virtualderived, virtualderived);
// CHECK-LABEL: @test_virtualderived_uninit()
// CHECK:       %uninit = alloca %struct.virtualderived, align
// CHECK-NEXT:  call void @{{.*}}virtualderived{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_virtualderived_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_virtualderived_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_virtualderived_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]

TEST_BRACES(virtualderived, virtualderived);
// CHECK-LABEL: @test_virtualderived_braces()
// CHECK:       %braces = alloca %struct.virtualderived, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 16, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}virtualderived{{.*}}%braces)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(matching, matching);
// CHECK-LABEL: @test_matching_uninit()
// CHECK:       %uninit = alloca %union.matching, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_matching_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_matching_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_matching_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i32 0, {{.*}} align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(matching, matching);
// CHECK-LABEL: @test_matching_braces()
// CHECK:       %braces = alloca %union.matching, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(matching, matching, { .f = 0xf00f });
// CHECK-LABEL: @test_matching_custom()
// CHECK:       %custom = alloca %union.matching, align
// CHECK-O0:  bitcast
// CHECK-O0:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-O0:  call void @{{.*}}used{{.*}}%custom)
// CHECK-O1:  getelementptr
// CHECK-O1:  store i32 1198526208, i32* {{.*}}, align 4
// CHECK-NOT:   !annotation

TEST_UNINIT(matchingreverse, matchingreverse);
// CHECK-LABEL: @test_matchingreverse_uninit()
// CHECK:       %uninit = alloca %union.matchingreverse, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_matchingreverse_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_matchingreverse_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// PATTERN-O1: store float 0xFFFFFFFFE0000000, {{.+}}, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_matchingreverse_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0,{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i32 0, {{.*}} align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(matchingreverse, matchingreverse);
// CHECK-LABEL: @test_matchingreverse_braces()
// CHECK:       %braces = alloca %union.matchingreverse, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(matchingreverse, matchingreverse, { .i = 0xf00f });
// CHECK-LABEL: @test_matchingreverse_custom()
// CHECK:       %custom = alloca %union.matchingreverse, align
// CHECK-O0:    bitcast
// CHECK-O0:    call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-O0:    call void @{{.*}}used{{.*}}%custom)
// CHECK-O1:    store i32 61455, i32* %1, align 4
// CHECK-NOT:   !annotation

TEST_UNINIT(unmatched, unmatched);
// CHECK-LABEL: @test_unmatched_uninit()
// CHECK:       %uninit = alloca %union.unmatched, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_unmatched_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_unmatched_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_unmatched_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i32 0, {{.*}} align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(unmatched, unmatched);
// CHECK-LABEL: @test_unmatched_braces()
// CHECK:       %braces = alloca %union.unmatched, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(unmatched, unmatched, { .i = 0x3badbeef });
// CHECK-LABEL: @test_unmatched_custom()
// CHECK:       %custom = alloca %union.unmatched, align
// CHECK-O0:    bitcast
// CHECK-O0:    call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-O0:    call void @{{.*}}used{{.*}}%custom)
// CHECK-O1:    store i32 1001242351, i32* {{.*}}, align 4
// CHECK-NOT:   !annotation

TEST_UNINIT(unmatchedreverse, unmatchedreverse);
// CHECK-LABEL: @test_unmatchedreverse_uninit()
// CHECK:       %uninit = alloca %union.unmatchedreverse, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_unmatchedreverse_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_unmatchedreverse_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_unmatchedreverse_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1:  store i32 0, {{.*}} align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(unmatchedreverse, unmatchedreverse);
// CHECK-LABEL: @test_unmatchedreverse_braces()
// CHECK:       %braces = alloca %union.unmatchedreverse, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align [[ALIGN]] %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(unmatchedreverse, unmatchedreverse, { .c = 42  });
// CHECK-LABEL: @test_unmatchedreverse_custom()
// CHECK:       %custom = alloca %union.unmatchedreverse, align
// CHECK-O0:    bitcast
// CHECK-O0:    call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-O0:    call void @{{.*}}used{{.*}}%custom)
// PATTERN-O1:  store i32 -1431655894, i32* {{.*}}, align 4
// ZERO-O1:     store i32 42, i32* {{.*}}, align 4

TEST_UNINIT(unmatchedfp, unmatchedfp);
// CHECK-LABEL: @test_unmatchedfp_uninit()
// CHECK:       %uninit = alloca %union.unmatchedfp, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_unmatchedfp_uninit()
// PATTERN-O0: call void @llvm.memcpy{{.*}} @__const.test_unmatchedfp_uninit.uninit{{.+}}), !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_unmatchedfp_uninit()
// ZERO-O0: call void @llvm.memset{{.*}}, i8 0, {{.+}}), !annotation [[AUTO_INIT]]
// ZERO-O1: store i64 0, {{.*}} align 8, !annotation [[AUTO_INIT]]

TEST_BRACES(unmatchedfp, unmatchedfp);
// CHECK-LABEL: @test_unmatchedfp_braces()
// CHECK:       %braces = alloca %union.unmatchedfp, align
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(unmatchedfp, unmatchedfp, { .d = 3.1415926535897932384626433 });
// CHECK-LABEL: @test_unmatchedfp_custom()
// CHECK:       %custom = alloca %union.unmatchedfp, align
// CHECK-O0:    bitcast
// CHECK-O0:    call void @llvm.memcpy
// CHECK-NOT:   !annotation
// CHECK-O0:    call void @{{.*}}used{{.*}}%custom)
// CHECK-O1:    store i64 4614256656552045848, i64* %1, align 8
// CHECK-NOT:   !annotation

TEST_UNINIT(emptyenum, emptyenum);
// CHECK-LABEL: @test_emptyenum_uninit()
// CHECK:       %uninit = alloca i32, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_emptyenum_uninit()
// PATTERN: store i32 [[I32]], i32* %uninit, align 4, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_emptyenum_uninit()
// ZERO: store i32 0, i32* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(emptyenum, emptyenum);
// CHECK-LABEL: @test_emptyenum_braces()
// CHECK:       %braces = alloca i32, align [[ALIGN:[0-9]*]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  store i32 0, i32* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(emptyenum, emptyenum, { (emptyenum)42 });
// CHECK-LABEL: @test_emptyenum_custom()
// CHECK:       %custom = alloca i32, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i32 42, i32* %custom, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(smallenum, smallenum);
// CHECK-LABEL: @test_smallenum_uninit()
// CHECK:       %uninit = alloca i32, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_smallenum_uninit()
// PATTERN: store i32 [[I32]], i32* %uninit, align 4, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_smallenum_uninit()
// ZERO: store i32 0, i32* %uninit, align 4, !annotation [[AUTO_INIT]]

TEST_BRACES(smallenum, smallenum);
// CHECK-LABEL: @test_smallenum_braces()
// CHECK:       %braces = alloca i32, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i32 0, i32* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(smallenum, smallenum, { (smallenum)42 });
// CHECK-LABEL: @test_smallenum_custom()
// CHECK:       %custom = alloca i32, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store i32 42, i32* %custom, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(intvec16, int  __attribute__((vector_size(16))));
// CHECK-LABEL: @test_intvec16_uninit()
// CHECK:       %uninit = alloca <4 x i32>, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_intvec16_uninit()
// PATTERN: store <4 x i32> <i32 [[I32]], i32 [[I32]], i32 [[I32]], i32 [[I32]]>, <4 x i32>* %uninit, align 16, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_intvec16_uninit()
// ZERO: store <4 x i32> zeroinitializer, <4 x i32>* %uninit, align 16, !annotation [[AUTO_INIT]]

TEST_BRACES(intvec16, int  __attribute__((vector_size(16))));
// CHECK-LABEL: @test_intvec16_braces()
// CHECK:       %braces = alloca <4 x i32>, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store <4 x i32> zeroinitializer, <4 x i32>* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(intvec16, int  __attribute__((vector_size(16))), { 0x44444444, 0x44444444, 0x44444444, 0x44444444 });
// CHECK-LABEL: @test_intvec16_custom()
// CHECK:       %custom = alloca <4 x i32>, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store <4 x i32> <i32 1145324612, i32 1145324612, i32 1145324612, i32 1145324612>, <4 x i32>* %custom, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(longlongvec32, long long  __attribute__((vector_size(32))));
// CHECK-LABEL: @test_longlongvec32_uninit()
// CHECK:       %uninit = alloca <4 x i64>, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_longlongvec32_uninit()
// PATTERN: store <4 x i64> <i64 [[I64]], i64 [[I64]], i64 [[I64]], i64 [[I64]]>, <4 x i64>* %uninit, align 32, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_longlongvec32_uninit()
// ZERO: store <4 x i64> zeroinitializer, <4 x i64>* %uninit, align 32, !annotation [[AUTO_INIT]]

TEST_BRACES(longlongvec32, long long  __attribute__((vector_size(32))));
// CHECK-LABEL: @test_longlongvec32_braces()
// CHECK:       %braces = alloca <4 x i64>, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store <4 x i64> zeroinitializer, <4 x i64>* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(longlongvec32, long long  __attribute__((vector_size(32))), { 0x3333333333333333, 0x3333333333333333, 0x3333333333333333, 0x3333333333333333 });
// CHECK-LABEL: @test_longlongvec32_custom()
// CHECK:       %custom = alloca <4 x i64>, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store <4 x i64> <i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323>, <4 x i64>* %custom, align [[ALIGN]]
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(floatvec16, float  __attribute__((vector_size(16))));
// CHECK-LABEL: @test_floatvec16_uninit()
// CHECK:       %uninit = alloca <4 x float>, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_floatvec16_uninit()
// PATTERN: store <4 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>, <4 x float>* %uninit, align 16, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_floatvec16_uninit()
// ZERO: store <4 x float> zeroinitializer, <4 x float>* %uninit, align 16, !annotation [[AUTO_INIT]]

TEST_BRACES(floatvec16, float  __attribute__((vector_size(16))));
// CHECK-LABEL: @test_floatvec16_braces()
// CHECK:       %braces = alloca <4 x float>, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store <4 x float> zeroinitializer, <4 x float>* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(floatvec16, float  __attribute__((vector_size(16))), { 3.1415926535897932384626433, 3.1415926535897932384626433, 3.1415926535897932384626433, 3.1415926535897932384626433 });
// CHECK-LABEL: @test_floatvec16_custom()
// CHECK:       %custom = alloca <4 x float>, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store <4 x float> <float 0x400921FB60000000, float 0x400921FB60000000, float 0x400921FB60000000, float 0x400921FB60000000>, <4 x float>* %custom, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(doublevec32, double  __attribute__((vector_size(32))));
// CHECK-LABEL: @test_doublevec32_uninit()
// CHECK:       %uninit = alloca <4 x double>, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_doublevec32_uninit()
// PATTERN: store <4 x double> <double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF>, <4 x double>* %uninit, align 32, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_doublevec32_uninit()
// ZERO: store <4 x double> zeroinitializer, <4 x double>* %uninit, align 32, !annotation [[AUTO_INIT]]

TEST_BRACES(doublevec32, double  __attribute__((vector_size(32))));
// CHECK-LABEL: @test_doublevec32_braces()
// CHECK:       %braces = alloca <4 x double>, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store <4 x double> zeroinitializer, <4 x double>* %braces, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(doublevec32, double  __attribute__((vector_size(32))), { 3.1415926535897932384626433, 3.1415926535897932384626433, 3.1415926535897932384626433, 3.1415926535897932384626433 });
// CHECK-LABEL: @test_doublevec32_custom()
// CHECK:       %custom = alloca <4 x double>, align [[ALIGN:[0-9]*]]
// CHECK-NEXT:  store <4 x double> <double 0x400921FB54442D18, double 0x400921FB54442D18, double 0x400921FB54442D18, double 0x400921FB54442D18>, <4 x double>* %custom, align [[ALIGN]]
// CHECK-NOT:   !annotation
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

// TODO: This vector has tail padding
TEST_UNINIT(doublevec24, double  __attribute__((vector_size(24))));
// CHECK-LABEL: @test_doublevec24_uninit()
// CHECK:       %uninit = alloca <3 x double>, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_doublevec24_uninit()
// PATTERN: store <3 x double> <double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF>, <3 x double>* %uninit, align 32, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_doublevec24_uninit()
// ZERO: store <3 x double> zeroinitializer, <3 x double>* %uninit, align 32, !annotation [[AUTO_INIT]]

// TODO: This vector has tail padding
TEST_UNINIT(longdoublevec32, long double  __attribute__((vector_size(sizeof(long double)*2))));
// CHECK-LABEL: @test_longdoublevec32_uninit()
// CHECK:       %uninit = alloca <2 x x86_fp80>, align
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)
// PATTERN-LABEL: @test_longdoublevec32_uninit()
// PATTERN: store <2 x x86_fp80> <x86_fp80 0xKFFFFFFFFFFFFFFFFFFFF, x86_fp80 0xKFFFFFFFFFFFFFFFFFFFF>, <2 x x86_fp80>* %uninit, align 32, !annotation [[AUTO_INIT]]
// ZERO-LABEL: @test_longdoublevec32_uninit()
// ZERO: store <2 x x86_fp80> zeroinitializer, <2 x x86_fp80>* %uninit, align 32, !annotation [[AUTO_INIT]]

} // extern "C"

// PATTERN: [[AUTO_INIT]] = !{!"auto-init"}
// ZERO: [[AUTO_INIT]] = !{!"auto-init"}
