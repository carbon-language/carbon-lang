// RUN: %clang_cc1 -triple i686--windows -fms-compatibility -Oz -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple thumbv7--windows -fms-compatibility -Oz -emit-llvm %s -o - | FileCheck %s

void *test_InterlockedExchangePointer(void * volatile *Target, void *Value) {
  return _InterlockedExchangePointer(Target, Value);
}

// CHECK: define{{.*}}i8* @test_InterlockedExchangePointer(i8** %Target, i8* %Value){{.*}}{
// CHECK: entry:
// CHECK:   %0 = bitcast i8** %Target to i32*
// CHECK:   %1 = ptrtoint i8* %Value to i32
// CHECK:   %2 = atomicrmw xchg i32* %0, i32 %1 seq_cst
// CHECK:   %3 = inttoptr i32 %2 to i8*
// CHECK:   ret i8* %3
// CHECK: }

void *test_InterlockedCompareExchangePointer(void * volatile *Destination,
                                             void *Exchange, void *Comparand) {
  return _InterlockedCompareExchangePointer(Destination, Exchange, Comparand);
}

// CHECK: define{{.*}}i8* @test_InterlockedCompareExchangePointer(i8** %Destination, i8* %Exchange, i8* %Comparand){{.*}}{
// CHECK: entry:
// CHECK:   %0 = bitcast i8** %Destination to i32*
// CHECK:   %1 = ptrtoint i8* %Exchange to i32
// CHECK:   %2 = ptrtoint i8* %Comparand to i32
// CHECK:   %3 = cmpxchg volatile i32* %0, i32 %2, i32 %1 seq_cst seq_cst
// CHECK:   %4 = extractvalue { i32, i1 } %3, 0
// CHECK:   %5 = inttoptr i32 %4 to i8*
// CHECK:   ret i8* %5
// CHECK: }

long test_InterlockedExchange(long *Target, long Value) {
  return _InterlockedExchange(Target, Value);
}

// CHECK: define{{.*}}i32 @test_InterlockedExchange(i32* %Target, i32 %Value){{.*}}{
// CHECK: entry:
// CHECK:   %0 = atomicrmw xchg i32* %Target, i32 %Value seq_cst
// CHECK:   ret i32 %0
// CHECK: }
