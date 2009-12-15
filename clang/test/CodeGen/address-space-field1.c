// RUN: %clang_cc1 -emit-llvm < %s -o - | FileCheck %s
// CHECK:%struct.S = type { i32, i32 }
// CHECK:define void @test_addrspace(%struct.S addrspace(1)* %p1, %struct.S addrspace(2)* %p2) nounwind
// CHECK:  [[p1addr:%.*]] = alloca %struct.S addrspace(1)*       ; <%struct.S addrspace(1)**> [#uses=3]
// CHECK:  [[p2addr:%.*]] = alloca %struct.S addrspace(2)*       ; <%struct.S addrspace(2)**> [#uses=3]
// CHECK:  store %struct.S addrspace(1)* %p1, %struct.S addrspace(1)** [[p1addr]]
// CHECK:  store %struct.S addrspace(2)* %p2, %struct.S addrspace(2)** [[p2addr]]
// CHECK:  [[t0:%.*]] = load %struct.S addrspace(2)** [[p2addr]]   ; <%struct.S addrspace(2)*> [#uses=1]
// CHECK:  [[t1:%.*]] = getelementptr inbounds %struct.S addrspace(2)* [[t0]], i32 0, i32 1 ; <i32 addrspace(2)*> [#uses=1]
// CHECK:  [[t2:%.*]] = load i32 addrspace(2)* [[t1]]            ; <i32> [#uses=1]
// CHECK:  [[t3:%.*]] = load %struct.S addrspace(1)** [[p1addr]]  ; <%struct.S addrspace(1)*> [#uses=1]
// CHECK:  [[t4:%.*]] = getelementptr inbounds %struct.S addrspace(1)* [[t3]], i32 0, i32 0 ; <i32 addrspace(1)*> [#uses=1]
// CHECK:  store i32 [[t2]], i32 addrspace(1)* [[t4]]
// CHECK:  [[t5:%.*]] = load %struct.S addrspace(2)** [[p2addr]]  ; <%struct.S addrspace(2)*> [#uses=1]
// CHECK:  [[t6:%.*]] = getelementptr inbounds %struct.S addrspace(2)* [[t5]], i32 0, i32 0 ; <i32 addrspace(2)*> [#uses=1]
// CHECK:  [[t7:%.*]] = load i32 addrspace(2)* [[t6]]            ; <i32> [#uses=1]
// CHECK:  [[t8:%.*]] = load %struct.S addrspace(1)** [[p1addr]]  ; <%struct.S addrspace(1)*> [#uses=1]
// CHECK:  [[t9:%.*]] = getelementptr inbounds %struct.S addrspace(1)* [[t8]], i32 0, i32 1 ; <i32 addrspace(1)*> [#uses=1]
// CHECK:  store i32 [[t7]], i32 addrspace(1)* [[t9]]
// CHECK:  ret void
// CHECK:}

// Check that we don't lose the address space when accessing a member
// of a structure.

#define __addr1    __attribute__((address_space(1)))
#define __addr2    __attribute__((address_space(2)))

typedef struct S {
  int a;
  int b;
} S;

void test_addrspace(__addr1 S* p1, __addr2 S*p2) {
  // swap
  p1->a = p2->b;
  p1->b = p2->a;
}
