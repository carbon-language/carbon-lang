// RUN: clang-cc -emit-llvm < %s -o - | FileCheck %s
// CHECK:%struct.S = type { i32, i32 }
// CHECK:define void @test_addrspace(%struct.S addrspace(1)* %p1, %struct.S addrspace(2)* %p2) nounwind {
// CHECK:entry:
// CHECK:  %p1.addr = alloca %struct.S addrspace(1)*       ; <%struct.S addrspace(1)**> [#uses=3]
// CHECK:  %p2.addr = alloca %struct.S addrspace(2)*       ; <%struct.S addrspace(2)**> [#uses=3]
// CHECK:  store %struct.S addrspace(1)* %p1, %struct.S addrspace(1)** %p1.addr
// CHECK:  store %struct.S addrspace(2)* %p2, %struct.S addrspace(2)** %p2.addr
// CHECK:  %tmp = load %struct.S addrspace(2)** %p2.addr   ; <%struct.S addrspace(2)*> [#uses=1]
// CHECK:  %tmp1 = getelementptr inbounds %struct.S addrspace(2)* %tmp, i32 0, i32 1 ; <i32 addrspace(2)*> [#uses=1]
// CHECK:  %tmp2 = load i32 addrspace(2)* %tmp1            ; <i32> [#uses=1]
// CHECK:  %tmp3 = load %struct.S addrspace(1)** %p1.addr  ; <%struct.S addrspace(1)*> [#uses=1]
// CHECK:  %tmp4 = getelementptr inbounds %struct.S addrspace(1)* %tmp3, i32 0, i32 0 ; <i32 addrspace(1)*> [#uses=1]
// CHECK:  store i32 %tmp2, i32 addrspace(1)* %tmp4
// CHECK:  %tmp5 = load %struct.S addrspace(2)** %p2.addr  ; <%struct.S addrspace(2)*> [#uses=1]
// CHECK:  %tmp6 = getelementptr inbounds %struct.S addrspace(2)* %tmp5, i32 0, i32 0 ; <i32 addrspace(2)*> [#uses=1]
// CHECK:  %tmp7 = load i32 addrspace(2)* %tmp6            ; <i32> [#uses=1]
// CHECK:  %tmp8 = load %struct.S addrspace(1)** %p1.addr  ; <%struct.S addrspace(1)*> [#uses=1]
// CHECK:  %tmp9 = getelementptr inbounds %struct.S addrspace(1)* %tmp8, i32 0, i32 1 ; <i32 addrspace(1)*> [#uses=1]
// CHECK:  store i32 %tmp7, i32 addrspace(1)* %tmp9
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
