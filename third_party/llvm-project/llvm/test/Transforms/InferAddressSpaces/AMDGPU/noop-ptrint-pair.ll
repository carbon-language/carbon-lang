; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -o - -infer-address-spaces %s | FileCheck -check-prefixes=COMMON,AMDGCN %s
; RUN: opt -S -o - -infer-address-spaces -assume-default-is-flat-addrspace %s | FileCheck -check-prefixes=COMMON,NOTTI %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7"

; COMMON-LABEL: @noop_ptrint_pair(
; AMDGCN-NEXT: store i32 0, i32 addrspace(1)* %{{.*}}
; AMDGCN-NEXT: ret void
; NOTTI-NEXT: %1 = ptrtoint i32 addrspace(1)* %x.coerce to i64
; NOTTI-NEXT: %2 = inttoptr i64 %1 to i32*
; NOTTI-NEXT: store i32 0, i32* %2
; NOTTI-NEXT: ret void
define void @noop_ptrint_pair(i32 addrspace(1)* %x.coerce) {
  %1 = ptrtoint i32 addrspace(1)* %x.coerce to i64
  %2 = inttoptr i64 %1 to i32*
  store i32 0, i32* %2
  ret void
}

; COMMON-LABEL: @non_noop_ptrint_pair(
; AMDGCN-NEXT: ptrtoint i32 addrspace(3)* %{{.*}} to i64
; AMDGCN-NEXT: inttoptr i64 %{{.*}} to i32*
; AMDGCN-NEXT: store i32 0, i32* %{{.*}}
; AMDGCN-NEXT: ret void
; NOTTI-NEXT: ptrtoint i32 addrspace(3)* %{{.*}} to i64
; NOTTI-NEXT: inttoptr i64 %{{.*}} to i32*
; NOTTI-NEXT: store i32 0, i32* %{{.*}}
; NOTTI-NEXT: ret void
define void @non_noop_ptrint_pair(i32 addrspace(3)* %x.coerce) {
  %1 = ptrtoint i32 addrspace(3)* %x.coerce to i64
  %2 = inttoptr i64 %1 to i32*
  store i32 0, i32* %2
  ret void
}

; COMMON-LABEL: @non_noop_ptrint_pair2(
; AMDGCN-NEXT: ptrtoint i32 addrspace(1)* %{{.*}} to i32
; AMDGCN-NEXT: inttoptr i32 %{{.*}} to i32*
; AMDGCN-NEXT: store i32 0, i32* %{{.*}}
; AMDGCN-NEXT: ret void
; NOTTI-NEXT: ptrtoint i32 addrspace(1)* %{{.*}} to i32
; NOTTI-NEXT: inttoptr i32 %{{.*}} to i32*
; NOTTI-NEXT: store i32 0, i32* %{{.*}}
; NOTTI-NEXT: ret void
define void @non_noop_ptrint_pair2(i32 addrspace(1)* %x.coerce) {
  %1 = ptrtoint i32 addrspace(1)* %x.coerce to i32
  %2 = inttoptr i32 %1 to i32*
  store i32 0, i32* %2
  ret void
}

@g = addrspace(1) global i32 0, align 4
@l = addrspace(3) global i32 0, align 4

; COMMON-LABEL: @noop_ptrint_pair_ce(
; AMDGCN-NEXT: store i32 0, i32 addrspace(1)* @g
; AMDGCN-NEXT: ret void
; NOTTI-NEXT: store i32 0, i32* inttoptr (i64 ptrtoint (i32 addrspace(1)* @g to i64) to i32*)
; NOTTI-NEXT: ret void
define void @noop_ptrint_pair_ce() {
  store i32 0, i32* inttoptr (i64 ptrtoint (i32 addrspace(1)* @g to i64) to i32*)
  ret void
}

; COMMON-LABEL: @noop_ptrint_pair_ce2(
; AMDGCN-NEXT: ret i32* addrspacecast (i32 addrspace(1)* @g to i32*)
; NOTTI-NEXT: ret i32* inttoptr (i64 ptrtoint (i32 addrspace(1)* @g to i64) to i32*)
define i32* @noop_ptrint_pair_ce2() {
  ret i32* inttoptr (i64 ptrtoint (i32 addrspace(1)* @g to i64) to i32*)
}

; COMMON-LABEL: @noop_ptrint_pair_ce3(
; AMDGCN-NEXT: %i = inttoptr i64 ptrtoint (i32 addrspace(1)* @g to i64) to i32*
; AMDGCN-NEXT: ret void
; NOTTI-NEXT: %i = inttoptr i64 ptrtoint (i32 addrspace(1)* @g to i64) to i32*
; NOTTI-NEXT: ret void
define void @noop_ptrint_pair_ce3() {
  %i = inttoptr i64 ptrtoint (i32 addrspace(1)* @g to i64) to i32*
  ret void
}

; COMMON-LABEL: @non_noop_ptrint_pair_ce(
; AMDGCN-NEXT: store i32 0, i32* inttoptr (i64 ptrtoint (i32 addrspace(3)* @l to i64) to i32*)
; AMDGCN-NEXT: ret void
; NOTTI-NEXT: store i32 0, i32* inttoptr (i64 ptrtoint (i32 addrspace(3)* @l to i64) to i32*)
; NOTTI-NEXT: ret void
define void @non_noop_ptrint_pair_ce() {
  store i32 0, i32* inttoptr (i64 ptrtoint (i32 addrspace(3)* @l to i64) to i32*)
  ret void
}

; COMMON-LABEL: @non_noop_ptrint_pair_ce2(
; AMDGCN-NEXT: ret i32* inttoptr (i64 ptrtoint (i32 addrspace(3)* @l to i64) to i32*)
; NOTTI-NEXT: ret i32* inttoptr (i64 ptrtoint (i32 addrspace(3)* @l to i64) to i32*)
define i32* @non_noop_ptrint_pair_ce2() {
  ret i32* inttoptr (i64 ptrtoint (i32 addrspace(3)* @l to i64) to i32*)
}

; COMMON-LABEL: @non_noop_ptrint_pair_ce3(
; AMDGCN-NEXT: ret i32* inttoptr (i32 ptrtoint (i32 addrspace(1)* @g to i32) to i32*)
; NOTTI-NEXT: ret i32* inttoptr (i32 ptrtoint (i32 addrspace(1)* @g to i32) to i32*)
define i32* @non_noop_ptrint_pair_ce3() {
  ret i32* inttoptr (i32 ptrtoint (i32 addrspace(1)* @g to i32) to i32*)
}

; COMMON-LABEL: @non_noop_ptrint_pair_ce4(
; AMDGCN-NEXT: ret i32* inttoptr (i128 ptrtoint (i32 addrspace(3)* @l to i128) to i32*)
; NOTTI-NEXT: ret i32* inttoptr (i128 ptrtoint (i32 addrspace(3)* @l to i128) to i32*)
define i32* @non_noop_ptrint_pair_ce4() {
  ret i32* inttoptr (i128 ptrtoint (i32 addrspace(3)* @l to i128) to i32*)
}
