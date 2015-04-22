; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu %s -o - -O0 | FileCheck --check-prefix=NOOPT %s

declare void @g(i32)

define void @basic(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 3, label %bb0
    i32 1, label %bb1
    i32 4, label %bb1
    i32 5, label %bb0
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
return: ret void

; Should be lowered as straight compares in -O0 mode.
; NOOPT-LABEL: basic
; NOOPT: subl $3, %eax
; NOOPT: je
; NOOPT: subl $1, %eax
; NOOPT: je
; NOOPT: subl $4, %eax
; NOOPT: je
; NOOPT: subl $5, %eax
; NOOPT: je

; Jump table otherwise.
; CHECK-LABEL: basic
; CHECK: decl
; CHECK: cmpl $4
; CHECK: ja
; CHECK: jmpq *.LJTI
}


define void @simple_ranges(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 0, label %bb0
    i32 1, label %bb0
    i32 2, label %bb0
    i32 3, label %bb0
    i32 100, label %bb1
    i32 101, label %bb1
    i32 102, label %bb1
    i32 103, label %bb1
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
return: ret void

; Should be lowered to two range checks.
; CHECK-LABEL: simple_ranges
; CHECK: leal -100
; CHECK: cmpl $4
; CHECK: jae
; CHECK: cmpl $3
; CHECK: ja
}


define void @jt_is_better(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 0, label %bb0
    i32 2, label %bb0
    i32 4, label %bb0
    i32 1, label %bb1
    i32 3, label %bb1
    i32 5, label %bb1

    i32 6, label %bb2
    i32 7, label %bb3
    i32 8, label %bb4
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 2) br label %return
bb3: tail call void @g(i32 3) br label %return
bb4: tail call void @g(i32 4) br label %return
return: ret void

; Cases 0-5 could be lowered with two bit tests,
; but with 6-8, the whole switch is suitable for a jump table.
; CHECK-LABEL: jt_is_better
; CHECK: cmpl $8
; CHECK: jbe
; CHECK: jmpq *.LJTI
}


define void @bt_is_better(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 0, label %bb0
    i32 3, label %bb0
    i32 6, label %bb0
    i32 1, label %bb1
    i32 4, label %bb1
    i32 7, label %bb1
    i32 2, label %bb2
    i32 5, label %bb2
    i32 8, label %bb2

  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 2) br label %return
return: ret void

; This could be lowered as a jump table, but bit tests is more efficient.
; CHECK-LABEL: bt_is_better
; 73 = 2^0 + 2^3 + 2^6
; CHECK: movl $73
; CHECK: btl
; 146 = 2^1 + 2^4 + 2^7
; CHECK: movl $146
; CHECK: btl
; 292 = 2^2 + 2^5 + 2^8
; CHECK: movl $292
; CHECK: btl
}


define void @optimal_pivot1(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 100, label %bb0
    i32 200, label %bb1
    i32 300, label %bb0
    i32 400, label %bb1
    i32 500, label %bb0
    i32 600, label %bb1

  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
return: ret void

; Should pivot around 400 for two subtrees of equal size.
; CHECK-LABEL: optimal_pivot1
; CHECK-NOT: cmpl
; CHECK: cmpl $399
}


define void @optimal_pivot2(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 100, label %bb0   i32 101, label %bb1   i32 102, label %bb2   i32 103, label %bb3
    i32 200, label %bb0   i32 201, label %bb1   i32 202, label %bb2   i32 203, label %bb3
    i32 300, label %bb0   i32 301, label %bb1   i32 302, label %bb2   i32 303, label %bb3
    i32 400, label %bb0   i32 401, label %bb1   i32 402, label %bb2   i32 403, label %bb3

  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 2) br label %return
bb3: tail call void @g(i32 3) br label %return
return: ret void

; Should pivot around 300 for two subtrees with two jump tables each.
; CHECK-LABEL: optimal_pivot2
; CHECK-NOT: cmpl
; CHECK: cmpl $299
; CHECK: jmpq *.LJTI
; CHECK: jmpq *.LJTI
; CHECK: jmpq *.LJTI
; CHECK: jmpq *.LJTI
}


define void @optimal_jump_table1(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 0,  label %bb0
    i32 5,  label %bb1
    i32 6,  label %bb2
    i32 12, label %bb3
    i32 13, label %bb4
    i32 15, label %bb5
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 2) br label %return
bb3: tail call void @g(i32 3) br label %return
bb4: tail call void @g(i32 4) br label %return
bb5: tail call void @g(i32 5) br label %return
return: ret void

; Splitting in the largest gap (between 6 and 12) would yield suboptimal result.
; Expecting a jump table from 5 to 15.
; CHECK-LABEL: optimal_jump_table1
; CHECK: leal -5
; CHECK: cmpl $10
; CHECK: jmpq *.LJTI
}


define void @optimal_jump_table2(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 0,  label %bb0
    i32 1,  label %bb1
    i32 2,  label %bb2
    i32 9,  label %bb3
    i32 14, label %bb4
    i32 15, label %bb5
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 2) br label %return
bb3: tail call void @g(i32 3) br label %return
bb4: tail call void @g(i32 4) br label %return
bb5: tail call void @g(i32 5) br label %return
return: ret void

; Partitioning the cases to the minimum number of dense sets is not good enough.
; This can be partitioned as {0,1,2,9},{14,15} or {0,1,2},{9,14,15}. The former
; should be preferred. Expecting a table from 0-9.
; CHECK-LABEL: optimal_jump_table2
; CHECK: cmpl $9
; CHECK: jmpq *.LJTI
}


define void @optimal_jump_table3(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 1,  label %bb0
    i32 2,  label %bb1
    i32 3,  label %bb2
    i32 10, label %bb3
    i32 13, label %bb0
    i32 14, label %bb1
    i32 15, label %bb2
    i32 20, label %bb3
    i32 25, label %bb4
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 2) br label %return
bb3: tail call void @g(i32 3) br label %return
bb4: tail call void @g(i32 4) br label %return
return: ret void

; Splitting to maximize left-right density sum and gap size would split this
; between 3 and 10, and then between 20 and 25. It's better to build a table
; from 1-20.
; CHECK-LABEL: optimal_jump_table3
; CHECK: leal -1
; CHECK: cmpl $19
; CHECK: jmpq *.LJTI
}

%struct.S = type { %struct.S*, i32 }
define void @phi_node_trouble(%struct.S* %s) {
entry:
  br label %header
header:
  %ptr = phi %struct.S* [ %s, %entry ], [ %next, %loop ]
  %bool = icmp eq %struct.S* %ptr, null
  br i1 %bool, label %exit, label %loop
loop:
  %nextptr = getelementptr inbounds %struct.S, %struct.S* %ptr, i64 0, i32 0
  %next = load %struct.S*, %struct.S** %nextptr
  %xptr = getelementptr inbounds %struct.S, %struct.S* %next, i64 0, i32 1
  %x = load i32, i32* %xptr
  switch i32 %x, label %exit [
    i32 4, label %header
    i32 36, label %exit2
    i32 69, label %exit2
    i32 25, label %exit2
  ]
exit:
  ret void
exit2:
  ret void

; This will be lowered to a comparison with 4 and then bit tests. Make sure
; that the phi node in %header gets a value from the comparison block.
; CHECK-LABEL: phi_node_trouble
; CHECK: movq (%[[REG1:[a-z]+]]), %[[REG1]]
; CHECK: movl 8(%[[REG1]]), %[[REG2:[a-z]+]]
; CHECK: cmpl $4, %[[REG2]]
}


define void @default_only(i32 %x) {
entry:
  br label %sw
return:
  ret void
sw:
  switch i32 %x, label %return [
  ]

; Branch directly to the default.
; (In optimized builds the switch is removed earlier.)
; NOOPT-LABEL: default_only
; NOOPT: .[[L:[A-Z0-9_]+]]:
; NOOPT-NEXT: retq
; NOOPT: jmp .[[L]]
}
