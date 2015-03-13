; RUN: opt < %s -instsimplify -S | FileCheck %s
target datalayout = "p:32:32"

; Check some past-the-end subtleties.

@opte_a = global i32 0
@opte_b = global i32 0

; Comparing base addresses of two distinct globals. Never equal.

define zeroext i1 @no_offsets() {
  %t = icmp eq i32* @opte_a, @opte_b
  ret i1 %t
  ; CHECK: no_offsets(
  ; CHECK: ret i1 false
}

; Comparing past-the-end addresses of two distinct globals. Never equal.

define zeroext i1 @both_past_the_end() {
  %x = getelementptr i32, i32* @opte_a, i32 1
  %y = getelementptr i32, i32* @opte_b, i32 1
  %t = icmp eq i32* %x, %y
  ret i1 %t
  ; CHECK: both_past_the_end(
  ; CHECK-NOT: ret i1 true
  ; TODO: refine this
}

; Comparing past-the-end addresses of one global to the base address
; of another. Can't fold this.

define zeroext i1 @just_one_past_the_end() {
  %x = getelementptr i32, i32* @opte_a, i32 1
  %t = icmp eq i32* %x, @opte_b
  ret i1 %t
  ; CHECK: just_one_past_the_end(
  ; CHECK: ret i1 icmp eq (i32* getelementptr inbounds (i32, i32* @opte_a, i32 1), i32* @opte_b)
}

; Comparing base addresses of two distinct allocas. Never equal.

define zeroext i1 @no_alloca_offsets() {
  %m = alloca i32
  %n = alloca i32
  %t = icmp eq i32* %m, %n
  ret i1 %t
  ; CHECK: no_alloca_offsets(
  ; CHECK: ret i1 false
}

; Comparing past-the-end addresses of two distinct allocas. Never equal.

define zeroext i1 @both_past_the_end_alloca() {
  %m = alloca i32
  %n = alloca i32
  %x = getelementptr i32, i32* %m, i32 1
  %y = getelementptr i32, i32* %n, i32 1
  %t = icmp eq i32* %x, %y
  ret i1 %t
  ; CHECK: both_past_the_end_alloca(
  ; CHECK-NOT: ret i1 true
  ; TODO: refine this
}

; Comparing past-the-end addresses of one alloca to the base address
; of another. Can't fold this.

define zeroext i1 @just_one_past_the_end_alloca() {
  %m = alloca i32
  %n = alloca i32
  %x = getelementptr i32, i32* %m, i32 1
  %t = icmp eq i32* %x, %n
  ret i1 %t
  ; CHECK: just_one_past_the_end_alloca(
  ; CHECK: ret i1 %t
}
