; RUN: llc -mtriple=aarch64-none-linux-gnu -o - %s | FileCheck %s

;; Check that the llvm aarch64 backend can handle arrays of
;; structs and vice versa when passed from IR.
;; (this layering is something clang would normally simplify)
;;
;; Some of these examples are not ABI compliant and they're not
;; meant to be. For instance according to the ABI an aggregate
;; with more than 4 members must go in memory. This restriction
;; is applied earlier in the compilation process so here we do
;; see 8 member types in registers.
;;
;; When we have more than 8 members we simply run out of registers
;; and that's what produces the 8 limit here.

;; Plain arrays

define [ 0 x double ] @array_0() {
; CHECK-LABEL: array_0:
; CHECK:  ret
  ret [ 0 x double ] zeroinitializer
}

define [ 1 x double ] @array_1() {
; CHECK-LABEL: array_1:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  ret
  ret [ 1 x double ] zeroinitializer
}

define [ 8 x double ] @array_8() {
; CHECK-LABEL: array_8:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  movi d1, #0000000000000000
; CHECK-NEXT:  movi d2, #0000000000000000
; CHECK-NEXT:  movi d3, #0000000000000000
; CHECK-NEXT:  movi d4, #0000000000000000
; CHECK-NEXT:  movi d5, #0000000000000000
; CHECK-NEXT:  movi d6, #0000000000000000
; CHECK-NEXT:  movi d7, #0000000000000000
; CHECK-NEXT:  ret
  ret [ 8 x double ] zeroinitializer
}

;; > 8 items goes on the stack

define [ 9 x double ] @array_9() {
; CHECK-LABEL: array_9:
; CHECK:      movi v0.2d, #0000000000000000
; CHECK-NEXT: str xzr, [x8, #64]
; CHECK-NEXT: stp q0, q0, [x8, #32]
; CHECK-NEXT: stp q0, q0, [x8]
; CHECK-NEXT:  ret
  ret [ 9 x double ] zeroinitializer
}

;; Won't use any registers, just checking for assumptions.
%T_STRUCT_0M = type { }

define %T_STRUCT_0M @struct_zero_fields() {
; CHECK-LABEL: struct_zero_fields:
; CHECK:  ret
  ret %T_STRUCT_0M zeroinitializer
}

define [ 1 x %T_STRUCT_0M ] @array_of_struct_zero_fields() {
; CHECK-LABEL: array_of_struct_zero_fields:
; CHECK:  ret
  ret [ 1 x %T_STRUCT_0M ] zeroinitializer
}

define [ 2 x %T_STRUCT_0M ] @array_of_struct_zero_fields_in_struct() {
; CHECK-LABEL: array_of_struct_zero_fields_in_struct:
; CHECK:  ret
  ret [ 2 x %T_STRUCT_0M ] zeroinitializer
}

%T_STRUCT_1M = type { i32 }

define %T_STRUCT_1M @struct_one_field() {
; CHECK-LABEL: struct_one_field:
; CHECK:       w0, wzr
; CHECK-NEXT:  ret
  ret %T_STRUCT_1M zeroinitializer
}

define [ 1 x %T_STRUCT_1M ] @array_of_struct_one_field() {
; CHECK-LABEL: array_of_struct_one_field:
; CHECK:       w0, wzr
; CHECK-NEXT:  ret
  ret [ 1 x %T_STRUCT_1M ] zeroinitializer
}

;; This one will be a reg block
define [ 2 x %T_STRUCT_1M ] @array_of_struct_one_field_2() {
; CHECK-LABEL: array_of_struct_one_field_2:
; CHECK:       w0, wzr
; CHECK:       w1, wzr
; CHECK-NEXT:  ret
  ret [ 2 x %T_STRUCT_1M ] zeroinitializer
}

;; Different types for each field, will not be put in a reg block
%T_STRUCT_DIFFM = type { double, i32 }

define %T_STRUCT_DIFFM @struct_different_field_types() {
; CHECK-LABEL: struct_different_field_types:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  w0, wzr
; CHECK-NEXT:  ret
  ret %T_STRUCT_DIFFM zeroinitializer
}

define [ 1 x %T_STRUCT_DIFFM ] @array_of_struct_different_field_types() {
; CHECK-LABEL: array_of_struct_different_field_types:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  w0, wzr
; CHECK-NEXT:  ret
  ret [ 1 x %T_STRUCT_DIFFM ] zeroinitializer
}

define [ 2 x %T_STRUCT_DIFFM ] @array_of_struct_different_field_types_2() {
; CHECK-LABEL: array_of_struct_different_field_types_2:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  movi d1, #0000000000000000
; CHECK-NEXT:  w0, wzr
; CHECK-NEXT:  w1, wzr
; CHECK-NEXT:  ret
  ret [ 2 x %T_STRUCT_DIFFM ] zeroinitializer
}

;; Each field is the same type, can be put in a reg block
%T_STRUCT_SAMEM = type { double, double }

;; Here isn't a block as such, we just allocate two consecutive registers
define %T_STRUCT_SAMEM @struct_same_field_types() {
; CHECK-LABEL: struct_same_field_types:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  movi d1, #0000000000000000
; CHECK-NEXT:  ret
  ret %T_STRUCT_SAMEM zeroinitializer
}

define [ 1 x %T_STRUCT_SAMEM ] @array_of_struct_same_field_types() {
; CHECK-LABEL: array_of_struct_same_field_types:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  movi d1, #0000000000000000
; CHECK-NEXT:  ret
  ret [ 1 x %T_STRUCT_SAMEM ] zeroinitializer
}

define [ 2 x %T_STRUCT_SAMEM ] @array_of_struct_same_field_types_2() {
; CHECK-LABEL: array_of_struct_same_field_types_2:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  movi d1, #0000000000000000
; CHECK-NEXT:  movi d2, #0000000000000000
; CHECK-NEXT:  movi d3, #0000000000000000
; CHECK-NEXT:  ret
  ret [ 2 x %T_STRUCT_SAMEM ] zeroinitializer
}

;; Same field type but integer this time. Put into x registers instead.
%T_STRUCT_SAMEM_INT = type { i64, i64 }

define %T_STRUCT_SAMEM_INT @struct_same_field_types_int() {
; CHECK-LABEL: struct_same_field_types_int:
; CHECK:       x0, xzr
; CHECK-NEXT:  x1, xzr
; CHECK-NEXT:  ret
  ret %T_STRUCT_SAMEM_INT zeroinitializer
}

define [ 1 x %T_STRUCT_SAMEM_INT ] @array_of_struct_same_field_types_int() {
; CHECK-LABEL: array_of_struct_same_field_types_int:
; CHECK:       x0, xzr
; CHECK-NEXT:  x1, xzr
; CHECK-NEXT:  ret
  ret [ 1 x %T_STRUCT_SAMEM_INT ] zeroinitializer
}

define [ 2 x %T_STRUCT_SAMEM_INT ] @array_of_struct_same_field_types_int_2() {
; CHECK-LABEL: array_of_struct_same_field_types_int_2:
; CHECK:       x0, xzr
; CHECK-NEXT:  x1, xzr
; CHECK-NEXT:  x2, xzr
; CHECK-NEXT:  x3, xzr
; CHECK-NEXT:  ret
  ret [ 2 x %T_STRUCT_SAMEM_INT ] zeroinitializer
}

;; An aggregate of more than 8 items must go in memory.
;; 4x2 struct fields = 8 items so it goes in a block.

define [ 4 x %T_STRUCT_SAMEM ] @array_of_struct_8_fields() {
; CHECK-LABEL: array_of_struct_8_fields:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  movi d1, #0000000000000000
; CHECK-NEXT:  movi d2, #0000000000000000
; CHECK-NEXT:  movi d3, #0000000000000000
; CHECK-NEXT:  movi d4, #0000000000000000
; CHECK-NEXT:  movi d5, #0000000000000000
; CHECK-NEXT:  movi d6, #0000000000000000
; CHECK-NEXT:  movi d7, #0000000000000000
; CHECK-NEXT:  ret
  ret [ 4 x %T_STRUCT_SAMEM ] zeroinitializer
}

;; 5x2 fields = 10 so it is returned in memory.

define [ 5 x %T_STRUCT_SAMEM ] @array_of_struct_in_memory() {
; CHECK-LABEL: array_of_struct_in_memory:
; CHECK:       movi    v0.2d, #0000000000000000
; CHECK-NEXT:  stp     q0, q0, [x8, #48]
; CHECK-NEXT:  stp     q0, q0, [x8, #16]
; CHECK-NEXT:  str     q0, [x8]
; CHECK-NEXT:  ret
  ret [ 5 x %T_STRUCT_SAMEM ] zeroinitializer
}

;; A struct whose field is an array.
%T_STRUCT_ARRAYM = type { [ 2 x double ]};

define %T_STRUCT_ARRAYM @struct_array_field() {
; CHECK-LABEL: struct_array_field:
; CHECK:       movi    d0, #0000000000000000
; CHECK-NEXT:  movi    d1, #0000000000000000
; CHECK-NEXT:  ret
  ret %T_STRUCT_ARRAYM zeroinitializer
}

define [ 1 x %T_STRUCT_ARRAYM ] @array_of_struct_array_field() {
; CHECK-LABEL: array_of_struct_array_field:
; CHECK:       movi    d0, #0000000000000000
; CHECK-NEXT:  movi    d1, #0000000000000000
; CHECK-NEXT:  ret
  ret [ 1 x %T_STRUCT_ARRAYM ] zeroinitializer
}

define [ 2 x %T_STRUCT_ARRAYM ] @array_of_struct_array_field_2() {
; CHECK-LABEL: array_of_struct_array_field_2:
; CHECK:       movi    d0, #0000000000000000
; CHECK-NEXT:  movi    d1, #0000000000000000
; CHECK-NEXT:  movi    d2, #0000000000000000
; CHECK-NEXT:  movi    d3, #0000000000000000
; CHECK-NEXT:  ret
  ret [ 2 x %T_STRUCT_ARRAYM ] zeroinitializer
}

;; All non-aggregate fields must have the same type, all through the
;; overall aggreagate. This is false here because of the i32.
%T_NESTED_STRUCT_DIFFM = type {
  [ 1 x { { double, double } } ],
  [ 1 x { { double, i32 } } ]
};

define %T_NESTED_STRUCT_DIFFM @struct_nested_different_field_types() {
; CHECK-LABEL: struct_nested_different_field_types:
; CHECK:       movi d0, #0000000000000000
; CHECK:       movi d1, #0000000000000000
; CHECK:       movi d2, #0000000000000000
; CHECK-NEXT:  w0, wzr
; CHECK-NEXT:  ret
  ret %T_NESTED_STRUCT_DIFFM zeroinitializer
}

define [ 1 x %T_NESTED_STRUCT_DIFFM ] @array_of_struct_nested_different_field_types() {
; CHECK-LABEL: array_of_struct_nested_different_field_types:
; CHECK:       movi d0, #0000000000000000
; CHECK:       movi d1, #0000000000000000
; CHECK:       movi d2, #0000000000000000
; CHECK-NEXT:  w0, wzr
; CHECK-NEXT:  ret
  ret [ 1 x %T_NESTED_STRUCT_DIFFM ] zeroinitializer
}

define [ 2 x %T_NESTED_STRUCT_DIFFM ] @array_of_struct_nested_different_field_types_2() {
; CHECK-LABEL: array_of_struct_nested_different_field_types_2:
; CHECK:       movi d0, #0000000000000000
; CHECK:       movi d1, #0000000000000000
; CHECK:       movi d2, #0000000000000000
; CHECK-NEXT:  movi d3, #0000000000000000
; CHECK-NEXT:  movi d4, #0000000000000000
; CHECK-NEXT:  movi d5, #0000000000000000
; CHECK-NEXT:  w0, wzr
; CHECK-NEXT:  w1, wzr
; CHECK-NEXT:  ret
  ret [ 2 x %T_NESTED_STRUCT_DIFFM ] zeroinitializer
}

;; All fields here are the same type, more nesting to stress the recursive walk.
%T_NESTED_STRUCT_SAMEM = type {
  { { double} },
  { [ 2 x { double, double } ] }
};

define %T_NESTED_STRUCT_SAMEM @struct_nested_same_field_types() {
; CHECK-LABEL: struct_nested_same_field_types:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  movi d1, #0000000000000000
; CHECK-NEXT:  movi d2, #0000000000000000
; CHECK-NEXT:  movi d3, #0000000000000000
; CHECK-NEXT:  movi d4, #0000000000000000
; CHECK-NEXT:  ret
  ret %T_NESTED_STRUCT_SAMEM zeroinitializer
}

define [ 1 x %T_NESTED_STRUCT_SAMEM ] @array_of_struct_nested_same_field_types() {
; CHECK-LABEL: array_of_struct_nested_same_field_types:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  movi d1, #0000000000000000
; CHECK-NEXT:  movi d2, #0000000000000000
; CHECK-NEXT:  movi d3, #0000000000000000
; CHECK-NEXT:  movi d4, #0000000000000000
; CHECK-NEXT:  ret
  ret [ 1 x %T_NESTED_STRUCT_SAMEM ] zeroinitializer
}

;; 2 x (1 + (2 x 2)) = 10 so this is returned in memory
define [ 2 x %T_NESTED_STRUCT_SAMEM ] @array_of_struct_nested_same_field_types_2() {
; CHECK-LABEL: array_of_struct_nested_same_field_types_2:
; CHECK:      movi    v0.2d, #0000000000000000
; CHECK-NEXT: stp     q0, q0, [x8, #48]
; CHECK-NEXT: stp     q0, q0, [x8, #16]
; CHECK-NEXT: str     q0, [x8]
; CHECK-NEXT: ret
  ret [ 2 x %T_NESTED_STRUCT_SAMEM ] zeroinitializer
}

;; Check combinations of call, return and argument passing

%T_IN_BLOCK = type [ 2 x { double, { double, double } } ]

define %T_IN_BLOCK @return_in_block() {
; CHECK-LABEL: return_in_block:
; CHECK:      movi d0, #0000000000000000
; CHECK-NEXT: movi d1, #0000000000000000
; CHECK-NEXT: movi d2, #0000000000000000
; CHECK-NEXT: movi d3, #0000000000000000
; CHECK-NEXT: movi d4, #0000000000000000
; CHECK-NEXT: movi d5, #0000000000000000
; CHECK-NEXT: ret
  ret %T_IN_BLOCK zeroinitializer
}

@in_block_store = dso_local global %T_IN_BLOCK zeroinitializer, align 8

define void @caller_in_block() {
; CHECK-LABEL: caller_in_block:
; CHECK: bl   return_in_block
; CHECK-NEXT: adrp x8, in_block_store
; CHECK-NEXT: add x8, x8, :lo12:in_block_store
; CHECK-NEXT: stp d0, d1, [x8]
; CHECK-NEXT: stp d2, d3, [x8, #16]
; CHECK-NEXT: stp d4, d5, [x8, #32]
; CHECK-NEXT: ldr x30, [sp], #16
; CHECK-NEXT: ret
  %1 = call %T_IN_BLOCK @return_in_block()
  store %T_IN_BLOCK %1, %T_IN_BLOCK* @in_block_store
  ret void
}

define void @callee_in_block(%T_IN_BLOCK %a) {
; CHECK-LABEL: callee_in_block:
; CHECK:      adrp x8, in_block_store
; CHECK-NEXT: add x8, x8, :lo12:in_block_store
; CHECK-NEXT: stp d4, d5, [x8, #32]
; CHECK-NEXT: stp d2, d3, [x8, #16]
; CHECK-NEXT: stp d0, d1, [x8]
; CHECK-NEXT: ret
  store %T_IN_BLOCK %a, %T_IN_BLOCK* @in_block_store
  ret void
}

define void @argument_in_block() {
; CHECK-LABEL: argument_in_block:
; CHECK:      adrp x8, in_block_store
; CHECK-NEXT: add x8, x8, :lo12:in_block_store
; CHECK-NEXT: ldp d4, d5, [x8, #32]
; CHECK-NEXT: ldp d2, d3, [x8, #16]
; CHECK-NEXT: ldp d0, d1, [x8]
; CHECK-NEXT: bl callee_in_block
  %1 = load %T_IN_BLOCK, %T_IN_BLOCK* @in_block_store
  call void @callee_in_block(%T_IN_BLOCK %1)
  ret void
}

%T_IN_MEMORY = type [ 3 x { double, { double, double } } ]

define %T_IN_MEMORY @return_in_memory() {
; CHECK-LABEL: return_in_memory:
; CHECK:       movi v0.2d, #0000000000000000
; CHECK-NEXT:  str xzr, [x8, #64]
; CHECK-NEXT:  stp q0, q0, [x8, #32]
; CHECK-NEXT:  stp q0, q0, [x8]
; CHECK-NEXT:  ret
  ret %T_IN_MEMORY zeroinitializer
}

@in_memory_store = dso_local global %T_IN_MEMORY zeroinitializer, align 8

define void @caller_in_memory() {
; CHECK-LABEL: caller_in_memory:
; CHECK:      add     x8, sp, #8
; CHECK-NEXT: bl      return_in_memory
; CHECK-NEXT: ldr     d0, [sp, #72]
; CHECK-NEXT: ldur    q1, [sp, #24]
; CHECK-NEXT: ldur    q2, [sp, #8]
; CHECK-NEXT: ldur    q3, [sp, #56]
; CHECK-NEXT: ldur    q4, [sp, #40]
; CHECK-NEXT: ldr     x30, [sp, #80]
; CHECK-NEXT: adrp    x8, in_memory_store
; CHECK-NEXT: add     x8, x8, :lo12:in_memory_store
; CHECK-NEXT: stp     q2, q1, [x8]
; CHECK-NEXT: stp     q4, q3, [x8, #32]
; CHECK-NEXT: str     d0, [x8, #64]
; CHECK-NEXT: add     sp, sp, #96
; CHECK-NEXT: ret
  %1 = call %T_IN_MEMORY @return_in_memory()
  store %T_IN_MEMORY %1, %T_IN_MEMORY* @in_memory_store
  ret void
}

define void @callee_in_memory(%T_IN_MEMORY %a) {
; CHECK-LABEL: callee_in_memory:
; CHECK:      ldp     q0, q1, [sp, #32]
; CHECK-NEXT: ldr     d2, [sp, #64]
; CHECK-NEXT: ldp     q3, q4, [sp]
; CHECK-NEXT: adrp    x8, in_memory_store
; CHECK-NEXT: add     x8, x8, :lo12:in_memory_store
; CHECK-NEXT: str     d2, [x8, #64]
; CHECK-NEXT: stp     q0, q1, [x8, #32]
; CHECK-NEXT: stp     q3, q4, [x8]
; CHECK-NEXT: ret
  store %T_IN_MEMORY %a, %T_IN_MEMORY* @in_memory_store
  ret void
}

define void @argument_in_memory() {
; CHECK-LABEL: argument_in_memory:
; CHECK:      adrp    x8, in_memory_store
; CHECK-NEXT: add     x8, x8, :lo12:in_memory_store
; CHECK-NEXT: ldp     q0, q1, [x8]
; CHECK-NEXT: ldp     q2, q3, [x8, #32]
; CHECK-NEXT: ldr     d4, [x8, #64]
; CHECK-NEXT: stp     q0, q1, [sp]
; CHECK-NEXT: stp     q2, q3, [sp, #32]
; CHECK-NEXT: str     d4, [sp, #64]
; CHECK-NEXT: bl      callee_in_memory
  %1 = load %T_IN_MEMORY, %T_IN_MEMORY* @in_memory_store
  call void @callee_in_memory(%T_IN_MEMORY %1)
  ret void
}

%T_NO_BLOCK = type [ 2 x { double, { i32 } } ]

define %T_NO_BLOCK @return_no_block() {
; CHECK-LABEL: return_no_block:
; CHECK:       movi d0, #0000000000000000
; CHECK-NEXT:  movi d1, #0000000000000000
; CHECK-NEXT:  mov w0, wzr
; CHECK-NEXT:  mov w1, wzr
; CHECK-NEXT:  ret
  ret %T_NO_BLOCK zeroinitializer
}

@no_block_store = dso_local global %T_NO_BLOCK zeroinitializer, align 8

define void @caller_no_block() {
; CHECK-LABEL: caller_no_block:
; CHECK:       bl return_no_block
; CHECK-NEXT:  adrp x8, no_block_store
; CHECK-NEXT:  add x8, x8, :lo12:no_block_store
; CHECK-NEXT:  str d0, [x8]
; CHECK-NEXT:  str w0, [x8, #8]
; CHECK-NEXT:  str d1, [x8, #16]
; CHECK-NEXT:  str w1, [x8, #24]
; CHECK-NEXT:  ldr x30, [sp], #16
; CHECK-NEXT:  ret
  %1 = call %T_NO_BLOCK @return_no_block()
  store %T_NO_BLOCK %1, %T_NO_BLOCK* @no_block_store
  ret void
}

define void @callee_no_block(%T_NO_BLOCK %a) {
; CHECK-LABEL: callee_no_block:
; CHECK:       adrp x8, no_block_store
; CHECK-NEXT:  add x8, x8, :lo12:no_block_store
; CHECK-NEXT:  str w1, [x8, #24]
; CHECK-NEXT:  str d1, [x8, #16]
; CHECK-NEXT:  str w0, [x8, #8]
; CHECK-NEXT:  str d0, [x8]
; CHECK-NEXT:  ret
  store %T_NO_BLOCK %a, %T_NO_BLOCK* @no_block_store
  ret void
}

define void @argument_no_block() {
; CHECK-LABEL: argument_no_block:
; CHECK:       adrp x8, no_block_store
; CHECK-NEXT:  add x8, x8, :lo12:no_block_store
; CHECK-NEXT:  ldr w1, [x8, #24]
; CHECK-NEXT:  ldr d1, [x8, #16]
; CHECK-NEXT:  ldr w0, [x8, #8]
; CHECK-NEXT:  ldr d0, [x8]
; CHECK-NEXT:  bl callee_no_block
; CHECK-NEXT:  ldr x30, [sp], #16
; CHECK-NEXT:  ret
  %1 = load %T_NO_BLOCK, %T_NO_BLOCK* @no_block_store
  call void @callee_no_block(%T_NO_BLOCK %1)
  ret void
}
