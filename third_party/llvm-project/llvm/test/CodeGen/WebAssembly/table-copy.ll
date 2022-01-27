; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s | FileCheck %s

%extern = type opaque
%externref = type %extern addrspace(10)* ;; addrspace 10 is nonintegral

@externref_table1 = local_unnamed_addr addrspace(1) global [0 x %externref] undef
@externref_table2 = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare void @llvm.wasm.table.copy(i8 addrspace(1)*, i8 addrspace(1)*, i32, i32, i32) nounwind readonly

define void @table_copy(i32 %dst, i32 %src, i32 %len) {
; CHECK-LABEL: table_copy:
; CHECK-NEXT:  .functype	table_copy (i32, i32, i32) -> ()
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    1
; CHECK-NEXT:  local.get    2
; CHECK-NEXT:  table.copy	externref_table1, externref_table2
; CHECK-NEXT:  end_function
  %tableptr1 = getelementptr [0 x %externref], [0 x %externref] addrspace(1)* @externref_table1, i32 0, i32 0
  %tb1 = bitcast %externref addrspace(1)* %tableptr1 to i8 addrspace(1)*
  %tableptr2 = getelementptr [0 x %externref], [0 x %externref] addrspace(1)* @externref_table2, i32 0, i32 0
  %tb2 = bitcast %externref addrspace(1)* %tableptr2 to i8 addrspace(1)*
  call void @llvm.wasm.table.copy(i8 addrspace(1)* %tb1, i8 addrspace(1)* %tb2, i32 %dst, i32 %src, i32 %len)
  ret void
}

; Testing copying from a table to itself at different offsets
; Copies len items from table1 at src to table1 at src+off
define void @self_table_copy(i32 %src, i32 %off, i32 %len) {
; CHECK-LABEL: self_table_copy:
; CHECK-NEXT:  .functype	self_table_copy (i32, i32, i32) -> ()
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    1
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    2
; CHECK-NEXT:  table.copy	externref_table1, externref_table1
; CHECK-NEXT:  end_function
  %dst = add nsw i32 %src, %off
  %tableptr1 = getelementptr [0 x %externref], [0 x %externref] addrspace(1)* @externref_table1, i32 0, i32 0
  %tb1 = bitcast %externref addrspace(1)* %tableptr1 to i8 addrspace(1)*
  call void @llvm.wasm.table.copy(i8 addrspace(1)* %tb1, i8 addrspace(1)* %tb1, i32 %dst, i32 %src, i32 %len)
  ret void
}
