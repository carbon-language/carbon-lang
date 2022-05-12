; RUN: llc -march=hexagon < %s  | FileCheck -check-prefix=INITARRAY %s
; RUN: llc -march=hexagon < %s  -use-ctors | FileCheck -check-prefix=CTOR %s

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_P10066.ii, i8* null }]
define internal void @_GLOBAL__sub_I_P10066.ii() {
entry:
  ret void
}

;CTOR: .section	.ctors
;CTOR-NOT:  section	.init_array

;INITARRAY: section	.init_array
;INITARRAY-NOT: .section	.ctors
