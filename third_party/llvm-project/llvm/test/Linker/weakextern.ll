; RUN: llvm-link %s %s %p/testlink.ll -S | FileCheck %s
; CHECK: kallsyms_names = extern_weak
; CHECK: Inte = global i32
; CHECK: MyVar = external global i32

@kallsyms_names = extern_weak global [0 x i8]
@MyVar = extern_weak global i32
@Inte = extern_weak global i32

define weak [0 x i8]* @use_kallsyms_names() {
  ret [0 x i8]* @kallsyms_names
}
