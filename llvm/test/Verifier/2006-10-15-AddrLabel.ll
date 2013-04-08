; RUN: not llvm-as < %s > /dev/null 2> %t
; RUN: FileCheck %s --input-file=%t
; CHECK: basic block pointers are invalid

define i32 @main() {
         %foo  = call i8* %llvm.stacksave()
         %foop = bitcast i8* %foo to label*
         %nret = load label* %foop
         br label %nret
}
