; RUN: llvm-as < %s > /dev/null
; RUN: verify-uselistorder %s

define <4 x i32> @foo() {
        ret <4 x i32> zeroinitializer
}

