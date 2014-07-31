; RUN: llvm-as < %s | llvm-dis | grep bitcast
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

define i1 @main(i32 %X) {
  %res = bitcast i1 true to i1
  ret i1 %res
}
