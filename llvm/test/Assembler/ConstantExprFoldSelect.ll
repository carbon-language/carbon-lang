; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s -preserve-bc-use-list-order
; PR18319

define void @function() {
  %c = trunc <4 x i16> select (<4 x i1> <i1 undef, i1 undef, i1 false, i1 true>, <4 x i16> <i16 undef, i16 2, i16 3, i16 4>, <4 x i16> <i16 -1, i16 -2, i16 -3, i16 -4>) to <4 x i8>
; CHECK: <i16 undef, i16 -2, i16 -3, i16 4>
  ret void
}
