; This tests a hack put into place specifically for the C++ libstdc++ library.
; It uses an ugly hack which is cleaned up by the funcresolve pass.
;
; RUN: llvm-as < %s | opt -funcresolve | llvm-dis | grep @X | grep '{'

@X = external global { i32 }
@X = global [ 4 x i8 ] zeroinitializer

define i32* @test() {
  %P = getelementptr {i32}* @X, i64 0, i32 0
  ret i32* %P
}
