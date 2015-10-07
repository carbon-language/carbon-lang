; RUN: llc < %s -mtriple=i686-pc-linux | FileCheck %s

declare i32 @__gxx_personality_v0(...)
declare void @good(i32 %a, i32 %b, i32 %c, i32 %d)
declare void @large(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f)
declare void @empty()

; We use an invoke, and expect a .cfi_escape GNU_ARGS_SIZE with size 16
; before the invocation
; CHECK-LABEL: test1:
; CHECK: .cfi_escape 0x2e, 0x10
; CHECK-NEXT: pushl   $4
; CHECK-NEXT: pushl   $3
; CHECK-NEXT: pushl   $2
; CHECK-NEXT: pushl   $1
; CHECK-NEXT: call
; CHECK-NEXT: addl $16, %esp
define void @test1() optsize personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue unwind label %cleanup
continue:
  ret void
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; If the function has no handlers, we don't need to generate GNU_ARGS_SIZE,
; even if it has an unwind table.
; CHECK-LABEL: test2:
; CHECK-NOT: .cfi_escape
; CHECK: pushl   $4
; CHECK-NEXT: pushl   $3
; CHECK-NEXT: pushl   $2
; CHECK-NEXT: pushl   $1
; CHECK-NEXT: call
; CHECK-NEXT: addl $16, %esp
define void @test2() optsize personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; If we did not end up using any pushes, no need for GNU_ARGS_SIZE anywhere
; CHECK-LABEL: test3:
; CHECK-NOT: .cfi_escape
; CHECK-NOT: pushl
; CHECK: retl
define void @test3() optsize personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @empty()
          to label %continue unwind label %cleanup
continue:
  ret void
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; Different sized stacks need different GNU_ARGS_SIZEs
; CHECK-LABEL: test4:
; CHECK: .cfi_escape 0x2e, 0x10
; CHECK-NEXT: pushl   $4
; CHECK-NEXT: pushl   $3
; CHECK-NEXT: pushl   $2
; CHECK-NEXT: pushl   $1
; CHECK-NEXT: call
; CHECK-NEXT: addl $16, %esp
; CHECK: .cfi_escape 0x2e, 0x20
; CHECK-NEXT: subl    $8, %esp
; CHECK-NEXT: pushl   $11
; CHECK-NEXT: pushl   $10
; CHECK-NEXT: pushl   $9
; CHECK-NEXT: pushl   $8
; CHECK-NEXT: pushl   $7
; CHECK-NEXT: pushl   $6
; CHECK-NEXT: calll   large
; CHECK-NEXT: addl $32, %esp
define void @test4() optsize personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue1 unwind label %cleanup
continue1:
  invoke void @large(i32 6, i32 7, i32 8, i32 9, i32 10, i32 11)
          to label %continue2 unwind label %cleanup
continue2:
  ret void          
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; If we did use pushes, we need to reset GNU_ARGS_SIZE before a call
; without parameters
; CHECK-LABEL: test5:
; CHECK: .cfi_escape 0x2e, 0x10
; CHECK-NEXT: pushl   $4
; CHECK-NEXT: pushl   $3
; CHECK-NEXT: pushl   $2
; CHECK-NEXT: pushl   $1
; CHECK-NEXT: call
; CHECK-NEXT: addl $16, %esp
; CHECK: .cfi_escape 0x2e, 0x00
; CHECK-NEXT: call
define void @test5() optsize personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue1 unwind label %cleanup
continue1:
  invoke void @empty()
          to label %continue2 unwind label %cleanup
continue2:
  ret void          
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; This is actually inefficient - we don't need to repeat the .cfi_escape twice.
; CHECK-LABEL: test6:
; CHECK: .cfi_escape 0x2e, 0x10
; CHECK: call
; CHECK: .cfi_escape 0x2e, 0x10
; CHECK: call
define void @test6() optsize personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue1 unwind label %cleanup
continue1:
  invoke void @good(i32 5, i32 6, i32 7, i32 8)
          to label %continue2 unwind label %cleanup
continue2:
  ret void          
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}
