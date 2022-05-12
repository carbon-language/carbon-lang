; RUN: llc < %s -mtriple=x86_64-none-none-gnux32 -mcpu=generic | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-none-none-gnux32 -mcpu=generic -fast-isel | FileCheck %s
;
; Ensures that landingpad instructions in x32 use the right Exception Pointer
; and Exception Selector registers.

declare void @foo()
declare void @bar(i8*, i32) noreturn
declare i32 @__gxx_personality_v0(...)

define void @test1() uwtable personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @foo() to label %done unwind label %lpad
done:
  ret void
lpad:
  %0 = landingpad { i8*, i32 } cleanup
; The Exception Pointer is %eax; the Exception Selector, %edx.
; CHECK: LBB{{[^%]*}} %lpad
; CHECK-DAG: movl %eax, {{.*}}
; CHECK-DAG: movl %edx, {{.*}}
; CHECK: callq bar
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  call void @bar(i8* %1, i32 %2)
  unreachable
}
