! Test the `-emit-llvm` option

! UNSUPPORTED: system-windows
! Windows is currently not supported in flang/lib/Optimizer/CodeGen/Target.cpp

!------------
! RUN COMMAND
!------------
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

!----------------
! EXPECTED OUTPUT
!----------------
! CHECK: ; ModuleID = 'FIRModule'
! CHECK: define void @_QQmain()
! CHECK-NEXT:  ret void
! CHECK-NEXT: }

!------
! INPUT
!------
end program
