! Test the options for generating LLVM byte-code `-emit-llvm-bc` option

!-------------
! RUN COMMANDS
!-------------
! RUN: %flang -emit-llvm -c %s -o - | llvm-dis -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm-bc %s -o - | llvm-dis -o - | FileCheck %s

!----------------
! EXPECTED OUTPUT
!----------------
! CHECK: define void @_QQmain()
! CHECK-NEXT:  ret void
! CHECK-NEXT: }

!------
! INPUT
!------
end program
