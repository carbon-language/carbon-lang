! RUN: bbc -emit-fir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPfail_image_test
subroutine fail_image_test(fail)
  logical :: fail
! CHECK:  cond_br {{.*}}, ^[[BB1:.*]], ^[[BB2:.*]]
! CHECK: ^[[BB1]]:
  if (fail) then
! CHECK:  {{.*}} = fir.call @_FortranAFailImageStatement() : () -> none
! CHECK-NEXT:  fir.unreachable
   FAIL IMAGE
  end if
! CHECK: ^[[BB2]]:
! CHECK-NEXT:  br ^[[BB3:.*]]
! CHECK-NEXT: ^[[BB3]]
! CEHCK-NEXT:  return
  return
end subroutine
! CHECK-LABEL: func private @_FortranAFailImageStatement() -> none attributes {fir.runtime}
