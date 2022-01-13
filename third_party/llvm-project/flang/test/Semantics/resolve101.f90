! RUN: %python %S/test_errors.py %s %flang_fc1

! Ensure that spurious errors do not arise from FinishSpecificationPart
! checking on a nested specification part.
real, save :: x
interface
  subroutine subr(x)
    real, intent(in) :: x
    ! SAVE attribute checking should not complain at the
    ! end of this specification part about a dummy argument
    ! having the SAVE attribute.
  end subroutine
end interface
end
