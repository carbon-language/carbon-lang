! RUN: %llvmgcc -S %s
! PR2437
program main
  implicit none
  call build (77)
contains
  subroutine build (order)
    integer :: order, i, j


    call test (1, order, 3,  (/ (i, i = 1, order, 3) /))
    call test (order, 1, -3, (/ (i, i = order, 1, -3) /))

    do j = -10, 10
      call test (order + j, order, 5,  (/ (i, i = order + j, order, 5) /))
      call test (order + j, order, -5, (/ (i, i = order + j, order, -5) /))
    end do

  end subroutine build

  subroutine test (from, to, step, values)
    integer, dimension (:) :: values
    integer :: from, to, step, last, i

    last = 0
    do i = from, to, step
      last = last + 1
      if (values (last) .ne. i) call abort
    end do
    if (size (values, dim = 1) .ne. last) call abort
  end subroutine test
end program main
