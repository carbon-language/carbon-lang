!RUN: %S/test_errors.sh %s %t %flang -fopenmp
!REQUIRES: shell
! OpenMP Version 4.5
! 2.15.3.3 parallel private Clause
program omp_parallel_private
  integer :: i, j, a(10), b(10), c(10)
  integer :: k = 10
  type my_type
    integer :: array(10)
  end type my_type

  type(my_type) :: my_var

  real :: arr(10)
  integer :: intx = 10

  do i = 1, 10
    arr(i) = 0.0
  end do

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a PRIVATE or SHARED clause or on the ALLOCATE directive.
  !$omp parallel private(arr,intx,my_var%array(1))
  do i = 1, 10
    c(i) = a(i) + b(i) + k
    my_var%array(i) = k+intx
    arr(i) = k
  end do
  !$omp end parallel
end program omp_parallel_private
