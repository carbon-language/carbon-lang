! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct

program omp
  integer i, j, k

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(3)
  do i = 0, 10
    !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
    if (i .lt. 1) cycle
    do j = 0, 10
      do k  = 0, 10
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(3)
  do i = 0, 10
    do j = 0, 10
      !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
      if (i .lt. 1) cycle
      do k  = 0, 10
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(2)
  do i = 0, 10
    !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
    if (i .lt. 1) cycle
    do j = 0, 10
      do k  = 0, 10
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do


  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(2)
  foo: do i = 0, 10
    !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
    if (i .lt. 1) cycle foo
    do j = 0, 10
      do k  = 0, 10
        print *, i, j, k
      end do
    end do
  end do foo
  !$omp end do


  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do collapse(3)
  do 60 i=2,200,2
    do j=1,10
      !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
      if(i==100) cycle
      do k=1,10
        print *,i
      end do
    end do
  60 continue
  !$omp end do

  !$omp do  collapse(3)
  foo: do i = 0, 10
    foo1: do j = 0, 10
         foo2:  do k  = 0, 10
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             if (k .lt. 1) cycle foo
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do

  !$omp do  collapse(3)
  foo: do i = 0, 10
    foo1: do j = 0, 10
         foo2:  do k  = 0, 10
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             if (k .lt. 1) cycle foo1
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do


  !$omp do  collapse(2)
  foo: do i = 0, 10
    foo1: do j = 0, 10
         foo2:  do k  = 0, 10
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             if (k .lt. 1) cycle foo
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do


  !$omp do  ordered(2)
  foo: do i = 0, 10
    foo1: do j = 0, 10
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             if (k .lt. 1) cycle foo
         foo2:  do k  = 0, 10
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(2) ordered(3)
  foo: do i = 0, 10
    foo1: do j = 0, 10
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             if (k .lt. 1) cycle foo
         foo2:  do k  = 0, 10
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do

end program omp
