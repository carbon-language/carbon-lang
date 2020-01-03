subroutine forall1
  real :: a(9)
  !ERROR: 'i' is already declared in this scoping unit
  forall (i=1:8, i=1:9)  a(i) = i
  forall (j=1:8)
    !ERROR: 'j' is already declared in this scoping unit
    forall (j=1:9)
    end forall
  end forall
end

subroutine forall2
  integer, pointer :: a(:)
  integer, target :: b(10,10)
  forall (i=1:10)
    !ERROR: Impure procedure 'f_impure' may not be referenced in a FORALL
    a(f_impure(i):) => b(i,:)
  end forall
contains
  impure integer function f_impure(i)
    f_impure = i
  end
end
