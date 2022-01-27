! RUN: %python %S/test_symbols.py %s %flang_fc1
!DEF: /main MainProgram
program main
 !DEF: /main/x POINTER ObjectEntity REAL(4)
 pointer :: x
 !REF: /main/x
 real x
 !DEF: /main/y EXTERNAL, POINTER (Function) ProcEntity REAL(4)
 pointer :: y
 !REF: /main/y
 procedure (real) :: y
 !DEF: /main/z (Implicit) ObjectEntity REAL(4)
 !REF: /main/y
 z = y()
end program
