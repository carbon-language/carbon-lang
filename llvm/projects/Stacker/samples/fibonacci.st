#
# Fibonacci Algorithm in Stacker.
#
: print >d CR;
: fibonacci RROT DUP2 + print 3 PICK -- ; 
: MAIN 0 print 1 print 44 WHILE fibonacci END ;
