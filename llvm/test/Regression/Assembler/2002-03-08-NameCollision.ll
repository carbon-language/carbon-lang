; Method arguments were being checked for collisions at the global scope before
; the method object was created by the parser.  Because of this, false collisions
; could occur that would cause the following error message to be produced:
;
;    Redefinition of value named 'X' in the 'int *' type plane!
;
; Fixed by delaying binding of variable names until _after_ the method symtab is
; created.
;

%X = global int 4

declare int "xxx"(int * %X)

implementation

