;; Date:     Jul 29, 2003.
;; From:     test/Programs/MultiSource/Ptrdist-bc
;; Function: ---
;; Global:   %yy_ec = internal constant [256 x sbyte] ...
;;           A subset of this array is used in the test below.
;;
;; Error:    Character '\07' was being emitted as '\a', at yy_ec[38].
;;	     When loaded, this returned the value 97 ('a'), instead of 7.
;; 
;; Incorrect LLC Output for the array yy_ec was:
;; yy_ec_1094:
;; 	.ascii	"\000\001\001\001\001\001\001\001\001\002\003\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\002\004\005\001\001\006\a\001\b\t\n\v\f\r\016\017\020\020\020\020\020\020\020\020\020\020\001\021\022\023\024\001\001\025\025\025\025\025\025\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\026\027\030\031\032\001\033\034\035\036\037 !\"#$%&'()*+,-./$0$1$234\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001"
;;


%yy_ec = internal constant [6 x sbyte] c"\06\07\01\08\01\09"

%.str_3 = internal constant [8 x sbyte] c"[%d] = \00"
%.str_4 = internal constant [4 x sbyte] c"%d\0A\00"

implementation

declare int %printf(sbyte*, ...)

int %main() {
entry:
	br label %loopentry
loopentry:
	%i = phi long [0, %entry], [%inc.i, %loopentry]
	%cptr = getelementptr [6 x sbyte]* %yy_ec, long 0, long %i
	%c = load sbyte* %cptr
	%ignore = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([8 x sbyte]* %.str_3, long 0, long 0), long %i)
	%ignore2 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([4 x sbyte]* %.str_4, long 0, long 0), sbyte %c)
	%inc.i = add long %i, 1
	%done = setle long %inc.i, 5
	br bool %done, label %loopentry, label %exit.1
exit.1:
	ret int 0
};
