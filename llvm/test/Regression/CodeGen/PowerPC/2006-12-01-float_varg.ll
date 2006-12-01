; RUN: llvm-as < %s | llc -march=ppc64 -o - | tail -n +2 > Output/2006-12-01-float_varg.s
; RUN: as -arch ppc64 Output/2006-12-01-float_varg.s -o Output/2006-12-01-float_varg.o
; RUN: gcc -arch ppc64 Output/2006-12-01-float_varg.o -o Output/2006-12-01-float_varg.exe
; RUN: Output/2006-12-01-float_varg.exe | grep "foo 1.230000 1.231210 3.100000 1.310000"

%str = internal constant [17 x sbyte] c"foo %f %f %f %f\0A\00"

implementation

declare int %printf(sbyte*, ...)

int %main(int %argc, sbyte** %argv) {
entry:
	%tmp = tail call int (sbyte*, ...)* %printf( sbyte* getelementptr ([17 x sbyte]* %str, int 0, uint 0), double 1.23, double 1.23121, double 3.1, double 1.31 )
	ret int 0
}
