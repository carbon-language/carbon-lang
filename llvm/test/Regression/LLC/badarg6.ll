; On this code, llc did not pass the sixth argument (%reg321) to printf.
; It passed the first five in %o0 - %o4, but never initialized %o5.
; Fix in  SparcInstrSelection.cpp: 
; 2030c2030
; -                 if (i < target.getRegInfo().GetNumOfIntArgRegs())
; +                 if (i <= target.getRegInfo().GetNumOfIntArgRegs())
; 

%.LC12 = internal global [44 x sbyte] c"\09\09M = %g, I = %g, V = %g\0A\09\09O = %g, E = %g\0A\0A\00"           ; <[44 x sbyte]*>

implementation;

declare int %printf(sbyte*, ...)

declare double %opaque(double)

int %main(int %argc, sbyte** %argv) {

bb25:
	%b = setle int %argc, 2
	br bool %b, label %bb42, label %bb43

bb42:
	%reg315 = call double (double)* %opaque(double 3.0)
	%reg316 = call double (double)* %opaque(double 3.1)
	%reg317 = call double (double)* %opaque(double 3.2)
	%reg318 = call double (double)* %opaque(double 3.3)
	%reg319 = call double (double)* %opaque(double 3.4)
	br label %bb43

bb43:
        %reg321 = phi double [ 2.000000e-01, %bb25 ], [ %reg315, %bb42 ]        
        %reg322 = phi double [ 6.000000e+00, %bb25 ], [ %reg316, %bb42 ]        
        %reg323 = phi double [ 0xBFF0000000000000, %bb25 ], [ %reg317, %bb42 ]  
        %reg324 = phi double [ 0xBFF0000000000000, %bb25 ], [ %reg318, %bb42 ]  
        %reg325 = phi double [ 1.000000e+00, %bb25 ], [ %reg319, %bb42 ]        

	%reg609 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([44 x sbyte]* %.LC12, long 0, long 0), double %reg325, double %reg324, double %reg323, double %reg322, double %reg321 )

	ret int 0
}
