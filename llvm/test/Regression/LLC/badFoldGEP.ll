;; GetMemInstArgs() folded the two getElementPtr instructions together,
;; producing an illegal getElementPtr.  That's because the type generated
;; by the last index for the first one is a structure field, not an array
;; element, and the second one indexes off that structure field.
;; The code is legal but not type-safe and the two GEPs should not be folded.
;; 
;; This code fragment is from Spec/CINT2000/197.parser/197.parser.bc,
;; file post_process.c, function build_domain().
;; (Modified to replace store with load and return load value.)
;; 

%Domain = type { sbyte*, int, int*, int, int, int*, %Domain* }
%domain_array = uninitialized global [497 x %Domain] 

implementation; Functions:

declare void %opaque([497 x %Domain]*)

int %main(int %argc, sbyte** %argv) {
bb0:					;[#uses=0]
	call void %opaque([497 x %Domain]* %domain_array)
	%cann-indvar-idxcast = cast int %argc to long
        %reg841 = getelementptr [497 x %Domain]* %domain_array, long 0, long %cann-indvar-idxcast, ubyte 3
        %reg846 = getelementptr int* %reg841, long 1
        %reg820 = load int* %reg846
	ret int %reg820
}
