; Test resolvable and unresolvable calls through function pointers:
; -- both should be retained in function graphs until resolved or until main
; -- former should get resolved in or before main() and never appear in GG
; -- latter should remain unresolved in main() and copied to GG
; -- globals in GG pointed to by latter should be marked I, but not other nodes
;
; FIXME: KnownPtr should be just S.
; RUN: opt -analyze %s -datastructure-gc -dsgc-check-flags=KnownPtr:SI,UnknownPtr:SI -dsgc-dspass=bu

%Z = internal global int 0
%X = internal global int 0
%M = internal global int 0
%.str_1 = internal constant [9 x sbyte] c"&Z = %p\0A\00"

implementation

declare int %printf(sbyte*, ...)
declare void %exit_dummy(int*)

internal void %makeCalls(void (int*)* %GpKnown.1, void (int*)* %GpUnknown.1,
                         int* %GpKnownPtr, int* %GpUnknownPtr) {
	%tmp.0 = load int* %Z
	%tmp.1.not = setne int %tmp.0, 0
	br bool %tmp.1.not, label %else, label %then

then:
	; pass to exit_dummy: never resolved
	call void %GpUnknown.1( int* %GpUnknownPtr )
	%tmp.61 = load int* %Z
	%inc1 = add int %tmp.61, 1
	store int %inc1, int* %Z
	%tmp.71 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([9 x sbyte]* %.str_1, long 0, long 0), int* %Z )
	ret void

else:
	; pass to knownF: resolved in main
	call void %GpKnown.1( int* %GpKnownPtr )
	%tmp.6 = load int* %Z
	%inc = add int %tmp.6, 1
	store int %inc, int* %Z

	; "known external": resolved here
	%tmp.7 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([9 x sbyte]* %.str_1, long 0, long 0), int* %Z )
	ret void
}

internal void %knownF(int* %Y.1) {
	%tmp.1 = seteq int* %Y.1, null
	br bool %tmp.1, label %then, label %UnifiedExitNode

then:
	call void %knownF( int* %Y.1 )   ; direct call to self: resolved here
	br label %UnifiedExitNode

UnifiedExitNode:
	ret void
}

int %main(int %argc.1) {
        %KnownPtr = alloca int
        %UnknownPtr = alloca int
	store int 1, int* %Z
	call void %makeCalls( void (int*)* %knownF, void (int*)* %exit_dummy,
                              int* %KnownPtr, int* %UnknownPtr )
	ret int 0
}

