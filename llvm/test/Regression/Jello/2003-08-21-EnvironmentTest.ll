;
; Regression Test: EnvironmentTest.ll
;
; Description:
;	This is a regression test that verifies that the JIT passes the
;	environment to the main() function.
;

target endian = little
target pointersize = 32
	%struct..TorRec = type { int, void ()* }

implementation   ; Functions:

declare uint %strlen(sbyte*)

declare void %exit(int)

internal void %__main() {
entry:		; No predecessors!
	ret void
}

int %main(int %argc.1, sbyte** %argv.1, sbyte** %envp.1) {
entry:		; No predecessors!
	call void %__main( )
	%tmp.2 = load sbyte** %envp.1		; <sbyte*> [#uses=2]
	%tmp.3 = call uint %strlen( sbyte* %tmp.2 )		; <uint> [#uses=1]
	%tmp.0 = call int %write( int 1, sbyte* %tmp.2, uint %tmp.3 )		; <int> [#uses=0]
	call void %exit( int 0 )
	ret int 0
}

declare int %write(int, sbyte*, uint)
