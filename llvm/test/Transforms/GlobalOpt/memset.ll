; both globals are write only, delete them.

; RUN: llvm-upgrade < %s | llvm-as | opt -globalopt | llvm-dis | \
; RUN:   not grep internal

%G0 = internal global [58 x sbyte] c"asdlfkajsdlfkajsd;lfkajds;lfkjasd;flkajsd;lkfja;sdlkfjasd\00"

%G1 = internal global [4 x int] [ int 1, int 2, int 3, int 4]

implementation   ; Functions:

declare void %llvm.memcpy.i32(sbyte*, sbyte*, uint, uint)
declare void %llvm.memset.i32(sbyte*, ubyte, uint, uint)

void %foo() {
        %Blah = alloca [58 x sbyte]             ; <[58 x sbyte]*> [#uses=2]
        %tmp3 = cast [58 x sbyte]* %Blah to sbyte*
	call void %llvm.memcpy.i32( sbyte* cast ([4 x int]* %G1 to sbyte*), sbyte* %tmp3, uint 16, uint 1)
 	call void %llvm.memset.i32( sbyte* getelementptr ([58 x sbyte]* %G0, int 0, int 0), ubyte 17, uint 58, uint 1)
	ret void
}


