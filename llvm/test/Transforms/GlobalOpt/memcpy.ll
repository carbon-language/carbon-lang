; RUN: llvm-upgrade < %s | llvm-as | opt -globalopt | llvm-dis | \
; RUN:   grep {G1 = internal constant}

%G1 = internal global [58 x sbyte] c"asdlfkajsdlfkajsd;lfkajds;lfkjasd;flkajsd;lkfja;sdlkfjasd\00"

implementation

declare void %llvm.memcpy.i32(sbyte*, sbyte*, uint, uint)

void %foo() {
        %Blah = alloca [58 x sbyte]             ; <[58 x sbyte]*> [#uses=2]
        %tmp.0 = getelementptr [58 x sbyte]* %Blah, int 0, int 0                ; <sbyte*> [#uses=1]
        call void %llvm.memcpy.i32( sbyte* %tmp.0, sbyte* getelementptr ([58 x sbyte]* %G1, int 0, int 0), uint 58, uint 1 )
	ret void
}


