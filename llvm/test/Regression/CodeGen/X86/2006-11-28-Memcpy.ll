; RUN: llvm-as < %s | llc -march=x86 &&
; RUN: llvm-as < %s | llc -march=x86 | grep 3721182122 | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -march=x86 | grep 'movl _bytes2' | wc -l | grep 1
; PR1022, 1023

%fmt = constant [4 x sbyte] c"%x\0A\00"
%bytes = constant [4 x sbyte] c"\AA\BB\CC\DD"
%bytes2 = global [4 x sbyte] c"\AA\BB\CC\DD"


int %test1() {
        %y = alloca uint
        %c = cast uint* %y to sbyte*
        %z = getelementptr [4 x sbyte]* %bytes, int 0, int 0
        call void %llvm.memcpy.i32( sbyte* %c, sbyte* %z, uint 4, uint 1 )
        %r = load uint* %y
        %t = cast [4 x sbyte]* %fmt to sbyte*
        %tmp = call int (sbyte*, ...)* %printf( sbyte* %t, uint %r )
        ret int 0
}

void %test2() {
        %y = alloca uint
        %c = cast uint* %y to sbyte*
        %z = getelementptr [4 x sbyte]* %bytes2, int 0, int 0
        call void %llvm.memcpy.i32( sbyte* %c, sbyte* %z, uint 4, uint 1 )
        %r = load uint* %y
        %t = cast [4 x sbyte]* %fmt to sbyte*
        %tmp = call int (sbyte*, ...)* %printf( sbyte* %t, uint %r )
        ret void
}

declare void %llvm.memcpy.i32(sbyte*, sbyte*, uint, uint)
declare int %printf(sbyte*, ...)
