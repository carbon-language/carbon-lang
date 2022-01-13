; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@i = global i32 10, align 4
@.str = private unnamed_addr constant [5 x i8] c"%i \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32, i32* @i, align 4
  %sub = sub nsw i32 0, %0
; 16:	neg	${{[0-9]+}}, ${{[0-9]+}}
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i32 0, i32 0), i32 %sub)
  ret i32 0
}

declare i32 @printf(i8*, ...)
