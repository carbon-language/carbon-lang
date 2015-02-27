; RUN: llc -mtriple=arm64-apple-ios < %s
@bar = common global i32 0, align 4

; Leaf function which uses all callee-saved registers and allocates >= 256 bytes on the stack
; this will cause processFunctionBeforeCalleeSavedScan() to spill LR as an additional scratch
; register.
;
; This is a crash-only regression test for rdar://15124582.
define i32 @foo(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h) nounwind {
entry:
  %stack = alloca [128 x i32], align 4
  %0 = bitcast [128 x i32]* %stack to i8*
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [128 x i32], [128 x i32]* %stack, i64 0, i64 %idxprom
  store i32 %b, i32* %arrayidx, align 4
  %1 = load volatile i32, i32* @bar, align 4
  %2 = load volatile i32, i32* @bar, align 4
  %3 = load volatile i32, i32* @bar, align 4
  %4 = load volatile i32, i32* @bar, align 4
  %5 = load volatile i32, i32* @bar, align 4
  %6 = load volatile i32, i32* @bar, align 4
  %7 = load volatile i32, i32* @bar, align 4
  %8 = load volatile i32, i32* @bar, align 4
  %9 = load volatile i32, i32* @bar, align 4
  %10 = load volatile i32, i32* @bar, align 4
  %11 = load volatile i32, i32* @bar, align 4
  %12 = load volatile i32, i32* @bar, align 4
  %13 = load volatile i32, i32* @bar, align 4
  %14 = load volatile i32, i32* @bar, align 4
  %15 = load volatile i32, i32* @bar, align 4
  %16 = load volatile i32, i32* @bar, align 4
  %17 = load volatile i32, i32* @bar, align 4
  %18 = load volatile i32, i32* @bar, align 4
  %19 = load volatile i32, i32* @bar, align 4
  %20 = load volatile i32, i32* @bar, align 4
  %idxprom1 = sext i32 %c to i64
  %arrayidx2 = getelementptr inbounds [128 x i32], [128 x i32]* %stack, i64 0, i64 %idxprom1
  %21 = load i32, i32* %arrayidx2, align 4
  %factor = mul i32 %h, -2
  %factor67 = mul i32 %g, -2
  %factor68 = mul i32 %f, -2
  %factor69 = mul i32 %e, -2
  %factor70 = mul i32 %d, -2
  %factor71 = mul i32 %c, -2
  %factor72 = mul i32 %b, -2
  %sum = add i32 %2, %1
  %sum73 = add i32 %sum, %3
  %sum74 = add i32 %sum73, %4
  %sum75 = add i32 %sum74, %5
  %sum76 = add i32 %sum75, %6
  %sum77 = add i32 %sum76, %7
  %sum78 = add i32 %sum77, %8
  %sum79 = add i32 %sum78, %9
  %sum80 = add i32 %sum79, %10
  %sum81 = add i32 %sum80, %11
  %sum82 = add i32 %sum81, %12
  %sum83 = add i32 %sum82, %13
  %sum84 = add i32 %sum83, %14
  %sum85 = add i32 %sum84, %15
  %sum86 = add i32 %sum85, %16
  %sum87 = add i32 %sum86, %17
  %sum88 = add i32 %sum87, %18
  %sum89 = add i32 %sum88, %19
  %sum90 = add i32 %sum89, %20
  %sub15 = sub i32 %21, %sum90
  %sub16 = add i32 %sub15, %factor
  %sub17 = add i32 %sub16, %factor67
  %sub18 = add i32 %sub17, %factor68
  %sub19 = add i32 %sub18, %factor69
  %sub20 = add i32 %sub19, %factor70
  %sub21 = add i32 %sub20, %factor71
  %add = add i32 %sub21, %factor72
  ret i32 %add
}
