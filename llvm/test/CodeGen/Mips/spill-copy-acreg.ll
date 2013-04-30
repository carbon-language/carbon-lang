; RUN: llc -march=mipsel -mattr=+dsp < %s

@g1 = common global i64 0, align 8
@g2 = common global i64 0, align 8
@g3 = common global i64 0, align 8

define i64 @test_acreg_copy(i32 %a0, i32 %a1, i32 %a2, i32 %a3) {
entry:
  %0 = load i64* @g1, align 8
  %1 = tail call i64 @llvm.mips.maddu(i64 %0, i32 %a0, i32 %a1)
  %2 = tail call i64 @llvm.mips.maddu(i64 %0, i32 %a2, i32 %a3)
  store i64 %1, i64* @g1, align 8
  store i64 %2, i64* @g2, align 8
  tail call void @foo1()
  store i64 %2, i64* @g3, align 8
  ret i64 %1
}

declare i64 @llvm.mips.maddu(i64, i32, i32)

declare void @foo1()
