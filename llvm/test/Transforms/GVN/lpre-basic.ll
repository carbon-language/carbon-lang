; RUN: opt < %s -gvn -enable-load-pre -S | grep {%PRE = phi}

define i32 @test(i32* %p, i1 %C) {
block1:
	br i1 %C, label %block2, label %block3

block2:
 br label %block4

block3:
  %b = bitcast i32 0 to i32
  store i32 %b, i32* %p
  br label %block4

block4:
  %PRE = load i32* %p
  ret i32 %PRE
}
