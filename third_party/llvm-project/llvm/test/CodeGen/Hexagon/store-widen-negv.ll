; RUN: llc -march=hexagon < %s | FileCheck %s
; We shouldn't see a 32-bit expansion of -120, just the uint8 value.
; CHECK: #136
define i32 @foo([4 x i8]* %ptr) {
entry:
  %msb = getelementptr inbounds [4 x i8], [4 x i8]* %ptr, i32 0, i32 3
  %lsb = getelementptr inbounds [4 x i8], [4 x i8]* %ptr, i32 0, i32 2
  store i8 0, i8* %msb
  store i8 -120, i8* %lsb, align 2
  ret i32 0
}
