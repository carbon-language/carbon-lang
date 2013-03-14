; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s
; Check that the mis-aligned load doesn't cause compiler to assert.

declare i32 @_hi(i64) #1
@temp1 = common global i32 0, align 4

define i32 @CSDRSEARCH_executeSearchManager() #0 {
entry:
  %temp = alloca i32, align 4
  %0 = load i32* @temp1, align 4
  store i32 %0, i32* %temp, align 4
  %1 = bitcast i32* %temp to i64*
  %2 = load i64* %1, align 8
  %call = call i32 @_hi(i64 %2)
  ret i32 %call
}
