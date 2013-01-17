; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s -O0

; This is a regression test which makes sure that the offset check
; is available for STRiw_indexed instruction. This is required
; by 'Hexagon Expand Predicate Spill Code' pass.

define i32 @f(i32 %a, i32 %b) nounwind {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32* %a.addr, align 4
  %1 = load i32* %b.addr, align 4
  %cmp = icmp sgt i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %2 = load i32* %a.addr, align 4
  %3 = load i32* %b.addr, align 4
  %add = add nsw i32 %2, %3
  store i32 %add, i32* %retval
  br label %return

if.else:
  %4 = load i32* %a.addr, align 4
  %5 = load i32* %b.addr, align 4
  %sub = sub nsw i32 %4, %5
  store i32 %sub, i32* %retval
  br label %return

return:
  %6 = load i32* %retval
  ret i32 %6
}
