; RUN: verify-uselistorder < %s -preserve-bc-use-list-order -num-shuffles=5

@a = global [4 x i1] [i1 0, i1 1, i1 0, i1 1]
@b = alias i1* getelementptr ([4 x i1]* @a, i64 0, i64 2)

; Check use-list order of constants used by globals.
@glob1 = global i5 7
@glob2 = global i5 7
@glob3 = global i5 7

; Check use-list order between variables and aliases.
@target = global i3 zeroinitializer
@alias1 = alias i3* @target
@alias2 = alias i3* @target
@alias3 = alias i3* @target
@var1 = global i3* @target
@var2 = global i3* @target
@var3 = global i3* @target

define i64 @f(i64 %f) {
entry:
  %sum = add i64 %f, 0
  ret i64 %sum
}

define i64 @g(i64 %g) {
entry:
  %sum = add i64 %g, 0
  ret i64 %sum
}

define i64 @h(i64 %h) {
entry:
  %sum = add i64 %h, 0
  ret i64 %sum
}

define i64 @i(i64 %i) {
entry:
  %sum = add i64 %i, 1
  ret i64 %sum
}

define i64 @j(i64 %j) {
entry:
  %sum = add i64 %j, 1
  ret i64 %sum
}

define i64 @k(i64 %k) {
entry:
  %sum = add i64 %k, 1
  ret i64 %sum
}

define i64 @l(i64 %l) {
entry:
  %sum = add i64 %l, 1
  ret i64 %sum
}

define i1 @loadb() {
entry:
  %b = load i1* @b
  ret i1 %b
}

define i1 @loada() {
entry:
  %a = load i1* getelementptr ([4 x i1]* @a, i64 0, i64 2)
  ret i1 %a
}

define i32 @f32(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  br label %first

second:
  %eh = mul i32 %e, %h
  %sum = add i32 %eh, %ef
  br label %exit

exit:
  %product = phi i32 [%ef, %first], [%sum, %second]
  ret i32 %product

first:
  %e = add i32 %a, 7
  %f = add i32 %b, 7
  %g = add i32 %c, 8
  %h = add i32 %d, 8
  %ef = mul i32 %e, %f
  %gh = mul i32 %g, %h
  %gotosecond = icmp slt i32 %gh, -9
  br i1 %gotosecond, label %second, label %exit
}
