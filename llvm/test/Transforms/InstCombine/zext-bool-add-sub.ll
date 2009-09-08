; RUN: opt < %s -instcombine -S | not grep zext

define i32 @a(i1 %x) {
entry:
        %y = zext i1 %x to i32
        %res = add i32 %y, 1
        ret i32 %res
}

define i32 @b(i1 %x) {
entry:
        %y = zext i1 %x to i32
        %res = add i32 %y, -1
        ret i32 %res
}

define i32 @c(i1 %x) {
entry:
        %y = zext i1 %x to i32
        %res = sub i32 0, %y
        ret i32 %res
}

define i32 @d(i1 %x) {
entry:
        %y = zext i1 %x to i32
        %res = sub i32 3, %y
        ret i32 %res
}
