; RUN: llc < %s | grep larl | count 3

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-ibm-linux"
@bar = common global i64 0, align 8		; <i64*> [#uses=3]

define i64 @foo() nounwind readonly {
entry:
	%tmp = load i64* @bar		; <i64> [#uses=1]
	ret i64 %tmp
}

define i64* @foo2() nounwind readnone {
entry:
	ret i64* @bar
}

define i64* @foo3(i64 %idx) nounwind readnone {
entry:
	%add.ptr.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr2 = getelementptr i64* @bar, i64 %add.ptr.sum		; <i64*> [#uses=1]
	ret i64* %add.ptr2
}
