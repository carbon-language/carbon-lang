; RUN: llvm-as < %s | llc -mtriple=i386-pc-linux-gnu -stats |& grep phielim | grep {Number of EH try blocks skipped} | grep 4
; PR3784

	%struct.c38002a__arr___XUB = type { i32, i32 }
	%struct.c38002a__arr_name = type { [0 x i32]*, %struct.c38002a__arr___XUB* }
	%struct.c38002a__rec = type { i32, %struct.c38002a__arr_name }

define void @_ada_c38002a() {
entry:
	%0 = invoke i8* @__gnat_malloc(i32 12)
			to label %invcont unwind label %lpad		; <i8*> [#uses=0]

invcont:		; preds = %entry
	%1 = invoke i8* @__gnat_malloc(i32 20)
			to label %invcont1 unwind label %lpad		; <i8*> [#uses=0]

invcont1:		; preds = %invcont
	%2 = invoke i32 @report__ident_int(i32 2)
			to label %.noexc unwind label %lpad		; <i32> [#uses=0]

.noexc:		; preds = %invcont1
	%3 = invoke i32 @report__ident_int(i32 3)
			to label %.noexc88 unwind label %lpad		; <i32> [#uses=0]

.noexc88:		; preds = %.noexc
	unreachable

lpad:		; preds = %.noexc, %invcont1, %invcont, %entry
	%r.0 = phi %struct.c38002a__rec* [ null, %entry ], [ null, %invcont ], [ null, %invcont1 ], [ null, %.noexc ]		; <%struct.c38002a__rec*> [#uses=1]
	%4 = getelementptr %struct.c38002a__rec* %r.0, i32 0, i32 0		; <i32*> [#uses=1]
	%5 = load i32* %4, align 4		; <i32> [#uses=0]
	ret void
}

declare i32 @report__ident_int(i32)

declare i8* @__gnat_malloc(i32)
