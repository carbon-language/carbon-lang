; RUN: llvm-as < %s | opt -simplify-libcalls
; PR4641

	%struct.__sFILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, i8*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64, %struct.pthread_mutex*, %struct.pthread*, i32, i32, %union.anon }
	%struct.__sbuf = type { i8*, i32, [4 x i8] }
	%struct.pthread = type opaque
	%struct.pthread_mutex = type opaque
	%union.anon = type { i64, [120 x i8] }
@.str13 = external constant [2 x i8]		; <[2 x i8]*> [#uses=1]
@.str14 = external constant [2 x i8]		; <[2 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	br i1 undef, label %if.then.i, label %xmalloc.exit

if.then.i:		; preds = %entry
	unreachable

xmalloc.exit:		; preds = %entry
	br i1 undef, label %if.then.i11, label %xmalloc.exit13

if.then.i11:		; preds = %xmalloc.exit
	unreachable

xmalloc.exit13:		; preds = %xmalloc.exit
	br label %while.cond

while.cond:		; preds = %sw.bb124, %if.end59, %if.end, %while.cond, %xmalloc.exit13
	switch i32 undef, label %sw.default [
		i32 -1, label %for.cond
		i32 0, label %while.cond
		i32 4097, label %sw.bb36
		i32 102, label %sw.bb55
		i32 111, label %sw.bb55
		i32 108, label %sw.bb62
		i32 99, label %sw.bb84
		i32 86, label %sw.bb114
		i32 66, label %sw.bb124
	]

sw.bb36:		; preds = %while.cond
	br i1 undef, label %if.end, label %if.then

if.then:		; preds = %sw.bb36
	unreachable

if.end:		; preds = %sw.bb36
	br label %while.cond

sw.bb55:		; preds = %while.cond, %while.cond
	br i1 undef, label %if.end59, label %if.then58

if.then58:		; preds = %sw.bb55
	br label %if.end59

if.end59:		; preds = %if.then58, %sw.bb55
	br label %while.cond

sw.bb62:		; preds = %while.cond
	unreachable

sw.bb84:		; preds = %while.cond
	unreachable

sw.bb114:		; preds = %while.cond
	unreachable

sw.bb124:		; preds = %while.cond
	br label %while.cond

sw.default:		; preds = %while.cond
	unreachable

for.cond:		; preds = %while.cond
	br i1 undef, label %if.end167, label %if.then8.i

if.then8.i:		; preds = %for.cond
	unreachable

if.end167:		; preds = %for.cond
	br i1 undef, label %if.then174, label %if.end175

if.then174:		; preds = %if.end167
	unreachable

if.end175:		; preds = %if.end167
	br i1 undef, label %if.then179, label %if.end181

if.then179:		; preds = %if.end175
	unreachable

if.end181:		; preds = %if.end175
	br i1 undef, label %if.then.i.i189, label %while.cond.i194

if.then.i.i189:		; preds = %if.end181
	unreachable

while.cond.i194:		; preds = %if.end181
	br i1 undef, label %while.body.i198, label %for.cond.i.i202

while.body.i198:		; preds = %while.cond.i194
	unreachable

for.cond.i.i202:		; preds = %while.cond.i194
	br i1 undef, label %if.end197, label %if.then191

if.then191:		; preds = %for.cond.i.i202
	unreachable

if.end197:		; preds = %for.cond.i.i202
	br label %for.cond.i144

for.cond.i144:		; preds = %for.body.i145, %if.end197
	br i1 undef, label %for.body.i145, label %for.cond24.i

for.body.i145:		; preds = %for.cond.i144
	br label %for.cond.i144

for.cond24.i:		; preds = %for.cond.i144
	br label %for.cond78.i

for.cond78.i:		; preds = %for.body84.i, %for.cond24.i
	br i1 undef, label %for.end94.i, label %for.body84.i

for.body84.i:		; preds = %for.cond78.i
	br label %for.cond78.i

for.end94.i:		; preds = %for.cond78.i
	br i1 undef, label %if.then.i.i139, label %linebuffer_init.exit142

if.then.i.i139:		; preds = %for.end94.i
	br label %linebuffer_init.exit142

linebuffer_init.exit142:		; preds = %if.then.i.i139, %for.end94.i
	br i1 undef, label %if.then.i.i124, label %linebuffer_init.exit129

if.then.i.i124:		; preds = %linebuffer_init.exit142
	unreachable

linebuffer_init.exit129:		; preds = %linebuffer_init.exit142
	br i1 undef, label %if.then.i.i110, label %linebuffer_init.exit113

if.then.i.i110:		; preds = %linebuffer_init.exit129
	unreachable

linebuffer_init.exit113:		; preds = %linebuffer_init.exit129
	br i1 undef, label %if.then.i.i98, label %linebuffer_init.exit

if.then.i.i98:		; preds = %linebuffer_init.exit113
	br label %linebuffer_init.exit

linebuffer_init.exit:		; preds = %if.then.i.i98, %linebuffer_init.exit113
	br i1 undef, label %if.then227, label %while.cond.i50

if.then227:		; preds = %linebuffer_init.exit
	unreachable

while.cond.i50:		; preds = %linebuffer_init.exit
	br i1 undef, label %while.end339, label %while.body334

while.body334:		; preds = %while.cond.i50
	unreachable

while.end339:		; preds = %while.cond.i50
	br i1 undef, label %if.then344, label %if.end346

if.then344:		; preds = %while.end339
	unreachable

if.end346:		; preds = %while.end339
	call void @exit(i32 0) nounwind
	%cond392 = select i1 undef, i8* getelementptr ([2 x i8]* @.str13, i32 0, i32 0), i8* getelementptr ([2 x i8]* @.str14, i32 0, i32 0)		; <i8*> [#uses=1]
	%call393 = call %struct.__sFILE* @fopen(i8* undef, i8* %cond392) nounwind		; <%struct.__sFILE*> [#uses=0]
	unreachable
}

declare %struct.__sFILE* @fopen(i8*, i8*)

declare void @exit(i32)
