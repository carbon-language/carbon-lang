; RUN: llvm-as < %s | opt -raise
	
	%Tree = type %struct.tree*
	%struct.tree = type { int, double, double, %Tree, %Tree, %Tree, %Tree }

implementation   ; Functions:

void %reverse(%Tree %t) {
bb0:					;[#uses=0]
	%cast219 = cast %Tree %t to sbyte***		; <sbyte***> [#uses=7]
	%cond220 = seteq sbyte*** %cast219, null		; <bool> [#uses=1]
	br bool %cond220, label %bb5, label %bb2

bb2:					;[#uses=3]
	%reg2221 = getelementptr sbyte*** %cast219, long 6		; <sbyte***> [#uses=1]
	%reg108 = load sbyte*** %reg2221		; <sbyte**> [#uses=3]
	%reg2251 = getelementptr sbyte** %reg108, long 5		; <sbyte**> [#uses=1]
	store sbyte* null, sbyte** %reg2251
	%reg2281 = getelementptr sbyte*** %cast219, long 6		; <sbyte***> [#uses=1]
	store sbyte** null, sbyte*** %reg2281
	%reg2311 = getelementptr sbyte*** %cast219, long 5		; <sbyte***> [#uses=1]
	%reg114 = load sbyte*** %reg2311		; <sbyte**> [#uses=2]
	%cond234 = seteq sbyte** %reg114, null		; <bool> [#uses=1]
	br bool %cond234, label %bb4, label %bb3

bb3:					;[#uses=4]
	%reg115 = phi sbyte*** [ %cast117, %bb3 ], [ %cast219, %bb2 ]		; <sbyte***> [#uses=2]
	%reg116 = phi sbyte** [ %cast118, %bb3 ], [ %reg114, %bb2 ]		; <sbyte**> [#uses=4]
	%reg236 = getelementptr sbyte** %reg116, long 5		; <sbyte**> [#uses=1]
	%reg110 = load sbyte** %reg236		; <sbyte*> [#uses=1]
	%reg239 = getelementptr sbyte** %reg116, long 5		; <sbyte**> [#uses=1]
	%cast241 = cast sbyte*** %reg115 to sbyte*		; <sbyte*> [#uses=1]
	store sbyte* %cast241, sbyte** %reg239
	%reg242 = getelementptr sbyte*** %reg115, long 6		; <sbyte***> [#uses=1]
	store sbyte** %reg116, sbyte*** %reg242
	%cast117 = cast sbyte** %reg116 to sbyte***		; <sbyte***> [#uses=1]
	%cast118 = cast sbyte* %reg110 to sbyte**		; <sbyte**> [#uses=2]
	%cond245 = setne sbyte** %cast118, null		; <bool> [#uses=1]
	br bool %cond245, label %bb3, label %bb4

bb4:					;[#uses=2]
	%reg247 = getelementptr sbyte*** %cast219, long 5		; <sbyte***> [#uses=1]
	store sbyte** %reg108, sbyte*** %reg247
	%reg250 = getelementptr sbyte** %reg108, long 6		; <sbyte**> [#uses=2]
	cast sbyte** %reg250 to sbyte****		; <sbyte****>:0 [#uses=0]
	%cast252 = cast sbyte*** %cast219 to sbyte*		; <sbyte*> [#uses=1]
	store sbyte* %cast252, sbyte** %reg250
	br label %bb5

bb5:					;[#uses=2]
	ret void
}

