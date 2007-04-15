; RUN: llvm-upgrade < %s | llvm-as | opt -std-compile-opts | llvm-dis | \
; RUN:   %prcontext strstr 2 | grep -v declare | grep bb36.outer:
; END.

@str = internal constant [68 x i8] c"Dot. date. datum. 123. Some more doubtful demonstration dummy data.\00"		; <[68 x i8]*> [#uses=1]
@str1 = internal constant [5 x i8] c"ummy\00"		; <[5 x i8]*> [#uses=1]
@str2 = internal constant [6 x i8] c" data\00"		; <[6 x i8]*> [#uses=1]
@str3 = internal constant [3 x i8] c"by\00"		; <[3 x i8]*> [#uses=1]

i32 @stringSearch_Clib(i32 %count) {
entry:
	%count_addr = alloca i32		; <i32*> [#uses=2]
	%retval = alloca i32, align 4		; <i32*> [#uses=2]
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	%i = alloca i32, align 4		; <i32*> [#uses=5]
	%c = alloca i32, align 4		; <i32*> [#uses=9]
	%j = alloca i32, align 4		; <i32*> [#uses=4]
	%p = alloca i8*, align 4		; <i8**> [#uses=6]
	%b = alloca [68 x i8], align 16		; <[68 x i8]*> [#uses=6]
	"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %count, i32* %count_addr
	store i32 0, i32* %c
	%b1 = bitcast [68 x i8]* %b to i8*		; <i8*> [#uses=1]
	%tmp2 = getelementptr [68 x i8]* @str, i32 0, i32 0		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %b1, i8* %tmp2, i32 68, i32 1 )
	store i32 0, i32* %j
	br label %bb41

bb:		; preds = %bb41
	store i32 0, i32* %i
	%tmp3 = load i32* %i		; <i32> [#uses=1]
	store i32 %tmp3, i32* %c
	br label %bb36

bb4:		; preds = %bb36
	%b5 = bitcast [68 x i8]* %b to i8*		; <i8*> [#uses=1]
	%tmp6 = getelementptr [5 x i8]* @str1, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp7 = call i8* @strstr( i8* %b5, i8* %tmp6 )		; <i8*> [#uses=1]
	store i8* %tmp7, i8** %p
	%tmp8 = load i8** %p		; <i8*> [#uses=1]
	%ttmp8 = icmp ne i8* %tmp8, null		; <i1>:0 [#uses=1]
	%ttmp10 = zext i1 %ttmp8 to i8		; <i8>:1 [#uses=1]
	%ttmp7 = icmp ne i8 %ttmp10, 0		; <i1>:2 [#uses=1]
	br i1 %ttmp7, label %cond_true, label %cond_next

cond_true:		; preds = %bb4
	%tmp9 = load i8** %p		; <i8*> [#uses=1]
	%tmp910 = ptrtoint i8* %tmp9 to i32		; <i32> [#uses=1]
	%b11 = bitcast [68 x i8]* %b to i8*		; <i8*> [#uses=1]
	%b1112 = ptrtoint i8* %b11 to i32		; <i32> [#uses=1]
	%tmp13 = sub i32 %tmp910, %b1112		; <i32> [#uses=1]
	%tmp14 = load i32* %c		; <i32> [#uses=1]
	%tmp15 = add i32 %tmp13, %tmp14		; <i32> [#uses=1]
	store i32 %tmp15, i32* %c
	br label %cond_next

cond_next:		; preds = %cond_true, %bb4
	%b16 = bitcast [68 x i8]* %b to i8*		; <i8*> [#uses=1]
	%tmp17 = getelementptr [6 x i8]* @str2, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp18 = call i8* @strstr( i8* %b16, i8* %tmp17 )		; <i8*> [#uses=1]
	store i8* %tmp18, i8** %p
	%tmp19 = load i8** %p		; <i8*> [#uses=1]
	%ttmp6 = icmp ne i8* %tmp19, null		; <i1>:3 [#uses=1]
	%ttmp9 = zext i1 %ttmp6 to i8		; <i8>:4 [#uses=1]
	%ttmp4 = icmp ne i8 %ttmp9, 0		; <i1>:5 [#uses=1]
	br i1 %ttmp4, label %cond_true20, label %cond_next28

cond_true20:		; preds = %cond_next
	%tmp21 = load i8** %p		; <i8*> [#uses=1]
	%tmp2122 = ptrtoint i8* %tmp21 to i32		; <i32> [#uses=1]
	%b23 = bitcast [68 x i8]* %b to i8*		; <i8*> [#uses=1]
	%b2324 = ptrtoint i8* %b23 to i32		; <i32> [#uses=1]
	%tmp25 = sub i32 %tmp2122, %b2324		; <i32> [#uses=1]
	%tmp26 = load i32* %c		; <i32> [#uses=1]
	%tmp27 = add i32 %tmp25, %tmp26		; <i32> [#uses=1]
	store i32 %tmp27, i32* %c
	br label %cond_next28

cond_next28:		; preds = %cond_true20, %cond_next
	%b29 = bitcast [68 x i8]* %b to i8*		; <i8*> [#uses=1]
	%tmp30 = getelementptr [3 x i8]* @str3, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp31 = call i32 @strcspn( i8* %b29, i8* %tmp30 )		; <i32> [#uses=1]
	%tmp32 = load i32* %c		; <i32> [#uses=1]
	%tmp33 = add i32 %tmp31, %tmp32		; <i32> [#uses=1]
	store i32 %tmp33, i32* %c
	%tmp34 = load i32* %i		; <i32> [#uses=1]
	%tmp35 = add i32 %tmp34, 1		; <i32> [#uses=1]
	store i32 %tmp35, i32* %i
	br label %bb36

bb36:		; preds = %cond_next28, %bb
	%tmp37 = load i32* %i		; <i32> [#uses=1]
	%ttmp3= icmp sle i32 %tmp37, 249		; <i1>:6 [#uses=1]
	%ttmp12 = zext i1 %ttmp3 to i8		; <i8>:7 [#uses=1]
	%ttmp1 = icmp ne i8 %ttmp12, 0		; <i1>:8 [#uses=1]
	br i1 %ttmp1, label %bb4, label %bb38

bb38:		; preds = %bb36
	%tmp39 = load i32* %j		; <i32> [#uses=1]
	%tmp40 = add i32 %tmp39, 1		; <i32> [#uses=1]
	store i32 %tmp40, i32* %j
	br label %bb41

bb41:		; preds = %bb38, %entry
	%tmp42 = load i32* %j		; <i32> [#uses=1]
	%tmp43 = load i32* %count_addr		; <i32> [#uses=1]
	%ttmp2 = icmp slt i32 %tmp42, %tmp43		; <i1>:9 [#uses=1]
	%ttmp11 = zext i1 %ttmp2 to i8		; <i8>:10 [#uses=1]
	%ttmp5 = icmp ne i8 %ttmp11, 0		; <i1>:11 [#uses=1]
	br i1 %ttmp5, label %bb, label %bb44

bb44:		; preds = %bb41
	%tmp45 = load i32* %c		; <i32> [#uses=1]
	store i32 %tmp45, i32* %tmp
	%tmp46 = load i32* %tmp		; <i32> [#uses=1]
	store i32 %tmp46, i32* %retval
	br label %return

return:		; preds = %bb44
	%retval47 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval47
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare i8* @strstr(i8*, i8*)

declare i32 @strcspn(i8*, i8*)
