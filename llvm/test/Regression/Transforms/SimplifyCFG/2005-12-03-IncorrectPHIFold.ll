; Make sure this doesn't turn into an infinite loop

; RUN: llvm-as < %s | opt -simplifycfg -constprop -simplifycfg | llvm-dis | grep bb86

	%struct.anon = type { uint, int, int, int, [1024 x sbyte] }
%_zero_ = external global %struct.anon*		; <%struct.anon**> [#uses=2]
%_one_ = external global %struct.anon*		; <%struct.anon**> [#uses=4]
%str = internal constant [4 x sbyte] c"%d\0A\00"		; <[4 x sbyte]*> [#uses=1]

implementation   ; Functions:


declare int %bc_compare(%struct.anon*, %struct.anon*)

declare void %free_num(%struct.anon**)

declare %struct.anon* %copy_num(%struct.anon*)

declare void %init_num(%struct.anon**)

declare %struct.anon* %new_num(int, int)

declare void %int2num(%struct.anon**, int)

declare void %bc_multiply(%struct.anon*, %struct.anon*, %struct.anon**, int)

declare void %bc_raise(%struct.anon*, %struct.anon*, %struct.anon**, int)

declare int %bc_divide(%struct.anon*, %struct.anon*, %struct.anon**, int)

declare void %bc_add(%struct.anon*, %struct.anon*, %struct.anon**)

declare int %_do_compare(%struct.anon*, %struct.anon*, int, int)

declare int %printf(sbyte*, ...)

int %bc_sqrt(%struct.anon** %num, int %scale) {
entry:
	%guess = alloca %struct.anon*		; <%struct.anon**> [#uses=15]
	%guess1 = alloca %struct.anon*		; <%struct.anon**> [#uses=12]
	%point5 = alloca %struct.anon*		; <%struct.anon**> [#uses=4]
	%tmp = load %struct.anon** %num		; <%struct.anon*> [#uses=1]
	%tmp1 = load %struct.anon** %_zero_		; <%struct.anon*> [#uses=1]
	%tmp = call int %bc_compare( %struct.anon* %tmp, %struct.anon* %tmp1 )		; <int> [#uses=2]
	%tmp = setlt int %tmp, 0		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	ret int 0

cond_false:		; preds = %entry
	%tmp5 = seteq int %tmp, 0		; <bool> [#uses=1]
	br bool %tmp5, label %cond_true6, label %cond_next13

cond_true6:		; preds = %cond_false
	call void %free_num( %struct.anon** %num )
	%tmp8 = load %struct.anon** %_zero_		; <%struct.anon*> [#uses=1]
	%tmp9 = call %struct.anon* %copy_num( %struct.anon* %tmp8 )		; <%struct.anon*> [#uses=1]
	store %struct.anon* %tmp9, %struct.anon** %num
	ret int 1

cond_next13:		; preds = %cond_false
	%tmp15 = load %struct.anon** %num		; <%struct.anon*> [#uses=1]
	%tmp16 = load %struct.anon** %_one_		; <%struct.anon*> [#uses=1]
	%tmp17 = call int %bc_compare( %struct.anon* %tmp15, %struct.anon* %tmp16 )		; <int> [#uses=2]
	%tmp19 = seteq int %tmp17, 0		; <bool> [#uses=1]
	br bool %tmp19, label %cond_true20, label %cond_next27

cond_true20:		; preds = %cond_next13
	call void %free_num( %struct.anon** %num )
	%tmp22 = load %struct.anon** %_one_		; <%struct.anon*> [#uses=1]
	%tmp23 = call %struct.anon* %copy_num( %struct.anon* %tmp22 )		; <%struct.anon*> [#uses=1]
	store %struct.anon* %tmp23, %struct.anon** %num
	ret int 1

cond_next27:		; preds = %cond_next13
	%tmp29 = load %struct.anon** %num		; <%struct.anon*> [#uses=1]
	%tmp30 = getelementptr %struct.anon* %tmp29, int 0, uint 2		; <int*> [#uses=1]
	%tmp31 = load int* %tmp30		; <int> [#uses=2]
	%tmp33 = setge int %tmp31, %scale		; <bool> [#uses=1]
	%max = select bool %tmp33, int %tmp31, int %scale		; <int> [#uses=4]
	%tmp35 = add int %max, 2		; <int> [#uses=2]
	call void %init_num( %struct.anon** %guess )
	call void %init_num( %struct.anon** %guess1 )
	%tmp36 = call %struct.anon* %new_num( int 1, int 1 )		; <%struct.anon*> [#uses=2]
	store %struct.anon* %tmp36, %struct.anon** %point5
	%tmp = getelementptr %struct.anon* %tmp36, int 0, uint 4, int 1		; <sbyte*> [#uses=1]
	store sbyte 5, sbyte* %tmp
	%tmp39 = setlt int %tmp17, 0		; <bool> [#uses=1]
	br bool %tmp39, label %cond_true40, label %cond_false43

cond_true40:		; preds = %cond_next27
	%tmp41 = load %struct.anon** %_one_		; <%struct.anon*> [#uses=1]
	%tmp42 = call %struct.anon* %copy_num( %struct.anon* %tmp41 )		; <%struct.anon*> [#uses=1]
	store %struct.anon* %tmp42, %struct.anon** %guess
	br label %bb80.outer

cond_false43:		; preds = %cond_next27
	call void %int2num( %struct.anon** %guess, int 10 )
	%tmp45 = load %struct.anon** %num		; <%struct.anon*> [#uses=1]
	%tmp46 = getelementptr %struct.anon* %tmp45, int 0, uint 1		; <int*> [#uses=1]
	%tmp47 = load int* %tmp46		; <int> [#uses=1]
	call void %int2num( %struct.anon** %guess1, int %tmp47 )
	%tmp48 = load %struct.anon** %guess1		; <%struct.anon*> [#uses=1]
	%tmp49 = load %struct.anon** %point5		; <%struct.anon*> [#uses=1]
	call void %bc_multiply( %struct.anon* %tmp48, %struct.anon* %tmp49, %struct.anon** %guess1, int %max )
	%tmp51 = load %struct.anon** %guess1		; <%struct.anon*> [#uses=1]
	%tmp52 = getelementptr %struct.anon* %tmp51, int 0, uint 2		; <int*> [#uses=1]
	store int 0, int* %tmp52
	%tmp53 = load %struct.anon** %guess		; <%struct.anon*> [#uses=1]
	%tmp54 = load %struct.anon** %guess1		; <%struct.anon*> [#uses=1]
	call void %bc_raise( %struct.anon* %tmp53, %struct.anon* %tmp54, %struct.anon** %guess, int %max )
	br label %bb80.outer

bb80.outer:		; preds = %cond_true77, %cond_next56
	%done.1.ph = phi int [ 1, %cond_true83 ], [0, %cond_true40], [0, %cond_false43]		; <int> [#uses=1]
	br label %bb80

bb80:		; preds = %bb80.outer, %cond_true83
	%tmp82 = seteq int %done.1.ph, 0		; <bool> [#uses=1]
	br bool %tmp82, label %cond_true83, label %bb86

cond_true83:		; preds = %bb80
	%tmp71 = call int %_do_compare( %struct.anon* null, %struct.anon* null, int 0, int 1 )		; <int> [#uses=2]
	%tmp76 = seteq int %tmp71, 0		; <bool> [#uses=1]
	br bool %tmp76, label %bb80.outer, label %bb80

bb86:		; preds = %bb80
	call void %free_num( %struct.anon** %num )
	%tmp88 = load %struct.anon** %guess		; <%struct.anon*> [#uses=1]
	%tmp89 = load %struct.anon** %_one_		; <%struct.anon*> [#uses=1]
	%tmp92 = call int %bc_divide( %struct.anon* %tmp88, %struct.anon* %tmp89, %struct.anon** %num, int %max )		; <int> [#uses=0]
	call void %free_num( %struct.anon** %guess )
	call void %free_num( %struct.anon** %guess1 )
	call void %free_num( %struct.anon** %point5 )
	ret int 1
}

