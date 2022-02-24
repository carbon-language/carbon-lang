; RUN: llc < %s -mtriple=i386-apple-darwin -asm-verbose | grep "#" | not grep -v "##"

	%struct.AGenericCall = type { %struct.AGenericManager*, %struct.ComponentParameters*, i32* }
	%struct.AGenericManager = type <{ i8 }>
	%struct.ComponentInstanceRecord = type opaque
	%struct.ComponentParameters = type { [1 x i64] }

define i32 @_ZN12AGenericCall10MapIDPtrAtEsRP23ComponentInstanceRecord(%struct.AGenericCall* %this, i16 signext  %param, %struct.ComponentInstanceRecord** %instance) {
entry:
	%tmp4 = icmp slt i16 %param, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	%tmp1415 = shl i16 %param, 3		; <i16> [#uses=1]
	%tmp17 = getelementptr %struct.AGenericCall, %struct.AGenericCall* %this, i32 0, i32 1		; <%struct.ComponentParameters**> [#uses=1]
	%tmp18 = load %struct.ComponentParameters*, %struct.ComponentParameters** %tmp17, align 8		; <%struct.ComponentParameters*> [#uses=1]
	%tmp1920 = bitcast %struct.ComponentParameters* %tmp18 to i8*		; <i8*> [#uses=1]
	%tmp212223 = sext i16 %tmp1415 to i64		; <i64> [#uses=1]
	%tmp24 = getelementptr i8, i8* %tmp1920, i64 %tmp212223		; <i8*> [#uses=1]
	%tmp2425 = bitcast i8* %tmp24 to i64*		; <i64*> [#uses=1]
	%tmp28 = load i64, i64* %tmp2425, align 8		; <i64> [#uses=1]
	%tmp2829 = inttoptr i64 %tmp28 to i32*		; <i32*> [#uses=1]
	%tmp31 = getelementptr %struct.AGenericCall, %struct.AGenericCall* %this, i32 0, i32 2		; <i32**> [#uses=1]
	store i32* %tmp2829, i32** %tmp31, align 8
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
	%tmp4243 = shl i16 %param, 3		; <i16> [#uses=1]
	%tmp46 = getelementptr %struct.AGenericCall, %struct.AGenericCall* %this, i32 0, i32 1		; <%struct.ComponentParameters**> [#uses=1]
	%tmp47 = load %struct.ComponentParameters*, %struct.ComponentParameters** %tmp46, align 8		; <%struct.ComponentParameters*> [#uses=1]
	%tmp4849 = bitcast %struct.ComponentParameters* %tmp47 to i8*		; <i8*> [#uses=1]
	%tmp505152 = sext i16 %tmp4243 to i64		; <i64> [#uses=1]
	%tmp53 = getelementptr i8, i8* %tmp4849, i64 %tmp505152		; <i8*> [#uses=1]
	%tmp5354 = bitcast i8* %tmp53 to i64*		; <i64*> [#uses=1]
	%tmp58 = load i64, i64* %tmp5354, align 8		; <i64> [#uses=1]
	%tmp59 = icmp eq i64 %tmp58, 0		; <i1> [#uses=1]
	br i1 %tmp59, label %UnifiedReturnBlock, label %cond_true63

cond_true63:		; preds = %cond_next
	%tmp65 = getelementptr %struct.AGenericCall, %struct.AGenericCall* %this, i32 0, i32 0		; <%struct.AGenericManager**> [#uses=1]
	%tmp66 = load %struct.AGenericManager*, %struct.AGenericManager** %tmp65, align 8		; <%struct.AGenericManager*> [#uses=1]
	%tmp69 = tail call i32 @_ZN15AGenericManager24DefaultComponentInstanceERP23ComponentInstanceRecord( %struct.AGenericManager* %tmp66, %struct.ComponentInstanceRecord** %instance )		; <i32> [#uses=1]
	ret i32 %tmp69

UnifiedReturnBlock:		; preds = %cond_next
	ret i32 undef
}

declare i32 @_ZN15AGenericManager24DefaultComponentInstanceERP23ComponentInstanceRecord(%struct.AGenericManager*, %struct.ComponentInstanceRecord**)
