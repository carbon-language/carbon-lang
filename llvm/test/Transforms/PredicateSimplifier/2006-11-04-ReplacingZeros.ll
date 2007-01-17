; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify -disable-output

int %test_wp_B_slice(int %select_method) {
entry:
	br label %bb309

cond_true114:		; preds = %bb309
	%tmp130 = setlt int 0, 128		; <bool> [#uses=1]
	%min = select bool %tmp130, int 0, int 127		; <int> [#uses=2]
	%tmp143 = load int* null		; <int> [#uses=1]
	br bool false, label %bb303, label %bb314

cond_true166:		; preds = %bb303
	ret int 0

cond_false200:		; preds = %bb303
	%tmp205 = sdiv int %min, 2		; <int> [#uses=1]
	%iftmp.380.0.p = select bool false, int 0, int %tmp205		; <int> [#uses=0]
	ret int 0

bb303:		; preds = %cond_true114
	%tmp165 = seteq int %min, 0		; <bool> [#uses=1]
	br bool %tmp165, label %cond_true166, label %cond_false200

bb309:		; preds = %bb19
	br bool false, label %cond_true114, label %bb314

bb314:		; preds = %bb309
	ret int 0
}
