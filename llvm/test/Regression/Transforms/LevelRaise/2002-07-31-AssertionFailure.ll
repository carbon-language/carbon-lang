; RUN: llvm-as < %s | opt -raise -raise-start-inst=cast271

	%CON_list = type { %CON_list*, %CON_node* }
	%CON_node = type { %DIS_list*, %DIS_list*, int }
	%DIS_list = type { %DIS_list*, %DIS_node* }
	%DIS_node = type { %CON_list*, %List_o_links*, int }
	%List_o_links = type { int, int, int, %List_o_links* }

implementation   ; Functions:

%CON_node* %build_CON_node(int %reg107) {
        br label %bb5

bb2:                                    ;[#uses=3]
        %reg126 = phi sbyte* [ %reg126, %bb2 ]
        br bool true, label %bb2, label %bb5

bb5:                                    ;[#uses=2]
        %reg125 = phi sbyte* [ %reg126, %bb2], [ null, %0 ]
        %reg263 = malloc sbyte*, uint 3         ; <sbyte**> [#uses=4]
        %reg2641 = getelementptr sbyte** %reg263, long 1                ; <sbyte**> [#uses=1]
        store sbyte* %reg125, sbyte** %reg2641
        store sbyte* %reg125, sbyte** %reg263
        %cast271 = cast sbyte** %reg263 to %CON_node*           ; <%CON_node*> [#uses=1]
        ret %CON_node* %cast271
}
