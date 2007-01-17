; Test Case for PR1080
; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

%str = internal constant [4 x sbyte] c"-ga\00"		; <[4 x sbyte]*> [#uses=5]

int %main(int %argc, sbyte** %argv) {
entry:
	%tmp65 = getelementptr sbyte** %argv, int 1		; <sbyte**> [#uses=1]
	%tmp66 = load sbyte** %tmp65		; <sbyte*> [#uses=3]
	br bool icmp ne (
          uint sub (
            uint ptrtoint (
              sbyte* getelementptr ([4 x sbyte]* %str, int 0, long 1) 
              to uint), 
            uint ptrtoint ([4 x sbyte]* %str to uint)
          ), 
          uint 1), 
        label %exit_1, label %exit_2

exit_1:
        ret int 0;
exit_2:
        ret int 1;
}
