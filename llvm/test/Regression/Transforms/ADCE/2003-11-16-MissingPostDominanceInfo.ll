; RUN: llvm-as < %s | opt -adce | lli
declare void %exit(int)

int %main(int %argc) {
  %C = seteq int %argc, 1
  br bool %C, label %Cond, label %Done

Cond:
  br bool %C, label %Loop, label %Done

Loop:
  call void %exit(int 0)
  br label %Loop

Done:
	ret int 1
}
