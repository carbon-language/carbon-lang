; RUN: llvm-as < %s | opt -inline -disable-output

declare int %External()

implementation

internal int %Callee() {
  %I = call int %External()
  %J = add int %I, %I
  ret int %J
}

int %Caller() {
	%V = invoke int %Callee() to label %Ok except label %Bad
Ok:
  ret int %V
Bad:
  ret int 0
}

