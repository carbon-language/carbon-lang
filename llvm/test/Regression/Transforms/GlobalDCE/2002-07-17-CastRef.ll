; RUN: llvm-as < %s | opt -globaldce
;
implementation

internal void %func() {  ; Not dead, can be reachable via X
  ret void
}

void %main() {
	%X = cast void()* %func to int*
	ret void
}
