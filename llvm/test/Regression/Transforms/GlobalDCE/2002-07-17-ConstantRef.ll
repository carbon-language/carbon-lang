; RUN: llvm-as < %s | opt -globaldce
;
%X = global void() * %func
implementation

internal void %func() {  ; Not dead, can be reachable via X
  ret void
}

void %main() {
	ret void
}
