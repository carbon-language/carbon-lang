; RUN: llvm-as < %s | opt -instcombine -globaldce | llvm-dis | not grep Array

; Pulling the cast out of the load allows us to eliminate the load, and then 
; the whole array.

%unop = type {int }
%op = type {float}

%Array = internal constant [1 x %op* (%op*)*] [ %op* (%op*)* %foo ]

implementation

%op* %foo(%op* %X) {
  ret %op* %X
}

%unop* %caller(%op* %O) {
   %tmp = load %unop* (%op*)** cast ([1 x %op* (%op*)*]* %Array to %unop* (%op*)**)
   %tmp.2 = call %unop* (%op*)* %tmp(%op* %O)
   ret %unop* %tmp.2
}

