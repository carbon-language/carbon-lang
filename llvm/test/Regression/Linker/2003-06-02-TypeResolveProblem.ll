; RUN: echo "%T = type opaque" | as > %t.2.bc
; RUN: as < %s > %t.1.bc
; RUN: link %t.[12].bc

%T = type opaque
%a = constant { %T* }  { %T* null }

