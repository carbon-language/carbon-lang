; RUN: echo "%T = type opaque" | as > Output/%s.2.bc
; RUN: as < %s > Output/%s.1.bc
; RUN: link Output/%s.[12].bc

%T = type opaque
%a = constant { %T* }  { %T* null }

