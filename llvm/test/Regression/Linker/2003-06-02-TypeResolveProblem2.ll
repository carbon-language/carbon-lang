; RUN: echo "%T = type int" | as > %t.1.bc
; RUN: as < %s > %t.2.bc
; RUN: link %t.[12].bc

%T = type opaque

%X = constant {%T*} {%T* null }

