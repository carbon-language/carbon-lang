; RUN: echo "%T = type int" | as > Output/%s.1.bc
; RUN: as < %s > Output/%s.2.bc
; RUN: link Output/%s.[12].bc

%T = type opaque

%X = constant {%T*} {%T* null }

