; RUN: if as < %s | opt -globaldce | dis | grep global
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi


%X = uninitialized global int
%Y = internal global int 7

