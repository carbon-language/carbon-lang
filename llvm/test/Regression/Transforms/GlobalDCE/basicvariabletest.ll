; RUN: as < %s | opt -globaldce | dis | not grep global

%X = uninitialized global int
%Y = internal global int 7

