; Make sure that global variables do not collide if they have the same name,
; but different types.

%X = global int 5
%X = global long 7
