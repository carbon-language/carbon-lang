; RUN: as < %s | opt -funcresolve | dis | not grep foo

; The funcresolve pass was resolving the two foo's together in this test,
; adding a ConstantPointerRef to one of them.  Then because of this
; reference, it wasn't able to delete the dead declaration. :(

declare int %foo(...)
declare int %foo(int)


