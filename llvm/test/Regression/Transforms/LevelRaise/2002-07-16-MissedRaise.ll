; This case fails raise because the store requires that it's argument is of a 
; particular type, but the gep is unable to propogate types backwards through 
; it, because it doesn't know what type to ask it's operand to be.
;
; This could be fixed by making all stores add themselves to a list, and check
; their arguments are consistent AFTER all other values are propogated.

; RUN: as < %s | opt -raise | dis | not grep '= cast' 

        %Tree = type %struct.tree*
        %struct.tree = type { int, double, double, %Tree, %Tree, %Tree, %Tree }

void %reverse(%Tree %t) {
bb0:                                    ;[#uses=0]
        %cast219 = cast %Tree %t to sbyte***            ; <sbyte***> [#uses=2]
        %reg2221 = getelementptr sbyte*** %cast219, long 6              ; <sbyte***> [#uses=1]
        %reg108 = load sbyte*** %reg2221                ; <sbyte**> [#uses=2]
        %reg247 = getelementptr sbyte*** %cast219, long 5               ; <sbyte***> [#uses=1]
        store sbyte** %reg108, sbyte*** %reg247
        ret void
}

