; reduced from DOOM.
%union._XEvent = type { int }
%.X_event_9 = global %union._XEvent zeroinitializer

implementation   ; Functions:
void %I_InitGraphics() {
shortcirc_next.3:		; preds = %no_exit.1
	%tmp.319 = load int* getelementptr ({ int, int }* cast (%union._XEvent* %.X_event_9 to { int, int }*), long 0, ubyte 1)		; <int> [#uses=1]
    ret void
}
