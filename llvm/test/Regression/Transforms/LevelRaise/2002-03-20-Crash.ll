; Level raise is making an incorrect transformation, which causes incorrect 
; bytecode to be generated.
;
; RUN: llvm-as < %s | opt -raise | llvm-dis
;

	%Village = type { [4 x \3 *], \2 *, { \2 *, { int, int, int, \5 * } *, \2 * }, { int, int, int, { \2 *, { int, int, int, \6 * } *, \2 * }, { \2 *, { int, int, int, \6 * } *, \2 * }, { \2 *, { int, int, int, \6 * } *, \2 * }, { \2 *, { int, int, int, \6 * } *, \2 * } }, int, int }
implementation

%Village *"get_results"(%Village * %village)
begin
bb0:					;[#uses=1]
	%cast121 = cast int 24 to ulong		; <%Village *> [#uses=1]
	%A = cast %Village* %village to ulong
	%reg123 = add ulong %A, %cast121		; <%Village *> [#uses=1]
	%reg123 = cast ulong %reg123 to %Village*
	%idx = getelementptr %Village * %reg123, long 0, ubyte 0, long 0		; <%Village *> [#uses=1]
	%reg118 = load %Village** %idx
	ret %Village *%reg118
end
