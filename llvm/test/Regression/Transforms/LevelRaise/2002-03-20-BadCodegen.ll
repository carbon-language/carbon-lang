; This testcase tests whether the raise pass generates bad code for a 
; getelementptr instruction... with a bad level raise pass, this code
; will segfault on execution.
;
; RUN: llvm-as < %s | opt -raise  |lli -abort-on-exception  
 
 %Village = type { [4 x \3 *], \2 *, { \2 *, { int, int, int, \5 * } *, \2 * }, { int, int, int, { \2 *, { int, int, int, \6 * } *, \2 * }, { \2 *, { int, int,
  int, \6 * } *, \2 * }, { \2 *, { int, int, int, \6 * } *, \2 * }, { \2 *, { int, int, int, \6 * } *, \2 * } }, int, int }



implementation

void "foo"(%Village *%V)
begin
	ret void
end

int %main(int %argc, sbyte **%argv) {
; <label>:0         ;[#uses=0]
  %fval = alloca %Village *, uint 4   ; <%Village * *> [#uses=1]
  %reg115 = malloc sbyte, uint 184    ; <sbyte *> [#uses=2]
  br label %bb4

bb4:          ;[#uses=2]
  %reg130 = shl int %argc, ubyte 2   ; <int> [#uses=1]
  %reg131 = add int %reg130, 3    ; <int> [#uses=1]
  %reg132 = add int %reg131, 1    ; <int> [#uses=1]
  %cast323 = cast sbyte * %reg115 to %Village *   ; <%Village *> [#uses=1]
  call void %foo(%Village * %cast323)    ; <%Village *> [#uses=0]
  br label %bb6

bb6:          ;[#uses=3]
  %reg176 = phi int [ %reg177, %bb6 ], [ 0, %bb4 ]    ; <int> [#uses=2]
  %cast370 = cast int %reg176 to int    ; <int> [#uses=1]
  %cast366 = cast int %reg176 to uint   ; <uint> [#uses=1]
  %reg159 = shl uint %cast366, ubyte 3    ; <uint> [#uses=1]
  %cast161 = cast uint %reg159 to ulong   ; <ulong> [#uses=1]
  %cast367 = cast %Village * * %fval to ulong   ; <sbyte *> [#uses=1]
  %reg169 = add ulong %cast367, %cast161
  %cast368 = cast ulong %reg169 to sbyte * *    ; <sbyte * *> [#uses=1]
  %reg170 = load sbyte * * %cast368   ; <sbyte *> [#uses=1]
  %V = cast sbyte *%reg170 to %Village*
  call void %foo(%Village *%V)
  %reg177 = add int %cast370, 1   ; <int> [#uses=2]
  %cond303 = setle int %reg177, 3   ; <bool> [#uses=1]
  br bool %cond303, label %bb6, label %bb7

bb7:          ;[#uses=1]
  ret int 0
}
