; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t
; RUN:  FileCheck %s -check-prefix=X86 < %t
; RUN:  FileCheck %s -check-prefix=X86-GOT-EQUIV < %t
; RUN:  FileCheck %s -check-prefix=X86-NOGOT-EQUIV < %t

; GOT equivalent globals references can be replaced by the GOT entry of the
; final symbol instead.

%struct.data = type { i32, %struct.anon }
%struct.anon = type { i32, i32 }

; Check that these got equivalent symbols are never emitted.

; X86-GOT-EQUIV-NOT: L_localgotequiv
; X86-GOT-EQUIV-NOT: l_extgotequiv
@localfoo = global i32 42
@localgotequiv = private unnamed_addr constant i32* @localfoo

@extfoo = external global i32
@extgotequiv = private unnamed_addr constant i32* @extfoo

; Don't replace GOT equivalent usage within instructions and emit the GOT
; equivalent since it can't be replaced by the GOT entry. @bargotequiv is
; used by an instruction inside @t0.
;
; X86: l_bargotequiv:
; X86-NEXT:  .quad   _extbar
@extbar = external global i32
@bargotequiv = private unnamed_addr constant i32* @extbar

@table = global [4 x %struct.data] [
  %struct.data { i32 1, %struct.anon { i32 2, i32 3 } },
; Test GOT equivalent usage inside nested constant arrays.

; X86: .long   5
; X86-NOT: .long   _localgotequiv-(_table+20)
; X86-NEXT: .long   _localfoo@GOTPCREL+4
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 trunc (i64 sub (i64 ptrtoint (i32** @localgotequiv to i64),
                        i64 ptrtoint (i32* getelementptr inbounds ([4 x %struct.data], [4 x %struct.data]* @table, i32 0, i64 1, i32 1, i32 1) to i64))
                        to i32)}
  },
; X86: .long   5
; X86-NOT: _extgotequiv-(_table+32)
; X86-NEXT: .long   _extfoo@GOTPCREL+4
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 trunc (i64 sub (i64 ptrtoint (i32** @extgotequiv to i64),
                        i64 ptrtoint (i32* getelementptr inbounds ([4 x %struct.data], [4 x %struct.data]* @table, i32 0, i64 2, i32 1, i32 1) to i64))
                        to i32)}
  },
; Test support for arbitrary constants into the GOTPCREL offset.

; X86: .long   5
; X86-NOT: _extgotequiv-(_table+44)
; X86-NEXT: .long   _extfoo@GOTPCREL+28
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 add (i32 trunc (i64 sub (i64 ptrtoint (i32** @extgotequiv to i64),
                                 i64 ptrtoint (i32* getelementptr inbounds ([4 x %struct.data], [4 x %struct.data]* @table, i32 0, i64 3, i32 1, i32 1) to i64))
                                 to i32), i32 24)}
  }
], align 16

; Test multiple uses of GOT equivalents.

; X86-LABEL: _delta
; X86: .long   _extfoo@GOTPCREL+4
@delta = global i32 trunc (i64 sub (i64 ptrtoint (i32** @extgotequiv to i64),
                                    i64 ptrtoint (i32* @delta to i64))
                           to i32)

; X86-LABEL: _deltaplus:
; X86: .long   _localfoo@GOTPCREL+59
@deltaplus = global i32 add (i32 trunc (i64 sub (i64 ptrtoint (i32** @localgotequiv to i64),
                                        i64 ptrtoint (i32* @deltaplus to i64))
                                        to i32), i32 55)

define i32 @t0(i32 %a) {
  %x = add i32 trunc (i64 sub (i64 ptrtoint (i32** @bargotequiv to i64),
                               i64 ptrtoint (i32 (i32)* @t0 to i64))
                           to i32), %a
  ret i32 %x
}

; Also test direct instruction uses.
define i32** @t1() {
  ret i32** @bargotequiv
}

; Do not crash when a pattern cannot be matched as a GOT equivalent
define void @foo() {
; X86-NOGOT-EQUIV-LABEL: _foo:
; X86-NOGOT-EQUIV: leaq  _b(%rip), %rax
  store i8** @b, i8*** null
  ret void
}
@a = external global i8
@b = internal unnamed_addr constant i8* @a

; X86-NOGOT-EQUIV-LABEL: _c:
; X86-NOGOT-EQUIV:   .quad _b
@c = global i8** @b
