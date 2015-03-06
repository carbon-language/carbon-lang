; RUN: llc -mtriple=arm64-apple-darwin %s -o %t
; RUN:  FileCheck %s -check-prefix=ARM < %t
; RUN:  FileCheck %s -check-prefix=ARM-GOT-EQUIV < %t

; GOT equivalent globals references can be replaced by the GOT entry of the
; final symbol instead.

%struct.data = type { i32, %struct.anon }
%struct.anon = type { i32, i32 }

; Check that these got equivalent symbols are emitted on ARM64. Since ARM64 does
; not support encoding an extra offset with @GOT, we still need to emit the
; equivalents for use by such IR constructs.

; ARM-GOT-EQUIV: {{.*}}extgotequiv:
; ARM-GOT-EQUIV: {{.*}}localgotequiv:
@localfoo = global i32 42
@localgotequiv = private unnamed_addr constant i32* @localfoo

@extfoo = external global i32
@extgotequiv = private unnamed_addr constant i32* @extfoo

; Don't replace GOT equivalent usage within instructions and emit the GOT
; equivalent since it can't be replaced by the GOT entry. @bargotequiv is
; used by an instruction inside @t0.
;
; ARM: {{.*}}bargotequiv:
; ARM-NEXT:  .quad   _extbar
@extbar = external global i32
@bargotequiv = private unnamed_addr constant i32* @extbar

@table = global [4 x %struct.data] [
; ARM-LABEL: _table
  %struct.data { i32 1, %struct.anon { i32 2, i32 3 } },
; Test GOT equivalent usage inside nested constant arrays.

; ARM: .long   5
; ARM-NOT: .long   _localgotequiv-(_table+20)
; ARM-NEXT: Ltmp1:
; ARM-NEXT: .long _localfoo@GOT-Ltmp1
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 trunc (i64 sub (i64 ptrtoint (i32** @localgotequiv to i64),
                        i64 ptrtoint (i32* getelementptr inbounds ([4 x %struct.data]* @table, i32 0, i64 1, i32 1, i32 1) to i64))
                        to i32)}
  },

; ARM: .long   5
; ARM-NOT: _extgotequiv-(_table+32)
; ARM-NEXT: Ltmp2:
; ARM-NEXT: _extfoo@GOT-Ltmp2
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 trunc (i64 sub (i64 ptrtoint (i32** @extgotequiv to i64),
                        i64 ptrtoint (i32* getelementptr inbounds ([4 x %struct.data]* @table, i32 0, i64 2, i32 1, i32 1) to i64))
                        to i32)}
  },
; Test support for arbitrary constants into the GOTPCREL offset, which is
; supported on x86-64 but not on ARM64

; ARM: .long   5
; ARM-NEXT: .long ({{.*}}extgotequiv-(_table+44))+24
  %struct.data { i32 4, %struct.anon { i32 5,
    i32 add (i32 trunc (i64 sub (i64 ptrtoint (i32** @extgotequiv to i64),
                                 i64 ptrtoint (i32* getelementptr inbounds ([4 x %struct.data]* @table, i32 0, i64 3, i32 1, i32 1) to i64))
                                 to i32), i32 24)}
  }
], align 16

; Test multiple uses of GOT equivalents.

; ARM-LABEL: _delta
; ARM: Ltmp3:
; ARM-NEXT:  .long _extfoo@GOT-Ltmp3
@delta = global i32 trunc (i64 sub (i64 ptrtoint (i32** @extgotequiv to i64),
                                    i64 ptrtoint (i32* @delta to i64))
                           to i32)

; ARM-LABEL: _deltaplus:
; ARM: .long  ({{.*}}localgotequiv-_deltaplus)+55
@deltaplus = global i32 add (i32 trunc (i64 sub (i64 ptrtoint (i32** @localgotequiv to i64),
                                        i64 ptrtoint (i32* @deltaplus to i64))
                                        to i32), i32 55)

define i32 @t0(i32 %a) {
  %x = add i32 trunc (i64 sub (i64 ptrtoint (i32** @bargotequiv to i64),
                               i64 ptrtoint (i32 (i32)* @t0 to i64))
                           to i32), %a
  ret i32 %x
}
