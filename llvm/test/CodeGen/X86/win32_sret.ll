; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s -check-prefix=WIN32
; RUN: llc < %s -mtriple=i686-pc-mingw32 | FileCheck %s -check-prefix=MINGW_X86
; RUN: llc < %s -mtriple=i386-pc-linux | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -O0 -mtriple=i686-pc-win32 | FileCheck %s -check-prefix=WIN32
; RUN: llc < %s -O0 -mtriple=i686-pc-mingw32 | FileCheck %s -check-prefix=MINGW_X86
; RUN: llc < %s -O0 -mtriple=i386-pc-linux | FileCheck %s -check-prefix=LINUX

; The SysV ABI used by most Unixes and Mingw on x86 specifies that an sret pointer
; is callee-cleanup. However, in MSVC's cdecl calling convention, sret pointer
; arguments are caller-cleanup like normal arguments.

define void @sret1(i8* sret %x) nounwind {
entry:
; WIN32:      sret1
; WIN32:      movb $42, (%eax)
; WIN32-NOT:  popl %eax
; WIN32:    {{ret$}}

; MINGW_X86:  sret1
; MINGW_X86:  ret $4

; LINUX:      sret1
; LINUX:      ret $4

  store i8 42, i8* %x, align 4
  ret void
}

define void @sret2(i8* sret %x, i8 %y) nounwind {
entry:
; WIN32:      sret2
; WIN32:      movb {{.*}}, (%eax)
; WIN32-NOT:  popl %eax
; WIN32:    {{ret$}}

; MINGW_X86:  sret2
; MINGW_X86:  ret $4

; LINUX:      sret2
; LINUX:      ret $4

  store i8 %y, i8* %x
  ret void
}

define void @sret3(i8* sret %x, i8* %y) nounwind {
entry:
; WIN32:      sret3
; WIN32:      movb $42, (%eax)
; WIN32-NOT:  movb $13, (%eax)
; WIN32-NOT:  popl %eax
; WIN32:    {{ret$}}

; MINGW_X86:  sret3
; MINGW_X86:  ret $4

; LINUX:      sret3
; LINUX:      ret $4

  store i8 42, i8* %x
  store i8 13, i8* %y
  ret void
}

; PR15556
%struct.S4 = type { i32, i32, i32 }

define void @sret4(%struct.S4* noalias sret %agg.result) {
entry:
; WIN32:     sret4
; WIN32:     movl $42, (%eax)
; WIN32-NOT: popl %eax
; WIN32:   {{ret$}}

; MINGW_X86: sret4
; MINGW_X86: ret $4

; LINUX:     sret4
; LINUX:     ret $4

  %x = getelementptr inbounds %struct.S4* %agg.result, i32 0, i32 0
  store i32 42, i32* %x, align 4
  ret void
}
