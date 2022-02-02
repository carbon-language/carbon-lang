; RUN: llc -mtriple mips-unknown-linux-gnu -mattr=+micromips \
; RUN:     -relocation-model=static -O3 -filetype=obj -o - %s | \
; RUN:     llvm-readelf -r - | FileCheck %s --check-prefixes=STATIC
; RUN: llc -mtriple mips-unknown-linux-gnu -mattr=+micromips \
; RUN:     -relocation-model=pic -O3 -filetype=obj -o - %s | \
; RUN:     llvm-readelf -r - | FileCheck %s --check-prefixes=PIC
; RUN: llc -mtriple mips-unknown-linux-gnu -mattr=+micromips \
; RUN:     -relocation-model=static -O3 -filetype=obj -o - %s | \
; RUN:     llvm-objdump -s -j .gcc_except_table - | FileCheck %s --check-prefix=EXCEPT-TABLE-STATIC
; RUN: llc -mtriple mips-unknown-linux-gnu -mattr=+micromips \
; RUN:     -relocation-model=pic -O3 -filetype=obj -o - %s | \
; RUN:     llvm-objdump -s -j .gcc_except_table - | FileCheck %s --check-prefix=EXCEPT-TABLE-PIC

; STATIC-LABEL: Relocation section '.rel.eh_frame'
; STATIC-DAG: R_MIPS_32 00000000 DW.ref.__gxx_personality_v0
; STATIC-DAG: R_MIPS_32 00000000 .text
; STATIC-DAG: R_MIPS_32 00000000 .gcc_except_table

; PIC-LABEL: Relocation section '.rel.eh_frame'
; PIC-DAG: R_MIPS_32   00000000 DW.ref.__gxx_personality_v0
; PIC-DAG: R_MIPS_PC32
; PIC-DAG: R_MIPS_32   00000000 .gcc_except_table

; CHECK-READELF: DW.ref.__gxx_personality_v0
; CHECK-READELF-STATIC-NEXT: R_MIPS_32 00000000 .text
; CHECK-READELF-PIC-NEXT: R_MIPS_PC32
; CHECK-READELF-NEXT: .gcc_except_table

; EXCEPT-TABLE-STATIC: 0000 ff9b1501 0c011500 00150e23 01231e00  ...........#.#..
; EXCEPT-TABLE-STATIC: 0010 00010000 00000000
; EXCEPT-TABLE-PIC:    0000 ff9b1501 0c012d00 002d133f 013f2a00 ......-..-.?.?*.
; EXCEPT-TABLE-PIC:    0010 00010000 00000000                    ........

@_ZTIi = external constant i8*

define dso_local i32 @main() local_unnamed_addr personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %exception.i = tail call i8* @__cxa_allocate_exception(i32 4) nounwind
  %0 = bitcast i8* %exception.i to i32*
  store i32 5, i32* %0, align 16
  invoke void @__cxa_throw(i8* %exception.i, i8* bitcast (i8** @_ZTIi to i8*), i8* null) noreturn
          to label %.noexc unwind label %return

.noexc:
  unreachable

return:
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = tail call i8* @__cxa_begin_catch(i8* %2) nounwind
  tail call void @__cxa_end_catch()
  ret i32 0
}

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

declare i8* @__cxa_allocate_exception(i32) local_unnamed_addr

declare void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr
