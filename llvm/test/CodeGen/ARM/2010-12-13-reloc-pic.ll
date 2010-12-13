; RUN: llc  %s -mtriple=armv7-linux-gnueabi -arm-use-movt -relocation-model=pic -filetype=obj -o - | \
; RUN:    elf-dump --dump-section-data | FileCheck  -check-prefix=PIC01 %s

;; Fixme: Reduce this test further, or even better, 
;; redo as .s -> .o test once ARM AsmParser is working better

; ModuleID = 'large2.pnacl.bc'
target triple = "armv7-none-linux-gnueabi"

%struct._Bigint = type { %struct._Bigint*, i32, i32, i32, i32, [1 x i32] }
%struct.__FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, %struct._reent*, i8*, i32 (%struct._reent*, i8*, i8*, i32)*, i32 (%struct._reent*, i8*, i8*, i32)*, i32 (%struct._reent*, i8*, i32, i32)*, i32 (%struct._reent*, i8*)*, %struct.__sbuf, i8*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i32, %struct._flock_t, %struct._mbstate_t, i32 }
%struct.__sbuf = type { i8*, i32 }
%struct.__tm = type { i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct._atexit = type { %struct._atexit*, i32, [32 x void ()*], %struct._on_exit_args* }
%struct._flock_t = type { i32, i32, i32, i32, i32 }
%struct._glue = type { %struct._glue*, i32, %struct.__FILE* }
%struct._mbstate_t = type { i32, %union.anon }
%struct._misc_reent = type { i8*, %struct._mbstate_t, %struct._mbstate_t, %struct._mbstate_t, [8 x i8], i32, %struct._mbstate_t, %struct._mbstate_t, %struct._mbstate_t, %struct._mbstate_t, %struct._mbstate_t }
%struct._mprec = type { %struct._Bigint*, i32, %struct._Bigint*, %struct._Bigint** }
%struct._on_exit_args = type { [32 x i8*], [32 x i8*], i32, i32 }
%struct._rand48 = type { [3 x i16], [3 x i16], i16, i64 }
%struct._reent = type { %struct.__FILE*, %struct.__FILE*, %struct.__FILE*, i32, i32, i8*, i32, i32, i8*, %struct._mprec*, void (%struct._reent*)*, i32, i32, i8*, %struct._rand48*, %struct.__tm*, i8*, void (i32)**, %struct._atexit*, %struct._atexit, %struct._glue, %struct.__FILE*, %struct._misc_reent*, i8* }
%union.anon = type { i32 }

@buf = constant [2 x i8] c"x\00", align 4
@_impure_ptr = external thread_local global %struct._reent*
@.str = private constant [22 x i8] c"This should fault...\0A\00", align 4
@.str1 = private constant [40 x i8] c"We're still running. This is not good.\0A\00", align 4

define i32 @main() nounwind {
entry:
  %0 = load %struct._reent** @_impure_ptr, align 4
  %1 = getelementptr inbounds %struct._reent* %0, i32 0, i32 1
  %2 = load %struct.__FILE** %1, align 4
  %3 = bitcast %struct.__FILE* %2 to i8*
  %4 = tail call i32 @fwrite(i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i32 1, i32 21, i8* %3) nounwind
  %5 = load %struct._reent** @_impure_ptr, align 4
  %6 = getelementptr inbounds %struct._reent* %5, i32 0, i32 1
  %7 = load %struct.__FILE** %6, align 4
  %8 = tail call i32 @fflush(%struct.__FILE* %7) nounwind
  store i8 121, i8* getelementptr inbounds ([2 x i8]* @buf, i32 0, i32 0), align 4
  %9 = load %struct._reent** @_impure_ptr, align 4
  %10 = getelementptr inbounds %struct._reent* %9, i32 0, i32 1
  %11 = load %struct.__FILE** %10, align 4
  %12 = bitcast %struct.__FILE* %11 to i8*
  %13 = tail call i32 @fwrite(i8* getelementptr inbounds ([40 x i8]* @.str1, i32 0, i32 0), i32 1, i32 39, i8* %12) nounwind
  ret i32 1
}


; PIC01:             Relocation 0x00000000
; PIC01-NEXT:        'r_offset', 0x0000001c
; PIC01-NEXT:          'r_sym'
; PIC01-NEXT:          'r_type', 0x0000001b


; PIC01:             Relocation 0x00000001
; PIC01-NEXT:      'r_offset', 0x00000038
; PIC01-NEXT:        'r_sym'
; PIC01-NEXT:        'r_type', 0x0000001b

; PIC01:              Relocation 0x00000002
; PIC01-NEXT:      'r_offset', 0x00000044
; PIC01-NEXT:        'r_sym'
; PIC01-NEXT:        'r_type', 0x0000001b

; PIC01:              Relocation 0x00000003
; PIC01-NEXT:      'r_offset', 0x00000070
; PIC01-NEXT:        'r_sym'
; PIC01-NEXT:        'r_type', 0x0000001b

; PIC01:              Relocation 0x00000004
; PIC01-NEXT:      'r_offset', 0x0000007c
; PIC01-NEXT:        'r_sym'
; PIC01-NEXT:        'r_type', 0x00000019


; PIC01:              Relocation 0x00000005
; PIC01-NEXT:      'r_offset', 0x00000080
; PIC01-NEXT:        'r_sym'
; PIC01-NEXT:        'r_type', 0x00000018

; PIC01:              Relocation 0x00000006
; PIC01-NEXT:      'r_offset', 0x00000084
; PIC01-NEXT:        'r_sym'
; PIC01-NEXT:        'r_type', 0x00000068

; PIC01:              Relocation 0x00000007
; PIC01-NEXT:      'r_offset', 0x00000088
; PIC01-NEXT:        'r_sym'
; PIC01-NEXT:        'r_type', 0x0000001a

; PIC01:              Relocation 0x00000008
; PIC01-NEXT:      'r_offset', 0x0000008c
; PIC01-NEXT:        'r_sym'
; PIC01-NEXT:        'r_type', 0x00000018

declare i32 @fwrite(i8* nocapture, i32, i32, i8* nocapture) nounwind

declare i32 @fflush(%struct.__FILE* nocapture) nounwind
