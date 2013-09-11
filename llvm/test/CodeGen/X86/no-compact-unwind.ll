; RUN: llc < %s -mtriple x86_64-apple-macosx10.8.0 -mcpu corei7 -filetype=obj -o - \
; RUN:  | llvm-objdump -triple x86_64-apple-macosx10.8.0 -s - \
; RUN:  | FileCheck -check-prefix=CU %s
; RUN: llc < %s -mtriple x86_64-apple-darwin11 -mcpu corei7 \
; RUN:  | llvm-mc -triple x86_64-apple-darwin11 -filetype=obj -o - \
; RUN:  | llvm-objdump -triple x86_64-apple-darwin11 -s - \
; RUN:  | FileCheck -check-prefix=FROM-ASM %s

%"struct.dyld::MappedRanges" = type { [400 x %struct.anon], %"struct.dyld::MappedRanges"* }
%struct.anon = type { %class.ImageLoader*, i64, i64 }
%class.ImageLoader = type { i32 (...)**, i8*, i8*, i32, i64, i64, i32, i32, %"struct.ImageLoader::recursive_lock"*, i16, i16, [4 x i8] }
%"struct.ImageLoader::recursive_lock" = type { i32, i32 }

@G1 = external hidden global %"struct.dyld::MappedRanges", align 8

declare void @OSMemoryBarrier() optsize

; This compact unwind encoding indicates that we could not generate correct
; compact unwind encodings for this function. This then defaults to using the
; DWARF EH frame.

; CU:      Contents of section __compact_unwind:
; CU-NEXT: 0048 00000000 00000000 42000000 00000004
; CU-NEXT: 0058 00000000 00000000 00000000 00000000

; FROM-ASM:      Contents of section __compact_unwind:
; FROM-ASM-NEXT: 0048 00000000 00000000 42000000 00000004
; FROM-ASM-NEXT: 0058 00000000 00000000 00000000 00000000

define void @func(%class.ImageLoader* %image) optsize ssp uwtable {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc10, %entry
  %p.019 = phi %"struct.dyld::MappedRanges"* [ @G1, %entry ], [ %1, %for.inc10 ]
  br label %for.body3

for.body3:                                        ; preds = %for.inc, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.inc ]
  %image4 = getelementptr inbounds %"struct.dyld::MappedRanges"* %p.019, i64 0, i32 0, i64 %indvars.iv, i32 0
  %0 = load %class.ImageLoader** %image4, align 8
  %cmp5 = icmp eq %class.ImageLoader* %0, %image
  br i1 %cmp5, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body3
  tail call void @OSMemoryBarrier() optsize
  store %class.ImageLoader* null, %class.ImageLoader** %image4, align 8
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 400
  br i1 %exitcond, label %for.inc10, label %for.body3

for.inc10:                                        ; preds = %for.inc
  %next = getelementptr inbounds %"struct.dyld::MappedRanges"* %p.019, i64 0, i32 1
  %1 = load %"struct.dyld::MappedRanges"** %next, align 8
  %cmp = icmp eq %"struct.dyld::MappedRanges"* %1, null
  br i1 %cmp, label %for.end11, label %for.cond1.preheader

for.end11:                                        ; preds = %for.inc10
  ret void
}
