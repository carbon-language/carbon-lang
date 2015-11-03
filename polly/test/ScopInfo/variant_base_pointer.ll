; RUN: opt %loadPolly -polly-ignore-aliasing -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-ignore-aliasing -polly-codegen -analyze < %s
;
; CHECK:       Invariant Accesses: {
; CHECK-NEXT:  }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @cli_hex2int() {
entry:
  br label %if.end

if.end:                                           ; preds = %entry
  %call = call i16** @__ctype_b_loc() #0
  %tmp = load i16*, i16** %call, align 8
  %arrayidx = getelementptr inbounds i16, i16* %tmp, i64 0
  %tmp1 = load i16, i16* %arrayidx, align 2
  store i16 3, i16 *%arrayidx, align 2
  br i1 false, label %if.then.2, label %if.end.3

if.then.2:                                        ; preds = %if.end
  br label %cleanup

if.end.3:                                         ; preds = %if.end
  br label %cleanup

cleanup:                                          ; preds = %if.end.3, %if.then.2
  ret void
}

; Function Attrs: nounwind readnone
declare i16** @__ctype_b_loc() #0

attributes #0 = { nounwind readnone }
