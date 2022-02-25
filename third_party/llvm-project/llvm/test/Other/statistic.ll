; RUN: opt < %s -o /dev/null -instsimplify -stats -stats-json 2>&1 | FileCheck %s --check-prefix=JSON
; RUN: opt < %s -o /dev/null -instsimplify -stats -stats-json -info-output-file %t && FileCheck %s < %t --check-prefix=JSON
; RUN: opt < %s -o /dev/null -instsimplify -stats -stats-json -time-passes -enable-new-pm=0 2>&1 | FileCheck %s --check-prefixes=JSON,JSONTIME
; RUN: opt < %s -o /dev/null -instsimplify -stats -stats-json -time-passes -info-output-file %t -enable-new-pm=0 && FileCheck %s < %t --check-prefixes=JSON,JSONTIME
; RUN: opt < %s -o /dev/null -instsimplify -stats 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt < %s -o /dev/null -instsimplify -stats -info-output-file %t && FileCheck %s < %t --check-prefix=DEFAULT
; REQUIRES: asserts

; JSON: {
; JSON-DAG:   "instsimplify.NumSimplified": 1
; JSONTIME-DAG:   "time.pass.instsimplify.wall"
; JSONTIME-DAG:   "time.pass.instsimplify.user"
; JSONTIME-DAG:   "time.pass.instsimplify.sys"
; JSON: }

; DEFAULT: 1 instsimplify - Number of redundant instructions removed

define i32 @foo() {
  %res = add i32 5, 4
  ret i32 %res
}
