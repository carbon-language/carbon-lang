; RUN: opt -time-trace -time-trace-file %t.json \
; RUN:     -disable-output -disable-verify \
; RUN:     -passes='default<O3>' %s
; RUN: cat %t.json \
; RUN:  | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
; RUN:  | FileCheck %s

; CHECK: "name": "Total FunctionToLoopPassAdaptor

define void @foo(i1 %x, i8* %p1, i8* %p2) {
entry:
  store i8 42, i8* %p1
  br i1 %x, label %loop, label %exit

loop:
  %tmp1 = load i8, i8* %p2
  br label %loop

exit:
  ret void
}

declare void @bar()

