; RUN: opt < %s -gvn -o /dev/null  -pass-remarks-output=%t -S -pass-remarks=gvn \
; RUN:     2>&1 | FileCheck %s
; RUN: cat %t | FileCheck -check-prefix=YAML %s

; CHECK:      remark: <unknown>:0:0: load of type i32 eliminated{{$}}
; CHECK-NEXT: remark: <unknown>:0:0: load of type i32 eliminated{{$}}
; CHECK-NEXT: remark: <unknown>:0:0: load of type i32 eliminated{{$}}
; CHECK-NOT:  remark:

; YAML:      --- !Passed
; YAML-NEXT: Pass:            gvn
; YAML-NEXT: Name:            LoadElim
; YAML-NEXT: Function:        arg
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'load of type '
; YAML-NEXT:   - Type:            i32
; YAML-NEXT:   - String:          ' eliminated'
; YAML-NEXT: ...
; YAML-NEXT: --- !Passed
; YAML-NEXT: Pass:            gvn
; YAML-NEXT: Name:            LoadElim
; YAML-NEXT: Function:        const
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'load of type '
; YAML-NEXT:   - Type:            i32
; YAML-NEXT:   - String:          ' eliminated'
; YAML-NEXT: ...
; YAML-NEXT: --- !Passed
; YAML-NEXT: Pass:            gvn
; YAML-NEXT: Name:            LoadElim
; YAML-NEXT: Function:        inst
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'load of type '
; YAML-NEXT:   - Type:            i32
; YAML-NEXT:   - String:          ' eliminated'
; YAML-NEXT: ...


define i32 @arg(i32* %p, i32 %i) {
entry:
  store i32 %i, i32* %p
  %load = load i32, i32* %p
  ret i32 %load
}

define i32 @const(i32* %p) {
entry:
  store i32 4, i32* %p
  %load = load i32, i32* %p
  ret i32 %load
}

define i32 @inst(i32* %p) {
entry:
  %load1 = load i32, i32* %p
  %load = load i32, i32* %p
  %add = add i32 %load1, %load
  ret i32 %add
}
