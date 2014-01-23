; RUN: llc -mcpu=invalid -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CPU
; CPU:     'invalid' is not a recognized processor for this target (ignoring processor)
; CPU-NOT: 'invalid' is not a recognized processor for this target (ignoring processor)

; RUN: llc -mattr=+foo,+bar -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=FEATURE
; FEATURE:      'foo' is not a recognized feature for this target (ignoring feature)
; FEATURE-NEXT: 'bar' is not a recognized feature for this target (ignoring feature)
; FEATURE-NOT:  'foo' is not a recognized feature for this target (ignoring feature)
; FEATURE-NOT:  'bar' is not a recognized feature for this target (ignoring feature)

define void @foo() {
entry:
  ret void
}

