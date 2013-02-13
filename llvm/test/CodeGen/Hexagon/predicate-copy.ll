; RUN: llc -march=hexagon -mcpu=hexagonv4 -O3 < %s | FileCheck %s

; CHECK: r{{[0-9]+}} = p{{[0-9]+}}
define i1 @foo() {
entry:
  ret i1 false
}

