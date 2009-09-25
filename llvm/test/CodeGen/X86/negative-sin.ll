; RUN: llc < %s -enable-unsafe-fp-math -march=x86-64 | \
; RUN:   not egrep {addsd|subsd|xor}

declare double @sin(double %f)

define double @foo(double %e)
{
  %f = fsub double 0.0, %e
  %g = call double @sin(double %f) readonly
  %h = fsub double 0.0, %g
  ret double %h
}
