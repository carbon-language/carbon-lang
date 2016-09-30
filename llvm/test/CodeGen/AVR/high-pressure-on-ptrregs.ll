; RUN: llc < %s -march=avr | FileCheck %s

; This tests how LLVM handles IR which puts very high
; presure on the PTRREGS class for the register allocator.
;
; This causes a problem because we only have one small register
; class for loading and storing from pointers - 'PTRREGS'.
; One of these registers is also used for the frame pointer, meaning
; that we only ever have two registers available for these operations.
;
; There is an existing bug filed for this issue - PR14879.
;
; The specific failure:
; LLVM ERROR: ran out of registers during register allocation
;
; It has been assembled from the following c code:
;
; struct ss
; {
;   int a;
;   int b;
;   int c;
; };
;
; void loop(struct ss *x, struct ss **y, int z)
; {
;   int i;
;   for (i=0; i<z; ++i)
;   {
;     x->c += y[i]->b;
;   }
; }

%struct.ss = type { i16, i16, i16 }

; CHECK-LABEL: loop
define void @loop(%struct.ss* %x, %struct.ss** %y, i16 %z) {
entry:
  %x.addr = alloca %struct.ss*, align 2
  %y.addr = alloca %struct.ss**, align 2
  %z.addr = alloca i16, align 2
  %i = alloca i16, align 2
  store %struct.ss* %x, %struct.ss** %x.addr, align 2
  store %struct.ss** %y, %struct.ss*** %y.addr, align 2
  store i16 %z, i16* %z.addr, align 2
  store i16 0, i16* %i, align 2
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i16, i16* %i, align 2
  %1 = load i16, i16* %z.addr, align 2
  %cmp = icmp slt i16 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i16, i16* %i, align 2
  %3 = load %struct.ss**, %struct.ss*** %y.addr, align 2
  %arrayidx = getelementptr inbounds %struct.ss*, %struct.ss** %3, i16 %2
  %4 = load %struct.ss*, %struct.ss** %arrayidx, align 2
  %b = getelementptr inbounds %struct.ss, %struct.ss* %4, i32 0, i32 1
  %5 = load i16, i16* %b, align 2
  %6 = load %struct.ss*, %struct.ss** %x.addr, align 2
  %c = getelementptr inbounds %struct.ss, %struct.ss* %6, i32 0, i32 2
  %7 = load i16, i16* %c, align 2
  %add = add nsw i16 %7, %5
  store i16 %add, i16* %c, align 2
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %8 = load i16, i16* %i, align 2
  %inc = add nsw i16 %8, 1
  store i16 %inc, i16* %i, align 2
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

