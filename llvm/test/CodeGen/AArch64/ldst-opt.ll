; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-atomic-cfg-tidy=0 -verify-machineinstrs -o - %s | FileCheck %s

; This file contains tests for the AArch64 load/store optimizer.

%padding = type { i8*, i8*, i8*, i8* }
%s.word = type { i32, i32 }
%s.doubleword = type { i64, i32 }
%s.quadword = type { fp128, i32 }
%s.float = type { float, i32 }
%s.double = type { double, i32 }
%struct.word = type { %padding, %s.word }
%struct.doubleword = type { %padding, %s.doubleword }
%struct.quadword = type { %padding, %s.quadword }
%struct.float = type { %padding, %s.float }
%struct.double = type { %padding, %s.double }

; Check the following transform:
;
; (ldr|str) X, [x0, #32]
;  ...
; add x0, x0, #32
;  ->
; (ldr|str) X, [x0, #32]!
;
; with X being either w1, x1, s0, d0 or q0.

declare void @bar_word(%s.word*, i32)

define void @load-pre-indexed-word(%struct.word* %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-word
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.word* %ptr, i64 0, i32 1, i32 0
  %add = load i32* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.word* %ptr, i64 0, i32 1
  tail call void @bar_word(%s.word* %c, i32 %add)
  ret void
}

define void @store-pre-indexed-word(%struct.word* %ptr, i32 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-word
; CHECK: str w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.word* %ptr, i64 0, i32 1, i32 0
  store i32 %val, i32* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.word* %ptr, i64 0, i32 1
  tail call void @bar_word(%s.word* %c, i32 %val)
  ret void
}

declare void @bar_doubleword(%s.doubleword*, i64)

define void @load-pre-indexed-doubleword(%struct.doubleword* %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-doubleword
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.doubleword* %ptr, i64 0, i32 1, i32 0
  %add = load i64* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.doubleword* %ptr, i64 0, i32 1
  tail call void @bar_doubleword(%s.doubleword* %c, i64 %add)
  ret void
}

define void @store-pre-indexed-doubleword(%struct.doubleword* %ptr, i64 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-doubleword
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.doubleword* %ptr, i64 0, i32 1, i32 0
  store i64 %val, i64* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.doubleword* %ptr, i64 0, i32 1
  tail call void @bar_doubleword(%s.doubleword* %c, i64 %val)
  ret void
}

declare void @bar_quadword(%s.quadword*, fp128)

define void @load-pre-indexed-quadword(%struct.quadword* %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-quadword
; CHECK: ldr q{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.quadword* %ptr, i64 0, i32 1, i32 0
  %add = load fp128* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.quadword* %ptr, i64 0, i32 1
  tail call void @bar_quadword(%s.quadword* %c, fp128 %add)
  ret void
}

define void @store-pre-indexed-quadword(%struct.quadword* %ptr, fp128 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-quadword
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.quadword* %ptr, i64 0, i32 1, i32 0
  store fp128 %val, fp128* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.quadword* %ptr, i64 0, i32 1
  tail call void @bar_quadword(%s.quadword* %c, fp128 %val)
  ret void
}

declare void @bar_float(%s.float*, float)

define void @load-pre-indexed-float(%struct.float* %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-float
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.float* %ptr, i64 0, i32 1, i32 0
  %add = load float* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.float* %ptr, i64 0, i32 1
  tail call void @bar_float(%s.float* %c, float %add)
  ret void
}

define void @store-pre-indexed-float(%struct.float* %ptr, float %val) nounwind {
; CHECK-LABEL: store-pre-indexed-float
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.float* %ptr, i64 0, i32 1, i32 0
  store float %val, float* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.float* %ptr, i64 0, i32 1
  tail call void @bar_float(%s.float* %c, float %val)
  ret void
}

declare void @bar_double(%s.double*, double)

define void @load-pre-indexed-double(%struct.double* %ptr) nounwind {
; CHECK-LABEL: load-pre-indexed-double
; CHECK: ldr d{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.double* %ptr, i64 0, i32 1, i32 0
  %add = load double* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.double* %ptr, i64 0, i32 1
  tail call void @bar_double(%s.double* %c, double %add)
  ret void
}

define void @store-pre-indexed-double(%struct.double* %ptr, double %val) nounwind {
; CHECK-LABEL: store-pre-indexed-double
; CHECK: str d{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.double* %ptr, i64 0, i32 1, i32 0
  store double %val, double* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.double* %ptr, i64 0, i32 1
  tail call void @bar_double(%s.double* %c, double %val)
  ret void
}

; Check the following transform:
;
; add x8, x8, #16
;  ...
; ldr X, [x8]
;  ->
; ldr X, [x8, #16]!
;
; with X being either w0, x0, s0, d0 or q0.

%pre.struct.i32 = type { i32, i32, i32}
%pre.struct.i64 = type { i32, i64, i64}
%pre.struct.i128 = type { i32, <2 x i64>, <2 x i64>}
%pre.struct.float = type { i32, float, float}
%pre.struct.double = type { i32, double, double}

define i32 @load-pre-indexed-word2(%pre.struct.i32** %this, i1 %cond,
                                   %pre.struct.i32* %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-word2
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}, #4]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load %pre.struct.i32** %this
  %gep1 = getelementptr inbounds %pre.struct.i32* %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i32* %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi i32* [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load i32* %retptr
  ret i32 %ret
}

define i64 @load-pre-indexed-doubleword2(%pre.struct.i64** %this, i1 %cond,
                                         %pre.struct.i64* %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-doubleword2
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}, #8]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load %pre.struct.i64** %this
  %gep1 = getelementptr inbounds %pre.struct.i64* %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i64* %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi i64* [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load i64* %retptr
  ret i64 %ret
}

define <2 x i64> @load-pre-indexed-quadword2(%pre.struct.i128** %this, i1 %cond,
                                             %pre.struct.i128* %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-quadword2
; CHECK: ldr q{{[0-9]+}}, [x{{[0-9]+}}, #16]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load %pre.struct.i128** %this
  %gep1 = getelementptr inbounds %pre.struct.i128* %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i128* %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi <2 x i64>* [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load <2 x i64>* %retptr
  ret <2 x i64> %ret
}

define float @load-pre-indexed-float2(%pre.struct.float** %this, i1 %cond,
                                      %pre.struct.float* %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-float2
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+}}, #4]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load %pre.struct.float** %this
  %gep1 = getelementptr inbounds %pre.struct.float* %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.float* %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi float* [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load float* %retptr
  ret float %ret
}

define double @load-pre-indexed-double2(%pre.struct.double** %this, i1 %cond,
                                        %pre.struct.double* %load2) nounwind {
; CHECK-LABEL: load-pre-indexed-double2
; CHECK: ldr d{{[0-9]+}}, [x{{[0-9]+}}, #8]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load %pre.struct.double** %this
  %gep1 = getelementptr inbounds %pre.struct.double* %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.double* %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi double* [ %gep1, %if.then ], [ %gep2, %if.end ]
  %ret = load double* %retptr
  ret double %ret
}

; Check the following transform:
;
; add x8, x8, #16
;  ...
; str X, [x8]
;  ->
; str X, [x8, #16]!
;
; with X being either w0, x0, s0, d0 or q0.

define void @store-pre-indexed-word2(%pre.struct.i32** %this, i1 %cond,
                                     %pre.struct.i32* %load2,
                                     i32 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-word2
; CHECK: str w{{[0-9]+}}, [x{{[0-9]+}}, #4]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load %pre.struct.i32** %this
  %gep1 = getelementptr inbounds %pre.struct.i32* %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i32* %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi i32* [ %gep1, %if.then ], [ %gep2, %if.end ]
  store i32 %val, i32* %retptr
  ret void
}

define void @store-pre-indexed-doubleword2(%pre.struct.i64** %this, i1 %cond,
                                           %pre.struct.i64* %load2,
                                           i64 %val) nounwind {
; CHECK-LABEL: store-pre-indexed-doubleword2
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}, #8]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load %pre.struct.i64** %this
  %gep1 = getelementptr inbounds %pre.struct.i64* %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i64* %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi i64* [ %gep1, %if.then ], [ %gep2, %if.end ]
  store i64 %val, i64* %retptr
  ret void
}

define void @store-pre-indexed-quadword2(%pre.struct.i128** %this, i1 %cond,
                                         %pre.struct.i128* %load2,
                                         <2 x i64> %val) nounwind {
; CHECK-LABEL: store-pre-indexed-quadword2
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}, #16]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load %pre.struct.i128** %this
  %gep1 = getelementptr inbounds %pre.struct.i128* %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.i128* %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi <2 x i64>* [ %gep1, %if.then ], [ %gep2, %if.end ]
  store <2 x i64> %val, <2 x i64>* %retptr
  ret void
}

define void @store-pre-indexed-float2(%pre.struct.float** %this, i1 %cond,
                                      %pre.struct.float* %load2,
                                      float %val) nounwind {
; CHECK-LABEL: store-pre-indexed-float2
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+}}, #4]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load %pre.struct.float** %this
  %gep1 = getelementptr inbounds %pre.struct.float* %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.float* %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi float* [ %gep1, %if.then ], [ %gep2, %if.end ]
  store float %val, float* %retptr
  ret void
}

define void @store-pre-indexed-double2(%pre.struct.double** %this, i1 %cond,
                                      %pre.struct.double* %load2,
                                      double %val) nounwind {
; CHECK-LABEL: store-pre-indexed-double2
; CHECK: str d{{[0-9]+}}, [x{{[0-9]+}}, #8]!
  br i1 %cond, label %if.then, label %if.end
if.then:
  %load1 = load %pre.struct.double** %this
  %gep1 = getelementptr inbounds %pre.struct.double* %load1, i64 0, i32 1
  br label %return
if.end:
  %gep2 = getelementptr inbounds %pre.struct.double* %load2, i64 0, i32 2
  br label %return
return:
  %retptr = phi double* [ %gep1, %if.then ], [ %gep2, %if.end ]
  store double %val, double* %retptr
  ret void
}

; Check the following transform:
;
; ldr X, [x20]
;  ...
; add x20, x20, #32
;  ->
; ldr X, [x20], #32
;
; with X being either w0, x0, s0, d0 or q0.

define void @load-post-indexed-word(i32* %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-word
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}], #16
entry:
  %gep1 = getelementptr i32* %array, i64 2
  br label %body

body:
  %iv2 = phi i32* [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i32* %iv2, i64 -1
  %load = load i32* %gep2
  call void @use-word(i32 %load)
  %load2 = load i32* %iv2
  call void @use-word(i32 %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i32* %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @load-post-indexed-doubleword(i64* %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-doubleword
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}], #32
entry:
  %gep1 = getelementptr i64* %array, i64 2
  br label %body

body:
  %iv2 = phi i64* [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i64* %iv2, i64 -1
  %load = load i64* %gep2
  call void @use-doubleword(i64 %load)
  %load2 = load i64* %iv2
  call void @use-doubleword(i64 %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i64* %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @load-post-indexed-quadword(<2 x i64>* %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-quadword
; CHECK: ldr q{{[0-9]+}}, [x{{[0-9]+}}], #64
entry:
  %gep1 = getelementptr <2 x i64>* %array, i64 2
  br label %body

body:
  %iv2 = phi <2 x i64>* [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr <2 x i64>* %iv2, i64 -1
  %load = load <2 x i64>* %gep2
  call void @use-quadword(<2 x i64> %load)
  %load2 = load <2 x i64>* %iv2
  call void @use-quadword(<2 x i64> %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr <2 x i64>* %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @load-post-indexed-float(float* %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-float
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+}}], #16
entry:
  %gep1 = getelementptr float* %array, i64 2
  br label %body

body:
  %iv2 = phi float* [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr float* %iv2, i64 -1
  %load = load float* %gep2
  call void @use-float(float %load)
  %load2 = load float* %iv2
  call void @use-float(float %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr float* %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @load-post-indexed-double(double* %array, i64 %count) nounwind {
; CHECK-LABEL: load-post-indexed-double
; CHECK: ldr d{{[0-9]+}}, [x{{[0-9]+}}], #32
entry:
  %gep1 = getelementptr double* %array, i64 2
  br label %body

body:
  %iv2 = phi double* [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr double* %iv2, i64 -1
  %load = load double* %gep2
  call void @use-double(double %load)
  %load2 = load double* %iv2
  call void @use-double(double %load2)
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr double* %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

; Check the following transform:
;
; str X, [x20]
;  ...
; add x20, x20, #32
;  ->
; str X, [x20], #32
;
; with X being either w0, x0, s0, d0 or q0.

define void @store-post-indexed-word(i32* %array, i64 %count, i32 %val) nounwind {
; CHECK-LABEL: store-post-indexed-word
; CHECK: str w{{[0-9]+}}, [x{{[0-9]+}}], #16
entry:
  %gep1 = getelementptr i32* %array, i64 2
  br label %body

body:
  %iv2 = phi i32* [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i32* %iv2, i64 -1
  %load = load i32* %gep2
  call void @use-word(i32 %load)
  store i32 %val, i32* %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i32* %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @store-post-indexed-doubleword(i64* %array, i64 %count, i64 %val) nounwind {
; CHECK-LABEL: store-post-indexed-doubleword
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}], #32
entry:
  %gep1 = getelementptr i64* %array, i64 2
  br label %body

body:
  %iv2 = phi i64* [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr i64* %iv2, i64 -1
  %load = load i64* %gep2
  call void @use-doubleword(i64 %load)
  store i64 %val, i64* %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr i64* %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @store-post-indexed-quadword(<2 x i64>* %array, i64 %count, <2 x i64> %val) nounwind {
; CHECK-LABEL: store-post-indexed-quadword
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}], #64
entry:
  %gep1 = getelementptr <2 x i64>* %array, i64 2
  br label %body

body:
  %iv2 = phi <2 x i64>* [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr <2 x i64>* %iv2, i64 -1
  %load = load <2 x i64>* %gep2
  call void @use-quadword(<2 x i64> %load)
  store <2 x i64> %val, <2 x i64>* %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr <2 x i64>* %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @store-post-indexed-float(float* %array, i64 %count, float %val) nounwind {
; CHECK-LABEL: store-post-indexed-float
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+}}], #16
entry:
  %gep1 = getelementptr float* %array, i64 2
  br label %body

body:
  %iv2 = phi float* [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr float* %iv2, i64 -1
  %load = load float* %gep2
  call void @use-float(float %load)
  store float %val, float* %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr float* %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

define void @store-post-indexed-double(double* %array, i64 %count, double %val) nounwind {
; CHECK-LABEL: store-post-indexed-double
; CHECK: str d{{[0-9]+}}, [x{{[0-9]+}}], #32
entry:
  %gep1 = getelementptr double* %array, i64 2
  br label %body

body:
  %iv2 = phi double* [ %gep3, %body ], [ %gep1, %entry ]
  %iv = phi i64 [ %iv.next, %body ], [ %count, %entry ]
  %gep2 = getelementptr double* %iv2, i64 -1
  %load = load double* %gep2
  call void @use-double(double %load)
  store double %val, double* %iv2
  %iv.next = add i64 %iv, -4
  %gep3 = getelementptr double* %iv2, i64 4
  %cond = icmp eq i64 %iv.next, 0
  br i1 %cond, label %exit, label %body

exit:
  ret void
}

declare void @use-word(i32)
declare void @use-doubleword(i64)
declare void @use-quadword(<2 x i64>)
declare void @use-float(float)
declare void @use-double(double)

; Check the following transform:
;
; (ldr|str) X, [x20]
;  ...
; sub x20, x20, #16
;  ->
; (ldr|str) X, [x20], #-16
;
; with X being either w0, x0, s0, d0 or q0.

define void @post-indexed-sub-word(i32* %a, i32* %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-word
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}], #-8
; CHECK: str w{{[0-9]+}}, [x{{[0-9]+}}], #-8
  br label %for.body
for.body:
  %phi1 = phi i32* [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi i32* [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr i32* %phi1, i64 -1
  %load1 = load i32* %gep1
  %gep2 = getelementptr i32* %phi2, i64 -1
  store i32 %load1, i32* %gep2
  %load2 = load i32* %phi1
  store i32 %load2, i32* %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr i32* %phi2, i64 -2
  %gep4 = getelementptr i32* %phi1, i64 -2
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-sub-doubleword(i64* %a, i64* %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-doubleword
; CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}], #-16
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}], #-16
  br label %for.body
for.body:
  %phi1 = phi i64* [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi i64* [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr i64* %phi1, i64 -1
  %load1 = load i64* %gep1
  %gep2 = getelementptr i64* %phi2, i64 -1
  store i64 %load1, i64* %gep2
  %load2 = load i64* %phi1
  store i64 %load2, i64* %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr i64* %phi2, i64 -2
  %gep4 = getelementptr i64* %phi1, i64 -2
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-sub-quadword(<2 x i64>* %a, <2 x i64>* %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-quadword
; CHECK: ldr q{{[0-9]+}}, [x{{[0-9]+}}], #-32
; CHECK: str q{{[0-9]+}}, [x{{[0-9]+}}], #-32
  br label %for.body
for.body:
  %phi1 = phi <2 x i64>* [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi <2 x i64>* [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr <2 x i64>* %phi1, i64 -1
  %load1 = load <2 x i64>* %gep1
  %gep2 = getelementptr <2 x i64>* %phi2, i64 -1
  store <2 x i64> %load1, <2 x i64>* %gep2
  %load2 = load <2 x i64>* %phi1
  store <2 x i64> %load2, <2 x i64>* %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr <2 x i64>* %phi2, i64 -2
  %gep4 = getelementptr <2 x i64>* %phi1, i64 -2
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-sub-float(float* %a, float* %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-float
; CHECK: ldr s{{[0-9]+}}, [x{{[0-9]+}}], #-8
; CHECK: str s{{[0-9]+}}, [x{{[0-9]+}}], #-8
  br label %for.body
for.body:
  %phi1 = phi float* [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi float* [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr float* %phi1, i64 -1
  %load1 = load float* %gep1
  %gep2 = getelementptr float* %phi2, i64 -1
  store float %load1, float* %gep2
  %load2 = load float* %phi1
  store float %load2, float* %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr float* %phi2, i64 -2
  %gep4 = getelementptr float* %phi1, i64 -2
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}

define void @post-indexed-sub-double(double* %a, double* %b, i64 %count) nounwind {
; CHECK-LABEL: post-indexed-sub-double
; CHECK: ldr d{{[0-9]+}}, [x{{[0-9]+}}], #-16
; CHECK: str d{{[0-9]+}}, [x{{[0-9]+}}], #-16
  br label %for.body
for.body:
  %phi1 = phi double* [ %gep4, %for.body ], [ %b, %0 ]
  %phi2 = phi double* [ %gep3, %for.body ], [ %a, %0 ]
  %i = phi i64 [ %dec.i, %for.body], [ %count, %0 ]
  %gep1 = getelementptr double* %phi1, i64 -1
  %load1 = load double* %gep1
  %gep2 = getelementptr double* %phi2, i64 -1
  store double %load1, double* %gep2
  %load2 = load double* %phi1
  store double %load2, double* %phi2
  %dec.i = add nsw i64 %i, -1
  %gep3 = getelementptr double* %phi2, i64 -2
  %gep4 = getelementptr double* %phi1, i64 -2
  %cond = icmp sgt i64 %dec.i, 0
  br i1 %cond, label %for.body, label %end
end:
  ret void
}
