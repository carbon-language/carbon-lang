; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr10 | FileCheck -check-prefixes=CHECK-PWR9 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr9 | FileCheck -check-prefixes=CHECK-PWR9 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 | FileCheck -check-prefixes=CHECK-PWR8 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr10 | FileCheck -check-prefixes=CHECK-PWR10 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr9 | FileCheck -check-prefixes=CHECK-PWR9 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr8 | FileCheck -check-prefixes=CHECK-PWR8 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr7 | FileCheck -check-prefixes=CHECK-PWR7 %s

; Exponent is a variable
define void @vpow_var(double* nocapture %z, double* nocapture readonly %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vpow_var
; CHECK-PWR10:       __powd2_P10
; CHECK-PWR9:        __powd2_P9
; CHECK-PWR8:        __powd2_P8
; CHECK-PWR7:        __powd2_P7
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %z, i64 %index
  %next.gep31 = getelementptr double, double* %y, i64 %index
  %next.gep32 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep32 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = bitcast double* %next.gep31 to <2 x double>*
  %wide.load33 = load <2 x double>, <2 x double>* %1, align 8
  %2 = call ninf afn nsz <2 x double> @__powd2(<2 x double> %wide.load, <2 x double> %wide.load33)
  %3 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %2, <2 x double>* %3, align 8
  %index.next = add i64 %index, 2
  %4 = icmp eq i64 %index.next, 1024
  br i1 %4, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25
define void @vpow_const(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vpow_const
; CHECK-PWR10:       __powd2_P10
; CHECK-PWR9:        __powd2_P9
; CHECK-PWR8:        __powd2_P8
; CHECK-PWR7:        __powd2_P7
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call ninf afn nsz <2 x double> @__powd2(<2 x double> %wide.load, <2 x double> <double 7.600000e-01, double 7.600000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25 and they are different 
define void @vpow_noeq_const(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vpow_noeq_const
; CHECK-PWR10:       __powd2_P10
; CHECK-PWR9:        __powd2_P9
; CHECK-PWR8:        __powd2_P8
; CHECK-PWR7:        __powd2_P7
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call ninf afn nsz <2 x double> @__powd2(<2 x double> %wide.load, <2 x double> <double 7.700000e-01, double 7.600000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25 and they are different 
define void @vpow_noeq075_const(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vpow_noeq075_const
; CHECK-PWR10:       __powd2_P10
; CHECK-PWR9:        __powd2_P9
; CHECK-PWR8:        __powd2_P8
; CHECK-PWR7:        __powd2_P7
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call ninf afn nsz <2 x double> @__powd2(<2 x double> %wide.load, <2 x double> <double 7.700000e-01, double 7.500000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25 and they are different 
define void @vpow_noeq025_const(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vpow_noeq025_const
; CHECK-PWR10:       __powd2_P10
; CHECK-PWR9:        __powd2_P9
; CHECK-PWR8:        __powd2_P8
; CHECK-PWR7:        __powd2_P7
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call ninf afn nsz <2 x double> @__powd2(<2 x double> %wide.load, <2 x double> <double 7.700000e-01, double 2.500000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.75
define void @vpow_075(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vpow_075
; CHECK-NOT:         __powd2_P{{[7,8,9,10]}}
; CHECK:             xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call ninf afn <2 x double> @__powd2(<2 x double> %wide.load, <2 x double> <double 7.500000e-01, double 7.500000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.25
define void @vpow_025(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vpow_025
; CHECK-NOT:         __powd2_P{{[7,8,9,10]}}
; CHECK:             xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call ninf afn nsz <2 x double> @__powd2(<2 x double> %wide.load, <2 x double> <double 2.500000e-01, double 2.500000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.75 but no proper fast-math flags
define void @vpow_075_nofast(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vpow_075_nofast
; CHECK-PWR10:       __powd2_P10
; CHECK-PWR9:        __powd2_P9
; CHECK-PWR8:        __powd2_P8
; CHECK-PWR7:        __powd2_P7
; CHECK-NOT:         xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call <2 x double> @__powd2(<2 x double> %wide.load, <2 x double> <double 7.500000e-01, double 7.500000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.25 but no proper fast-math flags
define void @vpow_025_nofast(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vpow_025_nofast
; CHECK-PWR10:       __powd2_P10
; CHECK-PWR9:        __powd2_P9
; CHECK-PWR8:        __powd2_P8
; CHECK-PWR7:        __powd2_P7
; CHECK-NOT:         xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call <2 x double> @__powd2(<2 x double> %wide.load, <2 x double> <double 2.500000e-01, double 2.500000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare <2 x double> @__powd2(<2 x double>, <2 x double>) #1
