; RUN: llc -O2 -march=hexagon < %s | FileCheck %s
; CHECK-DAG: insert(r{{[0-9]*}},#17,#0)
; CHECK-DAG: insert(r{{[0-9]*}},#18,#0)
; CHECK-DAG: insert(r{{[0-9]*}},#22,#0)
; CHECK-DAG: insert(r{{[0-9]*}},#12,#0)

; C source:
; typedef struct {
;   unsigned x1:23;
;   unsigned x2:17;
;   unsigned x3:18;
;   unsigned x4:22;
;   unsigned x5:12;
; } structx_t;
;
; void foo(structx_t *px, int y1, int y2, int y3, int y4, int y5) {
;   px->x1 = y1;
;   px->x2 = y2;
;   px->x3 = y3;
;   px->x4 = y4;
;   px->x5 = y5;
; }

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

%struct.structx_t = type { [3 x i8], i8, [3 x i8], i8, [3 x i8], i8, [3 x i8], i8, [2 x i8], [2 x i8] }

define void @foo(%struct.structx_t* nocapture %px, i32 %y1, i32 %y2, i32 %y3, i32 %y4, i32 %y5) nounwind {
entry:
  %bf.value = and i32 %y1, 8388607
  %0 = bitcast %struct.structx_t* %px to i32*
  %1 = load i32, i32* %0, align 4
  %2 = and i32 %1, -8388608
  %3 = or i32 %2, %bf.value
  store i32 %3, i32* %0, align 4
  %bf.value1 = and i32 %y2, 131071
  %bf.field.offs = getelementptr %struct.structx_t, %struct.structx_t* %px, i32 0, i32 0, i32 4
  %4 = bitcast i8* %bf.field.offs to i32*
  %5 = load i32, i32* %4, align 4
  %6 = and i32 %5, -131072
  %7 = or i32 %6, %bf.value1
  store i32 %7, i32* %4, align 4
  %bf.value2 = and i32 %y3, 262143
  %bf.field.offs3 = getelementptr %struct.structx_t, %struct.structx_t* %px, i32 0, i32 0, i32 8
  %8 = bitcast i8* %bf.field.offs3 to i32*
  %9 = load i32, i32* %8, align 4
  %10 = and i32 %9, -262144
  %11 = or i32 %10, %bf.value2
  store i32 %11, i32* %8, align 4
  %bf.value4 = and i32 %y4, 4194303
  %bf.field.offs5 = getelementptr %struct.structx_t, %struct.structx_t* %px, i32 0, i32 0, i32 12
  %12 = bitcast i8* %bf.field.offs5 to i32*
  %13 = load i32, i32* %12, align 4
  %14 = and i32 %13, -4194304
  %15 = or i32 %14, %bf.value4
  store i32 %15, i32* %12, align 4
  %bf.value6 = and i32 %y5, 4095
  %bf.field.offs7 = getelementptr %struct.structx_t, %struct.structx_t* %px, i32 0, i32 0, i32 16
  %16 = bitcast i8* %bf.field.offs7 to i32*
  %17 = load i32, i32* %16, align 4
  %18 = and i32 %17, -4096
  %19 = or i32 %18, %bf.value6
  store i32 %19, i32* %16, align 4
  ret void
}
