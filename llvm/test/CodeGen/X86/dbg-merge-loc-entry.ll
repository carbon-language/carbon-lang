; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin8"

;CHECK: Ldebug_loc2:
;CHECK-NEXT:         .quad   Ltmp11
;CHECK-NEXT:         .quad   Lfunc_end0
;CHECK-NEXT:         .short  1                       ## Loc expr size
;CHECK-NEXT:         .byte   85                      ## DW_OP_reg5
;CHECK-NEXT:         .quad   0
;CHECK-NEXT:         .quad   0


%0 = type { i64, i1 }

@__clz_tab = external unnamed_addr constant [256 x i8]

define hidden i128 @__divti3(i128 %u, i128 %v) nounwind readnone {
entry:
  tail call void @llvm.dbg.value(metadata !{i128 %u}, i64 0, metadata !103), !dbg !111
  tail call void @llvm.dbg.value(metadata !{i128 %v}, i64 0, metadata !104), !dbg !111
  tail call void @llvm.dbg.declare(metadata !{null}, metadata !108), !dbg !112
  tail call void @llvm.dbg.declare(metadata !{null}, metadata !109), !dbg !113
  tail call void @llvm.dbg.value(metadata !114, i64 0, metadata !105), !dbg !115
  %0 = trunc i128 %u to i64
  %sroa.store.elt15 = lshr i128 %u, 64
  %1 = trunc i128 %sroa.store.elt15 to i64
  %2 = trunc i128 %v to i64
  %sroa.store.elt = lshr i128 %v, 64
  %3 = trunc i128 %sroa.store.elt to i64
  %4 = icmp slt i64 %1, 0, !dbg !116
  br i1 %4, label %bb, label %bb1, !dbg !116

bb:                                               ; preds = %entry
  tail call void @llvm.dbg.value(metadata !117, i64 0, metadata !105), !dbg !118
  %5 = sub nsw i128 0, %u, !dbg !118
  %6 = trunc i128 %5 to i64
  %sroa.store.elt18 = lshr i128 %5, 64
  %7 = trunc i128 %sroa.store.elt18 to i64
  br label %bb1, !dbg !118

bb1:                                              ; preds = %bb, %entry
  %uu.0.1.0 = phi i64 [ %7, %bb ], [ %1, %entry ]
  %uu.0.0.0 = phi i64 [ %6, %bb ], [ %0, %entry ]
  %c.0 = phi i64 [ -1, %bb ], [ 0, %entry ]
  %8 = icmp slt i64 %3, 0, !dbg !119
  br i1 %8, label %bb2, label %bb4, !dbg !119

bb2:                                              ; preds = %bb1
  %not3 = xor i64 %c.0, -1, !dbg !120
  tail call void @llvm.dbg.value(metadata !{i64 %not3}, i64 0, metadata !105), !dbg !120
  %9 = sub nsw i128 0, %v, !dbg !120
  %10 = trunc i128 %9 to i64
  %sroa.store.elt11 = lshr i128 %9, 64
  %11 = trunc i128 %sroa.store.elt11 to i64
  br label %bb4, !dbg !120

bb4:                                              ; preds = %bb2, %bb1
  %vv.0.1.0 = phi i64 [ %11, %bb2 ], [ %3, %bb1 ]
  %vv.0.0.0 = phi i64 [ %10, %bb2 ], [ %2, %bb1 ]
  %c.1 = phi i64 [ %not3, %bb2 ], [ %c.0, %bb1 ]
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !14) nounwind, !dbg !121
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !15) nounwind, !dbg !121
  tail call void @llvm.dbg.value(metadata !123, i64 0, metadata !16) nounwind, !dbg !121
  tail call void @llvm.dbg.declare(metadata !{null}, metadata !17) nounwind, !dbg !124
  tail call void @llvm.dbg.declare(metadata !{null}, metadata !30) nounwind, !dbg !125
  tail call void @llvm.dbg.declare(metadata !{null}, metadata !31) nounwind, !dbg !126
  tail call void @llvm.dbg.declare(metadata !{null}, metadata !43) nounwind, !dbg !127
  tail call void @llvm.dbg.value(metadata !{i64 %vv.0.0.0}, i64 0, metadata !32) nounwind, !dbg !128
  tail call void @llvm.dbg.value(metadata !{i64 %vv.0.1.0}, i64 0, metadata !35) nounwind, !dbg !129
  tail call void @llvm.dbg.value(metadata !{i64 %uu.0.0.0}, i64 0, metadata !36) nounwind, !dbg !130
  tail call void @llvm.dbg.value(metadata !{i64 %uu.0.1.0}, i64 0, metadata !37) nounwind, !dbg !131
  %12 = icmp eq i64 %vv.0.1.0, 0, !dbg !132
  br i1 %12, label %bb.i, label %bb73.i, !dbg !132

bb.i:                                             ; preds = %bb4
  %13 = icmp ugt i64 %vv.0.0.0, %uu.0.1.0, !dbg !133
  br i1 %13, label %bb4.i, label %bb21.i, !dbg !133

bb2.i:                                            ; preds = %bb4.i
  %tmp154.i = shl i64 255, %.cast.i
  %14 = and i64 %tmp154.i, %vv.0.0.0
  %15 = icmp eq i64 %14, 0, !dbg !134
  br i1 %15, label %bb3.i, label %bb5.i, !dbg !134

bb3.i:                                            ; preds = %bb2.i
  %indvar.next20.i = add i64 %indvar19.i, 1
  br label %bb4.i, !dbg !134

bb4.i:                                            ; preds = %bb.i, %bb3.i
  %indvar19.i = phi i64 [ %indvar.next20.i, %bb3.i ], [ 0, %bb.i ]
  %tmp24 = mul i64 %indvar19.i, -8
  %.cast.i = add i64 %tmp24, 56
  %16 = icmp eq i64 %.cast.i, 0, !dbg !134
  br i1 %16, label %bb5.i, label %bb2.i, !dbg !134

bb5.i:                                            ; preds = %bb4.i, %bb2.i
  %.cast6.i = and i64 %.cast.i, 4294967288
  %17 = lshr i64 %vv.0.0.0, %.cast6.i, !dbg !134
  %18 = getelementptr inbounds [256 x i8]* @__clz_tab, i64 0, i64 %17, !dbg !134
  %19 = load i8* %18, align 1, !dbg !134
  %20 = zext i8 %19 to i64, !dbg !134
  %21 = add i64 %20, %.cast.i, !dbg !134
  tail call void @llvm.dbg.value(metadata !135, i64 0, metadata !42) nounwind, !dbg !134
  %22 = icmp eq i64 %21, 64
  br i1 %22, label %bb12.i, label %bb7.i, !dbg !136

bb7.i:                                            ; preds = %bb5.i
  %23 = sub i64 64, %21, !dbg !134
  %.cast8.i = and i64 %23, 4294967295
  %24 = shl i64 %vv.0.0.0, %.cast8.i, !dbg !137
  tail call void @llvm.dbg.value(metadata !{i64 %24}, i64 0, metadata !32) nounwind, !dbg !137
  %25 = shl i64 %uu.0.1.0, %.cast8.i, !dbg !138
  %.cast10.i = and i64 %21, 4294967295
  %26 = lshr i64 %uu.0.0.0, %.cast10.i, !dbg !138
  %27 = or i64 %25, %26, !dbg !138
  tail call void @llvm.dbg.value(metadata !{i64 %27}, i64 0, metadata !37) nounwind, !dbg !138
  %28 = shl i64 %uu.0.0.0, %.cast8.i, !dbg !139
  tail call void @llvm.dbg.value(metadata !{i64 %28}, i64 0, metadata !36) nounwind, !dbg !139
  br label %bb12.i, !dbg !139

bb12.i:                                           ; preds = %bb7.i, %bb5.i
  %n1.0.i = phi i64 [ %27, %bb7.i ], [ %uu.0.1.0, %bb5.i ]
  %n0.0.i = phi i64 [ %28, %bb7.i ], [ %uu.0.0.0, %bb5.i ]
  %d0.0.i = phi i64 [ %24, %bb7.i ], [ %vv.0.0.0, %bb5.i ]
  %29 = lshr i64 %d0.0.i, 32, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %29}, i64 0, metadata !47) nounwind, !dbg !140
  %30 = and i64 %d0.0.i, 4294967295, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %30}, i64 0, metadata !49) nounwind, !dbg !140
  %31 = urem i64 %n1.0.i, %29, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %31}, i64 0, metadata !52) nounwind, !dbg !140
  %32 = udiv i64 %n1.0.i, %29, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %32}, i64 0, metadata !50) nounwind, !dbg !140
  %33 = mul i64 %32, %30, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %33}, i64 0, metadata !54) nounwind, !dbg !140
  %34 = shl i64 %31, 32
  %35 = lshr i64 %n0.0.i, 32, !dbg !140
  %36 = or i64 %34, %35, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %36}, i64 0, metadata !52) nounwind, !dbg !140
  %37 = icmp ult i64 %36, %33, !dbg !140
  br i1 %37, label %bb13.i, label %bb16.i, !dbg !140

bb13.i:                                           ; preds = %bb12.i
  %38 = add i64 %32, -1
  tail call void @llvm.dbg.value(metadata !{i64 %38}, i64 0, metadata !50) nounwind, !dbg !140
  %uadd153.i = tail call %0 @llvm.uadd.with.overflow.i64(i64 %36, i64 %d0.0.i) nounwind
  %39 = extractvalue %0 %uadd153.i, 0
  tail call void @llvm.dbg.value(metadata !{i64 %39}, i64 0, metadata !52) nounwind, !dbg !140
  %40 = extractvalue %0 %uadd153.i, 1
  %.not.i = xor i1 %40, true
  %41 = icmp ult i64 %39, %33, !dbg !140
  %or.cond.i = and i1 %41, %.not.i
  br i1 %or.cond.i, label %bb15.i, label %bb16.i, !dbg !140

bb15.i:                                           ; preds = %bb13.i
  %42 = add i64 %32, -2
  tail call void @llvm.dbg.value(metadata !{i64 %42}, i64 0, metadata !50) nounwind, !dbg !140
  %43 = add i64 %39, %d0.0.i, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %43}, i64 0, metadata !52) nounwind, !dbg !140
  br label %bb16.i, !dbg !140

bb16.i:                                           ; preds = %bb15.i, %bb13.i, %bb12.i
  %__r1.0.i = phi i64 [ %43, %bb15.i ], [ %39, %bb13.i ], [ %36, %bb12.i ]
  %__q1.0.i = phi i64 [ %42, %bb15.i ], [ %38, %bb13.i ], [ %32, %bb12.i ]
  %44 = sub i64 %__r1.0.i, %33, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %44}, i64 0, metadata !52) nounwind, !dbg !140
  %45 = urem i64 %44, %29, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %45}, i64 0, metadata !53) nounwind, !dbg !140
  %46 = udiv i64 %44, %29, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %46}, i64 0, metadata !51) nounwind, !dbg !140
  %47 = mul i64 %46, %30, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %47}, i64 0, metadata !54) nounwind, !dbg !140
  %48 = shl i64 %45, 32
  %49 = and i64 %n0.0.i, 4294967295, !dbg !140
  %50 = or i64 %48, %49, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %50}, i64 0, metadata !53) nounwind, !dbg !140
  %51 = icmp ult i64 %50, %47, !dbg !140
  br i1 %51, label %bb17.i, label %bb20.i, !dbg !140

bb17.i:                                           ; preds = %bb16.i
  %52 = add i64 %46, -1
  tail call void @llvm.dbg.value(metadata !{i64 %52}, i64 0, metadata !51) nounwind, !dbg !140
  %uadd152.i = tail call %0 @llvm.uadd.with.overflow.i64(i64 %50, i64 %d0.0.i) nounwind
  tail call void @llvm.dbg.value(metadata !135, i64 0, metadata !53) nounwind, !dbg !140
  %53 = extractvalue %0 %uadd152.i, 1
  br i1 %53, label %bb20.i, label %bb18.i, !dbg !140

bb18.i:                                           ; preds = %bb17.i
  %54 = extractvalue %0 %uadd152.i, 0
  %55 = add i64 %46, -2
  %56 = icmp ult i64 %54, %47, !dbg !140
  %..i = select i1 %56, i64 %55, i64 %52
  br label %bb20.i

bb20.i:                                           ; preds = %bb18.i, %bb17.i, %bb16.i
  %__q0.0.i = phi i64 [ %52, %bb17.i ], [ %46, %bb16.i ], [ %..i, %bb18.i ]
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !53) nounwind, !dbg !140
  %57 = shl i64 %__q1.0.i, 32
  %58 = or i64 %__q0.0.i, %57, !dbg !140
  tail call void @llvm.dbg.value(metadata !{i64 %58}, i64 0, metadata !39) nounwind, !dbg !140
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !36) nounwind, !dbg !140
  tail call void @llvm.dbg.value(metadata !114, i64 0, metadata !40) nounwind, !dbg !141
  br label %__udivmodti4.exit, !dbg !141

bb21.i:                                           ; preds = %bb.i
  %59 = icmp eq i64 %vv.0.0.0, 0, !dbg !142
  br i1 %59, label %bb22.i, label %bb23.i, !dbg !142

bb22.i:                                           ; preds = %bb21.i
  %60 = udiv i64 1, %vv.0.0.0, !dbg !143
  tail call void @llvm.dbg.value(metadata !{i64 %60}, i64 0, metadata !32) nounwind, !dbg !143
  br label %bb23.i, !dbg !143

bb23.i:                                           ; preds = %bb22.i, %bb21.i
  %d0.1.i = phi i64 [ %60, %bb22.i ], [ %vv.0.0.0, %bb21.i ]
  tail call void @llvm.dbg.value(metadata !{i64 %d0.1.i}, i64 0, metadata !55) nounwind, !dbg !144
  tail call void @llvm.dbg.value(metadata !145, i64 0, metadata !57) nounwind, !dbg !144
  br label %bb29.i, !dbg !144

bb26.i:                                           ; preds = %bb29.i
  %tmp151.i = shl i64 255, %.cast27.i
  %61 = and i64 %tmp151.i, %d0.1.i
  %62 = icmp eq i64 %61, 0, !dbg !144
  br i1 %62, label %bb28.i, label %bb30.i, !dbg !144

bb28.i:                                           ; preds = %bb26.i
  %indvar.next16.i = add i64 %indvar15.i, 1
  br label %bb29.i, !dbg !144

bb29.i:                                           ; preds = %bb28.i, %bb23.i
  %indvar15.i = phi i64 [ %indvar.next16.i, %bb28.i ], [ 0, %bb23.i ]
  %tmp22 = mul i64 %indvar15.i, -8
  %.cast27.i = add i64 %tmp22, 56
  %63 = icmp eq i64 %.cast27.i, 0, !dbg !144
  br i1 %63, label %bb30.i, label %bb26.i, !dbg !144

bb30.i:                                           ; preds = %bb29.i, %bb26.i
  %.cast31.i = and i64 %.cast27.i, 4294967288
  %64 = lshr i64 %d0.1.i, %.cast31.i, !dbg !144
  %65 = getelementptr inbounds [256 x i8]* @__clz_tab, i64 0, i64 %64, !dbg !144
  %66 = load i8* %65, align 1, !dbg !144
  %67 = zext i8 %66 to i64, !dbg !144
  %68 = add i64 %67, %.cast27.i, !dbg !144
  tail call void @llvm.dbg.value(metadata !135, i64 0, metadata !42) nounwind, !dbg !144
  %69 = icmp eq i64 %68, 64
  br i1 %69, label %bb32.i, label %bb33.i, !dbg !146

bb32.i:                                           ; preds = %bb30.i
  %70 = sub i64 %uu.0.1.0, %d0.1.i, !dbg !147
  tail call void @llvm.dbg.value(metadata !{i64 %70}, i64 0, metadata !37) nounwind, !dbg !147
  tail call void @llvm.dbg.value(metadata !148, i64 0, metadata !40) nounwind, !dbg !149
  br label %bb54.i, !dbg !149

bb33.i:                                           ; preds = %bb30.i
  %71 = sub i64 64, %68, !dbg !144
  tail call void @llvm.dbg.value(metadata !{i64 %68}, i64 0, metadata !41) nounwind, !dbg !150
  %.cast34.i = and i64 %71, 4294967295
  %72 = shl i64 %d0.1.i, %.cast34.i, !dbg !151
  tail call void @llvm.dbg.value(metadata !{i64 %72}, i64 0, metadata !32) nounwind, !dbg !151
  %.cast35.i = and i64 %68, 4294967295
  %73 = lshr i64 %uu.0.1.0, %.cast35.i, !dbg !152
  tail call void @llvm.dbg.value(metadata !{i64 %73}, i64 0, metadata !38) nounwind, !dbg !152
  %74 = shl i64 %uu.0.1.0, %.cast34.i, !dbg !153
  %75 = lshr i64 %uu.0.0.0, %.cast35.i, !dbg !153
  %76 = or i64 %74, %75, !dbg !153
  tail call void @llvm.dbg.value(metadata !{i64 %76}, i64 0, metadata !37) nounwind, !dbg !153
  %77 = shl i64 %uu.0.0.0, %.cast34.i, !dbg !154
  tail call void @llvm.dbg.value(metadata !{i64 %77}, i64 0, metadata !36) nounwind, !dbg !154
  %78 = lshr i64 %72, 32, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %78}, i64 0, metadata !58) nounwind, !dbg !155
  %79 = and i64 %72, 4294967295, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %79}, i64 0, metadata !60) nounwind, !dbg !155
  %80 = urem i64 %73, %78, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %80}, i64 0, metadata !63) nounwind, !dbg !155
  %81 = udiv i64 %73, %78, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %81}, i64 0, metadata !61) nounwind, !dbg !155
  %82 = mul i64 %81, %79, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %82}, i64 0, metadata !65) nounwind, !dbg !155
  %83 = shl i64 %80, 32
  %84 = lshr i64 %76, 32, !dbg !155
  %85 = or i64 %83, %84, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %85}, i64 0, metadata !63) nounwind, !dbg !155
  %86 = icmp ult i64 %85, %82, !dbg !155
  br i1 %86, label %bb46.i, label %bb49.i, !dbg !155

bb46.i:                                           ; preds = %bb33.i
  %87 = add i64 %81, -1
  tail call void @llvm.dbg.value(metadata !{i64 %87}, i64 0, metadata !61) nounwind, !dbg !155
  %uadd150.i = tail call %0 @llvm.uadd.with.overflow.i64(i64 %85, i64 %72) nounwind
  %88 = extractvalue %0 %uadd150.i, 0
  tail call void @llvm.dbg.value(metadata !{i64 %88}, i64 0, metadata !63) nounwind, !dbg !155
  %89 = extractvalue %0 %uadd150.i, 1
  %.not1.i = xor i1 %89, true
  %90 = icmp ult i64 %88, %82, !dbg !155
  %or.cond2.i = and i1 %90, %.not1.i
  br i1 %or.cond2.i, label %bb48.i, label %bb49.i, !dbg !155

bb48.i:                                           ; preds = %bb46.i
  %91 = add i64 %81, -2
  tail call void @llvm.dbg.value(metadata !{i64 %91}, i64 0, metadata !61) nounwind, !dbg !155
  %92 = add i64 %88, %72, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %92}, i64 0, metadata !63) nounwind, !dbg !155
  br label %bb49.i, !dbg !155

bb49.i:                                           ; preds = %bb48.i, %bb46.i, %bb33.i
  %__q141.0.i = phi i64 [ %91, %bb48.i ], [ %87, %bb46.i ], [ %81, %bb33.i ]
  %__r143.0.i = phi i64 [ %92, %bb48.i ], [ %88, %bb46.i ], [ %85, %bb33.i ]
  %93 = sub i64 %__r143.0.i, %82, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %93}, i64 0, metadata !63) nounwind, !dbg !155
  %94 = urem i64 %93, %78, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %94}, i64 0, metadata !64) nounwind, !dbg !155
  %95 = udiv i64 %93, %78, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %95}, i64 0, metadata !62) nounwind, !dbg !155
  %96 = mul i64 %95, %79, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %96}, i64 0, metadata !65) nounwind, !dbg !155
  %97 = shl i64 %94, 32
  %98 = and i64 %76, 4294967295, !dbg !155
  %99 = or i64 %97, %98, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %99}, i64 0, metadata !64) nounwind, !dbg !155
  %100 = icmp ult i64 %99, %96, !dbg !155
  br i1 %100, label %bb50.i, label %bb53.i, !dbg !155

bb50.i:                                           ; preds = %bb49.i
  %101 = add i64 %95, -1
  tail call void @llvm.dbg.value(metadata !{i64 %101}, i64 0, metadata !62) nounwind, !dbg !155
  %uadd149.i = tail call %0 @llvm.uadd.with.overflow.i64(i64 %99, i64 %72) nounwind
  %102 = extractvalue %0 %uadd149.i, 0
  tail call void @llvm.dbg.value(metadata !{i64 %102}, i64 0, metadata !64) nounwind, !dbg !155
  %103 = extractvalue %0 %uadd149.i, 1
  %.not3.i = xor i1 %103, true
  %104 = icmp ult i64 %102, %96, !dbg !155
  %or.cond4.i = and i1 %104, %.not3.i
  br i1 %or.cond4.i, label %bb52.i, label %bb53.i, !dbg !155

bb52.i:                                           ; preds = %bb50.i
  %105 = add i64 %95, -2
  tail call void @llvm.dbg.value(metadata !{i64 %105}, i64 0, metadata !62) nounwind, !dbg !155
  %106 = add i64 %102, %72, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %106}, i64 0, metadata !64) nounwind, !dbg !155
  br label %bb53.i, !dbg !155

bb53.i:                                           ; preds = %bb52.i, %bb50.i, %bb49.i
  %__q042.0.i = phi i64 [ %105, %bb52.i ], [ %101, %bb50.i ], [ %95, %bb49.i ]
  %__r044.0.i = phi i64 [ %106, %bb52.i ], [ %102, %bb50.i ], [ %99, %bb49.i ]
  %107 = sub i64 %__r044.0.i, %96, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %107}, i64 0, metadata !64) nounwind, !dbg !155
  %108 = shl i64 %__q141.0.i, 32
  %109 = or i64 %__q042.0.i, %108, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %109}, i64 0, metadata !40) nounwind, !dbg !155
  tail call void @llvm.dbg.value(metadata !{i64 %107}, i64 0, metadata !37) nounwind, !dbg !155
  br label %bb54.i, !dbg !155

bb54.i:                                           ; preds = %bb53.i, %bb32.i
  %q1.0.i = phi i64 [ 1, %bb32.i ], [ %109, %bb53.i ]
  %n1.1.i = phi i64 [ %70, %bb32.i ], [ %107, %bb53.i ]
  %n0.1.i = phi i64 [ %uu.0.0.0, %bb32.i ], [ %77, %bb53.i ]
  %d0.2.i = phi i64 [ %d0.1.i, %bb32.i ], [ %72, %bb53.i ]
  %110 = lshr i64 %d0.2.i, 32, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %110}, i64 0, metadata !66) nounwind, !dbg !156
  %111 = and i64 %d0.2.i, 4294967295, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %111}, i64 0, metadata !68) nounwind, !dbg !156
  %112 = urem i64 %n1.1.i, %110, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %112}, i64 0, metadata !71) nounwind, !dbg !156
  %113 = udiv i64 %n1.1.i, %110, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %113}, i64 0, metadata !69) nounwind, !dbg !156
  %114 = mul i64 %113, %111, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %114}, i64 0, metadata !73) nounwind, !dbg !156
  %115 = shl i64 %112, 32
  %116 = lshr i64 %n0.1.i, 32, !dbg !156
  %117 = or i64 %115, %116, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %117}, i64 0, metadata !71) nounwind, !dbg !156
  %118 = icmp ult i64 %117, %114, !dbg !156
  br i1 %118, label %bb62.i, label %bb65.i, !dbg !156

bb62.i:                                           ; preds = %bb54.i
  %119 = add i64 %113, -1
  tail call void @llvm.dbg.value(metadata !{i64 %119}, i64 0, metadata !69) nounwind, !dbg !156
  %uadd148.i = tail call %0 @llvm.uadd.with.overflow.i64(i64 %117, i64 %d0.2.i) nounwind
  %120 = extractvalue %0 %uadd148.i, 0
  tail call void @llvm.dbg.value(metadata !{i64 %120}, i64 0, metadata !71) nounwind, !dbg !156
  %121 = extractvalue %0 %uadd148.i, 1
  %.not5.i = xor i1 %121, true
  %122 = icmp ult i64 %120, %114, !dbg !156
  %or.cond6.i = and i1 %122, %.not5.i
  br i1 %or.cond6.i, label %bb64.i, label %bb65.i, !dbg !156

bb64.i:                                           ; preds = %bb62.i
  %123 = add i64 %113, -2
  tail call void @llvm.dbg.value(metadata !{i64 %123}, i64 0, metadata !69) nounwind, !dbg !156
  %124 = add i64 %120, %d0.2.i, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %124}, i64 0, metadata !71) nounwind, !dbg !156
  br label %bb65.i, !dbg !156

bb65.i:                                           ; preds = %bb64.i, %bb62.i, %bb54.i
  %__q157.0.i = phi i64 [ %123, %bb64.i ], [ %119, %bb62.i ], [ %113, %bb54.i ]
  %__r159.0.i = phi i64 [ %124, %bb64.i ], [ %120, %bb62.i ], [ %117, %bb54.i ]
  %125 = sub i64 %__r159.0.i, %114, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %125}, i64 0, metadata !71) nounwind, !dbg !156
  %126 = urem i64 %125, %110, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %126}, i64 0, metadata !72) nounwind, !dbg !156
  %127 = udiv i64 %125, %110, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %127}, i64 0, metadata !70) nounwind, !dbg !156
  %128 = mul i64 %127, %111, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %128}, i64 0, metadata !73) nounwind, !dbg !156
  %129 = shl i64 %126, 32
  %130 = and i64 %n0.1.i, 4294967295, !dbg !156
  %131 = or i64 %129, %130, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %131}, i64 0, metadata !72) nounwind, !dbg !156
  %132 = icmp ult i64 %131, %128, !dbg !156
  br i1 %132, label %bb66.i, label %bb69.i, !dbg !156

bb66.i:                                           ; preds = %bb65.i
  %133 = add i64 %127, -1
  tail call void @llvm.dbg.value(metadata !{i64 %133}, i64 0, metadata !70) nounwind, !dbg !156
  %uadd147.i = tail call %0 @llvm.uadd.with.overflow.i64(i64 %131, i64 %d0.2.i) nounwind
  tail call void @llvm.dbg.value(metadata !135, i64 0, metadata !72) nounwind, !dbg !156
  %134 = extractvalue %0 %uadd147.i, 1
  br i1 %134, label %bb69.i, label %bb67.i, !dbg !156

bb67.i:                                           ; preds = %bb66.i
  %135 = extractvalue %0 %uadd147.i, 0
  %136 = add i64 %127, -2
  %137 = icmp ult i64 %135, %128, !dbg !156
  %.7.i = select i1 %137, i64 %136, i64 %133
  br label %bb69.i

bb69.i:                                           ; preds = %bb67.i, %bb66.i, %bb65.i
  %__q058.0.i = phi i64 [ %133, %bb66.i ], [ %127, %bb65.i ], [ %.7.i, %bb67.i ]
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !72) nounwind, !dbg !156
  %138 = shl i64 %__q157.0.i, 32
  %139 = or i64 %__q058.0.i, %138, !dbg !156
  tail call void @llvm.dbg.value(metadata !{i64 %139}, i64 0, metadata !39) nounwind, !dbg !156
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !36) nounwind, !dbg !156
  br label %__udivmodti4.exit, !dbg !156

bb73.i:                                           ; preds = %bb4
  %140 = icmp ugt i64 %vv.0.1.0, %uu.0.1.0, !dbg !157
  br i1 %140, label %__udivmodti4.exit, label %bb82.i, !dbg !157

bb79.i:                                           ; preds = %bb82.i
  %tmp.i = shl i64 255, %.cast80.i
  %141 = and i64 %tmp.i, %vv.0.1.0
  %142 = icmp eq i64 %141, 0, !dbg !158
  br i1 %142, label %bb81.i, label %bb83.i, !dbg !158

bb81.i:                                           ; preds = %bb79.i
  %indvar.next.i = add i64 %indvar.i, 1
  br label %bb82.i, !dbg !158

bb82.i:                                           ; preds = %bb73.i, %bb81.i
  %indvar.i = phi i64 [ %indvar.next.i, %bb81.i ], [ 0, %bb73.i ]
  %tmp = mul i64 %indvar.i, -8
  %.cast80.i = add i64 %tmp, 56
  %143 = icmp eq i64 %.cast80.i, 0, !dbg !158
  br i1 %143, label %bb83.i, label %bb79.i, !dbg !158

bb83.i:                                           ; preds = %bb82.i, %bb79.i
  %.cast84.i = and i64 %.cast80.i, 4294967288
  %144 = lshr i64 %vv.0.1.0, %.cast84.i, !dbg !158
  %145 = getelementptr inbounds [256 x i8]* @__clz_tab, i64 0, i64 %144, !dbg !158
  %146 = load i8* %145, align 1, !dbg !158
  %147 = zext i8 %146 to i64, !dbg !158
  %148 = add i64 %147, %.cast80.i, !dbg !158
  tail call void @llvm.dbg.value(metadata !135, i64 0, metadata !42) nounwind, !dbg !158
  %149 = icmp eq i64 %148, 64
  br i1 %149, label %bb85.i, label %bb92.i, !dbg !159

bb85.i:                                           ; preds = %bb83.i
  %150 = icmp ugt i64 %uu.0.1.0, %vv.0.1.0, !dbg !160
  %151 = icmp uge i64 %uu.0.0.0, %vv.0.0.0, !dbg !160
  %152 = or i1 %150, %151, !dbg !160
  %.8.i = zext i1 %152 to i64
  tail call void @llvm.dbg.value(metadata !114, i64 0, metadata !40) nounwind, !dbg !161
  br label %__udivmodti4.exit

bb92.i:                                           ; preds = %bb83.i
  %153 = sub i64 64, %148, !dbg !158
  tail call void @llvm.dbg.value(metadata !{i64 %148}, i64 0, metadata !41) nounwind, !dbg !162
  %.cast93.i = and i64 %153, 4294967295
  %154 = shl i64 %vv.0.1.0, %.cast93.i, !dbg !163
  %.cast94.i = and i64 %148, 4294967295
  %155 = lshr i64 %vv.0.0.0, %.cast94.i, !dbg !163
  %156 = or i64 %154, %155, !dbg !163
  tail call void @llvm.dbg.value(metadata !{i64 %156}, i64 0, metadata !35) nounwind, !dbg !163
  %157 = shl i64 %vv.0.0.0, %.cast93.i, !dbg !164
  tail call void @llvm.dbg.value(metadata !{i64 %157}, i64 0, metadata !32) nounwind, !dbg !164
  %158 = lshr i64 %uu.0.1.0, %.cast94.i, !dbg !165
  tail call void @llvm.dbg.value(metadata !{i64 %158}, i64 0, metadata !38) nounwind, !dbg !165
  %159 = shl i64 %uu.0.1.0, %.cast93.i, !dbg !166
  %160 = lshr i64 %uu.0.0.0, %.cast94.i, !dbg !166
  %161 = or i64 %159, %160, !dbg !166
  tail call void @llvm.dbg.value(metadata !{i64 %161}, i64 0, metadata !37) nounwind, !dbg !166
  %162 = shl i64 %uu.0.0.0, %.cast93.i, !dbg !167
  tail call void @llvm.dbg.value(metadata !{i64 %162}, i64 0, metadata !36) nounwind, !dbg !167
  %163 = lshr i64 %156, 32, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %163}, i64 0, metadata !82) nounwind, !dbg !168
  %164 = and i64 %156, 4294967295, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %164}, i64 0, metadata !84) nounwind, !dbg !168
  %165 = urem i64 %158, %163, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %165}, i64 0, metadata !87) nounwind, !dbg !168
  %166 = udiv i64 %158, %163, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %166}, i64 0, metadata !85) nounwind, !dbg !168
  %167 = mul i64 %166, %164, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %167}, i64 0, metadata !89) nounwind, !dbg !168
  %168 = shl i64 %165, 32
  %169 = lshr i64 %161, 32, !dbg !168
  %170 = or i64 %168, %169, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %170}, i64 0, metadata !87) nounwind, !dbg !168
  %171 = icmp ult i64 %170, %167, !dbg !168
  br i1 %171, label %bb107.i, label %bb110.i, !dbg !168

bb107.i:                                          ; preds = %bb92.i
  %172 = add i64 %166, -1
  tail call void @llvm.dbg.value(metadata !{i64 %172}, i64 0, metadata !85) nounwind, !dbg !168
  %uadd146.i = tail call %0 @llvm.uadd.with.overflow.i64(i64 %170, i64 %156) nounwind
  %173 = extractvalue %0 %uadd146.i, 0
  tail call void @llvm.dbg.value(metadata !{i64 %173}, i64 0, metadata !87) nounwind, !dbg !168
  %174 = extractvalue %0 %uadd146.i, 1
  %.not9.i = xor i1 %174, true
  %175 = icmp ult i64 %173, %167, !dbg !168
  %or.cond10.i = and i1 %175, %.not9.i
  br i1 %or.cond10.i, label %bb109.i, label %bb110.i, !dbg !168

bb109.i:                                          ; preds = %bb107.i
  %176 = add i64 %166, -2
  tail call void @llvm.dbg.value(metadata !{i64 %176}, i64 0, metadata !85) nounwind, !dbg !168
  %177 = add i64 %173, %156, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %177}, i64 0, metadata !87) nounwind, !dbg !168
  br label %bb110.i, !dbg !168

bb110.i:                                          ; preds = %bb109.i, %bb107.i, %bb92.i
  %__q1102.0.i = phi i64 [ %176, %bb109.i ], [ %172, %bb107.i ], [ %166, %bb92.i ]
  %__r1104.0.i = phi i64 [ %177, %bb109.i ], [ %173, %bb107.i ], [ %170, %bb92.i ]
  %178 = sub i64 %__r1104.0.i, %167, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %178}, i64 0, metadata !87) nounwind, !dbg !168
  %179 = urem i64 %178, %163, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %179}, i64 0, metadata !88) nounwind, !dbg !168
  %180 = udiv i64 %178, %163, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %180}, i64 0, metadata !86) nounwind, !dbg !168
  %181 = mul i64 %180, %164, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %181}, i64 0, metadata !89) nounwind, !dbg !168
  %182 = shl i64 %179, 32
  %183 = and i64 %161, 4294967295, !dbg !168
  %184 = or i64 %182, %183, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %184}, i64 0, metadata !88) nounwind, !dbg !168
  %185 = icmp ult i64 %184, %181, !dbg !168
  br i1 %185, label %bb111.i, label %bb114.i, !dbg !168

bb111.i:                                          ; preds = %bb110.i
  %186 = add i64 %180, -1
  tail call void @llvm.dbg.value(metadata !{i64 %186}, i64 0, metadata !86) nounwind, !dbg !168
  %uadd145.i = tail call %0 @llvm.uadd.with.overflow.i64(i64 %184, i64 %156) nounwind
  %187 = extractvalue %0 %uadd145.i, 0
  tail call void @llvm.dbg.value(metadata !{i64 %187}, i64 0, metadata !88) nounwind, !dbg !168
  %188 = extractvalue %0 %uadd145.i, 1
  %.not11.i = xor i1 %188, true
  %189 = icmp ult i64 %187, %181, !dbg !168
  %or.cond12.i = and i1 %189, %.not11.i
  br i1 %or.cond12.i, label %bb113.i, label %bb114.i, !dbg !168

bb113.i:                                          ; preds = %bb111.i
  %190 = add i64 %180, -2
  tail call void @llvm.dbg.value(metadata !{i64 %190}, i64 0, metadata !86) nounwind, !dbg !168
  %191 = add i64 %187, %156, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %191}, i64 0, metadata !88) nounwind, !dbg !168
  br label %bb114.i, !dbg !168

bb114.i:                                          ; preds = %bb113.i, %bb111.i, %bb110.i
  %__q0103.0.i = phi i64 [ %190, %bb113.i ], [ %186, %bb111.i ], [ %180, %bb110.i ]
  %__r0105.0.i = phi i64 [ %191, %bb113.i ], [ %187, %bb111.i ], [ %184, %bb110.i ]
  %192 = sub i64 %__r0105.0.i, %181, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %192}, i64 0, metadata !88) nounwind, !dbg !168
  %193 = shl i64 %__q1102.0.i, 32
  %194 = or i64 %__q0103.0.i, %193, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %194}, i64 0, metadata !39) nounwind, !dbg !168
  tail call void @llvm.dbg.value(metadata !{i64 %192}, i64 0, metadata !37) nounwind, !dbg !168
  %195 = and i64 %__q0103.0.i, 4294967295, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %195}, i64 0, metadata !95) nounwind, !dbg !169
  %196 = lshr i64 %194, 32, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %196}, i64 0, metadata !97) nounwind, !dbg !169
  %197 = and i64 %157, 4294967295, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %197}, i64 0, metadata !96) nounwind, !dbg !169
  %198 = lshr i64 %157, 32, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %198}, i64 0, metadata !98) nounwind, !dbg !169
  %199 = mul i64 %195, %197, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %199}, i64 0, metadata !90) nounwind, !dbg !169
  %200 = mul i64 %195, %198, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %200}, i64 0, metadata !92) nounwind, !dbg !169
  %201 = mul i64 %196, %197, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %201}, i64 0, metadata !93) nounwind, !dbg !169
  %202 = mul i64 %196, %198, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %202}, i64 0, metadata !94) nounwind, !dbg !169
  %203 = lshr i64 %199, 32, !dbg !169
  %204 = add i64 %203, %200, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %204}, i64 0, metadata !92) nounwind, !dbg !169
  %uadd.i = tail call %0 @llvm.uadd.with.overflow.i64(i64 %204, i64 %201) nounwind
  %205 = extractvalue %0 %uadd.i, 0
  tail call void @llvm.dbg.value(metadata !{i64 %205}, i64 0, metadata !92) nounwind, !dbg !169
  %206 = extractvalue %0 %uadd.i, 1
  %207 = add i64 %202, 4294967296, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %207}, i64 0, metadata !94) nounwind, !dbg !169
  %__x3.0.i = select i1 %206, i64 %207, i64 %202
  %208 = lshr i64 %205, 32, !dbg !169
  %209 = add i64 %__x3.0.i, %208, !dbg !169
  tail call void @llvm.dbg.value(metadata !{i64 %209}, i64 0, metadata !79) nounwind, !dbg !169
  tail call void @llvm.dbg.value(metadata !135, i64 0, metadata !81) nounwind, !dbg !169
  %210 = icmp ugt i64 %209, %192, !dbg !170
  br i1 %210, label %bb121.i, label %bb117.i, !dbg !170

bb117.i:                                          ; preds = %bb114.i
  %211 = and i64 %199, 4294967295, !dbg !169
  %212 = shl i64 %205, 32
  %213 = or i64 %212, %211
  %214 = icmp eq i64 %209, %192, !dbg !170
  %215 = icmp ugt i64 %213, %162, !dbg !170
  %216 = and i1 %214, %215, !dbg !170
  br i1 %216, label %bb121.i, label %__udivmodti4.exit, !dbg !170

bb121.i:                                          ; preds = %bb117.i, %bb114.i
  %217 = add i64 %194, -1
  tail call void @llvm.dbg.value(metadata !{i64 %217}, i64 0, metadata !39) nounwind, !dbg !171
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !99) nounwind, !dbg !172
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !79) nounwind, !dbg !172
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !81) nounwind, !dbg !172
  br label %__udivmodti4.exit, !dbg !172

__udivmodti4.exit:                                ; preds = %bb20.i, %bb69.i, %bb73.i, %bb85.i, %bb117.i, %bb121.i
  %q1.2.i = phi i64 [ 0, %bb85.i ], [ 0, %bb20.i ], [ %q1.0.i, %bb69.i ], [ 0, %bb73.i ], [ 0, %bb121.i ], [ 0, %bb117.i ]
  %q0.3.i = phi i64 [ %.8.i, %bb85.i ], [ %58, %bb20.i ], [ %139, %bb69.i ], [ 0, %bb73.i ], [ %217, %bb121.i ], [ %194, %bb117.i ]
  %218 = zext i64 %q0.3.i to i128
  %219 = zext i64 %q1.2.i to i128
  %220 = shl i128 %219, 64
  %221 = or i128 %220, %218
  tail call void @llvm.dbg.value(metadata !{i128 %221}, i64 0, metadata !110), !dbg !122
  %222 = icmp eq i64 %c.1, 0, !dbg !173
  %223 = sub nsw i128 0, %221, !dbg !174
  tail call void @llvm.dbg.value(metadata !{i128 %223}, i64 0, metadata !110), !dbg !174
  %w.0 = select i1 %222, i128 %221, i128 %223
  ret i128 %w.0, !dbg !175
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

declare %0 @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone

!llvm.dbg.sp = !{!0, !9}
!llvm.dbg.lv.__udivmodti4 = !{!14, !15, !16, !17, !30, !31, !32, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !46, !47, !49, !50, !51, !52, !53, !54, !55, !57, !58, !60, !61, !62, !63, !64, !65, !66, !68, !69, !70, !71, !72, !73, !74, !76, !77, !79, !81, !82, !84, !85, !86, !87, !88, !89, !90, !92, !93, !94, !95, !96, !97, !98, !99, !101}
!llvm.dbg.lv.__divti3 = !{!103, !104, !105, !108, !109, !110}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"__udivmodti4", metadata !"__udivmodti4", metadata !"", metadata !1, i32 879, metadata !3, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"foobar.c", metadata !"/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 1, metadata !"foobar.c", metadata !"/tmp", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5, metadata !5, metadata !5, metadata !8}
!5 = metadata !{i32 589846, metadata !6, metadata !"UTItype", metadata !6, i32 166, i64 0, i64 0, i64 0, i32 0, metadata !7} ; [ DW_TAG_typedef ]
!6 = metadata !{i32 589865, metadata !"foobar.h", metadata !"/tmp", metadata !2} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 589860, metadata !1, metadata !"", metadata !1, i32 0, i64 128, i64 128, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 589839, metadata !1, metadata !"", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !5} ; [ DW_TAG_pointer_type ]
!9 = metadata !{i32 589870, i32 0, metadata !1, metadata !"__divti3", metadata !"__divti3", metadata !"__divti3", metadata !1, i32 1094, metadata !10, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i128 (i128, i128)* @__divti3} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null} ; [ DW_TAG_subroutine_type ]
!11 = metadata !{metadata !12, metadata !12, metadata !12}
!12 = metadata !{i32 589846, metadata !6, metadata !"TItype", metadata !6, i32 160, i64 0, i64 0, i64 0, i32 0, metadata !13} ; [ DW_TAG_typedef ]
!13 = metadata !{i32 589860, metadata !1, metadata !"", metadata !1, i32 0, i64 128, i64 128, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!14 = metadata !{i32 590081, metadata !0, metadata !"n", metadata !1, i32 878, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 590081, metadata !0, metadata !"d", metadata !1, i32 878, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!16 = metadata !{i32 590081, metadata !0, metadata !"rp", metadata !1, i32 878, metadata !8, i32 0} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 590080, metadata !18, metadata !"nn", metadata !1, i32 880, metadata !19, i32 0} ; [ DW_TAG_auto_variable ]
!18 = metadata !{i32 589835, metadata !0, i32 879, i32 0, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!19 = metadata !{i32 589846, metadata !1, metadata !"DWunion", metadata !1, i32 879, i64 0, i64 0, i64 0, i32 0, metadata !20} ; [ DW_TAG_typedef ]
!20 = metadata !{i32 589847, metadata !1, metadata !"", metadata !6, i32 432, i64 128, i64 128, i64 0, i32 0, null, metadata !21, i32 0, null} ; [ DW_TAG_union_type ]
!21 = metadata !{metadata !22, metadata !29}
!22 = metadata !{i32 589837, metadata !20, metadata !"s", metadata !6, i32 433, i64 128, i64 64, i64 0, i32 0, metadata !23} ; [ DW_TAG_member ]
!23 = metadata !{i32 589843, metadata !1, metadata !"DWstruct", metadata !6, i32 424, i64 128, i64 64, i64 0, i32 0, null, metadata !24, i32 0, null} ; [ DW_TAG_structure_type ]
!24 = metadata !{metadata !25, metadata !28}
!25 = metadata !{i32 589837, metadata !23, metadata !"low", metadata !6, i32 424, i64 64, i64 64, i64 0, i32 0, metadata !26} ; [ DW_TAG_member ]
!26 = metadata !{i32 589846, metadata !6, metadata !"DItype", metadata !6, i32 156, i64 0, i64 0, i64 0, i32 0, metadata !27} ; [ DW_TAG_typedef ]
!27 = metadata !{i32 589860, metadata !1, metadata !"long int", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!28 = metadata !{i32 589837, metadata !23, metadata !"high", metadata !6, i32 424, i64 64, i64 64, i64 64, i32 0, metadata !26} ; [ DW_TAG_member ]
!29 = metadata !{i32 589837, metadata !20, metadata !"ll", metadata !6, i32 434, i64 128, i64 128, i64 0, i32 0, metadata !12} ; [ DW_TAG_member ]
!30 = metadata !{i32 590080, metadata !18, metadata !"dd", metadata !1, i32 881, metadata !19, i32 0} ; [ DW_TAG_auto_variable ]
!31 = metadata !{i32 590080, metadata !18, metadata !"rr", metadata !1, i32 882, metadata !19, i32 0} ; [ DW_TAG_auto_variable ]
!32 = metadata !{i32 590080, metadata !18, metadata !"d0", metadata !1, i32 883, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!33 = metadata !{i32 589846, metadata !6, metadata !"UDItype", metadata !6, i32 159, i64 0, i64 0, i64 0, i32 0, metadata !34} ; [ DW_TAG_typedef ]
!34 = metadata !{i32 589860, metadata !1, metadata !"long unsigned int", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!35 = metadata !{i32 590080, metadata !18, metadata !"d1", metadata !1, i32 883, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!36 = metadata !{i32 590080, metadata !18, metadata !"n0", metadata !1, i32 883, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!37 = metadata !{i32 590080, metadata !18, metadata !"n1", metadata !1, i32 883, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!38 = metadata !{i32 590080, metadata !18, metadata !"n2", metadata !1, i32 883, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!39 = metadata !{i32 590080, metadata !18, metadata !"q0", metadata !1, i32 884, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!40 = metadata !{i32 590080, metadata !18, metadata !"q1", metadata !1, i32 884, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!41 = metadata !{i32 590080, metadata !18, metadata !"b", metadata !1, i32 885, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!42 = metadata !{i32 590080, metadata !18, metadata !"bm", metadata !1, i32 885, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!43 = metadata !{i32 590080, metadata !18, metadata !"ww", metadata !1, i32 1086, metadata !19, i32 0} ; [ DW_TAG_auto_variable ]
!44 = metadata !{i32 590080, metadata !45, metadata !"__xr", metadata !1, i32 933, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!45 = metadata !{i32 589835, metadata !18, i32 933, i32 0, metadata !1, i32 1} ; [ DW_TAG_lexical_block ]
!46 = metadata !{i32 590080, metadata !45, metadata !"__a", metadata !1, i32 933, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!47 = metadata !{i32 590080, metadata !48, metadata !"__d1", metadata !1, i32 945, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!48 = metadata !{i32 589835, metadata !18, i32 945, i32 0, metadata !1, i32 2} ; [ DW_TAG_lexical_block ]
!49 = metadata !{i32 590080, metadata !48, metadata !"__d0", metadata !1, i32 945, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!50 = metadata !{i32 590080, metadata !48, metadata !"__q1", metadata !1, i32 945, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!51 = metadata !{i32 590080, metadata !48, metadata !"__q0", metadata !1, i32 945, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!52 = metadata !{i32 590080, metadata !48, metadata !"__r1", metadata !1, i32 945, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!53 = metadata !{i32 590080, metadata !48, metadata !"__r0", metadata !1, i32 945, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!54 = metadata !{i32 590080, metadata !48, metadata !"__m", metadata !1, i32 945, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!55 = metadata !{i32 590080, metadata !56, metadata !"__xr", metadata !1, i32 957, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!56 = metadata !{i32 589835, metadata !18, i32 957, i32 0, metadata !1, i32 3} ; [ DW_TAG_lexical_block ]
!57 = metadata !{i32 590080, metadata !56, metadata !"__a", metadata !1, i32 957, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!58 = metadata !{i32 590080, metadata !59, metadata !"__d1", metadata !1, i32 982, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!59 = metadata !{i32 589835, metadata !18, i32 982, i32 0, metadata !1, i32 4} ; [ DW_TAG_lexical_block ]
!60 = metadata !{i32 590080, metadata !59, metadata !"__d0", metadata !1, i32 982, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!61 = metadata !{i32 590080, metadata !59, metadata !"__q1", metadata !1, i32 982, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!62 = metadata !{i32 590080, metadata !59, metadata !"__q0", metadata !1, i32 982, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!63 = metadata !{i32 590080, metadata !59, metadata !"__r1", metadata !1, i32 982, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!64 = metadata !{i32 590080, metadata !59, metadata !"__r0", metadata !1, i32 982, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!65 = metadata !{i32 590080, metadata !59, metadata !"__m", metadata !1, i32 982, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!66 = metadata !{i32 590080, metadata !67, metadata !"__d1", metadata !1, i32 987, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!67 = metadata !{i32 589835, metadata !18, i32 987, i32 0, metadata !1, i32 5} ; [ DW_TAG_lexical_block ]
!68 = metadata !{i32 590080, metadata !67, metadata !"__d0", metadata !1, i32 987, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!69 = metadata !{i32 590080, metadata !67, metadata !"__q1", metadata !1, i32 987, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!70 = metadata !{i32 590080, metadata !67, metadata !"__q0", metadata !1, i32 987, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!71 = metadata !{i32 590080, metadata !67, metadata !"__r1", metadata !1, i32 987, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!72 = metadata !{i32 590080, metadata !67, metadata !"__r0", metadata !1, i32 987, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!73 = metadata !{i32 590080, metadata !67, metadata !"__m", metadata !1, i32 987, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!74 = metadata !{i32 590080, metadata !75, metadata !"__xr", metadata !1, i32 1022, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!75 = metadata !{i32 589835, metadata !18, i32 1022, i32 0, metadata !1, i32 6} ; [ DW_TAG_lexical_block ]
!76 = metadata !{i32 590080, metadata !75, metadata !"__a", metadata !1, i32 1022, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!77 = metadata !{i32 590080, metadata !78, metadata !"__x", metadata !1, i32 1036, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!78 = metadata !{i32 589835, metadata !18, i32 1036, i32 0, metadata !1, i32 7} ; [ DW_TAG_lexical_block ]
!79 = metadata !{i32 590080, metadata !80, metadata !"m1", metadata !1, i32 1052, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!80 = metadata !{i32 589835, metadata !18, i32 1055, i32 0, metadata !1, i32 8} ; [ DW_TAG_lexical_block ]
!81 = metadata !{i32 590080, metadata !80, metadata !"m0", metadata !1, i32 1052, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!82 = metadata !{i32 590080, metadata !83, metadata !"__d1", metadata !1, i32 1063, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!83 = metadata !{i32 589835, metadata !80, i32 1063, i32 0, metadata !1, i32 9} ; [ DW_TAG_lexical_block ]
!84 = metadata !{i32 590080, metadata !83, metadata !"__d0", metadata !1, i32 1063, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!85 = metadata !{i32 590080, metadata !83, metadata !"__q1", metadata !1, i32 1063, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!86 = metadata !{i32 590080, metadata !83, metadata !"__q0", metadata !1, i32 1063, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!87 = metadata !{i32 590080, metadata !83, metadata !"__r1", metadata !1, i32 1063, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!88 = metadata !{i32 590080, metadata !83, metadata !"__r0", metadata !1, i32 1063, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!89 = metadata !{i32 590080, metadata !83, metadata !"__m", metadata !1, i32 1063, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!90 = metadata !{i32 590080, metadata !91, metadata !"__x0", metadata !1, i32 1064, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!91 = metadata !{i32 589835, metadata !80, i32 1064, i32 0, metadata !1, i32 10} ; [ DW_TAG_lexical_block ]
!92 = metadata !{i32 590080, metadata !91, metadata !"__x1", metadata !1, i32 1064, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!93 = metadata !{i32 590080, metadata !91, metadata !"__x2", metadata !1, i32 1064, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!94 = metadata !{i32 590080, metadata !91, metadata !"__x3", metadata !1, i32 1064, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!95 = metadata !{i32 590080, metadata !91, metadata !"__ul", metadata !1, i32 1064, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!96 = metadata !{i32 590080, metadata !91, metadata !"__vl", metadata !1, i32 1064, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!97 = metadata !{i32 590080, metadata !91, metadata !"__uh", metadata !1, i32 1064, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!98 = metadata !{i32 590080, metadata !91, metadata !"__vh", metadata !1, i32 1064, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!99 = metadata !{i32 590080, metadata !100, metadata !"__x", metadata !1, i32 1069, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!100 = metadata !{i32 589835, metadata !80, i32 1069, i32 0, metadata !1, i32 11} ; [ DW_TAG_lexical_block ]
!101 = metadata !{i32 590080, metadata !102, metadata !"__x", metadata !1, i32 1077, metadata !33, i32 0} ; [ DW_TAG_auto_variable ]
!102 = metadata !{i32 589835, metadata !80, i32 1077, i32 0, metadata !1, i32 12} ; [ DW_TAG_lexical_block ]
!103 = metadata !{i32 590081, metadata !9, metadata !"u", metadata !1, i32 1093, metadata !12, i32 0} ; [ DW_TAG_arg_variable ]
!104 = metadata !{i32 590081, metadata !9, metadata !"v", metadata !1, i32 1093, metadata !12, i32 0} ; [ DW_TAG_arg_variable ]
!105 = metadata !{i32 590080, metadata !106, metadata !"c", metadata !1, i32 1095, metadata !107, i32 0} ; [ DW_TAG_auto_variable ]
!106 = metadata !{i32 589835, metadata !9, i32 1094, i32 0, metadata !1, i32 13} ; [ DW_TAG_lexical_block ]
!107 = metadata !{i32 589846, metadata !6, metadata !"word_type", metadata !6, i32 424, i64 0, i64 0, i64 0, i32 0, metadata !27} ; [ DW_TAG_typedef ]
!108 = metadata !{i32 590080, metadata !106, metadata !"uu", metadata !1, i32 1096, metadata !19, i32 0} ; [ DW_TAG_auto_variable ]
!109 = metadata !{i32 590080, metadata !106, metadata !"vv", metadata !1, i32 1097, metadata !19, i32 0} ; [ DW_TAG_auto_variable ]
!110 = metadata !{i32 590080, metadata !106, metadata !"w", metadata !1, i32 1098, metadata !12, i32 0} ; [ DW_TAG_auto_variable ]
!111 = metadata !{i32 1093, i32 0, metadata !9, null}
!112 = metadata !{i32 1096, i32 0, metadata !106, null}
!113 = metadata !{i32 1097, i32 0, metadata !106, null}
!114 = metadata !{i64 0}
!115 = metadata !{i32 1095, i32 0, metadata !106, null}
!116 = metadata !{i32 1100, i32 0, metadata !106, null}
!117 = metadata !{i64 -1}                         
!118 = metadata !{i32 1101, i32 0, metadata !106, null}
!119 = metadata !{i32 1103, i32 0, metadata !106, null}
!120 = metadata !{i32 1104, i32 0, metadata !106, null}
!121 = metadata !{i32 878, i32 0, metadata !0, metadata !122}
!122 = metadata !{i32 1107, i32 0, metadata !106, null}
!123 = metadata !{i128* null}
!124 = metadata !{i32 880, i32 0, metadata !18, metadata !122}
!125 = metadata !{i32 881, i32 0, metadata !18, metadata !122}
!126 = metadata !{i32 882, i32 0, metadata !18, metadata !122}
!127 = metadata !{i32 1086, i32 0, metadata !18, metadata !122}
!128 = metadata !{i32 887, i32 0, metadata !18, metadata !122}
!129 = metadata !{i32 888, i32 0, metadata !18, metadata !122}
!130 = metadata !{i32 889, i32 0, metadata !18, metadata !122}
!131 = metadata !{i32 890, i32 0, metadata !18, metadata !122}
!132 = metadata !{i32 927, i32 0, metadata !18, metadata !122}
!133 = metadata !{i32 929, i32 0, metadata !18, metadata !122}
!134 = metadata !{i32 933, i32 0, metadata !45, metadata !122}
!135 = metadata !{null}
!136 = metadata !{i32 935, i32 0, metadata !18, metadata !122}
!137 = metadata !{i32 940, i32 0, metadata !18, metadata !122}
!138 = metadata !{i32 941, i32 0, metadata !18, metadata !122}
!139 = metadata !{i32 942, i32 0, metadata !18, metadata !122}
!140 = metadata !{i32 945, i32 0, metadata !48, metadata !122}
!141 = metadata !{i32 946, i32 0, metadata !18, metadata !122}
!142 = metadata !{i32 954, i32 0, metadata !18, metadata !122}
!143 = metadata !{i32 955, i32 0, metadata !18, metadata !122}
!144 = metadata !{i32 957, i32 0, metadata !56, metadata !122}
!145 = metadata !{i64 56}
!146 = metadata !{i32 959, i32 0, metadata !18, metadata !122}
!147 = metadata !{i32 968, i32 0, metadata !18, metadata !122}
!148 = metadata !{i64 1}
!149 = metadata !{i32 969, i32 0, metadata !18, metadata !122}
!150 = metadata !{i32 975, i32 0, metadata !18, metadata !122}
!151 = metadata !{i32 977, i32 0, metadata !18, metadata !122}
!152 = metadata !{i32 978, i32 0, metadata !18, metadata !122}
!153 = metadata !{i32 979, i32 0, metadata !18, metadata !122}
!154 = metadata !{i32 980, i32 0, metadata !18, metadata !122}
!155 = metadata !{i32 982, i32 0, metadata !59, metadata !122}
!156 = metadata !{i32 987, i32 0, metadata !67, metadata !122}
!157 = metadata !{i32 1003, i32 0, metadata !18, metadata !122}
!158 = metadata !{i32 1022, i32 0, metadata !75, metadata !122}
!159 = metadata !{i32 1023, i32 0, metadata !18, metadata !122}
!160 = metadata !{i32 1033, i32 0, metadata !18, metadata !122}
!161 = metadata !{i32 1041, i32 0, metadata !18, metadata !122}
!162 = metadata !{i32 1055, i32 0, metadata !80, metadata !122}
!163 = metadata !{i32 1057, i32 0, metadata !80, metadata !122}
!164 = metadata !{i32 1058, i32 0, metadata !80, metadata !122}
!165 = metadata !{i32 1059, i32 0, metadata !80, metadata !122}
!166 = metadata !{i32 1060, i32 0, metadata !80, metadata !122}
!167 = metadata !{i32 1061, i32 0, metadata !80, metadata !122}
!168 = metadata !{i32 1063, i32 0, metadata !83, metadata !122}
!169 = metadata !{i32 1064, i32 0, metadata !91, metadata !122}
!170 = metadata !{i32 1066, i32 0, metadata !80, metadata !122}
!171 = metadata !{i32 1068, i32 0, metadata !80, metadata !122}
!172 = metadata !{i32 1069, i32 0, metadata !100, metadata !122}
!173 = metadata !{i32 1108, i32 0, metadata !106, null}
!174 = metadata !{i32 1109, i32 0, metadata !106, null}
!175 = metadata !{i32 1111, i32 0, metadata !106, null}
