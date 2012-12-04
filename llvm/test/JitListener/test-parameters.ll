; RUN: llvm-jitlistener %s | FileCheck %s

; CHECK: Method load [1]: _Z15test_parametersPfPA2_dR11char_structPPitm, Size = 170
; CHECK: Method load [2]: _Z3foov, Size = 3
; CHECK: Method load [3]: main, Size = 146
; CHECK: Method unload [1]
; CHECK: Method unload [2]
; CHECK: Method unload [3]

; ModuleID = 'test-parameters.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.char_struct = type { i8, [2 x i8] }

@compound_char = global %struct.char_struct zeroinitializer, align 1
@_ZZ4mainE1d = private unnamed_addr constant [2 x [2 x double]] [[2 x double] [double 0.000000e+00, double 1.000000e+00], [2 x double] [double 2.000000e+00, double 3.000000e+00]], align 16

define i32 @_Z3foov() nounwind uwtable {
entry:
  ret i32 0, !dbg !32
}

define double @_Z15test_parametersPfPA2_dR11char_structPPitm(float* %pf, [2 x double]* %ppd, %struct.char_struct* %s, i32** %ppn, i16 zeroext %us, i64 %l) nounwind uwtable {
entry:
  %pf.addr = alloca float*, align 8
  %ppd.addr = alloca [2 x double]*, align 8
  %s.addr = alloca %struct.char_struct*, align 8
  %ppn.addr = alloca i32**, align 8
  %us.addr = alloca i16, align 2
  %l.addr = alloca i64, align 8
  %result = alloca double, align 8
  store float* %pf, float** %pf.addr, align 8
  call void @llvm.dbg.declare(metadata !{float** %pf.addr}, metadata !34), !dbg !37
  store [2 x double]* %ppd, [2 x double]** %ppd.addr, align 8
  call void @llvm.dbg.declare(metadata !{[2 x double]** %ppd.addr}, metadata !38), !dbg !41
  store %struct.char_struct* %s, %struct.char_struct** %s.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.char_struct** %s.addr}, metadata !42), !dbg !44
  store i32** %ppn, i32*** %ppn.addr, align 8
  call void @llvm.dbg.declare(metadata !{i32*** %ppn.addr}, metadata !45), !dbg !48
  store i16 %us, i16* %us.addr, align 2
  call void @llvm.dbg.declare(metadata !{i16* %us.addr}, metadata !49), !dbg !51
  store i64 %l, i64* %l.addr, align 8
  call void @llvm.dbg.declare(metadata !{i64* %l.addr}, metadata !52), !dbg !55
  call void @llvm.dbg.declare(metadata !{double* %result}, metadata !56), !dbg !58
  %0 = load float** %pf.addr, align 8, !dbg !59
  %arrayidx = getelementptr inbounds float* %0, i64 0, !dbg !59
  %1 = load float* %arrayidx, !dbg !59
  %conv = fpext float %1 to double, !dbg !59
  %2 = load [2 x double]** %ppd.addr, align 8, !dbg !59
  %arrayidx1 = getelementptr inbounds [2 x double]* %2, i64 1, !dbg !59
  %arrayidx2 = getelementptr inbounds [2 x double]* %arrayidx1, i32 0, i64 1, !dbg !59
  %3 = load double* %arrayidx2, !dbg !59
  %mul = fmul double %conv, %3, !dbg !59
  %4 = load %struct.char_struct** %s.addr, !dbg !59
  %c = getelementptr inbounds %struct.char_struct* %4, i32 0, i32 0, !dbg !59
  %5 = load i8* %c, align 1, !dbg !59
  %conv3 = sext i8 %5 to i32, !dbg !59
  %conv4 = sitofp i32 %conv3 to double, !dbg !59
  %mul5 = fmul double %mul, %conv4, !dbg !59
  %6 = load i16* %us.addr, align 2, !dbg !59
  %conv6 = zext i16 %6 to i32, !dbg !59
  %conv7 = sitofp i32 %conv6 to double, !dbg !59
  %mul8 = fmul double %mul5, %conv7, !dbg !59
  %7 = load i64* %l.addr, align 8, !dbg !59
  %conv9 = uitofp i64 %7 to double, !dbg !59
  %mul10 = fmul double %mul8, %conv9, !dbg !59
  %call = call i32 @_Z3foov(), !dbg !60
  %conv11 = sitofp i32 %call to double, !dbg !60
  %add = fadd double %mul10, %conv11, !dbg !60
  store double %add, double* %result, align 8, !dbg !60
  %8 = load double* %result, align 8, !dbg !61
  ret double %8, !dbg !61
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define i32 @main(i32 %argc, i8** %argv) nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %s = alloca %struct.char_struct, align 1
  %f = alloca float, align 4
  %d = alloca [2 x [2 x double]], align 16
  %result = alloca double, align 8
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %argc.addr}, metadata !62), !dbg !63
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8*** %argv.addr}, metadata !64), !dbg !67
  call void @llvm.dbg.declare(metadata !{%struct.char_struct* %s}, metadata !68), !dbg !70
  call void @llvm.dbg.declare(metadata !{float* %f}, metadata !71), !dbg !72
  store float 0.000000e+00, float* %f, align 4, !dbg !73
  call void @llvm.dbg.declare(metadata !{[2 x [2 x double]]* %d}, metadata !74), !dbg !77
  %0 = bitcast [2 x [2 x double]]* %d to i8*, !dbg !78
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast ([2 x [2 x double]]* @_ZZ4mainE1d to i8*), i64 32, i32 16, i1 false), !dbg !78
  %c = getelementptr inbounds %struct.char_struct* %s, i32 0, i32 0, !dbg !79
  store i8 97, i8* %c, align 1, !dbg !79
  %c2 = getelementptr inbounds %struct.char_struct* %s, i32 0, i32 1, !dbg !80
  %arrayidx = getelementptr inbounds [2 x i8]* %c2, i32 0, i64 0, !dbg !80
  store i8 48, i8* %arrayidx, align 1, !dbg !80
  %c21 = getelementptr inbounds %struct.char_struct* %s, i32 0, i32 1, !dbg !81
  %arrayidx2 = getelementptr inbounds [2 x i8]* %c21, i32 0, i64 1, !dbg !81
  store i8 49, i8* %arrayidx2, align 1, !dbg !81
  call void @llvm.dbg.declare(metadata !{double* %result}, metadata !82), !dbg !83
  %arraydecay = getelementptr inbounds [2 x [2 x double]]* %d, i32 0, i32 0, !dbg !84
  %call = call double @_Z15test_parametersPfPA2_dR11char_structPPitm(float* %f, [2 x double]* %arraydecay, %struct.char_struct* %s, i32** null, i16 zeroext 10, i64 42), !dbg !84
  store double %call, double* %result, align 8, !dbg !84
  %1 = load double* %result, align 8, !dbg !85
  %cmp = fcmp oeq double %1, 0.000000e+00, !dbg !85
  %cond = select i1 %cmp, i32 0, i32 -1, !dbg !85
  ret i32 %cond, !dbg !85
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 4, metadata !"test-parameters.cpp", metadata !"/home/athirumurthi/dev/opencl-mc/build/RH64/Debug/backend/llvm", metadata !"clang version 3.0 (branches/release_30 36797)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !17} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5, metadata !12, metadata !16}
!5 = metadata !{i32 720942, i32 0, metadata !6, metadata !"foo", metadata !"foo", metadata !"_Z3foov", metadata !6, i32 28, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, i32 ()* @_Z3foov, null, null, metadata !10} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !"test-parameters.cpp", metadata !"/home/athirumurthi/dev/opencl-mc/build/RH64/Debug/backend/llvm", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!12 = metadata !{i32 720942, i32 0, metadata !6, metadata !"test_parameters", metadata !"test_parameters", metadata !"_Z15test_parametersPfPA2_dR11char_structPPitm", metadata !6, i32 33, metadata !13, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, double (float*, [2 x double]*, %struct.char_struct*, i32**, i16, i64)* @_Z15test_parametersPfPA2_dR11char_structPPitm, null, null, metadata !10} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !14, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!14 = metadata !{metadata !15}
!15 = metadata !{i32 720932, null, metadata !"double", null, i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!16 = metadata !{i32 720942, i32 0, metadata !6, metadata !"main", metadata !"main", metadata !"", metadata !6, i32 39, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, i32 (i32, i8**)* @main, null, null, metadata !10} ; [ DW_TAG_subprogram ]
!17 = metadata !{metadata !18}
!18 = metadata !{metadata !19}
!19 = metadata !{i32 720948, i32 0, null, metadata !"compound_char", metadata !"compound_char", metadata !"", metadata !6, i32 25, metadata !20, i32 0, i32 1, %struct.char_struct* @compound_char} ; [ DW_TAG_variable ]
!20 = metadata !{i32 720898, null, metadata !"char_struct", metadata !6, i32 22, i64 24, i64 8, i32 0, i32 0, null, metadata !21, i32 0, null, null} ; [ DW_TAG_class_type ]
!21 = metadata !{metadata !22, metadata !24, metadata !28}
!22 = metadata !{i32 720909, metadata !20, metadata !"c", metadata !6, i32 23, i64 8, i64 8, i64 0, i32 0, metadata !23} ; [ DW_TAG_member ]
!23 = metadata !{i32 720932, null, metadata !"char", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!24 = metadata !{i32 720909, metadata !20, metadata !"c2", metadata !6, i32 24, i64 16, i64 8, i64 8, i32 0, metadata !25} ; [ DW_TAG_member ]
!25 = metadata !{i32 720897, null, metadata !"", null, i32 0, i64 16, i64 8, i32 0, i32 0, metadata !23, metadata !26, i32 0, i32 0} ; [ DW_TAG_array_type ]
!26 = metadata !{metadata !27}
!27 = metadata !{i32 720929, i64 0, i64 2}        ; [ DW_TAG_subrange_type ]
!28 = metadata !{i32 720942, i32 0, metadata !20, metadata !"char_struct", metadata !"char_struct", metadata !"", metadata !6, i32 22, metadata !29, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !10} ; [ DW_TAG_subprogram ]
!29 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !30, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!30 = metadata !{null, metadata !31}
!31 = metadata !{i32 720911, i32 0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !20} ; [ DW_TAG_pointer_type ]
!32 = metadata !{i32 29, i32 3, metadata !33, null}
!33 = metadata !{i32 720907, metadata !5, i32 28, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!34 = metadata !{i32 721153, metadata !12, metadata !"pf", metadata !6, i32 16777248, metadata !35, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!35 = metadata !{i32 720911, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !36} ; [ DW_TAG_pointer_type ]
!36 = metadata !{i32 720932, null, metadata !"float", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!37 = metadata !{i32 32, i32 31, metadata !12, null}
!38 = metadata !{i32 721153, metadata !12, metadata !"ppd", metadata !6, i32 33554464, metadata !39, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!39 = metadata !{i32 720911, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !40} ; [ DW_TAG_pointer_type ]
!40 = metadata !{i32 720897, null, metadata !"", null, i32 0, i64 128, i64 64, i32 0, i32 0, metadata !15, metadata !26, i32 0, i32 0} ; [ DW_TAG_array_type ]
!41 = metadata !{i32 32, i32 42, metadata !12, null}
!42 = metadata !{i32 721153, metadata !12, metadata !"s", metadata !6, i32 50331680, metadata !43, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!43 = metadata !{i32 720912, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !20} ; [ DW_TAG_reference_type ]
!44 = metadata !{i32 32, i32 72, metadata !12, null}
!45 = metadata !{i32 721153, metadata !12, metadata !"ppn", metadata !6, i32 67108896, metadata !46, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!46 = metadata !{i32 720911, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !47} ; [ DW_TAG_pointer_type ]
!47 = metadata !{i32 720911, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ]
!48 = metadata !{i32 32, i32 81, metadata !12, null}
!49 = metadata !{i32 721153, metadata !12, metadata !"us", metadata !6, i32 83886112, metadata !50, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!50 = metadata !{i32 720932, null, metadata !"unsigned short", null, i32 0, i64 16, i64 16, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!51 = metadata !{i32 32, i32 105, metadata !12, null}
!52 = metadata !{i32 721153, metadata !12, metadata !"l", metadata !6, i32 100663328, metadata !53, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!53 = metadata !{i32 720934, null, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !54} ; [ DW_TAG_const_type ]
!54 = metadata !{i32 720932, null, metadata !"long unsigned int", null, i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!55 = metadata !{i32 32, i32 135, metadata !12, null}
!56 = metadata !{i32 721152, metadata !57, metadata !"result", metadata !6, i32 34, metadata !15, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!57 = metadata !{i32 720907, metadata !12, i32 33, i32 1, metadata !6, i32 1} ; [ DW_TAG_lexical_block ]
!58 = metadata !{i32 34, i32 10, metadata !57, null}
!59 = metadata !{i32 34, i32 59, metadata !57, null}
!60 = metadata !{i32 34, i32 54, metadata !57, null}
!61 = metadata !{i32 35, i32 3, metadata !57, null}
!62 = metadata !{i32 721153, metadata !16, metadata !"argc", metadata !6, i32 16777254, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!63 = metadata !{i32 38, i32 14, metadata !16, null}
!64 = metadata !{i32 721153, metadata !16, metadata !"argv", metadata !6, i32 33554470, metadata !65, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!65 = metadata !{i32 720911, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !66} ; [ DW_TAG_pointer_type ]
!66 = metadata !{i32 720911, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !23} ; [ DW_TAG_pointer_type ]
!67 = metadata !{i32 38, i32 26, metadata !16, null}
!68 = metadata !{i32 721152, metadata !69, metadata !"s", metadata !6, i32 40, metadata !20, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!69 = metadata !{i32 720907, metadata !16, i32 39, i32 1, metadata !6, i32 2} ; [ DW_TAG_lexical_block ]
!70 = metadata !{i32 40, i32 22, metadata !69, null}
!71 = metadata !{i32 721152, metadata !69, metadata !"f", metadata !6, i32 41, metadata !36, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!72 = metadata !{i32 41, i32 9, metadata !69, null}
!73 = metadata !{i32 41, i32 16, metadata !69, null}
!74 = metadata !{i32 721152, metadata !69, metadata !"d", metadata !6, i32 42, metadata !75, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!75 = metadata !{i32 720897, null, metadata !"", null, i32 0, i64 256, i64 64, i32 0, i32 0, metadata !15, metadata !76, i32 0, i32 0} ; [ DW_TAG_array_type ]
!76 = metadata !{metadata !27, metadata !27}
!77 = metadata !{i32 42, i32 10, metadata !69, null}
!78 = metadata !{i32 42, i32 38, metadata !69, null}
!79 = metadata !{i32 44, i32 3, metadata !69, null}
!80 = metadata !{i32 45, i32 3, metadata !69, null}
!81 = metadata !{i32 46, i32 3, metadata !69, null}
!82 = metadata !{i32 721152, metadata !69, metadata !"result", metadata !6, i32 48, metadata !15, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!83 = metadata !{i32 48, i32 10, metadata !69, null}
!84 = metadata !{i32 48, i32 19, metadata !69, null}
!85 = metadata !{i32 49, i32 3, metadata !69, null}
