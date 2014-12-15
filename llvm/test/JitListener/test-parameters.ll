; RUN: llvm-jitlistener %s | FileCheck %s

; CHECK: Method load [1]: _Z15test_parametersPfPA2_dR11char_structPPitm, Size = 170
; CHECK:   Line info @ 0: test-parameters.cpp, line 33
; CHECK:   Line info @ 35: test-parameters.cpp, line 34
; CHECK:   Line info @ 165: test-parameters.cpp, line 35
; CHECK: Method load [2]: _Z3foov, Size = 3
; CHECK:   Line info @ 0: test-parameters.cpp, line 28
; CHECK:   Line info @ 2: test-parameters.cpp, line 29
; CHECK: Method load [3]: main, Size = 146
; CHECK:   Line info @ 0: test-parameters.cpp, line 39
; CHECK:   Line info @ 21: test-parameters.cpp, line 41
; CHECK:   Line info @ 39: test-parameters.cpp, line 42
; CHECK:   Line info @ 60: test-parameters.cpp, line 44
; CHECK:   Line info @ 80: test-parameters.cpp, line 48
; CHECK:   Line info @ 90: test-parameters.cpp, line 45
; CHECK:   Line info @ 95: test-parameters.cpp, line 46
; CHECK:   Line info @ 114: test-parameters.cpp, line 48
; CHECK:   Line info @ 141: test-parameters.cpp, line 49
; CHECK:   Line info @ 146: test-parameters.cpp, line 49
; CHECK: Method unload [1]
; CHECK: Method unload [2]
; CHECK: Method unload [3]

; ModuleID = 'test-parameters.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.char_struct = type { i8, [2 x i8] }

@compound_char = global %struct.char_struct zeroinitializer, align 1
@_ZZ4mainE1d = private unnamed_addr constant [2 x [2 x double]] [[2 x double] [double 0.000000e+00, double 1.000000e+00], [2 x double] [double 2.000000e+00, double 3.000000e+00]], align 16

define i32 @_Z3foov() nounwind uwtable {
entry:
  ret i32 0, !dbg !46
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
  call void @llvm.dbg.declare(metadata float** %pf.addr, metadata !48, metadata !{!"0x102"}), !dbg !49
  store [2 x double]* %ppd, [2 x double]** %ppd.addr, align 8
  call void @llvm.dbg.declare(metadata [2 x double]** %ppd.addr, metadata !50, metadata !{!"0x102"}), !dbg !49
  store %struct.char_struct* %s, %struct.char_struct** %s.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.char_struct** %s.addr, metadata !51, metadata !{!"0x102"}), !dbg !49
  store i32** %ppn, i32*** %ppn.addr, align 8
  call void @llvm.dbg.declare(metadata i32*** %ppn.addr, metadata !52, metadata !{!"0x102"}), !dbg !49
  store i16 %us, i16* %us.addr, align 2
  call void @llvm.dbg.declare(metadata i16* %us.addr, metadata !53, metadata !{!"0x102"}), !dbg !49
  store i64 %l, i64* %l.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %l.addr, metadata !54, metadata !{!"0x102"}), !dbg !49
  call void @llvm.dbg.declare(metadata double* %result, metadata !55, metadata !{!"0x102"}), !dbg !57
  %0 = load float** %pf.addr, align 8, !dbg !57
  %arrayidx = getelementptr inbounds float* %0, i64 0, !dbg !57
  %1 = load float* %arrayidx, align 4, !dbg !57
  %conv = fpext float %1 to double, !dbg !57
  %2 = load [2 x double]** %ppd.addr, align 8, !dbg !57
  %arrayidx1 = getelementptr inbounds [2 x double]* %2, i64 1, !dbg !57
  %arrayidx2 = getelementptr inbounds [2 x double]* %arrayidx1, i32 0, i64 1, !dbg !57
  %3 = load double* %arrayidx2, align 8, !dbg !57
  %mul = fmul double %conv, %3, !dbg !57
  %4 = load %struct.char_struct** %s.addr, align 8, !dbg !57
  %c = getelementptr inbounds %struct.char_struct* %4, i32 0, i32 0, !dbg !57
  %5 = load i8* %c, align 1, !dbg !57
  %conv3 = sext i8 %5 to i32, !dbg !57
  %conv4 = sitofp i32 %conv3 to double, !dbg !57
  %mul5 = fmul double %mul, %conv4, !dbg !57
  %6 = load i16* %us.addr, align 2, !dbg !57
  %conv6 = zext i16 %6 to i32, !dbg !57
  %conv7 = sitofp i32 %conv6 to double, !dbg !57
  %mul8 = fmul double %mul5, %conv7, !dbg !57
  %7 = load i64* %l.addr, align 8, !dbg !57
  %conv9 = uitofp i64 %7 to double, !dbg !57
  %mul10 = fmul double %mul8, %conv9, !dbg !57
  %call = call i32 @_Z3foov(), !dbg !57
  %conv11 = sitofp i32 %call to double, !dbg !57
  %add = fadd double %mul10, %conv11, !dbg !57
  store double %add, double* %result, align 8, !dbg !57
  %8 = load double* %result, align 8, !dbg !58
  ret double %8, !dbg !58
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

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
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !59, metadata !{!"0x102"}), !dbg !60
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !61, metadata !{!"0x102"}), !dbg !60
  call void @llvm.dbg.declare(metadata %struct.char_struct* %s, metadata !62, metadata !{!"0x102"}), !dbg !64
  call void @llvm.dbg.declare(metadata float* %f, metadata !65, metadata !{!"0x102"}), !dbg !66
  store float 0.000000e+00, float* %f, align 4, !dbg !66
  call void @llvm.dbg.declare(metadata [2 x [2 x double]]* %d, metadata !67, metadata !{!"0x102"}), !dbg !70
  %0 = bitcast [2 x [2 x double]]* %d to i8*, !dbg !70
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast ([2 x [2 x double]]* @_ZZ4mainE1d to i8*), i64 32, i32 16, i1 false), !dbg !70
  %c = getelementptr inbounds %struct.char_struct* %s, i32 0, i32 0, !dbg !71
  store i8 97, i8* %c, align 1, !dbg !71
  %c2 = getelementptr inbounds %struct.char_struct* %s, i32 0, i32 1, !dbg !72
  %arrayidx = getelementptr inbounds [2 x i8]* %c2, i32 0, i64 0, !dbg !72
  store i8 48, i8* %arrayidx, align 1, !dbg !72
  %c21 = getelementptr inbounds %struct.char_struct* %s, i32 0, i32 1, !dbg !73
  %arrayidx2 = getelementptr inbounds [2 x i8]* %c21, i32 0, i64 1, !dbg !73
  store i8 49, i8* %arrayidx2, align 1, !dbg !73
  call void @llvm.dbg.declare(metadata double* %result, metadata !74, metadata !{!"0x102"}), !dbg !75
  %arraydecay = getelementptr inbounds [2 x [2 x double]]* %d, i32 0, i32 0, !dbg !75
  %call = call double @_Z15test_parametersPfPA2_dR11char_structPPitm(float* %f, [2 x double]* %arraydecay, %struct.char_struct* %s, i32** null, i16 zeroext 10, i64 42), !dbg !75
  store double %call, double* %result, align 8, !dbg !75
  %1 = load double* %result, align 8, !dbg !76
  %cmp = fcmp oeq double %1, 0.000000e+00, !dbg !76
  %cond = select i1 %cmp, i32 0, i32 -1, !dbg !76
  ret i32 %cond, !dbg !76
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!78}

!0 = !{!"0x11\004\00clang version 3.3 (ssh://akaylor@git-amr-1.devtools.intel.com:29418/ssg_llvm-clang2 gitosis@miro.kw.intel.com:clang.git 39450d0469e0d5589ad39fd0b20b5742750619a0) (ssh://akaylor@git-amr-1.devtools.intel.com:29418/ssg_llvm-llvm gitosis@miro.kw.intel.com:llvm.git 376642ed620ecae05b68c7bc81f79aeb2065abe0)\001\00\000\00\000", !77, !1, !1, !3, !43, null} ; [ DW_TAG_compile_unit ] [/home/akaylor/dev/test-parameters.cpp] [DW_LANG_C_plus_plus]
!1 = !{i32 0}
!3 = !{!5, !10, !38}
!5 = !{!"0x2e\00foo\00foo\00_Z3foov\0027\000\001\000\006\00256\000\0028", !77, !6, !7, null, i32 ()* @_Z3foov, null, null, !1} ; [ DW_TAG_subprogram ] [line 27] [def] [scope 28] [foo]
!6 = !{!"0x29", !77} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, !"", null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0x2e\00test_parameters\00test_parameters\00_Z15test_parametersPfPA2_dR11char_structPPitm\0032\000\001\000\006\00256\000\0033", !77, !6, !11, null, double (float*, [2 x double]*, %struct.char_struct*, i32**, i16, i64)* @_Z15test_parametersPfPA2_dR11char_structPPitm, null, null, !1} ; [ DW_TAG_subprogram ] [line 32] [def] [scope 33] [test_parameters]
!11 = !{!"0x15\00\000\000\000\000\000\000", i32 0, !"", null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!13, !14, !16, !20, !33, !35, !36}
!13 = !{!"0x24\00double\000\0064\0064\000\000\004", null, null} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!14 = !{!"0xf\00\000\0064\0064\000\000", null, !"", !15} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from float]
!15 = !{!"0x24\00float\000\0032\0032\000\000\004", null, null} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!16 = !{!"0xf\00\000\0064\0064\000\000", null, !"", !17} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!17 = !{!"0x1\00\000\00128\0064\000\000", null, !"", !13, !18, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 64, offset 0] [from double]
!18 = !{!19}
!19 = !{!"0x21\000\002"}        ; [ DW_TAG_subrange_type ] [0, 1]
!20 = !{!"0x10\00\000\000\000\000\000", null, null, !21} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from char_struct]
!21 = !{!"0x13\00char_struct\0022\0024\008\000\000\000", !77, null, null, !22, null, null, null} ; [ DW_TAG_structure_type ] [char_struct] [line 22, size 24, align 8, offset 0] [def] [from ]
!22 = !{!23, !25, !27}
!23 = !{!"0xd\00c\0023\008\008\000\000", !77, !21, !24} ; [ DW_TAG_member ] [c] [line 23, size 8, align 8, offset 0] [from char]
!24 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!25 = !{!"0xd\00c2\0024\0016\008\008\000", !77, !21, !26} ; [ DW_TAG_member ] [c2] [line 24, size 16, align 8, offset 8] [from ]
!26 = !{!"0x1\00\000\0016\008\000\000", null, !"", !24, !18, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 16, align 8, offset 0] [from char]
!27 = !{!"0x2e\00char_struct\00char_struct\00\0022\000\000\000\006\00320\000\0022", !77, !21, !28, null, null, null, i32 0, !31} ; [ DW_TAG_subprogram ] [line 22] [char_struct]
!28 = !{!"0x15\00\000\000\000\000\000\000", i32 0, !"", null, !29, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!29 = !{null, !30}
!30 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, !"", !21} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char_struct]
!31 = !{!32}
!32 = !{!"0x24"}                      ; [ DW_TAG_base_type ] [line 0, size 0, align 0, offset 0]
!33 = !{!"0xf\00\000\0064\0064\000\000", null, !"", !34} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!34 = !{!"0xf\00\000\0064\0064\000\000", null, !"", !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!35 = !{!"0x24\00unsigned short\000\0016\0016\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned short] [line 0, size 16, align 16, offset 0, enc DW_ATE_unsigned]
!36 = !{!"0x26\00\000\000\000\000\000", null, !"", !37} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from long unsigned int]
!37 = !{!"0x24\00long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ] [long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!38 = !{!"0x2e\00main\00main\00\0038\000\001\000\006\00256\000\0039", !77, !6, !39, null, i32 (i32, i8**)* @main, null, null, !1} ; [ DW_TAG_subprogram ] [line 38] [def] [scope 39] [main]
!39 = !{!"0x15\00\000\000\000\000\000\000", i32 0, !"", null, !40, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!40 = !{!9, !9, !41}
!41 = !{!"0xf\00\000\0064\0064\000\000", null, !"", !42} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!42 = !{!"0xf\00\000\0064\0064\000\000", null, !"", !24} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char]
!43 = !{!45}
!45 = !{!"0x34\00compound_char\00compound_char\00\0025\000\001", null, !6, !21, %struct.char_struct* @compound_char, null} ; [ DW_TAG_variable ] [compound_char] [line 25] [def]
!46 = !{i32 29, i32 0, !47, null}
!47 = !{!"0xb\0028\000\000", !77, !5} ; [ DW_TAG_lexical_block ] [/home/akaylor/dev/test-parameters.cpp]
!48 = !{!"0x101\00pf\0016777248\000", !10, !6, !14} ; [ DW_TAG_arg_variable ] [pf] [line 32]
!49 = !{i32 32, i32 0, !10, null}
!50 = !{!"0x101\00ppd\0033554464\000", !10, !6, !16} ; [ DW_TAG_arg_variable ] [ppd] [line 32]
!51 = !{!"0x101\00s\0050331680\000", !10, !6, !20} ; [ DW_TAG_arg_variable ] [s] [line 32]
!52 = !{!"0x101\00ppn\0067108896\000", !10, !6, !33} ; [ DW_TAG_arg_variable ] [ppn] [line 32]
!53 = !{!"0x101\00us\0083886112\000", !10, !6, !35} ; [ DW_TAG_arg_variable ] [us] [line 32]
!54 = !{!"0x101\00l\00100663328\000", !10, !6, !36} ; [ DW_TAG_arg_variable ] [l] [line 32]
!55 = !{!"0x100\00result\0034\000", !56, !6, !13} ; [ DW_TAG_auto_variable ] [result] [line 34]
!56 = !{!"0xb\0033\000\001", !77, !10} ; [ DW_TAG_lexical_block ] [/home/akaylor/dev/test-parameters.cpp]
!57 = !{i32 34, i32 0, !56, null}
!58 = !{i32 35, i32 0, !56, null}
!59 = !{!"0x101\00argc\0016777254\000", !38, !6, !9} ; [ DW_TAG_arg_variable ] [argc] [line 38]
!60 = !{i32 38, i32 0, !38, null}
!61 = !{!"0x101\00argv\0033554470\000", !38, !6, !41} ; [ DW_TAG_arg_variable ] [argv] [line 38]
!62 = !{!"0x100\00s\0040\000", !63, !6, !21} ; [ DW_TAG_auto_variable ] [s] [line 40]
!63 = !{!"0xb\0039\000\002", !77, !38} ; [ DW_TAG_lexical_block ] [/home/akaylor/dev/test-parameters.cpp]
!64 = !{i32 40, i32 0, !63, null}
!65 = !{!"0x100\00f\0041\000", !63, !6, !15} ; [ DW_TAG_auto_variable ] [f] [line 41]
!66 = !{i32 41, i32 0, !63, null}
!67 = !{!"0x100\00d\0042\000", !63, !6, !68} ; [ DW_TAG_auto_variable ] [d] [line 42]
!68 = !{!"0x1\00\000\00256\0064\000\000", null, !"", !13, !69, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 256, align 64, offset 0] [from double]
!69 = !{!19, !19}
!70 = !{i32 42, i32 0, !63, null}
!71 = !{i32 44, i32 0, !63, null}
!72 = !{i32 45, i32 0, !63, null}
!73 = !{i32 46, i32 0, !63, null}
!74 = !{!"0x100\00result\0048\000", !63, !6, !13} ; [ DW_TAG_auto_variable ] [result] [line 48]
!75 = !{i32 48, i32 0, !63, null}
!76 = !{i32 49, i32 0, !63, null}
!77 = !{!"test-parameters.cpp", !"/home/akaylor/dev"}
!78 = !{i32 1, !"Debug Info Version", i32 2}
