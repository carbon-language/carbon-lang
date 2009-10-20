; RUN: opt -ipsccp <%s -S | FileCheck %s
; PR4313
; the return value of a multiple-return value invoke must not be left undefined
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%0 = type <{ %1, %2, [4 x i8], %3 }>
%1 = type { i32 (...)**, i32, i32, i32, i32, %1*, i8*, i8*, i8* }
%2 = type { i32 }
%3 = type <{ %4, %5 }>
%4 = type { i32 (...)**, i8 }
%5 = type <{ [33 x i8], [191 x i8] }>
%6 = type { %1, %7, %10 }
%7 = type { %8 }
%8 = type { %9 }
%9 = type { i8* }
%10 = type { %11 }
%11 = type { %12 }
%12 = type { i32 (...)** }
%13 = type { %1, %14, %18, %10 }
%14 = type { %15 }
%15 = type { %16 }
%16 = type { %17 }
%17 = type { %8*, %8*, %8* }
%18 = type { %19 }
%19 = type { %20 }
%20 = type { i32*, i32*, i32* }
%21 = type { %22, %22 }
%22 = type { %23 }
%23 = type { %24 }
%24 = type { %25, %26, i64 }
%25 = type <{ i8 }>
%26 = type { i32, %26*, %26*, %26* }
%27 = type { %28, %15, i8* }
%28 = type { %29 }
%29 = type { %30 }
%30 = type { %31*, %31*, %31* }
%31 = type { %32*, %9 }
%32 = type { i32 (...)**, i8*, i8*, i8 }
%33 = type { i64, [12 x i32] }
%34 = type { %35 }
%35 = type { i32, i32, i32, i32, i32, i32, %36 }
%36 = type { %36*, %36* }
%37 = type { i32, %8, %9, %15, %38, %42 }
%38 = type { %39 }
%39 = type { %40 }
%40 = type { %41*, %41*, %41* }
%41 = type { %8, %12*, i32, %12* }
%42 = type { %43 }
%43 = type { %44 }
%44 = type { %37**, %37**, %37** }
%45 = type { i32 (...)**, i8*, i8*, i8*, i32, i8 }
%46 = type { %47, %37*, %12*, %8, %15, %37*, %50 }
%47 = type { %48 }
%48 = type { %49 }
%49 = type { i8*, i8*, i8* }
%50 = type { %51 }
%51 = type { %52 }
%52 = type { %46**, %46**, %46** }
%53 = type { %21*, %54, %63, %63, %22, %22, %22, %22, %22, %22, %37*, %37*, %37*, %72 }
%54 = type { %37*, %22, %55, %59, %18 }
%55 = type { %56 }
%56 = type { %57 }
%57 = type { %58*, %58*, %58* }
%58 = type { %37*, i32, i32 }
%59 = type { %60 }
%60 = type { %61 }
%61 = type { %62*, %62*, %62* }
%62 = type { %37*, %8, %42, %18, i32, i32, i32, %42, %8, %8 }
%63 = type { %64 }
%64 = type { %65 }
%65 = type { %66*, %66*, %66* }
%66 = type { %37*, %8, %8, %8, %8, %67, i32, i8, i8, %68 }
%67 = type { %18, %42, %18, %42 }
%68 = type { %69 }
%69 = type { %70 }
%70 = type { %71*, %71*, %71* }
%71 = type { i32, i32 }
%72 = type { %73 }
%73 = type { %74 }
%74 = type { %75*, %75*, %75* }
%75 = type { %76*, %46*, %46*, %42, i32 }
%76 = type { %77, %78 }
%77 = type { %12, %12* }
%78 = type { %79 }
%79 = type { %80 }
%80 = type { %12**, %12**, %12** }
%81 = type { %12, %21*, %53 }
%82 = type { %22, %8 }
%83 = type { %26, %84 }
%84 = type { i32, %22 }
%s2i64 = type { i64, i64 }
%85 = type { %26* }
%86 = type { %27*, i8*, %32*, i8*, i32, %8, i64, i32 }
%87 = type { %86, %88, %22, %95* }
%88 = type { %89 }
%89 = type { %90 }
%90 = type { %91*, %91*, %91* }
%91 = type { %92 }
%92 = type { %93 }
%93 = type { %94*, %94*, %94* }
%94 = type { %8, %18, %12*, %9 }
%95 = type { %37, %42 }

@_ZNSs4_Rep20_S_empty_rep_storageE = external global [4 x i64] ; <[4 x i64]*> [#uses=1]
@.str111723 = external constant [1 x i8], align 1 ; <[1 x i8]*> [#uses=1]
@.str181730 = external constant [4 x i8], align 1 ; <[4 x i8]*> [#uses=1]
@.str721784 = external constant [37 x i8], align 8 ; <[37 x i8]*> [#uses=1]
@_ZN12_GLOBAL__N_16ActionE = external global %0, align 32 ; <%0*> [#uses=1]
@_ZN12_GLOBAL__N_114OutputFilenameE = external global %6, align 32 ; <%6*> [#uses=1]
@_ZN12_GLOBAL__N_111IncludeDirsE = external global %13, align 32 ; <%13*> [#uses=1]
@.str533653 = external constant [2 x i8], align 1 ; <[2 x i8]*> [#uses=1]
@_ZN4llvm7RecordsE = external global %21, align 32 ; <%21*> [#uses=2]
@_ZL6SrcMgr = external global %27, align 32       ; <%27*> [#uses=2]
@.str3723 = external constant [88 x i8], align 8  ; <[88 x i8]*> [#uses=1]
@.str13724 = external constant [136 x i8], align 8 ; <[136 x i8]*> [#uses=1]

@_ZL20__gthrw_pthread_oncePiPFvvE = alias weak i32 (i32*, void ()*)* @pthread_once ; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj = alias weak i8* (i32)* @pthread_getspecific ; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv = alias weak i32 (i32, i8*)* @pthread_setspecific ; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_ = alias weak i32 (i64*, %33*, i8* (i8*)*, i8*)* @pthread_create ; <i32 (i64*, %33*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm = alias weak i32 (i64)* @pthread_cancel ; <i32 (i64)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = alias weak i32 (%34*)* @pthread_mutex_lock ; <i32 (%34*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = alias weak i32 (%34*)* @pthread_mutex_trylock ; <i32 (%34*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = alias weak i32 (%34*)* @pthread_mutex_unlock ; <i32 (%34*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = alias weak i32 (%34*, %2*)* @pthread_mutex_init ; <i32 (%34*, %2*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create ; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej = alias weak i32 (i32)* @pthread_key_delete ; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = alias weak i32 (%2*)* @pthread_mutexattr_init ; <i32 (%2*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = alias weak i32 (%2*, i32)* @pthread_mutexattr_settype ; <i32 (%2*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = alias weak i32 (%2*)* @pthread_mutexattr_destroy ; <i32 (%2*)*> [#uses=0]

declare void @_ZNSsC1EPKcRKSaIcE(%8*, i8*, %25*)

declare i8* @_Znwm(i64)

declare zeroext i8 @_ZNK4llvm6Record12isSubClassOfENS_9StringRefE(%37*, i64, i64) align 2

declare i32 @_ZNKSs7compareEPKc(%8*, i8*)

declare %45* @_ZN4llvm11raw_ostreamlsEPKc(%45*, i8*) align 2

declare void @_ZNSsC1ERKSs(%8*, %8*)

declare %26* @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(%26*)

declare void @_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_EPKS3_RKS6_(%8* noalias sret, i8*, %8*)

declare %15* @_ZNSt6vectorISsSaISsEEaSERKS1_(%15*, %15*) align 2

declare void @_ZNSt6vectorISsSaISsEE9push_backERKSs(%15*, %8*) align 2

declare i32 @_ZNK4llvm15TreePatternNode10getTypeNumEj(%46*, i32) align 2

declare void @_ZN4llvm18CodeGenDAGPatternsD1Ev(%53*) align 2

declare void @_ZN4llvm18CodeGenDAGPatternsC1ERNS_12RecordKeeperE(%53*, %21*) align 2

declare void @_ZNK4llvm14PatternToMatch17getPredicateCheckEv(%8* noalias sret, %75*) align 2

define internal void @0(%81*, %45*) align 2 {
  invoke void @_ZNSsC1ERKSs(%8* undef, %8* null)
          to label %3 unwind label %28

; <label>:3                                       ; preds = %2
  %4 = getelementptr inbounds i8* null, i64 -24   ; <i8*> [#uses=1]
  %5 = icmp eq i8* %4, bitcast ([4 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*) ; <i1> [#uses=1]
  br i1 %5, label %7, label %6

; <label>:6                                       ; preds = %3
  unreachable

; <label>:7                                       ; preds = %3
  br i1 undef, label %9, label %8

; <label>:8                                       ; preds = %7
  invoke void @_ZStplIcSt11char_traitsIcESaIcEESbIT_T0_T1_EPKS3_RKS6_(%8* noalias sret null, i8* getelementptr inbounds ([37 x i8]* @.str721784, i64 0, i64 0), %8* undef)
          to label %10 unwind label %29

; <label>:9                                       ; preds = %7
  unreachable

; <label>:10                                      ; preds = %8
  invoke void @_ZNSsC1ERKSs(%8* null, %8* null)
          to label %11 unwind label %30

; <label>:11                                      ; preds = %10
  %12 = invoke %45* @_ZN4llvm11raw_ostreamlsEPKc(%45* %1, i8* getelementptr inbounds ([88 x i8]* @.str3723, i64 0, i64 0))
          to label %13 unwind label %31           ; <%45*> [#uses=1]

; <label>:13                                      ; preds = %11
  %14 = invoke %45* @_ZN4llvm11raw_ostreamlsEPKc(%45* %12, i8* getelementptr inbounds ([136 x i8]* @.str13724, i64 0, i64 0))
          to label %15 unwind label %31           ; <%45*> [#uses=0]

; <label>:15                                      ; preds = %13
  %16 = getelementptr inbounds i8* null, i64 -24  ; <i8*> [#uses=1]
  %17 = icmp eq i8* %16, bitcast ([4 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*) ; <i1> [#uses=1]
  br i1 %17, label %19, label %18

; <label>:18                                      ; preds = %15
  unreachable

; <label>:19                                      ; preds = %15
  %20 = getelementptr inbounds i8* null, i64 -24  ; <i8*> [#uses=1]
  %21 = icmp eq i8* %20, bitcast ([4 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE to i8*) ; <i1> [#uses=1]
  br i1 %21, label %23, label %22

; <label>:22                                      ; preds = %19
  unreachable

; <label>:23                                      ; preds = %19
  invoke void @_ZNSsC1ERKSs(%8* undef, %8* undef)
          to label %24 unwind label %29

; <label>:24                                      ; preds = %23
  invoke void @_ZNSsC1ERKSs(%8* undef, %8* undef)
          to label %26 unwind label %25

; <label>:25                                      ; preds = %24
  unreachable

; <label>:26                                      ; preds = %24
  invoke void @f4(%82* undef, %53* undef)
          to label %27 unwind label %32

; <label>:27                                      ; preds = %26
  unreachable

; <label>:28                                      ; preds = %2
  unreachable

; <label>:29                                      ; preds = %23, %8
  unreachable

; <label>:30                                      ; preds = %10
  unreachable

; <label>:31                                      ; preds = %13, %11
  unreachable

; <label>:32                                      ; preds = %26
  unreachable
}

declare void @_ZNSt3mapISsN12_GLOBAL__N_115InstructionMemoESt4lessISsESaISt4pairIKSsS1_EEED1Ev(%22*) align 2

declare void @_ZNSt3mapISsN12_GLOBAL__N_115InstructionMemoESt4lessISsESaISt4pairIKSsS1_EEEC1ERKS8_(%22*, %22*) align 2

declare %83* @_ZNSt8_Rb_treeIN4llvm3MVT15SimpleValueTypeESt4pairIKS2_St3mapISsN12_GLOBAL__N_115InstructionMemoESt4lessISsESaIS3_IKSsS7_EEEESt10_Select1stISE_ES8_IS2_ESaISE_EE14_M_create_nodeERKSE_(%23*, %84*) align 2

declare %22* @_ZNSt3mapIN12_GLOBAL__N_117OperandsSignatureES_ISsS_IN4llvm3MVT15SimpleValueTypeES_IS4_S_ISsNS0_15InstructionMemoESt4lessISsESaISt4pairIKSsS5_EEES6_IS4_ESaIS8_IKS4_SC_EEESD_SaIS8_ISE_SH_EEES7_SaIS8_IS9_SK_EEES6_IS1_ESaIS8_IKS1_SN_EEEixERSP_(%22*, %14*) align 2

define internal %22* @1(%22*, i32*) align 2 {
  unreachable
}

declare void @test(i64);

define internal %22* @f3(%22*, i32*) align 2 {
  %3 = alloca %85, align 8                        ; <%85*> [#uses=1]
  %4 = alloca %84, align 8                        ; <%84*> [#uses=5]
  invoke void @_ZNSt3mapISsN12_GLOBAL__N_115InstructionMemoESt4lessISsESaISt4pairIKSsS1_EEEC1ERKS8_(%22* undef, %22* undef)
          to label %5 unwind label %30

; <label>:5                                       ; preds = %2
  %6 = getelementptr inbounds %26* null, i64 1, i32 0 ; <i32*> [#uses=1]
  %7 = load i32* %6, align 4                      ; <i32> [#uses=0]
  br i1 false, label %8, label %11

; <label>:8                                       ; preds = %5
  %9 = invoke %83* @_ZNSt8_Rb_treeIN4llvm3MVT15SimpleValueTypeESt4pairIKS2_St3mapISsN12_GLOBAL__N_115InstructionMemoESt4lessISsESaIS3_IKSsS7_EEEESt10_Select1stISE_ES8_IS2_ESaISE_EE14_M_create_nodeERKSE_(%23* undef, %84* %4)
          to label %10 unwind label %31           ; <%83*> [#uses=0]

; <label>:10                                      ; preds = %8
  unreachable

; <label>:11                                      ; preds = %5
  %12 = getelementptr inbounds %85* %3, i64 0, i32 0 ; <%26**> [#uses=1]
  %13 = load %26** %12, align 8                   ; <%26*> [#uses=0]
  br i1 false, label %14, label %17

; <label>:14                                      ; preds = %11
  %15 = invoke %83* @_ZNSt8_Rb_treeIN4llvm3MVT15SimpleValueTypeESt4pairIKS2_St3mapISsN12_GLOBAL__N_115InstructionMemoESt4lessISsESaIS3_IKSsS7_EEEESt10_Select1stISE_ES8_IS2_ESaISE_EE14_M_create_nodeERKSE_(%23* undef, %84* %4)
          to label %16 unwind label %31           ; <%83*> [#uses=0]

; <label>:16                                      ; preds = %14
  unreachable

; <label>:17                                      ; preds = %11
  %18 = invoke %26* @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(%26* undef)
          to label %19 unwind label %31           ; <%26*> [#uses=0]

; <label>:19                                      ; preds = %17
  %20 = getelementptr inbounds %84* %4, i64 0, i32 0 ; <i32*> [#uses=1]
  %21 = load i32* %20, align 8                    ; <i32> [#uses=0]
  br i1 false, label %22, label %25

; <label>:22                                      ; preds = %19
  %23 = invoke %83* @_ZNSt8_Rb_treeIN4llvm3MVT15SimpleValueTypeESt4pairIKS2_St3mapISsN12_GLOBAL__N_115InstructionMemoESt4lessISsESaIS3_IKSsS7_EEEESt10_Select1stISE_ES8_IS2_ESaISE_EE14_M_create_nodeERKSE_(%23* undef, %84* %4)
          to label %24 unwind label %31           ; <%83*> [#uses=0]

; <label>:24                                      ; preds = %22
  unreachable

; <label>:25                                      ; preds = %19
  %26 = invoke %s2i64 @f_3(%23* undef, %84* %4)
          to label %l1 unwind label %31

; <label>:27                                      ; preds = %25
  %eval = extractvalue %s2i64 %inv, 0                   ; <i64> [#uses=0]
  call void @test(i64 %eval)
; CHECK: = invoke %s2i64 @f_3
; CHECK: %eval = extractvalue %s2i64 %inv, 0
; CHECK-NEXT: call void @test(i64 %eval)
  invoke void @_ZNSt3mapISsN12_GLOBAL__N_115InstructionMemoESt4lessISsESaISt4pairIKSsS1_EEED1Ev(%22* undef)
          to label %29 unwind label %30

l1:
  %28 = extractvalue %s2i64 %26, 0
  call void @test(i64 %28)
;CHECK: call void @test(i64 5)
;CHECK-NEXT: %inv = invoke %s2i64 @f2
  %inv = invoke %s2i64 @f2(%23* undef, %84* %4)
          to label %27 unwind label %31           ; <%s2i64> [#uses=1]


; <label>:29                                      ; preds = %27
  unreachable

; <label>:30                                      ; preds = %27, %2
  unreachable

; <label>:31                                      ; preds = %25, %22, %17, %14, %8
  unreachable
}
define internal %s2i64 @f2(%23*, %84*) align 2 {
  br i1 undef, label %3, label %4

; <label>:3                                       ; preds = %2
  br label %4

; <label>:4                                       ; preds = %3, %2
  %5 = insertvalue %s2i64 undef, i64 4, 1        ; <%s2i64> [#uses=1]
  %6 = ptrtoint %84* %1 to i64
  %7 = insertvalue %s2i64 %5, i64 %6, 0        ; <%s2i64> [#uses=1]
  ret %s2i64 %7
}
define internal %s2i64 @f_3(%23*, %84*) align 2 {
  br i1 undef, label %3, label %4

; <label>:3                                       ; preds = %2
  br label %4

; <label>:4                                       ; preds = %3, %2
  %5 = insertvalue %s2i64 undef, i64 4, 1        ; <%s2i64> [#uses=1]
  %6 = insertvalue %s2i64 %5, i64 5, 0        ; <%s2i64> [#uses=1]
  ret %s2i64 %6
}


declare %22* @_ZNSt3mapISsS_IN4llvm3MVT15SimpleValueTypeES_IS2_S_ISsN12_GLOBAL__N_115InstructionMemoESt4lessISsESaISt4pairIKSsS4_EEES5_IS2_ESaIS7_IKS2_SB_EEESC_SaIS7_ISD_SG_EEES6_SaIS7_IS8_SJ_EEEixERS8_(%22*, %8*) align 2

define internal void @f4(%82*, %53*) align 2 {
  invoke void @_ZNSsC1ERKSs(%8* undef, %8* undef)
          to label %3 unwind label %57

; <label>:3                                       ; preds = %2
  br i1 undef, label %6, label %4

; <label>:4                                       ; preds = %3
  br i1 undef, label %5, label %6

; <label>:5                                       ; preds = %4
  br label %6

; <label>:6                                       ; preds = %5, %4, %3
  br i1 undef, label %8, label %7

; <label>:7                                       ; preds = %6
  br i1 undef, label %56, label %9

; <label>:8                                       ; preds = %6
  unreachable

; <label>:9                                       ; preds = %7
  %10 = icmp eq %12* null, null                   ; <i1> [#uses=1]
  br i1 %10, label %12, label %11

; <label>:11                                      ; preds = %9
  unreachable

; <label>:12                                      ; preds = %9
  %13 = getelementptr inbounds %46* null, i64 0, i32 6, i32 0, i32 0, i32 0 ; <%46***> [#uses=0]
  %14 = icmp eq i32 0, 0                          ; <i1> [#uses=1]
  br i1 %14, label %16, label %15

; <label>:15                                      ; preds = %12
  unreachable

; <label>:16                                      ; preds = %12
  br i1 undef, label %19, label %17

; <label>:17                                      ; preds = %16
  %18 = icmp eq %12* null, null                   ; <i1> [#uses=1]
  br i1 %18, label %21, label %20

; <label>:19                                      ; preds = %16
  unreachable

; <label>:20                                      ; preds = %17
  unreachable

; <label>:21                                      ; preds = %17
  %22 = invoke i32 @_ZNK4llvm15TreePatternNode10getTypeNumEj(%46* null, i32 0)
          to label %23 unwind label %58           ; <i32> [#uses=0]

; <label>:23                                      ; preds = %21
  %24 = invoke zeroext i8 @_ZNK4llvm6Record12isSubClassOfENS_9StringRefE(%37* undef, i64 undef, i64 undef)
          to label %25 unwind label %58           ; <i8> [#uses=0]

; <label>:25                                      ; preds = %23
  br i1 undef, label %26, label %27

; <label>:26                                      ; preds = %25
  unreachable

; <label>:27                                      ; preds = %25
  %28 = invoke i8* @_Znwm(i64 24)
          to label %29 unwind label %59           ; <i8*> [#uses=0]

; <label>:29                                      ; preds = %27
  %30 = icmp eq %12* null, null                   ; <i1> [#uses=1]
  br i1 %30, label %32, label %31

; <label>:31                                      ; preds = %29
  unreachable

; <label>:32                                      ; preds = %29
  %33 = invoke i32 @_ZNKSs7compareEPKc(%8* undef, i8* getelementptr inbounds ([4 x i8]* @.str181730, i64 0, i64 0))
          to label %34 unwind label %59           ; <i32> [#uses=1]

; <label>:34                                      ; preds = %32
  %35 = icmp eq i32 %33, 0                        ; <i1> [#uses=1]
  br i1 %35, label %37, label %36

; <label>:36                                      ; preds = %34
  unreachable

; <label>:37                                      ; preds = %34
  invoke void @_ZNSsC1EPKcRKSaIcE(%8* undef, i8* getelementptr inbounds ([1 x i8]* @.str111723, i64 0, i64 0), %25* undef)
          to label %38 unwind label %60

; <label>:38                                      ; preds = %37
  invoke void @_ZNSt6vectorISsSaISsEE9push_backERKSs(%15* undef, %8* undef)
          to label %39 unwind label %61

; <label>:39                                      ; preds = %38
  br i1 undef, label %42, label %40

; <label>:40                                      ; preds = %39
  br i1 undef, label %41, label %42

; <label>:41                                      ; preds = %40
  unreachable

; <label>:42                                      ; preds = %40, %39
  invoke void @_ZNK4llvm14PatternToMatch17getPredicateCheckEv(%8* noalias sret undef, %75* null)
          to label %43 unwind label %59

; <label>:43                                      ; preds = %42
  %44 = icmp eq %12* null, null                   ; <i1> [#uses=1]
  br i1 %44, label %46, label %45

; <label>:45                                      ; preds = %43
  unreachable

; <label>:46                                      ; preds = %43
  invoke void @_ZNSsC1ERKSs(%8* undef, %8* undef)
          to label %47 unwind label %62

; <label>:47                                      ; preds = %46
  %48 = invoke %22* @_ZNSt3mapIN12_GLOBAL__N_117OperandsSignatureES_ISsS_IN4llvm3MVT15SimpleValueTypeES_IS4_S_ISsNS0_15InstructionMemoESt4lessISsESaISt4pairIKSsS5_EEES6_IS4_ESaIS8_IKS4_SC_EEESD_SaIS8_ISE_SH_EEES7_SaIS8_IS9_SK_EEES6_IS1_ESaIS8_IKS1_SN_EEEixERSP_(%22* undef, %14* undef)
          to label %49 unwind label %63           ; <%22*> [#uses=1]

; <label>:49                                      ; preds = %47
  %50 = invoke %22* @_ZNSt3mapISsS_IN4llvm3MVT15SimpleValueTypeES_IS2_S_ISsN12_GLOBAL__N_115InstructionMemoESt4lessISsESaISt4pairIKSsS4_EEES5_IS2_ESaIS7_IKS2_SB_EEESC_SaIS7_ISD_SG_EEES6_SaIS7_IS8_SJ_EEEixERS8_(%22* %48, %8* undef)
          to label %51 unwind label %63           ; <%22*> [#uses=1]

; <label>:51                                      ; preds = %49
  %52 = invoke %22* @1(%22* %50, i32* undef)
          to label %53 unwind label %63           ; <%22*> [#uses=1]

; <label>:53                                      ; preds = %51
  %54 = invoke %22* @f3(%22* %52, i32* undef)
          to label %55 unwind label %63           ; <%22*> [#uses=0]

; <label>:55                                      ; preds = %53
  unreachable

; <label>:56                                      ; preds = %7
  ret void

; <label>:57                                      ; preds = %2
  unreachable

; <label>:58                                      ; preds = %23, %21
  unreachable

; <label>:59                                      ; preds = %42, %32, %27
  unreachable

; <label>:60                                      ; preds = %37
  unreachable

; <label>:61                                      ; preds = %38
  unreachable

; <label>:62                                      ; preds = %46
  unreachable

; <label>:63                                      ; preds = %53, %51, %49, %47
  unreachable
}

declare i32 @_ZN4llvm7TGLexer8LexTokenEv(%86*) align 2

declare void @_ZN4llvm8TGParserD1Ev(%87*) align 2

declare i32 @_ZN4llvm9SourceMgr18AddNewSourceBufferEPNS_12MemoryBufferENS_5SMLocE(%27*, %32*, %9* noalias) align 2

define i32 @main(i32, i8**) {
  br i1 undef, label %3, label %4

; <label>:3                                       ; preds = %2
  unreachable

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 0, 0                           ; <i1> [#uses=1]
  br i1 %5, label %7, label %6

; <label>:6                                       ; preds = %4
  unreachable

; <label>:7                                       ; preds = %4
  br i1 undef, label %8, label %9

; <label>:8                                       ; preds = %7
  br i1 undef, label %10, label %9

; <label>:9                                       ; preds = %8, %7
  br i1 undef, label %14, label %18

; <label>:10                                      ; preds = %8
  br label %11

; <label>:11                                      ; preds = %12, %10
  br i1 undef, label %12, label %13

; <label>:12                                      ; preds = %11
  br label %11

; <label>:13                                      ; preds = %11
  unreachable

; <label>:14                                      ; preds = %9
  br i1 undef, label %15, label %17

; <label>:15                                      ; preds = %14
  br i1 undef, label %17, label %16

; <label>:16                                      ; preds = %15
  unreachable

; <label>:17                                      ; preds = %15, %14
  unreachable

; <label>:18                                      ; preds = %9
  %19 = invoke i32 @_ZN4llvm9SourceMgr18AddNewSourceBufferEPNS_12MemoryBufferENS_5SMLocE(%27* @_ZL6SrcMgr, %32* undef, %9* noalias undef)
          to label %20 unwind label %34           ; <i32> [#uses=0]

; <label>:20                                      ; preds = %18
  %21 = invoke %15* @_ZNSt6vectorISsSaISsEEaSERKS1_(%15* getelementptr inbounds (%27* @_ZL6SrcMgr, i64 0, i32 1), %15* getelementptr inbounds (%13* @_ZN12_GLOBAL__N_111IncludeDirsE, i64 0, i32 1, i32 0))
          to label %22 unwind label %34           ; <%15*> [#uses=0]

; <label>:22                                      ; preds = %20
  %23 = getelementptr inbounds %27* null, i64 0, i32 0, i32 0, i32 0, i32 1 ; <%31**> [#uses=1]
  %24 = load %31** %23, align 8                   ; <%31*> [#uses=1]
  %25 = ptrtoint %31* %24 to i64                  ; <i64> [#uses=1]
  %26 = sub i64 %25, 0                            ; <i64> [#uses=1]
  %27 = icmp ugt i64 %26, 15                      ; <i1> [#uses=1]
  br i1 %27, label %29, label %28

; <label>:28                                      ; preds = %22
  unreachable

; <label>:29                                      ; preds = %22
  %30 = invoke i32 @_ZN4llvm7TGLexer8LexTokenEv(%86* undef)
          to label %31 unwind label %35           ; <i32> [#uses=0]

; <label>:31                                      ; preds = %29
  invoke void @_ZN4llvm8TGParserD1Ev(%87* null)
          to label %32 unwind label %34

; <label>:32                                      ; preds = %31
  %33 = icmp eq i8 0, 0                           ; <i1> [#uses=1]
  br i1 %33, label %36, label %56

; <label>:34                                      ; preds = %31, %20, %18
  unreachable

; <label>:35                                      ; preds = %29
  unreachable

; <label>:36                                      ; preds = %32
  %37 = invoke i32 @_ZNKSs7compareEPKc(%8* getelementptr inbounds (%6* @_ZN12_GLOBAL__N_114OutputFilenameE, i64 0, i32 1, i32 0), i8* getelementptr inbounds ([2 x i8]* @.str533653, i64 0, i64 0))
          to label %38 unwind label %57           ; <i32> [#uses=1]

; <label>:38                                      ; preds = %36
  %39 = icmp eq i32 %37, 0                        ; <i1> [#uses=1]
  br i1 %39, label %43, label %40

; <label>:40                                      ; preds = %38
  %41 = invoke i8* @_Znwm(i64 56)
          to label %42 unwind label %58           ; <i8*> [#uses=0]

; <label>:42                                      ; preds = %40
  unreachable

; <label>:43                                      ; preds = %38
  %44 = load i32* getelementptr inbounds (%0* @_ZN12_GLOBAL__N_16ActionE, i64 0, i32 1, i32 0), align 8 ; <i32> [#uses=1]
  switch i32 %44, label %56 [
    i32 0, label %45
    i32 12, label %48
    i32 13, label %51
  ]

; <label>:45                                      ; preds = %43
  br i1 undef, label %46, label %47

; <label>:46                                      ; preds = %45
  unreachable

; <label>:47                                      ; preds = %45
  unreachable

; <label>:48                                      ; preds = %43
  invoke void @_ZN4llvm18CodeGenDAGPatternsC1ERNS_12RecordKeeperE(%53* undef, %21* @_ZN4llvm7RecordsE)
          to label %50 unwind label %49

; <label>:49                                      ; preds = %48
  unreachable

; <label>:50                                      ; preds = %48
  unreachable

; <label>:51                                      ; preds = %43
  invoke void @_ZN4llvm18CodeGenDAGPatternsC1ERNS_12RecordKeeperE(%53* undef, %21* @_ZN4llvm7RecordsE)
          to label %53 unwind label %52

; <label>:52                                      ; preds = %51
  unreachable

; <label>:53                                      ; preds = %51
  invoke void @0(%81* undef, %45* null)
          to label %54 unwind label %60

; <label>:54                                      ; preds = %53
  invoke void @_ZN4llvm18CodeGenDAGPatternsD1Ev(%53* undef)
          to label %55 unwind label %59

; <label>:55                                      ; preds = %54
  unreachable

; <label>:56                                      ; preds = %43, %32
  ret i32 1

; <label>:57                                      ; preds = %36
  unreachable

; <label>:58                                      ; preds = %40
  unreachable

; <label>:59                                      ; preds = %54
  unreachable

; <label>:60                                      ; preds = %53
  unreachable
}

declare i8* @pthread_getspecific(i32) nounwind

declare i32 @pthread_setspecific(i32, i8*) nounwind

declare i32 @pthread_key_delete(i32) nounwind

declare i32 @pthread_key_create(i32*, void (i8*)*) nounwind

declare i32 @pthread_mutex_trylock(%34*) nounwind

declare i32 @pthread_mutex_unlock(%34*) nounwind

declare i32 @pthread_mutex_lock(%34*) nounwind

declare i32 @pthread_mutexattr_init(%2*) nounwind

declare i32 @pthread_mutexattr_settype(%2*, i32) nounwind

declare i32 @pthread_mutex_init(%34*, %2*) nounwind

declare i32 @pthread_mutexattr_destroy(%2*) nounwind

declare extern_weak i32 @pthread_once(i32*, void ()*)

declare extern_weak i32 @pthread_create(i64*, %33*, i8* (i8*)*, i8*)

declare extern_weak i32 @pthread_cancel(i64)
