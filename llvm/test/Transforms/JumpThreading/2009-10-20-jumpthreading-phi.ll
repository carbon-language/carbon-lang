; RUN: opt -jump-threading -verify %s -disable-output
; PR5258
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%0 = type { i64, [12 x i32] }
%1 = type { %2 }
%2 = type { i32, i32, i32, i32, i32, i32, %3 }
%3 = type { %3*, %3* }
%4 = type { i32 }
%5 = type { %6*, %84, %87, %89, %92, %95, %69, %79*, %102*, %69, %69, %69 }
%6 = type { %7* }
%7 = type { %8, %10, %28, %34, %38, %40, %40, %40, %40, %40, %40, %40, %21*, %21*, %45, %46, %45, %45, %15, %15, %15, %15, %15, %15, %15, %15, %49, %49, %49, %49, %49, %9, %9, %51, %51, %51, %51, %51, %51, %53, %56 }
%8 = type { %9, i32, i32 }
%9 = type { i8* }
%10 = type { i32, %11*, i32, i32 }
%11 = type { %12, %21* }
%12 = type { %13, %15* }
%13 = type { i32, %14 }
%14 = type { i64 }
%15 = type { %16, i8, [3 x i8], i32, %6*, %15*, %17, i32, %20* }
%16 = type { i32 (...)** }
%17 = type { %18 }
%18 = type { %19 }
%19 = type { %16**, %16**, %16** }
%20 = type { %15*, %16* }
%21 = type { %22, %13 }
%22 = type { %23 }
%23 = type { %24, %26*, i32 }
%24 = type { i32 (...)**, i8, i8, i8, i16, %25, %26*, %27* }
%25 = type { %15* }
%26 = type { %24*, %26*, %14 }
%27 = type { %4, %24* }
%28 = type { i32, %29*, i32, i32 }
%29 = type { %30, %33* }
%30 = type { %31 }
%31 = type { %32*, %14, i16, i16 }
%32 = type { i16, i16, i32, i32 }
%33 = type { %22, %31 }
%34 = type { %35, %37 }
%35 = type { %36*, i32, i32, i32, i32 }
%36 = type { i32, %4* }
%37 = type <{ i8 }>
%38 = type { %39 }
%39 = type { i32 (...)**, i8**, i32, i32 }
%40 = type { %16, %41, %41, %41, %45 }
%41 = type { %42 }
%42 = type { %43 }
%43 = type { %37, %44, i64 }
%44 = type { i32, %44*, %44*, %44* }
%45 = type { %9, i32, i8 }
%46 = type { %47, %24*, i8* }
%47 = type { %48, [8 x i8*] }
%48 = type { i8**, i32, i32, i32, [1 x i8*] }
%49 = type { %50 }
%50 = type { %15 }
%51 = type { %52, %41 }
%52 = type { %41 }
%53 = type { i32, %54*, i32, i32 }
%54 = type { %24*, %55* }
%55 = type { %14, %55*, %24* }
%56 = type { %57, %34 }
%57 = type { i32, %58*, i32, i32 }
%58 = type <{ %59*, [8 x i8], %83 }>
%59 = type { %23, %60, %62* }
%60 = type { %61, %59* }
%61 = type { %59* }
%62 = type { %24, %63, %65, %67* }
%63 = type { %64, %62* }
%64 = type { %62* }
%65 = type { %66, %59* }
%66 = type { %61 }
%67 = type { %68, %70, %72, %74, %79*, %80 }
%68 = type { %22, %5*, i32, %69 }
%69 = type { %9 }
%70 = type { %71, %67* }
%71 = type { %67* }
%72 = type { %73, %62* }
%73 = type { %64 }
%74 = type { %75, %77* }
%75 = type { %76 }
%76 = type { %77* }
%77 = type { %24, %78, %67* }
%78 = type { %76, %77* }
%79 = type { %34, i32 }
%80 = type { %81* }
%81 = type <{ %9, i32, [4 x i8], %82 }>
%82 = type <{ [33 x i8], [31 x i8] }>
%83 = type <{ [33 x i8], [63 x i8] }>
%84 = type { %85* }
%85 = type { %68, %86, i8 }
%86 = type { %84, %85* }
%87 = type { %88, %67* }
%88 = type { %70 }
%89 = type { %90* }
%90 = type { %68, %91 }
%91 = type { %89, %90* }
%92 = type { %93 }
%93 = type { %94 }
%94 = type { %69*, %69*, %69* }
%95 = type { %96, %99* }
%96 = type { %97 }
%97 = type { %98, %99* }
%98 = type { %99* }
%99 = type <{ %100, %97, %5*, %101 }>
%100 = type { [52 x i8], i32 }
%101 = type <{ [33 x i8], [95 x i8] }>
%102 = type { %16, %41, i32 }

@_ZL20__gthrw_pthread_oncePiPFvvE = alias weak i32 (i32*, void ()*)* @pthread_once ; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj = alias weak i8* (i32)* @pthread_getspecific ; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv = alias weak i32 (i32, i8*)* @pthread_setspecific ; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_ = alias weak i32 (i64*, %0*, i8* (i8*)*, i8*)* @pthread_create ; <i32 (i64*, %0*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm = alias weak i32 (i64)* @pthread_cancel ; <i32 (i64)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = alias weak i32 (%1*)* @pthread_mutex_lock ; <i32 (%1*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = alias weak i32 (%1*)* @pthread_mutex_trylock ; <i32 (%1*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = alias weak i32 (%1*)* @pthread_mutex_unlock ; <i32 (%1*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = alias weak i32 (%1*, %4*)* @pthread_mutex_init ; <i32 (%1*, %4*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create ; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej = alias weak i32 (i32)* @pthread_key_delete ; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = alias weak i32 (%4*)* @pthread_mutexattr_init ; <i32 (%4*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = alias weak i32 (%4*, i32)* @pthread_mutexattr_settype ; <i32 (%4*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = alias weak i32 (%4*)* @pthread_mutexattr_destroy ; <i32 (%4*)*> [#uses=0]

define fastcc zeroext i8 @_ZN4llvm6Linker11LinkModulesEPNS_6ModuleES2_PSs(%5*, %5*, %69*) nounwind align 2 {
  br i1 false, label %4, label %5

; <label>:4                                       ; preds = %3
  unreachable

; <label>:5                                       ; preds = %3
  br i1 false, label %6, label %7

; <label>:6                                       ; preds = %5
  unreachable

; <label>:7                                       ; preds = %5
  br i1 false, label %8, label %11

; <label>:8                                       ; preds = %7
  br i1 undef, label %10, label %9

; <label>:9                                       ; preds = %8
  unreachable

; <label>:10                                      ; preds = %8
  unreachable

; <label>:11                                      ; preds = %7
  br i1 undef, label %13, label %12

; <label>:12                                      ; preds = %11
  unreachable

; <label>:13                                      ; preds = %11
  br i1 undef, label %15, label %14

; <label>:14                                      ; preds = %13
  br label %15

; <label>:15                                      ; preds = %14, %13
  br i1 undef, label %17, label %16

; <label>:16                                      ; preds = %15
  unreachable

; <label>:17                                      ; preds = %15
  br i1 undef, label %19, label %18

; <label>:18                                      ; preds = %17
  unreachable

; <label>:19                                      ; preds = %17
  br i1 false, label %20, label %21

; <label>:20                                      ; preds = %19
  unreachable

; <label>:21                                      ; preds = %19
  br i1 undef, label %22, label %23

; <label>:22                                      ; preds = %21
  br label %23

; <label>:23                                      ; preds = %22, %21
  br i1 false, label %24, label %25

; <label>:24                                      ; preds = %23
  unreachable

; <label>:25                                      ; preds = %23
  br i1 undef, label %29, label %26

; <label>:26                                      ; preds = %25
  br i1 undef, label %28, label %27

; <label>:27                                      ; preds = %26
  unreachable

; <label>:28                                      ; preds = %26
  unreachable

; <label>:29                                      ; preds = %25
  br i1 undef, label %31, label %30

; <label>:30                                      ; preds = %29
  unreachable

; <label>:31                                      ; preds = %29
  br i1 undef, label %32, label %33

; <label>:32                                      ; preds = %31
  br label %33

; <label>:33                                      ; preds = %32, %31
  br i1 false, label %34, label %35

; <label>:34                                      ; preds = %33
  unreachable

; <label>:35                                      ; preds = %33
  br i1 undef, label %36, label %37

; <label>:36                                      ; preds = %35
  br label %37

; <label>:37                                      ; preds = %36, %35
  br i1 undef, label %39, label %38

; <label>:38                                      ; preds = %37
  br i1 false, label %39, label %40

; <label>:39                                      ; preds = %38, %37
  unreachable

; <label>:40                                      ; preds = %38
  %41 = load i8* undef, align 8                   ; <i8> [#uses=1]
  switch i8 %41, label %42 [
    i8 4, label %43
    i8 2, label %43
    i8 3, label %43
  ]

; <label>:42                                      ; preds = %40
  unreachable

; <label>:43                                      ; preds = %40, %40, %40
  %44 = trunc i32 undef to i5                     ; <i5> [#uses=1]
  switch i5 %44, label %45 [
    i5 7, label %50
    i5 9, label %50
  ]

; <label>:45                                      ; preds = %43
  br i1 undef, label %47, label %46

; <label>:46                                      ; preds = %45
  br label %47

; <label>:47                                      ; preds = %46, %45
  %48 = icmp eq %85* null, null                   ; <i1> [#uses=1]
  br i1 %48, label %50, label %49

; <label>:49                                      ; preds = %47
  unreachable

; <label>:50                                      ; preds = %47, %43, %43
  %51 = phi %68* [ null, %43 ], [ undef, %47 ], [ null, %43 ] ; <%68*> [#uses=4]
  %52 = phi %68* [ null, %43 ], [ undef, %47 ], [ null, %43 ] ; <%68*> [#uses=1]
  %53 = icmp eq %68* %52, null                    ; <i1> [#uses=1]
  br i1 %53, label %54, label %59

; <label>:54                                      ; preds = %50
  %55 = trunc i32 undef to i5                     ; <i5> [#uses=1]
  switch i5 %55, label %56 [
    i5 7, label %59
    i5 8, label %59
    i5 9, label %59
  ]

; <label>:56                                      ; preds = %54
  br i1 undef, label %58, label %57

; <label>:57                                      ; preds = %56
  br label %58

; <label>:58                                      ; preds = %57, %56
  br label %59

; <label>:59                                      ; preds = %58, %54, %54, %54, %50
  %60 = phi %68* [ %51, %50 ], [ %51, %54 ], [ %51, %54 ], [ %51, %54 ], [ undef, %58 ] ; <%68*> [#uses=0]
  br i1 undef, label %62, label %61

; <label>:61                                      ; preds = %59
  br label %62

; <label>:62                                      ; preds = %61, %59
  switch i8 undef, label %64 [
    i8 3, label %63
    i8 2, label %65
  ]

; <label>:63                                      ; preds = %62
  unreachable

; <label>:64                                      ; preds = %62
  unreachable

; <label>:65                                      ; preds = %62
  switch i8 undef, label %67 [
    i8 0, label %66
    i8 1, label %66
  ]

; <label>:66                                      ; preds = %65, %65
  unreachable

; <label>:67                                      ; preds = %65
  unreachable
}

declare i32 @pthread_mutex_trylock(%1*) nounwind

declare i32 @pthread_mutex_unlock(%1*) nounwind

declare i32 @pthread_mutex_lock(%1*) nounwind

declare i32 @pthread_mutexattr_init(%4*) nounwind

declare i32 @pthread_mutexattr_settype(%4*, i32) nounwind

declare i32 @pthread_mutex_init(%1*, %4*) nounwind

declare i32 @pthread_mutexattr_destroy(%4*) nounwind

declare i8* @pthread_getspecific(i32) nounwind

declare i32 @pthread_setspecific(i32, i8*) nounwind

declare i32 @pthread_key_delete(i32) nounwind

declare i32 @pthread_key_create(i32*, void (i8*)*) nounwind

declare i32 @pthread_once(i32*, void ()*)

declare i32 @pthread_create(i64*, %0*, i8* (i8*)*, i8*)

declare i32 @pthread_cancel(i64)
