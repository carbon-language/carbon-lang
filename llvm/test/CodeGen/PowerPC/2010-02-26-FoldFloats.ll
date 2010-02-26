; RUN: llc < %s -O3 | FileCheck %s
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128-n32"
target triple = "powerpc-apple-darwin9.6"

; There should be no stfs spills
; CHECK: main:
; CHECK-NOT: stfs
; CHECK: .section

@.str66 = external constant [3 x i8], align 4     ; <[3 x i8]*> [#uses=1]
@.str31 = external constant [6 x i8], align 4     ; <[6 x i8]*> [#uses=1]
@.str61 = external constant [21 x i8], align 4    ; <[21 x i8]*> [#uses=1]
@.str101 = external constant [61 x i8], align 4   ; <[61 x i8]*> [#uses=1]
@.str104 = external constant [31 x i8], align 4   ; <[31 x i8]*> [#uses=1]
@.str105 = external constant [45 x i8], align 4   ; <[45 x i8]*> [#uses=1]
@.str112 = external constant [38 x i8], align 4   ; <[38 x i8]*> [#uses=1]
@.str121 = external constant [36 x i8], align 4   ; <[36 x i8]*> [#uses=1]
@.str12293 = external constant [67 x i8], align 4 ; <[67 x i8]*> [#uses=1]
@.str123 = external constant [68 x i8], align 4   ; <[68 x i8]*> [#uses=1]
@.str124 = external constant [52 x i8], align 4   ; <[52 x i8]*> [#uses=1]
@.str125 = external constant [51 x i8], align 4   ; <[51 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** %argv) noreturn nounwind {
entry:
  br i1 undef, label %bb4.i1, label %my_fopen.exit

bb4.i1:                                           ; preds = %entry
  unreachable

my_fopen.exit:                                    ; preds = %entry
  br i1 undef, label %bb.i, label %bb1.i

bb.i:                                             ; preds = %my_fopen.exit
  unreachable

bb1.i:                                            ; preds = %my_fopen.exit
  br label %bb134.i

bb2.i:                                            ; preds = %bb134.i
  %0 = icmp eq i32 undef, 0                       ; <i1> [#uses=1]
  br i1 %0, label %bb20.i, label %bb21.i

bb20.i:                                           ; preds = %bb2.i
  br label %bb134.i

bb21.i:                                           ; preds = %bb2.i
  %1 = call i32 @strcmp(i8* undef, i8* getelementptr inbounds ([6 x i8]* @.str31, i32 0, i32 0)) nounwind readonly ; <i32> [#uses=0]
  br i1 undef, label %bb30.i, label %bb31.i

bb30.i:                                           ; preds = %bb21.i
  br label %bb134.i

bb31.i:                                           ; preds = %bb21.i
  br i1 undef, label %bb41.i, label %bb44.i

bb41.i:                                           ; preds = %bb31.i
  %2 = icmp slt i32 undef, %argc                  ; <i1> [#uses=1]
  br i1 %2, label %bb1.i77.i, label %bb2.i78.i

bb1.i77.i:                                        ; preds = %bb41.i
  %3 = load float* undef, align 4                 ; <float> [#uses=2]
  %4 = fcmp ugt float %3, 0.000000e+00            ; <i1> [#uses=1]
  br i1 %4, label %bb43.i, label %bb42.i

bb2.i78.i:                                        ; preds = %bb41.i
  unreachable

bb42.i:                                           ; preds = %bb1.i77.i
  unreachable

bb43.i:                                           ; preds = %bb1.i77.i
  br label %bb134.i

bb44.i:                                           ; preds = %bb31.i
  br i1 undef, label %bb45.i, label %bb49.i

bb45.i:                                           ; preds = %bb44.i
  %5 = icmp slt i32 undef, %argc                  ; <i1> [#uses=1]
  br i1 %5, label %bb1.i72.i, label %bb2.i73.i

bb1.i72.i:                                        ; preds = %bb45.i
  %6 = load float* undef, align 4                 ; <float> [#uses=3]
  %7 = fcmp ult float %6, 1.000000e+00            ; <i1> [#uses=1]
  %or.cond.i = and i1 undef, %7                   ; <i1> [#uses=1]
  br i1 %or.cond.i, label %bb48.i, label %bb47.i

bb2.i73.i:                                        ; preds = %bb45.i
  unreachable

bb47.i:                                           ; preds = %bb1.i72.i
  unreachable

bb48.i:                                           ; preds = %bb1.i72.i
  br label %bb134.i

bb49.i:                                           ; preds = %bb44.i
  br i1 undef, label %bb50.i, label %bb53.i

bb50.i:                                           ; preds = %bb49.i
  br i1 false, label %bb1.i67.i, label %bb2.i68.i

bb1.i67.i:                                        ; preds = %bb50.i
  br i1 false, label %read_float_option.exit69.i, label %bb1.i67.bb2.i68_crit_edge.i

bb1.i67.bb2.i68_crit_edge.i:                      ; preds = %bb1.i67.i
  br label %bb2.i68.i

bb2.i68.i:                                        ; preds = %bb1.i67.bb2.i68_crit_edge.i, %bb50.i
  unreachable

read_float_option.exit69.i:                       ; preds = %bb1.i67.i
  br i1 undef, label %bb52.i, label %bb51.i

bb51.i:                                           ; preds = %read_float_option.exit69.i
  unreachable

bb52.i:                                           ; preds = %read_float_option.exit69.i
  br label %bb134.i

bb53.i:                                           ; preds = %bb49.i
  %8 = call i32 @strcmp(i8* undef, i8* getelementptr inbounds ([21 x i8]* @.str61, i32 0, i32 0)) nounwind readonly ; <i32> [#uses=0]
  br i1 false, label %bb89.i, label %bb92.i

bb89.i:                                           ; preds = %bb53.i
  br i1 undef, label %bb1.i27.i, label %bb2.i28.i

bb1.i27.i:                                        ; preds = %bb89.i
  unreachable

bb2.i28.i:                                        ; preds = %bb89.i
  unreachable

bb92.i:                                           ; preds = %bb53.i
  br i1 undef, label %bb93.i, label %bb96.i

bb93.i:                                           ; preds = %bb92.i
  br i1 undef, label %bb1.i22.i, label %bb2.i23.i

bb1.i22.i:                                        ; preds = %bb93.i
  br i1 undef, label %bb95.i, label %bb94.i

bb2.i23.i:                                        ; preds = %bb93.i
  unreachable

bb94.i:                                           ; preds = %bb1.i22.i
  unreachable

bb95.i:                                           ; preds = %bb1.i22.i
  br label %bb134.i

bb96.i:                                           ; preds = %bb92.i
  br i1 undef, label %bb97.i, label %bb100.i

bb97.i:                                           ; preds = %bb96.i
  %9 = icmp slt i32 undef, %argc                  ; <i1> [#uses=1]
  br i1 %9, label %bb1.i17.i, label %bb2.i18.i

bb1.i17.i:                                        ; preds = %bb97.i
  %10 = call i32 (i8*, i8*, ...)* @"\01_sscanf$LDBL128"(i8* undef, i8* getelementptr inbounds ([3 x i8]* @.str66, i32 0, i32 0), float* undef) nounwind ; <i32> [#uses=1]
  %phitmp.i16.i = icmp eq i32 %10, 1              ; <i1> [#uses=1]
  br i1 %phitmp.i16.i, label %read_float_option.exit19.i, label %bb1.i17.bb2.i18_crit_edge.i

bb1.i17.bb2.i18_crit_edge.i:                      ; preds = %bb1.i17.i
  br label %bb2.i18.i

bb2.i18.i:                                        ; preds = %bb1.i17.bb2.i18_crit_edge.i, %bb97.i
  unreachable

read_float_option.exit19.i:                       ; preds = %bb1.i17.i
  br i1 false, label %bb99.i, label %bb98.i

bb98.i:                                           ; preds = %read_float_option.exit19.i
  unreachable

bb99.i:                                           ; preds = %read_float_option.exit19.i
  br label %bb134.i

bb100.i:                                          ; preds = %bb96.i
  br i1 false, label %bb101.i, label %bb104.i

bb101.i:                                          ; preds = %bb100.i
  br i1 false, label %bb1.i12.i, label %bb2.i13.i

bb1.i12.i:                                        ; preds = %bb101.i
  br i1 undef, label %bb102.i, label %bb103.i

bb2.i13.i:                                        ; preds = %bb101.i
  unreachable

bb102.i:                                          ; preds = %bb1.i12.i
  unreachable

bb103.i:                                          ; preds = %bb1.i12.i
  br label %bb134.i

bb104.i:                                          ; preds = %bb100.i
  unreachable

bb134.i:                                          ; preds = %bb103.i, %bb99.i, %bb95.i, %bb52.i, %bb48.i, %bb43.i, %bb30.i, %bb20.i, %bb1.i
  %annealing_sched.1.0 = phi float [ 1.000000e+01, %bb1.i ], [ %annealing_sched.1.0, %bb20.i ], [ 1.000000e+00, %bb30.i ], [ %annealing_sched.1.0, %bb43.i ], [ %annealing_sched.1.0, %bb48.i ], [ %annealing_sched.1.0, %bb52.i ], [ %annealing_sched.1.0, %bb95.i ], [ %annealing_sched.1.0, %bb99.i ], [ %annealing_sched.1.0, %bb103.i ] ; <float> [#uses=8]
  %annealing_sched.2.0 = phi float [ 1.000000e+02, %bb1.i ], [ %annealing_sched.2.0, %bb20.i ], [ %annealing_sched.2.0, %bb30.i ], [ %3, %bb43.i ], [ %annealing_sched.2.0, %bb48.i ], [ %annealing_sched.2.0, %bb52.i ], [ %annealing_sched.2.0, %bb95.i ], [ %annealing_sched.2.0, %bb99.i ], [ %annealing_sched.2.0, %bb103.i ] ; <float> [#uses=8]
  %annealing_sched.3.0 = phi float [ 0x3FE99999A0000000, %bb1.i ], [ %annealing_sched.3.0, %bb20.i ], [ %annealing_sched.3.0, %bb30.i ], [ %annealing_sched.3.0, %bb43.i ], [ %6, %bb48.i ], [ %annealing_sched.3.0, %bb52.i ], [ %annealing_sched.3.0, %bb95.i ], [ %annealing_sched.3.0, %bb99.i ], [ %annealing_sched.3.0, %bb103.i ] ; <float> [#uses=8]
  %annealing_sched.4.0 = phi float [ 0x3F847AE140000000, %bb1.i ], [ %annealing_sched.4.0, %bb20.i ], [ %annealing_sched.4.0, %bb30.i ], [ %annealing_sched.4.0, %bb43.i ], [ %annealing_sched.4.0, %bb48.i ], [ 0.000000e+00, %bb52.i ], [ %annealing_sched.4.0, %bb95.i ], [ %annealing_sched.4.0, %bb99.i ], [ %annealing_sched.4.0, %bb103.i ] ; <float> [#uses=8]
  %router_opts.0.0 = phi float [ 0.000000e+00, %bb1.i ], [ %router_opts.0.0, %bb20.i ], [ 1.000000e+04, %bb30.i ], [ %router_opts.0.0, %bb43.i ], [ %router_opts.0.0, %bb48.i ], [ %router_opts.0.0, %bb52.i ], [ %router_opts.0.0, %bb95.i ], [ %router_opts.0.0, %bb99.i ], [ %router_opts.0.0, %bb103.i ] ; <float> [#uses=8]
  %router_opts.1.0 = phi float [ 5.000000e-01, %bb1.i ], [ %router_opts.1.0, %bb20.i ], [ 1.000000e+04, %bb30.i ], [ %router_opts.1.0, %bb43.i ], [ %router_opts.1.0, %bb48.i ], [ %router_opts.1.0, %bb52.i ], [ undef, %bb95.i ], [ %router_opts.1.0, %bb99.i ], [ %router_opts.1.0, %bb103.i ] ; <float> [#uses=7]
  %router_opts.2.0 = phi float [ 1.500000e+00, %bb1.i ], [ %router_opts.2.0, %bb20.i ], [ %router_opts.2.0, %bb30.i ], [ %router_opts.2.0, %bb43.i ], [ %router_opts.2.0, %bb48.i ], [ %router_opts.2.0, %bb52.i ], [ %router_opts.2.0, %bb95.i ], [ undef, %bb99.i ], [ %router_opts.2.0, %bb103.i ] ; <float> [#uses=8]
  %router_opts.3.0 = phi float [ 0x3FC99999A0000000, %bb1.i ], [ %router_opts.3.0, %bb20.i ], [ %router_opts.3.0, %bb30.i ], [ %router_opts.3.0, %bb43.i ], [ %router_opts.3.0, %bb48.i ], [ %router_opts.3.0, %bb52.i ], [ %router_opts.3.0, %bb95.i ], [ %router_opts.3.0, %bb99.i ], [ 0.000000e+00, %bb103.i ] ; <float> [#uses=8]
  %11 = phi float [ 0x3FC99999A0000000, %bb1.i ], [ %11, %bb20.i ], [ %11, %bb30.i ], [ %11, %bb43.i ], [ %11, %bb48.i ], [ %11, %bb52.i ], [ %11, %bb95.i ], [ %11, %bb99.i ], [ 0.000000e+00, %bb103.i ] ; <float> [#uses=8]
  %12 = phi float [ 1.500000e+00, %bb1.i ], [ %12, %bb20.i ], [ %12, %bb30.i ], [ %12, %bb43.i ], [ %12, %bb48.i ], [ %12, %bb52.i ], [ %12, %bb95.i ], [ undef, %bb99.i ], [ %12, %bb103.i ] ; <float> [#uses=8]
  %13 = phi float [ 5.000000e-01, %bb1.i ], [ %13, %bb20.i ], [ 1.000000e+04, %bb30.i ], [ %13, %bb43.i ], [ %13, %bb48.i ], [ %13, %bb52.i ], [ undef, %bb95.i ], [ %13, %bb99.i ], [ %13, %bb103.i ] ; <float> [#uses=7]
  %14 = phi float [ 0.000000e+00, %bb1.i ], [ %14, %bb20.i ], [ 1.000000e+04, %bb30.i ], [ %14, %bb43.i ], [ %14, %bb48.i ], [ %14, %bb52.i ], [ %14, %bb95.i ], [ %14, %bb99.i ], [ %14, %bb103.i ] ; <float> [#uses=8]
  %15 = phi float [ 0x3FE99999A0000000, %bb1.i ], [ %15, %bb20.i ], [ %15, %bb30.i ], [ %15, %bb43.i ], [ %6, %bb48.i ], [ %15, %bb52.i ], [ %15, %bb95.i ], [ %15, %bb99.i ], [ %15, %bb103.i ] ; <float> [#uses=8]
  %16 = phi float [ 0x3F847AE140000000, %bb1.i ], [ %16, %bb20.i ], [ %16, %bb30.i ], [ %16, %bb43.i ], [ %16, %bb48.i ], [ 0.000000e+00, %bb52.i ], [ %16, %bb95.i ], [ %16, %bb99.i ], [ %16, %bb103.i ] ; <float> [#uses=8]
  %17 = phi float [ 1.000000e+01, %bb1.i ], [ %17, %bb20.i ], [ 1.000000e+00, %bb30.i ], [ %17, %bb43.i ], [ %17, %bb48.i ], [ %17, %bb52.i ], [ %17, %bb95.i ], [ %17, %bb99.i ], [ %17, %bb103.i ] ; <float> [#uses=8]
  %18 = icmp slt i32 undef, %argc                 ; <i1> [#uses=1]
  br i1 %18, label %bb2.i, label %bb135.i

bb135.i:                                          ; preds = %bb134.i
  br i1 undef, label %bb141.i, label %bb142.i

bb141.i:                                          ; preds = %bb135.i
  unreachable

bb142.i:                                          ; preds = %bb135.i
  br i1 undef, label %bb145.i, label %bb144.i

bb144.i:                                          ; preds = %bb142.i
  unreachable

bb145.i:                                          ; preds = %bb142.i
  br i1 undef, label %bb146.i, label %bb147.i

bb146.i:                                          ; preds = %bb145.i
  unreachable

bb147.i:                                          ; preds = %bb145.i
  br i1 undef, label %bb148.i, label %bb155.i

bb148.i:                                          ; preds = %bb147.i
  br label %bb155.i

bb155.i:                                          ; preds = %bb148.i, %bb147.i
  br i1 undef, label %bb156.i, label %bb161.i

bb156.i:                                          ; preds = %bb155.i
  unreachable

bb161.i:                                          ; preds = %bb155.i
  br i1 undef, label %bb162.i, label %bb163.i

bb162.i:                                          ; preds = %bb161.i
  %19 = fpext float %17 to double                 ; <double> [#uses=1]
  %20 = call i32 (i8*, ...)* @"\01_printf$LDBL128"(i8* getelementptr inbounds ([61 x i8]* @.str101, i32 0, i32 0), double %19) nounwind ; <i32> [#uses=0]
  unreachable

bb163.i:                                          ; preds = %bb161.i
  %21 = fpext float %16 to double                 ; <double> [#uses=1]
  %22 = call i32 (i8*, ...)* @"\01_printf$LDBL128"(i8* getelementptr inbounds ([31 x i8]* @.str104, i32 0, i32 0), double %21) nounwind ; <i32> [#uses=0]
  %23 = fpext float %15 to double                 ; <double> [#uses=1]
  %24 = call i32 (i8*, ...)* @"\01_printf$LDBL128"(i8* getelementptr inbounds ([45 x i8]* @.str105, i32 0, i32 0), double %23) nounwind ; <i32> [#uses=0]
  %25 = call i32 (i8*, ...)* @"\01_printf$LDBL128"(i8* getelementptr inbounds ([38 x i8]* @.str112, i32 0, i32 0), double undef) nounwind ; <i32> [#uses=0]
  br i1 undef, label %parse_command.exit, label %bb176.i

bb176.i:                                          ; preds = %bb163.i
  br i1 undef, label %bb177.i, label %bb178.i

bb177.i:                                          ; preds = %bb176.i
  unreachable

bb178.i:                                          ; preds = %bb176.i
  %26 = call i32 (i8*, ...)* @"\01_printf$LDBL128"(i8* getelementptr inbounds ([36 x i8]* @.str121, i32 0, i32 0), double undef) nounwind ; <i32> [#uses=0]
  %27 = fpext float %14 to double                 ; <double> [#uses=1]
  %28 = call i32 (i8*, ...)* @"\01_printf$LDBL128"(i8* getelementptr inbounds ([67 x i8]* @.str12293, i32 0, i32 0), double %27) nounwind ; <i32> [#uses=0]
  %29 = fpext float %13 to double                 ; <double> [#uses=1]
  %30 = call i32 (i8*, ...)* @"\01_printf$LDBL128"(i8* getelementptr inbounds ([68 x i8]* @.str123, i32 0, i32 0), double %29) nounwind ; <i32> [#uses=0]
  %31 = fpext float %12 to double                 ; <double> [#uses=1]
  %32 = call i32 (i8*, ...)* @"\01_printf$LDBL128"(i8* getelementptr inbounds ([52 x i8]* @.str124, i32 0, i32 0), double %31) nounwind ; <i32> [#uses=0]
  %33 = fpext float %11 to double                 ; <double> [#uses=1]
  %34 = call i32 (i8*, ...)* @"\01_printf$LDBL128"(i8* getelementptr inbounds ([51 x i8]* @.str125, i32 0, i32 0), double %33) nounwind ; <i32> [#uses=0]
  unreachable

parse_command.exit:                               ; preds = %bb163.i
  br i1 undef, label %bb4.i152.i, label %my_fopen.exit.i

bb4.i152.i:                                       ; preds = %parse_command.exit
  unreachable

my_fopen.exit.i:                                  ; preds = %parse_command.exit
  br i1 undef, label %bb.i6.i99, label %bb49.preheader.i.i

bb.i6.i99:                                        ; preds = %my_fopen.exit.i
  br i1 undef, label %bb3.i.i100, label %bb1.i8.i

bb1.i8.i:                                         ; preds = %bb.i6.i99
  unreachable

bb3.i.i100:                                       ; preds = %bb.i6.i99
  unreachable

bb49.preheader.i.i:                               ; preds = %my_fopen.exit.i
  br i1 undef, label %bb7.i11.i, label %bb50.i.i

bb7.i11.i:                                        ; preds = %bb49.preheader.i.i
  unreachable

bb50.i.i:                                         ; preds = %bb49.preheader.i.i
  br i1 undef, label %bb.i.i.i20.i, label %my_calloc.exit.i.i.i

bb.i.i.i20.i:                                     ; preds = %bb50.i.i
  unreachable

my_calloc.exit.i.i.i:                             ; preds = %bb50.i.i
  br i1 undef, label %bb.i.i37.i.i, label %alloc_hash_table.exit.i21.i

bb.i.i37.i.i:                                     ; preds = %my_calloc.exit.i.i.i
  unreachable

alloc_hash_table.exit.i21.i:                      ; preds = %my_calloc.exit.i.i.i
  br i1 undef, label %bb51.i.i, label %bb3.i23.i.i

bb51.i.i:                                         ; preds = %alloc_hash_table.exit.i21.i
  unreachable

bb3.i23.i.i:                                      ; preds = %alloc_hash_table.exit.i21.i
  br i1 undef, label %bb.i8.i.i, label %bb.nph.i.i

bb.nph.i.i:                                       ; preds = %bb3.i23.i.i
  unreachable

bb.i8.i.i:                                        ; preds = %bb3.i.i34.i, %bb3.i23.i.i
  br i1 undef, label %bb3.i.i34.i, label %bb1.i.i32.i

bb1.i.i32.i:                                      ; preds = %bb.i8.i.i
  unreachable

bb3.i.i34.i:                                      ; preds = %bb.i8.i.i
  br i1 undef, label %free_hash_table.exit.i.i, label %bb.i8.i.i

free_hash_table.exit.i.i:                         ; preds = %bb3.i.i34.i
  br i1 undef, label %check_netlist.exit.i, label %bb59.i.i

bb59.i.i:                                         ; preds = %free_hash_table.exit.i.i
  unreachable

check_netlist.exit.i:                             ; preds = %free_hash_table.exit.i.i
  br label %bb.i.i3.i

bb.i.i3.i:                                        ; preds = %bb3.i.i4.i, %check_netlist.exit.i
  br i1 false, label %bb3.i.i4.i, label %bb1.i.i.i122

bb1.i.i.i122:                                     ; preds = %bb1.i.i.i122, %bb.i.i3.i
  br i1 false, label %bb3.i.i4.i, label %bb1.i.i.i122

bb3.i.i4.i:                                       ; preds = %bb1.i.i.i122, %bb.i.i3.i
  br i1 undef, label %read_net.exit, label %bb.i.i3.i

read_net.exit:                                    ; preds = %bb3.i.i4.i
  br i1 undef, label %bb.i44, label %bb3.i47

bb.i44:                                           ; preds = %read_net.exit
  unreachable

bb3.i47:                                          ; preds = %read_net.exit
  br i1 false, label %bb9.i50, label %bb8.i49

bb8.i49:                                          ; preds = %bb3.i47
  unreachable

bb9.i50:                                          ; preds = %bb3.i47
  br i1 undef, label %bb11.i51, label %bb12.i52

bb11.i51:                                         ; preds = %bb9.i50
  unreachable

bb12.i52:                                         ; preds = %bb9.i50
  br i1 undef, label %bb.i.i53, label %my_malloc.exit.i54

bb.i.i53:                                         ; preds = %bb12.i52
  unreachable

my_malloc.exit.i54:                               ; preds = %bb12.i52
  br i1 undef, label %bb.i2.i55, label %my_malloc.exit3.i56

bb.i2.i55:                                        ; preds = %my_malloc.exit.i54
  unreachable

my_malloc.exit3.i56:                              ; preds = %my_malloc.exit.i54
  br i1 undef, label %bb.i.i.i57, label %my_malloc.exit.i.i

bb.i.i.i57:                                       ; preds = %my_malloc.exit3.i56
  unreachable

my_malloc.exit.i.i:                               ; preds = %my_malloc.exit3.i56
  br i1 undef, label %bb, label %bb10

bb:                                               ; preds = %my_malloc.exit.i.i
  unreachable

bb10:                                             ; preds = %my_malloc.exit.i.i
  br i1 false, label %bb12, label %bb11

bb11:                                             ; preds = %bb10
  unreachable

bb12:                                             ; preds = %bb10
  store float %annealing_sched.1.0, float* null, align 4
  store float %annealing_sched.2.0, float* undef, align 8
  store float %annealing_sched.3.0, float* undef, align 4
  store float %annealing_sched.4.0, float* undef, align 8
  store float %router_opts.0.0, float* undef, align 8
  store float %router_opts.1.0, float* undef, align 4
  store float %router_opts.2.0, float* null, align 8
  store float %router_opts.3.0, float* undef, align 4
  br i1 undef, label %place_and_route.exit, label %bb7.i22

bb7.i22:                                          ; preds = %bb12
  br i1 false, label %bb8.i23, label %bb9.i26

bb8.i23:                                          ; preds = %bb7.i22
  unreachable

bb9.i26:                                          ; preds = %bb7.i22
  unreachable

place_and_route.exit:                             ; preds = %bb12
  unreachable
}

declare i32 @"\01_printf$LDBL128"(i8*, ...) nounwind

declare i32 @strcmp(i8* nocapture, i8* nocapture) nounwind readonly

declare i32 @"\01_sscanf$LDBL128"(i8*, i8*, ...) nounwind
