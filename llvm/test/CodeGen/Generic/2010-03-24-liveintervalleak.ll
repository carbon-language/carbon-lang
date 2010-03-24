; RUN: llc <%s
%0 = type { i8, %1, i8, %1, i8, %1 }
%1 = type { i32 }
%2 = type { i16, i8, i8, i8, [3 x [10 x i8]], [29 x i8] }
%3 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%4 = type { i32, i32, i16, i16, %5, %6, i32, %8, [16 x %7], i32, i32, i32, i32 }
%5 = type { i32, i16, i16, i32, i32, i32, i16, i16 }
%6 = type { i16, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, i16, i16, i16, i16, i16, i32, i32, i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, [16 x %7] }
%7 = type { i32, i32 }
%8 = type { i16, i8, i8, i32, i32, i32, i32, i32, i64, i32, i32, i16, i16, i16, i16, i16, i16, i32, i32, i32, i32, i16, i16, i64, i64, i64, i64, i32, i32, [16 x %7] }
@glob2 = internal constant [39 x i8] c"ClamAV-Test-File-detected-via-bytecode\00" ; <[39 x i8]*> [#uses=1]
@glob9 = internal constant [5 x i8] c"EP: \00"    ; <[5 x i8]*> [#uses=1]
@glob10 = internal constant [27 x i8] c"Couldn't read 5 bytes @EP\0A\00" ; <[27 x i8]*> [#uses=1]
@glob11 = internal constant [46 x i8] c"No 'mov ebx, cyphertext' found at entrypoint\0A\00" ; <[46 x i8]*> [#uses=1]
@glob12 = internal constant [21 x i8] c"VA of cyphertext is \00" ; <[21 x i8]*> [#uses=1]
@glob13 = internal constant [22 x i8] c"RVA of cyphertext is \00" ; <[22 x i8]*> [#uses=1]
@glob14 = internal constant [51 x i8] c"Can't locate the phisical offset of the cyphertext\00" ; <[51 x i8]*> [#uses=1]
@glob15 = internal constant [22 x i8] c"Cyphertext starts at \00" ; <[22 x i8]*> [#uses=1]
@glob16 = internal constant [35 x i8] c"Can't read 10 bytes of cyphertext\0A\00" ; <[35 x i8]*> [#uses=1]
@glob17 = internal constant [11 x i8] c"HELLO WORM\00" ; <[11 x i8]*> [#uses=1]
declare i32 @llvm.bswap.i32(i32) nounwind readnone
declare i32 @memcmp(i8*, i8*, i64)
declare i32 @read([128 x i8]*, i8*, i32)
declare i32 @seek([128 x i8]*, i32, i32)
declare i32 @setvirusname([128 x i8]*, i8*, i32)
declare i32 @debug_print_str([128 x i8]*, i8*, i32)
declare i32 @debug_print_uint([128 x i8]*, i32)
declare i32 @pe_rawaddr([128 x i8]*, i32)
define internal fastcc i32 @bc0f0([128 x i8]*) nounwind ssp sspreq {
  %2 = alloca [5 x i8]                            ; <[5 x i8]*> [#uses=3]
  %3 = alloca [11 x i8]                           ; <[11 x i8]*> [#uses=5]
  %4 = getelementptr inbounds [128 x i8]* %0, i32 0, i32 120 ; <i8*> [#uses=1]
  %5 = bitcast i8* %4 to %4**                     ; <%4**> [#uses=1]
  %6 = load %4** %5                               ; <%4*> [#uses=1]
  %g3_1 = bitcast %4* %6 to i8*                   ; <i8*> [#uses=1]
  %7 = getelementptr i8* %g3_1, i32 64            ; <i8*> [#uses=1]
  %8 = bitcast i8* %7 to %4**                     ; <%4**> [#uses=1]
  %9 = getelementptr inbounds [128 x i8]* %0, i32 0, i32 120 ; <i8*> [#uses=1]
  %10 = bitcast i8* %9 to %4**                    ; <%4**> [#uses=1]
  %11 = load %4** %10                             ; <%4*> [#uses=1]
  %g3_2 = bitcast %4* %11 to i8*                  ; <i8*> [#uses=1]
  %12 = getelementptr i8* %g3_2, i32 4            ; <i8*> [#uses=1]
  %13 = bitcast i8* %12 to %4**                   ; <%4**> [#uses=1]
  %14 = bitcast %4** %13 to i32*                  ; <i32*> [#uses=1]
  %15 = load i32* %14                             ; <i32> [#uses=2]
  %16 = call i32 @debug_print_str([128 x i8]* %0, i8* getelementptr inbounds ([5 x i8]* @glob9, i32 0, i32 0), i32 0) ; <i32> [#uses=0]
  %17 = call i32 @debug_print_uint([128 x i8]* %0, i32 %15) ; <i32> [#uses=0]
  %18 = call i32 @seek([128 x i8]* %0, i32 %15, i32 0) ; <i32> [#uses=0]
  %19 = getelementptr [5 x i8]* %2, i32 0, i32 0  ; <i8*> [#uses=1]
  %20 = call i32 @read([128 x i8]* %0, i8* %19, i32 5) ; <i32> [#uses=1]
  %21 = icmp eq i32 %20, 5                        ; <i1> [#uses=1]
  br i1 %21, label %24, label %22
  %23 = call i32 @debug_print_str([128 x i8]* %0, i8* getelementptr inbounds ([27 x i8]* @glob10, i32 0, i32 0), i32 0) ; <i32> [#uses=0]
  ret i32 0
  %25 = getelementptr [5 x i8]* %2, i32 0, i32 0  ; <i8*> [#uses=1]
  %26 = load i8* %25                              ; <i8> [#uses=1]
  %27 = icmp eq i8 %26, -69                       ; <i1> [#uses=1]
  br i1 %27, label %30, label %28
  %29 = call i32 @debug_print_str([128 x i8]* %0, i8* getelementptr inbounds ([46 x i8]* @glob11, i32 0, i32 0), i32 0) ; <i32> [#uses=0]
  ret i32 0
  %31 = getelementptr [5 x i8]* %2, i32 0, i32 1  ; <i8*> [#uses=1]
  %32 = bitcast i8* %31 to i32*                   ; <i32*> [#uses=1]
  %33 = load i32* %32                             ; <i32> [#uses=2]
  br i1 false, label %34, label %36
  %35 = tail call i32 @llvm.bswap.i32(i32 %33) nounwind ; <i32> [#uses=1]
  br label %36
  %.06 = phi i32 [ %35, %34 ], [ %33, %30 ]       ; <i32> [#uses=2]
  %37 = call i32 @debug_print_str([128 x i8]* %0, i8* getelementptr inbounds ([21 x i8]* @glob12, i32 0, i32 0), i32 0) ; <i32> [#uses=0]
  %38 = call i32 @debug_print_uint([128 x i8]* %0, i32 %.06) ; <i32> [#uses=0]
  %39 = bitcast %4** %8 to i32*                   ; <i32*> [#uses=1]
  %40 = load i32* %39                             ; <i32> [#uses=1]
  %41 = sub i32 %.06, %40                         ; <i32> [#uses=2]
  %42 = call i32 @debug_print_str([128 x i8]* %0, i8* getelementptr inbounds ([22 x i8]* @glob13, i32 0, i32 0), i32 0) ; <i32> [#uses=0]
  %43 = call i32 @debug_print_uint([128 x i8]* %0, i32 %41) ; <i32> [#uses=0]
  %44 = call i32 @pe_rawaddr([128 x i8]* %0, i32 %41) ; <i32> [#uses=3]
  %45 = icmp eq i32 %44, -1                       ; <i1> [#uses=1]
  br i1 %45, label %46, label %48
  %47 = call i32 @debug_print_str([128 x i8]* %0, i8* getelementptr inbounds ([51 x i8]* @glob14, i32 0, i32 0), i32 0) ; <i32> [#uses=0]
  ret i32 0
  %49 = call i32 @debug_print_str([128 x i8]* %0, i8* getelementptr inbounds ([22 x i8]* @glob15, i32 0, i32 0), i32 0) ; <i32> [#uses=0]
  %50 = call i32 @debug_print_uint([128 x i8]* %0, i32 %44) ; <i32> [#uses=0]
  %51 = call i32 @seek([128 x i8]* %0, i32 %44, i32 0) ; <i32> [#uses=0]
  %52 = getelementptr [11 x i8]* %3, i32 0, i32 0 ; <i8*> [#uses=1]
  %53 = call i32 @read([128 x i8]* %0, i8* %52, i32 10) ; <i32> [#uses=1]
  %54 = icmp eq i32 %53, 10                       ; <i1> [#uses=1]
  br i1 %54, label %57, label %55
  %56 = call i32 @debug_print_str([128 x i8]* %0, i8* getelementptr inbounds ([35 x i8]* @glob16, i32 0, i32 0), i32 0) ; <i32> [#uses=0]
  ret i32 0
  %.05 = phi i32 [ 0, %48 ], [ %66, %65 ]         ; <i32> [#uses=4]
  %.0 = phi i8 [ 41, %48 ], [ %63, %65 ]          ; <i8> [#uses=1]
  %58 = getelementptr [11 x i8]* %3, i32 0, i32 %.05 ; <i8*> [#uses=2]
  %59 = icmp ult i32 %.05, 11                     ; <i1> [#uses=1]
  br i1 %59, label %60, label %79
  %61 = add i8 %.0, 1                             ; <i8> [#uses=1]
  %62 = load i8* %58                              ; <i8> [#uses=1]
  %63 = xor i8 %62, %61                           ; <i8> [#uses=2]
  %64 = icmp ult i32 %.05, 11                     ; <i1> [#uses=1]
  br i1 %64, label %65, label %79
  %66 = add i32 %.05, 1                           ; <i32> [#uses=2]
  %67 = icmp eq i32 %66, 10                       ; <i1> [#uses=1]
  br i1 %67, label %68, label %57
  %69 = getelementptr [11 x i8]* %3, i32 0, i32 0 ; <i8*> [#uses=1]
  %70 = tail call i32 @memcmp(i8* %69, i8* getelementptr inbounds ([11 x i8]* @glob17, i32 0, i32 0), i64 10) nounwind ; <i32> [#uses=1]
  %71 = icmp eq i32 %70, 0                        ; <i1> [#uses=1]
  br i1 %71, label %72, label %78
  %73 = getelementptr [11 x i8]* %3, i32 0, i32 10 ; <i8*> [#uses=1]
  %74 = bitcast i8* %73 to i1*                    ; <i1*> [#uses=1]
  %75 = getelementptr [11 x i8]* %3, i32 0, i32 0 ; <i8*> [#uses=1]
  %76 = call i32 @debug_print_str([128 x i8]* %0, i8* %75, i32 0) ; <i32> [#uses=0]
  %77 = call i32 @setvirusname([128 x i8]* %0, i8* getelementptr inbounds ([39 x i8]* @glob2, i32 0, i32 0), i32 0) ; <i32> [#uses=0]
  ret i32 0
  ret i32 0
  unreachable
}
