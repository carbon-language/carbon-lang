; RUN: opt -instcombine -S < %s | FileCheck -check-prefix=NODL %s
; RUN: opt -instcombine -S -default-data-layout="p:32:32:32-p1:16:16:16-n8:16:32:64" < %s | FileCheck -check-prefix=P32 %s

@G16 = internal constant [10 x i16] [i16 35, i16 82, i16 69, i16 81, i16 85,
                                     i16 73, i16 82, i16 69, i16 68, i16 0]

@G16_as1 = internal addrspace(1) constant [10 x i16] [i16 35, i16 82, i16 69, i16 81, i16 85,
                                                      i16 73, i16 82, i16 69, i16 68, i16 0]

@GD = internal constant [6 x double]
   [double -10.0, double 1.0, double 4.0, double 2.0, double -20.0, double -40.0]

%Foo = type { i32, i32, i32, i32 }

@GS = internal constant %Foo { i32 1, i32 4, i32 9, i32 14 }

@GStructArr = internal constant [4 x %Foo] [ %Foo { i32 1, i32 4, i32 9, i32 14 },
                                             %Foo { i32 5, i32 4, i32 6, i32 11 },
                                             %Foo { i32 6, i32 5, i32 9, i32 20 },
                                             %Foo { i32 12, i32 3, i32 9, i32 8 } ]


define i1 @test1(i32 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp eq i16 %Q, 0
  ret i1 %R
; NODL-LABEL: @test1(
; NODL-NEXT: %R = icmp eq i32 %X, 9
; NODL-NEXT: ret i1 %R

; P32-LABEL: @test1(
; P32-NEXT: %R = icmp eq i32 %X, 9
; P32-NEXT: ret i1 %R
}

define i1 @test1_noinbounds(i32 %X) {
  %P = getelementptr [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp eq i16 %Q, 0
  ret i1 %R
; NODL-LABEL: @test1_noinbounds(
; NODL-NEXT: %P = getelementptr [10 x i16]* @G16, i32 0, i32 %X

; P32-LABEL: @test1_noinbounds(
; P32-NEXT: %R = icmp eq i32 %X, 9
; P32-NEXT: ret i1 %R
}

define i1 @test1_noinbounds_i64(i64 %X) {
  %P = getelementptr [10 x i16]* @G16, i64 0, i64 %X
  %Q = load i16* %P
  %R = icmp eq i16 %Q, 0
  ret i1 %R
; NODL-LABEL: @test1_noinbounds_i64(
; NODL-NEXT: %P = getelementptr [10 x i16]* @G16, i64 0, i64 %X

; P32-LABEL: @test1_noinbounds_i64(
; P32: %R = icmp eq i32 %1, 9
; P32-NEXT: ret i1 %R
}

define i1 @test1_noinbounds_as1(i32 %x) {
  %p = getelementptr [10 x i16] addrspace(1)* @G16_as1, i16 0, i32 %x
  %q = load i16 addrspace(1)* %p
  %r = icmp eq i16 %q, 0
  ret i1 %r

; P32-LABEL: @test1_noinbounds_as1(
; P32-NEXT: trunc i32 %x to i16
; P32-NEXT: %r = icmp eq i16 %1, 9
; P32-NEXT: ret i1 %r
}

define i1 @test2(i32 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp slt i16 %Q, 85
  ret i1 %R
; NODL-LABEL: @test2(
; NODL-NEXT: %R = icmp ne i32 %X, 4
; NODL-NEXT: ret i1 %R
}

define i1 @test3(i32 %X) {
  %P = getelementptr inbounds [6 x double]* @GD, i32 0, i32 %X
  %Q = load double* %P
  %R = fcmp oeq double %Q, 1.0
  ret i1 %R
; NODL-LABEL: @test3(
; NODL-NEXT: %R = icmp eq i32 %X, 1
; NODL-NEXT: ret i1 %R

; P32-LABEL: @test3(
; P32-NEXT: %R = icmp eq i32 %X, 1
; P32-NEXT: ret i1 %R

}

define i1 @test4(i32 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp sle i16 %Q, 73
  ret i1 %R
; NODL-LABEL: @test4(
; NODL-NEXT: lshr i32 933, %X
; NODL-NEXT: and i32 {{.*}}, 1
; NODL-NEXT: %R = icmp ne i32 {{.*}}, 0
; NODL-NEXT: ret i1 %R

; P32-LABEL: @test4(
; P32-NEXT: lshr i32 933, %X
; P32-NEXT: and i32 {{.*}}, 1
; P32-NEXT: %R = icmp ne i32 {{.*}}, 0
; P32-NEXT: ret i1 %R
}

define i1 @test4_i16(i16 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i16 %X
  %Q = load i16* %P
  %R = icmp sle i16 %Q, 73
  ret i1 %R

; NODL-LABEL: @test4_i16(
; NODL-NEXT: lshr i16 933, %X
; NODL-NEXT: and i16 {{.*}}, 1
; NODL-NEXT: %R = icmp ne i16 {{.*}}, 0
; NODL-NEXT: ret i1 %R

; P32-LABEL: @test4_i16(
; P32-NEXT: sext i16 %X to i32
; P32-NEXT: lshr i32 933, %1
; P32-NEXT: and i32 {{.*}}, 1
; P32-NEXT: %R = icmp ne i32 {{.*}}, 0
; P32-NEXT: ret i1 %R
}

define i1 @test5(i32 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp eq i16 %Q, 69
  ret i1 %R
; NODL-LABEL: @test5(
; NODL-NEXT: icmp eq i32 %X, 2
; NODL-NEXT: icmp eq i32 %X, 7
; NODL-NEXT: %R = or i1
; NODL-NEXT: ret i1 %R

; P32-LABEL: @test5(
; P32-NEXT: icmp eq i32 %X, 2
; P32-NEXT: icmp eq i32 %X, 7
; P32-NEXT: %R = or i1
; P32-NEXT: ret i1 %R
}

define i1 @test6(i32 %X) {
  %P = getelementptr inbounds [6 x double]* @GD, i32 0, i32 %X
  %Q = load double* %P
  %R = fcmp ogt double %Q, 0.0
  ret i1 %R
; NODL-LABEL: @test6(
; NODL-NEXT: add i32 %X, -1
; NODL-NEXT: %R = icmp ult i32 {{.*}}, 3
; NODL-NEXT: ret i1 %R

; P32-LABEL: @test6(
; P32-NEXT: add i32 %X, -1
; P32-NEXT: %R = icmp ult i32 {{.*}}, 3
; P32-NEXT: ret i1 %R
}

define i1 @test7(i32 %X) {
  %P = getelementptr inbounds [6 x double]* @GD, i32 0, i32 %X
  %Q = load double* %P
  %R = fcmp olt double %Q, 0.0
  ret i1 %R
; NODL-LABEL: @test7(
; NODL-NEXT: add i32 %X, -1
; NODL-NEXT: %R = icmp ugt i32 {{.*}}, 2
; NODL-NEXT: ret i1 %R

; P32-LABEL: @test7(
; P32-NEXT: add i32 %X, -1
; P32-NEXT: %R = icmp ugt i32 {{.*}}, 2
; P32-NEXT: ret i1 %R
}

define i1 @test8(i32 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = and i16 %Q, 3
  %S = icmp eq i16 %R, 0
  ret i1 %S
; NODL-LABEL: @test8(
; NODL-NEXT: and i32 %X, -2
; NODL-NEXT: icmp eq i32 {{.*}}, 8
; NODL-NEXT: ret i1

; P32-LABEL: @test8(
; P32-NEXT: and i32 %X, -2
; P32-NEXT: icmp eq i32 {{.*}}, 8
; P32-NEXT: ret i1
}

@GA = internal constant [4 x { i32, i32 } ] [
  { i32, i32 } { i32 1, i32 0 },
  { i32, i32 } { i32 2, i32 1 },
  { i32, i32 } { i32 3, i32 1 },
  { i32, i32 } { i32 4, i32 0 }
]

define i1 @test9(i32 %X) {
  %P = getelementptr inbounds [4 x { i32, i32 } ]* @GA, i32 0, i32 %X, i32 1
  %Q = load i32* %P
  %R = icmp eq i32 %Q, 1
  ret i1 %R
; NODL-LABEL: @test9(
; NODL-NEXT: add i32 %X, -1
; NODL-NEXT: icmp ult i32 {{.*}}, 2
; NODL-NEXT: ret i1

; P32-LABEL: @test9(
; P32-NEXT: add i32 %X, -1
; P32-NEXT: icmp ult i32 {{.*}}, 2
; P32-NEXT: ret i1
}

define i1 @test10_struct(i32 %x) {
; NODL-LABEL: @test10_struct(
; NODL: getelementptr inbounds %Foo* @GS, i32 %x, i32 0

; P32-LABEL: @test10_struct(
; P32: getelementptr inbounds %Foo* @GS, i32 %x, i32 0
  %p = getelementptr inbounds %Foo* @GS, i32 %x, i32 0
  %q = load i32* %p
  %r = icmp eq i32 %q, 9
  ret i1 %r
}

define i1 @test10_struct_noinbounds(i32 %x) {
; NODL-LABEL: @test10_struct_noinbounds(
; NODL: getelementptr %Foo* @GS, i32 %x, i32 0

; P32-LABEL: @test10_struct_noinbounds(
; P32: getelementptr %Foo* @GS, i32 %x, i32 0
  %p = getelementptr %Foo* @GS, i32 %x, i32 0
  %q = load i32* %p
  %r = icmp eq i32 %q, 9
  ret i1 %r
}

; Test that the GEP indices are converted before we ever get here
; Index < ptr size
define i1 @test10_struct_i16(i16 %x){
; NODL-LABEL: @test10_struct_i16(
; NODL: getelementptr inbounds %Foo* @GS, i16 %x, i32 0

; P32-LABEL: @test10_struct_i16(
; P32: %1 = sext i16 %x to i32
; P32: getelementptr inbounds %Foo* @GS, i32 %1, i32 0
  %p = getelementptr inbounds %Foo* @GS, i16 %x, i32 0
  %q = load i32* %p
  %r = icmp eq i32 %q, 0
  ret i1 %r
}

; Test that the GEP indices are converted before we ever get here
; Index > ptr size
define i1 @test10_struct_i64(i64 %x){
; NODL-LABEL: @test10_struct_i64(
; NODL: getelementptr inbounds %Foo* @GS, i64 %x, i32 0

; P32-LABEL: @test10_struct_i64(
; P32: %1 = trunc i64 %x to i32
; P32: getelementptr inbounds %Foo* @GS, i32 %1, i32 0
  %p = getelementptr inbounds %Foo* @GS, i64 %x, i32 0
  %q = load i32* %p
  %r = icmp eq i32 %q, 0
  ret i1 %r
}

define i1 @test10_struct_noinbounds_i16(i16 %x) {
; NODL-LABEL: @test10_struct_noinbounds_i16(
; NODL: getelementptr %Foo* @GS, i16 %x, i32 0

; P32-LABEL: @test10_struct_noinbounds_i16(
; P32: %1 = sext i16 %x to i32
; P32: getelementptr %Foo* @GS, i32 %1, i32 0
  %p = getelementptr %Foo* @GS, i16 %x, i32 0
  %q = load i32* %p
  %r = icmp eq i32 %q, 0
  ret i1 %r
}

define i1 @test10_struct_arr(i32 %x) {
; NODL-LABEL: @test10_struct_arr(
; NODL-NEXT: %r = icmp ne i32 %x, 1
; NODL-NEXT: ret i1 %r

; P32-LABEL: @test10_struct_arr(
; P32-NEXT: %r = icmp ne i32 %x, 1
; P32-NEXT: ret i1 %r
  %p = getelementptr inbounds [4 x %Foo]* @GStructArr, i32 0, i32 %x, i32 2
  %q = load i32* %p
  %r = icmp eq i32 %q, 9
  ret i1 %r
}

define i1 @test10_struct_arr_noinbounds(i32 %x) {
; NODL-LABEL: @test10_struct_arr_noinbounds(
; NODL-NEXT  %p = getelementptr [4 x %Foo]* @GStructArr, i32 0, i32 %x, i32 2

; P32-LABEL: @test10_struct_arr_noinbounds(
; P32-NEXT  %p = getelementptr [4 x %Foo]* @GStructArr, i32 0, i32 %x, i32 2
  %p = getelementptr [4 x %Foo]* @GStructArr, i32 0, i32 %x, i32 2
  %q = load i32* %p
  %r = icmp eq i32 %q, 9
  ret i1 %r
}

define i1 @test10_struct_arr_i16(i16 %x) {
; NODL-LABEL: @test10_struct_arr_i16(
; NODL-NEXT: %r = icmp ne i16 %x, 1
; NODL-NEXT: ret i1 %r

; P32-LABEL: @test10_struct_arr_i16(
; P32-NEXT: %r = icmp ne i16 %x, 1
; P32-NEXT: ret i1 %r
  %p = getelementptr inbounds [4 x %Foo]* @GStructArr, i16 0, i16 %x, i32 2
  %q = load i32* %p
  %r = icmp eq i32 %q, 9
  ret i1 %r
}

define i1 @test10_struct_arr_i64(i64 %x) {
; NODL-LABEL: @test10_struct_arr_i64(
; NODL-NEXT: %r = icmp ne i64 %x, 1
; NODL-NEXT: ret i1 %r

; P32-LABEL: @test10_struct_arr_i64(
; P32-NEXT: trunc i64 %x to i32
; P32-NEXT: %r = icmp ne i32 %1, 1
; P32-NEXT: ret i1 %r
  %p = getelementptr inbounds [4 x %Foo]* @GStructArr, i64 0, i64 %x, i32 2
  %q = load i32* %p
  %r = icmp eq i32 %q, 9
  ret i1 %r
}

define i1 @test10_struct_arr_noinbounds_i16(i16 %x) {
; NODL-LABEL: @test10_struct_arr_noinbounds_i16(
; NODL-NEXT:  %p = getelementptr [4 x %Foo]* @GStructArr, i32 0, i16 %x, i32 2

; P32-LABEL: @test10_struct_arr_noinbounds_i16(
; P32-NEXT: %r = icmp ne i16 %x, 1
  %p = getelementptr [4 x %Foo]* @GStructArr, i32 0, i16 %x, i32 2
  %q = load i32* %p
  %r = icmp eq i32 %q, 9
  ret i1 %r
}

define i1 @test10_struct_arr_noinbounds_i64(i64 %x) {
; FIXME: Should be no trunc?
; NODL-LABEL: @test10_struct_arr_noinbounds_i64(
; NODL-NEXT:  %p = getelementptr [4 x %Foo]* @GStructArr, i32 0, i64 %x, i32 2

; P32-LABEL: @test10_struct_arr_noinbounds_i64(
; P32: %r = icmp ne i32 %1, 1
; P32-NEXT: ret i1 %r
  %p = getelementptr [4 x %Foo]* @GStructArr, i32 0, i64 %x, i32 2
  %q = load i32* %p
  %r = icmp eq i32 %q, 9
  ret i1 %r
}
