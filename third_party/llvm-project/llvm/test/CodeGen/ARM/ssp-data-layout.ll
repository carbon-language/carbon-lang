; RUN: llc < %s -frame-pointer=all -mcpu=cortex-a8 -mtriple arm-linux-gnu -target-abi=apcs -o - | FileCheck %s
;  This test is fairly fragile.  The goal is to ensure that "large" stack
;  objects are allocated closest to the stack protector (i.e., farthest away
;  from the Stack Pointer.)  In standard SSP mode this means that large (>=
;  ssp-buffer-size) arrays and structures containing such arrays are
;  closet to the protector.  With sspstrong and sspreq this means large
;  arrays/structures-with-arrays are closest, followed by small (< ssp-buffer-size)
;  arrays/structures-with-arrays, and then addr-taken variables.
;
;  Ideally, we only want verify that the objects appear in the correct groups
;  and that the groups have the correct relative stack offset.  The ordering
;  within a group is not relevant to this test.  Unfortunately, there is not
;  an elegant way to do this, so just match the offset for each object.

%struct.struct_large_char = type { [8 x i8] }
%struct.struct_large_char2 = type { [2 x i8], [8 x i8] }
%struct.struct_small_char = type { [2 x i8] }
%struct.struct_large_nonchar = type { [8 x i32] }
%struct.struct_small_nonchar = type { [2 x i16] }

define void @layout_ssp() ssp {
entry:
; Expected stack layout for ssp is
;  180 large_char          . Group 1, nested arrays, arrays >= ssp-buffer-size
;  172 struct_large_char   .
;  168 scalar1             | Everything else
;  164 scalar2
;  160 scalar3
;  156 addr-of
;  152 small_nonchar
;  112 large_nonchar
;  110 small_char
;  108 struct_small_char
;   72 struct_large_nonchar
;   68 struct_small_nonchar

; CHECK: layout_ssp:

; CHECK: bl get_scalar1
; CHECK: str r0, [sp, #168]
; CHECK: bl end_scalar1

; CHECK: bl get_scalar2
; CHECK: str r0, [sp, #164]
; CHECK: bl end_scalar

; CHECK: bl get_scalar3
; CHECK: str r0, [sp, #160]
; CHECK: bl end_scalar3

; CHECK: bl get_addrof
; CHECK: str r0, [sp, #156]
; CHECK: bl end_addrof

; CHECK: get_small_nonchar
; CHECK: strh r0, [sp, #152]
; CHECK: bl end_small_nonchar

; CHECK: bl get_large_nonchar
; CHECK: str r0, [sp, #112]
; CHECK: bl end_large_nonchar

; CHECK: bl get_small_char
; CHECK: strb r0, [sp, #110]
; CHECK: bl end_small_char

; CHECK: bl get_large_char
; CHECK: strb r0, [sp, #180]
; CHECK: bl end_large_char

; CHECK: bl get_struct_large_char
; CHECK: strb r0, [sp, #172]
; CHECK: bl end_struct_large_char

; CHECK: bl get_struct_small_char
; CHECK: strb r0, [sp, #108]
; CHECK: bl end_struct_small_char

; CHECK: bl get_struct_large_nonchar
; CHECK:str r0, [sp, #72]
; CHECK: bl end_struct_large_nonchar

; CHECK: bl get_struct_small_nonchar
; CHECK: strh r0, [sp, #68]
; CHECK: bl end_struct_small_nonchar
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %ptr = alloca i32, align 4
  %small2 = alloca [2 x i16], align 2
  %large2 = alloca [8 x i32], align 16
  %small = alloca [2 x i8], align 1
  %large = alloca [8 x i8], align 1
  %a = alloca %struct.struct_large_char, align 1
  %b = alloca %struct.struct_small_char, align 1
  %c = alloca %struct.struct_large_nonchar, align 8
  %d = alloca %struct.struct_small_nonchar, align 2
  %call = call i32 @get_scalar1()
  store i32 %call, i32* %x, align 4
  call void @end_scalar1()
  %call1 = call i32 @get_scalar2()
  store i32 %call1, i32* %y, align 4
  call void @end_scalar2()
  %call2 = call i32 @get_scalar3()
  store i32 %call2, i32* %z, align 4
  call void @end_scalar3()
  %call3 = call i32 @get_addrof()
  store i32 %call3, i32* %ptr, align 4
  call void @end_addrof()
  %call4 = call signext i16 @get_small_nonchar()
  %arrayidx = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i64 0
  store i16 %call4, i16* %arrayidx, align 2
  call void @end_small_nonchar()
  %call5 = call i32 @get_large_nonchar()
  %arrayidx6 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i64 0
  store i32 %call5, i32* %arrayidx6, align 4
  call void @end_large_nonchar()
  %call7 = call signext i8 @get_small_char()
  %arrayidx8 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i64 0
  store i8 %call7, i8* %arrayidx8, align 1
  call void @end_small_char()
  %call9 = call signext i8 @get_large_char()
  %arrayidx10 = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i64 0
  store i8 %call9, i8* %arrayidx10, align 1
  call void @end_large_char()
  %call11 = call signext i8 @get_struct_large_char()
  %foo = getelementptr inbounds %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %arrayidx12 = getelementptr inbounds [8 x i8], [8 x i8]* %foo, i32 0, i64 0
  store i8 %call11, i8* %arrayidx12, align 1
  call void @end_struct_large_char()
  %call13 = call signext i8 @get_struct_small_char()
  %foo14 = getelementptr inbounds %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %arrayidx15 = getelementptr inbounds [2 x i8], [2 x i8]* %foo14, i32 0, i64 0
  store i8 %call13, i8* %arrayidx15, align 1
  call void @end_struct_small_char()
  %call16 = call i32 @get_struct_large_nonchar()
  %foo17 = getelementptr inbounds %struct.struct_large_nonchar, %struct.struct_large_nonchar* %c, i32 0, i32 0
  %arrayidx18 = getelementptr inbounds [8 x i32], [8 x i32]* %foo17, i32 0, i64 0
  store i32 %call16, i32* %arrayidx18, align 4
  call void @end_struct_large_nonchar()
  %call19 = call signext i16 @get_struct_small_nonchar()
  %foo20 = getelementptr inbounds %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %arrayidx21 = getelementptr inbounds [2 x i16], [2 x i16]* %foo20, i32 0, i64 0
  store i16 %call19, i16* %arrayidx21, align 2
  call void @end_struct_small_nonchar()
  %arraydecay = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i32 0
  %arraydecay22 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i32 0
  %arraydecay23 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i32 0
  %arraydecay24 = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i32 0
  %0 = load i32, i32* %x, align 4
  %1 = load i32, i32* %y, align 4
  %2 = load i32, i32* %z, align 4
  %coerce.dive = getelementptr %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %3 = bitcast [8 x i8]* %coerce.dive to i64*
  %4 = load i64, i64* %3, align 1
  %coerce.dive25 = getelementptr %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %5 = bitcast [2 x i8]* %coerce.dive25 to i16*
  %6 = load i16, i16* %5, align 1
  %coerce.dive26 = getelementptr %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %7 = bitcast [2 x i16]* %coerce.dive26 to i32*
  %8 = load i32, i32* %7, align 1
  call void @takes_all(i64 %4, i16 %6, %struct.struct_large_nonchar* byval(%struct.struct_large_nonchar) align 4 %c, i32 %8, i8* %arraydecay, i8* %arraydecay22, i32* %arraydecay23, i16* %arraydecay24, i32* %ptr, i32 %0, i32 %1, i32 %2)
  ret void
}

define void @layout_sspstrong() sspstrong {
entry:
; Expected stack layout for sspstrong is
; 144  large_nonchar          . Group 1, nested arrays,
; 136  large_char             .  arrays >= ssp-buffer-size
; 128  struct_large_char      .
; 96   struct_large_nonchar   .
; 84+8 small_non_char         | Group 2, nested arrays,
; 90   small_char             |  arrays < ssp-buffer-size
; 88   struct_small_char      |
; 84   struct_small_nonchar   |
; 80   addrof                 * Group 3, addr-of local
; 76   scalar1                + Group 4, everything else
; 72   scalar2                +
; 68   scalar3                +
;
; CHECK: layout_sspstrong:

; CHECK: bl get_scalar1
; CHECK: str r0, [sp, #76]
; CHECK: bl end_scalar1

; CHECK: bl get_scalar2
; CHECK: str r0, [sp, #72]
; CHECK: bl end_scalar2

; CHECK: bl get_scalar3
; CHECK: str r0, [sp, #68]
; CHECK: bl end_scalar3

; CHECK: bl get_addrof
; CHECK: str r0, [sp, #80]
; CHECK: bl end_addrof

; CHECK: get_small_nonchar
; CHECK: strh r0, [sp, #92]
; CHECK: bl end_small_nonchar

; CHECK: bl get_large_nonchar
; CHECK: str r0, [sp, #144]
; CHECK: bl end_large_nonchar

; CHECK: bl get_small_char
; CHECK: strb r0, [sp, #90]
; CHECK: bl end_small_char

; CHECK: bl get_large_char
; CHECK: strb r0, [sp, #136]
; CHECK: bl end_large_char

; CHECK: bl get_struct_large_char
; CHECK: strb r0, [sp, #128]
; CHECK: bl end_struct_large_char

; CHECK: bl get_struct_small_char
; CHECK: strb r0, [sp, #88]
; CHECK: bl end_struct_small_char

; CHECK: bl get_struct_large_nonchar
; CHECK: str r0, [sp, #96]
; CHECK: bl end_struct_large_nonchar

; CHECK: bl get_struct_small_nonchar
; CHECK: strh r0, [sp, #84]
; CHECK: bl end_struct_small_nonchar
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %ptr = alloca i32, align 4
  %small2 = alloca [2 x i16], align 2
  %large2 = alloca [8 x i32], align 16
  %small = alloca [2 x i8], align 1
  %large = alloca [8 x i8], align 1
  %a = alloca %struct.struct_large_char, align 1
  %b = alloca %struct.struct_small_char, align 1
  %c = alloca %struct.struct_large_nonchar, align 8
  %d = alloca %struct.struct_small_nonchar, align 2
  %call = call i32 @get_scalar1()
  store i32 %call, i32* %x, align 4
  call void @end_scalar1()
  %call1 = call i32 @get_scalar2()
  store i32 %call1, i32* %y, align 4
  call void @end_scalar2()
  %call2 = call i32 @get_scalar3()
  store i32 %call2, i32* %z, align 4
  call void @end_scalar3()
  %call3 = call i32 @get_addrof()
  store i32 %call3, i32* %ptr, align 4
  call void @end_addrof()
  %call4 = call signext i16 @get_small_nonchar()
  %arrayidx = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i64 0
  store i16 %call4, i16* %arrayidx, align 2
  call void @end_small_nonchar()
  %call5 = call i32 @get_large_nonchar()
  %arrayidx6 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i64 0
  store i32 %call5, i32* %arrayidx6, align 4
  call void @end_large_nonchar()
  %call7 = call signext i8 @get_small_char()
  %arrayidx8 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i64 0
  store i8 %call7, i8* %arrayidx8, align 1
  call void @end_small_char()
  %call9 = call signext i8 @get_large_char()
  %arrayidx10 = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i64 0
  store i8 %call9, i8* %arrayidx10, align 1
  call void @end_large_char()
  %call11 = call signext i8 @get_struct_large_char()
  %foo = getelementptr inbounds %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %arrayidx12 = getelementptr inbounds [8 x i8], [8 x i8]* %foo, i32 0, i64 0
  store i8 %call11, i8* %arrayidx12, align 1
  call void @end_struct_large_char()
  %call13 = call signext i8 @get_struct_small_char()
  %foo14 = getelementptr inbounds %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %arrayidx15 = getelementptr inbounds [2 x i8], [2 x i8]* %foo14, i32 0, i64 0
  store i8 %call13, i8* %arrayidx15, align 1
  call void @end_struct_small_char()
  %call16 = call i32 @get_struct_large_nonchar()
  %foo17 = getelementptr inbounds %struct.struct_large_nonchar, %struct.struct_large_nonchar* %c, i32 0, i32 0
  %arrayidx18 = getelementptr inbounds [8 x i32], [8 x i32]* %foo17, i32 0, i64 0
  store i32 %call16, i32* %arrayidx18, align 4
  call void @end_struct_large_nonchar()
  %call19 = call signext i16 @get_struct_small_nonchar()
  %foo20 = getelementptr inbounds %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %arrayidx21 = getelementptr inbounds [2 x i16], [2 x i16]* %foo20, i32 0, i64 0
  store i16 %call19, i16* %arrayidx21, align 2
  call void @end_struct_small_nonchar()
  %arraydecay = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i32 0
  %arraydecay22 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i32 0
  %arraydecay23 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i32 0
  %arraydecay24 = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i32 0
  %0 = load i32, i32* %x, align 4
  %1 = load i32, i32* %y, align 4
  %2 = load i32, i32* %z, align 4
  %coerce.dive = getelementptr %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %3 = bitcast [8 x i8]* %coerce.dive to i64*
  %4 = load i64, i64* %3, align 1
  %coerce.dive25 = getelementptr %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %5 = bitcast [2 x i8]* %coerce.dive25 to i16*
  %6 = load i16, i16* %5, align 1
  %coerce.dive26 = getelementptr %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %7 = bitcast [2 x i16]* %coerce.dive26 to i32*
  %8 = load i32, i32* %7, align 1
  call void @takes_all(i64 %4, i16 %6, %struct.struct_large_nonchar* byval(%struct.struct_large_nonchar) align 4 %c, i32 %8, i8* %arraydecay, i8* %arraydecay22, i32* %arraydecay23, i16* %arraydecay24, i32* %ptr, i32 %0, i32 %1, i32 %2)
  ret void
}

define void @layout_sspreq() sspreq {
entry:
; Expected stack layout for sspreq is the same as sspstrong
;
; CHECK: layout_sspreq:

; CHECK: bl get_scalar1
; CHECK: str r0, [sp, #76]
; CHECK: bl end_scalar1

; CHECK: bl get_scalar2
; CHECK: str r0, [sp, #72]
; CHECK: bl end_scalar2

; CHECK: bl get_scalar3
; CHECK: str r0, [sp, #68]
; CHECK: bl end_scalar3

; CHECK: bl get_addrof
; CHECK: str r0, [sp, #80]
; CHECK: bl end_addrof

; CHECK: get_small_nonchar
; CHECK: strh r0, [sp, #92]
; CHECK: bl end_small_nonchar

; CHECK: bl get_large_nonchar
; CHECK: str r0, [sp, #144]
; CHECK: bl end_large_nonchar

; CHECK: bl get_small_char
; CHECK: strb r0, [sp, #90]
; CHECK: bl end_small_char

; CHECK: bl get_large_char
; CHECK: strb r0, [sp, #136]
; CHECK: bl end_large_char

; CHECK: bl get_struct_large_char
; CHECK: strb r0, [sp, #128]
; CHECK: bl end_struct_large_char

; CHECK: bl get_struct_small_char
; CHECK: strb r0, [sp, #88]
; CHECK: bl end_struct_small_char

; CHECK: bl get_struct_large_nonchar
; CHECK: str r0, [sp, #96]
; CHECK: bl end_struct_large_nonchar

; CHECK: bl get_struct_small_nonchar
; CHECK: strh r0, [sp, #84]
; CHECK: bl end_struct_small_nonchar
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %ptr = alloca i32, align 4
  %small2 = alloca [2 x i16], align 2
  %large2 = alloca [8 x i32], align 16
  %small = alloca [2 x i8], align 1
  %large = alloca [8 x i8], align 1
  %a = alloca %struct.struct_large_char, align 1
  %b = alloca %struct.struct_small_char, align 1
  %c = alloca %struct.struct_large_nonchar, align 8
  %d = alloca %struct.struct_small_nonchar, align 2
  %call = call i32 @get_scalar1()
  store i32 %call, i32* %x, align 4
  call void @end_scalar1()
  %call1 = call i32 @get_scalar2()
  store i32 %call1, i32* %y, align 4
  call void @end_scalar2()
  %call2 = call i32 @get_scalar3()
  store i32 %call2, i32* %z, align 4
  call void @end_scalar3()
  %call3 = call i32 @get_addrof()
  store i32 %call3, i32* %ptr, align 4
  call void @end_addrof()
  %call4 = call signext i16 @get_small_nonchar()
  %arrayidx = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i64 0
  store i16 %call4, i16* %arrayidx, align 2
  call void @end_small_nonchar()
  %call5 = call i32 @get_large_nonchar()
  %arrayidx6 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i64 0
  store i32 %call5, i32* %arrayidx6, align 4
  call void @end_large_nonchar()
  %call7 = call signext i8 @get_small_char()
  %arrayidx8 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i64 0
  store i8 %call7, i8* %arrayidx8, align 1
  call void @end_small_char()
  %call9 = call signext i8 @get_large_char()
  %arrayidx10 = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i64 0
  store i8 %call9, i8* %arrayidx10, align 1
  call void @end_large_char()
  %call11 = call signext i8 @get_struct_large_char()
  %foo = getelementptr inbounds %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %arrayidx12 = getelementptr inbounds [8 x i8], [8 x i8]* %foo, i32 0, i64 0
  store i8 %call11, i8* %arrayidx12, align 1
  call void @end_struct_large_char()
  %call13 = call signext i8 @get_struct_small_char()
  %foo14 = getelementptr inbounds %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %arrayidx15 = getelementptr inbounds [2 x i8], [2 x i8]* %foo14, i32 0, i64 0
  store i8 %call13, i8* %arrayidx15, align 1
  call void @end_struct_small_char()
  %call16 = call i32 @get_struct_large_nonchar()
  %foo17 = getelementptr inbounds %struct.struct_large_nonchar, %struct.struct_large_nonchar* %c, i32 0, i32 0
  %arrayidx18 = getelementptr inbounds [8 x i32], [8 x i32]* %foo17, i32 0, i64 0
  store i32 %call16, i32* %arrayidx18, align 4
  call void @end_struct_large_nonchar()
  %call19 = call signext i16 @get_struct_small_nonchar()
  %foo20 = getelementptr inbounds %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %arrayidx21 = getelementptr inbounds [2 x i16], [2 x i16]* %foo20, i32 0, i64 0
  store i16 %call19, i16* %arrayidx21, align 2
  call void @end_struct_small_nonchar()
  %arraydecay = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i32 0
  %arraydecay22 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i32 0
  %arraydecay23 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i32 0
  %arraydecay24 = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i32 0
  %0 = load i32, i32* %x, align 4
  %1 = load i32, i32* %y, align 4
  %2 = load i32, i32* %z, align 4
  %coerce.dive = getelementptr %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %3 = bitcast [8 x i8]* %coerce.dive to i64*
  %4 = load i64, i64* %3, align 1
  %coerce.dive25 = getelementptr %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %5 = bitcast [2 x i8]* %coerce.dive25 to i16*
  %6 = load i16, i16* %5, align 1
  %coerce.dive26 = getelementptr %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %7 = bitcast [2 x i16]* %coerce.dive26 to i32*
  %8 = load i32, i32* %7, align 1
  call void @takes_all(i64 %4, i16 %6, %struct.struct_large_nonchar* byval(%struct.struct_large_nonchar) align 4 %c, i32 %8, i8* %arraydecay, i8* %arraydecay22, i32* %arraydecay23, i16* %arraydecay24, i32* %ptr, i32 %0, i32 %1, i32 %2)
  ret void
}

define void @struct_with_protectable_arrays() sspstrong {
entry:
; Check to ensure that a structure which contains a small array followed by a
; large array is assigned to the stack properly as a large object.
; CHECK: struct_with_protectable_arrays:
; CHECK: bl get_struct_small_char
; CHECK: strb r0, [sp, #68]
; CHECK: bl end_struct_small_char
; CHECK: bl get_struct_large_char2
; CHECK: strb r0, [sp, #106]
; CHECK: bl end_struct_large_char2
  %a = alloca %struct.struct_small_char, align 1
  %b = alloca %struct.struct_large_char2, align 1
  %d1 = alloca %struct.struct_large_nonchar, align 8
  %d2 = alloca %struct.struct_small_nonchar, align 2
  %call = call signext i8 @get_struct_small_char()
  %foo = getelementptr inbounds %struct.struct_small_char, %struct.struct_small_char* %a, i32 0, i32 0
  %arrayidx = getelementptr inbounds [2 x i8], [2 x i8]* %foo, i32 0, i64 0
  store i8 %call, i8* %arrayidx, align 1
  call void @end_struct_small_char()
  %call1 = call signext i8 @get_struct_large_char2()
  %foo2 = getelementptr inbounds %struct.struct_large_char2, %struct.struct_large_char2* %b, i32 0, i32 1
  %arrayidx3 = getelementptr inbounds [8 x i8], [8 x i8]* %foo2, i32 0, i64 0
  store i8 %call1, i8* %arrayidx3, align 1
  call void @end_struct_large_char2()
  %0 = bitcast %struct.struct_large_char2* %b to %struct.struct_large_char*
  %coerce.dive = getelementptr %struct.struct_large_char, %struct.struct_large_char* %0, i32 0, i32 0
  %1 = bitcast [8 x i8]* %coerce.dive to i64*
  %2 = load i64, i64* %1, align 1
  %coerce.dive4 = getelementptr %struct.struct_small_char, %struct.struct_small_char* %a, i32 0, i32 0
  %3 = bitcast [2 x i8]* %coerce.dive4 to i16*
  %4 = load i16, i16* %3, align 1
  %coerce.dive5 = getelementptr %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d2, i32 0, i32 0
  %5 = bitcast [2 x i16]* %coerce.dive5 to i32*
  %6 = load i32, i32* %5, align 1
  call void @takes_all(i64 %2, i16 %4, %struct.struct_large_nonchar* byval(%struct.struct_large_nonchar) align 4 %d1, i32 %6, i8* null, i8* null, i32* null, i16* null, i32* null, i32 0, i32 0, i32 0)
  ret void
}

declare i32 @get_scalar1()
declare void @end_scalar1()

declare i32 @get_scalar2()
declare void @end_scalar2()

declare i32 @get_scalar3()
declare void @end_scalar3()

declare i32 @get_addrof()
declare void @end_addrof()

declare signext i16 @get_small_nonchar()
declare void @end_small_nonchar()

declare i32 @get_large_nonchar()
declare void @end_large_nonchar()

declare signext i8 @get_small_char()
declare void @end_small_char()

declare signext i8 @get_large_char()
declare void @end_large_char()

declare signext i8 @get_struct_large_char()
declare void @end_struct_large_char()

declare signext i8 @get_struct_large_char2()
declare void @end_struct_large_char2()

declare signext i8 @get_struct_small_char()
declare void @end_struct_small_char()

declare i32 @get_struct_large_nonchar()
declare void @end_struct_large_nonchar()

declare signext i16 @get_struct_small_nonchar()
declare void @end_struct_small_nonchar()

declare void @takes_all(i64, i16, %struct.struct_large_nonchar* byval(%struct.struct_large_nonchar) align 8, i32, i8*, i8*, i32*, i16*, i32*, i32, i32, i32)
