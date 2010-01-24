; RUN: llc < %s -o - -march=x86-64 -mattr=+sse42 -disable-mmx | FileCheck %s

; Test based on pr5626 to load/store
;

%i32vec3 = type <3 x i32>
define void @add3i32(%i32vec3*  sret %ret, %i32vec3* %ap, %i32vec3* %bp)  {
; CHECK: movaps
; CHECK: paddd
; CHECK: pextrd
; CHECK: movq
	%a = load %i32vec3* %ap, align 16
	%b = load %i32vec3* %bp, align 16
	%x = add %i32vec3 %a, %b
	store %i32vec3 %x, %i32vec3* %ret, align 16
	ret void
}

define void @add3i32_2(%i32vec3*  sret %ret, %i32vec3* %ap, %i32vec3* %bp)  {
; CHECK: movq
; CHECK: pinsrd
; CHECK: movq
; CHECK: pinsrd
; CHECK: paddd
; CHECK: pextrd
; CHECK: movq
	%a = load %i32vec3* %ap
	%b = load %i32vec3* %bp
	%x = add %i32vec3 %a, %b
	store %i32vec3 %x, %i32vec3* %ret
	ret void
}

%i32vec7 = type <7 x i32>
define void @add7i32(%i32vec7*  sret %ret, %i32vec7* %ap, %i32vec7* %bp)  {
; CHECK: movaps
; CHECK: movaps
; CHECK: paddd
; CHECK: paddd
; CHECK: pextrd
; CHECK: movq
; CHECK: movaps
	%a = load %i32vec7* %ap, align 16
	%b = load %i32vec7* %bp, align 16
	%x = add %i32vec7 %a, %b
	store %i32vec7 %x, %i32vec7* %ret, align 16
	ret void
}

%i32vec12 = type <12 x i32>
define void @add12i32(%i32vec12*  sret %ret, %i32vec12* %ap, %i32vec12* %bp)  {
; CHECK: movaps
; CHECK: movaps
; CHECK: movaps
; CHECK: paddd
; CHECK: paddd
; CHECK: paddd
; CHECK: movaps
; CHECK: movaps
; CHECK: movaps
	%a = load %i32vec12* %ap, align 16
	%b = load %i32vec12* %bp, align 16
	%x = add %i32vec12 %a, %b
	store %i32vec12 %x, %i32vec12* %ret, align 16
	ret void
}


%i16vec3 = type <3 x i16>
define void @add3i16(%i16vec3* nocapture sret %ret, %i16vec3* %ap, %i16vec3* %bp) nounwind {
; CHECK: movaps
; CHECK: paddw
; CHECK: movd
; CHECK: pextrw
	%a = load %i16vec3* %ap, align 16
	%b = load %i16vec3* %bp, align 16
	%x = add %i16vec3 %a, %b
	store %i16vec3 %x, %i16vec3* %ret, align 16
	ret void
}

%i16vec4 = type <4 x i16>
define void @add4i16(%i16vec4* nocapture sret %ret, %i16vec4* %ap, %i16vec4* %bp) nounwind {
; CHECK: movaps
; CHECK: paddw
; CHECK: movq
	%a = load %i16vec4* %ap, align 16
	%b = load %i16vec4* %bp, align 16
	%x = add %i16vec4 %a, %b
	store %i16vec4 %x, %i16vec4* %ret, align 16
	ret void
}

%i16vec12 = type <12 x i16>
define void @add12i16(%i16vec12* nocapture sret %ret, %i16vec12* %ap, %i16vec12* %bp) nounwind {
; CHECK: movaps
; CHECK: movaps
; CHECK: paddw
; CHECK: paddw
; CHECK: movq
; CHECK: movaps
	%a = load %i16vec12* %ap, align 16
	%b = load %i16vec12* %bp, align 16
	%x = add %i16vec12 %a, %b
	store %i16vec12 %x, %i16vec12* %ret, align 16
	ret void
}

%i16vec18 = type <18 x i16>
define void @add18i16(%i16vec18* nocapture sret %ret, %i16vec18* %ap, %i16vec18* %bp) nounwind {
; CHECK: movaps
; CHECK: movaps
; CHECK: movaps
; CHECK: paddw
; CHECK: paddw
; CHECK: paddw
; CHECK: movd
; CHECK: movaps
; CHECK: movaps
	%a = load %i16vec18* %ap, align 16
	%b = load %i16vec18* %bp, align 16
	%x = add %i16vec18 %a, %b
	store %i16vec18 %x, %i16vec18* %ret, align 16
	ret void
}


%i8vec3 = type <3 x i8>
define void @add3i8(%i8vec3* nocapture sret %ret, %i8vec3* %ap, %i8vec3* %bp) nounwind {
; CHECK: movaps
; CHECK: paddb
; CHECK: pextrb
; CHECK: movb
	%a = load %i8vec3* %ap, align 16
	%b = load %i8vec3* %bp, align 16
	%x = add %i8vec3 %a, %b
	store %i8vec3 %x, %i8vec3* %ret, align 16
	ret void
}

%i8vec31 = type <31 x i8>
define void @add31i8(%i8vec31* nocapture sret %ret, %i8vec31* %ap, %i8vec31* %bp) nounwind {
; CHECK: movaps
; CHECK: movaps
; CHECK: paddb
; CHECK: paddb
; CHECK: movq
; CHECK: pextrb
; CHECK: pextrw
	%a = load %i8vec31* %ap, align 16
	%b = load %i8vec31* %bp, align 16
	%x = add %i8vec31 %a, %b
	store %i8vec31 %x, %i8vec31* %ret, align 16
	ret void
}