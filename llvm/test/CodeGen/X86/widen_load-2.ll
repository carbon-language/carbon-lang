; RUN: llc < %s -o - -mcpu=generic -march=x86-64 -mattr=+sse4.2 | FileCheck %s

; Test based on pr5626 to load/store
;

%i32vec3 = type <3 x i32>
define void @add3i32(%i32vec3*  sret %ret, %i32vec3* %ap, %i32vec3* %bp)  {
; CHECK-LABEL: add3i32:
; CHECK:         movdqa  (%{{.*}}), %[[R0:xmm[0-9]+]]
; CHECK-NEXT:    paddd   (%{{.*}}), %[[R0]]
; CHECK-NEXT:    pextrd  $2, %[[R0]], 8(%{{.*}})
; CHECK-NEXT:    movq    %[[R0]], (%{{.*}})
	%a = load %i32vec3* %ap, align 16
	%b = load %i32vec3* %bp, align 16
	%x = add %i32vec3 %a, %b
	store %i32vec3 %x, %i32vec3* %ret, align 16
	ret void
}

define void @add3i32_2(%i32vec3*  sret %ret, %i32vec3* %ap, %i32vec3* %bp)  {
; CHECK-LABEL: add3i32_2:
; CHECK:         movq    (%{{.*}}), %[[R0:xmm[0-9]+]]
; CHECK-NEXT:    pinsrd  $2, 8(%{{.*}}), %[[R0]]
; CHECK-NEXT:    movq    (%{{.*}}), %[[R1:xmm[0-9]+]]
; CHECK-NEXT:    pinsrd  $2, 8(%{{.*}}), %[[R1]]
; CHECK-NEXT:    paddd   %[[R0]], %[[R1]]
; CHECK-NEXT:    pextrd  $2, %[[R1]], 8(%{{.*}})
; CHECK-NEXT:    movq    %[[R1]], (%{{.*}})
	%a = load %i32vec3* %ap, align 8
	%b = load %i32vec3* %bp, align 8
	%x = add %i32vec3 %a, %b
	store %i32vec3 %x, %i32vec3* %ret, align 8
	ret void
}

%i32vec7 = type <7 x i32>
define void @add7i32(%i32vec7*  sret %ret, %i32vec7* %ap, %i32vec7* %bp)  {
; CHECK-LABEL: add7i32:
; CHECK:         movdqa  (%{{.*}}), %[[R0:xmm[0-9]+]]
; CHECK-NEXT:    movdqa  16(%{{.*}}), %[[R1:xmm[0-9]+]]
; CHECK-NEXT:    paddd   (%{{.*}}), %[[R0]]
; CHECK-NEXT:    paddd   16(%{{.*}}), %[[R1]]
; CHECK-NEXT:    pextrd  $2, %[[R1]], 24(%{{.*}})
; CHECK-NEXT:    movq    %[[R1]], 16(%{{.*}})
; CHECK-NEXT:    movdqa  %[[R0]], (%{{.*}})
	%a = load %i32vec7* %ap, align 16
	%b = load %i32vec7* %bp, align 16
	%x = add %i32vec7 %a, %b
	store %i32vec7 %x, %i32vec7* %ret, align 16
	ret void
}

%i32vec12 = type <12 x i32>
define void @add12i32(%i32vec12*  sret %ret, %i32vec12* %ap, %i32vec12* %bp)  {
; CHECK-LABEL: add12i32:
; CHECK:         movdqa  (%{{.*}}), %[[R0:xmm[0-9]+]]
; CHECK-NEXT:    movdqa  16(%{{.*}}), %[[R1:xmm[0-9]+]]
; CHECK-NEXT:    movdqa  32(%{{.*}}), %[[R2:xmm[0-9]+]]
; CHECK-NEXT:    paddd   (%{{.*}}), %[[R0]]
; CHECK-NEXT:    paddd   16(%{{.*}}), %[[R1]]
; CHECK-NEXT:    paddd   32(%{{.*}}), %[[R2]]
; CHECK-NEXT:    movdqa  %[[R2]], 32(%{{.*}})
; CHECK-NEXT:    movdqa  %[[R1]], 16(%{{.*}})
; CHECK-NEXT:    movdqa  %[[R0]], (%{{.*}})
	%a = load %i32vec12* %ap, align 16
	%b = load %i32vec12* %bp, align 16
	%x = add %i32vec12 %a, %b
	store %i32vec12 %x, %i32vec12* %ret, align 16
	ret void
}


%i16vec3 = type <3 x i16>
define void @add3i16(%i16vec3* nocapture sret %ret, %i16vec3* %ap, %i16vec3* %bp) nounwind {
; CHECK-LABEL: add3i16:
; CHECK:         pmovzxwd (%{{.*}}), %[[R0:xmm[0-9]+]]
; CHECK-NEXT:    pmovzxwd (%{{.*}}), %[[R1:xmm[0-9]+]]
; CHECK-NEXT:    paddd    %[[R0]], %[[R1]]
; CHECK-NEXT:    movdqa   %[[R1]], %[[R0]]
; CHECK-NEXT:    pshufb   {{.*}}, %[[R0]]
; CHECK-NEXT:    pmovzxdq %[[R0]], %[[R0]]
; CHECK-NEXT:    pextrw   $4, %[[R1]], 4(%{{.*}})
; CHECK-NEXT:    movd     %[[R0]], (%{{.*}})
	%a = load %i16vec3* %ap, align 16
	%b = load %i16vec3* %bp, align 16
	%x = add %i16vec3 %a, %b
	store %i16vec3 %x, %i16vec3* %ret, align 16
	ret void
}

%i16vec4 = type <4 x i16>
define void @add4i16(%i16vec4* nocapture sret %ret, %i16vec4* %ap, %i16vec4* %bp) nounwind {
; CHECK-LABEL: add4i16:
; CHECK:         movq    (%{{.*}}), %[[R0:xmm[0-9]+]]
; CHECK-NEXT:    movq    (%{{.*}}), %[[R1:xmm[0-9]+]]
; CHECK-NEXT:    paddw   %[[R0]], %[[R1]]
; CHECK-NEXT:    movq    %[[R1]], (%{{.*}})
	%a = load %i16vec4* %ap, align 16
	%b = load %i16vec4* %bp, align 16
	%x = add %i16vec4 %a, %b
	store %i16vec4 %x, %i16vec4* %ret, align 16
	ret void
}

%i16vec12 = type <12 x i16>
define void @add12i16(%i16vec12* nocapture sret %ret, %i16vec12* %ap, %i16vec12* %bp) nounwind {
; CHECK-LABEL: add12i16:
; CHECK:         movdqa  (%{{.*}}), %[[R0:xmm[0-9]+]]
; CHECK-NEXT:    movdqa  16(%{{.*}}), %[[R1:xmm[0-9]+]]
; CHECK-NEXT:    paddw   (%{{.*}}), %[[R0]]
; CHECK-NEXT:    paddw   16(%{{.*}}), %[[R1]]
; CHECK-NEXT:    movq    %[[R1]], 16(%{{.*}})
; CHECK-NEXT:    movdqa  %[[R0]], (%{{.*}})
	%a = load %i16vec12* %ap, align 16
	%b = load %i16vec12* %bp, align 16
	%x = add %i16vec12 %a, %b
	store %i16vec12 %x, %i16vec12* %ret, align 16
	ret void
}

%i16vec18 = type <18 x i16>
define void @add18i16(%i16vec18* nocapture sret %ret, %i16vec18* %ap, %i16vec18* %bp) nounwind {
; CHECK-LABEL: add18i16:
; CHECK:         movdqa  (%{{.*}}), %[[R0:xmm[0-9]+]]
; CHECK-NEXT:    movdqa  16(%{{.*}}), %[[R1:xmm[0-9]+]]
; CHECK-NEXT:    movdqa  32(%{{.*}}), %[[R2:xmm[0-9]+]]
; CHECK-NEXT:    paddw   (%{{.*}}), %[[R0]]
; CHECK-NEXT:    paddw   16(%{{.*}}), %[[R1]]
; CHECK-NEXT:    paddw   32(%{{.*}}), %[[R2]]
; CHECK-NEXT:    movd    %[[R2]], 32(%{{.*}})
; CHECK-NEXT:    movdqa  %[[R1]], 16(%{{.*}})
; CHECK-NEXT:    movdqa  %[[R0]], (%{{.*}})
	%a = load %i16vec18* %ap, align 16
	%b = load %i16vec18* %bp, align 16
	%x = add %i16vec18 %a, %b
	store %i16vec18 %x, %i16vec18* %ret, align 16
	ret void
}


%i8vec3 = type <3 x i8>
define void @add3i8(%i8vec3* nocapture sret %ret, %i8vec3* %ap, %i8vec3* %bp) nounwind {
; CHECK-LABEL: add3i8:
; CHECK:         pmovzxbd (%{{.*}}), %[[R0:xmm[0-9]+]]
; CHECK-NEXT:    pmovzxbd (%{{.*}}), %[[R1:xmm[0-9]+]]
; CHECK-NEXT:    paddd    %[[R0]], %[[R1]]
; CHECK-NEXT:    movdqa   %[[R1]], %[[R0]]
; CHECK-NEXT:    pshufb   {{.*}}, %[[R0]]
; CHECK-NEXT:    pmovzxwq %[[R0]], %[[R0]]
; CHECK-NEXT:    pextrb   $8, %[[R1]], 2(%{{.*}})
; CHECK-NEXT:    movd     %[[R0]], %e[[R2:[abcd]]]x
; CHECK-NEXT:    movw     %[[R2]]x, (%{{.*}})
	%a = load %i8vec3* %ap, align 16
	%b = load %i8vec3* %bp, align 16
	%x = add %i8vec3 %a, %b
	store %i8vec3 %x, %i8vec3* %ret, align 16
	ret void
}

%i8vec31 = type <31 x i8>
define void @add31i8(%i8vec31* nocapture sret %ret, %i8vec31* %ap, %i8vec31* %bp) nounwind {
; CHECK-LABEL: add31i8:
; CHECK:         movdqa  (%{{.*}}), %[[R0:xmm[0-9]+]]
; CHECK-NEXT:    movdqa  16(%{{.*}}), %[[R1:xmm[0-9]+]]
; CHECK-NEXT:    paddb   (%{{.*}}), %[[R0]]
; CHECK-NEXT:    paddb   16(%{{.*}}), %[[R1]]
; CHECK-NEXT:    pextrb  $14, %[[R1]], 30(%{{.*}})
; CHECK-NEXT:    pextrw  $6, %[[R1]], 28(%{{.*}})
; CHECK-NEXT:    pextrd  $2, %[[R1]], 24(%{{.*}})
; CHECK-NEXT:    movq    %[[R1]], 16(%{{.*}})
; CHECK-NEXT:    movdqa  %[[R0]], (%{{.*}})
	%a = load %i8vec31* %ap, align 16
	%b = load %i8vec31* %bp, align 16
	%x = add %i8vec31 %a, %b
	store %i8vec31 %x, %i8vec31* %ret, align 16
	ret void
}


%i8vec3pack = type { <3 x i8>, i8 }
define void @rot(%i8vec3pack* nocapture sret %result, %i8vec3pack* %X, %i8vec3pack* %rot) nounwind {
; CHECK-LABEL: rot:
; CHECK:         movdqa  {{.*}}, %[[CONSTANT0:xmm[0-9]+]]
; CHECK-NEXT:    movdqa  {{.*}}, %[[SHUFFLE_MASK:xmm[0-9]+]]
; CHECK-NEXT:    pshufb  %[[SHUFFLE_MASK]], %[[CONSTANT0]]
; CHECK-NEXT:    pmovzxwq %[[CONSTANT0]], %[[CONSTANT0]]
; CHECK-NEXT:    movd    %[[CONSTANT0]], %e[[R0:[abcd]]]x
; CHECK-NEXT:    movw    %[[R0]]x, (%[[PTR0:.*]])
; CHECK-NEXT:    movb    $-98, 2(%[[PTR0]])
; CHECK-NEXT:    movdqa  {{.*}}, %[[CONSTANT1:xmm[0-9]+]]
; CHECK-NEXT:    pshufb  %[[SHUFFLE_MASK]], %[[CONSTANT1]]
; CHECK-NEXT:    pmovzxwq %[[CONSTANT1]], %[[CONSTANT1]]
; CHECK-NEXT:    movd    %[[CONSTANT1]], %e[[R1:[abcd]]]x
; CHECK-NEXT:    movw    %[[R1]]x, (%[[PTR1:.*]])
; CHECK-NEXT:    movb    $1, 2(%[[PTR1]])
; CHECK-NEXT:    pmovzxbd (%[[PTR0]]), %[[X0:xmm[0-9]+]]
; CHECK-NEXT:    pand    {{.*}}, %[[X0]]
; CHECK-NEXT:    pextrd  $1, %[[X0]], %e[[R0:[abcd]]]x
; CHECK-NEXT:    shrl    %e[[R0]]x
; CHECK-NEXT:    movd    %[[X0]], %e[[R1:[abcd]]]x
; CHECK-NEXT:    shrl    %e[[R1]]x
; CHECK-NEXT:    movd    %e[[R1]]x, %[[X1:xmm[0-9]+]]
; CHECK-NEXT:    pinsrd  $1, %e[[R0]]x, %[[X1]]
; CHECK-NEXT:    pextrd  $2, %[[X0]], %e[[R0:[abcd]]]x
; CHECK-NEXT:    shrl    %e[[R0]]x
; CHECK-NEXT:    pinsrd  $2, %e[[R0]]x, %[[X1]]
; CHECK-NEXT:    pextrd  $3, %[[X0]], %e[[R0:[abcd]]]x
; CHECK-NEXT:    pinsrd  $3, %e[[R0]]x, %[[X1]]
; CHECK-NEXT:    movdqa  %[[X1]], %[[X2:xmm[0-9]+]]
; CHECK-NEXT:    pshufb  %[[SHUFFLE_MASK]], %[[X2]]
; CHECK-NEXT:    pmovzxwq %[[X2]], %[[X3:xmm[0-9]+]]
; CHECK-NEXT:    pextrb  $8, %[[X1]], 2(%{{.*}})
; CHECK-NEXT:    movd    %[[X3]], %e[[R0:[abcd]]]x
; CHECK-NEXT:    movw    %[[R0]]x, (%{{.*}})

entry:
  %storetmp = bitcast %i8vec3pack* %X to <3 x i8>*
  store <3 x i8> <i8 -98, i8 -98, i8 -98>, <3 x i8>* %storetmp
  %storetmp1 = bitcast %i8vec3pack* %rot to <3 x i8>*
  store <3 x i8> <i8 1, i8 1, i8 1>, <3 x i8>* %storetmp1
  %tmp = load %i8vec3pack* %X
  %extractVec = extractvalue %i8vec3pack %tmp, 0
  %tmp2 = load %i8vec3pack* %rot
  %extractVec3 = extractvalue %i8vec3pack %tmp2, 0
  %shr = lshr <3 x i8> %extractVec, %extractVec3
  %storetmp4 = bitcast %i8vec3pack* %result to <3 x i8>*
  store <3 x i8> %shr, <3 x i8>* %storetmp4
  ret void
}

