; RUN: llc  < %s -march=mipsel | FileCheck %s

@a0 = external global i32
@b0 = external global i32*
@a1 = external global i32
@b1 = external global i32*
@a2 = external global i32
@b2 = external global i32*
@a3 = external global i32
@b3 = external global i32*
@a4 = external global i32
@b4 = external global i32*
@a5 = external global i32
@b5 = external global i32*
@a6 = external global i32
@b6 = external global i32*
@a7 = external global i32
@b7 = external global i32*
@a8 = external global i32
@b8 = external global i32*
@a9 = external global i32
@b9 = external global i32*
@a10 = external global i32
@b10 = external global i32*
@a11 = external global i32
@b11 = external global i32*
@a12 = external global i32
@b12 = external global i32*
@a13 = external global i32
@b13 = external global i32*
@a14 = external global i32
@b14 = external global i32*
@a15 = external global i32
@b15 = external global i32*
@a16 = external global i32
@b16 = external global i32*
@a17 = external global i32
@b17 = external global i32*
@a18 = external global i32
@b18 = external global i32*
@a19 = external global i32
@b19 = external global i32*
@a20 = external global i32
@b20 = external global i32*
@a21 = external global i32
@b21 = external global i32*
@a22 = external global i32
@b22 = external global i32*
@a23 = external global i32
@b23 = external global i32*
@a24 = external global i32
@b24 = external global i32*
@a25 = external global i32
@b25 = external global i32*
@a26 = external global i32
@b26 = external global i32*
@a27 = external global i32
@b27 = external global i32*
@a28 = external global i32
@b28 = external global i32*
@a29 = external global i32
@b29 = external global i32*
@c0 = external global i32*
@c1 = external global i32*
@c2 = external global i32*
@c3 = external global i32*
@c4 = external global i32*
@c5 = external global i32*
@c6 = external global i32*
@c7 = external global i32*
@c8 = external global i32*
@c9 = external global i32*
@c10 = external global i32*
@c11 = external global i32*
@c12 = external global i32*
@c13 = external global i32*
@c14 = external global i32*
@c15 = external global i32*
@c16 = external global i32*
@c17 = external global i32*
@c18 = external global i32*
@c19 = external global i32*
@c20 = external global i32*
@c21 = external global i32*
@c22 = external global i32*
@c23 = external global i32*
@c24 = external global i32*
@c25 = external global i32*
@c26 = external global i32*
@c27 = external global i32*
@c28 = external global i32*
@c29 = external global i32*

define i32 @f1() nounwind {
entry:
; CHECK: sw  $ra, {{[0-9]+}}($sp)            # 4-byte Folded Spill
; CHECK: $ra
; CHECK: lw  $ra, {{[0-9]+}}($sp)            # 4-byte Folded Reload
; CHECK: jr  $ra

  %0 = load i32* @a0, align 4
  %1 = load i32** @b0, align 4
  store i32 %0, i32* %1, align 4
  %2 = load i32* @a1, align 4
  %3 = load i32** @b1, align 4
  store i32 %2, i32* %3, align 4
  %4 = load i32* @a2, align 4
  %5 = load i32** @b2, align 4
  store i32 %4, i32* %5, align 4
  %6 = load i32* @a3, align 4
  %7 = load i32** @b3, align 4
  store i32 %6, i32* %7, align 4
  %8 = load i32* @a4, align 4
  %9 = load i32** @b4, align 4
  store i32 %8, i32* %9, align 4
  %10 = load i32* @a5, align 4
  %11 = load i32** @b5, align 4
  store i32 %10, i32* %11, align 4
  %12 = load i32* @a6, align 4
  %13 = load i32** @b6, align 4
  store i32 %12, i32* %13, align 4
  %14 = load i32* @a7, align 4
  %15 = load i32** @b7, align 4
  store i32 %14, i32* %15, align 4
  %16 = load i32* @a8, align 4
  %17 = load i32** @b8, align 4
  store i32 %16, i32* %17, align 4
  %18 = load i32* @a9, align 4
  %19 = load i32** @b9, align 4
  store i32 %18, i32* %19, align 4
  %20 = load i32* @a10, align 4
  %21 = load i32** @b10, align 4
  store i32 %20, i32* %21, align 4
  %22 = load i32* @a11, align 4
  %23 = load i32** @b11, align 4
  store i32 %22, i32* %23, align 4
  %24 = load i32* @a12, align 4
  %25 = load i32** @b12, align 4
  store i32 %24, i32* %25, align 4
  %26 = load i32* @a13, align 4
  %27 = load i32** @b13, align 4
  store i32 %26, i32* %27, align 4
  %28 = load i32* @a14, align 4
  %29 = load i32** @b14, align 4
  store i32 %28, i32* %29, align 4
  %30 = load i32* @a15, align 4
  %31 = load i32** @b15, align 4
  store i32 %30, i32* %31, align 4
  %32 = load i32* @a16, align 4
  %33 = load i32** @b16, align 4
  store i32 %32, i32* %33, align 4
  %34 = load i32* @a17, align 4
  %35 = load i32** @b17, align 4
  store i32 %34, i32* %35, align 4
  %36 = load i32* @a18, align 4
  %37 = load i32** @b18, align 4
  store i32 %36, i32* %37, align 4
  %38 = load i32* @a19, align 4
  %39 = load i32** @b19, align 4
  store i32 %38, i32* %39, align 4
  %40 = load i32* @a20, align 4
  %41 = load i32** @b20, align 4
  store i32 %40, i32* %41, align 4
  %42 = load i32* @a21, align 4
  %43 = load i32** @b21, align 4
  store i32 %42, i32* %43, align 4
  %44 = load i32* @a22, align 4
  %45 = load i32** @b22, align 4
  store i32 %44, i32* %45, align 4
  %46 = load i32* @a23, align 4
  %47 = load i32** @b23, align 4
  store i32 %46, i32* %47, align 4
  %48 = load i32* @a24, align 4
  %49 = load i32** @b24, align 4
  store i32 %48, i32* %49, align 4
  %50 = load i32* @a25, align 4
  %51 = load i32** @b25, align 4
  store i32 %50, i32* %51, align 4
  %52 = load i32* @a26, align 4
  %53 = load i32** @b26, align 4
  store i32 %52, i32* %53, align 4
  %54 = load i32* @a27, align 4
  %55 = load i32** @b27, align 4
  store i32 %54, i32* %55, align 4
  %56 = load i32* @a28, align 4
  %57 = load i32** @b28, align 4
  store i32 %56, i32* %57, align 4
  %58 = load i32* @a29, align 4
  %59 = load i32** @b29, align 4
  store i32 %58, i32* %59, align 4
  %60 = load i32* @a0, align 4
  %61 = load i32** @c0, align 4
  store i32 %60, i32* %61, align 4
  %62 = load i32* @a1, align 4
  %63 = load i32** @c1, align 4
  store i32 %62, i32* %63, align 4
  %64 = load i32* @a2, align 4
  %65 = load i32** @c2, align 4
  store i32 %64, i32* %65, align 4
  %66 = load i32* @a3, align 4
  %67 = load i32** @c3, align 4
  store i32 %66, i32* %67, align 4
  %68 = load i32* @a4, align 4
  %69 = load i32** @c4, align 4
  store i32 %68, i32* %69, align 4
  %70 = load i32* @a5, align 4
  %71 = load i32** @c5, align 4
  store i32 %70, i32* %71, align 4
  %72 = load i32* @a6, align 4
  %73 = load i32** @c6, align 4
  store i32 %72, i32* %73, align 4
  %74 = load i32* @a7, align 4
  %75 = load i32** @c7, align 4
  store i32 %74, i32* %75, align 4
  %76 = load i32* @a8, align 4
  %77 = load i32** @c8, align 4
  store i32 %76, i32* %77, align 4
  %78 = load i32* @a9, align 4
  %79 = load i32** @c9, align 4
  store i32 %78, i32* %79, align 4
  %80 = load i32* @a10, align 4
  %81 = load i32** @c10, align 4
  store i32 %80, i32* %81, align 4
  %82 = load i32* @a11, align 4
  %83 = load i32** @c11, align 4
  store i32 %82, i32* %83, align 4
  %84 = load i32* @a12, align 4
  %85 = load i32** @c12, align 4
  store i32 %84, i32* %85, align 4
  %86 = load i32* @a13, align 4
  %87 = load i32** @c13, align 4
  store i32 %86, i32* %87, align 4
  %88 = load i32* @a14, align 4
  %89 = load i32** @c14, align 4
  store i32 %88, i32* %89, align 4
  %90 = load i32* @a15, align 4
  %91 = load i32** @c15, align 4
  store i32 %90, i32* %91, align 4
  %92 = load i32* @a16, align 4
  %93 = load i32** @c16, align 4
  store i32 %92, i32* %93, align 4
  %94 = load i32* @a17, align 4
  %95 = load i32** @c17, align 4
  store i32 %94, i32* %95, align 4
  %96 = load i32* @a18, align 4
  %97 = load i32** @c18, align 4
  store i32 %96, i32* %97, align 4
  %98 = load i32* @a19, align 4
  %99 = load i32** @c19, align 4
  store i32 %98, i32* %99, align 4
  %100 = load i32* @a20, align 4
  %101 = load i32** @c20, align 4
  store i32 %100, i32* %101, align 4
  %102 = load i32* @a21, align 4
  %103 = load i32** @c21, align 4
  store i32 %102, i32* %103, align 4
  %104 = load i32* @a22, align 4
  %105 = load i32** @c22, align 4
  store i32 %104, i32* %105, align 4
  %106 = load i32* @a23, align 4
  %107 = load i32** @c23, align 4
  store i32 %106, i32* %107, align 4
  %108 = load i32* @a24, align 4
  %109 = load i32** @c24, align 4
  store i32 %108, i32* %109, align 4
  %110 = load i32* @a25, align 4
  %111 = load i32** @c25, align 4
  store i32 %110, i32* %111, align 4
  %112 = load i32* @a26, align 4
  %113 = load i32** @c26, align 4
  store i32 %112, i32* %113, align 4
  %114 = load i32* @a27, align 4
  %115 = load i32** @c27, align 4
  store i32 %114, i32* %115, align 4
  %116 = load i32* @a28, align 4
  %117 = load i32** @c28, align 4
  store i32 %116, i32* %117, align 4
  %118 = load i32* @a29, align 4
  %119 = load i32** @c29, align 4
  store i32 %118, i32* %119, align 4
  %120 = load i32* @a0, align 4
  ret i32 %120
}
