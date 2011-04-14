// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://9208404

typedef int MP4Err;
typedef float Float32;
typedef float float32_t;
typedef __attribute__((neon_vector_type(4))) float32_t float32x4_t;
typedef float vFloat __attribute__((__vector_size__(16)));
typedef vFloat VFLOAT;
typedef unsigned long UInt32;

extern int bar (float32x4_t const *p);

int foo (const Float32 *realBufPtr) {
  float32x4_t const *vRealPtr = (VFLOAT *)&realBufPtr[0];
  return bar(vRealPtr);
}

MP4Err autoCorrelation2nd_Neon(Float32 *alphar, Float32 *alphai,
			    const Float32 *realBufPtr,
			    const Float32 *imagBufPtr,
			    const UInt32 len)
{
  float32x4_t const *vRealPtr = (VFLOAT *)&realBufPtr[0];
  return 0;
}

