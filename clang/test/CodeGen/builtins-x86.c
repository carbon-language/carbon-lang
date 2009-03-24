// RUN: clang-cc -DUSE_64 -triple x86_64-unknown-unknown -emit-llvm -o %t %s &&
// RUN: clang-cc -DUSE_ALL -triple x86_64-unknown-unknown -fsyntax-only -o %t %s

#ifdef USE_ALL
#define USE_3DNOW
#define USE_64
#define USE_SSE4
#endif

// 64-bit
typedef char V8c __attribute__((vector_size(8 * sizeof(char))));
typedef signed short V4s __attribute__((vector_size(8)));
typedef signed int V2i __attribute__((vector_size(8)));
typedef signed long long V1LLi __attribute__((vector_size(8)));

typedef float V2f __attribute__((vector_size(8)));

// 128-bit
typedef char V16c __attribute__((vector_size(16)));
typedef signed short V8s __attribute__((vector_size(16)));
typedef signed int V4i __attribute__((vector_size(16)));
typedef signed long long V2LLi __attribute__((vector_size(16)));

typedef float V4f __attribute__((vector_size(16)));
typedef double V2d __attribute__((vector_size(16)));

void f0() {
  signed char         tmp_c;
//  unsigned char       tmp_Uc;
  signed short        tmp_s;
#ifdef USE_ALL
  unsigned short      tmp_Us;
#endif
  signed int          tmp_i;
  unsigned int        tmp_Ui;
  signed long long    tmp_LLi;
//  unsigned long long  tmp_ULLi;
  float               tmp_f;
  double              tmp_d;

  void*          tmp_vp;
  const void*    tmp_vCp;
  char*          tmp_cp; 
  const char*    tmp_cCp; 
  int*           tmp_ip;
  float*         tmp_fp;
  const float*   tmp_fCp;
  double*        tmp_dp;
  const double*  tmp_dCp;

#define imm_i 32
#define imm_i_0_2 0
#define imm_i_0_4 3
#define imm_i_0_8 7
#define imm_i_0_16 15
  // Check this.
#define imm_i_0_256 0

  V2i*   tmp_V2ip;
  V1LLi* tmp_V1LLip;
  V2LLi* tmp_V2LLip;

  // 64-bit
  V8c    tmp_V8c;
  V4s    tmp_V4s;
  V2i    tmp_V2i;
  V1LLi  tmp_V1LLi;
#ifdef USE_3DNOW
  V2f    tmp_V2f;
#endif

  // 128-bit
  V16c   tmp_V16c;
  V8s    tmp_V8s;
  V4i    tmp_V4i;
  V2LLi  tmp_V2LLi;
  V4f    tmp_V4f;
  V2d    tmp_V2d;

  tmp_i = __builtin_ia32_comieq(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comilt(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comile(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comigt(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comige(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comineq(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomieq(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomilt(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomile(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomigt(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomige(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomineq(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comisdeq(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_comisdlt(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_comisdle(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_comisdgt(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_comisdge(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_comisdneq(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdeq(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdlt(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdle(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdgt(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdge(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdneq(tmp_V2d, tmp_V2d);
  tmp_V4f = __builtin_ia32_addps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_subps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_mulps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_divps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_addss(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_subss(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_mulss(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_divss(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpeqps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpltps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpleps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpgtps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpgeps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpunordps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpneqps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpnltps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpnleps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpngtps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpngeps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpordps(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpeqss(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpltss(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpless(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpunordss(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpneqss(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpnltss(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpnless(tmp_V4f, tmp_V4f);
#ifdef USE_ALL
  tmp_V4i = __builtin_ia32_cmpngtss(tmp_V4f, tmp_V4f);
  tmp_V4i = __builtin_ia32_cmpngess(tmp_V4f, tmp_V4f);
#endif
  tmp_V4i = __builtin_ia32_cmpordss(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_minps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_maxps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_minss(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_maxss(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_andps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_andnps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_orps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_xorps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_movss(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_movhlps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_movlhps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_unpckhps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_unpcklps(tmp_V4f, tmp_V4f);
  tmp_V8c = __builtin_ia32_paddb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_paddw(tmp_V4s, tmp_V4s);
  tmp_V2i = __builtin_ia32_paddd(tmp_V2i, tmp_V2i);

  tmp_V1LLi = __builtin_ia32_paddq(tmp_V1LLi, tmp_V1LLi);
  tmp_V8c = __builtin_ia32_psubb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_psubw(tmp_V4s, tmp_V4s);
  tmp_V2i = __builtin_ia32_psubd(tmp_V2i, tmp_V2i);
  tmp_V1LLi = __builtin_ia32_psubq(tmp_V1LLi, tmp_V1LLi);
  tmp_V8c = __builtin_ia32_paddsb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_paddsw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_psubsb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_psubsw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_paddusb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_paddusw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_psubusb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_psubusw(tmp_V4s, tmp_V4s);
  tmp_V4s = __builtin_ia32_pmullw(tmp_V4s, tmp_V4s);
  tmp_V4s = __builtin_ia32_pmulhw(tmp_V4s, tmp_V4s);
  tmp_V4s = __builtin_ia32_pmulhuw(tmp_V4s, tmp_V4s);
  tmp_V1LLi = __builtin_ia32_pand(tmp_V1LLi, tmp_V1LLi);
  tmp_V1LLi = __builtin_ia32_pandn(tmp_V1LLi, tmp_V1LLi);
  tmp_V1LLi = __builtin_ia32_por(tmp_V1LLi, tmp_V1LLi);
  tmp_V1LLi = __builtin_ia32_pxor(tmp_V1LLi, tmp_V1LLi);
  tmp_V8c = __builtin_ia32_pavgb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_pavgw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_pcmpeqb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_pcmpeqw(tmp_V4s, tmp_V4s);
  tmp_V2i = __builtin_ia32_pcmpeqd(tmp_V2i, tmp_V2i);
  tmp_V8c = __builtin_ia32_pcmpgtb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_pcmpgtw(tmp_V4s, tmp_V4s);
  tmp_V2i = __builtin_ia32_pcmpgtd(tmp_V2i, tmp_V2i);
  tmp_V8c = __builtin_ia32_pmaxub(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_pmaxsw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_pminub(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_pminsw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_punpckhbw(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_punpckhwd(tmp_V4s, tmp_V4s);
  tmp_V2i = __builtin_ia32_punpckhdq(tmp_V2i, tmp_V2i);
  tmp_V8c = __builtin_ia32_punpcklbw(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_punpcklwd(tmp_V4s, tmp_V4s);
  tmp_V2i = __builtin_ia32_punpckldq(tmp_V2i, tmp_V2i);
  tmp_V2d = __builtin_ia32_addpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_subpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_mulpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_divpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_addsd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_subsd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_mulsd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_divsd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpeqpd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpltpd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmplepd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpgtpd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpgepd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpunordpd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpneqpd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpnltpd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpnlepd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpngtpd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpngepd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpordpd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpeqsd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpltsd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmplesd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpunordsd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpneqsd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpnltsd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpnlesd(tmp_V2d, tmp_V2d);
  tmp_V4i = __builtin_ia32_cmpordsd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_minpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_maxpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_minsd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_maxsd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_andpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_andnpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_orpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_xorpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_movsd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_unpckhpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_unpcklpd(tmp_V2d, tmp_V2d);
  tmp_V16c = __builtin_ia32_paddb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_paddw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_paddd128(tmp_V4i, tmp_V4i);
  tmp_V2LLi = __builtin_ia32_paddq128(tmp_V2LLi, tmp_V2LLi);
  tmp_V16c = __builtin_ia32_psubb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_psubw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_psubd128(tmp_V4i, tmp_V4i);
  tmp_V2LLi = __builtin_ia32_psubq128(tmp_V2LLi, tmp_V2LLi);
  tmp_V16c = __builtin_ia32_paddsb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_paddsw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_psubsb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_psubsw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_paddusb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_paddusw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_psubusb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_psubusw128(tmp_V8s, tmp_V8s);
  tmp_V8s = __builtin_ia32_pmullw128(tmp_V8s, tmp_V8s);
  tmp_V8s = __builtin_ia32_pmulhw128(tmp_V8s, tmp_V8s);
  tmp_V2LLi = __builtin_ia32_pand128(tmp_V2LLi, tmp_V2LLi);
  tmp_V2LLi = __builtin_ia32_pandn128(tmp_V2LLi, tmp_V2LLi);
  tmp_V2LLi = __builtin_ia32_por128(tmp_V2LLi, tmp_V2LLi);
  tmp_V2LLi = __builtin_ia32_pxor128(tmp_V2LLi, tmp_V2LLi);
  tmp_V16c = __builtin_ia32_pavgb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_pavgw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_pcmpeqb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_pcmpeqw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_pcmpeqd128(tmp_V4i, tmp_V4i);
  tmp_V16c = __builtin_ia32_pcmpgtb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_pcmpgtw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_pcmpgtd128(tmp_V4i, tmp_V4i);
  tmp_V16c = __builtin_ia32_pmaxub128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_pmaxsw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_pminub128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_pminsw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_punpckhbw128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_punpckhwd128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_punpckhdq128(tmp_V4i, tmp_V4i);
  tmp_V2LLi = __builtin_ia32_punpckhqdq128(tmp_V2LLi, tmp_V2LLi);
  tmp_V16c = __builtin_ia32_punpcklbw128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_punpcklwd128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_punpckldq128(tmp_V4i, tmp_V4i);
  tmp_V2LLi = __builtin_ia32_punpcklqdq128(tmp_V2LLi, tmp_V2LLi);
  tmp_V8s = __builtin_ia32_packsswb128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_packssdw128(tmp_V4i, tmp_V4i);
  tmp_V8s = __builtin_ia32_packuswb128(tmp_V8s, tmp_V8s);
  tmp_V8s = __builtin_ia32_pmulhuw128(tmp_V8s, tmp_V8s);
  tmp_V4f = __builtin_ia32_addsubps(tmp_V4f, tmp_V4f);
  tmp_V2d = __builtin_ia32_addsubpd(tmp_V2d, tmp_V2d);
  tmp_V4f = __builtin_ia32_haddps(tmp_V4f, tmp_V4f);
  tmp_V2d = __builtin_ia32_haddpd(tmp_V2d, tmp_V2d);
  tmp_V4f = __builtin_ia32_hsubps(tmp_V4f, tmp_V4f);
  tmp_V2d = __builtin_ia32_hsubpd(tmp_V2d, tmp_V2d);
  tmp_V8s = __builtin_ia32_phaddw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_phaddw(tmp_V4s, tmp_V4s);
  tmp_V4i = __builtin_ia32_phaddd128(tmp_V4i, tmp_V4i);
  tmp_V2i = __builtin_ia32_phaddd(tmp_V2i, tmp_V2i);
  tmp_V8s = __builtin_ia32_phaddsw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_phaddsw(tmp_V4s, tmp_V4s);
  tmp_V8s = __builtin_ia32_phsubw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_phsubw(tmp_V4s, tmp_V4s);
  tmp_V4i = __builtin_ia32_phsubd128(tmp_V4i, tmp_V4i);
  tmp_V2i = __builtin_ia32_phsubd(tmp_V2i, tmp_V2i);
  tmp_V8s = __builtin_ia32_phsubsw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_phsubsw(tmp_V4s, tmp_V4s);
  tmp_V16c = __builtin_ia32_pmaddubsw128(tmp_V16c, tmp_V16c);
  tmp_V8c = __builtin_ia32_pmaddubsw(tmp_V8c, tmp_V8c);
  tmp_V8s = __builtin_ia32_pmulhrsw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_pmulhrsw(tmp_V4s, tmp_V4s);
  tmp_V16c = __builtin_ia32_pshufb128(tmp_V16c, tmp_V16c);
  tmp_V8c = __builtin_ia32_pshufb(tmp_V8c, tmp_V8c);
  tmp_V16c = __builtin_ia32_psignb128(tmp_V16c, tmp_V16c);
  tmp_V8c = __builtin_ia32_psignb(tmp_V8c, tmp_V8c);
  tmp_V8s = __builtin_ia32_psignw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_psignw(tmp_V4s, tmp_V4s);
  tmp_V4i = __builtin_ia32_psignd128(tmp_V4i, tmp_V4i);
  tmp_V2i = __builtin_ia32_psignd(tmp_V2i, tmp_V2i);
  tmp_V16c = __builtin_ia32_pabsb128(tmp_V16c);
  tmp_V8c = __builtin_ia32_pabsb(tmp_V8c);
  tmp_V8s = __builtin_ia32_pabsw128(tmp_V8s);
  tmp_V4s = __builtin_ia32_pabsw(tmp_V4s);
  tmp_V4i = __builtin_ia32_pabsd128(tmp_V4i);
  tmp_V2i = __builtin_ia32_pabsd(tmp_V2i);
  tmp_V4s = __builtin_ia32_psllw(tmp_V4s, tmp_V1LLi);
  tmp_V2i = __builtin_ia32_pslld(tmp_V2i, tmp_V1LLi);
  tmp_V1LLi = __builtin_ia32_psllq(tmp_V1LLi, tmp_V1LLi);
  tmp_V4s = __builtin_ia32_psrlw(tmp_V4s, tmp_V1LLi);
  tmp_V2i = __builtin_ia32_psrld(tmp_V2i, tmp_V1LLi);
  tmp_V1LLi = __builtin_ia32_psrlq(tmp_V1LLi, tmp_V1LLi);
  tmp_V4s = __builtin_ia32_psraw(tmp_V4s, tmp_V1LLi);
  tmp_V2i = __builtin_ia32_psrad(tmp_V2i, tmp_V1LLi);
#ifdef USE_ALL
  tmp_V4s = __builtin_ia32_pshufw(tmp_V4s, imm_i);
#endif
  tmp_V2i = __builtin_ia32_pmaddwd(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_packsswb(tmp_V4s, tmp_V4s);
  tmp_V4s = __builtin_ia32_packssdw(tmp_V2i, tmp_V2i);
  tmp_V8c = __builtin_ia32_packuswb(tmp_V4s, tmp_V4s);

  (void) __builtin_ia32_ldmxcsr(tmp_Ui);
  tmp_Ui = __builtin_ia32_stmxcsr();
  tmp_V4f = __builtin_ia32_cvtpi2ps(tmp_V4f, tmp_V2i);
  tmp_V2i = __builtin_ia32_cvtps2pi(tmp_V4f);
  tmp_V4f = __builtin_ia32_cvtsi2ss(tmp_V4f, tmp_i);
#ifdef USE_64
  tmp_V4f = __builtin_ia32_cvtsi642ss(tmp_V4f, tmp_LLi);
#endif
  tmp_i = __builtin_ia32_cvtss2si(tmp_V4f);
#ifdef USE_64
  tmp_LLi = __builtin_ia32_cvtss2si64(tmp_V4f);
#endif
  tmp_V2i = __builtin_ia32_cvttps2pi(tmp_V4f);
  tmp_i = __builtin_ia32_cvttss2si(tmp_V4f);
#ifdef USE_64
  tmp_LLi = __builtin_ia32_cvttss2si64(tmp_V4f);
#endif
  (void) __builtin_ia32_maskmovq(tmp_V8c, tmp_V8c, tmp_cp);
  tmp_V4f = __builtin_ia32_loadups(tmp_fCp);
  (void) __builtin_ia32_storeups(tmp_fp, tmp_V4f);
  tmp_V4f = __builtin_ia32_loadhps(tmp_V4f, tmp_V2ip);
  tmp_V4f = __builtin_ia32_loadlps(tmp_V4f, tmp_V2ip);
  (void) __builtin_ia32_storehps(tmp_V2ip, tmp_V4f);
  (void) __builtin_ia32_storelps(tmp_V2ip, tmp_V4f);
  tmp_i = __builtin_ia32_movmskps(tmp_V4f);
  tmp_i = __builtin_ia32_pmovmskb(tmp_V8c);
  (void) __builtin_ia32_movntps(tmp_fp, tmp_V4f);
  (void) __builtin_ia32_movntq(tmp_V1LLip, tmp_V1LLi);
  (void) __builtin_ia32_sfence();

  tmp_V4s = __builtin_ia32_psadbw(tmp_V8c, tmp_V8c);
  tmp_V4f = __builtin_ia32_rcpps(tmp_V4f);
  tmp_V4f = __builtin_ia32_rcpss(tmp_V4f);
  tmp_V4f = __builtin_ia32_rsqrtps(tmp_V4f);
  tmp_V4f = __builtin_ia32_rsqrtss(tmp_V4f);
  tmp_V4f = __builtin_ia32_sqrtps(tmp_V4f);
  tmp_V4f = __builtin_ia32_sqrtss(tmp_V4f);
  tmp_V4f = __builtin_ia32_shufps(tmp_V4f, tmp_V4f, imm_i);
#ifdef USE_3DNOW
  (void) __builtin_ia32_femms();
  tmp_V8c = __builtin_ia32_pavgusb(tmp_V8c, tmp_V8c);
  tmp_V2i = __builtin_ia32_pf2id(tmp_V2f);
  tmp_V2f = __builtin_ia32_pfacc(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfadd(tmp_V2f, tmp_V2f);
  tmp_V2i = __builtin_ia32_pfcmpeq(tmp_V2f, tmp_V2f);
  tmp_V2i = __builtin_ia32_pfcmpge(tmp_V2f, tmp_V2f);
  tmp_V2i = __builtin_ia32_pfcmpgt(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfmax(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfmin(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfmul(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfrcp(tmp_V2f);
  tmp_V2f = __builtin_ia32_pfrcpit1(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfrcpit2(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfrsqrt(tmp_V2f);
  tmp_V2f = __builtin_ia32_pfrsqit1(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfsub(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfsubr(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pi2fd(tmp_V2i);
  tmp_V4s = __builtin_ia32_pmulhrw(tmp_V4s, tmp_V4s);
#endif
#ifdef USE_3DNOWA
  tmp_V2i = __builtin_ia32_pf2iw(tmp_V2f);
  tmp_V2f = __builtin_ia32_pfnacc(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfpnacc(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pi2fw(tmp_V2i);
  tmp_V2f = __builtin_ia32_pswapdsf(tmp_V2f);
  tmp_V2i = __builtin_ia32_pswapdsi(tmp_V2i);
#endif
  (void) __builtin_ia32_maskmovdqu(tmp_V16c, tmp_V16c, tmp_cp);
  tmp_V2d = __builtin_ia32_loadupd(tmp_dCp);
  (void) __builtin_ia32_storeupd(tmp_dp, tmp_V2d);
  tmp_V2d = __builtin_ia32_loadhpd(tmp_V2d, tmp_dCp);
  tmp_V2d = __builtin_ia32_loadlpd(tmp_V2d, tmp_dCp);
  tmp_i = __builtin_ia32_movmskpd(tmp_V2d);
  tmp_i = __builtin_ia32_pmovmskb128(tmp_V16c);
  (void) __builtin_ia32_movnti(tmp_ip, tmp_i);
  (void) __builtin_ia32_movntpd(tmp_dp, tmp_V2d);
  (void) __builtin_ia32_movntdq(tmp_V2LLip, tmp_V2LLi);
  tmp_V4i = __builtin_ia32_pshufd(tmp_V4i, imm_i);
  tmp_V8s = __builtin_ia32_pshuflw(tmp_V8s, imm_i);
  tmp_V8s = __builtin_ia32_pshufhw(tmp_V8s, imm_i);
  tmp_V2LLi = __builtin_ia32_psadbw128(tmp_V16c, tmp_V16c);
  tmp_V2d = __builtin_ia32_sqrtpd(tmp_V2d);
  tmp_V2d = __builtin_ia32_sqrtsd(tmp_V2d);
  tmp_V2d = __builtin_ia32_shufpd(tmp_V2d, tmp_V2d, imm_i);
  tmp_V2d = __builtin_ia32_cvtdq2pd(tmp_V4i);
  tmp_V4f = __builtin_ia32_cvtdq2ps(tmp_V4i);
  tmp_V2LLi = __builtin_ia32_cvtpd2dq(tmp_V2d);
  tmp_V2i = __builtin_ia32_cvtpd2pi(tmp_V2d);
  tmp_V4f = __builtin_ia32_cvtpd2ps(tmp_V2d);
  tmp_V4i = __builtin_ia32_cvttpd2dq(tmp_V2d);
  tmp_V2i = __builtin_ia32_cvttpd2pi(tmp_V2d);
  tmp_V2d = __builtin_ia32_cvtpi2pd(tmp_V2i);
  tmp_i = __builtin_ia32_cvtsd2si(tmp_V2d);
  tmp_i = __builtin_ia32_cvttsd2si(tmp_V2d);
#ifdef USE_64
  tmp_LLi = __builtin_ia32_cvtsd2si64(tmp_V2d);
  tmp_LLi = __builtin_ia32_cvttsd2si64(tmp_V2d);
#endif
  tmp_V4i = __builtin_ia32_cvtps2dq(tmp_V4f);
  tmp_V2d = __builtin_ia32_cvtps2pd(tmp_V4f);
  tmp_V4i = __builtin_ia32_cvttps2dq(tmp_V4f);
  tmp_V2d = __builtin_ia32_cvtsi2sd(tmp_V2d, tmp_i);
#ifdef USE_64
  tmp_V2d = __builtin_ia32_cvtsi642sd(tmp_V2d, tmp_LLi);
#endif
  tmp_V4f = __builtin_ia32_cvtsd2ss(tmp_V4f, tmp_V2d);
  tmp_V2d = __builtin_ia32_cvtss2sd(tmp_V2d, tmp_V4f);
  (void) __builtin_ia32_clflush(tmp_vCp);
  (void) __builtin_ia32_lfence();
  (void) __builtin_ia32_mfence();
  tmp_V16c = __builtin_ia32_loaddqu(tmp_cCp);
  (void) __builtin_ia32_storedqu(tmp_cp, tmp_V16c);
  tmp_V4s = __builtin_ia32_psllwi(tmp_V4s, tmp_i);
  tmp_V2i = __builtin_ia32_pslldi(tmp_V2i, tmp_i);
  tmp_V1LLi = __builtin_ia32_psllqi(tmp_V1LLi, tmp_i);
  tmp_V4s = __builtin_ia32_psrawi(tmp_V4s, tmp_i);
  tmp_V2i = __builtin_ia32_psradi(tmp_V2i, tmp_i);
  tmp_V4s = __builtin_ia32_psrlwi(tmp_V4s, tmp_i);
  tmp_V2i = __builtin_ia32_psrldi(tmp_V2i, tmp_i);
  tmp_V1LLi = __builtin_ia32_psrlqi(tmp_V1LLi, tmp_i);
  tmp_V1LLi = __builtin_ia32_pmuludq(tmp_V2i, tmp_V2i);
  tmp_V2LLi = __builtin_ia32_pmuludq128(tmp_V4i, tmp_V4i);
  tmp_V8s = __builtin_ia32_psraw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_psrad128(tmp_V4i, tmp_V4i);
  tmp_V8s = __builtin_ia32_psrlw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_psrld128(tmp_V4i, tmp_V4i);
  tmp_V2LLi = __builtin_ia32_psrlq128(tmp_V2LLi, tmp_V2LLi);
  tmp_V8s = __builtin_ia32_psllw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_pslld128(tmp_V4i, tmp_V4i);
  tmp_V2LLi = __builtin_ia32_psllq128(tmp_V2LLi, tmp_V2LLi);
  tmp_V8s = __builtin_ia32_psllwi128(tmp_V8s, tmp_i);
  tmp_V4i = __builtin_ia32_pslldi128(tmp_V4i, tmp_i);
  tmp_V2LLi = __builtin_ia32_psllqi128(tmp_V2LLi, tmp_i);
  tmp_V8s = __builtin_ia32_psrlwi128(tmp_V8s, tmp_i);
  tmp_V4i = __builtin_ia32_psrldi128(tmp_V4i, tmp_i);
  tmp_V2LLi = __builtin_ia32_psrlqi128(tmp_V2LLi, tmp_i);
  tmp_V8s = __builtin_ia32_psrawi128(tmp_V8s, tmp_i);
  tmp_V4i = __builtin_ia32_psradi128(tmp_V4i, tmp_i);
  tmp_V8s = __builtin_ia32_pmaddwd128(tmp_V8s, tmp_V8s);
  (void) __builtin_ia32_monitor(tmp_vp, tmp_Ui, tmp_Ui);
  (void) __builtin_ia32_mwait(tmp_Ui, tmp_Ui);
#ifdef USE_ALL
  tmp_V4f = __builtin_ia32_movshdup(tmp_V4f);
  tmp_V4f = __builtin_ia32_movsldup(tmp_V4f);
#endif
  tmp_V16c = __builtin_ia32_lddqu(tmp_cCp);
  tmp_V2LLi = __builtin_ia32_palignr128(tmp_V2LLi, tmp_V2LLi, imm_i);
  tmp_V1LLi = __builtin_ia32_palignr(tmp_V1LLi, tmp_V1LLi, imm_i);
  tmp_V2i = __builtin_ia32_vec_init_v2si(tmp_i, tmp_i);
  tmp_V4s = __builtin_ia32_vec_init_v4hi(tmp_s, tmp_s, tmp_s, tmp_s);
  tmp_V8c = __builtin_ia32_vec_init_v8qi(tmp_c, tmp_c, tmp_c, tmp_c, tmp_c, tmp_c, tmp_c, tmp_c);
  tmp_d = __builtin_ia32_vec_ext_v2df(tmp_V2d, imm_i_0_2);
  tmp_LLi = __builtin_ia32_vec_ext_v2di(tmp_V2LLi, imm_i_0_2);
  tmp_f = __builtin_ia32_vec_ext_v4sf(tmp_V4f, imm_i_0_4);
  tmp_i = __builtin_ia32_vec_ext_v4si(tmp_V4i, imm_i_0_4);
#ifdef USE_ALL
  tmp_Us = __builtin_ia32_vec_ext_v8hi(tmp_V8s, imm_i_0_8);
  tmp_s = __builtin_ia32_vec_ext_v4hi(tmp_V4s, imm_i_0_4);
#endif
  tmp_i = __builtin_ia32_vec_ext_v2si(tmp_V2i, imm_i_0_2);
  tmp_V8s = __builtin_ia32_vec_set_v8hi(tmp_V8s, tmp_s, imm_i_0_8);
  tmp_V4s = __builtin_ia32_vec_set_v4hi(tmp_V4s, tmp_s, imm_i_0_4);
  tmp_V4i = __builtin_ia32_movqv4si(tmp_V4i);
  tmp_V4i = __builtin_ia32_loadlv4si(tmp_V2ip);
  (void) __builtin_ia32_storelv4si(tmp_V2ip, tmp_V2LLi);
#ifdef USE_SSE4
  tmp_V16c = __builtin_ia32_pblendvb128(tmp_V16c, tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_pblendw128(tmp_V8s, tmp_V8s, imm_i_0_256);
  tmp_V2d = __builtin_ia32_blendpd(tmp_V2d, tmp_V2d, imm_i_0_256);
  tmp_V4f = __builtin_ia32_blendps(tmp_V4f, tmp_V4f, imm_i_0_256);
  tmp_V2d = __builtin_ia32_blendvpd(tmp_V2d, tmp_V2d, tmp_V2d);
  tmp_V4f = __builtin_ia32_blendvps(tmp_V4f, tmp_V4f, tmp_V4f);
  tmp_V8s = __builtin_ia32_packusdw128(tmp_V4i, tmp_V4i);
  tmp_V16c = __builtin_ia32_pmaxsb128(tmp_V16c, tmp_V16c);
  tmp_V4i = __builtin_ia32_pmaxsd128(tmp_V4i, tmp_V4i);
  tmp_V4i = __builtin_ia32_pmaxud128(tmp_V4i, tmp_V4i);
  tmp_V8s = __builtin_ia32_pmaxuw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_pminsb128(tmp_V16c, tmp_V16c);
  tmp_V4i = __builtin_ia32_pminsd128(tmp_V4i, tmp_V4i);
  tmp_V4i = __builtin_ia32_pminud128(tmp_V4i, tmp_V4i);
  tmp_V8s = __builtin_ia32_pminuw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_pmovsxbd128(tmp_V16c);
  tmp_V2LLi = __builtin_ia32_pmovsxbq128(tmp_V16c);
  tmp_V8s = __builtin_ia32_pmovsxbw128(tmp_V16c);
  tmp_V2LLi = __builtin_ia32_pmovsxdq128(tmp_V4i);
  tmp_V4i = __builtin_ia32_pmovsxwd128(tmp_V8s);
  tmp_V2LLi = __builtin_ia32_pmovsxwq128(tmp_V8s);
  tmp_V4i = __builtin_ia32_pmovzxbd128(tmp_V16c);
  tmp_V2LLi = __builtin_ia32_pmovzxbq128(tmp_V16c);
  tmp_V8s = __builtin_ia32_pmovzxbw128(tmp_V16c);
  tmp_V2LLi = __builtin_ia32_pmovzxdq128(tmp_V4i);
  tmp_V4i = __builtin_ia32_pmovzxwd128(tmp_V8s);
  tmp_V2LLi = __builtin_ia32_pmovzxwq128(tmp_V8s);
  tmp_V2LLi = __builtin_ia32_pmuldq128(tmp_V4i, tmp_V4i);
  tmp_V4i = __builtin_ia32_pmulld128(tmp_V4i, tmp_V4i);
  tmp_V4f = __builtin_ia32_roundps(tmp_V4f, imm_i_0_16);
  //  tmp_V4f = __builtin_ia32_roundss(tmp_V4f, tmp_V4f, imm_i_0_16);
  //  tmp_V2d = __builtin_ia32_roundsd(tmp_V2d, tmp_V2d, imm_i_0_16);
  tmp_V2d = __builtin_ia32_roundpd(tmp_V2d, imm_i_0_16);
  tmp_V16c = __builtin_ia32_vec_set_v16qi(tmp_V16c, tmp_i, tmp_i);
  tmp_V4i  = __builtin_ia32_vec_set_v4si(tmp_V4i, tmp_i, tmp_i);
  tmp_V4f = __builtin_ia32_insertps128(tmp_V4f, tmp_V4f, tmp_i);
  tmp_V2LLi = __builtin_ia32_vec_set_v2di(tmp_V2LLi, tmp_LLi, tmp_i);
#endif
}


