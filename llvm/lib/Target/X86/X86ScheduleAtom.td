//===- X86ScheduleAtom.td - X86 Atom Scheduling Definitions -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the Intel Atom (Bonnell)
// processors.
//
//===----------------------------------------------------------------------===//

//
// Scheduling information derived from the "Intel 64 and IA32 Architectures
// Optimization Reference Manual", Chapter 13, Section 4.
// Functional Units
//    Port 0
def Port0 : FuncUnit; // ALU: ALU0, shift/rotate, load/store
                      // SIMD/FP: SIMD ALU, Shuffle,SIMD/FP multiply, divide
def Port1 : FuncUnit; // ALU: ALU1, bit processing, jump, and LEA
                      // SIMD/FP: SIMD ALU, FP Adder

def AtomItineraries : ProcessorItineraries<
  [ Port0, Port1 ],
  [], [
  // P0 only
  // InstrItinData<class, [InstrStage<N, [P0]>] >,
  // P0 or P1
  // InstrItinData<class, [InstrStage<N, [P0, P1]>] >,
  // P0 and P1
  // InstrItinData<class, [InstrStage<N, [P0], 0>,  InstrStage<N, [P1]>] >,
  //
  // Default is 1 cycle, port0 or port1
  InstrItinData<IIC_DEFAULT, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_ALU_MEM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_ALU_NONMEM, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_LEA, [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_LEA_16, [InstrStage<2, [Port0, Port1]>] >,
  // mul
  InstrItinData<IIC_MUL8, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_MUL16_MEM, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_MUL16_REG, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_MUL32_MEM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_MUL32_REG, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_MUL64, [InstrStage<12, [Port0, Port1]>] >,
  // imul by al, ax, eax, rax
  InstrItinData<IIC_IMUL8, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL16_MEM, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL16_REG, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL32_MEM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL32_REG, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL64, [InstrStage<12, [Port0, Port1]>] >,
  // imul reg by reg|mem
  InstrItinData<IIC_IMUL16_RM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL16_RR, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL32_RM, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_IMUL32_RR, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_IMUL64_RM, [InstrStage<12, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL64_RR, [InstrStage<12, [Port0, Port1]>] >,
  // imul reg = reg/mem * imm
  InstrItinData<IIC_IMUL16_RRI, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL32_RRI, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_IMUL64_RRI, [InstrStage<14, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL16_RMI, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL32_RMI, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_IMUL64_RMI, [InstrStage<14, [Port0, Port1]>] >,
  // idiv
  InstrItinData<IIC_IDIV8, [InstrStage<62, [Port0, Port1]>] >,
  InstrItinData<IIC_IDIV16, [InstrStage<62, [Port0, Port1]>] >,
  InstrItinData<IIC_IDIV32, [InstrStage<62, [Port0, Port1]>] >,
  InstrItinData<IIC_IDIV64, [InstrStage<130, [Port0, Port1]>] >,
  // div
  InstrItinData<IIC_DIV8_REG, [InstrStage<50, [Port0, Port1]>] >,
  InstrItinData<IIC_DIV8_MEM, [InstrStage<68, [Port0, Port1]>] >,
  InstrItinData<IIC_DIV16, [InstrStage<50, [Port0, Port1]>] >,
  InstrItinData<IIC_DIV32, [InstrStage<50, [Port0, Port1]>] >,
  InstrItinData<IIC_DIV64, [InstrStage<130, [Port0, Port1]>] >,
  // neg/not/inc/dec
  InstrItinData<IIC_UNARY_REG, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_UNARY_MEM, [InstrStage<1, [Port0]>] >,
  // add/sub/and/or/xor/adc/sbc/cmp/test
  InstrItinData<IIC_BIN_NONMEM, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_BIN_MEM, [InstrStage<1, [Port0]>] >,
  // shift/rotate
  InstrItinData<IIC_SR, [InstrStage<1, [Port0]>] >,
  // shift double
  InstrItinData<IIC_SHD16_REG_IM, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD16_REG_CL, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD16_MEM_IM, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD16_MEM_CL, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD32_REG_IM, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD32_REG_CL, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD32_MEM_IM, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD32_MEM_CL, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD64_REG_IM, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD64_REG_CL, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD64_MEM_IM, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD64_MEM_CL, [InstrStage<9, [Port0, Port1]>] >,
  // cmov
  InstrItinData<IIC_CMOV16_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_CMOV16_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_CMOV32_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_CMOV32_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_CMOV64_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_CMOV64_RR, [InstrStage<1, [Port0, Port1]>] >,
  // set
  InstrItinData<IIC_SET_M, [InstrStage<2, [Port0, Port1]>] >, 
  InstrItinData<IIC_SET_R, [InstrStage<1, [Port0, Port1]>] >,
  // jcc
  InstrItinData<IIC_Jcc, [InstrStage<1, [Port1]>] >,
  // jcxz/jecxz/jrcxz
  InstrItinData<IIC_JCXZ, [InstrStage<4, [Port0, Port1]>] >,
  // jmp rel
  InstrItinData<IIC_JMP_REL, [InstrStage<1, [Port1]>] >,
  // jmp indirect
  InstrItinData<IIC_JMP_REG, [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_JMP_MEM, [InstrStage<2, [Port0, Port1]>] >,
  // jmp far
  InstrItinData<IIC_JMP_FAR_MEM, [InstrStage<32, [Port0, Port1]>] >,
  InstrItinData<IIC_JMP_FAR_PTR, [InstrStage<31, [Port0, Port1]>] >,
  // loop/loope/loopne
  InstrItinData<IIC_LOOP, [InstrStage<18, [Port0, Port1]>] >,
  InstrItinData<IIC_LOOPE, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_LOOPNE, [InstrStage<17, [Port0, Port1]>] >,
  // call - all but reg/imm
  InstrItinData<IIC_CALL_RI, [InstrStage<1, [Port0], 0>,
                              InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_CALL_MEM, [InstrStage<15, [Port0, Port1]>] >,
  InstrItinData<IIC_CALL_FAR_MEM, [InstrStage<40, [Port0, Port1]>] >,
  InstrItinData<IIC_CALL_FAR_PTR, [InstrStage<39, [Port0, Port1]>] >,
  //ret
  InstrItinData<IIC_RET, [InstrStage<79, [Port0, Port1]>] >,
  InstrItinData<IIC_RET_IMM, [InstrStage<1, [Port0], 0>,  InstrStage<1, [Port1]>] >,
  //sign extension movs
  InstrItinData<IIC_MOVSX,[InstrStage<1, [Port0] >] >,
  InstrItinData<IIC_MOVSX_R16_R8, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_MOVSX_R16_M8, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_MOVSX_R16_R16, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_MOVSX_R32_R32, [InstrStage<1, [Port0, Port1]>] >,
  //zero extension movs
  InstrItinData<IIC_MOVZX,[InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MOVZX_R16_R8, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_MOVZX_R16_M8, [InstrStage<3, [Port0, Port1]>] >,

  InstrItinData<IIC_REP_MOVS, [InstrStage<75, [Port0, Port1]>] >,
  InstrItinData<IIC_REP_STOS, [InstrStage<74, [Port0, Port1]>] >,

  // SSE binary operations
  // arithmetic fp scalar
  InstrItinData<IIC_SSE_ALU_F32S_RR, [InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F32S_RM, [InstrStage<5, [Port0], 0>,
                                   InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F64S_RR, [InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F64S_RM, [InstrStage<5, [Port0], 0>,
                                   InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_MUL_F32S_RR, [InstrStage<4, [Port0]>] >,
  InstrItinData<IIC_SSE_MUL_F32S_RM, [InstrStage<4, [Port0]>] >,
  InstrItinData<IIC_SSE_MUL_F64S_RR, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_MUL_F64S_RM, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_DIV_F32S_RR, [InstrStage<34, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F32S_RM, [InstrStage<34, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F64S_RR, [InstrStage<62, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F64S_RM, [InstrStage<62, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_COMIS_RR, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_COMIS_RM, [InstrStage<10, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_HADDSUB_RR, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_HADDSUB_RM, [InstrStage<9, [Port0, Port1]>] >,

  // arithmetic fp parallel
  InstrItinData<IIC_SSE_ALU_F32P_RR, [InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F32P_RM, [InstrStage<5, [Port0], 0>,
                                   InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F64P_RR, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F64P_RM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MUL_F32P_RR, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_MUL_F32P_RM, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_MUL_F64P_RR, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MUL_F64P_RM, [InstrStage<10, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F32P_RR, [InstrStage<70, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F32P_RM, [InstrStage<70, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F64P_RR, [InstrStage<125, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F64P_RM, [InstrStage<125, [Port0, Port1]>] >,

  // bitwise parallel
  InstrItinData<IIC_SSE_BIT_P_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_BIT_P_RM, [InstrStage<1, [Port0]>] >,

  // arithmetic int parallel
  InstrItinData<IIC_SSE_INTALU_P_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_INTALU_P_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_INTALUQ_P_RR, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_INTALUQ_P_RM, [InstrStage<3, [Port0, Port1]>] >,

  // multiply int parallel
  InstrItinData<IIC_SSE_INTMUL_P_RR, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_INTMUL_P_RM, [InstrStage<5, [Port0]>] >,

  // shift parallel
  InstrItinData<IIC_SSE_INTSH_P_RR, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_INTSH_P_RM, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_INTSH_P_RI, [InstrStage<1, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_CMPP_RR, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CMPP_RM, [InstrStage<7, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_SHUFP, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_PSHUF, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_UNPCK, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_SQRTP_RR, [InstrStage<13, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_SQRTP_RM, [InstrStage<14, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_SQRTS_RR, [InstrStage<11, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_SQRTS_RM, [InstrStage<12, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_RCPP_RR, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_RCPP_RM, [InstrStage<10, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_RCPS_RR, [InstrStage<4, [Port0]>] >,
  InstrItinData<IIC_SSE_RCPS_RM, [InstrStage<4, [Port0]>] >,

  InstrItinData<IIC_SSE_MOVMSK, [InstrStage<3, [Port0]>] >,
  InstrItinData<IIC_SSE_MASKMOV, [InstrStage<2, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_PEXTRW, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PINSRW, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_PABS_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PABS_RM, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_MOV_S_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MOV_S_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_MOV_S_MR, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_MOVA_P_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MOVA_P_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_MOVA_P_MR, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_MOVU_P_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MOVU_P_RM, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MOVU_P_MR, [InstrStage<2, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_MOV_LH, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_LDDQU, [InstrStage<3, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_MOVDQ, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_MOVD_ToGP, [InstrStage<3, [Port0]>] >,
  InstrItinData<IIC_SSE_MOVQ_RR, [InstrStage<1, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_MOVNT, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_PREFETCH, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_PAUSE, [InstrStage<17, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_LFENCE, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MFENCE, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_SFENCE, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_LDMXCSR, [InstrStage<5, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_STMXCSR, [InstrStage<15, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_PHADDSUBD_RR, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBD_RM, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBSW_RR, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBSW_RM, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBW_RR, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBW_RM, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PSHUFB_RR, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PSHUFB_RM, [InstrStage<5, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PSIGN_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PSIGN_RM, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_PMADD, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_PMULHRSW, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_PALIGNR, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_MWAIT, [InstrStage<46, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MONITOR, [InstrStage<45, [Port0, Port1]>] >,

  // conversions
  // to/from PD ...
  InstrItinData<IIC_SSE_CVT_PD_RR, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_PD_RM, [InstrStage<8, [Port0, Port1]>] >,
  // to/from PS except to/from PD and PS2PI
  InstrItinData<IIC_SSE_CVT_PS_RR, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_PS_RM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_Scalar_RR, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_Scalar_RM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI32_RR, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI32_RM, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI64_RR, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI64_RM, [InstrStage<10, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SD2SI_RR, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SD2SI_RM, [InstrStage<9, [Port0, Port1]>] >,

  InstrItinData<IIC_CMPX_LOCK, [InstrStage<14, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPX_LOCK_8, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPX_LOCK_8B, [InstrStage<18, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPX_LOCK_16B, [InstrStage<22, [Port0, Port1]>] >,

  InstrItinData<IIC_XADD_LOCK_MEM, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_XADD_LOCK_MEM, [InstrStage<3, [Port0, Port1]>] >
  ]>;

