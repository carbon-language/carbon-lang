//===- X86ScheduleSLM.td - X86 Atom Scheduling Definitions -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the Intel Atom
// (Silvermont) processor.
//
//===----------------------------------------------------------------------===//

def IEC_RSV0 : FuncUnit;
def IEC_RSV1 : FuncUnit;
def FPC_RSV0 : FuncUnit;
def FPC_RSV1 : FuncUnit;
def MEC_RSV : FuncUnit;














def SLMItineraries : ProcessorItineraries<
  [ IEC_RSV0, IEC_RSV1, FPC_RSV0, FPC_RSV1, MEC_RSV ],
  [], [
  // [InstrStage<N, [FPC_RSV0, FPC_RSV1]>] 
  // [InstrStage<N, [FPC_RSV0, FPC_RSV1], 0>, InstrStage<N, [MEC_RSV]>]
  // [InstrStage<N, [IEC_RSV0, IEC_RSV1]>] 
  // [InstrStage<N, [IEC_RSV0, IEC_RSV1], 0>,InstrStage<N,[MEC_RSV]>]
  //
  // Default is 1 cycle, IEC_RSV0 or IEC_RSV1
  //InstrItinData<IIC_DEFAULT, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_ALU_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_ALU_NONMEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LEA, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LEA_16, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // mul
  InstrItinData<IIC_MUL8, [InstrStage<4, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MUL16_MEM, [InstrStage<4, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_MUL16_REG, [InstrStage<4, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MUL32_MEM, [InstrStage<3, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<3, [MEC_RSV]>] >,
  InstrItinData<IIC_MUL32_REG, [InstrStage<3, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MUL64, [InstrStage<4, [IEC_RSV0, IEC_RSV1]>] >,
  // imul by al, ax, eax, rax
  InstrItinData<IIC_IMUL8, [InstrStage<6, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IMUL16_MEM, [InstrStage<6, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<6, [MEC_RSV]>] >,
  InstrItinData<IIC_IMUL16_REG, [InstrStage<6, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IMUL32_MEM, [InstrStage<6, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<6, [MEC_RSV]>] >,
  InstrItinData<IIC_IMUL32_REG, [InstrStage<6, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IMUL64, [InstrStage<6, [IEC_RSV0, IEC_RSV1]>] >,
  // imul reg by reg|mem
  InstrItinData<IIC_IMUL16_RM, [InstrStage<4, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_IMUL16_RR, [InstrStage<4, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IMUL32_RM, [InstrStage<3, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<3, [MEC_RSV]>] >,
  InstrItinData<IIC_IMUL32_RR, [InstrStage<3, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IMUL64_RM, [InstrStage<4, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_IMUL64_RR, [InstrStage<4, [IEC_RSV0, IEC_RSV1]>]  >,
  // imul reg = reg/mem * imm
  InstrItinData<IIC_IMUL16_RRI, [InstrStage<4, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IMUL32_RRI, [InstrStage<3, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IMUL64_RRI, [InstrStage<4, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IMUL16_RMI, [InstrStage<4, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_IMUL32_RMI, [InstrStage<3, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<3, [MEC_RSV]>] >,
  InstrItinData<IIC_IMUL64_RMI, [InstrStage<4, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  // idiv - min latency
  InstrItinData<IIC_IDIV8, [InstrStage<34, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IDIV16, [InstrStage<35, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IDIV32, [InstrStage<35, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IDIV64, [InstrStage<49, [IEC_RSV0, IEC_RSV1]>] >,
  // div - min latency
  InstrItinData<IIC_DIV8_REG, [InstrStage<25, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_DIV8_MEM, [InstrStage<25, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<25, [MEC_RSV]>] >,
  InstrItinData<IIC_DIV16, [InstrStage<26, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_DIV32, [InstrStage<26, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_DIV64, [InstrStage<38, [IEC_RSV0, IEC_RSV1]>] >,
  // neg/not/inc/dec
  InstrItinData<IIC_UNARY_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_UNARY_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  // add/sub/and/or/xor/adc/sbc/cmp/test
  InstrItinData<IIC_BIN_NONMEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_BIN_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  // adc/sbb
  InstrItinData<IIC_BIN_CARRY_NONMEM, [InstrStage<2, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_BIN_CARRY_MEM, [InstrStage<2, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<2, [MEC_RSV]>] >,
  // shift/rotate
  InstrItinData<IIC_SR, [InstrStage<1, [IEC_RSV0], 0>,
                   InstrStage<1, [MEC_RSV]>] >,
  // shift double
  InstrItinData<IIC_SHD16_REG_IM, [InstrStage<2, [IEC_RSV0]>] >,
  InstrItinData<IIC_SHD16_REG_CL, [InstrStage<4, [IEC_RSV0]>] >,
  InstrItinData<IIC_SHD16_MEM_IM, [InstrStage<2, [IEC_RSV0], 0>,
                   InstrStage<2, [MEC_RSV]>] >,
  InstrItinData<IIC_SHD16_MEM_CL, [InstrStage<4, [IEC_RSV0], 0>,
                   InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_SHD32_REG_IM, [InstrStage<2, [IEC_RSV0]>] >,
  InstrItinData<IIC_SHD32_REG_CL, [InstrStage<4, [IEC_RSV0]>] >,
  InstrItinData<IIC_SHD32_MEM_IM, [InstrStage<2, [IEC_RSV0], 0>,
                   InstrStage<2, [MEC_RSV]>] >,
  InstrItinData<IIC_SHD32_MEM_CL, [InstrStage<4, [IEC_RSV0], 0>,
                   InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_SHD64_REG_IM, [InstrStage<2, [IEC_RSV0]>] >,
  InstrItinData<IIC_SHD64_REG_CL, [InstrStage<4, [IEC_RSV0]>] >,
  InstrItinData<IIC_SHD64_MEM_IM, [InstrStage<2, [IEC_RSV0], 0>,
                   InstrStage<2, [MEC_RSV]>] >,
  InstrItinData<IIC_SHD64_MEM_CL, [InstrStage<4, [IEC_RSV0], 0>,
                   InstrStage<4, [MEC_RSV]>] >,
  // cmov
  InstrItinData<IIC_CMOV16_RM, [InstrStage<2, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<2, [MEC_RSV]>] >,
  InstrItinData<IIC_CMOV16_RR, [InstrStage<2, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CMOV32_RM, [InstrStage<2, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<2, [MEC_RSV]>] >,
  InstrItinData<IIC_CMOV32_RR, [InstrStage<2, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CMOV64_RM, [InstrStage<2, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<2, [MEC_RSV]>] >,
  InstrItinData<IIC_CMOV64_RR, [InstrStage<2, [IEC_RSV0, IEC_RSV1]>] >,
  // set
  InstrItinData<IIC_SET_M, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_SET_R, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // jcc
  InstrItinData<IIC_Jcc, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // jcxz/jecxz/jrcxz
  InstrItinData<IIC_JCXZ, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // jmp rel
  InstrItinData<IIC_JMP_REL, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // jmp indirect
  InstrItinData<IIC_JMP_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_JMP_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  // jmp far
  InstrItinData<IIC_JMP_FAR_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_JMP_FAR_PTR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // loop/loope/loopne
  InstrItinData<IIC_LOOP, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LOOPE, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LOOPNE, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // call - all but reg/imm
  InstrItinData<IIC_CALL_RI, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CALL_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_CALL_FAR_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_CALL_FAR_PTR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  //ret
  InstrItinData<IIC_RET, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_RET_IMM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  //sign extension movs
  InstrItinData<IIC_MOVSX, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOVSX_R16_R8, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOVSX_R16_M8, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_MOVSX_R16_R16, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOVSX_R32_R32, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  //zero extension movs
  InstrItinData<IIC_MOVZX, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOVZX_R16_R8, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOVZX_R16_M8, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,

  InstrItinData<IIC_REP_MOVS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_REP_STOS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  // SSE binary operations
  // arithmetic fp scalar
  InstrItinData<IIC_SSE_ALU_F32S_RR, [InstrStage<3, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_ALU_F32S_RM, [InstrStage<3, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<3, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_ALU_F64S_RR, [InstrStage<3, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_ALU_F64S_RM, [InstrStage<3, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<3, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_MUL_F32S_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MUL_F32S_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_MUL_F64S_RR, [InstrStage<2, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MUL_F64S_RM, [InstrStage<2, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<2, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_DIV_F32S_RR, [InstrStage<13, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_DIV_F32S_RM, [InstrStage<13, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<13, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_DIV_F64S_RR, [InstrStage<13, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_DIV_F64S_RM, [InstrStage<13, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<13, [MEC_RSV]>] >,

  InstrItinData<IIC_SSE_COMIS_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_COMIS_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,

  InstrItinData<IIC_SSE_HADDSUB_RR, [InstrStage<6, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_HADDSUB_RM, [InstrStage<6, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<6, [MEC_RSV]>] >,

  // arithmetic fp parallel
  InstrItinData<IIC_SSE_ALU_F32P_RR, [InstrStage<3, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_ALU_F32P_RM, [InstrStage<3, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<3, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_ALU_F64P_RR, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_ALU_F64P_RM, [InstrStage<4, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_MUL_F32P_RR, [InstrStage<2, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MUL_F32P_RM, [InstrStage<2, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<2, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_MUL_F64P_RR, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MUL_F64P_RM, [InstrStage<4, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_DIV_F32P_RR, [InstrStage<27, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_DIV_F32P_RM, [InstrStage<27, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<27, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_DIV_F64P_RR, [InstrStage<27, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_DIV_F64P_RM, [InstrStage<27, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<27, [MEC_RSV]>] >,

  // bitwise parallel
  InstrItinData<IIC_SSE_BIT_P_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_BIT_P_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,

  // arithmetic int parallel
  InstrItinData<IIC_SSE_INTALU_P_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_INTALU_P_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_INTALUQ_P_RR, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_INTALUQ_P_RM, [InstrStage<4, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,

  // multiply int parallel
  InstrItinData<IIC_SSE_INTMUL_P_RR, [InstrStage<5, [FPC_RSV0]>] >,
  InstrItinData<IIC_SSE_INTMUL_P_RM, [InstrStage<5, [FPC_RSV0], 0>,
                   InstrStage<5, [MEC_RSV]>] >,

  // shift parallel
  InstrItinData<IIC_SSE_INTSH_P_RR, [InstrStage<2, [FPC_RSV0]>] >,
  InstrItinData<IIC_SSE_INTSH_P_RM, [InstrStage<2, [FPC_RSV0], 0>,
                   InstrStage<2, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_INTSH_P_RI, [InstrStage<1, [FPC_RSV0]>] >,

  InstrItinData<IIC_SSE_INTSHDQ_P_RI, [InstrStage<1, [FPC_RSV0]>] >,

  InstrItinData<IIC_SSE_SHUFP, [InstrStage<1, [FPC_RSV0]>] >,
  InstrItinData<IIC_SSE_PSHUF_RI, [InstrStage<1, [FPC_RSV0]>] >,
  InstrItinData<IIC_SSE_PSHUF_MI, [InstrStage<1, [FPC_RSV0], 0>,
                   InstrStage<1, [MEC_RSV]>] >,

  InstrItinData<IIC_SSE_UNPCK, [InstrStage<1, [FPC_RSV0]>] >,

  InstrItinData<IIC_SSE_SQRTPS_RR, [InstrStage<26, [FPC_RSV0]>] >,
  InstrItinData<IIC_SSE_SQRTPS_RM, [InstrStage<26, [FPC_RSV0], 0>,
                   InstrStage<26, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_SQRTSS_RR, [InstrStage<13, [FPC_RSV0]>] >,
  InstrItinData<IIC_SSE_SQRTSS_RM, [InstrStage<13, [FPC_RSV0], 0>,
                   InstrStage<13, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_SQRTPD_RR, [InstrStage<26, [FPC_RSV0]>] >,
  InstrItinData<IIC_SSE_SQRTPD_RM, [InstrStage<26, [FPC_RSV0], 0>,
                   InstrStage<26, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_SQRTSD_RR, [InstrStage<13, [FPC_RSV0]>] >,
  InstrItinData<IIC_SSE_SQRTSD_RM, [InstrStage<13, [FPC_RSV0], 0>,
                   InstrStage<13, [MEC_RSV]>] >,

  InstrItinData<IIC_SSE_RCPP_RR, [InstrStage<9, [FPC_RSV0]>] >,
  InstrItinData<IIC_SSE_RCPP_RM, [InstrStage<9, [FPC_RSV0], 0>,
                   InstrStage<9, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_RCPS_RR, [InstrStage<4, [FPC_RSV0]>] >,
  InstrItinData<IIC_SSE_RCPS_RM, [InstrStage<4, [FPC_RSV0], 0>,
                   InstrStage<4, [MEC_RSV]>] >,

  InstrItinData<IIC_SSE_MOVMSK, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MASKMOV, [InstrStage<5, [FPC_RSV0, FPC_RSV1]>] >,

  InstrItinData<IIC_SSE_PEXTRW, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_PINSRW, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,

  InstrItinData<IIC_SSE_PABS_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_PABS_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,

  InstrItinData<IIC_SSE_MOV_S_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MOV_S_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_MOV_S_MR, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,

  InstrItinData<IIC_SSE_MOVA_P_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MOVA_P_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_MOVA_P_MR, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,

  InstrItinData<IIC_SSE_MOVU_P_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MOVU_P_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_MOVU_P_MR, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,

  InstrItinData<IIC_SSE_MOV_LH, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,

  InstrItinData<IIC_SSE_LDDQU, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,

  InstrItinData<IIC_SSE_MOVDQ, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MOVD_ToGP, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MOVQ_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,

  InstrItinData<IIC_SSE_MOVNT, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_SSE_PREFETCH, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SSE_PAUSE, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SSE_LFENCE, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SSE_MFENCE, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SSE_SFENCE, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SSE_LDMXCSR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SSE_STMXCSR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_SSE_PHADDSUBD_RR, [InstrStage<6, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBD_RM, [InstrStage<6, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<6, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_PHADDSUBSW_RR, [InstrStage<9, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBSW_RM, [InstrStage<9, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<9, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_PHADDSUBW_RR, [InstrStage<9, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<9, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_PHADDSUBW_RM, [InstrStage<9, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<9, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_PSHUFB_RR, [InstrStage<5, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_PSHUFB_RM, [InstrStage<5, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<5, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_PSIGN_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_PSIGN_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,

  InstrItinData<IIC_SSE_PMADD, [InstrStage<5, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_PMULHRSW, [InstrStage<5, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_PALIGNRR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_PALIGNRM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_MWAIT, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SSE_MONITOR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  // conversions
  // to/from PD ...
  InstrItinData<IIC_SSE_CVT_PD_RR, [InstrStage<5, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_CVT_PD_RM, [InstrStage<5, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<5, [MEC_RSV]>] >,
  // to/from PS except to/from PD and PS2PI
  InstrItinData<IIC_SSE_CVT_PS_RR, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_CVT_PS_RM, [InstrStage<4, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_CVT_Scalar_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_CVT_Scalar_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI32_RR, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI32_RM, [InstrStage<4, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI64_RR, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI64_RM, [InstrStage<4, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_CVT_SD2SI_RR, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_CVT_SD2SI_RM, [InstrStage<4, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,

  // MMX MOVs
  InstrItinData<IIC_MMX_MOV_MM_RM,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_MOV_REG_MM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_MOVQ_RM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_MOVQ_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // other MMX
  InstrItinData<IIC_MMX_ALU_RM,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_ALU_RR,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_ALUQ_RM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_ALUQ_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PHADDSUBW_RM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PHADDSUBW_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PHADDSUBD_RM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PHADDSUBD_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PMUL, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_MISC_FUNC_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_MISC_FUNC_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PSADBW,   [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_SHIFT_RI, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_SHIFT_RM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_SHIFT_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_UNPCK_H_RM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_UNPCK_H_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_UNPCK_L, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PCK_RM,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PCK_RR,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PSHUF,   [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PEXTR,   [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_PINSRW,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_MASKMOV, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // conversions
  // from/to PD
  InstrItinData<IIC_MMX_CVT_PD_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_CVT_PD_RM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // from/to PI
  InstrItinData<IIC_MMX_CVT_PS_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MMX_CVT_PS_RM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_CMPX_LOCK, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CMPX_LOCK_8, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CMPX_LOCK_8B, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CMPX_LOCK_16B, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_XADD_LOCK_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_XADD_LOCK_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_FILD, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FLD,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FLD80, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_FST,   [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FST80, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FIST,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_FLDZ,   [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FUCOM,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FUCOMI, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FCOMI,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FNSTSW, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FNSTCW, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FLDCW,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FNINIT, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FFREE,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FNCLEX, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_WAIT,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FXAM,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FNOP,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FLDL,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_F2XM1,  [InstrStage<88, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_FYL2X,  [InstrStage<296, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_FPTAN,  [InstrStage<281, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_FPATAN,  [InstrStage<296, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_FXTRACT,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FPREM1,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FPSTP,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FPREM,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FYL2XP1,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FSINCOS,  [InstrStage<281, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_FRNDINT,  [InstrStage<25, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_FSCALE,  [InstrStage<74, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_FCOMPP,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FXSAVE,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FXRSTOR,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_FXCH, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,

  // System instructions
  InstrItinData<IIC_CPUID, [InstrStage<60, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_INT,   [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_INT3,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_INVD,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_INVLPG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IRET,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_HLT,   [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LXS,   [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LTR,   [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_RDTSC, [InstrStage<30, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_RSM,   [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SIDT,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SGDT,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SLDT,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_STR,    [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SWAPGS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SYSCALL, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SYS_ENTER_EXIT, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_IN_RR,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_IN_RI,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_OUT_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_OUT_IR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_INS,    [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_MOV_REG_DR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOV_DR_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // worst case for mov REG_CRx
  InstrItinData<IIC_MOV_REG_CR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOV_CR_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_MOV_REG_SR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOV_MEM_SR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOV_SR_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOV_SR_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // LAR
  InstrItinData<IIC_LAR_RM,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LAR_RR,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // LSL
  InstrItinData<IIC_LSL_RM,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LSL_RR,  [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_LGDT, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LIDT, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LLDT_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LLDT_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // push control register, segment registers
  InstrItinData<IIC_PUSH_CS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_PUSH_SR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // pop control register, segment registers
  InstrItinData<IIC_POP_SR,    [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_POP_SR_SS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // VERR, VERW
  InstrItinData<IIC_VERR,     [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_VERW_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_VERW_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // WRMSR, RDMSR
  InstrItinData<IIC_WRMSR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_RDMSR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_RDPMC, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  // SMSW, LMSW
  InstrItinData<IIC_SMSW, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LMSW_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LMSW_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_ENTER, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LEAVE, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_POP_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_POP_REG16, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_POP_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_POP_F, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_POP_FD, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_POP_A, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_PUSH_IMM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_PUSH_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_PUSH_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_PUSH_F, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_PUSH_A, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,

  InstrItinData<IIC_BSWAP, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_BIT_SCAN_MEM, [InstrStage<10, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<10, [MEC_RSV]>] >,
  InstrItinData<IIC_BIT_SCAN_REG, [InstrStage<10, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOVS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_STOS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_SCAS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CMPS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOV, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_MOV_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_AHF, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_BT_MI, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_BT_MR, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_BT_RI, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_BT_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_BTX_MI, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_BTX_MR, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_BTX_RI, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_BTX_RR, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_XCHG_REG, [InstrStage<5, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_XCHG_MEM, [InstrStage<5, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<5, [MEC_RSV]>] >,
  InstrItinData<IIC_XADD_REG, [InstrStage<5, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_XADD_MEM, [InstrStage<5, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<5, [MEC_RSV]>] >,
  InstrItinData<IIC_CMPXCHG_MEM, [InstrStage<6, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CMPXCHG_REG, [InstrStage<6, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CMPXCHG_MEM8, [InstrStage<6, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<6, [MEC_RSV]>] >,
  InstrItinData<IIC_CMPXCHG_REG8, [InstrStage<6, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<6, [MEC_RSV]>] >,
  InstrItinData<IIC_CMPXCHG_8B, [InstrStage<6, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CMPXCHG_16B, [InstrStage<6, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_LODS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_OUTS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CLC, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CLD, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CLI, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CMC, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CLTS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_STC, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_STI, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_STD, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_XLAT, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_AAA, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_AAD, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_AAM, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_AAS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_DAA, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_DAS, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_BOUND, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_ARPL_REG, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_ARPL_MEM, [InstrStage<1, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_MOVBE, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_AES, [InstrStage<8, [FPC_RSV0]>] >,
  InstrItinData<IIC_BLEND_NOMEM, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_BLEND_MEM, [InstrStage<4, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_BIT_SCAN_MEM, [InstrStage<10, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<10, [MEC_RSV]>] >,
  InstrItinData<IIC_BIT_SCAN_REG, [InstrStage<10, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CBW, [InstrStage<4, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CRC32_REG, [InstrStage<3, [IEC_RSV0, IEC_RSV1]>] >,
  InstrItinData<IIC_CRC32_MEM, [InstrStage<3, [IEC_RSV0, IEC_RSV1], 0>,
                  InstrStage<3, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_DPPD_RR, [InstrStage<12, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_DPPD_RM, [InstrStage<12, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<12, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_DPPS_RR, [InstrStage<15, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_DPPS_RM, [InstrStage<15, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<15, [MEC_RSV]>] >,
  InstrItinData<IIC_MMX_EMMS, [InstrStage<10, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_EXTRACTPS_RR, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_EXTRACTPS_RM, [InstrStage<4, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_INSERTPS_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_INSERTPS_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_MPSADBW_RR, [InstrStage<1, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_MPSADBW_RM, [InstrStage<1, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<1, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_PMULLD_RR, [InstrStage<11, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_PMULLD_RM, [InstrStage<11, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<11, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_ROUNDPS_REG, [InstrStage<5, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_ROUNDPS_MEM, [InstrStage<5, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<5, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_ROUNDPD_REG, [InstrStage<4, [FPC_RSV0, FPC_RSV1]>] >,
  InstrItinData<IIC_SSE_ROUNDPD_MEM, [InstrStage<4, [FPC_RSV0, FPC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_POPCNT_RR, [InstrStage<4, [IEC_RSV1]>] >,
  InstrItinData<IIC_SSE_POPCNT_RM, [InstrStage<4, [IEC_RSV1], 0>,
                  InstrStage<4, [MEC_RSV]>] >,
  InstrItinData<IIC_SSE_PCLMULQDQ_RR, [InstrStage<10, [IEC_RSV1]>] >,
  InstrItinData<IIC_SSE_PCLMULQDQ_RM, [InstrStage<10, [IEC_RSV1], 0>,
                  InstrStage<10, [MEC_RSV]>] >,

  InstrItinData<IIC_NOP, [InstrStage<1, [IEC_RSV0, IEC_RSV1]>] >
  ]>;

// Silvermont machine model.
def SLMModel : SchedMachineModel {
  let IssueWidth = 2;  // Allows 2 instructions per scheduling group.
  let MinLatency = 1;  // InstrStage cycles overrides MinLatency.
                       // OperandCycles may be used for expected latency.
  let LoadLatency = 3; // Expected cycles, may be overriden by OperandCycles.
  let HighLatency = 30;// Expected, may be overriden by OperandCycles.

  let Itineraries = SLMItineraries;
}
