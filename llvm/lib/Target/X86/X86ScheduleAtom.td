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
  InstrItinData<IIC_CALL_RI, [InstrStage<1, [Port0], 0>,  InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_CALL_MEM, [InstrStage<15, [Port0, Port1]>] >,
  InstrItinData<IIC_CALL_FAR_MEM, [InstrStage<40, [Port0, Port1]>] >,
  InstrItinData<IIC_CALL_FAR_PTR, [InstrStage<39, [Port0, Port1]>] >,
  //ret
  InstrItinData<IIC_RET, [InstrStage<79, [Port0, Port1]>] >,
  InstrItinData<IIC_RET_IMM, [InstrStage<1, [Port0], 0>,  InstrStage<1, [Port1]>] >
]>;

