//===-- X86Schedule.td - X86 Scheduling Definitions --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for X86 
def IIC_DEFAULT     : InstrItinClass;
def IIC_ALU_MEM     : InstrItinClass;
def IIC_ALU_NONMEM  : InstrItinClass;
def IIC_LEA         : InstrItinClass;
def IIC_LEA_16      : InstrItinClass;
def IIC_MUL8        : InstrItinClass;
def IIC_MUL16_MEM   : InstrItinClass;
def IIC_MUL16_REG   : InstrItinClass;
def IIC_MUL32_MEM   : InstrItinClass;
def IIC_MUL32_REG   : InstrItinClass;
def IIC_MUL64       : InstrItinClass;
// imul by al, ax, eax, tax
def IIC_IMUL8       : InstrItinClass;
def IIC_IMUL16_MEM  : InstrItinClass;
def IIC_IMUL16_REG  : InstrItinClass;
def IIC_IMUL32_MEM  : InstrItinClass;
def IIC_IMUL32_REG  : InstrItinClass;
def IIC_IMUL64      : InstrItinClass;
// imul reg by reg|mem
def IIC_IMUL16_RM   : InstrItinClass;
def IIC_IMUL16_RR   : InstrItinClass;
def IIC_IMUL32_RM   : InstrItinClass;
def IIC_IMUL32_RR   : InstrItinClass;
def IIC_IMUL64_RM   : InstrItinClass;
def IIC_IMUL64_RR   : InstrItinClass;
// imul reg = reg/mem * imm
def IIC_IMUL16_RMI  : InstrItinClass;
def IIC_IMUL16_RRI  : InstrItinClass;
def IIC_IMUL32_RMI  : InstrItinClass;
def IIC_IMUL32_RRI  : InstrItinClass;
def IIC_IMUL64_RMI  : InstrItinClass;
def IIC_IMUL64_RRI  : InstrItinClass;
// div
def IIC_DIV8_MEM    : InstrItinClass;
def IIC_DIV8_REG    : InstrItinClass;
def IIC_DIV16       : InstrItinClass;
def IIC_DIV32       : InstrItinClass;
def IIC_DIV64       : InstrItinClass;
// idiv
def IIC_IDIV8       : InstrItinClass;
def IIC_IDIV16      : InstrItinClass;
def IIC_IDIV32      : InstrItinClass;
def IIC_IDIV64      : InstrItinClass;
// neg/not/inc/dec
def IIC_UNARY_REG   : InstrItinClass;
def IIC_UNARY_MEM   : InstrItinClass;
// add/sub/and/or/xor/adc/sbc/cmp/test
def IIC_BIN_MEM     : InstrItinClass;
def IIC_BIN_NONMEM  : InstrItinClass;
// shift/rotate
def IIC_SR          : InstrItinClass;
// shift double
def IIC_SHD16_REG_IM : InstrItinClass;
def IIC_SHD16_REG_CL : InstrItinClass;
def IIC_SHD16_MEM_IM : InstrItinClass;
def IIC_SHD16_MEM_CL : InstrItinClass;
def IIC_SHD32_REG_IM : InstrItinClass;
def IIC_SHD32_REG_CL : InstrItinClass;
def IIC_SHD32_MEM_IM : InstrItinClass;
def IIC_SHD32_MEM_CL : InstrItinClass;
def IIC_SHD64_REG_IM : InstrItinClass;
def IIC_SHD64_REG_CL : InstrItinClass;
def IIC_SHD64_MEM_IM : InstrItinClass;
def IIC_SHD64_MEM_CL : InstrItinClass;
// cmov
def IIC_CMOV16_RM : InstrItinClass;
def IIC_CMOV16_RR : InstrItinClass;
def IIC_CMOV32_RM : InstrItinClass;
def IIC_CMOV32_RR : InstrItinClass;
def IIC_CMOV64_RM : InstrItinClass;
def IIC_CMOV64_RR : InstrItinClass;
// set
def IIC_SET_R : InstrItinClass;
def IIC_SET_M : InstrItinClass;
// jmp/jcc/jcxz
def IIC_Jcc : InstrItinClass;
def IIC_JCXZ : InstrItinClass;
def IIC_JMP_REL : InstrItinClass;
def IIC_JMP_REG : InstrItinClass;
def IIC_JMP_MEM : InstrItinClass;
def IIC_JMP_FAR_MEM : InstrItinClass;
def IIC_JMP_FAR_PTR : InstrItinClass;
// loop
def IIC_LOOP : InstrItinClass;
def IIC_LOOPE : InstrItinClass;
def IIC_LOOPNE : InstrItinClass;
// call
def IIC_CALL_RI : InstrItinClass;
def IIC_CALL_MEM : InstrItinClass;
def IIC_CALL_FAR_MEM : InstrItinClass;
def IIC_CALL_FAR_PTR : InstrItinClass;
// ret
def IIC_RET : InstrItinClass;
def IIC_RET_IMM : InstrItinClass;

//===----------------------------------------------------------------------===//
// Processor instruction itineraries.

def GenericItineraries : ProcessorItineraries<[], [], []>;

include "X86ScheduleAtom.td"



