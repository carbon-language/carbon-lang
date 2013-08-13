//===- MipsMSAInstrInfo.td - MSA ASE instructions -*- tablegen ------------*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips MSA ASE instructions.
//
//===----------------------------------------------------------------------===//

// Instruction encoding.
class LD_B_ENC   : MSA_I5_FMT<0b110, 0b00, 0b000111>;
class LD_H_ENC   : MSA_I5_FMT<0b110, 0b01, 0b000111>;
class LD_W_ENC   : MSA_I5_FMT<0b110, 0b10, 0b000111>;
class LD_D_ENC   : MSA_I5_FMT<0b110, 0b11, 0b000111>;
class ST_B_ENC   : MSA_I5_FMT<0b111, 0b00, 0b000111>;
class ST_H_ENC   : MSA_I5_FMT<0b111, 0b01, 0b000111>;
class ST_W_ENC   : MSA_I5_FMT<0b111, 0b10, 0b000111>;
class ST_D_ENC   : MSA_I5_FMT<0b111, 0b11, 0b000111>;

// Instruction desc.
class LD_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                   ValueType TyNode, InstrItinClass itin, RegisterClass RCWD,
                   Operand MemOpnd = mem, ComplexPattern Addr = addr> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins MemOpnd:$addr);
  string AsmString = !strconcat(instr_asm, "\t$wd, $addr");
  list<dag> Pattern = [(set RCWD:$wd, (TyNode (OpNode Addr:$addr)))];
  InstrItinClass Itinerary = itin;
}

class ST_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                   ValueType TyNode, InstrItinClass itin, RegisterClass RCWD,
                   Operand MemOpnd = mem, ComplexPattern Addr = addr> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins RCWD:$wd, MemOpnd:$addr);
  string AsmString = !strconcat(instr_asm, "\t$wd, $addr");
  list<dag> Pattern = [(OpNode (TyNode RCWD:$wd), Addr:$addr)];
  InstrItinClass Itinerary = itin;
}

// Load/Store
class LD_B_DESC : LD_DESC_BASE<"ld.b", load, v16i8, NoItinerary, MSA128>;
class LD_H_DESC : LD_DESC_BASE<"ld.h", load, v8i16, NoItinerary, MSA128>;
class LD_W_DESC : LD_DESC_BASE<"ld.w", load, v4i32, NoItinerary, MSA128>;
class LD_D_DESC : LD_DESC_BASE<"ld.d", load, v2i64, NoItinerary, MSA128>;
class ST_B_DESC : ST_DESC_BASE<"st.b", store, v16i8, NoItinerary, MSA128>;
class ST_H_DESC : ST_DESC_BASE<"st.h", store, v8i16, NoItinerary, MSA128>;
class ST_W_DESC : ST_DESC_BASE<"st.w", store, v4i32, NoItinerary, MSA128>;
class ST_D_DESC : ST_DESC_BASE<"st.d", store, v2i64, NoItinerary, MSA128>;

// Instruction defs.
def LD_B: LD_B_ENC, LD_B_DESC, Requires<[HasMSA]>;
def LD_H: LD_H_ENC, LD_H_DESC, Requires<[HasMSA]>;
def LD_W: LD_W_ENC, LD_W_DESC, Requires<[HasMSA]>;
def LD_D: LD_D_ENC, LD_D_DESC, Requires<[HasMSA]>;

def ST_B: ST_B_ENC, ST_B_DESC, Requires<[HasMSA]>;
def ST_H: ST_H_ENC, ST_H_DESC, Requires<[HasMSA]>;
def ST_W: ST_W_ENC, ST_W_DESC, Requires<[HasMSA]>;
def ST_D: ST_D_ENC, ST_D_DESC, Requires<[HasMSA]>;

// Patterns.
class MSAPat<dag pattern, dag result, Predicate pred = HasMSA> :
  Pat<pattern, result>, Requires<[pred]>;

