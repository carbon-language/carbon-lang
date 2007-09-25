//==- X86InstrFPStack.td - Describe the X86 Instruction Set -------*- C++ -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 x87 FPU instruction set, defining the
// instructions, and properties of the instructions which are needed for code
// generation, machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FPStack specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDTX86FpGet     : SDTypeProfile<1, 0, [SDTCisFP<0>]>;
def SDTX86FpSet     : SDTypeProfile<0, 1, [SDTCisFP<0>]>;
def SDTX86Fld       : SDTypeProfile<1, 2, [SDTCisFP<0>,
                                           SDTCisPtrTy<1>, 
                                           SDTCisVT<2, OtherVT>]>;
def SDTX86Fst       : SDTypeProfile<0, 3, [SDTCisFP<0>,
                                           SDTCisPtrTy<1>, 
                                           SDTCisVT<2, OtherVT>]>;
def SDTX86Fild      : SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisPtrTy<1>,
                                           SDTCisVT<2, OtherVT>]>;
def SDTX86FpToIMem  : SDTypeProfile<0, 2, [SDTCisFP<0>, SDTCisPtrTy<1>]>;

def X86fpget        : SDNode<"X86ISD::FP_GET_RESULT", SDTX86FpGet,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;
def X86fpset        : SDNode<"X86ISD::FP_SET_RESULT", SDTX86FpSet,
                        [SDNPHasChain, SDNPOutFlag]>;
def X86fld          : SDNode<"X86ISD::FLD",      SDTX86Fld,
                        [SDNPHasChain]>;
def X86fst          : SDNode<"X86ISD::FST",      SDTX86Fst,
                        [SDNPHasChain, SDNPInFlag]>;
def X86fild         : SDNode<"X86ISD::FILD",     SDTX86Fild,
                        [SDNPHasChain]>;
def X86fildflag     : SDNode<"X86ISD::FILD_FLAG",SDTX86Fild,
                        [SDNPHasChain, SDNPOutFlag]>;
def X86fp_to_i16mem : SDNode<"X86ISD::FP_TO_INT16_IN_MEM", SDTX86FpToIMem,
                        [SDNPHasChain]>;
def X86fp_to_i32mem : SDNode<"X86ISD::FP_TO_INT32_IN_MEM", SDTX86FpToIMem,
                        [SDNPHasChain]>;
def X86fp_to_i64mem : SDNode<"X86ISD::FP_TO_INT64_IN_MEM", SDTX86FpToIMem,
                        [SDNPHasChain]>;

//===----------------------------------------------------------------------===//
// FPStack pattern fragments
//===----------------------------------------------------------------------===//

def fpimm0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fpimmneg0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(-0.0);
}]>;

def fpimm1 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+1.0);
}]>;

def fpimmneg1 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(-1.0);
}]>;

// Some 'special' instructions
let usesCustomDAGSchedInserter = 1 in {  // Expanded by the scheduler.
  def FP32_TO_INT16_IN_MEM : I<0, Pseudo,
                              (outs), (ins i16mem:$dst, RFP32:$src),
                              "#FP32_TO_INT16_IN_MEM PSEUDO!",
                              [(X86fp_to_i16mem RFP32:$src, addr:$dst)]>;
  def FP32_TO_INT32_IN_MEM : I<0, Pseudo,
                              (outs), (ins i32mem:$dst, RFP32:$src),
                              "#FP32_TO_INT32_IN_MEM PSEUDO!",
                              [(X86fp_to_i32mem RFP32:$src, addr:$dst)]>;
  def FP32_TO_INT64_IN_MEM : I<0, Pseudo,
                              (outs), (ins i64mem:$dst, RFP32:$src),
                              "#FP32_TO_INT64_IN_MEM PSEUDO!",
                              [(X86fp_to_i64mem RFP32:$src, addr:$dst)]>;
  def FP64_TO_INT16_IN_MEM : I<0, Pseudo,
                              (outs), (ins i16mem:$dst, RFP64:$src),
                              "#FP64_TO_INT16_IN_MEM PSEUDO!",
                              [(X86fp_to_i16mem RFP64:$src, addr:$dst)]>;
  def FP64_TO_INT32_IN_MEM : I<0, Pseudo,
                              (outs), (ins i32mem:$dst, RFP64:$src),
                              "#FP64_TO_INT32_IN_MEM PSEUDO!",
                              [(X86fp_to_i32mem RFP64:$src, addr:$dst)]>;
  def FP64_TO_INT64_IN_MEM : I<0, Pseudo,
                              (outs), (ins i64mem:$dst, RFP64:$src),
                              "#FP64_TO_INT64_IN_MEM PSEUDO!",
                              [(X86fp_to_i64mem RFP64:$src, addr:$dst)]>;
  def FP80_TO_INT16_IN_MEM : I<0, Pseudo,
                              (outs), (ins i16mem:$dst, RFP80:$src),
                              "#FP80_TO_INT16_IN_MEM PSEUDO!",
                              [(X86fp_to_i16mem RFP80:$src, addr:$dst)]>;
  def FP80_TO_INT32_IN_MEM : I<0, Pseudo,
                              (outs), (ins i32mem:$dst, RFP80:$src),
                              "#FP80_TO_INT32_IN_MEM PSEUDO!",
                              [(X86fp_to_i32mem RFP80:$src, addr:$dst)]>;
  def FP80_TO_INT64_IN_MEM : I<0, Pseudo,
                              (outs), (ins i64mem:$dst, RFP80:$src),
                              "#FP80_TO_INT64_IN_MEM PSEUDO!",
                              [(X86fp_to_i64mem RFP80:$src, addr:$dst)]>;
}

let isTerminator = 1 in
  let Defs = [FP0, FP1, FP2, FP3, FP4, FP5, FP6] in
    def FP_REG_KILL  : I<0, Pseudo, (outs), (ins), "#FP_REG_KILL", []>;

// All FP Stack operations are represented with four instructions here.  The
// first three instructions, generated by the instruction selector, use "RFP32"
// "RFP64" or "RFP80" registers: traditional register files to reference 32-bit,
// 64-bit or 80-bit floating point values.  These sizes apply to the values, 
// not the registers, which are always 80 bits; RFP32, RFP64 and RFP80 can be
// copied to each other without losing information.  These instructions are all
// pseudo instructions and use the "_Fp" suffix.
// In some cases there are additional variants with a mixture of different
// register sizes.
// The second instruction is defined with FPI, which is the actual instruction
// emitted by the assembler.  These use "RST" registers, although frequently
// the actual register(s) used are implicit.  These are always 80 bits.
// The FP stackifier pass converts one to the other after register allocation 
// occurs.
//
// Note that the FpI instruction should have instruction selection info (e.g.
// a pattern) and the FPI instruction should have emission info (e.g. opcode
// encoding and asm printing info).

// Pseudo Instructions for FP stack return values.
def FpGETRESULT32 : FpI_<(outs RFP32:$dst), (ins), SpecialFP,
                      [(set RFP32:$dst, X86fpget)]>;           // FPR = ST(0)

def FpGETRESULT64 : FpI_<(outs RFP64:$dst), (ins), SpecialFP,
                      [(set RFP64:$dst, X86fpget)]>;           // FPR = ST(0)

def FpGETRESULT80 : FpI_<(outs RFP80:$dst), (ins), SpecialFP,
                      [(set RFP80:$dst, X86fpget)]>;           // FPR = ST(0)

let Defs = [ST0] in {
def FpSETRESULT32 : FpI_<(outs), (ins RFP32:$src), SpecialFP,
                      [(X86fpset RFP32:$src)]>;// ST(0) = FPR

def FpSETRESULT64 : FpI_<(outs), (ins RFP64:$src), SpecialFP,
                      [(X86fpset RFP64:$src)]>;// ST(0) = FPR

def FpSETRESULT80 : FpI_<(outs), (ins RFP80:$src), SpecialFP,
                      [(X86fpset RFP80:$src)]>;// ST(0) = FPR
}

// FpIf32, FpIf64 - Floating Point Psuedo Instruction template.
// f32 instructions can use SSE1 and are predicated on FPStackf32 == !SSE1.
// f64 instructions can use SSE2 and are predicated on FPStackf64 == !SSE2.
// f80 instructions cannot use SSE and use neither of these.
class FpIf32<dag outs, dag ins, FPFormat fp, list<dag> pattern> :
  FpI_<outs, ins, fp, pattern>, Requires<[FPStackf32]>;
class FpIf64<dag outs, dag ins, FPFormat fp, list<dag> pattern> :
  FpI_<outs, ins, fp, pattern>, Requires<[FPStackf64]>;

// Register copies.  Just copies, the shortening ones do not truncate.
def MOV_Fp3232       : FpIf32<(outs RFP32:$dst), (ins RFP32:$src), SpecialFP, []>; 
def MOV_Fp3264       : FpIf32<(outs RFP64:$dst), (ins RFP32:$src), SpecialFP, []>; 
def MOV_Fp6432       : FpIf32<(outs RFP32:$dst), (ins RFP64:$src), SpecialFP, []>; 
def MOV_Fp6464       : FpIf64<(outs RFP64:$dst), (ins RFP64:$src), SpecialFP, []>; 
def MOV_Fp8032       : FpIf32<(outs RFP32:$dst), (ins RFP80:$src), SpecialFP, []>; 
def MOV_Fp3280       : FpIf32<(outs RFP80:$dst), (ins RFP32:$src), SpecialFP, []>; 
def MOV_Fp8064       : FpIf64<(outs RFP64:$dst), (ins RFP80:$src), SpecialFP, []>; 
def MOV_Fp6480       : FpIf64<(outs RFP80:$dst), (ins RFP64:$src), SpecialFP, []>; 
def MOV_Fp8080       : FpI_<(outs RFP80:$dst), (ins RFP80:$src), SpecialFP, []>; 

// Factoring for arithmetic.
multiclass FPBinary_rr<SDNode OpNode> {
// Register op register -> register
// These are separated out because they have no reversed form.
def _Fp32 : FpIf32<(outs RFP32:$dst), (ins RFP32:$src1, RFP32:$src2), TwoArgFP,
                [(set RFP32:$dst, (OpNode RFP32:$src1, RFP32:$src2))]>;
def _Fp64 : FpIf64<(outs RFP64:$dst), (ins RFP64:$src1, RFP64:$src2), TwoArgFP,
                [(set RFP64:$dst, (OpNode RFP64:$src1, RFP64:$src2))]>;
def _Fp80 : FpI_<(outs RFP80:$dst), (ins RFP80:$src1, RFP80:$src2), TwoArgFP,
                [(set RFP80:$dst, (OpNode RFP80:$src1, RFP80:$src2))]>;
}
// The FopST0 series are not included here because of the irregularities
// in where the 'r' goes in assembly output.
// These instructions cannot address 80-bit memory.
multiclass FPBinary<SDNode OpNode, Format fp, string asmstring> {
// ST(0) = ST(0) + [mem]
def _Fp32m  : FpIf32<(outs RFP32:$dst), (ins RFP32:$src1, f32mem:$src2), OneArgFPRW,
                  [(set RFP32:$dst, 
                    (OpNode RFP32:$src1, (loadf32 addr:$src2)))]>;
def _Fp64m  : FpIf64<(outs RFP64:$dst), (ins RFP64:$src1, f64mem:$src2), OneArgFPRW,
                  [(set RFP64:$dst, 
                    (OpNode RFP64:$src1, (loadf64 addr:$src2)))]>;
def _Fp64m32: FpIf64<(outs RFP64:$dst), (ins RFP64:$src1, f32mem:$src2), OneArgFPRW,
                  [(set RFP64:$dst, 
                    (OpNode RFP64:$src1, (f64 (extloadf32 addr:$src2))))]>;
def _Fp80m32: FpI_<(outs RFP80:$dst), (ins RFP80:$src1, f32mem:$src2), OneArgFPRW,
                  [(set RFP80:$dst, 
                    (OpNode RFP80:$src1, (f80 (extloadf32 addr:$src2))))]>;
def _Fp80m64: FpI_<(outs RFP80:$dst), (ins RFP80:$src1, f64mem:$src2), OneArgFPRW,
                  [(set RFP80:$dst, 
                    (OpNode RFP80:$src1, (f80 (extloadf64 addr:$src2))))]>;
def _F32m  : FPI<0xD8, fp, (outs), (ins f32mem:$src), 
                 !strconcat("f", !strconcat(asmstring, "{s}\t$src"))>;
def _F64m  : FPI<0xDC, fp, (outs), (ins f64mem:$src), 
                 !strconcat("f", !strconcat(asmstring, "{l}\t$src"))>;
// ST(0) = ST(0) + [memint]
def _FpI16m32 : FpIf32<(outs RFP32:$dst), (ins RFP32:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP32:$dst, (OpNode RFP32:$src1,
                                       (X86fild addr:$src2, i16)))]>;
def _FpI32m32 : FpIf32<(outs RFP32:$dst), (ins RFP32:$src1, i32mem:$src2), OneArgFPRW,
                    [(set RFP32:$dst, (OpNode RFP32:$src1,
                                       (X86fild addr:$src2, i32)))]>;
def _FpI16m64 : FpIf64<(outs RFP64:$dst), (ins RFP64:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP64:$dst, (OpNode RFP64:$src1,
                                       (X86fild addr:$src2, i16)))]>;
def _FpI32m64 : FpIf64<(outs RFP64:$dst), (ins RFP64:$src1, i32mem:$src2), OneArgFPRW,
                    [(set RFP64:$dst, (OpNode RFP64:$src1,
                                       (X86fild addr:$src2, i32)))]>;
def _FpI16m80 : FpI_<(outs RFP80:$dst), (ins RFP80:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP80:$dst, (OpNode RFP80:$src1,
                                       (X86fild addr:$src2, i16)))]>;
def _FpI32m80 : FpI_<(outs RFP80:$dst), (ins RFP80:$src1, i32mem:$src2), OneArgFPRW,
                    [(set RFP80:$dst, (OpNode RFP80:$src1,
                                       (X86fild addr:$src2, i32)))]>;
def _FI16m  : FPI<0xDE, fp, (outs), (ins i16mem:$src), 
                  !strconcat("fi", !strconcat(asmstring, "{s}\t$src"))>;
def _FI32m  : FPI<0xDA, fp, (outs), (ins i32mem:$src), 
                  !strconcat("fi", !strconcat(asmstring, "{l}\t$src"))>;
}

defm ADD : FPBinary_rr<fadd>;
defm SUB : FPBinary_rr<fsub>;
defm MUL : FPBinary_rr<fmul>;
defm DIV : FPBinary_rr<fdiv>;
defm ADD : FPBinary<fadd, MRM0m, "add">;
defm SUB : FPBinary<fsub, MRM4m, "sub">;
defm SUBR: FPBinary<fsub ,MRM5m, "subr">;
defm MUL : FPBinary<fmul, MRM1m, "mul">;
defm DIV : FPBinary<fdiv, MRM6m, "div">;
defm DIVR: FPBinary<fdiv, MRM7m, "divr">;

class FPST0rInst<bits<8> o, string asm>
  : FPI<o, AddRegFrm, (outs), (ins RST:$op), asm>, D8;
class FPrST0Inst<bits<8> o, string asm>
  : FPI<o, AddRegFrm, (outs), (ins RST:$op), asm>, DC;
class FPrST0PInst<bits<8> o, string asm>
  : FPI<o, AddRegFrm, (outs), (ins RST:$op), asm>, DE;

// NOTE: GAS and apparently all other AT&T style assemblers have a broken notion
// of some of the 'reverse' forms of the fsub and fdiv instructions.  As such,
// we have to put some 'r's in and take them out of weird places.
def ADD_FST0r   : FPST0rInst <0xC0, "fadd\t$op">;
def ADD_FrST0   : FPrST0Inst <0xC0, "fadd\t{%st(0), $op|$op, %ST(0)}">;
def ADD_FPrST0  : FPrST0PInst<0xC0, "faddp\t$op">;
def SUBR_FST0r  : FPST0rInst <0xE8, "fsubr\t$op">;
def SUB_FrST0   : FPrST0Inst <0xE8, "fsub{r}\t{%st(0), $op|$op, %ST(0)}">;
def SUB_FPrST0  : FPrST0PInst<0xE8, "fsub{r}p\t$op">;
def SUB_FST0r   : FPST0rInst <0xE0, "fsub\t$op">;
def SUBR_FrST0  : FPrST0Inst <0xE0, "fsub{|r}\t{%st(0), $op|$op, %ST(0)}">;
def SUBR_FPrST0 : FPrST0PInst<0xE0, "fsub{|r}p\t$op">;
def MUL_FST0r   : FPST0rInst <0xC8, "fmul\t$op">;
def MUL_FrST0   : FPrST0Inst <0xC8, "fmul\t{%st(0), $op|$op, %ST(0)}">;
def MUL_FPrST0  : FPrST0PInst<0xC8, "fmulp\t$op">;
def DIVR_FST0r  : FPST0rInst <0xF8, "fdivr\t$op">;
def DIV_FrST0   : FPrST0Inst <0xF8, "fdiv{r}\t{%st(0), $op|$op, %ST(0)}">;
def DIV_FPrST0  : FPrST0PInst<0xF8, "fdiv{r}p\t$op">;
def DIV_FST0r   : FPST0rInst <0xF0, "fdiv\t$op">;
def DIVR_FrST0  : FPrST0Inst <0xF0, "fdiv{|r}\t{%st(0), $op|$op, %ST(0)}">;
def DIVR_FPrST0 : FPrST0PInst<0xF0, "fdiv{|r}p\t$op">;

// Unary operations.
multiclass FPUnary<SDNode OpNode, bits<8> opcode, string asmstring> {
def _Fp32  : FpIf32<(outs RFP32:$dst), (ins RFP32:$src), OneArgFPRW,
                 [(set RFP32:$dst, (OpNode RFP32:$src))]>;
def _Fp64  : FpIf64<(outs RFP64:$dst), (ins RFP64:$src), OneArgFPRW,
                 [(set RFP64:$dst, (OpNode RFP64:$src))]>;
def _Fp80  : FpI_<(outs RFP80:$dst), (ins RFP80:$src), OneArgFPRW,
                 [(set RFP80:$dst, (OpNode RFP80:$src))]>;
def _F     : FPI<opcode, RawFrm, (outs), (ins), asmstring>, D9;
}

defm CHS : FPUnary<fneg, 0xE0, "fchs">;
defm ABS : FPUnary<fabs, 0xE1, "fabs">;
defm SQRT: FPUnary<fsqrt,0xFA, "fsqrt">;
defm SIN : FPUnary<fsin, 0xFE, "fsin">;
defm COS : FPUnary<fcos, 0xFF, "fcos">;

def TST_Fp32  : FpIf32<(outs), (ins RFP32:$src), OneArgFP,
                 []>;
def TST_Fp64  : FpIf64<(outs), (ins RFP64:$src), OneArgFP,
                 []>;
def TST_Fp80  : FpI_<(outs), (ins RFP80:$src), OneArgFP,
                 []>;
def TST_F  : FPI<0xE4, RawFrm, (outs), (ins), "ftst">, D9;

// Floating point cmovs.
multiclass FPCMov<PatLeaf cc> {
  def _Fp32  : FpIf32<(outs RFP32:$dst), (ins RFP32:$src1, RFP32:$src2), CondMovFP,
                     [(set RFP32:$dst, (X86cmov RFP32:$src1, RFP32:$src2,
                                        cc))]>;
  def _Fp64  : FpIf64<(outs RFP64:$dst), (ins RFP64:$src1, RFP64:$src2), CondMovFP,
                     [(set RFP64:$dst, (X86cmov RFP64:$src1, RFP64:$src2,
                                        cc))]>;
  def _Fp80  : FpI_<(outs RFP80:$dst), (ins RFP80:$src1, RFP80:$src2), CondMovFP,
                     [(set RFP80:$dst, (X86cmov RFP80:$src1, RFP80:$src2,
                                        cc))]>;
}
let isTwoAddress = 1 in {
defm CMOVB  : FPCMov<X86_COND_B>;
defm CMOVBE : FPCMov<X86_COND_BE>;
defm CMOVE  : FPCMov<X86_COND_E>;
defm CMOVP  : FPCMov<X86_COND_P>;
defm CMOVNB : FPCMov<X86_COND_AE>;
defm CMOVNBE: FPCMov<X86_COND_A>;
defm CMOVNE : FPCMov<X86_COND_NE>;
defm CMOVNP : FPCMov<X86_COND_NP>;
}

multiclass NEW_FPCMov<PatLeaf cc> {
  def _Fp32  : FpIf32<(outs RFP32:$dst), (ins RFP32:$src1, RFP32:$src2),
                       CondMovFP,
                     [(set RFP32:$dst, (X86cmov_new RFP32:$src1, RFP32:$src2,
                                        cc, EFLAGS))]>;
  def _Fp64  : FpIf64<(outs RFP64:$dst), (ins RFP64:$src1, RFP64:$src2),
                       CondMovFP,
                     [(set RFP64:$dst, (X86cmov_new RFP64:$src1, RFP64:$src2,
                                        cc, EFLAGS))]>;
  def _Fp80  : FpI_<(outs RFP80:$dst), (ins RFP80:$src1, RFP80:$src2),
                     CondMovFP,
                     [(set RFP80:$dst, (X86cmov_new RFP80:$src1, RFP80:$src2,
                                        cc, EFLAGS))]>;
}
let Uses = [EFLAGS], isTwoAddress = 1 in {
defm NEW_CMOVB  : NEW_FPCMov<X86_COND_B>;
defm NEW_CMOVBE : NEW_FPCMov<X86_COND_BE>;
defm NEW_CMOVE  : NEW_FPCMov<X86_COND_E>;
defm NEW_CMOVP  : NEW_FPCMov<X86_COND_P>;
defm NEW_CMOVNB : NEW_FPCMov<X86_COND_AE>;
defm NEW_CMOVNBE: NEW_FPCMov<X86_COND_A>;
defm NEW_CMOVNE : NEW_FPCMov<X86_COND_NE>;
defm NEW_CMOVNP : NEW_FPCMov<X86_COND_NP>;
}

// These are not factored because there's no clean way to pass DA/DB.
def CMOVB_F  : FPI<0xC0, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovb\t{$op, %st(0)|%ST(0), $op}">, DA;
def CMOVBE_F : FPI<0xD0, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovbe\t{$op, %st(0)|%ST(0), $op}">, DA;
def CMOVE_F  : FPI<0xC8, AddRegFrm, (outs RST:$op), (ins),
                  "fcmove\t{$op, %st(0)|%ST(0), $op}">, DA;
def CMOVP_F  : FPI<0xD8, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovu\t {$op, %st(0)|%ST(0), $op}">, DA;
def CMOVNB_F : FPI<0xC0, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovnb\t{$op, %st(0)|%ST(0), $op}">, DB;
def CMOVNBE_F: FPI<0xD0, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovnbe\t{$op, %st(0)|%ST(0), $op}">, DB;
def CMOVNE_F : FPI<0xC8, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovne\t{$op, %st(0)|%ST(0), $op}">, DB;
def CMOVNP_F : FPI<0xD8, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovnu\t{$op, %st(0)|%ST(0), $op}">, DB;

// Floating point loads & stores.
let isLoad = 1 in {
def LD_Fp32m   : FpIf32<(outs RFP32:$dst), (ins f32mem:$src), ZeroArgFP,
                  [(set RFP32:$dst, (loadf32 addr:$src))]>;
def LD_Fp64m   : FpIf64<(outs RFP64:$dst), (ins f64mem:$src), ZeroArgFP,
                  [(set RFP64:$dst, (loadf64 addr:$src))]>;
def LD_Fp80m   : FpI_<(outs RFP80:$dst), (ins f80mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (loadf80 addr:$src))]>;
}
def LD_Fp32m64 : FpIf64<(outs RFP64:$dst), (ins f32mem:$src), ZeroArgFP,
                  [(set RFP64:$dst, (f64 (extloadf32 addr:$src)))]>;
def LD_Fp64m80 : FpI_<(outs RFP80:$dst), (ins f64mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (f80 (extloadf64 addr:$src)))]>;
def LD_Fp32m80 : FpI_<(outs RFP80:$dst), (ins f32mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (f80 (extloadf32 addr:$src)))]>;
def ILD_Fp16m32: FpIf32<(outs RFP32:$dst), (ins i16mem:$src), ZeroArgFP,
                  [(set RFP32:$dst, (X86fild addr:$src, i16))]>;
def ILD_Fp32m32: FpIf32<(outs RFP32:$dst), (ins i32mem:$src), ZeroArgFP,
                  [(set RFP32:$dst, (X86fild addr:$src, i32))]>;
def ILD_Fp64m32: FpIf32<(outs RFP32:$dst), (ins i64mem:$src), ZeroArgFP,
                  [(set RFP32:$dst, (X86fild addr:$src, i64))]>;
def ILD_Fp16m64: FpIf64<(outs RFP64:$dst), (ins i16mem:$src), ZeroArgFP,
                  [(set RFP64:$dst, (X86fild addr:$src, i16))]>;
def ILD_Fp32m64: FpIf64<(outs RFP64:$dst), (ins i32mem:$src), ZeroArgFP,
                  [(set RFP64:$dst, (X86fild addr:$src, i32))]>;
def ILD_Fp64m64: FpIf64<(outs RFP64:$dst), (ins i64mem:$src), ZeroArgFP,
                  [(set RFP64:$dst, (X86fild addr:$src, i64))]>;
def ILD_Fp16m80: FpI_<(outs RFP80:$dst), (ins i16mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (X86fild addr:$src, i16))]>;
def ILD_Fp32m80: FpI_<(outs RFP80:$dst), (ins i32mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (X86fild addr:$src, i32))]>;
def ILD_Fp64m80: FpI_<(outs RFP80:$dst), (ins i64mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (X86fild addr:$src, i64))]>;

def ST_Fp32m   : FpIf32<(outs), (ins f32mem:$op, RFP32:$src), OneArgFP,
                  [(store RFP32:$src, addr:$op)]>;
def ST_Fp64m32 : FpIf64<(outs), (ins f32mem:$op, RFP64:$src), OneArgFP,
                  [(truncstoref32 RFP64:$src, addr:$op)]>;
def ST_Fp64m   : FpIf64<(outs), (ins f64mem:$op, RFP64:$src), OneArgFP,
                  [(store RFP64:$src, addr:$op)]>;
def ST_Fp80m32 : FpI_<(outs), (ins f32mem:$op, RFP80:$src), OneArgFP,
                  [(truncstoref32 RFP80:$src, addr:$op)]>;
def ST_Fp80m64 : FpI_<(outs), (ins f64mem:$op, RFP80:$src), OneArgFP,
                  [(truncstoref64 RFP80:$src, addr:$op)]>;
// FST does not support 80-bit memory target; FSTP must be used.

def ST_FpP32m    : FpIf32<(outs), (ins f32mem:$op, RFP32:$src), OneArgFP, []>;
def ST_FpP64m32  : FpIf64<(outs), (ins f32mem:$op, RFP64:$src), OneArgFP, []>;
def ST_FpP64m    : FpIf64<(outs), (ins f64mem:$op, RFP64:$src), OneArgFP, []>;
def ST_FpP80m32  : FpI_<(outs), (ins f32mem:$op, RFP80:$src), OneArgFP, []>;
def ST_FpP80m64  : FpI_<(outs), (ins f64mem:$op, RFP80:$src), OneArgFP, []>;
def ST_FpP80m    : FpI_<(outs), (ins f80mem:$op, RFP80:$src), OneArgFP,
                    [(store RFP80:$src, addr:$op)]>;
def IST_Fp16m32  : FpIf32<(outs), (ins i16mem:$op, RFP32:$src), OneArgFP, []>;
def IST_Fp32m32  : FpIf32<(outs), (ins i32mem:$op, RFP32:$src), OneArgFP, []>;
def IST_Fp64m32  : FpIf32<(outs), (ins i64mem:$op, RFP32:$src), OneArgFP, []>;
def IST_Fp16m64  : FpIf64<(outs), (ins i16mem:$op, RFP64:$src), OneArgFP, []>;
def IST_Fp32m64  : FpIf64<(outs), (ins i32mem:$op, RFP64:$src), OneArgFP, []>;
def IST_Fp64m64  : FpIf64<(outs), (ins i64mem:$op, RFP64:$src), OneArgFP, []>;
def IST_Fp16m80  : FpI_<(outs), (ins i16mem:$op, RFP80:$src), OneArgFP, []>;
def IST_Fp32m80  : FpI_<(outs), (ins i32mem:$op, RFP80:$src), OneArgFP, []>;
def IST_Fp64m80  : FpI_<(outs), (ins i64mem:$op, RFP80:$src), OneArgFP, []>;

def LD_F32m   : FPI<0xD9, MRM0m, (outs), (ins f32mem:$src), "fld{s}\t$src">;
def LD_F64m   : FPI<0xDD, MRM0m, (outs), (ins f64mem:$src), "fld{l}\t$src">;
def LD_F80m   : FPI<0xDB, MRM5m, (outs), (ins f80mem:$src), "fld{t}\t$src">;
def ILD_F16m  : FPI<0xDF, MRM0m, (outs), (ins i16mem:$src), "fild{s}\t$src">;
def ILD_F32m  : FPI<0xDB, MRM0m, (outs), (ins i32mem:$src), "fild{l}\t$src">;
def ILD_F64m  : FPI<0xDF, MRM5m, (outs), (ins i64mem:$src), "fild{ll}\t$src">;
def ST_F32m   : FPI<0xD9, MRM2m, (outs), (ins f32mem:$dst), "fst{s}\t$dst">;
def ST_F64m   : FPI<0xDD, MRM2m, (outs), (ins f64mem:$dst), "fst{l}\t$dst">;
def ST_FP32m  : FPI<0xD9, MRM3m, (outs), (ins f32mem:$dst), "fstp{s}\t$dst">;
def ST_FP64m  : FPI<0xDD, MRM3m, (outs), (ins f64mem:$dst), "fstp{l}\t$dst">;
def ST_FP80m  : FPI<0xDB, MRM7m, (outs), (ins f80mem:$dst), "fstp{t}\t$dst">;
def IST_F16m  : FPI<0xDF, MRM2m, (outs), (ins i16mem:$dst), "fist{s}\t$dst">;
def IST_F32m  : FPI<0xDB, MRM2m, (outs), (ins i32mem:$dst), "fist{l}\t$dst">;
def IST_FP16m : FPI<0xDF, MRM3m, (outs), (ins i16mem:$dst), "fistp{s}\t$dst">;
def IST_FP32m : FPI<0xDB, MRM3m, (outs), (ins i32mem:$dst), "fistp{l}\t$dst">;
def IST_FP64m : FPI<0xDF, MRM7m, (outs), (ins i64mem:$dst), "fistp{ll}\t$dst">;

// FISTTP requires SSE3 even though it's a FPStack op.
def ISTT_Fp16m32 : FpI_<(outs), (ins i16mem:$op, RFP32:$src), OneArgFP,
                    [(X86fp_to_i16mem RFP32:$src, addr:$op)]>,
                    Requires<[HasSSE3]>;
def ISTT_Fp32m32 : FpI_<(outs), (ins i32mem:$op, RFP32:$src), OneArgFP,
                    [(X86fp_to_i32mem RFP32:$src, addr:$op)]>,
                    Requires<[HasSSE3]>;
def ISTT_Fp64m32 : FpI_<(outs), (ins i64mem:$op, RFP32:$src), OneArgFP,
                    [(X86fp_to_i64mem RFP32:$src, addr:$op)]>,
                    Requires<[HasSSE3]>;
def ISTT_Fp16m64 : FpI_<(outs), (ins i16mem:$op, RFP64:$src), OneArgFP,
                    [(X86fp_to_i16mem RFP64:$src, addr:$op)]>,
                    Requires<[HasSSE3]>;
def ISTT_Fp32m64 : FpI_<(outs), (ins i32mem:$op, RFP64:$src), OneArgFP,
                    [(X86fp_to_i32mem RFP64:$src, addr:$op)]>,
                    Requires<[HasSSE3]>;
def ISTT_Fp64m64 : FpI_<(outs), (ins i64mem:$op, RFP64:$src), OneArgFP,
                    [(X86fp_to_i64mem RFP64:$src, addr:$op)]>,
                    Requires<[HasSSE3]>;
def ISTT_Fp16m80 : FpI_<(outs), (ins i16mem:$op, RFP80:$src), OneArgFP,
                    [(X86fp_to_i16mem RFP80:$src, addr:$op)]>,
                    Requires<[HasSSE3]>;
def ISTT_Fp32m80 : FpI_<(outs), (ins i32mem:$op, RFP80:$src), OneArgFP,
                    [(X86fp_to_i32mem RFP80:$src, addr:$op)]>,
                    Requires<[HasSSE3]>;
def ISTT_Fp64m80 : FpI_<(outs), (ins i64mem:$op, RFP80:$src), OneArgFP,
                    [(X86fp_to_i64mem RFP80:$src, addr:$op)]>,
                    Requires<[HasSSE3]>;

def ISTT_FP16m : FPI<0xDF, MRM1m, (outs), (ins i16mem:$dst), "fisttp{s}\t$dst">;
def ISTT_FP32m : FPI<0xDB, MRM1m, (outs), (ins i32mem:$dst), "fisttp{l}\t$dst">;
def ISTT_FP64m : FPI<0xDD, MRM1m, (outs), (ins i64mem:$dst), "fisttp{ll}\t$dst">;

// FP Stack manipulation instructions.
def LD_Frr   : FPI<0xC0, AddRegFrm, (outs), (ins RST:$op), "fld\t$op">, D9;
def ST_Frr   : FPI<0xD0, AddRegFrm, (outs), (ins RST:$op), "fst\t$op">, DD;
def ST_FPrr  : FPI<0xD8, AddRegFrm, (outs), (ins RST:$op), "fstp\t$op">, DD;
def XCH_F    : FPI<0xC8, AddRegFrm, (outs), (ins RST:$op), "fxch\t$op">, D9;

// Floating point constant loads.
let isReMaterializable = 1 in {
def LD_Fp032 : FpIf32<(outs RFP32:$dst), (ins), ZeroArgFP,
                [(set RFP32:$dst, fpimm0)]>;
def LD_Fp132 : FpIf32<(outs RFP32:$dst), (ins), ZeroArgFP,
                [(set RFP32:$dst, fpimm1)]>;
def LD_Fp064 : FpIf64<(outs RFP64:$dst), (ins), ZeroArgFP,
                [(set RFP64:$dst, fpimm0)]>;
def LD_Fp164 : FpIf64<(outs RFP64:$dst), (ins), ZeroArgFP,
                [(set RFP64:$dst, fpimm1)]>;
def LD_Fp080 : FpI_<(outs RFP80:$dst), (ins), ZeroArgFP,
                [(set RFP80:$dst, fpimm0)]>;
def LD_Fp180 : FpI_<(outs RFP80:$dst), (ins), ZeroArgFP,
                [(set RFP80:$dst, fpimm1)]>;
}

def LD_F0 : FPI<0xEE, RawFrm, (outs), (ins), "fldz">, D9;
def LD_F1 : FPI<0xE8, RawFrm, (outs), (ins), "fld1">, D9;


// Floating point compares.
def UCOM_Fpr32 : FpIf32<(outs), (ins RFP32:$lhs, RFP32:$rhs), CompareFP,
                  []>;  // FPSW = cmp ST(0) with ST(i)
def UCOM_FpIr32: FpIf32<(outs), (ins RFP32:$lhs, RFP32:$rhs), CompareFP,
                  [(X86cmp RFP32:$lhs, RFP32:$rhs)]>; // CC = ST(0) cmp ST(i)
def UCOM_Fpr64 : FpIf64<(outs), (ins RFP64:$lhs, RFP64:$rhs), CompareFP,
                  []>;  // FPSW = cmp ST(0) with ST(i)
def UCOM_FpIr64: FpIf64<(outs), (ins RFP64:$lhs, RFP64:$rhs), CompareFP,
                  [(X86cmp RFP64:$lhs, RFP64:$rhs)]>; // CC = ST(0) cmp ST(i)
def UCOM_Fpr80 : FpI_<(outs), (ins RFP80:$lhs, RFP80:$rhs), CompareFP,
                  []>;  // FPSW = cmp ST(0) with ST(i)
def UCOM_FpIr80: FpI_<(outs), (ins RFP80:$lhs, RFP80:$rhs), CompareFP,
                  [(X86cmp RFP80:$lhs, RFP80:$rhs)]>; // CC = ST(0) cmp ST(i)

let Defs = [EFLAGS], Uses = [ST0] in {
def UCOM_Fr    : FPI<0xE0, AddRegFrm,    // FPSW = cmp ST(0) with ST(i)
                    (outs), (ins RST:$reg),
                    "fucom\t$reg">, DD;
def UCOM_FPr   : FPI<0xE8, AddRegFrm,    // FPSW = cmp ST(0) with ST(i), pop
                    (outs), (ins RST:$reg),
                    "fucomp\t$reg">, DD;
def UCOM_FPPr  : FPI<0xE9, RawFrm,       // cmp ST(0) with ST(1), pop, pop
                    (outs), (ins),
                    "fucompp">, DA;

def UCOM_FIr   : FPI<0xE8, AddRegFrm,     // CC = cmp ST(0) with ST(i)
                    (outs), (ins RST:$reg),
                    "fucomi\t{$reg, %st(0)|%ST(0), $reg}">, DB;
def UCOM_FIPr  : FPI<0xE8, AddRegFrm,     // CC = cmp ST(0) with ST(i), pop
                    (outs), (ins RST:$reg),
                    "fucomip\t{$reg, %st(0)|%ST(0), $reg}">, DF;
}

// Floating point flag ops.
let Defs = [AX] in
def FNSTSW8r  : I<0xE0, RawFrm,                  // AX = fp flags
                  (outs), (ins), "fnstsw", []>, DF;

def FNSTCW16m : I<0xD9, MRM7m,                   // [mem16] = X87 control world
                  (outs), (ins i16mem:$dst), "fnstcw\t$dst", []>;
def FLDCW16m  : I<0xD9, MRM5m,                   // X87 control world = [mem16]
                  (outs), (ins i16mem:$dst), "fldcw\t$dst", []>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// Required for RET of f32 / f64 / f80 values.
def : Pat<(X86fld addr:$src, f32), (LD_Fp32m addr:$src)>;
def : Pat<(X86fld addr:$src, f64), (LD_Fp64m addr:$src)>;
def : Pat<(X86fld addr:$src, f80), (LD_Fp80m addr:$src)>;

// Required for CALL which return f32 / f64 / f80 values.
def : Pat<(X86fst RFP32:$src, addr:$op, f32), (ST_Fp32m addr:$op, RFP32:$src)>;
def : Pat<(X86fst RFP64:$src, addr:$op, f32), (ST_Fp64m32 addr:$op, RFP64:$src)>;
def : Pat<(X86fst RFP64:$src, addr:$op, f64), (ST_Fp64m addr:$op, RFP64:$src)>;
def : Pat<(X86fst RFP80:$src, addr:$op, f32), (ST_Fp80m32 addr:$op, RFP80:$src)>;
def : Pat<(X86fst RFP80:$src, addr:$op, f64), (ST_Fp80m64 addr:$op, RFP80:$src)>;
def : Pat<(X86fst RFP80:$src, addr:$op, f80), (ST_FpP80m addr:$op, RFP80:$src)>;

// Floating point constant -0.0 and -1.0
def : Pat<(f32 fpimmneg0), (CHS_Fp32 (LD_Fp032))>, Requires<[FPStackf32]>;
def : Pat<(f32 fpimmneg1), (CHS_Fp32 (LD_Fp132))>, Requires<[FPStackf32]>;
def : Pat<(f64 fpimmneg0), (CHS_Fp64 (LD_Fp064))>, Requires<[FPStackf64]>;
def : Pat<(f64 fpimmneg1), (CHS_Fp64 (LD_Fp164))>, Requires<[FPStackf64]>;
def : Pat<(f80 fpimmneg0), (CHS_Fp80 (LD_Fp080))>;
def : Pat<(f80 fpimmneg1), (CHS_Fp80 (LD_Fp180))>;

// Used to conv. i64 to f64 since there isn't a SSE version.
def : Pat<(X86fildflag addr:$src, i64), (ILD_Fp64m64 addr:$src)>;

def : Pat<(f64 (fextend RFP32:$src)), (MOV_Fp3264 RFP32:$src)>, Requires<[FPStackf32]>;
def : Pat<(f80 (fextend RFP32:$src)), (MOV_Fp3280 RFP32:$src)>, Requires<[FPStackf32]>;
def : Pat<(f80 (fextend RFP64:$src)), (MOV_Fp6480 RFP64:$src)>, Requires<[FPStackf64]>;
