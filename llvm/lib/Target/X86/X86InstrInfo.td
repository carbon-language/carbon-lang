//===- X86InstrInfo.td - Main X86 Instruction Definition ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 instruction set, defining the instructions, and
// properties of the instructions which are needed for code generation, machine
// code emission, and analysis.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// X86 specific DAG Nodes.
//

def SDTIntShiftDOp: SDTypeProfile<1, 3,
                                  [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                                   SDTCisInt<0>, SDTCisInt<3>]>;

def SDTX86CmpTest : SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisSameAs<1, 2>]>;

def SDTX86Cmov    : SDTypeProfile<1, 4,
                                  [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
                                   SDTCisVT<3, i8>, SDTCisVT<4, i32>]>;

// Unary and binary operator instructions that set EFLAGS as a side-effect.
def SDTUnaryArithWithFlags : SDTypeProfile<2, 1,
                                           [SDTCisInt<0>, SDTCisVT<1, i32>]>;

def SDTBinaryArithWithFlags : SDTypeProfile<2, 2,
                                            [SDTCisSameAs<0, 2>,
                                             SDTCisSameAs<0, 3>,
                                             SDTCisInt<0>, SDTCisVT<1, i32>]>;
def SDTX86BrCond  : SDTypeProfile<0, 3,
                                  [SDTCisVT<0, OtherVT>,
                                   SDTCisVT<1, i8>, SDTCisVT<2, i32>]>;

def SDTX86SetCC   : SDTypeProfile<1, 2,
                                  [SDTCisVT<0, i8>,
                                   SDTCisVT<1, i8>, SDTCisVT<2, i32>]>;
def SDTX86SetCC_C : SDTypeProfile<1, 2,
                                  [SDTCisInt<0>,
                                   SDTCisVT<1, i8>, SDTCisVT<2, i32>]>;

def SDTX86cas : SDTypeProfile<0, 3, [SDTCisPtrTy<0>, SDTCisInt<1>, 
                                     SDTCisVT<2, i8>]>;
def SDTX86cas8 : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;

def SDTX86atomicBinary : SDTypeProfile<2, 3, [SDTCisInt<0>, SDTCisInt<1>,
                                SDTCisPtrTy<2>, SDTCisInt<3>,SDTCisInt<4>]>;
def SDTX86Ret     : SDTypeProfile<0, -1, [SDTCisVT<0, i16>]>;

def SDT_X86CallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>]>;
def SDT_X86CallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32>]>;

def SDT_X86Call   : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;

def SDT_X86VASTART_SAVE_XMM_REGS : SDTypeProfile<0, -1, [SDTCisVT<0, i8>,
                                                         SDTCisVT<1, iPTR>,
                                                         SDTCisVT<2, iPTR>]>;

def SDTX86RepStr  : SDTypeProfile<0, 1, [SDTCisVT<0, OtherVT>]>;

def SDTX86Void    : SDTypeProfile<0, 0, []>;

def SDTX86Wrapper : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;

def SDT_X86TLSADDR : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def SDT_X86TLSCALL : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;

def SDT_X86EHRET : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def SDT_X86TCRET : SDTypeProfile<0, 2, [SDTCisPtrTy<0>, SDTCisVT<1, i32>]>;

def SDT_X86MEMBARRIER : SDTypeProfile<0, 0, []>;
def SDT_X86MEMBARRIERNoSSE : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def X86MemBarrier : SDNode<"X86ISD::MEMBARRIER", SDT_X86MEMBARRIER,
                            [SDNPHasChain]>;
def X86MemBarrierNoSSE : SDNode<"X86ISD::MEMBARRIER", SDT_X86MEMBARRIERNoSSE,
                                [SDNPHasChain]>;
def X86MFence : SDNode<"X86ISD::MFENCE", SDT_X86MEMBARRIER,
                        [SDNPHasChain]>;
def X86SFence : SDNode<"X86ISD::SFENCE", SDT_X86MEMBARRIER,
                        [SDNPHasChain]>;
def X86LFence : SDNode<"X86ISD::LFENCE", SDT_X86MEMBARRIER,
                        [SDNPHasChain]>;


def X86bsf     : SDNode<"X86ISD::BSF",      SDTUnaryArithWithFlags>;
def X86bsr     : SDNode<"X86ISD::BSR",      SDTUnaryArithWithFlags>;
def X86shld    : SDNode<"X86ISD::SHLD",     SDTIntShiftDOp>;
def X86shrd    : SDNode<"X86ISD::SHRD",     SDTIntShiftDOp>;

def X86cmp     : SDNode<"X86ISD::CMP" ,     SDTX86CmpTest>;
def X86bt      : SDNode<"X86ISD::BT",       SDTX86CmpTest>;

def X86cmov    : SDNode<"X86ISD::CMOV",     SDTX86Cmov>;
def X86brcond  : SDNode<"X86ISD::BRCOND",   SDTX86BrCond,
                        [SDNPHasChain]>;
def X86setcc   : SDNode<"X86ISD::SETCC",    SDTX86SetCC>;
def X86setcc_c : SDNode<"X86ISD::SETCC_CARRY", SDTX86SetCC_C>;

def X86cas : SDNode<"X86ISD::LCMPXCHG_DAG", SDTX86cas,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86cas8 : SDNode<"X86ISD::LCMPXCHG8_DAG", SDTX86cas8,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomAdd64 : SDNode<"X86ISD::ATOMADD64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore, 
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomSub64 : SDNode<"X86ISD::ATOMSUB64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore, 
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomOr64 : SDNode<"X86ISD::ATOMOR64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore, 
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomXor64 : SDNode<"X86ISD::ATOMXOR64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore, 
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomAnd64 : SDNode<"X86ISD::ATOMAND64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore, 
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomNand64 : SDNode<"X86ISD::ATOMNAND64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore, 
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomSwap64 : SDNode<"X86ISD::ATOMSWAP64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore, 
                         SDNPMayLoad, SDNPMemOperand]>;
def X86retflag : SDNode<"X86ISD::RET_FLAG", SDTX86Ret,
                        [SDNPHasChain, SDNPOptInFlag, SDNPVariadic]>;

def X86vastart_save_xmm_regs :
                 SDNode<"X86ISD::VASTART_SAVE_XMM_REGS",
                        SDT_X86VASTART_SAVE_XMM_REGS,
                        [SDNPHasChain, SDNPVariadic]>;

def X86callseq_start :
                 SDNode<"ISD::CALLSEQ_START", SDT_X86CallSeqStart,
                        [SDNPHasChain, SDNPOutFlag]>;
def X86callseq_end :
                 SDNode<"ISD::CALLSEQ_END",   SDT_X86CallSeqEnd,
                        [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;       

def X86call    : SDNode<"X86ISD::CALL",     SDT_X86Call,
                        [SDNPHasChain, SDNPOutFlag, SDNPOptInFlag,
                         SDNPVariadic]>;

def X86rep_stos: SDNode<"X86ISD::REP_STOS", SDTX86RepStr,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag, SDNPMayStore]>;
def X86rep_movs: SDNode<"X86ISD::REP_MOVS", SDTX86RepStr,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag, SDNPMayStore,
                         SDNPMayLoad]>;

def X86rdtsc   : SDNode<"X86ISD::RDTSC_DAG", SDTX86Void,
                        [SDNPHasChain, SDNPOutFlag, SDNPSideEffect]>;

def X86Wrapper    : SDNode<"X86ISD::Wrapper",     SDTX86Wrapper>;
def X86WrapperRIP : SDNode<"X86ISD::WrapperRIP",  SDTX86Wrapper>;

def X86tlsaddr : SDNode<"X86ISD::TLSADDR", SDT_X86TLSADDR,
                        [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def X86ehret : SDNode<"X86ISD::EH_RETURN", SDT_X86EHRET,
                        [SDNPHasChain]>;

def X86tcret : SDNode<"X86ISD::TC_RETURN", SDT_X86TCRET, 
                        [SDNPHasChain,  SDNPOptInFlag, SDNPVariadic]>;

def X86add_flag  : SDNode<"X86ISD::ADD",  SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86sub_flag  : SDNode<"X86ISD::SUB",  SDTBinaryArithWithFlags>;
def X86smul_flag : SDNode<"X86ISD::SMUL", SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86umul_flag : SDNode<"X86ISD::UMUL", SDTUnaryArithWithFlags,
                          [SDNPCommutative]>;
                          
def X86inc_flag  : SDNode<"X86ISD::INC",  SDTUnaryArithWithFlags>;
def X86dec_flag  : SDNode<"X86ISD::DEC",  SDTUnaryArithWithFlags>;
def X86or_flag   : SDNode<"X86ISD::OR",   SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86xor_flag  : SDNode<"X86ISD::XOR",  SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86and_flag  : SDNode<"X86ISD::AND",  SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;

def X86mul_imm : SDNode<"X86ISD::MUL_IMM", SDTIntBinOp>;

def X86MingwAlloca : SDNode<"X86ISD::MINGW_ALLOCA", SDTX86Void,
                            [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;
                            
def X86TLSCall : SDNode<"X86ISD::TLSCALL", SDT_X86TLSCALL,
                        []>;

//===----------------------------------------------------------------------===//
// X86 Operand Definitions.
//

// A version of ptr_rc which excludes SP, ESP, and RSP. This is used for
// the index operand of an address, to conform to x86 encoding restrictions.
def ptr_rc_nosp : PointerLikeRegClass<1>;

// *mem - Operand definitions for the funky X86 addressing mode operands.
//
def X86MemAsmOperand : AsmOperandClass {
  let Name = "Mem";
  let SuperClasses = [];
}
def X86AbsMemAsmOperand : AsmOperandClass {
  let Name = "AbsMem";
  let SuperClasses = [X86MemAsmOperand];
}
class X86MemOperand<string printMethod> : Operand<iPTR> {
  let PrintMethod = printMethod;
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc_nosp, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
}

def opaque32mem : X86MemOperand<"printopaquemem">;
def opaque48mem : X86MemOperand<"printopaquemem">;
def opaque80mem : X86MemOperand<"printopaquemem">;
def opaque512mem : X86MemOperand<"printopaquemem">;

def i8mem   : X86MemOperand<"printi8mem">;
def i16mem  : X86MemOperand<"printi16mem">;
def i32mem  : X86MemOperand<"printi32mem">;
def i64mem  : X86MemOperand<"printi64mem">;
def i128mem : X86MemOperand<"printi128mem">;
def i256mem : X86MemOperand<"printi256mem">;
def f32mem  : X86MemOperand<"printf32mem">;
def f64mem  : X86MemOperand<"printf64mem">;
def f80mem  : X86MemOperand<"printf80mem">;
def f128mem : X86MemOperand<"printf128mem">;
def f256mem : X86MemOperand<"printf256mem">;

// A version of i8mem for use on x86-64 that uses GR64_NOREX instead of
// plain GR64, so that it doesn't potentially require a REX prefix.
def i8mem_NOREX : Operand<i64> {
  let PrintMethod = "printi8mem";
  let MIOperandInfo = (ops GR64_NOREX, i8imm, GR64_NOREX_NOSP, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
}

// Special i32mem for addresses of load folding tail calls. These are not
// allowed to use callee-saved registers since they must be scheduled
// after callee-saved register are popped.
def i32mem_TC : Operand<i32> {
  let PrintMethod = "printi32mem";
  let MIOperandInfo = (ops GR32_TC, i8imm, GR32_TC, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
}


let ParserMatchClass = X86AbsMemAsmOperand,
    PrintMethod = "print_pcrel_imm" in {
def i32imm_pcrel : Operand<i32>;
def i16imm_pcrel : Operand<i16>;

def offset8 : Operand<i64>;
def offset16 : Operand<i64>;
def offset32 : Operand<i64>;
def offset64 : Operand<i64>;

// Branch targets have OtherVT type and print as pc-relative values.
def brtarget : Operand<OtherVT>;
def brtarget8 : Operand<OtherVT>;

}

def SSECC : Operand<i8> {
  let PrintMethod = "printSSECC";
}

class ImmSExtAsmOperandClass : AsmOperandClass {
  let SuperClasses = [ImmAsmOperand];
  let RenderMethod = "addImmOperands";
}

// Sign-extended immediate classes. We don't need to define the full lattice
// here because there is no instruction with an ambiguity between ImmSExti64i32
// and ImmSExti32i8.
//
// The strange ranges come from the fact that the assembler always works with
// 64-bit immediates, but for a 16-bit target value we want to accept both "-1"
// (which will be a -1ULL), and "0xFF" (-1 in 16-bits).

// [0, 0x7FFFFFFF]                                            |
//   [0xFFFFFFFF80000000, 0xFFFFFFFFFFFFFFFF]
def ImmSExti64i32AsmOperand : ImmSExtAsmOperandClass {
  let Name = "ImmSExti64i32";
}

// [0, 0x0000007F] | [0x000000000000FF80, 0x000000000000FFFF] |
//   [0xFFFFFFFFFFFFFF80, 0xFFFFFFFFFFFFFFFF]
def ImmSExti16i8AsmOperand : ImmSExtAsmOperandClass {
  let Name = "ImmSExti16i8";
  let SuperClasses = [ImmSExti64i32AsmOperand];
}

// [0, 0x0000007F] | [0x00000000FFFFFF80, 0x00000000FFFFFFFF] |
//   [0xFFFFFFFFFFFFFF80, 0xFFFFFFFFFFFFFFFF]
def ImmSExti32i8AsmOperand : ImmSExtAsmOperandClass {
  let Name = "ImmSExti32i8";
}

// [0, 0x0000007F]                                            |
//   [0xFFFFFFFFFFFFFF80, 0xFFFFFFFFFFFFFFFF]
def ImmSExti64i8AsmOperand : ImmSExtAsmOperandClass {
  let Name = "ImmSExti64i8";
  let SuperClasses = [ImmSExti16i8AsmOperand, ImmSExti32i8AsmOperand,
                      ImmSExti64i32AsmOperand];
}

// A couple of more descriptive operand definitions.
// 16-bits but only 8 bits are significant.
def i16i8imm  : Operand<i16> {
  let ParserMatchClass = ImmSExti16i8AsmOperand;
}
// 32-bits but only 8 bits are significant.
def i32i8imm  : Operand<i32> {
  let ParserMatchClass = ImmSExti32i8AsmOperand;
}

//===----------------------------------------------------------------------===//
// X86 Complex Pattern Definitions.
//

// Define X86 specific addressing mode.
def addr      : ComplexPattern<iPTR, 5, "SelectAddr", [], [SDNPWantParent]>;
def lea32addr : ComplexPattern<i32, 5, "SelectLEAAddr",
                               [add, sub, mul, X86mul_imm, shl, or, frameindex],
                               []>;
def tls32addr : ComplexPattern<i32, 5, "SelectTLSADDRAddr",
                               [tglobaltlsaddr], []>;

//===----------------------------------------------------------------------===//
// X86 Instruction Predicate Definitions.
def HasCMov      : Predicate<"Subtarget->hasCMov()">;
def NoCMov       : Predicate<"!Subtarget->hasCMov()">;

// FIXME: temporary hack to let codegen assert or generate poor code in case
// no AVX version of the desired intructions is present, this is better for
// incremental dev (without fallbacks it's easier to spot what's missing)
def HasMMX       : Predicate<"Subtarget->hasMMX() && !Subtarget->hasAVX()">;
def Has3DNow     : Predicate<"Subtarget->has3DNow()">;
def Has3DNowA    : Predicate<"Subtarget->has3DNowA()">;
def HasSSE1      : Predicate<"Subtarget->hasSSE1() && !Subtarget->hasAVX()">;
def HasSSE2      : Predicate<"Subtarget->hasSSE2() && !Subtarget->hasAVX()">;
def HasSSE3      : Predicate<"Subtarget->hasSSE3() && !Subtarget->hasAVX()">;
def HasSSSE3     : Predicate<"Subtarget->hasSSSE3() && !Subtarget->hasAVX()">;
def HasSSE41     : Predicate<"Subtarget->hasSSE41() && !Subtarget->hasAVX()">;
def HasSSE42     : Predicate<"Subtarget->hasSSE42() && !Subtarget->hasAVX()">;
def HasSSE4A     : Predicate<"Subtarget->hasSSE4A() && !Subtarget->hasAVX()">;

def HasAVX       : Predicate<"Subtarget->hasAVX()">;
def HasCLMUL     : Predicate<"Subtarget->hasCLMUL()">;
def HasFMA3      : Predicate<"Subtarget->hasFMA3()">;
def HasFMA4      : Predicate<"Subtarget->hasFMA4()">;
def FPStackf32   : Predicate<"!Subtarget->hasSSE1()">;
def FPStackf64   : Predicate<"!Subtarget->hasSSE2()">;
def In32BitMode  : Predicate<"!Subtarget->is64Bit()">;
def In64BitMode  : Predicate<"Subtarget->is64Bit()">;
def IsWin64      : Predicate<"Subtarget->isTargetWin64()">;
def NotWin64     : Predicate<"!Subtarget->isTargetWin64()">;
def SmallCode    : Predicate<"TM.getCodeModel() == CodeModel::Small">;
def KernelCode   : Predicate<"TM.getCodeModel() == CodeModel::Kernel">;
def FarData      : Predicate<"TM.getCodeModel() != CodeModel::Small &&"
                             "TM.getCodeModel() != CodeModel::Kernel">;
def NearData     : Predicate<"TM.getCodeModel() == CodeModel::Small ||"
                             "TM.getCodeModel() == CodeModel::Kernel">;
def IsStatic     : Predicate<"TM.getRelocationModel() == Reloc::Static">;
def IsNotPIC     : Predicate<"TM.getRelocationModel() != Reloc::PIC_">;
def OptForSize   : Predicate<"OptForSize">;
def OptForSpeed  : Predicate<"!OptForSize">;
def FastBTMem    : Predicate<"!Subtarget->isBTMemSlow()">;
def CallImmAddr  : Predicate<"Subtarget->IsLegalToCallImmediateAddr(TM)">;
def HasAES       : Predicate<"Subtarget->hasAES()">;

//===----------------------------------------------------------------------===//
// X86 Instruction Format Definitions.
//

include "X86InstrFormats.td"

//===----------------------------------------------------------------------===//
// Pattern fragments...
//

// X86 specific condition code. These correspond to CondCode in
// X86InstrInfo.h. They must be kept in synch.
def X86_COND_A   : PatLeaf<(i8 0)>;  // alt. COND_NBE
def X86_COND_AE  : PatLeaf<(i8 1)>;  // alt. COND_NC
def X86_COND_B   : PatLeaf<(i8 2)>;  // alt. COND_C
def X86_COND_BE  : PatLeaf<(i8 3)>;  // alt. COND_NA
def X86_COND_E   : PatLeaf<(i8 4)>;  // alt. COND_Z
def X86_COND_G   : PatLeaf<(i8 5)>;  // alt. COND_NLE
def X86_COND_GE  : PatLeaf<(i8 6)>;  // alt. COND_NL
def X86_COND_L   : PatLeaf<(i8 7)>;  // alt. COND_NGE
def X86_COND_LE  : PatLeaf<(i8 8)>;  // alt. COND_NG
def X86_COND_NE  : PatLeaf<(i8 9)>;  // alt. COND_NZ
def X86_COND_NO  : PatLeaf<(i8 10)>;
def X86_COND_NP  : PatLeaf<(i8 11)>; // alt. COND_PO
def X86_COND_NS  : PatLeaf<(i8 12)>;
def X86_COND_O   : PatLeaf<(i8 13)>;
def X86_COND_P   : PatLeaf<(i8 14)>; // alt. COND_PE
def X86_COND_S   : PatLeaf<(i8 15)>;

def immSext8 : PatLeaf<(imm), [{ return immSext8(N); }]>;

def i16immSExt8  : PatLeaf<(i16 immSext8)>;
def i32immSExt8  : PatLeaf<(i32 immSext8)>;

// Helper fragments for loads.
// It's always safe to treat a anyext i16 load as a i32 load if the i16 is
// known to be 32-bit aligned or better. Ditto for i8 to i16.
def loadi16 : PatFrag<(ops node:$ptr), (i16 (unindexedload node:$ptr)), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType == ISD::NON_EXTLOAD)
    return true;
  if (ExtType == ISD::EXTLOAD)
    return LD->getAlignment() >= 2 && !LD->isVolatile();
  return false;
}]>;

def loadi16_anyext : PatFrag<(ops node:$ptr), (i32 (unindexedload node:$ptr)),[{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType == ISD::EXTLOAD)
    return LD->getAlignment() >= 2 && !LD->isVolatile();
  return false;
}]>;

def loadi32 : PatFrag<(ops node:$ptr), (i32 (unindexedload node:$ptr)), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType == ISD::NON_EXTLOAD)
    return true;
  if (ExtType == ISD::EXTLOAD)
    return LD->getAlignment() >= 4 && !LD->isVolatile();
  return false;
}]>;

def loadi8  : PatFrag<(ops node:$ptr), (i8  (load node:$ptr))>;
def loadi64 : PatFrag<(ops node:$ptr), (i64 (load node:$ptr))>;
def loadf32 : PatFrag<(ops node:$ptr), (f32 (load node:$ptr))>;
def loadf64 : PatFrag<(ops node:$ptr), (f64 (load node:$ptr))>;
def loadf80 : PatFrag<(ops node:$ptr), (f80 (load node:$ptr))>;

def sextloadi16i8  : PatFrag<(ops node:$ptr), (i16 (sextloadi8 node:$ptr))>;
def sextloadi32i8  : PatFrag<(ops node:$ptr), (i32 (sextloadi8 node:$ptr))>;
def sextloadi32i16 : PatFrag<(ops node:$ptr), (i32 (sextloadi16 node:$ptr))>;

def zextloadi8i1   : PatFrag<(ops node:$ptr), (i8  (zextloadi1 node:$ptr))>;
def zextloadi16i1  : PatFrag<(ops node:$ptr), (i16 (zextloadi1 node:$ptr))>;
def zextloadi32i1  : PatFrag<(ops node:$ptr), (i32 (zextloadi1 node:$ptr))>;
def zextloadi16i8  : PatFrag<(ops node:$ptr), (i16 (zextloadi8 node:$ptr))>;
def zextloadi32i8  : PatFrag<(ops node:$ptr), (i32 (zextloadi8 node:$ptr))>;
def zextloadi32i16 : PatFrag<(ops node:$ptr), (i32 (zextloadi16 node:$ptr))>;

def extloadi8i1    : PatFrag<(ops node:$ptr), (i8  (extloadi1 node:$ptr))>;
def extloadi16i1   : PatFrag<(ops node:$ptr), (i16 (extloadi1 node:$ptr))>;
def extloadi32i1   : PatFrag<(ops node:$ptr), (i32 (extloadi1 node:$ptr))>;
def extloadi16i8   : PatFrag<(ops node:$ptr), (i16 (extloadi8 node:$ptr))>;
def extloadi32i8   : PatFrag<(ops node:$ptr), (i32 (extloadi8 node:$ptr))>;
def extloadi32i16  : PatFrag<(ops node:$ptr), (i32 (extloadi16 node:$ptr))>;


// An 'and' node with a single use.
def and_su : PatFrag<(ops node:$lhs, node:$rhs), (and node:$lhs, node:$rhs), [{
  return N->hasOneUse();
}]>;
// An 'srl' node with a single use.
def srl_su : PatFrag<(ops node:$lhs, node:$rhs), (srl node:$lhs, node:$rhs), [{
  return N->hasOneUse();
}]>;
// An 'trunc' node with a single use.
def trunc_su : PatFrag<(ops node:$src), (trunc node:$src), [{
  return N->hasOneUse();
}]>;

// Treat an 'or' node is as an 'add' if the or'ed bits are known to be zero.
def or_is_add : PatFrag<(ops node:$lhs, node:$rhs), (or node:$lhs, node:$rhs),[{
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    return CurDAG->MaskedValueIsZero(N->getOperand(0), CN->getAPIntValue());

  unsigned BitWidth = N->getValueType(0).getScalarType().getSizeInBits();
  APInt Mask = APInt::getAllOnesValue(BitWidth);
  APInt KnownZero0, KnownOne0;
  CurDAG->ComputeMaskedBits(N->getOperand(0), Mask, KnownZero0, KnownOne0, 0);
  APInt KnownZero1, KnownOne1;
  CurDAG->ComputeMaskedBits(N->getOperand(1), Mask, KnownZero1, KnownOne1, 0);
  return (~KnownZero0 & ~KnownZero1) == 0;
}]>;

//===----------------------------------------------------------------------===//
// Instruction list.
//

// ADJCALLSTACKDOWN/UP implicitly use/def ESP because they may be expanded into
// a stack adjustment and the codegen must know that they may modify the stack
// pointer before prolog-epilog rewriting occurs.
// Pessimistically assume ADJCALLSTACKDOWN / ADJCALLSTACKUP will become
// sub / add which can clobber EFLAGS.
let Defs = [ESP, EFLAGS], Uses = [ESP] in {
def ADJCALLSTACKDOWN32 : I<0, Pseudo, (outs), (ins i32imm:$amt),
                           "#ADJCALLSTACKDOWN",
                           [(X86callseq_start timm:$amt)]>,
                          Requires<[In32BitMode]>;
def ADJCALLSTACKUP32   : I<0, Pseudo, (outs), (ins i32imm:$amt1, i32imm:$amt2),
                           "#ADJCALLSTACKUP",
                           [(X86callseq_end timm:$amt1, timm:$amt2)]>,
                          Requires<[In32BitMode]>;
}

// x86-64 va_start lowering magic.
let usesCustomInserter = 1 in {
def VASTART_SAVE_XMM_REGS : I<0, Pseudo,
                              (outs),
                              (ins GR8:$al,
                                   i64imm:$regsavefi, i64imm:$offset,
                                   variable_ops),
                              "#VASTART_SAVE_XMM_REGS $al, $regsavefi, $offset",
                              [(X86vastart_save_xmm_regs GR8:$al,
                                                         imm:$regsavefi,
                                                         imm:$offset)]>;

// Dynamic stack allocation yields _alloca call for Cygwin/Mingw targets.  Calls
// to _alloca is needed to probe the stack when allocating more than 4k bytes in
// one go. Touching the stack at 4K increments is necessary to ensure that the
// guard pages used by the OS virtual memory manager are allocated in correct
// sequence.
// The main point of having separate instruction are extra unmodelled effects
// (compared to ordinary calls) like stack pointer change.

let Defs = [EAX, ESP, EFLAGS], Uses = [ESP] in
  def MINGW_ALLOCA : I<0, Pseudo, (outs), (ins),
                       "# dynamic stack allocation",
                       [(X86MingwAlloca)]>;
}

// Nop
let neverHasSideEffects = 1 in {
  def NOOP : I<0x90, RawFrm, (outs), (ins), "nop", []>;
  def NOOPW : I<0x1f, MRM0m, (outs), (ins i16mem:$zero),
                "nop{w}\t$zero", []>, TB, OpSize;
  def NOOPL : I<0x1f, MRM0m, (outs), (ins i32mem:$zero),
                "nop{l}\t$zero", []>, TB;
}

// PIC base construction.  This expands to code that looks like this:
//     call  $next_inst
//     popl %destreg"
let neverHasSideEffects = 1, isNotDuplicable = 1, Uses = [ESP] in
  def MOVPC32r : Ii32<0xE8, Pseudo, (outs GR32:$reg), (ins i32imm:$label),
                      "", []>;


// Constructing a stack frame.
def ENTER : Ii16<0xC8, RawFrmImm8, (outs), (ins i16imm:$len, i8imm:$lvl),
                 "enter\t$len, $lvl", []>;

let Defs = [EBP, ESP], Uses = [EBP, ESP], mayLoad = 1, neverHasSideEffects=1 in
def LEAVE    : I<0xC9, RawFrm,
                 (outs), (ins), "leave", []>, Requires<[In32BitMode]>;

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//
def POPCNT16rr : I<0xB8, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                   "popcnt{w}\t{$src, $dst|$dst, $src}", []>, OpSize, XS;
let mayLoad = 1 in
def POPCNT16rm : I<0xB8, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                   "popcnt{w}\t{$src, $dst|$dst, $src}", []>, OpSize, XS;
def POPCNT32rr : I<0xB8, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                   "popcnt{l}\t{$src, $dst|$dst, $src}", []>, XS;
let mayLoad = 1 in
def POPCNT32rm : I<0xB8, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                   "popcnt{l}\t{$src, $dst|$dst, $src}", []>, XS;

let Defs = [ESP], Uses = [ESP], neverHasSideEffects=1 in {
let mayLoad = 1 in {
def POP16r  : I<0x58, AddRegFrm, (outs GR16:$reg), (ins), "pop{w}\t$reg", []>,
  OpSize;
def POP32r  : I<0x58, AddRegFrm, (outs GR32:$reg), (ins), "pop{l}\t$reg", []>;
def POP16rmr: I<0x8F, MRM0r, (outs GR16:$reg), (ins), "pop{w}\t$reg", []>,
  OpSize;
def POP16rmm: I<0x8F, MRM0m, (outs i16mem:$dst), (ins), "pop{w}\t$dst", []>,
  OpSize;
def POP32rmr: I<0x8F, MRM0r, (outs GR32:$reg), (ins), "pop{l}\t$reg", []>;
def POP32rmm: I<0x8F, MRM0m, (outs i32mem:$dst), (ins), "pop{l}\t$dst", []>;
}

let mayStore = 1 in {
def PUSH16r  : I<0x50, AddRegFrm, (outs), (ins GR16:$reg), "push{w}\t$reg",[]>,
  OpSize;
def PUSH32r  : I<0x50, AddRegFrm, (outs), (ins GR32:$reg), "push{l}\t$reg",[]>;
def PUSH16rmr: I<0xFF, MRM6r, (outs), (ins GR16:$reg), "push{w}\t$reg",[]>,
  OpSize;
def PUSH16rmm: I<0xFF, MRM6m, (outs), (ins i16mem:$src), "push{w}\t$src",[]>,
  OpSize;
def PUSH32rmr: I<0xFF, MRM6r, (outs), (ins GR32:$reg), "push{l}\t$reg",[]>;
def PUSH32rmm: I<0xFF, MRM6m, (outs), (ins i32mem:$src), "push{l}\t$src",[]>;
}
}

let Defs = [ESP], Uses = [ESP], neverHasSideEffects = 1, mayStore = 1 in {
def PUSHi8   : Ii8<0x6a, RawFrm, (outs), (ins i32i8imm:$imm), 
                      "push{l}\t$imm", []>;
def PUSHi16  : Ii16<0x68, RawFrm, (outs), (ins i16imm:$imm), 
                      "push{w}\t$imm", []>, OpSize;
def PUSHi32  : Ii32<0x68, RawFrm, (outs), (ins i32imm:$imm), 
                      "push{l}\t$imm", []>;
}

let Defs = [ESP, EFLAGS], Uses = [ESP], mayLoad = 1, neverHasSideEffects=1 in {
def POPF16   : I<0x9D, RawFrm, (outs), (ins), "popf{w}", []>, OpSize;
def POPF32   : I<0x9D, RawFrm, (outs), (ins), "popf{l|d}", []>,
               Requires<[In32BitMode]>;
}
let Defs = [ESP], Uses = [ESP, EFLAGS], mayStore = 1, neverHasSideEffects=1 in {
def PUSHF16  : I<0x9C, RawFrm, (outs), (ins), "pushf{w}", []>, OpSize;
def PUSHF32  : I<0x9C, RawFrm, (outs), (ins), "pushf{l|d}", []>,
               Requires<[In32BitMode]>;
}

let Defs = [EDI, ESI, EBP, EBX, EDX, ECX, EAX, ESP], Uses = [ESP],
    mayLoad=1, neverHasSideEffects=1 in {
def POPA32   : I<0x61, RawFrm, (outs), (ins), "popa{l}", []>,
               Requires<[In32BitMode]>;
}
let Defs = [ESP], Uses = [EDI, ESI, EBP, EBX, EDX, ECX, EAX, ESP],
    mayStore=1, neverHasSideEffects=1 in {
def PUSHA32  : I<0x60, RawFrm, (outs), (ins), "pusha{l}", []>,
               Requires<[In32BitMode]>;
}

let Uses = [EFLAGS], Constraints = "$src = $dst" in     // GR32 = bswap GR32
  def BSWAP32r : I<0xC8, AddRegFrm,
                   (outs GR32:$dst), (ins GR32:$src),
                   "bswap{l}\t$dst", 
                   [(set GR32:$dst, (bswap GR32:$src))]>, TB;


// Bit scan instructions.
let Defs = [EFLAGS] in {
def BSF16rr  : I<0xBC, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                 "bsf{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, EFLAGS, (X86bsf GR16:$src))]>, TB, OpSize;
def BSF16rm  : I<0xBC, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                 "bsf{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, EFLAGS, (X86bsf (loadi16 addr:$src)))]>, TB,
                 OpSize;
def BSF32rr  : I<0xBC, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                 "bsf{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, EFLAGS, (X86bsf GR32:$src))]>, TB;
def BSF32rm  : I<0xBC, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                 "bsf{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, EFLAGS, (X86bsf (loadi32 addr:$src)))]>, TB;

def BSR16rr  : I<0xBD, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                 "bsr{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, EFLAGS, (X86bsr GR16:$src))]>, TB, OpSize;
def BSR16rm  : I<0xBD, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                 "bsr{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, EFLAGS, (X86bsr (loadi16 addr:$src)))]>, TB,
                 OpSize;
def BSR32rr  : I<0xBD, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                 "bsr{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, EFLAGS, (X86bsr GR32:$src))]>, TB;
def BSR32rm  : I<0xBD, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                 "bsr{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, EFLAGS, (X86bsr (loadi32 addr:$src)))]>, TB;
} // Defs = [EFLAGS]

let neverHasSideEffects = 1 in
def LEA16r   : I<0x8D, MRMSrcMem,
                 (outs GR16:$dst), (ins i32mem:$src),
                 "lea{w}\t{$src|$dst}, {$dst|$src}", []>, OpSize;
let isReMaterializable = 1 in
def LEA32r   : I<0x8D, MRMSrcMem,
                 (outs GR32:$dst), (ins i32mem:$src),
                 "lea{l}\t{$src|$dst}, {$dst|$src}",
                 [(set GR32:$dst, lea32addr:$src)]>, Requires<[In32BitMode]>;

let Defs = [ECX,EDI,ESI], Uses = [ECX,EDI,ESI], isCodeGenOnly = 1 in {
def REP_MOVSB : I<0xA4, RawFrm, (outs), (ins), "{rep;movsb|rep movsb}",
                  [(X86rep_movs i8)]>, REP;
def REP_MOVSW : I<0xA5, RawFrm, (outs), (ins), "{rep;movsw|rep movsw}",
                  [(X86rep_movs i16)]>, REP, OpSize;
def REP_MOVSD : I<0xA5, RawFrm, (outs), (ins), "{rep;movsl|rep movsd}",
                  [(X86rep_movs i32)]>, REP;
}

// These uses the DF flag in the EFLAGS register to inc or dec EDI and ESI
let Defs = [EDI,ESI], Uses = [EDI,ESI,EFLAGS] in {
def MOVSB : I<0xA4, RawFrm, (outs), (ins), "{movsb}", []>;
def MOVSW : I<0xA5, RawFrm, (outs), (ins), "{movsw}", []>, OpSize;
def MOVSD : I<0xA5, RawFrm, (outs), (ins), "{movsl|movsd}", []>;
}

let Defs = [ECX,EDI], Uses = [AL,ECX,EDI], isCodeGenOnly = 1 in
def REP_STOSB : I<0xAA, RawFrm, (outs), (ins), "{rep;stosb|rep stosb}",
                  [(X86rep_stos i8)]>, REP;
let Defs = [ECX,EDI], Uses = [AX,ECX,EDI], isCodeGenOnly = 1 in
def REP_STOSW : I<0xAB, RawFrm, (outs), (ins), "{rep;stosw|rep stosw}",
                  [(X86rep_stos i16)]>, REP, OpSize;
let Defs = [ECX,EDI], Uses = [EAX,ECX,EDI], isCodeGenOnly = 1 in
def REP_STOSD : I<0xAB, RawFrm, (outs), (ins), "{rep;stosl|rep stosd}",
                  [(X86rep_stos i32)]>, REP;

// These uses the DF flag in the EFLAGS register to inc or dec EDI and ESI
let Defs = [EDI], Uses = [AL,EDI,EFLAGS] in
def STOSB : I<0xAA, RawFrm, (outs), (ins), "{stosb}", []>;
let Defs = [EDI], Uses = [AX,EDI,EFLAGS] in
def STOSW : I<0xAB, RawFrm, (outs), (ins), "{stosw}", []>, OpSize;
let Defs = [EDI], Uses = [EAX,EDI,EFLAGS] in
def STOSD : I<0xAB, RawFrm, (outs), (ins), "{stosl|stosd}", []>;

def SCAS8 : I<0xAE, RawFrm, (outs), (ins), "scas{b}", []>;
def SCAS16 : I<0xAF, RawFrm, (outs), (ins), "scas{w}", []>, OpSize;
def SCAS32 : I<0xAF, RawFrm, (outs), (ins), "scas{l}", []>;

def CMPS8 : I<0xA6, RawFrm, (outs), (ins), "cmps{b}", []>;
def CMPS16 : I<0xA7, RawFrm, (outs), (ins), "cmps{w}", []>, OpSize;
def CMPS32 : I<0xA7, RawFrm, (outs), (ins), "cmps{l}", []>;


//===----------------------------------------------------------------------===//
//  Move Instructions.
//
let neverHasSideEffects = 1 in {
def MOV8rr  : I<0x88, MRMDestReg, (outs GR8 :$dst), (ins GR8 :$src),
                "mov{b}\t{$src, $dst|$dst, $src}", []>;
def MOV16rr : I<0x89, MRMDestReg, (outs GR16:$dst), (ins GR16:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32rr : I<0x89, MRMDestReg, (outs GR32:$dst), (ins GR32:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;
}
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def MOV8ri  : Ii8 <0xB0, AddRegFrm, (outs GR8 :$dst), (ins i8imm :$src),
                   "mov{b}\t{$src, $dst|$dst, $src}",
                   [(set GR8:$dst, imm:$src)]>;
def MOV16ri : Ii16<0xB8, AddRegFrm, (outs GR16:$dst), (ins i16imm:$src),
                   "mov{w}\t{$src, $dst|$dst, $src}",
                   [(set GR16:$dst, imm:$src)]>, OpSize;
def MOV32ri : Ii32<0xB8, AddRegFrm, (outs GR32:$dst), (ins i32imm:$src),
                   "mov{l}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, imm:$src)]>;
}

def MOV8mi  : Ii8 <0xC6, MRM0m, (outs), (ins i8mem :$dst, i8imm :$src),
                   "mov{b}\t{$src, $dst|$dst, $src}",
                   [(store (i8 imm:$src), addr:$dst)]>;
def MOV16mi : Ii16<0xC7, MRM0m, (outs), (ins i16mem:$dst, i16imm:$src),
                   "mov{w}\t{$src, $dst|$dst, $src}",
                   [(store (i16 imm:$src), addr:$dst)]>, OpSize;
def MOV32mi : Ii32<0xC7, MRM0m, (outs), (ins i32mem:$dst, i32imm:$src),
                   "mov{l}\t{$src, $dst|$dst, $src}",
                   [(store (i32 imm:$src), addr:$dst)]>;

/// moffs8, moffs16 and moffs32 versions of moves.  The immediate is a
/// 32-bit offset from the PC.  These are only valid in x86-32 mode.
def MOV8o8a : Ii32 <0xA0, RawFrm, (outs), (ins offset8:$src),
                   "mov{b}\t{$src, %al|%al, $src}", []>,
                   Requires<[In32BitMode]>;
def MOV16o16a : Ii32 <0xA1, RawFrm, (outs), (ins offset16:$src),
                      "mov{w}\t{$src, %ax|%ax, $src}", []>, OpSize,
                     Requires<[In32BitMode]>;
def MOV32o32a : Ii32 <0xA1, RawFrm, (outs), (ins offset32:$src),
                      "mov{l}\t{$src, %eax|%eax, $src}", []>,
                     Requires<[In32BitMode]>;
def MOV8ao8 : Ii32 <0xA2, RawFrm, (outs offset8:$dst), (ins),
                   "mov{b}\t{%al, $dst|$dst, %al}", []>,
                  Requires<[In32BitMode]>;
def MOV16ao16 : Ii32 <0xA3, RawFrm, (outs offset16:$dst), (ins),
                      "mov{w}\t{%ax, $dst|$dst, %ax}", []>, OpSize,
                     Requires<[In32BitMode]>;
def MOV32ao32 : Ii32 <0xA3, RawFrm, (outs offset32:$dst), (ins),
                      "mov{l}\t{%eax, $dst|$dst, %eax}", []>,
                     Requires<[In32BitMode]>;
                      

let isCodeGenOnly = 1 in {
def MOV8rr_REV : I<0x8A, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src),
                   "mov{b}\t{$src, $dst|$dst, $src}", []>;
def MOV16rr_REV : I<0x8B, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                    "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32rr_REV : I<0x8B, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                    "mov{l}\t{$src, $dst|$dst, $src}", []>;
}

let canFoldAsLoad = 1, isReMaterializable = 1 in {
def MOV8rm  : I<0x8A, MRMSrcMem, (outs GR8 :$dst), (ins i8mem :$src),
                "mov{b}\t{$src, $dst|$dst, $src}",
                [(set GR8:$dst, (loadi8 addr:$src))]>;
def MOV16rm : I<0x8B, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                "mov{w}\t{$src, $dst|$dst, $src}",
                [(set GR16:$dst, (loadi16 addr:$src))]>, OpSize;
def MOV32rm : I<0x8B, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                "mov{l}\t{$src, $dst|$dst, $src}",
                [(set GR32:$dst, (loadi32 addr:$src))]>;
}

def MOV8mr  : I<0x88, MRMDestMem, (outs), (ins i8mem :$dst, GR8 :$src),
                "mov{b}\t{$src, $dst|$dst, $src}",
                [(store GR8:$src, addr:$dst)]>;
def MOV16mr : I<0x89, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src),
                "mov{w}\t{$src, $dst|$dst, $src}",
                [(store GR16:$src, addr:$dst)]>, OpSize;
def MOV32mr : I<0x89, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                "mov{l}\t{$src, $dst|$dst, $src}",
                [(store GR32:$src, addr:$dst)]>;

/// Versions of MOV32rr, MOV32rm, and MOV32mr for i32mem_TC and GR32_TC.
let isCodeGenOnly = 1 in {
let neverHasSideEffects = 1 in
def MOV32rr_TC : I<0x89, MRMDestReg, (outs GR32_TC:$dst), (ins GR32_TC:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;

let mayLoad = 1,
    canFoldAsLoad = 1, isReMaterializable = 1 in
def MOV32rm_TC : I<0x8B, MRMSrcMem, (outs GR32_TC:$dst), (ins i32mem_TC:$src),
                "mov{l}\t{$src, $dst|$dst, $src}",
                []>;

let mayStore = 1 in
def MOV32mr_TC : I<0x89, MRMDestMem, (outs), (ins i32mem_TC:$dst, GR32_TC:$src),
                "mov{l}\t{$src, $dst|$dst, $src}",
                []>;
}

// Versions of MOV8rr, MOV8mr, and MOV8rm that use i8mem_NOREX and GR8_NOREX so
// that they can be used for copying and storing h registers, which can't be
// encoded when a REX prefix is present.
let isCodeGenOnly = 1 in {
let neverHasSideEffects = 1 in
def MOV8rr_NOREX : I<0x88, MRMDestReg,
                     (outs GR8_NOREX:$dst), (ins GR8_NOREX:$src),
                     "mov{b}\t{$src, $dst|$dst, $src}  # NOREX", []>;
let mayStore = 1 in
def MOV8mr_NOREX : I<0x88, MRMDestMem,
                     (outs), (ins i8mem_NOREX:$dst, GR8_NOREX:$src),
                     "mov{b}\t{$src, $dst|$dst, $src}  # NOREX", []>;
let mayLoad = 1,
    canFoldAsLoad = 1, isReMaterializable = 1 in
def MOV8rm_NOREX : I<0x8A, MRMSrcMem,
                     (outs GR8_NOREX:$dst), (ins i8mem_NOREX:$src),
                     "mov{b}\t{$src, $dst|$dst, $src}  # NOREX", []>;
}

//===----------------------------------------------------------------------===//
//  Fixed-Register Multiplication and Division Instructions...
//

// Extra precision multiplication

// AL is really implied by AX, but the registers in Defs must match the
// SDNode results (i8, i32).
let Defs = [AL,EFLAGS,AX], Uses = [AL] in
def MUL8r  : I<0xF6, MRM4r, (outs),  (ins GR8:$src), "mul{b}\t$src",
               // FIXME: Used for 8-bit mul, ignore result upper 8 bits.
               // This probably ought to be moved to a def : Pat<> if the
               // syntax can be accepted.
               [(set AL, (mul AL, GR8:$src)),
                (implicit EFLAGS)]>;     // AL,AH = AL*GR8

let Defs = [AX,DX,EFLAGS], Uses = [AX], neverHasSideEffects = 1 in
def MUL16r : I<0xF7, MRM4r, (outs),  (ins GR16:$src),
               "mul{w}\t$src", 
               []>, OpSize;    // AX,DX = AX*GR16

let Defs = [EAX,EDX,EFLAGS], Uses = [EAX], neverHasSideEffects = 1 in
def MUL32r : I<0xF7, MRM4r, (outs),  (ins GR32:$src),
               "mul{l}\t$src",
               []>; // EAX,EDX = EAX*GR32

let Defs = [AL,EFLAGS,AX], Uses = [AL] in
def MUL8m  : I<0xF6, MRM4m, (outs), (ins i8mem :$src),
               "mul{b}\t$src",
               // FIXME: Used for 8-bit mul, ignore result upper 8 bits.
               // This probably ought to be moved to a def : Pat<> if the
               // syntax can be accepted.
               [(set AL, (mul AL, (loadi8 addr:$src))),
                (implicit EFLAGS)]>;   // AL,AH = AL*[mem8]

let mayLoad = 1, neverHasSideEffects = 1 in {
let Defs = [AX,DX,EFLAGS], Uses = [AX] in
def MUL16m : I<0xF7, MRM4m, (outs), (ins i16mem:$src),
               "mul{w}\t$src",
               []>, OpSize; // AX,DX = AX*[mem16]

let Defs = [EAX,EDX,EFLAGS], Uses = [EAX] in
def MUL32m : I<0xF7, MRM4m, (outs), (ins i32mem:$src),
              "mul{l}\t$src",
              []>;          // EAX,EDX = EAX*[mem32]
}

let neverHasSideEffects = 1 in {
let Defs = [AL,EFLAGS,AX], Uses = [AL] in
def IMUL8r  : I<0xF6, MRM5r, (outs),  (ins GR8:$src), "imul{b}\t$src", []>;
              // AL,AH = AL*GR8
let Defs = [AX,DX,EFLAGS], Uses = [AX] in
def IMUL16r : I<0xF7, MRM5r, (outs),  (ins GR16:$src), "imul{w}\t$src", []>,
              OpSize;    // AX,DX = AX*GR16
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX] in
def IMUL32r : I<0xF7, MRM5r, (outs),  (ins GR32:$src), "imul{l}\t$src", []>;
              // EAX,EDX = EAX*GR32
let mayLoad = 1 in {
let Defs = [AL,EFLAGS,AX], Uses = [AL] in
def IMUL8m  : I<0xF6, MRM5m, (outs), (ins i8mem :$src),
                "imul{b}\t$src", []>;    // AL,AH = AL*[mem8]
let Defs = [AX,DX,EFLAGS], Uses = [AX] in
def IMUL16m : I<0xF7, MRM5m, (outs), (ins i16mem:$src),
                "imul{w}\t$src", []>, OpSize; // AX,DX = AX*[mem16]
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX] in
def IMUL32m : I<0xF7, MRM5m, (outs), (ins i32mem:$src),
                "imul{l}\t$src", []>;  // EAX,EDX = EAX*[mem32]
}
} // neverHasSideEffects

// unsigned division/remainder
let Defs = [AL,EFLAGS,AX], Uses = [AX] in
def DIV8r  : I<0xF6, MRM6r, (outs),  (ins GR8:$src),    // AX/r8 = AL,AH
               "div{b}\t$src", []>;
let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
def DIV16r : I<0xF7, MRM6r, (outs),  (ins GR16:$src),   // DX:AX/r16 = AX,DX
               "div{w}\t$src", []>, OpSize;
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in
def DIV32r : I<0xF7, MRM6r, (outs),  (ins GR32:$src),   // EDX:EAX/r32 = EAX,EDX
               "div{l}\t$src", []>;
let mayLoad = 1 in {
let Defs = [AL,EFLAGS,AX], Uses = [AX] in
def DIV8m  : I<0xF6, MRM6m, (outs), (ins i8mem:$src),   // AX/[mem8] = AL,AH
               "div{b}\t$src", []>;
let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
def DIV16m : I<0xF7, MRM6m, (outs), (ins i16mem:$src),  // DX:AX/[mem16] = AX,DX
               "div{w}\t$src", []>, OpSize;
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in
                                                    // EDX:EAX/[mem32] = EAX,EDX
def DIV32m : I<0xF7, MRM6m, (outs), (ins i32mem:$src),
               "div{l}\t$src", []>;
}

// Signed division/remainder.
let Defs = [AL,EFLAGS,AX], Uses = [AX] in
def IDIV8r : I<0xF6, MRM7r, (outs),  (ins GR8:$src),    // AX/r8 = AL,AH
               "idiv{b}\t$src", []>;
let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
def IDIV16r: I<0xF7, MRM7r, (outs),  (ins GR16:$src),   // DX:AX/r16 = AX,DX
               "idiv{w}\t$src", []>, OpSize;
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in
def IDIV32r: I<0xF7, MRM7r, (outs),  (ins GR32:$src),   // EDX:EAX/r32 = EAX,EDX
               "idiv{l}\t$src", []>;
let mayLoad = 1, mayLoad = 1 in {
let Defs = [AL,EFLAGS,AX], Uses = [AX] in
def IDIV8m : I<0xF6, MRM7m, (outs), (ins i8mem:$src),   // AX/[mem8] = AL,AH
               "idiv{b}\t$src", []>;
let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
def IDIV16m: I<0xF7, MRM7m, (outs), (ins i16mem:$src),  // DX:AX/[mem16] = AX,DX
               "idiv{w}\t$src", []>, OpSize;
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in
def IDIV32m: I<0xF7, MRM7m, (outs), (ins i32mem:$src), 
                                                    // EDX:EAX/[mem32] = EAX,EDX
               "idiv{l}\t$src", []>;
}

//===----------------------------------------------------------------------===//
//  Two address Instructions.
//
let Constraints = "$src1 = $dst" in {

// Conditional moves
let Uses = [EFLAGS] in {

let Predicates = [HasCMov] in {
let isCommutable = 1 in {
def CMOVB16rr : I<0x42, MRMSrcReg,       // if <u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovb{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_B, EFLAGS))]>,
                  TB, OpSize;
def CMOVB32rr : I<0x42, MRMSrcReg,       // if <u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovb{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_B, EFLAGS))]>,
                   TB;
def CMOVAE16rr: I<0x43, MRMSrcReg,       // if >=u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovae{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_AE, EFLAGS))]>,
                   TB, OpSize;
def CMOVAE32rr: I<0x43, MRMSrcReg,       // if >=u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovae{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_AE, EFLAGS))]>,
                   TB;
def CMOVE16rr : I<0x44, MRMSrcReg,       // if ==, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmove{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_E, EFLAGS))]>,
                   TB, OpSize;
def CMOVE32rr : I<0x44, MRMSrcReg,       // if ==, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmove{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_E, EFLAGS))]>,
                   TB;
def CMOVNE16rr: I<0x45, MRMSrcReg,       // if !=, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovne{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_NE, EFLAGS))]>,
                   TB, OpSize;
def CMOVNE32rr: I<0x45, MRMSrcReg,       // if !=, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovne{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_NE, EFLAGS))]>,
                   TB;
def CMOVBE16rr: I<0x46, MRMSrcReg,       // if <=u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovbe{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_BE, EFLAGS))]>,
                   TB, OpSize;
def CMOVBE32rr: I<0x46, MRMSrcReg,       // if <=u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovbe{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_BE, EFLAGS))]>,
                   TB;
def CMOVA16rr : I<0x47, MRMSrcReg,       // if >u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmova{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_A, EFLAGS))]>,
                   TB, OpSize;
def CMOVA32rr : I<0x47, MRMSrcReg,       // if >u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmova{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_A, EFLAGS))]>,
                   TB;
def CMOVL16rr : I<0x4C, MRMSrcReg,       // if <s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovl{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_L, EFLAGS))]>,
                   TB, OpSize;
def CMOVL32rr : I<0x4C, MRMSrcReg,       // if <s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovl{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_L, EFLAGS))]>,
                   TB;
def CMOVGE16rr: I<0x4D, MRMSrcReg,       // if >=s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovge{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_GE, EFLAGS))]>,
                   TB, OpSize;
def CMOVGE32rr: I<0x4D, MRMSrcReg,       // if >=s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovge{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_GE, EFLAGS))]>,
                   TB;
def CMOVLE16rr: I<0x4E, MRMSrcReg,       // if <=s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovle{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_LE, EFLAGS))]>,
                   TB, OpSize;
def CMOVLE32rr: I<0x4E, MRMSrcReg,       // if <=s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovle{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_LE, EFLAGS))]>,
                   TB;
def CMOVG16rr : I<0x4F, MRMSrcReg,       // if >s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovg{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_G, EFLAGS))]>,
                   TB, OpSize;
def CMOVG32rr : I<0x4F, MRMSrcReg,       // if >s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovg{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_G, EFLAGS))]>,
                   TB;
def CMOVS16rr : I<0x48, MRMSrcReg,       // if signed, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovs{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_S, EFLAGS))]>,
                  TB, OpSize;
def CMOVS32rr : I<0x48, MRMSrcReg,       // if signed, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovs{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_S, EFLAGS))]>,
                  TB;
def CMOVNS16rr: I<0x49, MRMSrcReg,       // if !signed, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovns{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_NS, EFLAGS))]>,
                  TB, OpSize;
def CMOVNS32rr: I<0x49, MRMSrcReg,       // if !signed, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovns{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_NS, EFLAGS))]>,
                  TB;
def CMOVP16rr : I<0x4A, MRMSrcReg,       // if parity, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovp{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_P, EFLAGS))]>,
                  TB, OpSize;
def CMOVP32rr : I<0x4A, MRMSrcReg,       // if parity, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovp{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_P, EFLAGS))]>,
                  TB;
def CMOVNP16rr : I<0x4B, MRMSrcReg,       // if !parity, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovnp{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                    X86_COND_NP, EFLAGS))]>,
                  TB, OpSize;
def CMOVNP32rr : I<0x4B, MRMSrcReg,       // if !parity, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovnp{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                    X86_COND_NP, EFLAGS))]>,
                  TB;
def CMOVO16rr : I<0x40, MRMSrcReg,       // if overflow, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovo{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_O, EFLAGS))]>,
                  TB, OpSize;
def CMOVO32rr : I<0x40, MRMSrcReg,       // if overflow, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovo{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_O, EFLAGS))]>,
                  TB;
def CMOVNO16rr : I<0x41, MRMSrcReg,       // if !overflow, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovno{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                    X86_COND_NO, EFLAGS))]>,
                  TB, OpSize;
def CMOVNO32rr : I<0x41, MRMSrcReg,       // if !overflow, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovno{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                    X86_COND_NO, EFLAGS))]>,
                  TB;
} // isCommutable = 1

def CMOVB16rm : I<0x42, MRMSrcMem,       // if <u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovb{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_B, EFLAGS))]>,
                  TB, OpSize;
def CMOVB32rm : I<0x42, MRMSrcMem,       // if <u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovb{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_B, EFLAGS))]>,
                   TB;
def CMOVAE16rm: I<0x43, MRMSrcMem,       // if >=u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovae{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_AE, EFLAGS))]>,
                   TB, OpSize;
def CMOVAE32rm: I<0x43, MRMSrcMem,       // if >=u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovae{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_AE, EFLAGS))]>,
                   TB;
def CMOVE16rm : I<0x44, MRMSrcMem,       // if ==, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmove{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_E, EFLAGS))]>,
                   TB, OpSize;
def CMOVE32rm : I<0x44, MRMSrcMem,       // if ==, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmove{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_E, EFLAGS))]>,
                   TB;
def CMOVNE16rm: I<0x45, MRMSrcMem,       // if !=, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovne{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_NE, EFLAGS))]>,
                   TB, OpSize;
def CMOVNE32rm: I<0x45, MRMSrcMem,       // if !=, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovne{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NE, EFLAGS))]>,
                   TB;
def CMOVBE16rm: I<0x46, MRMSrcMem,       // if <=u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovbe{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_BE, EFLAGS))]>,
                   TB, OpSize;
def CMOVBE32rm: I<0x46, MRMSrcMem,       // if <=u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovbe{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_BE, EFLAGS))]>,
                   TB;
def CMOVA16rm : I<0x47, MRMSrcMem,       // if >u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmova{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_A, EFLAGS))]>,
                   TB, OpSize;
def CMOVA32rm : I<0x47, MRMSrcMem,       // if >u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmova{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_A, EFLAGS))]>,
                   TB;
def CMOVL16rm : I<0x4C, MRMSrcMem,       // if <s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovl{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_L, EFLAGS))]>,
                   TB, OpSize;
def CMOVL32rm : I<0x4C, MRMSrcMem,       // if <s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovl{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_L, EFLAGS))]>,
                   TB;
def CMOVGE16rm: I<0x4D, MRMSrcMem,       // if >=s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovge{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_GE, EFLAGS))]>,
                   TB, OpSize;
def CMOVGE32rm: I<0x4D, MRMSrcMem,       // if >=s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovge{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_GE, EFLAGS))]>,
                   TB;
def CMOVLE16rm: I<0x4E, MRMSrcMem,       // if <=s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovle{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_LE, EFLAGS))]>,
                   TB, OpSize;
def CMOVLE32rm: I<0x4E, MRMSrcMem,       // if <=s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovle{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_LE, EFLAGS))]>,
                   TB;
def CMOVG16rm : I<0x4F, MRMSrcMem,       // if >s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovg{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_G, EFLAGS))]>,
                   TB, OpSize;
def CMOVG32rm : I<0x4F, MRMSrcMem,       // if >s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovg{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_G, EFLAGS))]>,
                   TB;
def CMOVS16rm : I<0x48, MRMSrcMem,       // if signed, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovs{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_S, EFLAGS))]>,
                  TB, OpSize;
def CMOVS32rm : I<0x48, MRMSrcMem,       // if signed, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovs{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_S, EFLAGS))]>,
                  TB;
def CMOVNS16rm: I<0x49, MRMSrcMem,       // if !signed, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovns{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_NS, EFLAGS))]>,
                  TB, OpSize;
def CMOVNS32rm: I<0x49, MRMSrcMem,       // if !signed, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovns{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NS, EFLAGS))]>,
                  TB;
def CMOVP16rm : I<0x4A, MRMSrcMem,       // if parity, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovp{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_P, EFLAGS))]>,
                  TB, OpSize;
def CMOVP32rm : I<0x4A, MRMSrcMem,       // if parity, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovp{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_P, EFLAGS))]>,
                  TB;
def CMOVNP16rm : I<0x4B, MRMSrcMem,       // if !parity, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovnp{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                    X86_COND_NP, EFLAGS))]>,
                  TB, OpSize;
def CMOVNP32rm : I<0x4B, MRMSrcMem,       // if !parity, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovnp{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                    X86_COND_NP, EFLAGS))]>,
                  TB;
def CMOVO16rm : I<0x40, MRMSrcMem,       // if overflow, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovo{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_O, EFLAGS))]>,
                  TB, OpSize;
def CMOVO32rm : I<0x40, MRMSrcMem,       // if overflow, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovo{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_O, EFLAGS))]>,
                  TB;
def CMOVNO16rm : I<0x41, MRMSrcMem,       // if !overflow, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovno{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                    X86_COND_NO, EFLAGS))]>,
                  TB, OpSize;
def CMOVNO32rm : I<0x41, MRMSrcMem,       // if !overflow, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovno{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                    X86_COND_NO, EFLAGS))]>,
                  TB;
} // Predicates = [HasCMov]

// X86 doesn't have 8-bit conditional moves. Use a customInserter to
// emit control flow. An alternative to this is to mark i8 SELECT as Promote,
// however that requires promoting the operands, and can induce additional
// i8 register pressure. Note that CMOV_GR8 is conservatively considered to
// clobber EFLAGS, because if one of the operands is zero, the expansion
// could involve an xor.
let usesCustomInserter = 1, Constraints = "", Defs = [EFLAGS] in {
def CMOV_GR8 : I<0, Pseudo,
                 (outs GR8:$dst), (ins GR8:$src1, GR8:$src2, i8imm:$cond),
                 "#CMOV_GR8 PSEUDO!",
                 [(set GR8:$dst, (X86cmov GR8:$src1, GR8:$src2,
                                          imm:$cond, EFLAGS))]>;

let Predicates = [NoCMov] in {
def CMOV_GR32 : I<0, Pseudo,
                    (outs GR32:$dst), (ins GR32:$src1, GR32:$src2, i8imm:$cond),
                    "#CMOV_GR32* PSEUDO!",
                    [(set GR32:$dst,
                      (X86cmov GR32:$src1, GR32:$src2, imm:$cond, EFLAGS))]>;
def CMOV_GR16 : I<0, Pseudo,
                    (outs GR16:$dst), (ins GR16:$src1, GR16:$src2, i8imm:$cond),
                    "#CMOV_GR16* PSEUDO!",
                    [(set GR16:$dst,
                      (X86cmov GR16:$src1, GR16:$src2, imm:$cond, EFLAGS))]>;
def CMOV_RFP32 : I<0, Pseudo,
                    (outs RFP32:$dst),
                    (ins RFP32:$src1, RFP32:$src2, i8imm:$cond),
                    "#CMOV_RFP32 PSEUDO!",
                    [(set RFP32:$dst,
                      (X86cmov RFP32:$src1, RFP32:$src2, imm:$cond,
                                                  EFLAGS))]>;
def CMOV_RFP64 : I<0, Pseudo,
                    (outs RFP64:$dst),
                    (ins RFP64:$src1, RFP64:$src2, i8imm:$cond),
                    "#CMOV_RFP64 PSEUDO!",
                    [(set RFP64:$dst,
                      (X86cmov RFP64:$src1, RFP64:$src2, imm:$cond,
                                                  EFLAGS))]>;
def CMOV_RFP80 : I<0, Pseudo,
                    (outs RFP80:$dst),
                    (ins RFP80:$src1, RFP80:$src2, i8imm:$cond),
                    "#CMOV_RFP80 PSEUDO!",
                    [(set RFP80:$dst,
                      (X86cmov RFP80:$src1, RFP80:$src2, imm:$cond,
                                                  EFLAGS))]>;
} // Predicates = [NoCMov]
} // UsesCustomInserter = 1, Constraints = "", Defs = [EFLAGS] 
} // Uses = [EFLAGS]


// unary instructions
let CodeSize = 2 in {
let Defs = [EFLAGS] in {
def NEG8r  : I<0xF6, MRM3r, (outs GR8 :$dst), (ins GR8 :$src1),
               "neg{b}\t$dst",
               [(set GR8:$dst, (ineg GR8:$src1)),
                (implicit EFLAGS)]>;
def NEG16r : I<0xF7, MRM3r, (outs GR16:$dst), (ins GR16:$src1),
               "neg{w}\t$dst",
               [(set GR16:$dst, (ineg GR16:$src1)),
                (implicit EFLAGS)]>, OpSize;
def NEG32r : I<0xF7, MRM3r, (outs GR32:$dst), (ins GR32:$src1),
               "neg{l}\t$dst",
               [(set GR32:$dst, (ineg GR32:$src1)),
                (implicit EFLAGS)]>;
                
let Constraints = "" in {
  def NEG8m  : I<0xF6, MRM3m, (outs), (ins i8mem :$dst),
                 "neg{b}\t$dst",
                 [(store (ineg (loadi8 addr:$dst)), addr:$dst),
                  (implicit EFLAGS)]>;
  def NEG16m : I<0xF7, MRM3m, (outs), (ins i16mem:$dst),
                 "neg{w}\t$dst",
                 [(store (ineg (loadi16 addr:$dst)), addr:$dst),
                  (implicit EFLAGS)]>, OpSize;
  def NEG32m : I<0xF7, MRM3m, (outs), (ins i32mem:$dst),
                 "neg{l}\t$dst",
                 [(store (ineg (loadi32 addr:$dst)), addr:$dst),
                  (implicit EFLAGS)]>;
} // Constraints = ""
} // Defs = [EFLAGS]

// Match xor -1 to not. Favors these over a move imm + xor to save code size.
let AddedComplexity = 15 in {
def NOT8r  : I<0xF6, MRM2r, (outs GR8 :$dst), (ins GR8 :$src1),
               "not{b}\t$dst",
               [(set GR8:$dst, (not GR8:$src1))]>;
def NOT16r : I<0xF7, MRM2r, (outs GR16:$dst), (ins GR16:$src1),
               "not{w}\t$dst",
               [(set GR16:$dst, (not GR16:$src1))]>, OpSize;
def NOT32r : I<0xF7, MRM2r, (outs GR32:$dst), (ins GR32:$src1),
               "not{l}\t$dst",
               [(set GR32:$dst, (not GR32:$src1))]>;
}
let Constraints = "" in {
  def NOT8m  : I<0xF6, MRM2m, (outs), (ins i8mem :$dst),
                 "not{b}\t$dst",
                 [(store (not (loadi8 addr:$dst)), addr:$dst)]>;
  def NOT16m : I<0xF7, MRM2m, (outs), (ins i16mem:$dst),
                 "not{w}\t$dst",
                 [(store (not (loadi16 addr:$dst)), addr:$dst)]>, OpSize;
  def NOT32m : I<0xF7, MRM2m, (outs), (ins i32mem:$dst),
                 "not{l}\t$dst",
                 [(store (not (loadi32 addr:$dst)), addr:$dst)]>;
} // Constraints = ""
} // CodeSize

// TODO: inc/dec is slow for P4, but fast for Pentium-M.
let Defs = [EFLAGS] in {
let CodeSize = 2 in
def INC8r  : I<0xFE, MRM0r, (outs GR8 :$dst), (ins GR8 :$src1),
               "inc{b}\t$dst",
               [(set GR8:$dst, EFLAGS, (X86inc_flag GR8:$src1))]>;

let isConvertibleToThreeAddress = 1, CodeSize = 1 in {  // Can xform into LEA.
def INC16r : I<0x40, AddRegFrm, (outs GR16:$dst), (ins GR16:$src1), 
               "inc{w}\t$dst",
               [(set GR16:$dst, EFLAGS, (X86inc_flag GR16:$src1))]>,
             OpSize, Requires<[In32BitMode]>;
def INC32r : I<0x40, AddRegFrm, (outs GR32:$dst), (ins GR32:$src1), 
               "inc{l}\t$dst",
               [(set GR32:$dst, EFLAGS, (X86inc_flag GR32:$src1))]>,
             Requires<[In32BitMode]>;
}
let Constraints = "", CodeSize = 2 in {
  def INC8m  : I<0xFE, MRM0m, (outs), (ins i8mem :$dst), "inc{b}\t$dst",
               [(store (add (loadi8 addr:$dst), 1), addr:$dst),
                (implicit EFLAGS)]>;
  def INC16m : I<0xFF, MRM0m, (outs), (ins i16mem:$dst), "inc{w}\t$dst",
               [(store (add (loadi16 addr:$dst), 1), addr:$dst),
                (implicit EFLAGS)]>,
               OpSize, Requires<[In32BitMode]>;
  def INC32m : I<0xFF, MRM0m, (outs), (ins i32mem:$dst), "inc{l}\t$dst",
               [(store (add (loadi32 addr:$dst), 1), addr:$dst),
                (implicit EFLAGS)]>,
               Requires<[In32BitMode]>;
} // Constraints = "", CodeSize = 2

let CodeSize = 2 in
def DEC8r  : I<0xFE, MRM1r, (outs GR8 :$dst), (ins GR8 :$src1),
               "dec{b}\t$dst",
               [(set GR8:$dst, EFLAGS, (X86dec_flag GR8:$src1))]>;
let isConvertibleToThreeAddress = 1, CodeSize = 1 in {   // Can xform into LEA.
def DEC16r : I<0x48, AddRegFrm, (outs GR16:$dst), (ins GR16:$src1), 
               "dec{w}\t$dst",
               [(set GR16:$dst, EFLAGS, (X86dec_flag GR16:$src1))]>,
             OpSize, Requires<[In32BitMode]>;
def DEC32r : I<0x48, AddRegFrm, (outs GR32:$dst), (ins GR32:$src1), 
               "dec{l}\t$dst",
               [(set GR32:$dst, EFLAGS, (X86dec_flag GR32:$src1))]>,
             Requires<[In32BitMode]>;
} // CodeSize = 2

let Constraints = "", CodeSize = 2 in {
  def DEC8m  : I<0xFE, MRM1m, (outs), (ins i8mem :$dst), "dec{b}\t$dst",
               [(store (add (loadi8 addr:$dst), -1), addr:$dst),
                (implicit EFLAGS)]>;
  def DEC16m : I<0xFF, MRM1m, (outs), (ins i16mem:$dst), "dec{w}\t$dst",
               [(store (add (loadi16 addr:$dst), -1), addr:$dst),
                (implicit EFLAGS)]>,
               OpSize, Requires<[In32BitMode]>;
  def DEC32m : I<0xFF, MRM1m, (outs), (ins i32mem:$dst), "dec{l}\t$dst",
               [(store (add (loadi32 addr:$dst), -1), addr:$dst),
                (implicit EFLAGS)]>,
               Requires<[In32BitMode]>;
} // Constraints = "", CodeSize = 2
} // Defs = [EFLAGS]

// Logical operators...
let Defs = [EFLAGS] in {
let isCommutable = 1 in {   // X = AND Y, Z   --> X = AND Z, Y
def AND8rr  : I<0x20, MRMDestReg,
               (outs GR8 :$dst), (ins GR8 :$src1, GR8 :$src2),
               "and{b}\t{$src2, $dst|$dst, $src2}",
               [(set GR8:$dst, EFLAGS, (X86and_flag GR8:$src1, GR8:$src2))]>;
def AND16rr : I<0x21, MRMDestReg,
                (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                "and{w}\t{$src2, $dst|$dst, $src2}",
                [(set GR16:$dst, EFLAGS, (X86and_flag GR16:$src1,
                                                      GR16:$src2))]>, OpSize;
def AND32rr : I<0x21, MRMDestReg, 
                (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                "and{l}\t{$src2, $dst|$dst, $src2}",
                [(set GR32:$dst, EFLAGS, (X86and_flag GR32:$src1,
                                                      GR32:$src2))]>;
}

// AND instructions with the destination register in REG and the source register
//   in R/M.  Included for the disassembler.
let isCodeGenOnly = 1 in {
def AND8rr_REV : I<0x22, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                  "and{b}\t{$src2, $dst|$dst, $src2}", []>;
def AND16rr_REV : I<0x23, MRMSrcReg, (outs GR16:$dst), 
                    (ins GR16:$src1, GR16:$src2),
                   "and{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize;
def AND32rr_REV : I<0x23, MRMSrcReg, (outs GR32:$dst), 
                    (ins GR32:$src1, GR32:$src2),
                   "and{l}\t{$src2, $dst|$dst, $src2}", []>;
}

def AND8rm   : I<0x22, MRMSrcMem, 
                 (outs GR8 :$dst), (ins GR8 :$src1, i8mem :$src2),
                 "and{b}\t{$src2, $dst|$dst, $src2}",
                [(set GR8:$dst, EFLAGS, (X86and_flag GR8:$src1,
                                                     (loadi8 addr:$src2)))]>;
def AND16rm  : I<0x23, MRMSrcMem, 
                 (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                 "and{w}\t{$src2, $dst|$dst, $src2}",
                [(set GR16:$dst, EFLAGS, (X86and_flag GR16:$src1,
                                                      (loadi16 addr:$src2)))]>,
               OpSize;
def AND32rm  : I<0x23, MRMSrcMem,
                 (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                 "and{l}\t{$src2, $dst|$dst, $src2}",
                [(set GR32:$dst, EFLAGS, (X86and_flag GR32:$src1,
                                                      (loadi32 addr:$src2)))]>;

def AND8ri   : Ii8<0x80, MRM4r, 
                   (outs GR8 :$dst), (ins GR8 :$src1, i8imm :$src2),
                   "and{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, EFLAGS, (X86and_flag GR8:$src1,
                                                        imm:$src2))]>;
def AND16ri  : Ii16<0x81, MRM4r, 
                    (outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                    "and{w}\t{$src2, $dst|$dst, $src2}",
                    [(set GR16:$dst, EFLAGS, (X86and_flag GR16:$src1,
                                                          imm:$src2))]>, OpSize;
def AND32ri  : Ii32<0x81, MRM4r, 
                    (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                    "and{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, EFLAGS, (X86and_flag GR32:$src1,
                                                          imm:$src2))]>;
def AND16ri8 : Ii8<0x83, MRM4r, 
                   (outs GR16:$dst), (ins GR16:$src1, i16i8imm:$src2),
                   "and{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, EFLAGS, (X86and_flag GR16:$src1,
                                                         i16immSExt8:$src2))]>,
                   OpSize;
def AND32ri8 : Ii8<0x83, MRM4r, 
                   (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                   "and{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, EFLAGS, (X86and_flag GR32:$src1,
                                                         i32immSExt8:$src2))]>;

let Constraints = "" in {
  def AND8mr   : I<0x20, MRMDestMem,
                   (outs), (ins i8mem :$dst, GR8 :$src),
                   "and{b}\t{$src, $dst|$dst, $src}",
                   [(store (and (load addr:$dst), GR8:$src), addr:$dst),
                    (implicit EFLAGS)]>;
  def AND16mr  : I<0x21, MRMDestMem,
                   (outs), (ins i16mem:$dst, GR16:$src),
                   "and{w}\t{$src, $dst|$dst, $src}",
                   [(store (and (load addr:$dst), GR16:$src), addr:$dst),
                    (implicit EFLAGS)]>,
                   OpSize;
  def AND32mr  : I<0x21, MRMDestMem,
                   (outs), (ins i32mem:$dst, GR32:$src),
                   "and{l}\t{$src, $dst|$dst, $src}",
                   [(store (and (load addr:$dst), GR32:$src), addr:$dst),
                    (implicit EFLAGS)]>;
  def AND8mi   : Ii8<0x80, MRM4m,
                     (outs), (ins i8mem :$dst, i8imm :$src),
                     "and{b}\t{$src, $dst|$dst, $src}",
                      [(store (and (loadi8 addr:$dst), imm:$src), addr:$dst),
                       (implicit EFLAGS)]>;
  def AND16mi  : Ii16<0x81, MRM4m,
                      (outs), (ins i16mem:$dst, i16imm:$src),
                      "and{w}\t{$src, $dst|$dst, $src}",
                      [(store (and (loadi16 addr:$dst), imm:$src), addr:$dst),
                       (implicit EFLAGS)]>,
                      OpSize;
  def AND32mi  : Ii32<0x81, MRM4m,
                      (outs), (ins i32mem:$dst, i32imm:$src),
                      "and{l}\t{$src, $dst|$dst, $src}",
                      [(store (and (loadi32 addr:$dst), imm:$src), addr:$dst),
                       (implicit EFLAGS)]>;
  def AND16mi8 : Ii8<0x83, MRM4m,
                     (outs), (ins i16mem:$dst, i16i8imm :$src),
                     "and{w}\t{$src, $dst|$dst, $src}",
                [(store (and (load addr:$dst), i16immSExt8:$src), addr:$dst),
                 (implicit EFLAGS)]>,
                     OpSize;
  def AND32mi8 : Ii8<0x83, MRM4m,
                     (outs), (ins i32mem:$dst, i32i8imm :$src),
                     "and{l}\t{$src, $dst|$dst, $src}",
                [(store (and (load addr:$dst), i32immSExt8:$src), addr:$dst),
                 (implicit EFLAGS)]>;

  def AND8i8 : Ii8<0x24, RawFrm, (outs), (ins i8imm:$src),
                   "and{b}\t{$src, %al|%al, $src}", []>;
  def AND16i16 : Ii16<0x25, RawFrm, (outs), (ins i16imm:$src),
                      "and{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
  def AND32i32 : Ii32<0x25, RawFrm, (outs), (ins i32imm:$src),
                      "and{l}\t{$src, %eax|%eax, $src}", []>;

} // Constraints = ""


let isCommutable = 1 in {   // X = OR Y, Z   --> X = OR Z, Y
def OR8rr    : I<0x08, MRMDestReg, (outs GR8 :$dst), 
                 (ins GR8 :$src1, GR8 :$src2),
                 "or{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, EFLAGS, (X86or_flag GR8:$src1, GR8:$src2))]>;
def OR16rr   : I<0x09, MRMDestReg, (outs GR16:$dst), 
                 (ins GR16:$src1, GR16:$src2),
                 "or{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, EFLAGS, (X86or_flag GR16:$src1,GR16:$src2))]>,
               OpSize;
def OR32rr   : I<0x09, MRMDestReg, (outs GR32:$dst), 
                 (ins GR32:$src1, GR32:$src2),
                 "or{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, EFLAGS, (X86or_flag GR32:$src1,GR32:$src2))]>;
}

// OR instructions with the destination register in REG and the source register
//   in R/M.  Included for the disassembler.
let isCodeGenOnly = 1 in {
def OR8rr_REV : I<0x0A, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                  "or{b}\t{$src2, $dst|$dst, $src2}", []>;
def OR16rr_REV : I<0x0B, MRMSrcReg, (outs GR16:$dst),
                   (ins GR16:$src1, GR16:$src2),
                   "or{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize;
def OR32rr_REV : I<0x0B, MRMSrcReg, (outs GR32:$dst), 
                   (ins GR32:$src1, GR32:$src2),
                   "or{l}\t{$src2, $dst|$dst, $src2}", []>;
}
                  
def OR8rm    : I<0x0A, MRMSrcMem, (outs GR8 :$dst), 
                 (ins GR8 :$src1, i8mem :$src2),
                 "or{b}\t{$src2, $dst|$dst, $src2}",
                [(set GR8:$dst, EFLAGS, (X86or_flag GR8:$src1,
                                                    (load addr:$src2)))]>;
def OR16rm   : I<0x0B, MRMSrcMem, (outs GR16:$dst), 
                 (ins GR16:$src1, i16mem:$src2),
                 "or{w}\t{$src2, $dst|$dst, $src2}",
                [(set GR16:$dst, EFLAGS, (X86or_flag GR16:$src1,
                                                     (load addr:$src2)))]>,
               OpSize;
def OR32rm   : I<0x0B, MRMSrcMem, (outs GR32:$dst), 
                 (ins GR32:$src1, i32mem:$src2),
                 "or{l}\t{$src2, $dst|$dst, $src2}",
                [(set GR32:$dst, EFLAGS, (X86or_flag GR32:$src1,
                                                     (load addr:$src2)))]>;

def OR8ri    : Ii8 <0x80, MRM1r, (outs GR8 :$dst), 
                    (ins GR8 :$src1, i8imm:$src2),
                    "or{b}\t{$src2, $dst|$dst, $src2}",
                    [(set GR8:$dst,EFLAGS, (X86or_flag GR8:$src1, imm:$src2))]>;
def OR16ri   : Ii16<0x81, MRM1r, (outs GR16:$dst), 
                    (ins GR16:$src1, i16imm:$src2),
                    "or{w}\t{$src2, $dst|$dst, $src2}", 
                    [(set GR16:$dst, EFLAGS, (X86or_flag GR16:$src1,
                                                        imm:$src2))]>, OpSize;
def OR32ri   : Ii32<0x81, MRM1r, (outs GR32:$dst), 
                    (ins GR32:$src1, i32imm:$src2),
                    "or{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, EFLAGS, (X86or_flag GR32:$src1,
                                                         imm:$src2))]>;

def OR16ri8  : Ii8<0x83, MRM1r, (outs GR16:$dst), 
                   (ins GR16:$src1, i16i8imm:$src2),
                   "or{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, EFLAGS, (X86or_flag GR16:$src1,
                                                i16immSExt8:$src2))]>, OpSize;
def OR32ri8  : Ii8<0x83, MRM1r, (outs GR32:$dst), 
                   (ins GR32:$src1, i32i8imm:$src2),
                   "or{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, EFLAGS, (X86or_flag GR32:$src1,
                                                        i32immSExt8:$src2))]>;
let Constraints = "" in {
  def OR8mr  : I<0x08, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src),
                 "or{b}\t{$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), GR8:$src), addr:$dst),
                  (implicit EFLAGS)]>;
  def OR16mr : I<0x09, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src),
                 "or{w}\t{$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), GR16:$src), addr:$dst),
                  (implicit EFLAGS)]>, OpSize;
  def OR32mr : I<0x09, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                 "or{l}\t{$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), GR32:$src), addr:$dst),
                  (implicit EFLAGS)]>;
  def OR8mi    : Ii8<0x80, MRM1m, (outs), (ins i8mem :$dst, i8imm:$src),
                 "or{b}\t{$src, $dst|$dst, $src}",
                 [(store (or (loadi8 addr:$dst), imm:$src), addr:$dst),
                  (implicit EFLAGS)]>;
  def OR16mi   : Ii16<0x81, MRM1m, (outs), (ins i16mem:$dst, i16imm:$src),
                 "or{w}\t{$src, $dst|$dst, $src}",
                 [(store (or (loadi16 addr:$dst), imm:$src), addr:$dst),
                  (implicit EFLAGS)]>,
                 OpSize;
  def OR32mi   : Ii32<0x81, MRM1m, (outs), (ins i32mem:$dst, i32imm:$src),
                 "or{l}\t{$src, $dst|$dst, $src}",
                 [(store (or (loadi32 addr:$dst), imm:$src), addr:$dst),
                  (implicit EFLAGS)]>;
  def OR16mi8  : Ii8<0x83, MRM1m, (outs), (ins i16mem:$dst, i16i8imm:$src),
                 "or{w}\t{$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), i16immSExt8:$src), addr:$dst),
                  (implicit EFLAGS)]>,
                     OpSize;
  def OR32mi8  : Ii8<0x83, MRM1m, (outs), (ins i32mem:$dst, i32i8imm:$src),
                 "or{l}\t{$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), i32immSExt8:$src), addr:$dst),
                  (implicit EFLAGS)]>;
                  
  def OR8i8 : Ii8 <0x0C, RawFrm, (outs), (ins i8imm:$src),
                   "or{b}\t{$src, %al|%al, $src}", []>;
  def OR16i16 : Ii16 <0x0D, RawFrm, (outs), (ins i16imm:$src),
                      "or{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
  def OR32i32 : Ii32 <0x0D, RawFrm, (outs), (ins i32imm:$src),
                      "or{l}\t{$src, %eax|%eax, $src}", []>;
} // Constraints = ""


let isCommutable = 1 in { // X = XOR Y, Z --> X = XOR Z, Y
  def XOR8rr   : I<0x30, MRMDestReg,
                   (outs GR8 :$dst), (ins GR8 :$src1, GR8 :$src2),
                   "xor{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, EFLAGS, (X86xor_flag GR8:$src1,
                                                        GR8:$src2))]>;
  def XOR16rr  : I<0x31, MRMDestReg, 
                   (outs GR16:$dst), (ins GR16:$src1, GR16:$src2), 
                   "xor{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, EFLAGS, (X86xor_flag GR16:$src1,
                                                         GR16:$src2))]>, OpSize;
  def XOR32rr  : I<0x31, MRMDestReg, 
                   (outs GR32:$dst), (ins GR32:$src1, GR32:$src2), 
                   "xor{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, EFLAGS, (X86xor_flag GR32:$src1,
                                                         GR32:$src2))]>;
} // isCommutable = 1

// XOR instructions with the destination register in REG and the source register
//   in R/M.  Included for the disassembler.
let isCodeGenOnly = 1 in {
def XOR8rr_REV : I<0x32, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                  "xor{b}\t{$src2, $dst|$dst, $src2}", []>;
def XOR16rr_REV : I<0x33, MRMSrcReg, (outs GR16:$dst), 
                    (ins GR16:$src1, GR16:$src2),
                   "xor{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize;
def XOR32rr_REV : I<0x33, MRMSrcReg, (outs GR32:$dst), 
                    (ins GR32:$src1, GR32:$src2),
                   "xor{l}\t{$src2, $dst|$dst, $src2}", []>;
}

def XOR8rm   : I<0x32, MRMSrcMem, 
                 (outs GR8 :$dst), (ins GR8:$src1, i8mem :$src2), 
                 "xor{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, EFLAGS, (X86xor_flag GR8:$src1,
                                                      (load addr:$src2)))]>;
def XOR16rm  : I<0x33, MRMSrcMem, 
                 (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2), 
                 "xor{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, EFLAGS, (X86xor_flag GR16:$src1,
                                                       (load addr:$src2)))]>,
                 OpSize;
def XOR32rm  : I<0x33, MRMSrcMem, 
                 (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2), 
                 "xor{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, EFLAGS, (X86xor_flag GR32:$src1,
                                                       (load addr:$src2)))]>;

def XOR8ri  : Ii8<0x80, MRM6r, 
                  (outs GR8:$dst), (ins GR8:$src1, i8imm:$src2), 
                  "xor{b}\t{$src2, $dst|$dst, $src2}",
                  [(set GR8:$dst, EFLAGS, (X86xor_flag GR8:$src1, imm:$src2))]>;
def XOR16ri : Ii16<0x81, MRM6r, 
                   (outs GR16:$dst), (ins GR16:$src1, i16imm:$src2), 
                   "xor{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, EFLAGS, (X86xor_flag GR16:$src1,
                                                         imm:$src2))]>, OpSize;
def XOR32ri  : Ii32<0x81, MRM6r, 
                    (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2), 
                    "xor{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, EFLAGS, (X86xor_flag GR32:$src1,
                                                          imm:$src2))]>;
def XOR16ri8 : Ii8<0x83, MRM6r, 
                   (outs GR16:$dst), (ins GR16:$src1, i16i8imm:$src2),
                   "xor{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, EFLAGS, (X86xor_flag GR16:$src1,
                                                         i16immSExt8:$src2))]>,
                   OpSize;
def XOR32ri8 : Ii8<0x83, MRM6r, 
                   (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                   "xor{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, EFLAGS, (X86xor_flag GR32:$src1,
                                                         i32immSExt8:$src2))]>;

let Constraints = "" in {
  def XOR8mr   : I<0x30, MRMDestMem,
                   (outs), (ins i8mem :$dst, GR8 :$src),
                   "xor{b}\t{$src, $dst|$dst, $src}",
                   [(store (xor (load addr:$dst), GR8:$src), addr:$dst),
                    (implicit EFLAGS)]>;
  def XOR16mr  : I<0x31, MRMDestMem,
                   (outs), (ins i16mem:$dst, GR16:$src),
                   "xor{w}\t{$src, $dst|$dst, $src}",
                   [(store (xor (load addr:$dst), GR16:$src), addr:$dst),
                    (implicit EFLAGS)]>,
                   OpSize;
  def XOR32mr  : I<0x31, MRMDestMem,
                   (outs), (ins i32mem:$dst, GR32:$src),
                   "xor{l}\t{$src, $dst|$dst, $src}",
                   [(store (xor (load addr:$dst), GR32:$src), addr:$dst),
                    (implicit EFLAGS)]>;
  def XOR8mi   : Ii8<0x80, MRM6m,
                     (outs), (ins i8mem :$dst, i8imm :$src),
                     "xor{b}\t{$src, $dst|$dst, $src}",
                    [(store (xor (loadi8 addr:$dst), imm:$src), addr:$dst),
                     (implicit EFLAGS)]>;
  def XOR16mi  : Ii16<0x81, MRM6m,
                      (outs), (ins i16mem:$dst, i16imm:$src),
                      "xor{w}\t{$src, $dst|$dst, $src}",
                   [(store (xor (loadi16 addr:$dst), imm:$src), addr:$dst),
                    (implicit EFLAGS)]>,
                      OpSize;
  def XOR32mi  : Ii32<0x81, MRM6m,
                      (outs), (ins i32mem:$dst, i32imm:$src),
                      "xor{l}\t{$src, $dst|$dst, $src}",
                   [(store (xor (loadi32 addr:$dst), imm:$src), addr:$dst),
                    (implicit EFLAGS)]>;
  def XOR16mi8 : Ii8<0x83, MRM6m,
                     (outs), (ins i16mem:$dst, i16i8imm :$src),
                     "xor{w}\t{$src, $dst|$dst, $src}",
                 [(store (xor (load addr:$dst), i16immSExt8:$src), addr:$dst),
                  (implicit EFLAGS)]>,
                     OpSize;
  def XOR32mi8 : Ii8<0x83, MRM6m,
                     (outs), (ins i32mem:$dst, i32i8imm :$src),
                     "xor{l}\t{$src, $dst|$dst, $src}",
                 [(store (xor (load addr:$dst), i32immSExt8:$src), addr:$dst),
                  (implicit EFLAGS)]>;
                  
  def XOR8i8   : Ii8 <0x34, RawFrm, (outs), (ins i8imm:$src),
                      "xor{b}\t{$src, %al|%al, $src}", []>;
  def XOR16i16 : Ii16<0x35, RawFrm, (outs), (ins i16imm:$src),
                      "xor{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
  def XOR32i32 : Ii32<0x35, RawFrm, (outs), (ins i32imm:$src),
                      "xor{l}\t{$src, %eax|%eax, $src}", []>;
} // Constraints = ""
} // Defs = [EFLAGS]

// Shift instructions
let Defs = [EFLAGS] in {
let Uses = [CL] in {
def SHL8rCL  : I<0xD2, MRM4r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "shl{b}\t{%cl, $dst|$dst, CL}",
                 [(set GR8:$dst, (shl GR8:$src1, CL))]>;
def SHL16rCL : I<0xD3, MRM4r, (outs GR16:$dst), (ins GR16:$src1),
                 "shl{w}\t{%cl, $dst|$dst, CL}",
                 [(set GR16:$dst, (shl GR16:$src1, CL))]>, OpSize;
def SHL32rCL : I<0xD3, MRM4r, (outs GR32:$dst), (ins GR32:$src1),
                 "shl{l}\t{%cl, $dst|$dst, CL}",
                 [(set GR32:$dst, (shl GR32:$src1, CL))]>;
} // Uses = [CL]

def SHL8ri   : Ii8<0xC0, MRM4r, (outs GR8 :$dst), (ins GR8 :$src1, i8imm:$src2),
                   "shl{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (shl GR8:$src1, (i8 imm:$src2)))]>;
                   
let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
def SHL16ri  : Ii8<0xC1, MRM4r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$src2),
                   "shl{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (shl GR16:$src1, (i8 imm:$src2)))]>, OpSize;
def SHL32ri  : Ii8<0xC1, MRM4r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$src2),
                   "shl{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (shl GR32:$src1, (i8 imm:$src2)))]>;

// NOTE: We don't include patterns for shifts of a register by one, because
// 'add reg,reg' is cheaper.

def SHL8r1   : I<0xD0, MRM4r, (outs GR8:$dst), (ins GR8:$src1),
                 "shl{b}\t$dst", []>;
def SHL16r1  : I<0xD1, MRM4r, (outs GR16:$dst), (ins GR16:$src1),
                 "shl{w}\t$dst", []>, OpSize;
def SHL32r1  : I<0xD1, MRM4r, (outs GR32:$dst), (ins GR32:$src1),
                 "shl{l}\t$dst", []>;

} // isConvertibleToThreeAddress = 1

let Constraints = "" in {
  let Uses = [CL] in {
  def SHL8mCL  : I<0xD2, MRM4m, (outs), (ins i8mem :$dst),
                   "shl{b}\t{%cl, $dst|$dst, CL}",
                   [(store (shl (loadi8 addr:$dst), CL), addr:$dst)]>;
  def SHL16mCL : I<0xD3, MRM4m, (outs), (ins i16mem:$dst),
                   "shl{w}\t{%cl, $dst|$dst, CL}",
                   [(store (shl (loadi16 addr:$dst), CL), addr:$dst)]>, OpSize;
  def SHL32mCL : I<0xD3, MRM4m, (outs), (ins i32mem:$dst),
                   "shl{l}\t{%cl, $dst|$dst, CL}",
                   [(store (shl (loadi32 addr:$dst), CL), addr:$dst)]>;
  }
  def SHL8mi   : Ii8<0xC0, MRM4m, (outs), (ins i8mem :$dst, i8imm:$src),
                     "shl{b}\t{$src, $dst|$dst, $src}",
                  [(store (shl (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def SHL16mi  : Ii8<0xC1, MRM4m, (outs), (ins i16mem:$dst, i8imm:$src),
                     "shl{w}\t{$src, $dst|$dst, $src}",
                 [(store (shl (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def SHL32mi  : Ii8<0xC1, MRM4m, (outs), (ins i32mem:$dst, i8imm:$src),
                     "shl{l}\t{$src, $dst|$dst, $src}",
                 [(store (shl (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;

  // Shift by 1
  def SHL8m1   : I<0xD0, MRM4m, (outs), (ins i8mem :$dst),
                   "shl{b}\t$dst",
                  [(store (shl (loadi8 addr:$dst), (i8 1)), addr:$dst)]>;
  def SHL16m1  : I<0xD1, MRM4m, (outs), (ins i16mem:$dst),
                   "shl{w}\t$dst",
                 [(store (shl (loadi16 addr:$dst), (i8 1)), addr:$dst)]>,
                     OpSize;
  def SHL32m1  : I<0xD1, MRM4m, (outs), (ins i32mem:$dst),
                   "shl{l}\t$dst",
                 [(store (shl (loadi32 addr:$dst), (i8 1)), addr:$dst)]>;
} // Constraints = ""

let Uses = [CL] in {
def SHR8rCL  : I<0xD2, MRM5r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "shr{b}\t{%cl, $dst|$dst, CL}",
                 [(set GR8:$dst, (srl GR8:$src1, CL))]>;
def SHR16rCL : I<0xD3, MRM5r, (outs GR16:$dst), (ins GR16:$src1),
                 "shr{w}\t{%cl, $dst|$dst, CL}",
                 [(set GR16:$dst, (srl GR16:$src1, CL))]>, OpSize;
def SHR32rCL : I<0xD3, MRM5r, (outs GR32:$dst), (ins GR32:$src1),
                 "shr{l}\t{%cl, $dst|$dst, CL}",
                 [(set GR32:$dst, (srl GR32:$src1, CL))]>;
}

def SHR8ri   : Ii8<0xC0, MRM5r, (outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                   "shr{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (srl GR8:$src1, (i8 imm:$src2)))]>;
def SHR16ri  : Ii8<0xC1, MRM5r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$src2),
                   "shr{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (srl GR16:$src1, (i8 imm:$src2)))]>, OpSize;
def SHR32ri  : Ii8<0xC1, MRM5r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$src2),
                   "shr{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (srl GR32:$src1, (i8 imm:$src2)))]>;

// Shift by 1
def SHR8r1   : I<0xD0, MRM5r, (outs GR8:$dst), (ins GR8:$src1),
                 "shr{b}\t$dst",
                 [(set GR8:$dst, (srl GR8:$src1, (i8 1)))]>;
def SHR16r1  : I<0xD1, MRM5r, (outs GR16:$dst), (ins GR16:$src1),
                 "shr{w}\t$dst",
                 [(set GR16:$dst, (srl GR16:$src1, (i8 1)))]>, OpSize;
def SHR32r1  : I<0xD1, MRM5r, (outs GR32:$dst), (ins GR32:$src1),
                 "shr{l}\t$dst",
                 [(set GR32:$dst, (srl GR32:$src1, (i8 1)))]>;

let Constraints = "" in {
  let Uses = [CL] in {
  def SHR8mCL  : I<0xD2, MRM5m, (outs), (ins i8mem :$dst),
                   "shr{b}\t{%cl, $dst|$dst, CL}",
                   [(store (srl (loadi8 addr:$dst), CL), addr:$dst)]>;
  def SHR16mCL : I<0xD3, MRM5m, (outs), (ins i16mem:$dst),
                   "shr{w}\t{%cl, $dst|$dst, CL}",
                   [(store (srl (loadi16 addr:$dst), CL), addr:$dst)]>,
                   OpSize;
  def SHR32mCL : I<0xD3, MRM5m, (outs), (ins i32mem:$dst),
                   "shr{l}\t{%cl, $dst|$dst, CL}",
                   [(store (srl (loadi32 addr:$dst), CL), addr:$dst)]>;
  }
  def SHR8mi   : Ii8<0xC0, MRM5m, (outs), (ins i8mem :$dst, i8imm:$src),
                     "shr{b}\t{$src, $dst|$dst, $src}",
                  [(store (srl (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def SHR16mi  : Ii8<0xC1, MRM5m, (outs), (ins i16mem:$dst, i8imm:$src),
                     "shr{w}\t{$src, $dst|$dst, $src}",
                 [(store (srl (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def SHR32mi  : Ii8<0xC1, MRM5m, (outs), (ins i32mem:$dst, i8imm:$src),
                     "shr{l}\t{$src, $dst|$dst, $src}",
                 [(store (srl (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;

  // Shift by 1
  def SHR8m1   : I<0xD0, MRM5m, (outs), (ins i8mem :$dst),
                   "shr{b}\t$dst",
                  [(store (srl (loadi8 addr:$dst), (i8 1)), addr:$dst)]>;
  def SHR16m1  : I<0xD1, MRM5m, (outs), (ins i16mem:$dst),
                   "shr{w}\t$dst",
                 [(store (srl (loadi16 addr:$dst), (i8 1)), addr:$dst)]>,OpSize;
  def SHR32m1  : I<0xD1, MRM5m, (outs), (ins i32mem:$dst),
                   "shr{l}\t$dst",
                 [(store (srl (loadi32 addr:$dst), (i8 1)), addr:$dst)]>;
} // Constraints = ""

let Uses = [CL] in {
def SAR8rCL  : I<0xD2, MRM7r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "sar{b}\t{%cl, $dst|$dst, CL}",
                 [(set GR8:$dst, (sra GR8:$src1, CL))]>;
def SAR16rCL : I<0xD3, MRM7r, (outs GR16:$dst), (ins GR16:$src1),
                 "sar{w}\t{%cl, $dst|$dst, CL}",
                 [(set GR16:$dst, (sra GR16:$src1, CL))]>, OpSize;
def SAR32rCL : I<0xD3, MRM7r, (outs GR32:$dst), (ins GR32:$src1),
                 "sar{l}\t{%cl, $dst|$dst, CL}",
                 [(set GR32:$dst, (sra GR32:$src1, CL))]>;
}

def SAR8ri   : Ii8<0xC0, MRM7r, (outs GR8 :$dst), (ins GR8 :$src1, i8imm:$src2),
                   "sar{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (sra GR8:$src1, (i8 imm:$src2)))]>;
def SAR16ri  : Ii8<0xC1, MRM7r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$src2),
                   "sar{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (sra GR16:$src1, (i8 imm:$src2)))]>,
                   OpSize;
def SAR32ri  : Ii8<0xC1, MRM7r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$src2),
                   "sar{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (sra GR32:$src1, (i8 imm:$src2)))]>;

// Shift by 1
def SAR8r1   : I<0xD0, MRM7r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "sar{b}\t$dst",
                 [(set GR8:$dst, (sra GR8:$src1, (i8 1)))]>;
def SAR16r1  : I<0xD1, MRM7r, (outs GR16:$dst), (ins GR16:$src1),
                 "sar{w}\t$dst",
                 [(set GR16:$dst, (sra GR16:$src1, (i8 1)))]>, OpSize;
def SAR32r1  : I<0xD1, MRM7r, (outs GR32:$dst), (ins GR32:$src1),
                 "sar{l}\t$dst",
                 [(set GR32:$dst, (sra GR32:$src1, (i8 1)))]>;

let Constraints = "" in {
  let Uses = [CL] in {
  def SAR8mCL  : I<0xD2, MRM7m, (outs), (ins i8mem :$dst),
                   "sar{b}\t{%cl, $dst|$dst, CL}",
                   [(store (sra (loadi8 addr:$dst), CL), addr:$dst)]>;
  def SAR16mCL : I<0xD3, MRM7m, (outs), (ins i16mem:$dst),
                   "sar{w}\t{%cl, $dst|$dst, CL}",
                   [(store (sra (loadi16 addr:$dst), CL), addr:$dst)]>, OpSize;
  def SAR32mCL : I<0xD3, MRM7m, (outs), (ins i32mem:$dst), 
                   "sar{l}\t{%cl, $dst|$dst, CL}",
                   [(store (sra (loadi32 addr:$dst), CL), addr:$dst)]>;
  }
  def SAR8mi   : Ii8<0xC0, MRM7m, (outs), (ins i8mem :$dst, i8imm:$src),
                     "sar{b}\t{$src, $dst|$dst, $src}",
                  [(store (sra (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def SAR16mi  : Ii8<0xC1, MRM7m, (outs), (ins i16mem:$dst, i8imm:$src),
                     "sar{w}\t{$src, $dst|$dst, $src}",
                 [(store (sra (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def SAR32mi  : Ii8<0xC1, MRM7m, (outs), (ins i32mem:$dst, i8imm:$src),
                     "sar{l}\t{$src, $dst|$dst, $src}",
                 [(store (sra (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;

  // Shift by 1
  def SAR8m1   : I<0xD0, MRM7m, (outs), (ins i8mem :$dst),
                   "sar{b}\t$dst",
                  [(store (sra (loadi8 addr:$dst), (i8 1)), addr:$dst)]>;
  def SAR16m1  : I<0xD1, MRM7m, (outs), (ins i16mem:$dst),
                   "sar{w}\t$dst",
                 [(store (sra (loadi16 addr:$dst), (i8 1)), addr:$dst)]>,
                     OpSize;
  def SAR32m1  : I<0xD1, MRM7m, (outs), (ins i32mem:$dst),
                   "sar{l}\t$dst",
                 [(store (sra (loadi32 addr:$dst), (i8 1)), addr:$dst)]>;
} // Constraints = ""

// Rotate instructions

def RCL8r1 : I<0xD0, MRM2r, (outs GR8:$dst), (ins GR8:$src1),
               "rcl{b}\t{1, $dst|$dst, 1}", []>;
let Uses = [CL] in {
def RCL8rCL : I<0xD2, MRM2r, (outs GR8:$dst), (ins GR8:$src1),
                "rcl{b}\t{%cl, $dst|$dst, CL}", []>;
}
def RCL8ri : Ii8<0xC0, MRM2r, (outs GR8:$dst), (ins GR8:$src1, i8imm:$cnt),
                 "rcl{b}\t{$cnt, $dst|$dst, $cnt}", []>;
  
def RCL16r1 : I<0xD1, MRM2r, (outs GR16:$dst), (ins GR16:$src1),
                "rcl{w}\t{1, $dst|$dst, 1}", []>, OpSize;
let Uses = [CL] in {
def RCL16rCL : I<0xD3, MRM2r, (outs GR16:$dst), (ins GR16:$src1),
                 "rcl{w}\t{%cl, $dst|$dst, CL}", []>, OpSize;
}
def RCL16ri : Ii8<0xC1, MRM2r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$cnt),
                  "rcl{w}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize;

def RCL32r1 : I<0xD1, MRM2r, (outs GR32:$dst), (ins GR32:$src1),
                "rcl{l}\t{1, $dst|$dst, 1}", []>;
let Uses = [CL] in {
def RCL32rCL : I<0xD3, MRM2r, (outs GR32:$dst), (ins GR32:$src1),
                 "rcl{l}\t{%cl, $dst|$dst, CL}", []>;
}
def RCL32ri : Ii8<0xC1, MRM2r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$cnt),
                  "rcl{l}\t{$cnt, $dst|$dst, $cnt}", []>;
                  
def RCR8r1 : I<0xD0, MRM3r, (outs GR8:$dst), (ins GR8:$src1),
               "rcr{b}\t{1, $dst|$dst, 1}", []>;
let Uses = [CL] in {
def RCR8rCL : I<0xD2, MRM3r, (outs GR8:$dst), (ins GR8:$src1),
                "rcr{b}\t{%cl, $dst|$dst, CL}", []>;
}
def RCR8ri : Ii8<0xC0, MRM3r, (outs GR8:$dst), (ins GR8:$src1, i8imm:$cnt),
                 "rcr{b}\t{$cnt, $dst|$dst, $cnt}", []>;
  
def RCR16r1 : I<0xD1, MRM3r, (outs GR16:$dst), (ins GR16:$src1),
                "rcr{w}\t{1, $dst|$dst, 1}", []>, OpSize;
let Uses = [CL] in {
def RCR16rCL : I<0xD3, MRM3r, (outs GR16:$dst), (ins GR16:$src1),
                 "rcr{w}\t{%cl, $dst|$dst, CL}", []>, OpSize;
}
def RCR16ri : Ii8<0xC1, MRM3r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$cnt),
                  "rcr{w}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize;

def RCR32r1 : I<0xD1, MRM3r, (outs GR32:$dst), (ins GR32:$src1),
                "rcr{l}\t{1, $dst|$dst, 1}", []>;
let Uses = [CL] in {
def RCR32rCL : I<0xD3, MRM3r, (outs GR32:$dst), (ins GR32:$src1),
                 "rcr{l}\t{%cl, $dst|$dst, CL}", []>;
}
def RCR32ri : Ii8<0xC1, MRM3r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$cnt),
                  "rcr{l}\t{$cnt, $dst|$dst, $cnt}", []>;

let Constraints = "" in {
def RCL8m1 : I<0xD0, MRM2m, (outs), (ins i8mem:$dst),
               "rcl{b}\t{1, $dst|$dst, 1}", []>;
def RCL8mi : Ii8<0xC0, MRM2m, (outs), (ins i8mem:$dst, i8imm:$cnt),
                 "rcl{b}\t{$cnt, $dst|$dst, $cnt}", []>;
def RCL16m1 : I<0xD1, MRM2m, (outs), (ins i16mem:$dst),
                "rcl{w}\t{1, $dst|$dst, 1}", []>, OpSize;
def RCL16mi : Ii8<0xC1, MRM2m, (outs), (ins i16mem:$dst, i8imm:$cnt),
                  "rcl{w}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize;
def RCL32m1 : I<0xD1, MRM2m, (outs), (ins i32mem:$dst),
                "rcl{l}\t{1, $dst|$dst, 1}", []>;
def RCL32mi : Ii8<0xC1, MRM2m, (outs), (ins i32mem:$dst, i8imm:$cnt),
                  "rcl{l}\t{$cnt, $dst|$dst, $cnt}", []>;
def RCR8m1 : I<0xD0, MRM3m, (outs), (ins i8mem:$dst),
               "rcr{b}\t{1, $dst|$dst, 1}", []>;
def RCR8mi : Ii8<0xC0, MRM3m, (outs), (ins i8mem:$dst, i8imm:$cnt),
                 "rcr{b}\t{$cnt, $dst|$dst, $cnt}", []>;
def RCR16m1 : I<0xD1, MRM3m, (outs), (ins i16mem:$dst),
                "rcr{w}\t{1, $dst|$dst, 1}", []>, OpSize;
def RCR16mi : Ii8<0xC1, MRM3m, (outs), (ins i16mem:$dst, i8imm:$cnt),
                  "rcr{w}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize;
def RCR32m1 : I<0xD1, MRM3m, (outs), (ins i32mem:$dst),
                "rcr{l}\t{1, $dst|$dst, 1}", []>;
def RCR32mi : Ii8<0xC1, MRM3m, (outs), (ins i32mem:$dst, i8imm:$cnt),
                  "rcr{l}\t{$cnt, $dst|$dst, $cnt}", []>;

let Uses = [CL] in {
def RCL8mCL : I<0xD2, MRM2m, (outs), (ins i8mem:$dst),
                "rcl{b}\t{%cl, $dst|$dst, CL}", []>;
def RCL16mCL : I<0xD3, MRM2m, (outs), (ins i16mem:$dst),
                 "rcl{w}\t{%cl, $dst|$dst, CL}", []>, OpSize;
def RCL32mCL : I<0xD3, MRM2m, (outs), (ins i32mem:$dst),
                 "rcl{l}\t{%cl, $dst|$dst, CL}", []>;
def RCR8mCL : I<0xD2, MRM3m, (outs), (ins i8mem:$dst),
                "rcr{b}\t{%cl, $dst|$dst, CL}", []>;
def RCR16mCL : I<0xD3, MRM3m, (outs), (ins i16mem:$dst),
                 "rcr{w}\t{%cl, $dst|$dst, CL}", []>, OpSize;
def RCR32mCL : I<0xD3, MRM3m, (outs), (ins i32mem:$dst),
                 "rcr{l}\t{%cl, $dst|$dst, CL}", []>;
}
} // Constraints = ""

// FIXME: provide shorter instructions when imm8 == 1
let Uses = [CL] in {
def ROL8rCL  : I<0xD2, MRM0r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "rol{b}\t{%cl, $dst|$dst, CL}",
                 [(set GR8:$dst, (rotl GR8:$src1, CL))]>;
def ROL16rCL : I<0xD3, MRM0r, (outs GR16:$dst), (ins GR16:$src1),
                 "rol{w}\t{%cl, $dst|$dst, CL}",
                 [(set GR16:$dst, (rotl GR16:$src1, CL))]>, OpSize;
def ROL32rCL : I<0xD3, MRM0r, (outs GR32:$dst), (ins GR32:$src1),
                 "rol{l}\t{%cl, $dst|$dst, CL}",
                 [(set GR32:$dst, (rotl GR32:$src1, CL))]>;
}

def ROL8ri   : Ii8<0xC0, MRM0r, (outs GR8 :$dst), (ins GR8 :$src1, i8imm:$src2),
                   "rol{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (rotl GR8:$src1, (i8 imm:$src2)))]>;
def ROL16ri  : Ii8<0xC1, MRM0r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$src2),
                   "rol{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (rotl GR16:$src1, (i8 imm:$src2)))]>, 
                   OpSize;
def ROL32ri  : Ii8<0xC1, MRM0r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$src2),
                   "rol{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (rotl GR32:$src1, (i8 imm:$src2)))]>;

// Rotate by 1
def ROL8r1   : I<0xD0, MRM0r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "rol{b}\t$dst",
                 [(set GR8:$dst, (rotl GR8:$src1, (i8 1)))]>;
def ROL16r1  : I<0xD1, MRM0r, (outs GR16:$dst), (ins GR16:$src1),
                 "rol{w}\t$dst",
                 [(set GR16:$dst, (rotl GR16:$src1, (i8 1)))]>, OpSize;
def ROL32r1  : I<0xD1, MRM0r, (outs GR32:$dst), (ins GR32:$src1),
                 "rol{l}\t$dst",
                 [(set GR32:$dst, (rotl GR32:$src1, (i8 1)))]>;

let Constraints = "" in {
  let Uses = [CL] in {
  def ROL8mCL  : I<0xD2, MRM0m, (outs), (ins i8mem :$dst),
                   "rol{b}\t{%cl, $dst|$dst, CL}",
                   [(store (rotl (loadi8 addr:$dst), CL), addr:$dst)]>;
  def ROL16mCL : I<0xD3, MRM0m, (outs), (ins i16mem:$dst),
                   "rol{w}\t{%cl, $dst|$dst, CL}",
                   [(store (rotl (loadi16 addr:$dst), CL), addr:$dst)]>, OpSize;
  def ROL32mCL : I<0xD3, MRM0m, (outs), (ins i32mem:$dst),
                   "rol{l}\t{%cl, $dst|$dst, CL}",
                   [(store (rotl (loadi32 addr:$dst), CL), addr:$dst)]>;
  }
  def ROL8mi   : Ii8<0xC0, MRM0m, (outs), (ins i8mem :$dst, i8imm:$src),
                     "rol{b}\t{$src, $dst|$dst, $src}",
                 [(store (rotl (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def ROL16mi  : Ii8<0xC1, MRM0m, (outs), (ins i16mem:$dst, i8imm:$src),
                     "rol{w}\t{$src, $dst|$dst, $src}",
                [(store (rotl (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def ROL32mi  : Ii8<0xC1, MRM0m, (outs), (ins i32mem:$dst, i8imm:$src),
                     "rol{l}\t{$src, $dst|$dst, $src}",
                [(store (rotl (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;

  // Rotate by 1
  def ROL8m1   : I<0xD0, MRM0m, (outs), (ins i8mem :$dst),
                   "rol{b}\t$dst",
                 [(store (rotl (loadi8 addr:$dst), (i8 1)), addr:$dst)]>;
  def ROL16m1  : I<0xD1, MRM0m, (outs), (ins i16mem:$dst),
                   "rol{w}\t$dst",
                [(store (rotl (loadi16 addr:$dst), (i8 1)), addr:$dst)]>,
                     OpSize;
  def ROL32m1  : I<0xD1, MRM0m, (outs), (ins i32mem:$dst),
                   "rol{l}\t$dst",
                [(store (rotl (loadi32 addr:$dst), (i8 1)), addr:$dst)]>;
} // Constraints = ""

let Uses = [CL] in {
def ROR8rCL  : I<0xD2, MRM1r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "ror{b}\t{%cl, $dst|$dst, CL}",
                 [(set GR8:$dst, (rotr GR8:$src1, CL))]>;
def ROR16rCL : I<0xD3, MRM1r, (outs GR16:$dst), (ins GR16:$src1),
                 "ror{w}\t{%cl, $dst|$dst, CL}",
                 [(set GR16:$dst, (rotr GR16:$src1, CL))]>, OpSize;
def ROR32rCL : I<0xD3, MRM1r, (outs GR32:$dst), (ins GR32:$src1),
                 "ror{l}\t{%cl, $dst|$dst, CL}",
                 [(set GR32:$dst, (rotr GR32:$src1, CL))]>;
}

def ROR8ri   : Ii8<0xC0, MRM1r, (outs GR8 :$dst), (ins GR8 :$src1, i8imm:$src2),
                   "ror{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (rotr GR8:$src1, (i8 imm:$src2)))]>;
def ROR16ri  : Ii8<0xC1, MRM1r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$src2),
                   "ror{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (rotr GR16:$src1, (i8 imm:$src2)))]>, 
                   OpSize;
def ROR32ri  : Ii8<0xC1, MRM1r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$src2),
                   "ror{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (rotr GR32:$src1, (i8 imm:$src2)))]>;

// Rotate by 1
def ROR8r1   : I<0xD0, MRM1r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "ror{b}\t$dst",
                 [(set GR8:$dst, (rotr GR8:$src1, (i8 1)))]>;
def ROR16r1  : I<0xD1, MRM1r, (outs GR16:$dst), (ins GR16:$src1),
                 "ror{w}\t$dst",
                 [(set GR16:$dst, (rotr GR16:$src1, (i8 1)))]>, OpSize;
def ROR32r1  : I<0xD1, MRM1r, (outs GR32:$dst), (ins GR32:$src1),
                 "ror{l}\t$dst",
                 [(set GR32:$dst, (rotr GR32:$src1, (i8 1)))]>;

let Constraints = "" in {
  let Uses = [CL] in {
  def ROR8mCL  : I<0xD2, MRM1m, (outs), (ins i8mem :$dst),
                   "ror{b}\t{%cl, $dst|$dst, CL}",
                   [(store (rotr (loadi8 addr:$dst), CL), addr:$dst)]>;
  def ROR16mCL : I<0xD3, MRM1m, (outs), (ins i16mem:$dst),
                   "ror{w}\t{%cl, $dst|$dst, CL}",
                   [(store (rotr (loadi16 addr:$dst), CL), addr:$dst)]>, OpSize;
  def ROR32mCL : I<0xD3, MRM1m, (outs), (ins i32mem:$dst), 
                   "ror{l}\t{%cl, $dst|$dst, CL}",
                   [(store (rotr (loadi32 addr:$dst), CL), addr:$dst)]>;
  }
  def ROR8mi   : Ii8<0xC0, MRM1m, (outs), (ins i8mem :$dst, i8imm:$src),
                     "ror{b}\t{$src, $dst|$dst, $src}",
                 [(store (rotr (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def ROR16mi  : Ii8<0xC1, MRM1m, (outs), (ins i16mem:$dst, i8imm:$src),
                     "ror{w}\t{$src, $dst|$dst, $src}",
                [(store (rotr (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def ROR32mi  : Ii8<0xC1, MRM1m, (outs), (ins i32mem:$dst, i8imm:$src),
                     "ror{l}\t{$src, $dst|$dst, $src}",
                [(store (rotr (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;

  // Rotate by 1
  def ROR8m1   : I<0xD0, MRM1m, (outs), (ins i8mem :$dst),
                   "ror{b}\t$dst",
                 [(store (rotr (loadi8 addr:$dst), (i8 1)), addr:$dst)]>;
  def ROR16m1  : I<0xD1, MRM1m, (outs), (ins i16mem:$dst),
                   "ror{w}\t$dst",
                [(store (rotr (loadi16 addr:$dst), (i8 1)), addr:$dst)]>,
                     OpSize;
  def ROR32m1  : I<0xD1, MRM1m, (outs), (ins i32mem:$dst),
                   "ror{l}\t$dst",
                [(store (rotr (loadi32 addr:$dst), (i8 1)), addr:$dst)]>;
} // Constraints = ""


// Double shift instructions (generalizations of rotate)
let Uses = [CL] in {
def SHLD32rrCL : I<0xA5, MRMDestReg, (outs GR32:$dst), 
                   (ins GR32:$src1, GR32:$src2),
                   "shld{l}\t{%cl, $src2, $dst|$dst, $src2, CL}",
                   [(set GR32:$dst, (X86shld GR32:$src1, GR32:$src2, CL))]>, TB;
def SHRD32rrCL : I<0xAD, MRMDestReg, (outs GR32:$dst),
                   (ins GR32:$src1, GR32:$src2),
                   "shrd{l}\t{%cl, $src2, $dst|$dst, $src2, CL}",
                   [(set GR32:$dst, (X86shrd GR32:$src1, GR32:$src2, CL))]>, TB;
def SHLD16rrCL : I<0xA5, MRMDestReg, (outs GR16:$dst), 
                   (ins GR16:$src1, GR16:$src2),
                   "shld{w}\t{%cl, $src2, $dst|$dst, $src2, CL}",
                   [(set GR16:$dst, (X86shld GR16:$src1, GR16:$src2, CL))]>,
                   TB, OpSize;
def SHRD16rrCL : I<0xAD, MRMDestReg, (outs GR16:$dst), 
                   (ins GR16:$src1, GR16:$src2),
                   "shrd{w}\t{%cl, $src2, $dst|$dst, $src2, CL}",
                   [(set GR16:$dst, (X86shrd GR16:$src1, GR16:$src2, CL))]>,
                   TB, OpSize;
}

let isCommutable = 1 in {  // These instructions commute to each other.
def SHLD32rri8 : Ii8<0xA4, MRMDestReg,
                     (outs GR32:$dst), 
                     (ins GR32:$src1, GR32:$src2, i8imm:$src3),
                     "shld{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR32:$dst, (X86shld GR32:$src1, GR32:$src2,
                                      (i8 imm:$src3)))]>,
                 TB;
def SHRD32rri8 : Ii8<0xAC, MRMDestReg,
                     (outs GR32:$dst), 
                     (ins GR32:$src1, GR32:$src2, i8imm:$src3),
                     "shrd{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR32:$dst, (X86shrd GR32:$src1, GR32:$src2,
                                      (i8 imm:$src3)))]>,
                 TB;
def SHLD16rri8 : Ii8<0xA4, MRMDestReg,
                     (outs GR16:$dst), 
                     (ins GR16:$src1, GR16:$src2, i8imm:$src3),
                     "shld{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR16:$dst, (X86shld GR16:$src1, GR16:$src2,
                                      (i8 imm:$src3)))]>,
                     TB, OpSize;
def SHRD16rri8 : Ii8<0xAC, MRMDestReg,
                     (outs GR16:$dst), 
                     (ins GR16:$src1, GR16:$src2, i8imm:$src3),
                     "shrd{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR16:$dst, (X86shrd GR16:$src1, GR16:$src2,
                                      (i8 imm:$src3)))]>,
                     TB, OpSize;
}

let Constraints = "" in {
  let Uses = [CL] in {
  def SHLD32mrCL : I<0xA5, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                     "shld{l}\t{%cl, $src2, $dst|$dst, $src2, CL}",
                     [(store (X86shld (loadi32 addr:$dst), GR32:$src2, CL),
                       addr:$dst)]>, TB;
  def SHRD32mrCL : I<0xAD, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                    "shrd{l}\t{%cl, $src2, $dst|$dst, $src2, CL}",
                    [(store (X86shrd (loadi32 addr:$dst), GR32:$src2, CL),
                      addr:$dst)]>, TB;
  }
  def SHLD32mri8 : Ii8<0xA4, MRMDestMem,
                      (outs), (ins i32mem:$dst, GR32:$src2, i8imm:$src3),
                      "shld{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shld (loadi32 addr:$dst), GR32:$src2,
                                        (i8 imm:$src3)), addr:$dst)]>,
                      TB;
  def SHRD32mri8 : Ii8<0xAC, MRMDestMem, 
                       (outs), (ins i32mem:$dst, GR32:$src2, i8imm:$src3),
                       "shrd{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                       [(store (X86shrd (loadi32 addr:$dst), GR32:$src2,
                                         (i8 imm:$src3)), addr:$dst)]>,
                       TB;

  let Uses = [CL] in {
  def SHLD16mrCL : I<0xA5, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                     "shld{w}\t{%cl, $src2, $dst|$dst, $src2, CL}",
                     [(store (X86shld (loadi16 addr:$dst), GR16:$src2, CL),
                       addr:$dst)]>, TB, OpSize;
  def SHRD16mrCL : I<0xAD, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                    "shrd{w}\t{%cl, $src2, $dst|$dst, $src2, CL}",
                    [(store (X86shrd (loadi16 addr:$dst), GR16:$src2, CL),
                      addr:$dst)]>, TB, OpSize;
  }
  def SHLD16mri8 : Ii8<0xA4, MRMDestMem,
                      (outs), (ins i16mem:$dst, GR16:$src2, i8imm:$src3),
                      "shld{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shld (loadi16 addr:$dst), GR16:$src2,
                                        (i8 imm:$src3)), addr:$dst)]>,
                      TB, OpSize;
  def SHRD16mri8 : Ii8<0xAC, MRMDestMem, 
                       (outs), (ins i16mem:$dst, GR16:$src2, i8imm:$src3),
                       "shrd{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shrd (loadi16 addr:$dst), GR16:$src2,
                                        (i8 imm:$src3)), addr:$dst)]>,
                       TB, OpSize;
} // Constraints = ""
} // Defs = [EFLAGS]


// Arithmetic.
let Defs = [EFLAGS] in {
let isCommutable = 1 in {   // X = ADD Y, Z   --> X = ADD Z, Y
// Register-Register Addition
def ADD8rr    : I<0x00, MRMDestReg, (outs GR8 :$dst),
                                    (ins GR8 :$src1, GR8 :$src2),
                  "add{b}\t{$src2, $dst|$dst, $src2}",
                  [(set GR8:$dst, EFLAGS, (X86add_flag GR8:$src1, GR8:$src2))]>;

let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
// Register-Register Addition
def ADD16rr  : I<0x01, MRMDestReg, (outs GR16:$dst),
                                   (ins GR16:$src1, GR16:$src2),
                 "add{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, EFLAGS, (X86add_flag GR16:$src1,
                                                       GR16:$src2))]>, OpSize;
def ADD32rr  : I<0x01, MRMDestReg, (outs GR32:$dst),
                                   (ins GR32:$src1, GR32:$src2),
                 "add{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, EFLAGS, (X86add_flag GR32:$src1,
                                                       GR32:$src2))]>;
} // end isConvertibleToThreeAddress
} // end isCommutable

// These are alternate spellings for use by the disassembler, we mark them as
// code gen only to ensure they aren't matched by the assembler.
let isCodeGenOnly = 1 in {
  def ADD8rr_alt: I<0x02, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                   "add{b}\t{$src2, $dst|$dst, $src2}", []>;
  def ADD16rr_alt: I<0x03, MRMSrcReg,(outs GR16:$dst),(ins GR16:$src1, GR16:$src2),
                    "add{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize;
  def ADD32rr_alt: I<0x03, MRMSrcReg,(outs GR32:$dst),(ins GR32:$src1, GR32:$src2),
                    "add{l}\t{$src2, $dst|$dst, $src2}", []>;
}

// Register-Memory Addition
def ADD8rm   : I<0x02, MRMSrcMem, (outs GR8 :$dst),
                                  (ins GR8 :$src1, i8mem :$src2),
                 "add{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, EFLAGS, (X86add_flag GR8:$src1,
                                                      (load addr:$src2)))]>;
def ADD16rm  : I<0x03, MRMSrcMem, (outs GR16:$dst),
                                  (ins GR16:$src1, i16mem:$src2),
                 "add{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, EFLAGS, (X86add_flag GR16:$src1,
                                                  (load addr:$src2)))]>, OpSize;
def ADD32rm  : I<0x03, MRMSrcMem, (outs GR32:$dst),
                                  (ins GR32:$src1, i32mem:$src2),
                 "add{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, EFLAGS, (X86add_flag GR32:$src1,
                                                       (load addr:$src2)))]>;
                  
// Register-Integer Addition
def ADD8ri    : Ii8<0x80, MRM0r, (outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                    "add{b}\t{$src2, $dst|$dst, $src2}",
                    [(set GR8:$dst, EFLAGS,
                          (X86add_flag GR8:$src1, imm:$src2))]>;

let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
// Register-Integer Addition
def ADD16ri  : Ii16<0x81, MRM0r, (outs GR16:$dst),
                                 (ins GR16:$src1, i16imm:$src2),
                    "add{w}\t{$src2, $dst|$dst, $src2}",
                    [(set GR16:$dst, EFLAGS,
                          (X86add_flag GR16:$src1, imm:$src2))]>, OpSize;
def ADD32ri  : Ii32<0x81, MRM0r, (outs GR32:$dst),
                                 (ins GR32:$src1, i32imm:$src2),
                    "add{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, EFLAGS, 
                          (X86add_flag GR32:$src1, imm:$src2))]>;
def ADD16ri8 : Ii8<0x83, MRM0r, (outs GR16:$dst),
                                (ins GR16:$src1, i16i8imm:$src2),
                   "add{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, EFLAGS,
                         (X86add_flag GR16:$src1, i16immSExt8:$src2))]>, OpSize;
def ADD32ri8 : Ii8<0x83, MRM0r, (outs GR32:$dst),
                                (ins GR32:$src1, i32i8imm:$src2),
                   "add{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, EFLAGS,
                         (X86add_flag GR32:$src1, i32immSExt8:$src2))]>;
}

let Constraints = "" in {
  // Memory-Register Addition
  def ADD8mr   : I<0x00, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src2),
                   "add{b}\t{$src2, $dst|$dst, $src2}",
                   [(store (add (load addr:$dst), GR8:$src2), addr:$dst),
                    (implicit EFLAGS)]>;
  def ADD16mr  : I<0x01, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                   "add{w}\t{$src2, $dst|$dst, $src2}",
                   [(store (add (load addr:$dst), GR16:$src2), addr:$dst),
                    (implicit EFLAGS)]>, OpSize;
  def ADD32mr  : I<0x01, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                   "add{l}\t{$src2, $dst|$dst, $src2}",
                   [(store (add (load addr:$dst), GR32:$src2), addr:$dst),
                    (implicit EFLAGS)]>;
  def ADD8mi   : Ii8<0x80, MRM0m, (outs), (ins i8mem :$dst, i8imm :$src2),
                     "add{b}\t{$src2, $dst|$dst, $src2}",
                   [(store (add (loadi8 addr:$dst), imm:$src2), addr:$dst),
                    (implicit EFLAGS)]>;
  def ADD16mi  : Ii16<0x81, MRM0m, (outs), (ins i16mem:$dst, i16imm:$src2),
                      "add{w}\t{$src2, $dst|$dst, $src2}",
                  [(store (add (loadi16 addr:$dst), imm:$src2), addr:$dst),
                   (implicit EFLAGS)]>, OpSize;
  def ADD32mi  : Ii32<0x81, MRM0m, (outs), (ins i32mem:$dst, i32imm:$src2),
                      "add{l}\t{$src2, $dst|$dst, $src2}",
                      [(store (add (loadi32 addr:$dst), imm:$src2), addr:$dst),
                       (implicit EFLAGS)]>;
  def ADD16mi8 : Ii8<0x83, MRM0m, (outs), (ins i16mem:$dst, i16i8imm :$src2),
                     "add{w}\t{$src2, $dst|$dst, $src2}",
                     [(store (add (load addr:$dst), i16immSExt8:$src2),
                                  addr:$dst),
                      (implicit EFLAGS)]>, OpSize;
  def ADD32mi8 : Ii8<0x83, MRM0m, (outs), (ins i32mem:$dst, i32i8imm :$src2),
                     "add{l}\t{$src2, $dst|$dst, $src2}",
                  [(store (add (load addr:$dst), i32immSExt8:$src2),
                               addr:$dst),
                   (implicit EFLAGS)]>;

  // addition to rAX
  def ADD8i8 : Ii8<0x04, RawFrm, (outs), (ins i8imm:$src),
                   "add{b}\t{$src, %al|%al, $src}", []>;
  def ADD16i16 : Ii16<0x05, RawFrm, (outs), (ins i16imm:$src),
                      "add{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
  def ADD32i32 : Ii32<0x05, RawFrm, (outs), (ins i32imm:$src),
                      "add{l}\t{$src, %eax|%eax, $src}", []>;
} // Constraints = ""

let Uses = [EFLAGS] in {
let isCommutable = 1 in {  // X = ADC Y, Z --> X = ADC Z, Y
def ADC8rr   : I<0x10, MRMDestReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                 "adc{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, (adde GR8:$src1, GR8:$src2))]>;
def ADC16rr  : I<0x11, MRMDestReg, (outs GR16:$dst),
                                   (ins GR16:$src1, GR16:$src2),
                 "adc{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (adde GR16:$src1, GR16:$src2))]>, OpSize;
def ADC32rr  : I<0x11, MRMDestReg, (outs GR32:$dst),
                                   (ins GR32:$src1, GR32:$src2),
                 "adc{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (adde GR32:$src1, GR32:$src2))]>;
}

let isCodeGenOnly = 1 in {
def ADC8rr_REV : I<0x12, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                 "adc{b}\t{$src2, $dst|$dst, $src2}", []>;
def ADC16rr_REV : I<0x13, MRMSrcReg, (outs GR16:$dst), 
                    (ins GR16:$src1, GR16:$src2),
                    "adc{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize;
def ADC32rr_REV : I<0x13, MRMSrcReg, (outs GR32:$dst), 
                    (ins GR32:$src1, GR32:$src2),
                    "adc{l}\t{$src2, $dst|$dst, $src2}", []>;
}

def ADC8rm   : I<0x12, MRMSrcMem , (outs GR8:$dst), 
                                   (ins GR8:$src1, i8mem:$src2),
                 "adc{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, (adde GR8:$src1, (load addr:$src2)))]>;
def ADC16rm  : I<0x13, MRMSrcMem , (outs GR16:$dst),
                                   (ins GR16:$src1, i16mem:$src2),
                 "adc{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (adde GR16:$src1, (load addr:$src2)))]>,
                 OpSize;
def ADC32rm  : I<0x13, MRMSrcMem , (outs GR32:$dst),
                                   (ins GR32:$src1, i32mem:$src2),
                 "adc{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (adde GR32:$src1, (load addr:$src2)))]>;
def ADC8ri   : Ii8<0x80, MRM2r, (outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                    "adc{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, (adde GR8:$src1, imm:$src2))]>;
def ADC16ri  : Ii16<0x81, MRM2r, (outs GR16:$dst),
                                 (ins GR16:$src1, i16imm:$src2),
                    "adc{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (adde GR16:$src1, imm:$src2))]>, OpSize;
def ADC16ri8 : Ii8<0x83, MRM2r, (outs GR16:$dst),
                                (ins GR16:$src1, i16i8imm:$src2),
                   "adc{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (adde GR16:$src1, i16immSExt8:$src2))]>,
                 OpSize;
def ADC32ri  : Ii32<0x81, MRM2r, (outs GR32:$dst),
                                 (ins GR32:$src1, i32imm:$src2),
                    "adc{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (adde GR32:$src1, imm:$src2))]>;
def ADC32ri8 : Ii8<0x83, MRM2r, (outs GR32:$dst),
                                (ins GR32:$src1, i32i8imm:$src2),
                   "adc{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (adde GR32:$src1, i32immSExt8:$src2))]>;

let Constraints = "" in {
  def ADC8mr   : I<0x10, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src2),
                   "adc{b}\t{$src2, $dst|$dst, $src2}",
                   [(store (adde (load addr:$dst), GR8:$src2), addr:$dst)]>;
  def ADC16mr  : I<0x11, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                   "adc{w}\t{$src2, $dst|$dst, $src2}",
                   [(store (adde (load addr:$dst), GR16:$src2), addr:$dst)]>,
                   OpSize;
  def ADC32mr  : I<0x11, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                   "adc{l}\t{$src2, $dst|$dst, $src2}",
                   [(store (adde (load addr:$dst), GR32:$src2), addr:$dst)]>;
  def ADC8mi   : Ii8<0x80, MRM2m, (outs), (ins i8mem:$dst, i8imm:$src2),
                      "adc{b}\t{$src2, $dst|$dst, $src2}",
                  [(store (adde (loadi8 addr:$dst), imm:$src2), addr:$dst)]>;
  def ADC16mi  : Ii16<0x81, MRM2m, (outs), (ins i16mem:$dst, i16imm:$src2),
                      "adc{w}\t{$src2, $dst|$dst, $src2}",
                  [(store (adde (loadi16 addr:$dst), imm:$src2), addr:$dst)]>,
                  OpSize;
  def ADC16mi8 : Ii8<0x83, MRM2m, (outs), (ins i16mem:$dst, i16i8imm :$src2),
                     "adc{w}\t{$src2, $dst|$dst, $src2}",
               [(store (adde (load addr:$dst), i16immSExt8:$src2), addr:$dst)]>,
               OpSize;
  def ADC32mi  : Ii32<0x81, MRM2m, (outs), (ins i32mem:$dst, i32imm:$src2),
                      "adc{l}\t{$src2, $dst|$dst, $src2}",
                  [(store (adde (loadi32 addr:$dst), imm:$src2), addr:$dst)]>;
  def ADC32mi8 : Ii8<0x83, MRM2m, (outs), (ins i32mem:$dst, i32i8imm :$src2),
                     "adc{l}\t{$src2, $dst|$dst, $src2}",
               [(store (adde (load addr:$dst), i32immSExt8:$src2), addr:$dst)]>;

  def ADC8i8 : Ii8<0x14, RawFrm, (outs), (ins i8imm:$src),
                   "adc{b}\t{$src, %al|%al, $src}", []>;
  def ADC16i16 : Ii16<0x15, RawFrm, (outs), (ins i16imm:$src),
                      "adc{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
  def ADC32i32 : Ii32<0x15, RawFrm, (outs), (ins i32imm:$src),
                      "adc{l}\t{$src, %eax|%eax, $src}", []>;
} // Constraints = ""
} // Uses = [EFLAGS]

// Register-Register Subtraction
def SUB8rr  : I<0x28, MRMDestReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                "sub{b}\t{$src2, $dst|$dst, $src2}",
                [(set GR8:$dst, EFLAGS,
                      (X86sub_flag GR8:$src1, GR8:$src2))]>;
def SUB16rr : I<0x29, MRMDestReg, (outs GR16:$dst), (ins GR16:$src1,GR16:$src2),
                "sub{w}\t{$src2, $dst|$dst, $src2}",
                [(set GR16:$dst, EFLAGS,
                      (X86sub_flag GR16:$src1, GR16:$src2))]>, OpSize;
def SUB32rr : I<0x29, MRMDestReg, (outs GR32:$dst), (ins GR32:$src1,GR32:$src2),
                "sub{l}\t{$src2, $dst|$dst, $src2}",
                [(set GR32:$dst, EFLAGS,
                      (X86sub_flag GR32:$src1, GR32:$src2))]>;

let isCodeGenOnly = 1 in {
def SUB8rr_REV : I<0x2A, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                   "sub{b}\t{$src2, $dst|$dst, $src2}", []>;
def SUB16rr_REV : I<0x2B, MRMSrcReg, (outs GR16:$dst), 
                    (ins GR16:$src1, GR16:$src2),
                    "sub{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize;
def SUB32rr_REV : I<0x2B, MRMSrcReg, (outs GR32:$dst), 
                    (ins GR32:$src1, GR32:$src2),
                    "sub{l}\t{$src2, $dst|$dst, $src2}", []>;
}

// Register-Memory Subtraction
def SUB8rm  : I<0x2A, MRMSrcMem, (outs GR8 :$dst),
                                 (ins GR8 :$src1, i8mem :$src2),
                "sub{b}\t{$src2, $dst|$dst, $src2}",
                [(set GR8:$dst, EFLAGS,
                      (X86sub_flag GR8:$src1, (load addr:$src2)))]>;
def SUB16rm : I<0x2B, MRMSrcMem, (outs GR16:$dst),
                                 (ins GR16:$src1, i16mem:$src2),
                "sub{w}\t{$src2, $dst|$dst, $src2}",
                [(set GR16:$dst, EFLAGS,
                      (X86sub_flag GR16:$src1, (load addr:$src2)))]>, OpSize;
def SUB32rm : I<0x2B, MRMSrcMem, (outs GR32:$dst),
                                 (ins GR32:$src1, i32mem:$src2),
                "sub{l}\t{$src2, $dst|$dst, $src2}",
                [(set GR32:$dst, EFLAGS,
                      (X86sub_flag GR32:$src1, (load addr:$src2)))]>;

// Register-Integer Subtraction
def SUB8ri   : Ii8 <0x80, MRM5r, (outs GR8:$dst),
                                 (ins GR8:$src1, i8imm:$src2),
                    "sub{b}\t{$src2, $dst|$dst, $src2}",
                    [(set GR8:$dst, EFLAGS,
                          (X86sub_flag GR8:$src1, imm:$src2))]>;
def SUB16ri  : Ii16<0x81, MRM5r, (outs GR16:$dst),
                                 (ins GR16:$src1, i16imm:$src2),
                    "sub{w}\t{$src2, $dst|$dst, $src2}",
                    [(set GR16:$dst, EFLAGS,
                          (X86sub_flag GR16:$src1, imm:$src2))]>, OpSize;
def SUB32ri  : Ii32<0x81, MRM5r, (outs GR32:$dst),
                                 (ins GR32:$src1, i32imm:$src2),
                    "sub{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, EFLAGS,
                          (X86sub_flag GR32:$src1, imm:$src2))]>;
def SUB16ri8 : Ii8<0x83, MRM5r, (outs GR16:$dst),
                                (ins GR16:$src1, i16i8imm:$src2),
                   "sub{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, EFLAGS,
                         (X86sub_flag GR16:$src1, i16immSExt8:$src2))]>, OpSize;
def SUB32ri8 : Ii8<0x83, MRM5r, (outs GR32:$dst),
                                (ins GR32:$src1, i32i8imm:$src2),
                   "sub{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, EFLAGS,
                         (X86sub_flag GR32:$src1, i32immSExt8:$src2))]>;

let Constraints = "" in {
  // Memory-Register Subtraction
  def SUB8mr   : I<0x28, MRMDestMem, (outs), (ins i8mem :$dst, GR8 :$src2),
                   "sub{b}\t{$src2, $dst|$dst, $src2}",
                   [(store (sub (load addr:$dst), GR8:$src2), addr:$dst),
                    (implicit EFLAGS)]>;
  def SUB16mr  : I<0x29, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                   "sub{w}\t{$src2, $dst|$dst, $src2}",
                   [(store (sub (load addr:$dst), GR16:$src2), addr:$dst),
                    (implicit EFLAGS)]>, OpSize;
  def SUB32mr  : I<0x29, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2), 
                   "sub{l}\t{$src2, $dst|$dst, $src2}",
                   [(store (sub (load addr:$dst), GR32:$src2), addr:$dst),
                    (implicit EFLAGS)]>;

  // Memory-Integer Subtraction
  def SUB8mi   : Ii8<0x80, MRM5m, (outs), (ins i8mem :$dst, i8imm:$src2), 
                     "sub{b}\t{$src2, $dst|$dst, $src2}",
                     [(store (sub (loadi8 addr:$dst), imm:$src2), addr:$dst),
                      (implicit EFLAGS)]>;
  def SUB16mi  : Ii16<0x81, MRM5m, (outs), (ins i16mem:$dst, i16imm:$src2), 
                      "sub{w}\t{$src2, $dst|$dst, $src2}",
                      [(store (sub (loadi16 addr:$dst), imm:$src2),addr:$dst),
                       (implicit EFLAGS)]>, OpSize;
  def SUB32mi  : Ii32<0x81, MRM5m, (outs), (ins i32mem:$dst, i32imm:$src2), 
                      "sub{l}\t{$src2, $dst|$dst, $src2}",
                      [(store (sub (loadi32 addr:$dst), imm:$src2),addr:$dst),
                       (implicit EFLAGS)]>;
  def SUB16mi8 : Ii8<0x83, MRM5m, (outs), (ins i16mem:$dst, i16i8imm :$src2), 
                     "sub{w}\t{$src2, $dst|$dst, $src2}",
                     [(store (sub (load addr:$dst), i16immSExt8:$src2),
                             addr:$dst),
                      (implicit EFLAGS)]>, OpSize;
  def SUB32mi8 : Ii8<0x83, MRM5m, (outs), (ins i32mem:$dst, i32i8imm :$src2),
                     "sub{l}\t{$src2, $dst|$dst, $src2}",
                     [(store (sub (load addr:$dst), i32immSExt8:$src2),
                             addr:$dst),
                      (implicit EFLAGS)]>;
                      
  def SUB8i8 : Ii8<0x2C, RawFrm, (outs), (ins i8imm:$src),
                   "sub{b}\t{$src, %al|%al, $src}", []>;
  def SUB16i16 : Ii16<0x2D, RawFrm, (outs), (ins i16imm:$src),
                      "sub{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
  def SUB32i32 : Ii32<0x2D, RawFrm, (outs), (ins i32imm:$src),
                      "sub{l}\t{$src, %eax|%eax, $src}", []>;
} // Constraints = ""

let Uses = [EFLAGS] in {
def SBB8rr     : I<0x18, MRMDestReg, (outs GR8:$dst),
                                     (ins GR8:$src1, GR8:$src2),
                  "sbb{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, (sube GR8:$src1, GR8:$src2))]>;
def SBB16rr    : I<0x19, MRMDestReg, (outs GR16:$dst),
                                     (ins GR16:$src1, GR16:$src2),
                  "sbb{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (sube GR16:$src1, GR16:$src2))]>, OpSize;
def SBB32rr    : I<0x19, MRMDestReg, (outs GR32:$dst),
                                      (ins GR32:$src1, GR32:$src2),
                  "sbb{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (sube GR32:$src1, GR32:$src2))]>;

let Constraints = "" in {
  def SBB8mr   : I<0x18, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src2), 
                   "sbb{b}\t{$src2, $dst|$dst, $src2}",
                   [(store (sube (load addr:$dst), GR8:$src2), addr:$dst)]>;
  def SBB16mr  : I<0x19, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2), 
                   "sbb{w}\t{$src2, $dst|$dst, $src2}",
                   [(store (sube (load addr:$dst), GR16:$src2), addr:$dst)]>,
                   OpSize;
  def SBB32mr  : I<0x19, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2), 
                   "sbb{l}\t{$src2, $dst|$dst, $src2}",
                   [(store (sube (load addr:$dst), GR32:$src2), addr:$dst)]>;
  def SBB8mi  : Ii8<0x80, MRM3m, (outs), (ins i8mem:$dst, i8imm:$src2), 
                    "sbb{b}\t{$src2, $dst|$dst, $src2}",
                   [(store (sube (loadi8 addr:$dst), imm:$src2), addr:$dst)]>;
  def SBB16mi  : Ii16<0x81, MRM3m, (outs), (ins i16mem:$dst, i16imm:$src2), 
                      "sbb{w}\t{$src2, $dst|$dst, $src2}",
                  [(store (sube (loadi16 addr:$dst), imm:$src2), addr:$dst)]>,
                  OpSize;
  def SBB16mi8 : Ii8<0x83, MRM3m, (outs), (ins i16mem:$dst, i16i8imm :$src2), 
                     "sbb{w}\t{$src2, $dst|$dst, $src2}",
               [(store (sube (load addr:$dst), i16immSExt8:$src2), addr:$dst)]>,
               OpSize;
  def SBB32mi  : Ii32<0x81, MRM3m, (outs), (ins i32mem:$dst, i32imm:$src2), 
                      "sbb{l}\t{$src2, $dst|$dst, $src2}",
                  [(store (sube (loadi32 addr:$dst), imm:$src2), addr:$dst)]>;
  def SBB32mi8 : Ii8<0x83, MRM3m, (outs), (ins i32mem:$dst, i32i8imm :$src2), 
                     "sbb{l}\t{$src2, $dst|$dst, $src2}",
               [(store (sube (load addr:$dst), i32immSExt8:$src2), addr:$dst)]>;
               
  def SBB8i8 : Ii8<0x1C, RawFrm, (outs), (ins i8imm:$src),
                   "sbb{b}\t{$src, %al|%al, $src}", []>;
  def SBB16i16 : Ii16<0x1D, RawFrm, (outs), (ins i16imm:$src),
                      "sbb{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
  def SBB32i32 : Ii32<0x1D, RawFrm, (outs), (ins i32imm:$src),
                      "sbb{l}\t{$src, %eax|%eax, $src}", []>;
} // Constraints = ""

let isCodeGenOnly = 1 in {
def SBB8rr_REV : I<0x1A, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                   "sbb{b}\t{$src2, $dst|$dst, $src2}", []>;
def SBB16rr_REV : I<0x1B, MRMSrcReg, (outs GR16:$dst), 
                    (ins GR16:$src1, GR16:$src2),
                    "sbb{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize;
def SBB32rr_REV : I<0x1B, MRMSrcReg, (outs GR32:$dst), 
                    (ins GR32:$src1, GR32:$src2),
                    "sbb{l}\t{$src2, $dst|$dst, $src2}", []>;
}

def SBB8rm   : I<0x1A, MRMSrcMem, (outs GR8:$dst), (ins GR8:$src1, i8mem:$src2),
                    "sbb{b}\t{$src2, $dst|$dst, $src2}",
                    [(set GR8:$dst, (sube GR8:$src1, (load addr:$src2)))]>;
def SBB16rm  : I<0x1B, MRMSrcMem, (outs GR16:$dst),
                                  (ins GR16:$src1, i16mem:$src2),
                    "sbb{w}\t{$src2, $dst|$dst, $src2}",
                    [(set GR16:$dst, (sube GR16:$src1, (load addr:$src2)))]>,
                    OpSize;
def SBB32rm  : I<0x1B, MRMSrcMem, (outs GR32:$dst),
                                  (ins GR32:$src1, i32mem:$src2),
                    "sbb{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, (sube GR32:$src1, (load addr:$src2)))]>;
def SBB8ri   : Ii8<0x80, MRM3r, (outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                    "sbb{b}\t{$src2, $dst|$dst, $src2}",
                    [(set GR8:$dst, (sube GR8:$src1, imm:$src2))]>;
def SBB16ri  : Ii16<0x81, MRM3r, (outs GR16:$dst),
                                 (ins GR16:$src1, i16imm:$src2),
                    "sbb{w}\t{$src2, $dst|$dst, $src2}",
                    [(set GR16:$dst, (sube GR16:$src1, imm:$src2))]>, OpSize;
def SBB16ri8 : Ii8<0x83, MRM3r, (outs GR16:$dst),
                                (ins GR16:$src1, i16i8imm:$src2),
                   "sbb{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (sube GR16:$src1, i16immSExt8:$src2))]>,
                   OpSize;
def SBB32ri  : Ii32<0x81, MRM3r, (outs GR32:$dst), 
                                 (ins GR32:$src1, i32imm:$src2),
                    "sbb{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, (sube GR32:$src1, imm:$src2))]>;
def SBB32ri8 : Ii8<0x83, MRM3r, (outs GR32:$dst),
                                (ins GR32:$src1, i32i8imm:$src2),
                   "sbb{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (sube GR32:$src1, i32immSExt8:$src2))]>;
} // Uses = [EFLAGS]
} // Defs = [EFLAGS]

let Defs = [EFLAGS] in {
let isCommutable = 1 in {  // X = IMUL Y, Z --> X = IMUL Z, Y
// Register-Register Signed Integer Multiply
def IMUL16rr : I<0xAF, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src1,GR16:$src2),
                 "imul{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, EFLAGS,
                       (X86smul_flag GR16:$src1, GR16:$src2))]>, TB, OpSize;
def IMUL32rr : I<0xAF, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src1,GR32:$src2),
                 "imul{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, EFLAGS,
                       (X86smul_flag GR32:$src1, GR32:$src2))]>, TB;
}

// Register-Memory Signed Integer Multiply
def IMUL16rm : I<0xAF, MRMSrcMem, (outs GR16:$dst),
                                  (ins GR16:$src1, i16mem:$src2),
                 "imul{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, EFLAGS,
                       (X86smul_flag GR16:$src1, (load addr:$src2)))]>,
               TB, OpSize;
def IMUL32rm : I<0xAF, MRMSrcMem, (outs GR32:$dst), 
                 (ins GR32:$src1, i32mem:$src2),
                 "imul{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, EFLAGS,
                       (X86smul_flag GR32:$src1, (load addr:$src2)))]>, TB;
} // Defs = [EFLAGS]
} // end Two Address instructions

// Suprisingly enough, these are not two address instructions!
let Defs = [EFLAGS] in {
// Register-Integer Signed Integer Multiply
def IMUL16rri  : Ii16<0x69, MRMSrcReg,                      // GR16 = GR16*I16
                      (outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                      "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR16:$dst, EFLAGS, 
                            (X86smul_flag GR16:$src1, imm:$src2))]>, OpSize;
def IMUL32rri  : Ii32<0x69, MRMSrcReg,                      // GR32 = GR32*I32
                      (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                      "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR32:$dst, EFLAGS,
                            (X86smul_flag GR32:$src1, imm:$src2))]>;
def IMUL16rri8 : Ii8<0x6B, MRMSrcReg,                       // GR16 = GR16*I8
                     (outs GR16:$dst), (ins GR16:$src1, i16i8imm:$src2),
                     "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR16:$dst, EFLAGS,
                           (X86smul_flag GR16:$src1, i16immSExt8:$src2))]>,
                 OpSize;
def IMUL32rri8 : Ii8<0x6B, MRMSrcReg,                       // GR32 = GR32*I8
                     (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                     "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR32:$dst, EFLAGS,
                           (X86smul_flag GR32:$src1, i32immSExt8:$src2))]>;

// Memory-Integer Signed Integer Multiply
def IMUL16rmi  : Ii16<0x69, MRMSrcMem,                     // GR16 = [mem16]*I16
                      (outs GR16:$dst), (ins i16mem:$src1, i16imm:$src2),
                      "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR16:$dst, EFLAGS,
                            (X86smul_flag (load addr:$src1), imm:$src2))]>,
                 OpSize;
def IMUL32rmi  : Ii32<0x69, MRMSrcMem,                     // GR32 = [mem32]*I32
                      (outs GR32:$dst), (ins i32mem:$src1, i32imm:$src2),
                      "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR32:$dst, EFLAGS,
                            (X86smul_flag (load addr:$src1), imm:$src2))]>;
def IMUL16rmi8 : Ii8<0x6B, MRMSrcMem,                       // GR16 = [mem16]*I8
                     (outs GR16:$dst), (ins i16mem:$src1, i16i8imm :$src2),
                     "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR16:$dst, EFLAGS,
                           (X86smul_flag (load addr:$src1),
                                         i16immSExt8:$src2))]>, OpSize;
def IMUL32rmi8 : Ii8<0x6B, MRMSrcMem,                       // GR32 = [mem32]*I8
                     (outs GR32:$dst), (ins i32mem:$src1, i32i8imm: $src2),
                     "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR32:$dst, EFLAGS,
                           (X86smul_flag (load addr:$src1),
                                         i32immSExt8:$src2))]>;
} // Defs = [EFLAGS]

//===----------------------------------------------------------------------===//
// Test instructions are just like AND, except they don't generate a result.
//
let Defs = [EFLAGS] in {
let isCommutable = 1 in {   // TEST X, Y   --> TEST Y, X
def TEST8rr  : I<0x84, MRMSrcReg, (outs),  (ins GR8:$src1, GR8:$src2),
                     "test{b}\t{$src2, $src1|$src1, $src2}",
                     [(set EFLAGS, (X86cmp (and_su GR8:$src1, GR8:$src2), 0))]>;
def TEST16rr : I<0x85, MRMSrcReg, (outs),  (ins GR16:$src1, GR16:$src2),
                     "test{w}\t{$src2, $src1|$src1, $src2}",
                     [(set EFLAGS, (X86cmp (and_su GR16:$src1, GR16:$src2),
                      0))]>,
                 OpSize;
def TEST32rr : I<0x85, MRMSrcReg, (outs),  (ins GR32:$src1, GR32:$src2),
                     "test{l}\t{$src2, $src1|$src1, $src2}",
                     [(set EFLAGS, (X86cmp (and_su GR32:$src1, GR32:$src2),
                      0))]>;
}

def TEST8i8  : Ii8<0xA8, RawFrm, (outs), (ins i8imm:$src),
                   "test{b}\t{$src, %al|%al, $src}", []>;
def TEST16i16 : Ii16<0xA9, RawFrm, (outs), (ins i16imm:$src),
                     "test{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def TEST32i32 : Ii32<0xA9, RawFrm, (outs), (ins i32imm:$src),
                     "test{l}\t{$src, %eax|%eax, $src}", []>;

def TEST8rm  : I<0x84, MRMSrcMem, (outs),  (ins GR8 :$src1, i8mem :$src2),
                     "test{b}\t{$src2, $src1|$src1, $src2}",
                     [(set EFLAGS, (X86cmp (and GR8:$src1, (loadi8 addr:$src2)),
                       0))]>;
def TEST16rm : I<0x85, MRMSrcMem, (outs),  (ins GR16:$src1, i16mem:$src2),
                     "test{w}\t{$src2, $src1|$src1, $src2}",
                     [(set EFLAGS, (X86cmp (and GR16:$src1,
                                         (loadi16 addr:$src2)), 0))]>, OpSize;
def TEST32rm : I<0x85, MRMSrcMem, (outs),  (ins GR32:$src1, i32mem:$src2),
                     "test{l}\t{$src2, $src1|$src1, $src2}",
                     [(set EFLAGS, (X86cmp (and GR32:$src1,
                                                (loadi32 addr:$src2)), 0))]>;

def TEST8ri  : Ii8 <0xF6, MRM0r,                     // flags = GR8  & imm8
                    (outs),  (ins GR8:$src1, i8imm:$src2),
                    "test{b}\t{$src2, $src1|$src1, $src2}",
                    [(set EFLAGS, (X86cmp (and_su GR8:$src1, imm:$src2), 0))]>;
def TEST16ri : Ii16<0xF7, MRM0r,                     // flags = GR16 & imm16
                    (outs),  (ins GR16:$src1, i16imm:$src2),
                    "test{w}\t{$src2, $src1|$src1, $src2}",
                    [(set EFLAGS, (X86cmp (and_su GR16:$src1, imm:$src2), 0))]>,
                    OpSize;
def TEST32ri : Ii32<0xF7, MRM0r,                     // flags = GR32 & imm32
                    (outs),  (ins GR32:$src1, i32imm:$src2),
                    "test{l}\t{$src2, $src1|$src1, $src2}",
                    [(set EFLAGS, (X86cmp (and_su GR32:$src1, imm:$src2), 0))]>;

def TEST8mi  : Ii8 <0xF6, MRM0m,                   // flags = [mem8]  & imm8
                    (outs), (ins i8mem:$src1, i8imm:$src2),
                    "test{b}\t{$src2, $src1|$src1, $src2}",
                    [(set EFLAGS, (X86cmp (and (loadi8 addr:$src1), imm:$src2),
                     0))]>;
def TEST16mi : Ii16<0xF7, MRM0m,                   // flags = [mem16] & imm16
                    (outs), (ins i16mem:$src1, i16imm:$src2),
                    "test{w}\t{$src2, $src1|$src1, $src2}",
                    [(set EFLAGS, (X86cmp (and (loadi16 addr:$src1), imm:$src2),
                     0))]>, OpSize;
def TEST32mi : Ii32<0xF7, MRM0m,                   // flags = [mem32] & imm32
                    (outs), (ins i32mem:$src1, i32imm:$src2),
                    "test{l}\t{$src2, $src1|$src1, $src2}",
                    [(set EFLAGS, (X86cmp (and (loadi32 addr:$src1), imm:$src2),
                     0))]>;
} // Defs = [EFLAGS]


// Condition code ops, incl. set if equal/not equal/...
let Defs = [EFLAGS], Uses = [AH], neverHasSideEffects = 1 in
def SAHF     : I<0x9E, RawFrm, (outs),  (ins), "sahf", []>;  // flags = AH
let Defs = [AH], Uses = [EFLAGS], neverHasSideEffects = 1 in
def LAHF     : I<0x9F, RawFrm, (outs),  (ins), "lahf", []>;  // AH = flags

let Uses = [EFLAGS] in {
// Use sbb to materialize carry bit.
let Defs = [EFLAGS], isCodeGenOnly = 1 in {
// FIXME: These are pseudo ops that should be replaced with Pat<> patterns.
// However, Pat<> can't replicate the destination reg into the inputs of the
// result.
// FIXME: Change these to have encoding Pseudo when X86MCCodeEmitter replaces
// X86CodeEmitter.
def SETB_C8r : I<0x18, MRMInitReg, (outs GR8:$dst), (ins), "",
                 [(set GR8:$dst, (X86setcc_c X86_COND_B, EFLAGS))]>;
def SETB_C16r : I<0x19, MRMInitReg, (outs GR16:$dst), (ins), "",
                 [(set GR16:$dst, (X86setcc_c X86_COND_B, EFLAGS))]>,
                OpSize;
def SETB_C32r : I<0x19, MRMInitReg, (outs GR32:$dst), (ins), "",
                 [(set GR32:$dst, (X86setcc_c X86_COND_B, EFLAGS))]>;
} // isCodeGenOnly

def SETEr    : I<0x94, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "sete\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_E, EFLAGS))]>,
               TB;                        // GR8 = ==
def SETEm    : I<0x94, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "sete\t$dst",
                 [(store (X86setcc X86_COND_E, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = ==

def SETNEr   : I<0x95, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setne\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NE, EFLAGS))]>,
               TB;                        // GR8 = !=
def SETNEm   : I<0x95, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setne\t$dst",
                 [(store (X86setcc X86_COND_NE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = !=

def SETLr    : I<0x9C, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setl\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_L, EFLAGS))]>,
               TB;                        // GR8 = <  signed
def SETLm    : I<0x9C, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setl\t$dst",
                 [(store (X86setcc X86_COND_L, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <  signed

def SETGEr   : I<0x9D, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setge\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_GE, EFLAGS))]>,
               TB;                        // GR8 = >= signed
def SETGEm   : I<0x9D, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setge\t$dst",
                 [(store (X86setcc X86_COND_GE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >= signed

def SETLEr   : I<0x9E, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setle\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_LE, EFLAGS))]>,
               TB;                        // GR8 = <= signed
def SETLEm   : I<0x9E, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setle\t$dst",
                 [(store (X86setcc X86_COND_LE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <= signed

def SETGr    : I<0x9F, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setg\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_G, EFLAGS))]>,
               TB;                        // GR8 = >  signed
def SETGm    : I<0x9F, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setg\t$dst",
                 [(store (X86setcc X86_COND_G, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >  signed

def SETBr    : I<0x92, MRM0r,
                 (outs GR8   :$dst), (ins),
                 "setb\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_B, EFLAGS))]>,
               TB;                        // GR8 = <  unsign
def SETBm    : I<0x92, MRM0m,
                 (outs), (ins i8mem:$dst),
                 "setb\t$dst",
                 [(store (X86setcc X86_COND_B, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <  unsign

def SETAEr   : I<0x93, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setae\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_AE, EFLAGS))]>,
               TB;                        // GR8 = >= unsign
def SETAEm   : I<0x93, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setae\t$dst",
                 [(store (X86setcc X86_COND_AE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >= unsign

def SETBEr   : I<0x96, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setbe\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_BE, EFLAGS))]>,
               TB;                        // GR8 = <= unsign
def SETBEm   : I<0x96, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setbe\t$dst",
                 [(store (X86setcc X86_COND_BE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <= unsign

def SETAr    : I<0x97, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "seta\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_A, EFLAGS))]>,
               TB;                        // GR8 = >  signed
def SETAm    : I<0x97, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "seta\t$dst",
                 [(store (X86setcc X86_COND_A, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >  signed

def SETSr    : I<0x98, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "sets\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_S, EFLAGS))]>,
               TB;                        // GR8 = <sign bit>
def SETSm    : I<0x98, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "sets\t$dst",
                 [(store (X86setcc X86_COND_S, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <sign bit>
def SETNSr   : I<0x99, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setns\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NS, EFLAGS))]>,
               TB;                        // GR8 = !<sign bit>
def SETNSm   : I<0x99, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setns\t$dst",
                 [(store (X86setcc X86_COND_NS, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = !<sign bit>

def SETPr    : I<0x9A, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setp\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_P, EFLAGS))]>,
               TB;                        // GR8 = parity
def SETPm    : I<0x9A, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setp\t$dst",
                 [(store (X86setcc X86_COND_P, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = parity
def SETNPr   : I<0x9B, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setnp\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NP, EFLAGS))]>,
               TB;                        // GR8 = not parity
def SETNPm   : I<0x9B, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setnp\t$dst",
                 [(store (X86setcc X86_COND_NP, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = not parity

def SETOr    : I<0x90, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "seto\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_O, EFLAGS))]>,
               TB;                        // GR8 = overflow
def SETOm    : I<0x90, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "seto\t$dst",
                 [(store (X86setcc X86_COND_O, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = overflow
def SETNOr   : I<0x91, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setno\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NO, EFLAGS))]>,
               TB;                        // GR8 = not overflow
def SETNOm   : I<0x91, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setno\t$dst",
                 [(store (X86setcc X86_COND_NO, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = not overflow
} // Uses = [EFLAGS]


// Integer comparisons
let Defs = [EFLAGS] in {
def CMP8i8 : Ii8<0x3C, RawFrm, (outs), (ins i8imm:$src),
                 "cmp{b}\t{$src, %al|%al, $src}", []>;
def CMP16i16 : Ii16<0x3D, RawFrm, (outs), (ins i16imm:$src),
                    "cmp{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def CMP32i32 : Ii32<0x3D, RawFrm, (outs), (ins i32imm:$src),
                    "cmp{l}\t{$src, %eax|%eax, $src}", []>;

def CMP8rr  : I<0x38, MRMDestReg,
                (outs), (ins GR8 :$src1, GR8 :$src2),
                "cmp{b}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp GR8:$src1, GR8:$src2))]>;
def CMP16rr : I<0x39, MRMDestReg,
                (outs), (ins GR16:$src1, GR16:$src2),
                "cmp{w}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp GR16:$src1, GR16:$src2))]>, OpSize;
def CMP32rr : I<0x39, MRMDestReg,
                (outs), (ins GR32:$src1, GR32:$src2),
                "cmp{l}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp GR32:$src1, GR32:$src2))]>;
def CMP8mr  : I<0x38, MRMDestMem,
                (outs), (ins i8mem :$src1, GR8 :$src2),
                "cmp{b}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp (loadi8 addr:$src1), GR8:$src2))]>;
def CMP16mr : I<0x39, MRMDestMem,
                (outs), (ins i16mem:$src1, GR16:$src2),
                "cmp{w}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp (loadi16 addr:$src1), GR16:$src2))]>,
                 OpSize;
def CMP32mr : I<0x39, MRMDestMem,
                (outs), (ins i32mem:$src1, GR32:$src2),
                "cmp{l}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp (loadi32 addr:$src1), GR32:$src2))]>;
def CMP8rm  : I<0x3A, MRMSrcMem,
                (outs), (ins GR8 :$src1, i8mem :$src2),
                "cmp{b}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp GR8:$src1, (loadi8 addr:$src2)))]>;
def CMP16rm : I<0x3B, MRMSrcMem,
                (outs), (ins GR16:$src1, i16mem:$src2),
                "cmp{w}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp GR16:$src1, (loadi16 addr:$src2)))]>,
                 OpSize;
def CMP32rm : I<0x3B, MRMSrcMem,
                (outs), (ins GR32:$src1, i32mem:$src2),
                "cmp{l}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp GR32:$src1, (loadi32 addr:$src2)))]>;

// These are alternate spellings for use by the disassembler, we mark them as
// code gen only to ensure they aren't matched by the assembler.
let isCodeGenOnly = 1 in {
  def CMP8rr_alt : I<0x3A, MRMSrcReg, (outs), (ins GR8:$src1, GR8:$src2),
                    "cmp{b}\t{$src2, $src1|$src1, $src2}", []>;
  def CMP16rr_alt : I<0x3B, MRMSrcReg, (outs), (ins GR16:$src1, GR16:$src2),
                     "cmp{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize;
  def CMP32rr_alt : I<0x3B, MRMSrcReg, (outs), (ins GR32:$src1, GR32:$src2),
                     "cmp{l}\t{$src2, $src1|$src1, $src2}", []>;
}

def CMP8ri  : Ii8<0x80, MRM7r,
                  (outs), (ins GR8:$src1, i8imm:$src2),
                  "cmp{b}\t{$src2, $src1|$src1, $src2}",
                  [(set EFLAGS, (X86cmp GR8:$src1, imm:$src2))]>;
def CMP16ri : Ii16<0x81, MRM7r,
                   (outs), (ins GR16:$src1, i16imm:$src2),
                   "cmp{w}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp GR16:$src1, imm:$src2))]>, OpSize;
def CMP32ri : Ii32<0x81, MRM7r,
                   (outs), (ins GR32:$src1, i32imm:$src2),
                   "cmp{l}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp GR32:$src1, imm:$src2))]>;
def CMP8mi  : Ii8 <0x80, MRM7m,
                   (outs), (ins i8mem :$src1, i8imm :$src2),
                   "cmp{b}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp (loadi8 addr:$src1), imm:$src2))]>;
def CMP16mi : Ii16<0x81, MRM7m,
                   (outs), (ins i16mem:$src1, i16imm:$src2),
                   "cmp{w}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp (loadi16 addr:$src1), imm:$src2))]>,
                   OpSize;
def CMP32mi : Ii32<0x81, MRM7m,
                   (outs), (ins i32mem:$src1, i32imm:$src2),
                   "cmp{l}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp (loadi32 addr:$src1), imm:$src2))]>;
def CMP16ri8 : Ii8<0x83, MRM7r,
                   (outs), (ins GR16:$src1, i16i8imm:$src2),
                   "cmp{w}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp GR16:$src1, i16immSExt8:$src2))]>,
                    OpSize;
def CMP16mi8 : Ii8<0x83, MRM7m,
                   (outs), (ins i16mem:$src1, i16i8imm:$src2),
                   "cmp{w}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp (loadi16 addr:$src1),
                                         i16immSExt8:$src2))]>, OpSize;
def CMP32mi8 : Ii8<0x83, MRM7m,
                   (outs), (ins i32mem:$src1, i32i8imm:$src2),
                   "cmp{l}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp (loadi32 addr:$src1),
                                         i32immSExt8:$src2))]>;
def CMP32ri8 : Ii8<0x83, MRM7r,
                   (outs), (ins GR32:$src1, i32i8imm:$src2),
                   "cmp{l}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp GR32:$src1, i32immSExt8:$src2))]>;
} // Defs = [EFLAGS]

// Bit tests.
// TODO: BTC, BTR, and BTS
let Defs = [EFLAGS] in {
def BT16rr : I<0xA3, MRMDestReg, (outs), (ins GR16:$src1, GR16:$src2),
               "bt{w}\t{$src2, $src1|$src1, $src2}",
               [(set EFLAGS, (X86bt GR16:$src1, GR16:$src2))]>, OpSize, TB;
def BT32rr : I<0xA3, MRMDestReg, (outs), (ins GR32:$src1, GR32:$src2),
               "bt{l}\t{$src2, $src1|$src1, $src2}",
               [(set EFLAGS, (X86bt GR32:$src1, GR32:$src2))]>, TB;

// Unlike with the register+register form, the memory+register form of the
// bt instruction does not ignore the high bits of the index. From ISel's
// perspective, this is pretty bizarre. Make these instructions disassembly
// only for now.

def BT16mr : I<0xA3, MRMDestMem, (outs), (ins i16mem:$src1, GR16:$src2),
               "bt{w}\t{$src2, $src1|$src1, $src2}", 
//               [(X86bt (loadi16 addr:$src1), GR16:$src2),
//                (implicit EFLAGS)]
               []
               >, OpSize, TB, Requires<[FastBTMem]>;
def BT32mr : I<0xA3, MRMDestMem, (outs), (ins i32mem:$src1, GR32:$src2),
               "bt{l}\t{$src2, $src1|$src1, $src2}", 
//               [(X86bt (loadi32 addr:$src1), GR32:$src2),
//                (implicit EFLAGS)]
               []
               >, TB, Requires<[FastBTMem]>;

def BT16ri8 : Ii8<0xBA, MRM4r, (outs), (ins GR16:$src1, i16i8imm:$src2),
                "bt{w}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt GR16:$src1, i16immSExt8:$src2))]>,
                OpSize, TB;
def BT32ri8 : Ii8<0xBA, MRM4r, (outs), (ins GR32:$src1, i32i8imm:$src2),
                "bt{l}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt GR32:$src1, i32immSExt8:$src2))]>, TB;
// Note that these instructions don't need FastBTMem because that
// only applies when the other operand is in a register. When it's
// an immediate, bt is still fast.
def BT16mi8 : Ii8<0xBA, MRM4m, (outs), (ins i16mem:$src1, i16i8imm:$src2),
                "bt{w}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt (loadi16 addr:$src1), i16immSExt8:$src2))
                 ]>, OpSize, TB;
def BT32mi8 : Ii8<0xBA, MRM4m, (outs), (ins i32mem:$src1, i32i8imm:$src2),
                "bt{l}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt (loadi32 addr:$src1), i32immSExt8:$src2))
                 ]>, TB;

def BTC16rr : I<0xBB, MRMDestReg, (outs), (ins GR16:$src1, GR16:$src2),
                "btc{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTC32rr : I<0xBB, MRMDestReg, (outs), (ins GR32:$src1, GR32:$src2),
                "btc{l}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTC16mr : I<0xBB, MRMDestMem, (outs), (ins i16mem:$src1, GR16:$src2),
                "btc{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTC32mr : I<0xBB, MRMDestMem, (outs), (ins i32mem:$src1, GR32:$src2),
                "btc{l}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTC16ri8 : Ii8<0xBA, MRM7r, (outs), (ins GR16:$src1, i16i8imm:$src2),
                    "btc{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTC32ri8 : Ii8<0xBA, MRM7r, (outs), (ins GR32:$src1, i32i8imm:$src2),
                    "btc{l}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTC16mi8 : Ii8<0xBA, MRM7m, (outs), (ins i16mem:$src1, i16i8imm:$src2),
                    "btc{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTC32mi8 : Ii8<0xBA, MRM7m, (outs), (ins i32mem:$src1, i32i8imm:$src2),
                    "btc{l}\t{$src2, $src1|$src1, $src2}", []>, TB;

def BTR16rr : I<0xB3, MRMDestReg, (outs), (ins GR16:$src1, GR16:$src2),
                "btr{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTR32rr : I<0xB3, MRMDestReg, (outs), (ins GR32:$src1, GR32:$src2),
                "btr{l}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTR16mr : I<0xB3, MRMDestMem, (outs), (ins i16mem:$src1, GR16:$src2),
                "btr{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTR32mr : I<0xB3, MRMDestMem, (outs), (ins i32mem:$src1, GR32:$src2),
                "btr{l}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTR16ri8 : Ii8<0xBA, MRM6r, (outs), (ins GR16:$src1, i16i8imm:$src2),
                    "btr{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTR32ri8 : Ii8<0xBA, MRM6r, (outs), (ins GR32:$src1, i32i8imm:$src2),
                    "btr{l}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTR16mi8 : Ii8<0xBA, MRM6m, (outs), (ins i16mem:$src1, i16i8imm:$src2),
                    "btr{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTR32mi8 : Ii8<0xBA, MRM6m, (outs), (ins i32mem:$src1, i32i8imm:$src2),
                    "btr{l}\t{$src2, $src1|$src1, $src2}", []>, TB;

def BTS16rr : I<0xAB, MRMDestReg, (outs), (ins GR16:$src1, GR16:$src2),
                "bts{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTS32rr : I<0xAB, MRMDestReg, (outs), (ins GR32:$src1, GR32:$src2),
                "bts{l}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTS16mr : I<0xAB, MRMDestMem, (outs), (ins i16mem:$src1, GR16:$src2),
                "bts{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTS32mr : I<0xAB, MRMDestMem, (outs), (ins i32mem:$src1, GR32:$src2),
                "bts{l}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTS16ri8 : Ii8<0xBA, MRM5r, (outs), (ins GR16:$src1, i16i8imm:$src2),
                    "bts{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTS32ri8 : Ii8<0xBA, MRM5r, (outs), (ins GR32:$src1, i32i8imm:$src2),
                    "bts{l}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTS16mi8 : Ii8<0xBA, MRM5m, (outs), (ins i16mem:$src1, i16i8imm:$src2),
                    "bts{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize, TB;
def BTS32mi8 : Ii8<0xBA, MRM5m, (outs), (ins i32mem:$src1, i32i8imm:$src2),
                    "bts{l}\t{$src2, $src1|$src1, $src2}", []>, TB;
} // Defs = [EFLAGS]

// Sign/Zero extenders
// Use movsbl intead of movsbw; we don't care about the high 16 bits
// of the register here. This has a smaller encoding and avoids a
// partial-register update.  Actual movsbw included for the disassembler.
def MOVSX16rr8W : I<0xBE, MRMSrcReg, (outs GR16:$dst), (ins GR8:$src),
                    "movs{bw|x}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def MOVSX16rm8W : I<0xBE, MRMSrcMem, (outs GR16:$dst), (ins i8mem:$src),
                    "movs{bw|x}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def MOVSX16rr8 : I<0xBE, MRMSrcReg, (outs GR16:$dst), (ins GR8 :$src),
                   "", [(set GR16:$dst, (sext GR8:$src))]>, TB;
def MOVSX16rm8 : I<0xBE, MRMSrcMem, (outs GR16:$dst), (ins i8mem :$src),
                   "", [(set GR16:$dst, (sextloadi16i8 addr:$src))]>, TB;
def MOVSX32rr8 : I<0xBE, MRMSrcReg, (outs GR32:$dst), (ins GR8 :$src),
                   "movs{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sext GR8:$src))]>, TB;
def MOVSX32rm8 : I<0xBE, MRMSrcMem, (outs GR32:$dst), (ins i8mem :$src),
                   "movs{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sextloadi32i8 addr:$src))]>, TB;
def MOVSX32rr16: I<0xBF, MRMSrcReg, (outs GR32:$dst), (ins GR16:$src),
                   "movs{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sext GR16:$src))]>, TB;
def MOVSX32rm16: I<0xBF, MRMSrcMem, (outs GR32:$dst), (ins i16mem:$src),
                   "movs{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sextloadi32i16 addr:$src))]>, TB;

// Use movzbl intead of movzbw; we don't care about the high 16 bits
// of the register here. This has a smaller encoding and avoids a
// partial-register update.  Actual movzbw included for the disassembler.
def MOVZX16rr8W : I<0xB6, MRMSrcReg, (outs GR16:$dst), (ins GR8:$src),
                    "movz{bw|x}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def MOVZX16rm8W : I<0xB6, MRMSrcMem, (outs GR16:$dst), (ins i8mem:$src),
                    "movz{bw|x}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;  
def MOVZX16rr8 : I<0xB6, MRMSrcReg, (outs GR16:$dst), (ins GR8 :$src),
                   "", [(set GR16:$dst, (zext GR8:$src))]>, TB;
def MOVZX16rm8 : I<0xB6, MRMSrcMem, (outs GR16:$dst), (ins i8mem :$src),
                   "", [(set GR16:$dst, (zextloadi16i8 addr:$src))]>, TB;
def MOVZX32rr8 : I<0xB6, MRMSrcReg, (outs GR32:$dst), (ins GR8 :$src),
                   "movz{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zext GR8:$src))]>, TB;
def MOVZX32rm8 : I<0xB6, MRMSrcMem, (outs GR32:$dst), (ins i8mem :$src),
                   "movz{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zextloadi32i8 addr:$src))]>, TB;
def MOVZX32rr16: I<0xB7, MRMSrcReg, (outs GR32:$dst), (ins GR16:$src),
                   "movz{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zext GR16:$src))]>, TB;
def MOVZX32rm16: I<0xB7, MRMSrcMem, (outs GR32:$dst), (ins i16mem:$src),
                   "movz{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zextloadi32i16 addr:$src))]>, TB;

// These are the same as the regular MOVZX32rr8 and MOVZX32rm8
// except that they use GR32_NOREX for the output operand register class
// instead of GR32. This allows them to operate on h registers on x86-64.
def MOVZX32_NOREXrr8 : I<0xB6, MRMSrcReg,
                         (outs GR32_NOREX:$dst), (ins GR8:$src),
                         "movz{bl|x}\t{$src, $dst|$dst, $src}  # NOREX",
                         []>, TB;
let mayLoad = 1 in
def MOVZX32_NOREXrm8 : I<0xB6, MRMSrcMem,
                         (outs GR32_NOREX:$dst), (ins i8mem:$src),
                         "movz{bl|x}\t{$src, $dst|$dst, $src}  # NOREX",
                         []>, TB;

let neverHasSideEffects = 1 in {
  let Defs = [AX], Uses = [AL] in
  def CBW : I<0x98, RawFrm, (outs), (ins),
              "{cbtw|cbw}", []>, OpSize;   // AX = signext(AL)
  let Defs = [EAX], Uses = [AX] in
  def CWDE : I<0x98, RawFrm, (outs), (ins),
              "{cwtl|cwde}", []>;   // EAX = signext(AX)

  let Defs = [AX,DX], Uses = [AX] in
  def CWD : I<0x99, RawFrm, (outs), (ins),
              "{cwtd|cwd}", []>, OpSize; // DX:AX = signext(AX)
  let Defs = [EAX,EDX], Uses = [EAX] in
  def CDQ : I<0x99, RawFrm, (outs), (ins),
              "{cltd|cdq}", []>; // EDX:EAX = signext(EAX)
}

//===----------------------------------------------------------------------===//
// Alias Instructions
//===----------------------------------------------------------------------===//

// Alias instructions that map movr0 to xor.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
// FIXME: Set encoding to pseudo.
let Defs = [EFLAGS], isReMaterializable = 1, isAsCheapAsAMove = 1,
    isCodeGenOnly = 1 in {
def MOV8r0   : I<0x30, MRMInitReg, (outs GR8 :$dst), (ins), "",
                 [(set GR8:$dst, 0)]>;

// We want to rewrite MOV16r0 in terms of MOV32r0, because it's a smaller
// encoding and avoids a partial-register update sometimes, but doing so
// at isel time interferes with rematerialization in the current register
// allocator. For now, this is rewritten when the instruction is lowered
// to an MCInst.
def MOV16r0   : I<0x31, MRMInitReg, (outs GR16:$dst), (ins),
                 "",
                 [(set GR16:$dst, 0)]>, OpSize;
                 
// FIXME: Set encoding to pseudo.
def MOV32r0  : I<0x31, MRMInitReg, (outs GR32:$dst), (ins), "",
                 [(set GR32:$dst, 0)]>;
}

//===----------------------------------------------------------------------===//
// Thread Local Storage Instructions
//

// ELF TLS Support
// All calls clobber the non-callee saved registers. ESP is marked as
// a use to prevent stack-pointer assignments that appear immediately
// before calls from potentially appearing dead.
let Defs = [EAX, ECX, EDX, FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0,
            MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
            XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
            XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
    Uses = [ESP] in
def TLS_addr32 : I<0, Pseudo, (outs), (ins i32mem:$sym),
                  "leal\t$sym, %eax; "
                  "call\t___tls_get_addr@PLT",
                  [(X86tlsaddr tls32addr:$sym)]>,
                  Requires<[In32BitMode]>;

// Darwin TLS Support
// For i386, the address of the thunk is passed on the stack, on return the 
// address of the variable is in %eax.  %ecx is trashed during the function 
// call.  All other registers are preserved.
let Defs = [EAX, ECX],
    Uses = [ESP],
    usesCustomInserter = 1 in
def TLSCall_32 : I<0, Pseudo, (outs), (ins i32mem:$sym),
                "# TLSCall_32",
                [(X86TLSCall addr:$sym)]>,
                Requires<[In32BitMode]>;
                

//===----------------------------------------------------------------------===//
// Atomic support
//

// Memory barriers

// TODO: Get this to fold the constant into the instruction.           
def OR32mrLocked  : I<0x09, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$zero),
                      "lock\n\t"
                      "or{l}\t{$zero, $dst|$dst, $zero}",
                      []>, Requires<[In32BitMode]>, LOCK;

let hasSideEffects = 1 in {
def Int_MemBarrier : I<0, Pseudo, (outs), (ins),
                     "#MEMBARRIER",
                     [(X86MemBarrier)]>, Requires<[HasSSE2]>;
}

// Atomic swap. These are just normal xchg instructions. But since a memory
// operand is referenced, the atomicity is ensured.
let Constraints = "$val = $dst" in {
def XCHG32rm : I<0x87, MRMSrcMem, (outs GR32:$dst), 
                 (ins GR32:$val, i32mem:$ptr),
               "xchg{l}\t{$val, $ptr|$ptr, $val}", 
               [(set GR32:$dst, (atomic_swap_32 addr:$ptr, GR32:$val))]>;
def XCHG16rm : I<0x87, MRMSrcMem, (outs GR16:$dst), 
                 (ins GR16:$val, i16mem:$ptr),
               "xchg{w}\t{$val, $ptr|$ptr, $val}", 
               [(set GR16:$dst, (atomic_swap_16 addr:$ptr, GR16:$val))]>, 
                OpSize;
def XCHG8rm  : I<0x86, MRMSrcMem, (outs GR8:$dst), (ins GR8:$val, i8mem:$ptr),
               "xchg{b}\t{$val, $ptr|$ptr, $val}", 
               [(set GR8:$dst, (atomic_swap_8 addr:$ptr, GR8:$val))]>;

def XCHG32rr : I<0x87, MRMSrcReg, (outs GR32:$dst), (ins GR32:$val, GR32:$src),
                 "xchg{l}\t{$val, $src|$src, $val}", []>;
def XCHG16rr : I<0x87, MRMSrcReg, (outs GR16:$dst), (ins GR16:$val, GR16:$src),
                 "xchg{w}\t{$val, $src|$src, $val}", []>, OpSize;
def XCHG8rr : I<0x86, MRMSrcReg, (outs GR8:$dst), (ins GR8:$val, GR8:$src),
                "xchg{b}\t{$val, $src|$src, $val}", []>;
}

def XCHG16ar : I<0x90, AddRegFrm, (outs), (ins GR16:$src),
                  "xchg{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def XCHG32ar : I<0x90, AddRegFrm, (outs), (ins GR32:$src),
                  "xchg{l}\t{$src, %eax|%eax, $src}", []>;

// Atomic compare and swap.
let Defs = [EAX, EFLAGS], Uses = [EAX] in {
def LCMPXCHG32 : I<0xB1, MRMDestMem, (outs), (ins i32mem:$ptr, GR32:$swap),
               "lock\n\t"
               "cmpxchg{l}\t{$swap, $ptr|$ptr, $swap}",
               [(X86cas addr:$ptr, GR32:$swap, 4)]>, TB, LOCK;
}
let Defs = [EAX, EDX, EFLAGS], Uses = [EAX, EBX, ECX, EDX] in {
def LCMPXCHG8B : I<0xC7, MRM1m, (outs), (ins i64mem:$ptr),
               "lock\n\t"
               "cmpxchg8b\t$ptr",
               [(X86cas8 addr:$ptr)]>, TB, LOCK;
}

let Defs = [AX, EFLAGS], Uses = [AX] in {
def LCMPXCHG16 : I<0xB1, MRMDestMem, (outs), (ins i16mem:$ptr, GR16:$swap),
               "lock\n\t"
               "cmpxchg{w}\t{$swap, $ptr|$ptr, $swap}",
               [(X86cas addr:$ptr, GR16:$swap, 2)]>, TB, OpSize, LOCK;
}
let Defs = [AL, EFLAGS], Uses = [AL] in {
def LCMPXCHG8 : I<0xB0, MRMDestMem, (outs), (ins i8mem:$ptr, GR8:$swap),
               "lock\n\t"
               "cmpxchg{b}\t{$swap, $ptr|$ptr, $swap}",
               [(X86cas addr:$ptr, GR8:$swap, 1)]>, TB, LOCK;
}

// Atomic exchange and add
let Constraints = "$val = $dst", Defs = [EFLAGS] in {
def LXADD32 : I<0xC1, MRMSrcMem, (outs GR32:$dst), (ins GR32:$val, i32mem:$ptr),
               "lock\n\t"
               "xadd{l}\t{$val, $ptr|$ptr, $val}",
               [(set GR32:$dst, (atomic_load_add_32 addr:$ptr, GR32:$val))]>,
                TB, LOCK;
def LXADD16 : I<0xC1, MRMSrcMem, (outs GR16:$dst), (ins GR16:$val, i16mem:$ptr),
               "lock\n\t"
               "xadd{w}\t{$val, $ptr|$ptr, $val}",
               [(set GR16:$dst, (atomic_load_add_16 addr:$ptr, GR16:$val))]>,
                TB, OpSize, LOCK;
def LXADD8  : I<0xC0, MRMSrcMem, (outs GR8:$dst), (ins GR8:$val, i8mem:$ptr),
               "lock\n\t"
               "xadd{b}\t{$val, $ptr|$ptr, $val}",
               [(set GR8:$dst, (atomic_load_add_8 addr:$ptr, GR8:$val))]>,
                TB, LOCK;
}

def XADD8rr : I<0xC0, MRMDestReg, (outs GR8:$dst), (ins GR8:$src),
                "xadd{b}\t{$src, $dst|$dst, $src}", []>, TB;
def XADD16rr : I<0xC1, MRMDestReg, (outs GR16:$dst), (ins GR16:$src),
                 "xadd{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def XADD32rr  : I<0xC1, MRMDestReg, (outs GR32:$dst), (ins GR32:$src),
                 "xadd{l}\t{$src, $dst|$dst, $src}", []>, TB;

let mayLoad = 1, mayStore = 1 in {
def XADD8rm   : I<0xC0, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src),
                 "xadd{b}\t{$src, $dst|$dst, $src}", []>, TB;
def XADD16rm  : I<0xC1, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src),
                 "xadd{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def XADD32rm  : I<0xC1, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                 "xadd{l}\t{$src, $dst|$dst, $src}", []>, TB;
}

def CMPXCHG8rr : I<0xB0, MRMDestReg, (outs GR8:$dst), (ins GR8:$src),
                   "cmpxchg{b}\t{$src, $dst|$dst, $src}", []>, TB;
def CMPXCHG16rr : I<0xB1, MRMDestReg, (outs GR16:$dst), (ins GR16:$src),
                    "cmpxchg{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def CMPXCHG32rr  : I<0xB1, MRMDestReg, (outs GR32:$dst), (ins GR32:$src),
                     "cmpxchg{l}\t{$src, $dst|$dst, $src}", []>, TB;

let mayLoad = 1, mayStore = 1 in {
def CMPXCHG8rm   : I<0xB0, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src),
                     "cmpxchg{b}\t{$src, $dst|$dst, $src}", []>, TB;
def CMPXCHG16rm  : I<0xB1, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src),
                     "cmpxchg{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def CMPXCHG32rm  : I<0xB1, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                     "cmpxchg{l}\t{$src, $dst|$dst, $src}", []>, TB;
}

let Defs = [EAX, EDX, EFLAGS], Uses = [EAX, EBX, ECX, EDX] in
def CMPXCHG8B : I<0xC7, MRM1m, (outs), (ins i64mem:$dst),
                  "cmpxchg8b\t$dst", []>, TB;

// Optimized codegen when the non-memory output is not used.
// FIXME: Use normal add / sub instructions and add lock prefix dynamically.
let Defs = [EFLAGS], mayLoad = 1, mayStore = 1 in {
def LOCK_ADD8mr  : I<0x00, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src2),
                    "lock\n\t"
                    "add{b}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_ADD16mr  : I<0x01, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                    "lock\n\t"
                    "add{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize, LOCK;
def LOCK_ADD32mr  : I<0x01, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                    "lock\n\t"
                    "add{l}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_ADD8mi   : Ii8<0x80, MRM0m, (outs), (ins i8mem :$dst, i8imm :$src2),
                    "lock\n\t"
                    "add{b}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_ADD16mi  : Ii16<0x81, MRM0m, (outs), (ins i16mem:$dst, i16imm:$src2),
                    "lock\n\t"
                     "add{w}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_ADD32mi  : Ii32<0x81, MRM0m, (outs), (ins i32mem:$dst, i32imm:$src2),
                    "lock\n\t"
                    "add{l}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_ADD16mi8 : Ii8<0x83, MRM0m, (outs), (ins i16mem:$dst, i16i8imm :$src2),
                    "lock\n\t"
                    "add{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize, LOCK;
def LOCK_ADD32mi8 : Ii8<0x83, MRM0m, (outs), (ins i32mem:$dst, i32i8imm :$src2),
                    "lock\n\t"
                    "add{l}\t{$src2, $dst|$dst, $src2}", []>, LOCK;

def LOCK_INC8m  : I<0xFE, MRM0m, (outs), (ins i8mem :$dst),
                    "lock\n\t"
                    "inc{b}\t$dst", []>, LOCK;
def LOCK_INC16m : I<0xFF, MRM0m, (outs), (ins i16mem:$dst),
                    "lock\n\t"
                    "inc{w}\t$dst", []>, OpSize, LOCK;
def LOCK_INC32m : I<0xFF, MRM0m, (outs), (ins i32mem:$dst),
                    "lock\n\t"
                    "inc{l}\t$dst", []>, LOCK;

def LOCK_SUB8mr   : I<0x28, MRMDestMem, (outs), (ins i8mem :$dst, GR8 :$src2),
                    "lock\n\t"
                    "sub{b}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_SUB16mr  : I<0x29, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                    "lock\n\t"
                    "sub{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize, LOCK;
def LOCK_SUB32mr  : I<0x29, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2), 
                    "lock\n\t"
                    "sub{l}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_SUB8mi   : Ii8<0x80, MRM5m, (outs), (ins i8mem :$dst, i8imm:$src2), 
                    "lock\n\t"
                    "sub{b}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_SUB16mi  : Ii16<0x81, MRM5m, (outs), (ins i16mem:$dst, i16imm:$src2), 
                    "lock\n\t"
                    "sub{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize, LOCK;
def LOCK_SUB32mi  : Ii32<0x81, MRM5m, (outs), (ins i32mem:$dst, i32imm:$src2), 
                    "lock\n\t"
                     "sub{l}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_SUB16mi8 : Ii8<0x83, MRM5m, (outs), (ins i16mem:$dst, i16i8imm :$src2),
                    "lock\n\t"
                     "sub{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize, LOCK;
def LOCK_SUB32mi8 : Ii8<0x83, MRM5m, (outs), (ins i32mem:$dst, i32i8imm :$src2),
                    "lock\n\t"
                     "sub{l}\t{$src2, $dst|$dst, $src2}", []>, LOCK;

def LOCK_DEC8m  : I<0xFE, MRM1m, (outs), (ins i8mem :$dst),
                    "lock\n\t"
                    "dec{b}\t$dst", []>, LOCK;
def LOCK_DEC16m : I<0xFF, MRM1m, (outs), (ins i16mem:$dst),
                    "lock\n\t"
                    "dec{w}\t$dst", []>, OpSize, LOCK;
def LOCK_DEC32m : I<0xFF, MRM1m, (outs), (ins i32mem:$dst),
                    "lock\n\t"
                    "dec{l}\t$dst", []>, LOCK;
}

// Atomic exchange, and, or, xor
let Constraints = "$val = $dst", Defs = [EFLAGS],
                  usesCustomInserter = 1 in {
def ATOMAND32 : I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMAND32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_and_32 addr:$ptr, GR32:$val))]>;
def ATOMOR32 : I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMOR32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_or_32 addr:$ptr, GR32:$val))]>;
def ATOMXOR32 : I<0, Pseudo,(outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMXOR32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_xor_32 addr:$ptr, GR32:$val))]>;
def ATOMNAND32 : I<0, Pseudo,(outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMNAND32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_nand_32 addr:$ptr, GR32:$val))]>;
def ATOMMIN32: I<0, Pseudo, (outs GR32:$dst), (ins i32mem:$ptr, GR32:$val),
               "#ATOMMIN32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_min_32 addr:$ptr, GR32:$val))]>;
def ATOMMAX32: I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMMAX32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_max_32 addr:$ptr, GR32:$val))]>;
def ATOMUMIN32: I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMUMIN32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_umin_32 addr:$ptr, GR32:$val))]>;
def ATOMUMAX32: I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMUMAX32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_umax_32 addr:$ptr, GR32:$val))]>;

def ATOMAND16 : I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMAND16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_and_16 addr:$ptr, GR16:$val))]>;
def ATOMOR16 : I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMOR16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_or_16 addr:$ptr, GR16:$val))]>;
def ATOMXOR16 : I<0, Pseudo,(outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMXOR16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_xor_16 addr:$ptr, GR16:$val))]>;
def ATOMNAND16 : I<0, Pseudo,(outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMNAND16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_nand_16 addr:$ptr, GR16:$val))]>;
def ATOMMIN16: I<0, Pseudo, (outs GR16:$dst), (ins i16mem:$ptr, GR16:$val),
               "#ATOMMIN16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_min_16 addr:$ptr, GR16:$val))]>;
def ATOMMAX16: I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMMAX16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_max_16 addr:$ptr, GR16:$val))]>;
def ATOMUMIN16: I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMUMIN16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_umin_16 addr:$ptr, GR16:$val))]>;
def ATOMUMAX16: I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMUMAX16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_umax_16 addr:$ptr, GR16:$val))]>;

def ATOMAND8 : I<0, Pseudo, (outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMAND8 PSEUDO!", 
               [(set GR8:$dst, (atomic_load_and_8 addr:$ptr, GR8:$val))]>;
def ATOMOR8 : I<0, Pseudo, (outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMOR8 PSEUDO!", 
               [(set GR8:$dst, (atomic_load_or_8 addr:$ptr, GR8:$val))]>;
def ATOMXOR8 : I<0, Pseudo,(outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMXOR8 PSEUDO!", 
               [(set GR8:$dst, (atomic_load_xor_8 addr:$ptr, GR8:$val))]>;
def ATOMNAND8 : I<0, Pseudo,(outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMNAND8 PSEUDO!", 
               [(set GR8:$dst, (atomic_load_nand_8 addr:$ptr, GR8:$val))]>;
}

let Constraints = "$val1 = $dst1, $val2 = $dst2", 
                  Defs = [EFLAGS, EAX, EBX, ECX, EDX],
                  Uses = [EAX, EBX, ECX, EDX],
                  mayLoad = 1, mayStore = 1,
                  usesCustomInserter = 1 in {
def ATOMAND6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMAND6432 PSEUDO!", []>;
def ATOMOR6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMOR6432 PSEUDO!", []>;
def ATOMXOR6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMXOR6432 PSEUDO!", []>;
def ATOMNAND6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMNAND6432 PSEUDO!", []>;
def ATOMADD6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMADD6432 PSEUDO!", []>;
def ATOMSUB6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMSUB6432 PSEUDO!", []>;
def ATOMSWAP6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMSWAP6432 PSEUDO!", []>;
}

// Lock instruction prefix
def LOCK_PREFIX : I<0xF0, RawFrm, (outs),  (ins), "lock", []>;

// Repeat string operation instruction prefixes
// These uses the DF flag in the EFLAGS register to inc or dec ECX
let Defs = [ECX], Uses = [ECX,EFLAGS] in {
// Repeat (used with INS, OUTS, MOVS, LODS and STOS)
def REP_PREFIX : I<0xF3, RawFrm, (outs),  (ins), "rep", []>;
// Repeat while not equal (used with CMPS and SCAS)
def REPNE_PREFIX : I<0xF2, RawFrm, (outs),  (ins), "repne", []>;
}


// String manipulation instructions

def LODSB : I<0xAC, RawFrm, (outs), (ins), "lodsb", []>;
def LODSW : I<0xAD, RawFrm, (outs), (ins), "lodsw", []>, OpSize;
def LODSD : I<0xAD, RawFrm, (outs), (ins), "lods{l|d}", []>;

def OUTSB : I<0x6E, RawFrm, (outs), (ins), "outsb", []>;
def OUTSW : I<0x6F, RawFrm, (outs), (ins), "outsw", []>, OpSize;
def OUTSD : I<0x6F, RawFrm, (outs), (ins), "outs{l|d}", []>;


// Flag instructions
def CLC : I<0xF8, RawFrm, (outs), (ins), "clc", []>;
def STC : I<0xF9, RawFrm, (outs), (ins), "stc", []>;
def CLI : I<0xFA, RawFrm, (outs), (ins), "cli", []>;
def STI : I<0xFB, RawFrm, (outs), (ins), "sti", []>;
def CLD : I<0xFC, RawFrm, (outs), (ins), "cld", []>;
def STD : I<0xFD, RawFrm, (outs), (ins), "std", []>;
def CMC : I<0xF5, RawFrm, (outs), (ins), "cmc", []>;

def CLTS : I<0x06, RawFrm, (outs), (ins), "clts", []>, TB;

// Table lookup instructions
def XLAT : I<0xD7, RawFrm, (outs), (ins), "xlatb", []>;



//===----------------------------------------------------------------------===//
// Subsystems.
//===----------------------------------------------------------------------===//

// Floating Point Stack Support
include "X86InstrFPStack.td"

// X86-64 Support
include "X86Instr64bit.td"

include "X86InstrControl.td"

// SIMD support (SSE, MMX and AVX)
include "X86InstrFragmentsSIMD.td"

// FMA - Fused Multiply-Add support (requires FMA)
include "X86InstrFMA.td"

// SSE, MMX and 3DNow! vector support.
include "X86InstrSSE.td"
include "X86InstrMMX.td"
include "X86Instr3DNow.td"

include "X86InstrVMX.td"

// System instructions.
include "X86InstrSystem.td"

// Compiler Pseudo Instructions and Pat Patterns
include "X86InstrCompiler.td"

