//===- ARMInstrFormats.td - ARM Instruction Formats --*- tablegen -*---------=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// ARM Instruction Format Definitions.
//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<5> val> {
  bits<5> Value = val;
}

def Pseudo      : Format<1>;
def MulFrm      : Format<2>;
def MulSMLAW    : Format<3>;
def MulSMULW    : Format<4>;
def MulSMLA     : Format<5>;
def MulSMUL     : Format<6>;
def Branch      : Format<7>;
def BranchMisc  : Format<8>;

def DPRdIm      : Format<9>;
def DPRdReg     : Format<10>;
def DPRdSoReg   : Format<11>;
def DPRdMisc    : Format<12>;
def DPRnIm      : Format<13>;
def DPRnReg     : Format<14>;
def DPRnSoReg   : Format<15>;
def DPRIm       : Format<16>;
def DPRReg      : Format<17>;
def DPRSoReg    : Format<18>;
def DPRImS      : Format<19>;
def DPRRegS     : Format<20>;
def DPRSoRegS   : Format<21>;

def LdFrm       : Format<22>;
def StFrm       : Format<23>;

def ArithMisc   : Format<24>;
def ThumbFrm    : Format<25>;
def VFPFrm      : Format<26>;



//===----------------------------------------------------------------------===//

// ARM Instruction templates.
//

class InstARM<bits<4> opcod, AddrMode am, SizeFlagVal sz, IndexMode im,
              Format f, string cstr>
  : Instruction {
  let Namespace = "ARM";

  bits<4> Opcode = opcod;
  AddrMode AM = am;
  bits<4> AddrModeBits = AM.Value;
  
  SizeFlagVal SZ = sz;
  bits<3> SizeFlag = SZ.Value;

  IndexMode IM = im;
  bits<2> IndexModeBits = IM.Value;
  
  Format F = f;
  bits<5> Form = F.Value;
  
  let Constraints = cstr;
}

class PseudoInst<dag oops, dag iops, string asm, list<dag> pattern>
  : InstARM<0, AddrModeNone, SizeSpecial, IndexModeNone, Pseudo, ""> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
  let Pattern = pattern;
}

// Almost all ARM instructions are predicable.
class I<bits<4> opcod, dag oops, dag iops, AddrMode am, SizeFlagVal sz,
        IndexMode im, Format f, string opc, string asm, string cstr,
        list<dag> pattern>
  : InstARM<opcod, am, sz, im, f, cstr> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p));
  let AsmString   = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

// Same as I except it can optionally modify CPSR. Note it's modeled as
// an input operand since by default it's a zero register. It will
// become an implicit def once it's "flipped".
class sI<bits<4> opcod, dag oops, dag iops, AddrMode am, SizeFlagVal sz,
         IndexMode im, Format f, string opc, string asm, string cstr,
         list<dag> pattern>
  : InstARM<opcod, am, sz, im, f, cstr> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p, cc_out:$s));
  let AsmString   = !strconcat(opc, !strconcat("${p}${s}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

class AI<bits<4> opcod, dag oops, dag iops, Format f, string opc,
         string asm, list<dag> pattern>
  : I<opcod, oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, opc,
      asm,"",pattern>;
class AsI<bits<4> opcod, dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : sI<opcod, oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, opc,
       asm,"",pattern>;
class AI1<bits<4> opcod, dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<opcod, oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern>;
class AsI1<bits<4> opcod, dag oops, dag iops, Format f, string opc,
           string asm, list<dag> pattern>
  : sI<opcod, oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, opc,
       asm, "", pattern>;
class AI2<bits<4> opcod, dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<opcod, oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern>;
class AI3<bits<4> opcod, dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<opcod, oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern>;
class AI4<bits<4> opcod, dag oops, dag iops, Format f, string opc,
          string asm, list<dag> pattern>
  : I<opcod, oops, iops, AddrMode4, Size4Bytes, IndexModeNone, f, opc,
      asm, "", pattern>;
class AI1x2<bits<4> opcod, dag oops, dag iops, Format f, string opc,
            string asm, list<dag> pattern>
  : I<opcod, oops, iops, AddrMode1, Size8Bytes, IndexModeNone, f, opc,
      asm, "", pattern>;

// Pre-indexed ops
class AI2pr<bits<4> opcod, dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<opcod, oops, iops, AddrMode2, Size4Bytes, IndexModePre, f, opc,
      asm, cstr, pattern>;
class AI3pr<bits<4> opcod, dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<opcod, oops, iops, AddrMode3, Size4Bytes, IndexModePre, f, opc,
      asm, cstr, pattern>;

// Post-indexed ops
class AI2po<bits<4> opcod, dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<opcod, oops, iops, AddrMode2, Size4Bytes, IndexModePost, f, opc,
      asm, cstr,pattern>;
class AI3po<bits<4> opcod, dag oops, dag iops, Format f, string opc,
            string asm, string cstr, list<dag> pattern>
  : I<opcod, oops, iops, AddrMode3, Size4Bytes, IndexModePost, f, opc,
      asm, cstr,pattern>;


// Special cases.
class XI<bits<4> opcod, dag oops, dag iops, AddrMode am, SizeFlagVal sz,
         IndexMode im, Format f, string asm, string cstr, list<dag> pattern>
  : InstARM<opcod, am, sz, im, f, cstr> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

class AXI<bits<4> opcod, dag oops, dag iops, Format f, string asm,
          list<dag> pattern>
  : XI<opcod, oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, f, asm,
       "", pattern>;
class AXI1<bits<4> opcod, dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<opcod, oops, iops, AddrMode1, Size4Bytes, IndexModeNone, f, asm,
       "", pattern>;
class AXI2<bits<4> opcod, dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<opcod, oops, iops, AddrMode2, Size4Bytes, IndexModeNone, f, asm,
       "", pattern>;
class AXI3<bits<4> opcod, dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<opcod, oops, iops, AddrMode3, Size4Bytes, IndexModeNone, f, asm,
       "", pattern>;
class AXI4<bits<4> opcod, dag oops, dag iops, Format f, string asm,
           list<dag> pattern>
  : XI<opcod, oops, iops, AddrMode4, Size4Bytes, IndexModeNone, f, asm,
       "", pattern>;

class AXIx2<bits<4> opcod, dag oops, dag iops, Format f, string asm,
            list<dag> pattern>
  : XI<opcod, oops, iops, AddrModeNone, Size8Bytes, IndexModeNone, f, asm,
       "", pattern>;

// BR_JT instructions
class JTI<bits<4> opcod, dag oops, dag iops, string asm, list<dag> pattern>
  : XI<opcod, oops, iops, AddrModeNone, SizeSpecial, IndexModeNone, BranchMisc,
       asm, "", pattern>;
class JTI1<bits<4> opcod, dag oops, dag iops, string asm, list<dag> pattern>
  : XI<opcod, oops, iops, AddrMode1, SizeSpecial, IndexModeNone, BranchMisc,
       asm, "", pattern>;
class JTI2<bits<4> opcod, dag oops, dag iops, string asm, list<dag> pattern>
  : XI<opcod, oops, iops, AddrMode2, SizeSpecial, IndexModeNone, BranchMisc,
       asm, "", pattern>;


//===----------------------------------------------------------------------===//

// ARMPat - Same as Pat<>, but requires that the compiler be in ARM mode.
class ARMPat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsARM];
}
class ARMV5TEPat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsARM, HasV5TE];
}
class ARMV6Pat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsARM, HasV6];
}
