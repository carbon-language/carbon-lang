//===- X86InstrInfo.td - Describe the X86 Instruction Set -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 instruction set, defining the instructions, and
// properties of the instructions which are needed for code generation, machine
// code emission, and analysis.
//
//===----------------------------------------------------------------------===//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<5> val> {
  bits<5> Value = val;
}

def Pseudo     : Format<0>; def RawFrm     : Format<1>;
def AddRegFrm  : Format<2>; def MRMDestReg : Format<3>;
def MRMDestMem : Format<4>; def MRMSrcReg  : Format<5>;
def MRMSrcMem  : Format<6>;
def MRM0r  : Format<16>; def MRM1r  : Format<17>; def MRM2r  : Format<18>;
def MRM3r  : Format<19>; def MRM4r  : Format<20>; def MRM5r  : Format<21>;
def MRM6r  : Format<22>; def MRM7r  : Format<23>;
def MRM0m  : Format<24>; def MRM1m  : Format<25>; def MRM2m  : Format<26>;
def MRM3m  : Format<27>; def MRM4m  : Format<28>; def MRM5m  : Format<29>;
def MRM6m  : Format<30>; def MRM7m  : Format<31>;

// ImmType - This specifies the immediate type used by an instruction. This is
// part of the ad-hoc solution used to emit machine instruction encodings by our
// machine code emitter.
class ImmType<bits<2> val> {
  bits<2> Value = val;
}
def NoImm  : ImmType<0>;
def Imm8   : ImmType<1>;
def Imm16  : ImmType<2>;
def Imm32  : ImmType<3>;

// MemType - This specifies the immediate type used by an instruction. This is
// part of the ad-hoc solution used to emit machine instruction encodings by our
// machine code emitter.
class MemType<bits<3> val> {
  bits<3> Value = val;
}
def NoMem  : MemType<0>;
def Mem8   : MemType<1>;
def Mem16  : MemType<2>;
def Mem32  : MemType<3>;
def Mem64  : MemType<4>;
def Mem80  : MemType<5>;
def Mem128 : MemType<6>;

// FPFormat - This specifies what form this FP instruction has.  This is used by
// the Floating-Point stackifier pass.
class FPFormat<bits<3> val> {
  bits<3> Value = val;
}
def NotFP      : FPFormat<0>;
def ZeroArgFP  : FPFormat<1>;
def OneArgFP   : FPFormat<2>;
def OneArgFPRW : FPFormat<3>;
def TwoArgFP   : FPFormat<4>;
def CondMovFP  : FPFormat<5>;
def SpecialFP  : FPFormat<6>;


class X86Inst<string nam, bits<8> opcod, Format f, MemType m, ImmType i> : Instruction {
  let Namespace = "X86";

  let Name = nam;
  bits<8> Opcode = opcod;
  Format Form = f;
  bits<5> FormBits = Form.Value;
  MemType MemT = m;
  bits<3> MemTypeBits = MemT.Value;
  ImmType ImmT = i;
  bits<2> ImmTypeBits = ImmT.Value;

  // Attributes specific to X86 instructions...
  bit hasOpSizePrefix = 0; // Does this inst have a 0x66 prefix?
  bit printImplicitUsesBefore = 0; // Should we print implicit uses before this inst?
  bit printImplicitUsesAfter = 0; // Should we print implicit uses after this inst?

  bits<4> Prefix = 0;       // Which prefix byte does this inst have?
  FPFormat FPForm;          // What flavor of FP instruction is this?
  bits<3> FPFormBits = 0;
}

class Imp<list<Register> uses, list<Register> defs> {
  list<Register> Uses = uses;
  list<Register> Defs = defs;
}

class Pattern<dag P> {
  dag Pattern = P;
}


// Prefix byte classes which are used to indicate to the ad-hoc machine code
// emitter that various prefix bytes are required.
class OpSize { bit hasOpSizePrefix = 1; }
class TB     { bits<4> Prefix = 1; }
class REP    { bits<4> Prefix = 2; }
class D8     { bits<4> Prefix = 3; }
class D9     { bits<4> Prefix = 4; }
class DA     { bits<4> Prefix = 5; }
class DB     { bits<4> Prefix = 6; }
class DC     { bits<4> Prefix = 7; }
class DD     { bits<4> Prefix = 8; }
class DE     { bits<4> Prefix = 9; }
class DF     { bits<4> Prefix = 10; }


//===----------------------------------------------------------------------===//
// Instruction templates...

class I<string n, bits<8> o, Format f> : X86Inst<n, o, f, NoMem, NoImm>;

class Im<string n, bits<8> o, Format f, MemType m> : X86Inst<n, o, f, m, NoImm>;
class Im8 <string n, bits<8> o, Format f> : Im<n, o, f, Mem8 >;
class Im16<string n, bits<8> o, Format f> : Im<n, o, f, Mem16>;
class Im32<string n, bits<8> o, Format f> : Im<n, o, f, Mem32>;

class Ii<string n, bits<8> o, Format f, ImmType i> : X86Inst<n, o, f, NoMem, i>;
class Ii8 <string n, bits<8> o, Format f> : Ii<n, o, f, Imm8 >;
class Ii16<string n, bits<8> o, Format f> : Ii<n, o, f, Imm16>;
class Ii32<string n, bits<8> o, Format f> : Ii<n, o, f, Imm32>;

class Im8i8 <string n, bits<8> o, Format f> : X86Inst<n, o, f, Mem8 , Imm8 >;
class Im16i16<string n, bits<8> o, Format f> : X86Inst<n, o, f, Mem16, Imm16>;
class Im32i32<string n, bits<8> o, Format f> : X86Inst<n, o, f, Mem32, Imm32>;

class Im16i8<string n, bits<8> o, Format f> : X86Inst<n, o, f, Mem16, Imm8>;
class Im32i8<string n, bits<8> o, Format f> : X86Inst<n, o, f, Mem32, Imm8>;

// Helper for shift instructions
class UsesCL { list<Register> Uses = [CL]; bit printImplicitUsesAfter = 1; }

//===----------------------------------------------------------------------===//
// Instruction list...
//

def PHI : I<"PHI", 0, Pseudo>;          // PHI node...

def NOOP : I<"nop", 0x90, RawFrm>;    // nop

def ADJCALLSTACKDOWN : I<"ADJCALLSTACKDOWN", 0, Pseudo>;
def ADJCALLSTACKUP   : I<"ADJCALLSTACKUP",   0, Pseudo>;
def IMPLICIT_USE     : I<"IMPLICIT_USE",     0, Pseudo>;
def IMPLICIT_DEF     : I<"IMPLICIT_DEF",     0, Pseudo>;
let isTerminator = 1 in
  let Defs = [FP0, FP1, FP2, FP3, FP4, FP5, FP6] in
    def FP_REG_KILL    : I<"FP_REG_KILL",      0, Pseudo>;
//===----------------------------------------------------------------------===//
//  Control Flow Instructions...
//

// Return instruction...
let isTerminator = 1, isReturn = 1 in
  def RET : I<"ret", 0xC3, RawFrm>, Pattern<(retvoid)>;

// All branches are RawFrm, Void, Branch, and Terminators
let isBranch = 1, isTerminator = 1 in
  class IBr<string name, bits<8> opcode> : I<name, opcode, RawFrm>;

def JMP : IBr<"jmp", 0xE9>, Pattern<(br basicblock)>;
def JB  : IBr<"jb" , 0x82>, TB;
def JAE : IBr<"jae", 0x83>, TB;
def JE  : IBr<"je" , 0x84>, TB, Pattern<(isVoid (unspec1 basicblock))>;
def JNE : IBr<"jne", 0x85>, TB;
def JBE : IBr<"jbe", 0x86>, TB;
def JA  : IBr<"ja" , 0x87>, TB;
def JS  : IBr<"js" , 0x88>, TB;
def JNS : IBr<"jns", 0x89>, TB;
def JL  : IBr<"jl" , 0x8C>, TB;
def JGE : IBr<"jge", 0x8D>, TB;
def JLE : IBr<"jle", 0x8E>, TB;
def JG  : IBr<"jg" , 0x8F>, TB;


//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1 in
  // All calls clobber the non-callee saved registers...
  let Defs = [EAX, ECX, EDX, FP0, FP1, FP2, FP3, FP4, FP5, FP6] in {
    def CALLpcrel32 : I <"call", 0xE8, RawFrm>;
    def CALL32r     : I <"call", 0xFF, MRM2r>;
    def CALL32m     : Im32<"call", 0xFF, MRM2m>;
  }

       
//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//
def LEAVE    : I<"leave", 0xC9, RawFrm>, Imp<[EBP,ESP],[EBP,ESP]>;
def POP32r   : I<"pop",   0x58, AddRegFrm>, Imp<[ESP],[ESP]>;

let isTwoAddress = 1 in                                    // R32 = bswap R32
  def BSWAP32r : I<"bswap", 0xC8, AddRegFrm>, TB;

def XCHG8rr  : I <"xchg", 0x86, MRMDestReg>;               // xchg R8, R8
def XCHG16rr : I <"xchg", 0x87, MRMDestReg>, OpSize;       // xchg R16, R16
def XCHG32rr : I <"xchg", 0x87, MRMDestReg>;               // xchg R32, R32
def XCHG8mr  : Im8 <"xchg", 0x86, MRMDestMem>;             // xchg [mem8], R8
def XCHG16mr : Im16<"xchg", 0x87, MRMDestMem>, OpSize;     // xchg [mem16], R16
def XCHG32mr : Im32<"xchg", 0x87, MRMDestMem>;             // xchg [mem32], R32
def XCHG8rm  : Im8 <"xchg", 0x86, MRMSrcMem >;             // xchg R8, [mem8]
def XCHG16rm : Im16<"xchg", 0x87, MRMSrcMem >, OpSize;     // xchg R16, [mem16]
def XCHG32rm : Im32<"xchg", 0x87, MRMSrcMem >;             // xchg R32, [mem32]

def LEA16r   : Im32<"lea", 0x8D, MRMSrcMem>, OpSize;          // R16 = lea [mem]
def LEA32r   : Im32<"lea", 0x8D, MRMSrcMem>;                  // R32 = lea [mem]


def REP_MOVSB : I<"rep movsb", 0xA4, RawFrm>, REP,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>;
def REP_MOVSW : I<"rep movsw", 0xA5, RawFrm>, REP, OpSize,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>;
def REP_MOVSD : I<"rep movsd", 0xA5, RawFrm>, REP,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>;

def REP_STOSB : I<"rep stosb", 0xAA, RawFrm>, REP,
                Imp<[AL,ECX,EDI], [ECX,EDI]>;
def REP_STOSW : I<"rep stosw", 0xAB, RawFrm>, REP, OpSize,
                Imp<[AX,ECX,EDI], [ECX,EDI]>;
def REP_STOSD : I<"rep stosd", 0xAB, RawFrm>, REP,
                Imp<[EAX,ECX,EDI], [ECX,EDI]>;

//===----------------------------------------------------------------------===//
//  Move Instructions...
//
def MOV8rr  : I    <"mov", 0x88, MRMDestReg>,         Pattern<(set R8 , R8 )>;
def MOV16rr : I    <"mov", 0x89, MRMDestReg>, OpSize, Pattern<(set R16, R16)>;
def MOV32rr : I    <"mov", 0x89, MRMDestReg>,         Pattern<(set R32, R32)>;
def MOV8ri  : Ii8  <"mov", 0xB0, AddRegFrm >,         Pattern<(set R8 , imm )>;
def MOV16ri : Ii16 <"mov", 0xB8, AddRegFrm >, OpSize, Pattern<(set R16, imm)>;
def MOV32ri : Ii32 <"mov", 0xB8, AddRegFrm >,         Pattern<(set R32, imm)>;
def MOV8mi  : Im8i8 <"mov", 0xC6, MRM0m     >;         // [mem8] = imm8
def MOV16mi : Im16i16<"mov", 0xC7, MRM0m     >, OpSize; // [mem16] = imm16
def MOV32mi : Im32i32<"mov", 0xC7, MRM0m     >;         // [mem32] = imm32

def MOV8rm  : Im8  <"mov", 0x8A, MRMSrcMem>;          // R8  = [mem8]
def MOV16rm : Im16 <"mov", 0x8B, MRMSrcMem>, OpSize,  // R16 = [mem16]
              Pattern<(set R16, (load (plus R32, (plus (times imm, R32), imm))))>;
def MOV32rm : Im32 <"mov", 0x8B, MRMSrcMem>,          // R32 = [mem32]
              Pattern<(set R32, (load (plus R32, (plus (times imm, R32), imm))))>;

def MOV8mr  : Im8  <"mov", 0x88, MRMDestMem>;         // [mem8] = R8
def MOV16mr : Im16 <"mov", 0x89, MRMDestMem>, OpSize; // [mem16] = R16
def MOV32mr : Im32 <"mov", 0x89, MRMDestMem>;         // [mem32] = R32

//===----------------------------------------------------------------------===//
//  Fixed-Register Multiplication and Division Instructions...
//

// Extra precision multiplication
def MUL8r  : I   <"mul", 0xF6, MRM4r>, Imp<[AL],[AX]>;               // AL,AH = AL*R8
def MUL16r : I   <"mul", 0xF7, MRM4r>, Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*R16
def MUL32r : I   <"mul", 0xF7, MRM4r>, Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*R32
def MUL8m  : Im8 <"mul", 0xF6, MRM4m>, Imp<[AL],[AX]>;               // AL,AH = AL*[mem8]
def MUL16m : Im16<"mul", 0xF7, MRM4m>, Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*[mem16]
def MUL32m : Im32<"mul", 0xF7, MRM4m>, Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*[mem32]

// unsigned division/remainder
def DIV8r  : I   <"div", 0xF6, MRM6r>, Imp<[AX],[AX]>;               // AX/r8 = AL,AH
def DIV16r : I   <"div", 0xF7, MRM6r>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/r16 = AX,DX
def DIV32r : I   <"div", 0xF7, MRM6r>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/r32 = EAX,EDX
def DIV8m  : Im8 <"div", 0xF6, MRM6m>, Imp<[AX],[AX]>;               // AX/[mem8] = AL,AH
def DIV16m : Im16<"div", 0xF7, MRM6m>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/[mem16] = AX,DX
def DIV32m : Im32<"div", 0xF7, MRM6m>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/[mem32] = EAX,EDX

// signed division/remainder
def IDIV8r : I   <"idiv",0xF6, MRM7r>, Imp<[AX],[AX]>;               // AX/r8 = AL,AH
def IDIV16r: I   <"idiv",0xF7, MRM7r>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/r16 = AX,DX
def IDIV32r: I   <"idiv",0xF7, MRM7r>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/r32 = EAX,EDX
def IDIV8m : Im8 <"idiv",0xF6, MRM7m>, Imp<[AX],[AX]>;               // AX/[mem8] = AL,AH
def IDIV16m: Im16<"idiv",0xF7, MRM7m>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/[mem16] = AX,DX
def IDIV32m: Im32<"idiv",0xF7, MRM7m>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/[mem32] = EAX,EDX

// Sign-extenders for division
def CBW    : I<"cbw", 0x98, RawFrm >, Imp<[AL],[AH]>;                // AX = signext(AL)
def CWD    : I<"cwd", 0x99, RawFrm >, Imp<[AX],[DX]>;                // DX:AX = signext(AX)
def CDQ    : I<"cdq", 0x99, RawFrm >, Imp<[EAX],[EDX]>;              // EDX:EAX = signext(EAX)

//===----------------------------------------------------------------------===//
//  Two address Instructions...
//
let isTwoAddress = 1 in {

// Conditional moves
def CMOVB16rr : I   <"cmovb", 0x42, MRMSrcReg>, TB, OpSize;        // if <u, R16 = R16
def CMOVB16rm : Im16<"cmovb", 0x42, MRMSrcMem>, TB, OpSize;        // if <u, R16 = [mem16]
def CMOVB32rr : I   <"cmovb", 0x42, MRMSrcReg>, TB;                // if <u, R32 = R32
def CMOVB32rm : Im32<"cmovb", 0x42, MRMSrcMem>, TB;                // if <u, R32 = [mem32]

def CMOVAE16rr: I   <"cmovae", 0x43, MRMSrcReg>, TB, OpSize;       // if >=u, R16 = R16
def CMOVAE16rm: Im16<"cmovae", 0x43, MRMSrcMem>, TB, OpSize;       // if >=u, R16 = [mem16]
def CMOVAE32rr: I   <"cmovae", 0x43, MRMSrcReg>, TB;               // if >=u, R32 = R32
def CMOVAE32rm: Im32<"cmovae", 0x43, MRMSrcMem>, TB;               // if >=u, R32 = [mem32]

def CMOVE16rr : I   <"cmove", 0x44, MRMSrcReg>, TB, OpSize;        // if ==, R16 = R16
def CMOVE16rm : Im16<"cmove", 0x44, MRMSrcMem>, TB, OpSize;        // if ==, R16 = [mem16]
def CMOVE32rr : I   <"cmove", 0x44, MRMSrcReg>, TB;                // if ==, R32 = R32
def CMOVE32rm : Im32<"cmove", 0x44, MRMSrcMem>, TB;                // if ==, R32 = [mem32]

def CMOVNE16rr: I   <"cmovne",0x45, MRMSrcReg>, TB, OpSize;        // if !=, R16 = R16
def CMOVNE16rm: Im16<"cmovne",0x45, MRMSrcMem>, TB, OpSize;        // if !=, R16 = [mem16]
def CMOVNE32rr: I   <"cmovne",0x45, MRMSrcReg>, TB;                // if !=, R32 = R32
def CMOVNE32rm: Im32<"cmovne",0x45, MRMSrcMem>, TB;                // if !=, R32 = [mem32]

def CMOVBE16rr: I   <"cmovbe",0x46, MRMSrcReg>, TB, OpSize;        // if <=u, R16 = R16
def CMOVBE16rm: Im16<"cmovbe",0x46, MRMSrcMem>, TB, OpSize;        // if <=u, R16 = [mem16]
def CMOVBE32rr: I   <"cmovbe",0x46, MRMSrcReg>, TB;                // if <=u, R32 = R32
def CMOVBE32rm: Im32<"cmovbe",0x46, MRMSrcMem>, TB;                // if <=u, R32 = [mem32]

def CMOVA16rr : I   <"cmova", 0x47, MRMSrcReg>, TB, OpSize;        // if >u, R16 = R16
def CMOVA16rm : Im16<"cmova", 0x47, MRMSrcMem>, TB, OpSize;        // if >u, R16 = [mem16]
def CMOVA32rr : I   <"cmova", 0x47, MRMSrcReg>, TB;                // if >u, R32 = R32
def CMOVA32rm : Im32<"cmova", 0x47, MRMSrcMem>, TB;                // if >u, R32 = [mem32]

def CMOVS16rr : I   <"cmovs", 0x48, MRMSrcReg>, TB, OpSize;        // if signed, R16 = R16
def CMOVS16rm : Im16<"cmovs", 0x48, MRMSrcMem>, TB, OpSize;        // if signed, R16 = [mem16]
def CMOVS32rr : I   <"cmovs", 0x48, MRMSrcReg>, TB;                // if signed, R32 = R32
def CMOVS32rm : Im32<"cmovs", 0x48, MRMSrcMem>, TB;                // if signed, R32 = [mem32]

def CMOVNS16rr: I   <"cmovns",0x49, MRMSrcReg>, TB, OpSize;        // if !signed, R16 = R16
def CMOVNS16rm: Im16<"cmovns",0x49, MRMSrcMem>, TB, OpSize;        // if !signed, R16 = [mem16]
def CMOVNS32rr: I   <"cmovns",0x49, MRMSrcReg>, TB;                // if !signed, R32 = R32
def CMOVNS32rm: Im32<"cmovns",0x49, MRMSrcMem>, TB;                // if !signed, R32 = [mem32]

def CMOVL16rr : I   <"cmovl", 0x4C, MRMSrcReg>, TB, OpSize;        // if <s, R16 = R16
def CMOVL16rm : Im16<"cmovl", 0x4C, MRMSrcMem>, TB, OpSize;        // if <s, R16 = [mem16]
def CMOVL32rr : I   <"cmovl", 0x4C, MRMSrcReg>, TB;                // if <s, R32 = R32
def CMOVL32rm : Im32<"cmovl", 0x4C, MRMSrcMem>, TB;                // if <s, R32 = [mem32]

def CMOVGE16rr: I   <"cmovge",0x4D, MRMSrcReg>, TB, OpSize;        // if >=s, R16 = R16
def CMOVGE16rm: Im16<"cmovge",0x4D, MRMSrcMem>, TB, OpSize;        // if >=s, R16 = [mem16]
def CMOVGE32rr: I   <"cmovge",0x4D, MRMSrcReg>, TB;                // if >=s, R32 = R32
def CMOVGE32rm: Im32<"cmovge",0x4D, MRMSrcMem>, TB;                // if >=s, R32 = [mem32]

def CMOVLE16rr: I   <"cmovle",0x4E, MRMSrcReg>, TB, OpSize;        // if <=s, R16 = R16
def CMOVLE16rm: Im16<"cmovle",0x4E, MRMSrcMem>, TB, OpSize;        // if <=s, R16 = [mem16]
def CMOVLE32rr: I   <"cmovle",0x4E, MRMSrcReg>, TB;                // if <=s, R32 = R32
def CMOVLE32rm: Im32<"cmovle",0x4E, MRMSrcMem>, TB;                // if <=s, R32 = [mem32]

def CMOVG16rr : I   <"cmovg", 0x4F, MRMSrcReg>, TB, OpSize;        // if >s, R16 = R16
def CMOVG16rm : Im16<"cmovg", 0x4F, MRMSrcMem>, TB, OpSize;        // if >s, R16 = [mem16]
def CMOVG32rr : I   <"cmovg", 0x4F, MRMSrcReg>, TB;                // if >s, R32 = R32
def CMOVG32rm : Im32<"cmovg", 0x4F, MRMSrcMem>, TB;                // if >s, R32 = [mem32]

// unary instructions
def NEG8r  : I   <"neg", 0xF6, MRM3r>;         // R8  = -R8  = 0-R8
def NEG16r : I   <"neg", 0xF7, MRM3r>, OpSize; // R16 = -R16 = 0-R16
def NEG32r : I   <"neg", 0xF7, MRM3r>;         // R32 = -R32 = 0-R32
def NEG8m  : Im8 <"neg", 0xF6, MRM3m>;         // [mem8]  = -[mem8]  = 0-[mem8]
def NEG16m : Im16<"neg", 0xF7, MRM3m>, OpSize; // [mem16] = -[mem16] = 0-[mem16]
def NEG32m : Im32<"neg", 0xF7, MRM3m>;         // [mem32] = -[mem32] = 0-[mem32]

def NOT8r  : I   <"not", 0xF6, MRM2r>;         // R8  = ~R8  = R8^-1
def NOT16r : I   <"not", 0xF7, MRM2r>, OpSize; // R16 = ~R16 = R16^-1
def NOT32r : I   <"not", 0xF7, MRM2r>;         // R32 = ~R32 = R32^-1
def NOT8m  : Im8 <"not", 0xF6, MRM2m>;         // [mem8]  = ~[mem8]  = [mem8^-1]
def NOT16m : Im16<"not", 0xF7, MRM2m>, OpSize; // [mem16] = ~[mem16] = [mem16^-1]
def NOT32m : Im32<"not", 0xF7, MRM2m>;         // [mem32] = ~[mem32] = [mem32^-1]

def INC8r  : I   <"inc", 0xFE, MRM0r>;         // ++R8
def INC16r : I   <"inc", 0xFF, MRM0r>, OpSize; // ++R16
def INC32r : I   <"inc", 0xFF, MRM0r>;         // ++R32
def INC8m  : Im8 <"inc", 0xFE, MRM0m>;         // ++R8
def INC16m : Im16<"inc", 0xFF, MRM0m>, OpSize; // ++R16
def INC32m : Im32<"inc", 0xFF, MRM0m>;         // ++R32

def DEC8r  : I   <"dec", 0xFE, MRM1r>;         // --R8
def DEC16r : I   <"dec", 0xFF, MRM1r>, OpSize; // --R16
def DEC32r : I   <"dec", 0xFF, MRM1r>;         // --R32
def DEC8m  : Im8 <"dec", 0xFE, MRM1m>;         // --[mem8]
def DEC16m : Im16<"dec", 0xFF, MRM1m>, OpSize; // --[mem16]
def DEC32m : Im32<"dec", 0xFF, MRM1m>;         // --[mem32]

// Logical operators...
def AND8rr   : I     <"and", 0x20, MRMDestReg>,         Pattern<(set R8 , (and R8 , R8 ))>;
def AND16rr  : I     <"and", 0x21, MRMDestReg>, OpSize, Pattern<(set R16, (and R16, R16))>;
def AND32rr  : I     <"and", 0x21, MRMDestReg>,         Pattern<(set R32, (and R32, R32))>;
def AND8mr   : Im8   <"and", 0x20, MRMDestMem>;            // [mem8]  &= R8
def AND16mr  : Im16  <"and", 0x21, MRMDestMem>, OpSize;    // [mem16] &= R16
def AND32mr  : Im32  <"and", 0x21, MRMDestMem>;            // [mem32] &= R32
def AND8rm   : Im8   <"and", 0x22, MRMSrcMem >;            // R8  &= [mem8]
def AND16rm  : Im16  <"and", 0x23, MRMSrcMem >, OpSize;    // R16 &= [mem16]
def AND32rm  : Im32  <"and", 0x23, MRMSrcMem >;            // R32 &= [mem32]

def AND8ri   : Ii8   <"and", 0x80, MRM4r     >,         Pattern<(set R8 , (and R8 , imm))>;
def AND16ri  : Ii16  <"and", 0x81, MRM4r     >, OpSize, Pattern<(set R16, (and R16, imm))>;
def AND32ri  : Ii32  <"and", 0x81, MRM4r     >,         Pattern<(set R32, (and R32, imm))>;
def AND8mi   : Im8i8  <"and", 0x80, MRM4m    >;            // [mem8]  &= imm8
def AND16mi  : Im16i16<"and", 0x81, MRM4m    >, OpSize;    // [mem16] &= imm16
def AND32mi  : Im32i32<"and", 0x81, MRM4m    >;            // [mem32] &= imm32

def AND16ri8 : Ii8   <"and", 0x83, MRM4r     >, OpSize;    // R16 &= imm8
def AND32ri8 : Ii8   <"and", 0x83, MRM4r     >;            // R32 &= imm8
def AND16mi8 : Im16i8<"and", 0x83, MRM4m     >, OpSize;    // [mem16] &= imm8
def AND32mi8 : Im32i8<"and", 0x83, MRM4m     >;            // [mem32] &= imm8


def OR8rr    : I     <"or" , 0x08, MRMDestReg>,         Pattern<(set R8 , (or  R8 , R8 ))>;
def OR16rr   : I     <"or" , 0x09, MRMDestReg>, OpSize, Pattern<(set R16, (or  R16, R16))>;
def OR32rr   : I     <"or" , 0x09, MRMDestReg>,         Pattern<(set R32, (or  R32, R32))>;
def OR8mr    : Im8   <"or" , 0x08, MRMDestMem>;            // [mem8]  |= R8
def OR16mr   : Im16  <"or" , 0x09, MRMDestMem>, OpSize;    // [mem16] |= R16
def OR32mr   : Im32  <"or" , 0x09, MRMDestMem>;            // [mem32] |= R32
def OR8rm    : Im8   <"or" , 0x0A, MRMSrcMem >;            // R8  |= [mem8]
def OR16rm   : Im16  <"or" , 0x0B, MRMSrcMem >, OpSize;    // R16 |= [mem16]
def OR32rm   : Im32  <"or" , 0x0B, MRMSrcMem >;            // R32 |= [mem32]

def OR8ri    : Ii8   <"or" , 0x80, MRM1r     >,         Pattern<(set R8 , (or  R8 , imm))>;
def OR16ri   : Ii16  <"or" , 0x81, MRM1r     >, OpSize, Pattern<(set R16, (or  R16, imm))>;
def OR32ri   : Ii32  <"or" , 0x81, MRM1r     >,         Pattern<(set R32, (or  R32, imm))>;
def OR8mi    : Im8i8  <"or" , 0x80, MRM1m    >;            // [mem8]  |= imm8
def OR16mi   : Im16i16<"or" , 0x81, MRM1m    >, OpSize;    // [mem16] |= imm16
def OR32mi   : Im32i32<"or" , 0x81, MRM1m    >;            // [mem32] |= imm32

def OR16ri8  : Ii8   <"or" , 0x83, MRM1r     >, OpSize;    // R16 |= imm8
def OR32ri8  : Ii8   <"or" , 0x83, MRM1r     >;            // R32 |= imm8
def OR16mi8  : Im16i8<"or" , 0x83, MRM1m     >, OpSize;    // [mem16] |= imm8
def OR32mi8  : Im32i8<"or" , 0x83, MRM1m     >;            // [mem32] |= imm8


def XOR8rr   : I     <"xor", 0x30, MRMDestReg>,         Pattern<(set R8 , (xor R8 , R8 ))>;
def XOR16rr  : I     <"xor", 0x31, MRMDestReg>, OpSize, Pattern<(set R16, (xor R16, R16))>;
def XOR32rr  : I     <"xor", 0x31, MRMDestReg>,         Pattern<(set R32, (xor R32, R32))>;
def XOR8mr   : Im8   <"xor", 0x30, MRMDestMem>;            // [mem8]  ^= R8
def XOR16mr  : Im16  <"xor", 0x31, MRMDestMem>, OpSize;    // [mem16] ^= R16
def XOR32mr  : Im32  <"xor", 0x31, MRMDestMem>;            // [mem32] ^= R32
def XOR8rm   : Im8   <"xor", 0x32, MRMSrcMem >;            // R8  ^= [mem8]
def XOR16rm  : Im16  <"xor", 0x33, MRMSrcMem >, OpSize;    // R16 ^= [mem16]
def XOR32rm  : Im32  <"xor", 0x33, MRMSrcMem >;            // R32 ^= [mem32]

def XOR8ri   : Ii8   <"xor", 0x80, MRM6r     >,         Pattern<(set R8 , (xor R8 , imm))>;
def XOR16ri  : Ii16  <"xor", 0x81, MRM6r     >, OpSize, Pattern<(set R16, (xor R16, imm))>;
def XOR32ri  : Ii32  <"xor", 0x81, MRM6r     >,         Pattern<(set R32, (xor R32, imm))>;
def XOR8mi   : Im8i8  <"xor", 0x80, MRM6m    >;            // [mem8] ^= R8
def XOR16mi  : Im16i16<"xor", 0x81, MRM6m    >, OpSize;    // [mem16] ^= R16
def XOR32mi  : Im32i32<"xor", 0x81, MRM6m    >;            // [mem32] ^= R32

def XOR16ri8 : Ii8   <"xor", 0x83, MRM6r     >, OpSize;    // R16 ^= imm8
def XOR32ri8 : Ii8   <"xor", 0x83, MRM6r     >;            // R32 ^= imm8
def XOR16mi8 : Im16i8<"xor", 0x83, MRM6m     >, OpSize;    // [mem16] ^= imm8
def XOR32mi8 : Im32i8<"xor", 0x83, MRM6m     >;            // [mem32] ^= imm8

// Shift instructions
// FIXME: provide shorter instructions when imm8 == 1
def SHL8rCL  : I     <"shl", 0xD2, MRM4r     >        , UsesCL; // R8  <<= cl
def SHL16rCL : I     <"shl", 0xD3, MRM4r     >, OpSize, UsesCL; // R16 <<= cl
def SHL32rCL : I     <"shl", 0xD3, MRM4r     >        , UsesCL; // R32 <<= cl
def SHL8mCL  : Im8   <"shl", 0xD2, MRM4m     >        , UsesCL; // [mem8]  <<= cl
def SHL16mCL : Im16  <"shl", 0xD3, MRM4m     >, OpSize, UsesCL; // [mem16] <<= cl
def SHL32mCL : Im32  <"shl", 0xD3, MRM4m     >        , UsesCL; // [mem32] <<= cl

def SHL8ri   : Ii8   <"shl", 0xC0, MRM4r     >;                 // R8  <<= imm8
def SHL16ri  : Ii8   <"shl", 0xC1, MRM4r     >, OpSize;         // R16 <<= imm8
def SHL32ri  : Ii8   <"shl", 0xC1, MRM4r     >;                 // R32 <<= imm8
def SHL8mi   : Im8i8 <"shl", 0xC0, MRM4m     >;                 // [mem8]  <<= imm8
def SHL16mi  : Im16i8<"shl", 0xC1, MRM4m     >, OpSize;         // [mem16] <<= imm8
def SHL32mi  : Im32i8<"shl", 0xC1, MRM4m     >;                 // [mem32] <<= imm8

def SHR8rCL  : I     <"shr", 0xD2, MRM5r     >        , UsesCL; // R8  >>= cl
def SHR16rCL : I     <"shr", 0xD3, MRM5r     >, OpSize, UsesCL; // R16 >>= cl
def SHR32rCL : I     <"shr", 0xD3, MRM5r     >        , UsesCL; // R32 >>= cl
def SHR8mCL  : Im8   <"shr", 0xD2, MRM5m     >        , UsesCL; // [mem8]  >>= cl
def SHR16mCL : Im16  <"shr", 0xD3, MRM5m     >, OpSize, UsesCL; // [mem16] >>= cl
def SHR32mCL : Im32  <"shr", 0xD3, MRM5m     >        , UsesCL; // [mem32] >>= cl

def SHR8ri   : Ii8   <"shr", 0xC0, MRM5r     >;                 // R8  >>= imm8
def SHR16ri  : Ii8   <"shr", 0xC1, MRM5r     >, OpSize;         // R16 >>= imm8
def SHR32ri  : Ii8   <"shr", 0xC1, MRM5r     >;                 // R32 >>= imm8
def SHR8mi   : Im8i8 <"shr", 0xC0, MRM5m     >;                 // [mem8]  >>= imm8
def SHR16mi  : Im16i8<"shr", 0xC1, MRM5m     >, OpSize;         // [mem16] >>= imm8
def SHR32mi  : Im32i8<"shr", 0xC1, MRM5m     >;                 // [mem32] >>= imm8

def SAR8rCL  : I     <"sar", 0xD2, MRM7r     >        , UsesCL; // R8  >>>= cl
def SAR16rCL : I     <"sar", 0xD3, MRM7r     >, OpSize, UsesCL; // R16 >>>= cl
def SAR32rCL : I     <"sar", 0xD3, MRM7r     >        , UsesCL; // R32 >>>= cl
def SAR8mCL  : Im8   <"sar", 0xD2, MRM7m     >        , UsesCL; // [mem8]  >>>= cl
def SAR16mCL : Im16  <"sar", 0xD3, MRM7m     >, OpSize, UsesCL; // [mem16] >>>= cl
def SAR32mCL : Im32  <"sar", 0xD3, MRM7m     >        , UsesCL; // [mem32] >>>= cl

def SAR8ri   : Ii8   <"sar", 0xC0, MRM7r     >;                 // R8  >>>= imm8
def SAR16ri  : Ii8   <"sar", 0xC1, MRM7r     >, OpSize;         // R16 >>>= imm8
def SAR32ri  : Ii8   <"sar", 0xC1, MRM7r     >;                 // R32 >>>= imm8
def SAR8mi   : Im8i8 <"sar", 0xC0, MRM7m     >;                 // [mem8]  >>>= imm8
def SAR16mi  : Im16i8<"sar", 0xC1, MRM7m     >, OpSize;         // [mem16] >>>= imm8
def SAR32mi  : Im32i8<"sar", 0xC1, MRM7m     >;                 // [mem32] >>>= imm8

def SHLD32rrCL : I     <"shld", 0xA5, MRMDestReg>, TB, UsesCL;    // R32 <<= R32,R32 cl
def SHLD32mrCL : Im32  <"shld", 0xA5, MRMDestMem>, TB, UsesCL;    // [mem32] <<= [mem32],R32 cl
def SHLD32rri8 : Ii8   <"shld", 0xA4, MRMDestReg>, TB;            // R32 <<= R32,R32 imm8
def SHLD32mri8 : Im32i8<"shld", 0xA4, MRMDestMem>, TB;            // [mem32] <<= [mem32],R32 imm8

def SHRD32rrCL : I     <"shrd", 0xAD, MRMDestReg>, TB, UsesCL;    // R32 >>= R32,R32 cl
def SHRD32mrCL : Im32  <"shrd", 0xAD, MRMDestMem>, TB, UsesCL;    // [mem32] >>= [mem32],R32 cl
def SHRD32rri8 : Ii8   <"shrd", 0xAC, MRMDestReg>, TB;            // R32 >>= R32,R32 imm8
def SHRD32mri8 : Im32i8<"shrd", 0xAC, MRMDestMem>, TB;            // [mem32] >>= [mem32],R32 imm8


// Arithmetic...
def ADD8rr   : I     <"add", 0x00, MRMDestReg>,         Pattern<(set R8 , (plus R8 , R8 ))>;
def ADD16rr  : I     <"add", 0x01, MRMDestReg>, OpSize, Pattern<(set R16, (plus R16, R16))>;
def ADD32rr  : I     <"add", 0x01, MRMDestReg>,         Pattern<(set R32, (plus R32, R32))>;
def ADD8mr   : Im8   <"add", 0x00, MRMDestMem>;         // [mem8]  += R8
def ADD16mr  : Im16  <"add", 0x01, MRMDestMem>, OpSize; // [mem16] += R16
def ADD32mr  : Im32  <"add", 0x01, MRMDestMem>;         // [mem32] += R32
def ADD8rm   : Im8   <"add", 0x02, MRMSrcMem >;         // R8  += [mem8]
def ADD16rm  : Im16  <"add", 0x03, MRMSrcMem >, OpSize; // R16 += [mem16]
def ADD32rm  : Im32  <"add", 0x03, MRMSrcMem >;         // R32 += [mem32]

def ADD8ri   : Ii8   <"add", 0x80, MRM0r     >,         Pattern<(set R8 , (plus R8 , imm))>;
def ADD16ri  : Ii16  <"add", 0x81, MRM0r     >, OpSize, Pattern<(set R16, (plus R16, imm))>;
def ADD32ri  : Ii32  <"add", 0x81, MRM0r     >,         Pattern<(set R32, (plus R32, imm))>;
def ADD8mi   : Im8i8  <"add", 0x80, MRM0m    >;         // [mem8] += I8
def ADD16mi  : Im16i16<"add", 0x81, MRM0m    >, OpSize; // [mem16] += I16
def ADD32mi  : Im32i32<"add", 0x81, MRM0m    >;         // [mem32] += I32

def ADD16ri8 : Ii8   <"add", 0x83, MRM0r     >, OpSize;   // ADDri with sign extended 8 bit imm
def ADD32ri8 : Ii8   <"add", 0x83, MRM0r     >;
def ADD16mi8 : Im16i8<"add", 0x83, MRM0m     >, OpSize; // [mem16] += I8
def ADD32mi8 : Im32i8<"add", 0x83, MRM0m     >;         // [mem32] += I8

def ADC32rr  : I     <"adc", 0x11, MRMDestReg>;         // R32 += R32+Carry
def ADC32rm  : Im32  <"adc", 0x11, MRMSrcMem >;         // R32 += [mem32]+Carry
def ADC32mr  : Im32  <"adc", 0x13, MRMDestMem>;         // [mem32] += R32+Carry


def SUB8rr   : I     <"sub", 0x28, MRMDestReg>,         Pattern<(set R8 , (minus R8 , R8 ))>;
def SUB16rr  : I     <"sub", 0x29, MRMDestReg>, OpSize, Pattern<(set R16, (minus R16, R16))>;
def SUB32rr  : I     <"sub", 0x29, MRMDestReg>,         Pattern<(set R32, (minus R32, R32))>;
def SUB8mr   : Im8   <"sub", 0x28, MRMDestMem>;         // [mem8]  -= R8
def SUB16mr  : Im16  <"sub", 0x29, MRMDestMem>, OpSize; // [mem16] -= R16
def SUB32mr  : Im32  <"sub", 0x29, MRMDestMem>;         // [mem32] -= R32
def SUB8rm   : Im8   <"sub", 0x2A, MRMSrcMem >;         // R8  -= [mem8]
def SUB16rm  : Im16  <"sub", 0x2B, MRMSrcMem >, OpSize; // R16 -= [mem16]
def SUB32rm  : Im32  <"sub", 0x2B, MRMSrcMem >;         // R32 -= [mem32]

def SUB8ri   : Ii8   <"sub", 0x80, MRM5r     >,         Pattern<(set R8 , (minus R8 , imm))>;
def SUB16ri  : Ii16  <"sub", 0x81, MRM5r     >, OpSize, Pattern<(set R16, (minus R16, imm))>;
def SUB32ri  : Ii32  <"sub", 0x81, MRM5r     >,         Pattern<(set R32, (minus R32, imm))>;
def SUB8mi   : Im8i8  <"sub", 0x80, MRM5m    >;         // [mem8] -= I8
def SUB16mi  : Im16i16<"sub", 0x81, MRM5m    >, OpSize; // [mem16] -= I16
def SUB32mi  : Im32i32<"sub", 0x81, MRM5m    >;         // [mem32] -= I32

def SUB16ri8 : Ii8   <"sub", 0x83, MRM5r     >, OpSize;
def SUB32ri8 : Ii8   <"sub", 0x83, MRM5r     >;
def SUB16mi8 : Im16i8<"sub", 0x83, MRM5m     >, OpSize; // [mem16] -= I8
def SUB32mi8 : Im32i8<"sub", 0x83, MRM5m     >;         // [mem32] -= I8

def SBB32rr  : I     <"sbb", 0x19, MRMDestReg>;         // R32 -= R32+Borrow
def SBB32rm  : Im32  <"sbb", 0x19, MRMSrcMem >;         // R32 -= [mem32]+Borrow
def SBB32mr  : Im32  <"sbb", 0x1B, MRMDestMem>;         // [mem32] -= R32+Borrow

def IMUL16rr : I     <"imul", 0xAF, MRMSrcReg>, TB, OpSize, Pattern<(set R16, (times R16, R16))>;
def IMUL32rr : I     <"imul", 0xAF, MRMSrcReg>, TB        , Pattern<(set R32, (times R32, R32))>;
def IMUL16rm : Im16  <"imul", 0xAF, MRMSrcMem>, TB, OpSize;
def IMUL32rm : Im32  <"imul", 0xAF, MRMSrcMem>, TB        ;

} // end Two Address instructions

// These are suprisingly enough not two address instructions!
def IMUL16rri  : Ii16  <"imul", 0x69, MRMSrcReg>,     OpSize;  // R16 = R16*I16
def IMUL32rri  : Ii32  <"imul", 0x69, MRMSrcReg>;              // R32 = R32*I32
def IMUL16rri8 : Ii8   <"imul", 0x6B, MRMSrcReg>,     OpSize;  // R16 = R16*I8
def IMUL32rri8 : Ii8   <"imul", 0x6B, MRMSrcReg>;              // R32 = R32*I8
def IMUL16rmi  : Im16i16<"imul",0x69, MRMSrcMem>,     OpSize;  // R16 = [mem16]*I16
def IMUL32rmi  : Im32i32<"imul",0x69, MRMSrcMem>;              // R32 = [mem32]*I32
def IMUL16rmi8 : Im16i8<"imul", 0x6B, MRMSrcMem>,     OpSize;  // R16 = [mem16]*I8
def IMUL32rmi8 : Im32i8<"imul", 0x6B, MRMSrcMem>;              // R32 = [mem32]*I8

//===----------------------------------------------------------------------===//
// Test instructions are just like AND, except they don't generate a result.
def TEST8rr  : I    <"test", 0x84, MRMDestReg>;          // flags = R8  & R8
def TEST16rr : I    <"test", 0x85, MRMDestReg>, OpSize;  // flags = R16 & R16
def TEST32rr : I    <"test", 0x85, MRMDestReg>;          // flags = R32 & R32
def TEST8mr  : Im8  <"test", 0x84, MRMDestMem>;          // flags = [mem8]  & R8
def TEST16mr : Im16 <"test", 0x85, MRMDestMem>, OpSize;  // flags = [mem16] & R16
def TEST32mr : Im32 <"test", 0x85, MRMDestMem>;          // flags = [mem32] & R32
def TEST8rm  : Im8  <"test", 0x84, MRMSrcMem >;          // flags = R8  & [mem8]
def TEST16rm : Im16 <"test", 0x85, MRMSrcMem >, OpSize;  // flags = R16 & [mem16]
def TEST32rm : Im32 <"test", 0x85, MRMSrcMem >;          // flags = R32 & [mem32]

def TEST8ri  : Ii8  <"test", 0xF6, MRM0r     >;          // flags = R8  & imm8
def TEST16ri : Ii16 <"test", 0xF7, MRM0r     >, OpSize;  // flags = R16 & imm16
def TEST32ri : Ii32 <"test", 0xF7, MRM0r     >;          // flags = R32 & imm32
def TEST8mi  : Im8i8  <"test", 0xF6, MRM0m     >;          // flags = [mem8]  & imm8
def TEST16mi : Im16i16<"test", 0xF7, MRM0m     >, OpSize;  // flags = [mem16] & imm16
def TEST32mi : Im32i32<"test", 0xF7, MRM0m     >;          // flags = [mem32] & imm32



// Condition code ops, incl. set if equal/not equal/...
def SAHF     : I  <"sahf" , 0x9E, RawFrm>, Imp<[AH],[]>;  // flags = AH
def LAHF     : I  <"lahf" , 0x9F, RawFrm>, Imp<[],[AH]>;  // AH = flags

def SETBr    : I  <"setb" , 0x92, MRM0r>, TB;            // R8 = <  unsign
def SETBm    : Im8<"setb" , 0x92, MRM0m>, TB;            // [mem8] = <  unsign
def SETAEr   : I  <"setae", 0x93, MRM0r>, TB;            // R8 = >= unsign
def SETAEm   : Im8<"setae", 0x93, MRM0m>, TB;            // [mem8] = >= unsign
def SETEr    : I  <"sete" , 0x94, MRM0r>, TB;            // R8 = ==
def SETEm    : Im8<"sete" , 0x94, MRM0m>, TB;            // [mem8] = ==
def SETNEr   : I  <"setne", 0x95, MRM0r>, TB;            // R8 = !=
def SETNEm   : Im8<"setne", 0x95, MRM0m>, TB;            // [mem8] = !=
def SETBEr   : I  <"setbe", 0x96, MRM0r>, TB;            // R8 = <= unsign
def SETBEm   : Im8<"setbe", 0x96, MRM0m>, TB;            // [mem8] = <= unsign
def SETAr    : I  <"seta" , 0x97, MRM0r>, TB;            // R8 = >  signed
def SETAm    : Im8<"seta" , 0x97, MRM0m>, TB;            // [mem8] = >  signed
def SETSr    : I  <"sets" , 0x98, MRM0r>, TB;            // R8 = <sign bit>
def SETSm    : Im8<"sets" , 0x98, MRM0m>, TB;            // [mem8] = <sign bit>
def SETNSr   : I  <"setns", 0x99, MRM0r>, TB;            // R8 = !<sign bit>
def SETNSm   : Im8<"setns", 0x99, MRM0m>, TB;            // [mem8] = !<sign bit>
def SETLr    : I  <"setl" , 0x9C, MRM0r>, TB;            // R8 = <  signed
def SETLm    : Im8<"setl" , 0x9C, MRM0m>, TB;            // [mem8] = <  signed
def SETGEr   : I  <"setge", 0x9D, MRM0r>, TB;            // R8 = >= signed
def SETGEm   : Im8<"setge", 0x9D, MRM0m>, TB;            // [mem8] = >= signed
def SETLEr   : I  <"setle", 0x9E, MRM0r>, TB;            // R8 = <= signed
def SETLEm   : Im8<"setle", 0x9E, MRM0m>, TB;            // [mem8] = <= signed
def SETGr    : I  <"setg" , 0x9F, MRM0r>, TB;            // R8 = <  signed
def SETGm    : Im8<"setg" , 0x9F, MRM0m>, TB;            // [mem8] = <  signed

// Integer comparisons
def CMP8rr  : I    <"cmp", 0x38, MRMDestReg>;              // compare R8, R8
def CMP16rr : I    <"cmp", 0x39, MRMDestReg>, OpSize;      // compare R16, R16
def CMP32rr : I    <"cmp", 0x39, MRMDestReg>,              // compare R32, R32
              Pattern<(isVoid (unspec2 R32, R32))>;
def CMP8mr  : Im8  <"cmp", 0x38, MRMDestMem>;              // compare [mem8], R8
def CMP16mr : Im16 <"cmp", 0x39, MRMDestMem>, OpSize;      // compare [mem16], R16
def CMP32mr : Im32 <"cmp", 0x39, MRMDestMem>;              // compare [mem32], R32
def CMP8rm  : Im8  <"cmp", 0x3A, MRMSrcMem >;              // compare R8, [mem8]
def CMP16rm : Im16 <"cmp", 0x3B, MRMSrcMem >, OpSize;      // compare R16, [mem16]
def CMP32rm : Im32 <"cmp", 0x3B, MRMSrcMem >;              // compare R32, [mem32]
def CMP8ri  : Ii8  <"cmp", 0x80, MRM7r     >;              // compare R8, imm8
def CMP16ri : Ii16 <"cmp", 0x81, MRM7r     >, OpSize;      // compare R16, imm16
def CMP32ri : Ii32 <"cmp", 0x81, MRM7r     >;              // compare R32, imm32
def CMP8mi  : Im8i8  <"cmp", 0x80, MRM7m   >;              // compare [mem8], imm8
def CMP16mi : Im16i16<"cmp", 0x81, MRM7m   >, OpSize;      // compare [mem16], imm16
def CMP32mi : Im32i32<"cmp", 0x81, MRM7m   >;              // compare [mem32], imm32

// Sign/Zero extenders
def MOVSX16rr8 : I   <"movsx", 0xBE, MRMSrcReg>, TB, OpSize; // R16 = signext(R8)
def MOVSX32rr8 : I   <"movsx", 0xBE, MRMSrcReg>, TB;         // R32 = signext(R8)
def MOVSX32rr16: I   <"movsx", 0xBF, MRMSrcReg>, TB;         // R32 = signext(R16)
def MOVSX16rm8 : Im8 <"movsx", 0xBE, MRMSrcMem>, TB, OpSize; // R16 = signext([mem8])
def MOVSX32rm8 : Im8 <"movsx", 0xBE, MRMSrcMem>, TB;         // R32 = signext([mem8])
def MOVSX32rm16: Im16<"movsx", 0xBF, MRMSrcMem>, TB;         // R32 = signext([mem16])

def MOVZX16rr8 : I   <"movzx", 0xB6, MRMSrcReg>, TB, OpSize; // R16 = zeroext(R8)
def MOVZX32rr8 : I   <"movzx", 0xB6, MRMSrcReg>, TB;         // R32 = zeroext(R8)
def MOVZX32rr16: I   <"movzx", 0xB7, MRMSrcReg>, TB;         // R32 = zeroext(R16)
def MOVZX16rm8 : Im8 <"movzx", 0xB6, MRMSrcMem>, TB, OpSize; // R16 = zeroext([mem8])
def MOVZX32rm8 : Im8 <"movzx", 0xB6, MRMSrcMem>, TB;         // R32 = zeroext([mem8])
def MOVZX32rm16: Im16<"movzx", 0xB7, MRMSrcMem>, TB;         // R32 = zeroext([mem16])


//===----------------------------------------------------------------------===//
// Floating point support
//===----------------------------------------------------------------------===//

// FIXME: These need to indicate mod/ref sets for FP regs... & FP 'TOP'

// Floating point instruction templates
class FPInst<string n, bits<8> o, Format F, FPFormat fp, MemType m, ImmType i>
  : X86Inst<n, o, F, m, i> { let FPForm = fp; let FPFormBits = FPForm.Value; }

class FPI<string n, bits<8> o, Format F, FPFormat fp> : FPInst<n, o, F, fp, NoMem, NoImm>;

class FPIM<string n, bits<8> o, Format F, FPFormat fp, MemType m> : FPInst<n, o, F, fp, m, NoImm>;

class FPI16m<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem16>;
class FPI32m<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem32>;
class FPI64m<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem64>;
class FPI80m<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem80>;

// Pseudo instructions for floating point.  We use these pseudo instructions
// because they can be expanded by the fp spackifier into one of many different
// forms of instructions for doing these operations.  Until the stackifier runs,
// we prefer to be abstract.
def FpMOV : FPI<"FMOV", 0, Pseudo, SpecialFP>;   // f1 = fmov f2
def FpADD : FPI<"FADD", 0, Pseudo, TwoArgFP>;    // f1 = fadd f2, f3
def FpSUB : FPI<"FSUB", 0, Pseudo, TwoArgFP>;    // f1 = fsub f2, f3
def FpMUL : FPI<"FMUL", 0, Pseudo, TwoArgFP>;    // f1 = fmul f2, f3
def FpDIV : FPI<"FDIV", 0, Pseudo, TwoArgFP>;    // f1 = fdiv f2, f3

def FpUCOM : FPI<"FUCOM", 0, Pseudo, TwoArgFP>;  // FPSW = fucom f1, f2
def FpGETRESULT : FPI<"FGETRESULT",0, Pseudo, SpecialFP>;  // FPR = ST(0)
def FpSETRESULT : FPI<"FSETRESULT",0, Pseudo, SpecialFP>;  // ST(0) = FPR


// Floating point cmovs...
let isTwoAddress = 1, Uses = [ST0], Defs = [ST0], printImplicitUsesBefore = 1 in {
  def FCMOVB  : FPI   <"fcmovb" , 0xC0, AddRegFrm, CondMovFP>, DA;     // fcmovb  ST(i) -> ST(0)
  def FCMOVBE : FPI   <"fcmovbe", 0xD0, AddRegFrm, CondMovFP>, DA;     // fcmovbe ST(i) -> ST(0)
  def FCMOVE  : FPI   <"fcmove" , 0xC8, AddRegFrm, CondMovFP>, DA;     // fcmove  ST(i) -> ST(0)
  def FCMOVAE : FPI   <"fcmovae", 0xC0, AddRegFrm, CondMovFP>, DB;     // fcmovae ST(i) -> ST(0)
  def FCMOVA  : FPI   <"fcmova" , 0xD0, AddRegFrm, CondMovFP>, DB;     // fcmova  ST(i) -> ST(0)
  def FCMOVNE : FPI   <"fcmovne", 0xC8, AddRegFrm, CondMovFP>, DB;     // fcmovne ST(i) -> ST(0)
}

// Floating point loads & stores...
def FLDrr   : FPI    <"fld"   , 0xC0, AddRegFrm, NotFP>, D9;        // push(ST(i))
def FLD32m  : FPI32m <"fld"   , 0xD9, MRM0m    , ZeroArgFP>;        // load float
def FLD64m  : FPI64m <"fld"   , 0xDD, MRM0m    , ZeroArgFP>;        // load double
def FLD80m  : FPI80m <"fld"   , 0xDB, MRM5m    , ZeroArgFP>;        // load extended
def FILD16m : FPI16m <"fild"  , 0xDF, MRM0m    , ZeroArgFP>;        // load signed short
def FILD32m : FPI32m <"fild"  , 0xDB, MRM0m    , ZeroArgFP>;        // load signed int
def FILD64m : FPI64m <"fild"  , 0xDF, MRM5m    , ZeroArgFP>;        // load signed long

def FSTrr    : FPI    <"fst" , 0xD0, AddRegFrm, NotFP   >, DD;      // ST(i) = ST(0)
def FSTPrr   : FPI    <"fstp", 0xD8, AddRegFrm, NotFP   >, DD;      // ST(i) = ST(0), pop
def FST32m   : FPI32m <"fst" , 0xD9, MRM2m    , OneArgFP>;          // store float
def FST64m   : FPI64m <"fst" , 0xDD, MRM2m    , OneArgFP>;          // store double
def FSTP32m  : FPI32m <"fstp", 0xD9, MRM3m    , OneArgFP>;          // store float, pop
def FSTP64m  : FPI64m <"fstp", 0xDD, MRM3m    , OneArgFP>;          // store double, pop
def FSTP80m  : FPI80m <"fstp", 0xDB, MRM7m    , OneArgFP>;          // store extended, pop

def FIST16m  : FPI16m <"fist",    0xDF, MRM2m , OneArgFP>;          // store signed short
def FIST32m  : FPI32m <"fist",    0xDB, MRM2m , OneArgFP>;          // store signed int
def FISTP16m : FPI16m <"fistp",   0xDF, MRM3m , NotFP   >;          // store signed short, pop
def FISTP32m : FPI32m <"fistp",   0xDB, MRM3m , NotFP   >;          // store signed int, pop
def FISTP64m : FPI64m <"fistpll", 0xDF, MRM7m , OneArgFP>;          // store signed long, pop

def FXCH     : FPI    <"fxch",    0xC8, AddRegFrm, NotFP>, D9;      // fxch ST(i), ST(0)

// Floating point constant loads...
def FLD0 : FPI<"fldz", 0xEE, RawFrm, ZeroArgFP>, D9;
def FLD1 : FPI<"fld1", 0xE8, RawFrm, ZeroArgFP>, D9;


// Unary operations...
def FCHS : FPI<"fchs", 0xE0, RawFrm, OneArgFPRW>, D9;           // f1 = fchs f2

def FTST : FPI<"ftst", 0xE4, RawFrm, OneArgFP>, D9;             // ftst ST(0)

// Binary arithmetic operations...
class FPST0rInst<string n, bits<8> o> : I<n, o, AddRegFrm>, D8 {
  list<Register> Uses = [ST0];
  list<Register> Defs = [ST0];
}
class FPrST0Inst<string n, bits<8> o> : I<n, o, AddRegFrm>, DC {
  bit printImplicitUsesAfter = 1;
  list<Register> Uses = [ST0];
}
class FPrST0PInst<string n, bits<8> o> : I<n, o, AddRegFrm>, DE {
  list<Register> Uses = [ST0];
}

def FADDST0r   : FPST0rInst <"fadd",    0xC0>;
def FADDrST0   : FPrST0Inst <"fadd",    0xC0>;
def FADDPrST0  : FPrST0PInst<"faddp",   0xC0>;

def FSUBRST0r  : FPST0rInst <"fsubr",   0xE8>;
def FSUBrST0   : FPrST0Inst <"fsub",    0xE8>;
def FSUBPrST0  : FPrST0PInst<"fsubp",   0xE8>;

def FSUBST0r   : FPST0rInst <"fsub",    0xE0>;
def FSUBRrST0  : FPrST0Inst <"fsubr",   0xE0>;
def FSUBRPrST0 : FPrST0PInst<"fsubrp",  0xE0>;

def FMULST0r   : FPST0rInst <"fmul",    0xC8>;
def FMULrST0   : FPrST0Inst <"fmul",    0xC8>;
def FMULPrST0  : FPrST0PInst<"fmulp",   0xC8>;

def FDIVRST0r  : FPST0rInst <"fdivr",   0xF8>;
def FDIVrST0   : FPrST0Inst <"fdiv",    0xF8>;
def FDIVPrST0  : FPrST0PInst<"fdivp",   0xF8>;

def FDIVST0r   : FPST0rInst <"fdiv",    0xF0>;   // ST(0) = ST(0) / ST(i)
def FDIVRrST0  : FPrST0Inst <"fdivr",   0xF0>;   // ST(i) = ST(0) / ST(i)
def FDIVRPrST0 : FPrST0PInst<"fdivrp",  0xF0>;   // ST(i) = ST(0) / ST(i), pop

// Floating point compares
def FUCOMr    : I<"fucom"  , 0xE0, AddRegFrm>, DD, Imp<[ST0],[]>;  // FPSW = compare ST(0) with ST(i)
def FUCOMPr   : I<"fucomp" , 0xE8, AddRegFrm>, DD, Imp<[ST0],[]>;  // FPSW = compare ST(0) with ST(i), pop
def FUCOMPPr  : I<"fucompp", 0xE9, RawFrm   >, DA, Imp<[ST0],[]>;  // compare ST(0) with ST(1), pop, pop

// Floating point flag ops
def FNSTSW8r  : I   <"fnstsw" , 0xE0, RawFrm>, DF, Imp<[],[AX]>;   // AX = fp flags
def FNSTCW16m : Im16<"fnstcw" , 0xD9, MRM7m >;                     // [mem16] = X87 control world
def FLDCW16m  : Im16<"fldcw"  , 0xD9, MRM5m >;                     // X87 control world = [mem16]


//===----------------------------------------------------------------------===//
//  Instruction Expanders
//

def RET_R32 : Expander<(ret R32:$reg),
                       [(MOV32rr EAX, R32:$reg),
                        (RET)]>;

// FIXME: This should eventually just be implemented by defining a frameidx as a
// value address for a load.
def LOAD_FI16 : Expander<(set R16:$dest, (load frameidx:$fi)),
                         [(MOV16rm R16:$dest, frameidx:$fi, 1, 0/*NoReg*/, 0)]>;

def LOAD_FI32 : Expander<(set R32:$dest, (load frameidx:$fi)),
                         [(MOV32rm R32:$dest, frameidx:$fi, 1, 0/*NoReg*/, 0)]>;


def LOAD_R16 : Expander<(set R16:$dest, (load R32:$src)),
                         [(MOV16rm R16:$dest, R32:$src, 1, 0/*NoReg*/, 0)]>;

def LOAD_R32 : Expander<(set R32:$dest, (load R32:$src)),
                         [(MOV32rm R32:$dest, R32:$src, 1, 0/*NoReg*/, 0)]>;

def BR_EQ : Expander<(brcond (seteq R32:$a1, R32:$a2),
                             basicblock:$d1, basicblock:$d2),
                     [(CMP32rr R32:$a1, R32:$a2),
                      (JE basicblock:$d1),
                      (JMP basicblock:$d2)]>;
