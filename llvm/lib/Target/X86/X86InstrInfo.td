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
def Mem80  : MemType<4>;
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
def SpecialFP  : FPFormat<5>;


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
  bit printImplicitUses = 0; // Should we print implicit uses of this inst?

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

class IM<string n, bits<8> o, Format f, MemType m> : X86Inst<n, o, f, m, NoImm>;
class IM8 <string n, bits<8> o, Format f> : IM<n, o, f, Mem8 >;
class IM16<string n, bits<8> o, Format f> : IM<n, o, f, Mem16>;
class IM32<string n, bits<8> o, Format f> : IM<n, o, f, Mem32>;

class II<string n, bits<8> o, Format f, ImmType i> : X86Inst<n, o, f, NoMem, i>;
class II8 <string n, bits<8> o, Format f> : II<n, o, f, Imm8 >;
class II16<string n, bits<8> o, Format f> : II<n, o, f, Imm16>;
class II32<string n, bits<8> o, Format f> : II<n, o, f, Imm32>;

class I8MI <string n, bits<8> o, Format f> : X86Inst<n, o, f, Mem8 , Imm8 >;
class I16MI<string n, bits<8> o, Format f> : X86Inst<n, o, f, Mem16, Imm16>;
class I32MI<string n, bits<8> o, Format f> : X86Inst<n, o, f, Mem32, Imm32>;

class IM16I8<string n, bits<8> o, Format f> : X86Inst<n, o, f, Mem16, Imm8>;
class IM32I8<string n, bits<8> o, Format f> : X86Inst<n, o, f, Mem32, Imm8>;

// Helper for shift instructions
class UsesCL { list<Register> Uses = [CL]; bit printImplicitUses = 1; }

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
    def CALLr32     : I <"call", 0xFF, MRM2r>;
    def CALLm32     : IM32<"call", 0xFF, MRM2m>;
  }

       
//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//
def LEAVE    : I<"leave", 0xC9, RawFrm>, Imp<[EBP,ESP],[EBP,ESP]>;
def POPr32   : I<"pop",   0x58, AddRegFrm>, Imp<[ESP],[ESP]>;

let isTwoAddress = 1 in                                    // R32 = bswap R32
  def BSWAPr32 : I<"bswap", 0xC8, AddRegFrm>, TB;

def XCHGrr8  : I <"xchg", 0x86, MRMDestReg>;               // xchg R8, R8
def XCHGrr16 : I <"xchg", 0x87, MRMDestReg>, OpSize;       // xchg R16, R16
def XCHGrr32 : I <"xchg", 0x87, MRMDestReg>;               // xchg R32, R32
def XCHGmr8  : IM8 <"xchg", 0x86, MRMDestMem>;             // xchg [mem8], R8
def XCHGmr16 : IM16<"xchg", 0x87, MRMDestMem>, OpSize;     // xchg [mem16], R16
def XCHGmr32 : IM32<"xchg", 0x87, MRMDestMem>;             // xchg [mem32], R32
def XCHGrm8  : IM8 <"xchg", 0x86, MRMSrcMem >;             // xchg R8, [mem8]
def XCHGrm16 : IM16<"xchg", 0x87, MRMSrcMem >, OpSize;     // xchg R16, [mem16]
def XCHGrm32 : IM32<"xchg", 0x87, MRMSrcMem >;             // xchg R32, [mem32]

def LEAr16   : IM32<"lea", 0x8D, MRMSrcMem>, OpSize;          // R16 = lea [mem]
def LEAr32   : IM32<"lea", 0x8D, MRMSrcMem>;                  // R32 = lea [mem]


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
def MOVrr8  : I    <"mov", 0x88, MRMDestReg>,         Pattern<(set R8 , R8 )>;
def MOVrr16 : I    <"mov", 0x89, MRMDestReg>, OpSize, Pattern<(set R16, R16)>;
def MOVrr32 : I    <"mov", 0x89, MRMDestReg>,         Pattern<(set R32, R32)>;
def MOVri8  : II8  <"mov", 0xB0, AddRegFrm >,         Pattern<(set R8 , imm )>;
def MOVri16 : II16 <"mov", 0xB8, AddRegFrm >, OpSize, Pattern<(set R16, imm)>;
def MOVri32 : II32 <"mov", 0xB8, AddRegFrm >,         Pattern<(set R32, imm)>;
def MOVmi8  : I8MI <"mov", 0xC6, MRM0m     >;         // [mem8] = imm8
def MOVmi16 : I16MI<"mov", 0xC7, MRM0m     >, OpSize; // [mem16] = imm16
def MOVmi32 : I32MI<"mov", 0xC7, MRM0m     >;         // [mem32] = imm32

def MOVrm8  : IM8  <"mov", 0x8A, MRMSrcMem>;          // R8  = [mem8]
def MOVrm16 : IM16 <"mov", 0x8B, MRMSrcMem>, OpSize,  // R16 = [mem16]
              Pattern<(set R16, (load (plus R32, (plus (times imm, R32), imm))))>;
def MOVrm32 : IM32 <"mov", 0x8B, MRMSrcMem>,          // R32 = [mem32]
              Pattern<(set R32, (load (plus R32, (plus (times imm, R32), imm))))>;

def MOVmr8  : IM8  <"mov", 0x88, MRMDestMem>;         // [mem8] = R8
def MOVmr16 : IM16 <"mov", 0x89, MRMDestMem>, OpSize; // [mem16] = R16
def MOVmr32 : IM32 <"mov", 0x89, MRMDestMem>;         // [mem32] = R32

//===----------------------------------------------------------------------===//
//  Fixed-Register Multiplication and Division Instructions...
//

// Extra precision multiplication
def MULr8  : I   <"mul", 0xF6, MRM4r>, Imp<[AL],[AX]>;               // AL,AH = AL*R8
def MULr16 : I   <"mul", 0xF7, MRM4r>, Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*R16
def MULr32 : I   <"mul", 0xF7, MRM4r>, Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*R32
def MULm8  : IM8 <"mul", 0xF6, MRM4m>, Imp<[AL],[AX]>;               // AL,AH = AL*[mem8]
def MULm16 : IM16<"mul", 0xF7, MRM4m>, Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*[mem16]
def MULm32 : IM32<"mul", 0xF7, MRM4m>, Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*[mem32]

// unsigned division/remainder
def DIVr8  : I   <"div", 0xF6, MRM6r>, Imp<[AX],[AX]>;               // AX/r8 = AL,AH
def DIVr16 : I   <"div", 0xF7, MRM6r>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/r16 = AX,DX
def DIVr32 : I   <"div", 0xF7, MRM6r>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/r32 = EAX,EDX
def DIVm8  : IM8 <"div", 0xF6, MRM6m>, Imp<[AX],[AX]>;               // AX/[mem8] = AL,AH
def DIVm16 : IM16<"div", 0xF7, MRM6m>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/[mem16] = AX,DX
def DIVm32 : IM32<"div", 0xF7, MRM6m>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/[mem32] = EAX,EDX

// signed division/remainder
def IDIVr8 : I   <"idiv",0xF6, MRM7r>, Imp<[AX],[AX]>;               // AX/r8 = AL,AH
def IDIVr16: I   <"idiv",0xF7, MRM7r>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/r16 = AX,DX
def IDIVr32: I   <"idiv",0xF7, MRM7r>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/r32 = EAX,EDX
def IDIVm8 : IM8 <"idiv",0xF6, MRM7m>, Imp<[AX],[AX]>;               // AX/[mem8] = AL,AH
def IDIVm16: IM16<"idiv",0xF7, MRM7m>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/[mem16] = AX,DX
def IDIVm32: IM32<"idiv",0xF7, MRM7m>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/[mem32] = EAX,EDX

// Sign-extenders for division
def CBW    : I<"cbw", 0x98, RawFrm >, Imp<[AL],[AH]>;                // AX = signext(AL)
def CWD    : I<"cwd", 0x99, RawFrm >, Imp<[AX],[DX]>;                // DX:AX = signext(AX)
def CDQ    : I<"cdq", 0x99, RawFrm >, Imp<[EAX],[EDX]>;              // EDX:EAX = signext(EAX)

//===----------------------------------------------------------------------===//
//  Two address Instructions...
//
let isTwoAddress = 1 in {

// Conditional moves.  These are modelled as X = cmovXX Y, Z.  Eventually
// register allocated to cmovXX XY, Z
def CMOVErr16 : I<"cmove", 0x44, MRMSrcReg>, TB, OpSize;        // if ==, R16 = R16
def CMOVNErr32: I<"cmovne",0x45, MRMSrcReg>, TB;                // if !=, R32 = R32
def CMOVSrr32 : I<"cmovs", 0x48, MRMSrcReg>, TB;                // if signed, R32 = R32

// unary instructions
def NEGr8  : I   <"neg", 0xF6, MRM3r>;         // R8  = -R8  = 0-R8
def NEGr16 : I   <"neg", 0xF7, MRM3r>, OpSize; // R16 = -R16 = 0-R16
def NEGr32 : I   <"neg", 0xF7, MRM3r>;         // R32 = -R32 = 0-R32
def NEGm8  : IM8 <"neg", 0xF6, MRM3m>;         // [mem8]  = -[mem8]  = 0-[mem8]
def NEGm16 : IM16<"neg", 0xF7, MRM3m>, OpSize; // [mem16] = -[mem16] = 0-[mem16]
def NEGm32 : IM32<"neg", 0xF7, MRM3m>;         // [mem32] = -[mem32] = 0-[mem32]

def NOTr8  : I   <"not", 0xF6, MRM2r>;         // R8  = ~R8  = R8^-1
def NOTr16 : I   <"not", 0xF7, MRM2r>, OpSize; // R16 = ~R16 = R16^-1
def NOTr32 : I   <"not", 0xF7, MRM2r>;         // R32 = ~R32 = R32^-1
def NOTm8  : IM8 <"not", 0xF6, MRM2m>;         // [mem8]  = ~[mem8]  = [mem8^-1]
def NOTm16 : IM16<"not", 0xF7, MRM2m>, OpSize; // [mem16] = ~[mem16] = [mem16^-1]
def NOTm32 : IM32<"not", 0xF7, MRM2m>;         // [mem32] = ~[mem32] = [mem32^-1]

def INCr8  : I   <"inc", 0xFE, MRM0r>;         // ++R8
def INCr16 : I   <"inc", 0xFF, MRM0r>, OpSize; // ++R16
def INCr32 : I   <"inc", 0xFF, MRM0r>;         // ++R32
def INCm8  : IM8 <"inc", 0xFE, MRM0m>;         // ++R8
def INCm16 : IM16<"inc", 0xFF, MRM0m>, OpSize; // ++R16
def INCm32 : IM32<"inc", 0xFF, MRM0m>;         // ++R32

def DECr8  : I   <"dec", 0xFE, MRM1r>;         // --R8
def DECr16 : I   <"dec", 0xFF, MRM1r>, OpSize; // --R16
def DECr32 : I   <"dec", 0xFF, MRM1r>;         // --R32
def DECm8  : IM8 <"dec", 0xFE, MRM1m>;         // --[mem8]
def DECm16 : IM16<"dec", 0xFF, MRM1m>, OpSize; // --[mem16]
def DECm32 : IM32<"dec", 0xFF, MRM1m>;         // --[mem32]

// Logical operators...
def ANDrr8   : I     <"and", 0x20, MRMDestReg>,         Pattern<(set R8 , (and R8 , R8 ))>;
def ANDrr16  : I     <"and", 0x21, MRMDestReg>, OpSize, Pattern<(set R16, (and R16, R16))>;
def ANDrr32  : I     <"and", 0x21, MRMDestReg>,         Pattern<(set R32, (and R32, R32))>;
def ANDmr8   : IM8   <"and", 0x20, MRMDestMem>;            // [mem8]  &= R8
def ANDmr16  : IM16  <"and", 0x21, MRMDestMem>, OpSize;    // [mem16] &= R16
def ANDmr32  : IM32  <"and", 0x21, MRMDestMem>;            // [mem32] &= R32
def ANDrm8   : IM8   <"and", 0x22, MRMSrcMem >;            // R8  &= [mem8]
def ANDrm16  : IM16  <"and", 0x23, MRMSrcMem >, OpSize;    // R16 &= [mem16]
def ANDrm32  : IM32  <"and", 0x23, MRMSrcMem >;            // R32 &= [mem32]

def ANDri8   : II8   <"and", 0x80, MRM4r     >,         Pattern<(set R8 , (and R8 , imm))>;
def ANDri16  : II16  <"and", 0x81, MRM4r     >, OpSize, Pattern<(set R16, (and R16, imm))>;
def ANDri32  : II32  <"and", 0x81, MRM4r     >,         Pattern<(set R32, (and R32, imm))>;
def ANDmi8   : I8MI  <"and", 0x80, MRM4m     >;            // [mem8]  &= imm8
def ANDmi16  : I16MI <"and", 0x81, MRM4m     >, OpSize;    // [mem16] &= imm16
def ANDmi32  : I32MI <"and", 0x81, MRM4m     >;            // [mem32] &= imm32

def ANDri16b : II8   <"and", 0x83, MRM4r     >, OpSize;    // R16 &= imm8
def ANDri32b : II8   <"and", 0x83, MRM4r     >;            // R32 &= imm8
def ANDmi16b : IM16I8<"and", 0x83, MRM4m     >, OpSize;    // [mem16] &= imm8
def ANDmi32b : IM32I8<"and", 0x83, MRM4m     >;            // [mem32] &= imm8


def ORrr8    : I     <"or" , 0x08, MRMDestReg>,         Pattern<(set R8 , (or  R8 , R8 ))>;
def ORrr16   : I     <"or" , 0x09, MRMDestReg>, OpSize, Pattern<(set R16, (or  R16, R16))>;
def ORrr32   : I     <"or" , 0x09, MRMDestReg>,         Pattern<(set R32, (or  R32, R32))>;
def ORmr8    : IM8   <"or" , 0x08, MRMDestMem>;            // [mem8]  |= R8
def ORmr16   : IM16  <"or" , 0x09, MRMDestMem>, OpSize;    // [mem16] |= R16
def ORmr32   : IM32  <"or" , 0x09, MRMDestMem>;            // [mem32] |= R32
def ORrm8    : IM8   <"or" , 0x0A, MRMSrcMem >;            // R8  |= [mem8]
def ORrm16   : IM16  <"or" , 0x0B, MRMSrcMem >, OpSize;    // R16 |= [mem16]
def ORrm32   : IM32  <"or" , 0x0B, MRMSrcMem >;            // R32 |= [mem32]

def ORri8    : II8   <"or" , 0x80, MRM1r     >,         Pattern<(set R8 , (or  R8 , imm))>;
def ORri16   : II16  <"or" , 0x81, MRM1r     >, OpSize, Pattern<(set R16, (or  R16, imm))>;
def ORri32   : II32  <"or" , 0x81, MRM1r     >,         Pattern<(set R32, (or  R32, imm))>;
def ORmi8    : I8MI  <"or" , 0x80, MRM1m     >;            // [mem8]  |= imm8
def ORmi16   : I16MI <"or" , 0x81, MRM1m     >, OpSize;    // [mem16] |= imm16
def ORmi32   : I32MI <"or" , 0x81, MRM1m     >;            // [mem32] |= imm32

def ORri16b  : II8   <"or" , 0x83, MRM1r     >, OpSize;    // R16 |= imm8
def ORri32b  : II8   <"or" , 0x83, MRM1r     >;            // R32 |= imm8
def ORmi16b  : IM16I8<"or" , 0x83, MRM1m     >, OpSize;    // [mem16] |= imm8
def ORmi32b  : IM32I8<"or" , 0x83, MRM1m     >;            // [mem32] |= imm8


def XORrr8   : I     <"xor", 0x30, MRMDestReg>,         Pattern<(set R8 , (xor R8 , R8 ))>;
def XORrr16  : I     <"xor", 0x31, MRMDestReg>, OpSize, Pattern<(set R16, (xor R16, R16))>;
def XORrr32  : I     <"xor", 0x31, MRMDestReg>,         Pattern<(set R32, (xor R32, R32))>;
def XORmr8   : IM8   <"xor", 0x30, MRMDestMem>;            // [mem8]  ^= R8
def XORmr16  : IM16  <"xor", 0x31, MRMDestMem>, OpSize;    // [mem16] ^= R16
def XORmr32  : IM32  <"xor", 0x31, MRMDestMem>;            // [mem32] ^= R32
def XORrm8   : IM8   <"xor", 0x32, MRMSrcMem >;            // R8  ^= [mem8]
def XORrm16  : IM16  <"xor", 0x33, MRMSrcMem >, OpSize;    // R16 ^= [mem16]
def XORrm32  : IM32  <"xor", 0x33, MRMSrcMem >;            // R32 ^= [mem32]

def XORri8   : II8   <"xor", 0x80, MRM6r     >,         Pattern<(set R8 , (xor R8 , imm))>;
def XORri16  : II16  <"xor", 0x81, MRM6r     >, OpSize, Pattern<(set R16, (xor R16, imm))>;
def XORri32  : II32  <"xor", 0x81, MRM6r     >,         Pattern<(set R32, (xor R32, imm))>;
def XORmi8   : I8MI  <"xor", 0x80, MRM6m     >;            // [mem8] ^= R8
def XORmi16  : I16MI <"xor", 0x81, MRM6m     >, OpSize;    // [mem16] ^= R16
def XORmi32  : I32MI <"xor", 0x81, MRM6m     >;            // [mem32] ^= R32

def XORri16b : II8   <"xor", 0x83, MRM6r     >, OpSize;    // R16 ^= imm8
def XORri32b : II8   <"xor", 0x83, MRM6r     >;            // R32 ^= imm8
def XORmi16b : IM16I8<"xor", 0x83, MRM6m     >, OpSize;    // [mem16] ^= imm8
def XORmi32b : IM32I8<"xor", 0x83, MRM6m     >;            // [mem32] ^= imm8

// Shift instructions
def SHLrCL8  : I     <"shl", 0xD2, MRM4r     >        , UsesCL; // R8  <<= cl
def SHLrCL16 : I     <"shl", 0xD3, MRM4r     >, OpSize, UsesCL; // R16 <<= cl
def SHLrCL32 : I     <"shl", 0xD3, MRM4r     >        , UsesCL; // R32 <<= cl
def SHLmCL8  : IM8   <"shl", 0xD2, MRM4m     >        , UsesCL; // [mem8]  <<= cl
def SHLmCL16 : IM16  <"shl", 0xD3, MRM4m     >, OpSize, UsesCL; // [mem16] <<= cl
def SHLmCL32 : IM32  <"shl", 0xD3, MRM4m     >        , UsesCL; // [mem32] <<= cl

def SHLri8   : II8   <"shl", 0xC0, MRM4r     >;                 // R8  <<= imm8
def SHLri16  : II8   <"shl", 0xC1, MRM4r     >, OpSize;         // R16 <<= imm8
def SHLri32  : II8   <"shl", 0xC1, MRM4r     >;                 // R32 <<= imm8
def SHLmi8   : I8MI  <"shl", 0xC0, MRM4m     >;                 // [mem8]  <<= imm8
def SHLmi16  : IM16I8<"shl", 0xC1, MRM4m     >, OpSize;         // [mem16] <<= imm8
def SHLmi32  : IM32I8<"shl", 0xC1, MRM4m     >;                 // [mem32] <<= imm8

def SHRrCL8  : I     <"shr", 0xD2, MRM5r     >        , UsesCL; // R8  >>= cl
def SHRrCL16 : I     <"shr", 0xD3, MRM5r     >, OpSize, UsesCL; // R16 >>= cl
def SHRrCL32 : I     <"shr", 0xD3, MRM5r     >        , UsesCL; // R32 >>= cl
def SHRmCL8  : IM8   <"shr", 0xD2, MRM5m     >        , UsesCL; // [mem8]  >>= cl
def SHRmCL16 : IM16  <"shr", 0xD3, MRM5m     >, OpSize, UsesCL; // [mem16] >>= cl
def SHRmCL32 : IM32  <"shr", 0xD3, MRM5m     >        , UsesCL; // [mem32] >>= cl

def SHRri8   : II8   <"shr", 0xC0, MRM5r     >;                 // R8  >>= imm8
def SHRri16  : II8   <"shr", 0xC1, MRM5r     >, OpSize;         // R16 >>= imm8
def SHRri32  : II8   <"shr", 0xC1, MRM5r     >;                 // R32 >>= imm8
def SHRmi8   : I8MI  <"shr", 0xC0, MRM5m     >;                 // [mem8]  >>= imm8
def SHRmi16  : IM16I8<"shr", 0xC1, MRM5m     >, OpSize;         // [mem16] >>= imm8
def SHRmi32  : IM32I8<"shr", 0xC1, MRM5m     >;                 // [mem32] >>= imm8

def SARrCL8  : I     <"sar", 0xD2, MRM7r     >        , UsesCL; // R8  >>>= cl
def SARrCL16 : I     <"sar", 0xD3, MRM7r     >, OpSize, UsesCL; // R16 >>>= cl
def SARrCL32 : I     <"sar", 0xD3, MRM7r     >        , UsesCL; // R32 >>>= cl
def SARmCL8  : IM8   <"sar", 0xD2, MRM7m     >        , UsesCL; // [mem8]  >>>= cl
def SARmCL16 : IM16  <"sar", 0xD3, MRM7m     >, OpSize, UsesCL; // [mem16] >>>= cl
def SARmCL32 : IM32  <"sar", 0xD3, MRM7m     >        , UsesCL; // [mem32] >>>= cl

def SARri8   : II8   <"sar", 0xC0, MRM7r     >;                 // R8  >>>= imm8
def SARri16  : II8   <"sar", 0xC1, MRM7r     >, OpSize;         // R16 >>>= imm8
def SARri32  : II8   <"sar", 0xC1, MRM7r     >;                 // R32 >>>= imm8
def SARmi8   : I8MI  <"sar", 0xC0, MRM7m     >;                 // [mem8]  >>>= imm8
def SARmi16  : IM16I8<"sar", 0xC1, MRM7m     >, OpSize;         // [mem16] >>>= imm8
def SARmi32  : IM32I8<"sar", 0xC1, MRM7m     >;                 // [mem32] >>>= imm8

def SHLDrrCL32 : I   <"shld", 0xA5, MRMDestReg>, TB, UsesCL;    // R32 <<= R32,R32 cl
def SHLDmrCL32 : I   <"shld", 0xA5, MRMDestMem>, TB, UsesCL;    // [mem32] <<= [mem32],R32 cl
def SHLDrri32  : II8 <"shld", 0xA4, MRMDestReg>, TB;            // R32 <<= R32,R32 imm8
def SHLDmri32  : II8 <"shld", 0xA4, MRMDestMem>, TB;            // [mem32] <<= [mem32],R32 imm8

def SHRDrrCL32 : I   <"shrd", 0xAD, MRMDestReg>, TB, UsesCL;    // R32 >>= R32,R32 cl
def SHRDmrCL32 : I   <"shrd", 0xAD, MRMDestMem>, TB, UsesCL;    // [mem32] >>= [mem32],R32 cl
def SHRDrri32  : II8 <"shrd", 0xAC, MRMDestReg>, TB;            // R32 >>= R32,R32 imm8
def SHRDmri32  : II8 <"shrd", 0xAC, MRMDestMem>, TB;            // [mem32] >>= [mem32],R32 imm8


// Arithmetic...
def ADDrr8   : I     <"add", 0x00, MRMDestReg>,         Pattern<(set R8 , (plus R8 , R8 ))>;
def ADDrr16  : I     <"add", 0x01, MRMDestReg>, OpSize, Pattern<(set R16, (plus R16, R16))>;
def ADDrr32  : I     <"add", 0x01, MRMDestReg>,         Pattern<(set R32, (plus R32, R32))>;
def ADDmr8   : IM8   <"add", 0x00, MRMDestMem>;         // [mem8]  += R8
def ADDmr16  : IM16  <"add", 0x01, MRMDestMem>, OpSize; // [mem16] += R16
def ADDmr32  : IM32  <"add", 0x01, MRMDestMem>;         // [mem32] += R32
def ADDrm8   : IM8   <"add", 0x02, MRMSrcMem >;         // R8  += [mem8]
def ADDrm16  : IM16  <"add", 0x03, MRMSrcMem >, OpSize; // R16 += [mem16]
def ADDrm32  : IM32  <"add", 0x03, MRMSrcMem >;         // R32 += [mem32]

def ADDri8   : II8   <"add", 0x80, MRM0r     >,         Pattern<(set R8 , (plus R8 , imm))>;
def ADDri16  : II16  <"add", 0x81, MRM0r     >, OpSize, Pattern<(set R16, (plus R16, imm))>;
def ADDri32  : II32  <"add", 0x81, MRM0r     >,         Pattern<(set R32, (plus R32, imm))>;
def ADDmi8   : I8MI  <"add", 0x80, MRM0m     >;         // [mem8] += I8
def ADDmi16  : I16MI <"add", 0x81, MRM0m     >, OpSize; // [mem16] += I16
def ADDmi32  : I32MI <"add", 0x81, MRM0m     >;         // [mem32] += I32

def ADDri16b : II8   <"add", 0x83, MRM0r     >, OpSize;   // ADDri with sign extended 8 bit imm
def ADDri32b : II8   <"add", 0x83, MRM0r     >;
def ADDmi16b : IM16I8<"add", 0x83, MRM0m     >, OpSize; // [mem16] += I8
def ADDmi32b : IM32I8<"add", 0x83, MRM0m     >;         // [mem32] += I8

def ADCrr32  : I     <"adc", 0x11, MRMDestReg>;         // R32 += R32+Carry
def ADCrm32  : I     <"adc", 0x11, MRMSrcMem >;         // R32 += [mem32]+Carry
def ADCmr32  : I     <"adc", 0x13, MRMDestMem>;         // [mem32] += R32+Carry


def SUBrr8   : I     <"sub", 0x28, MRMDestReg>,         Pattern<(set R8 , (minus R8 , R8 ))>;
def SUBrr16  : I     <"sub", 0x29, MRMDestReg>, OpSize, Pattern<(set R16, (minus R16, R16))>;
def SUBrr32  : I     <"sub", 0x29, MRMDestReg>,         Pattern<(set R32, (minus R32, R32))>;
def SUBmr8   : IM8   <"sub", 0x28, MRMDestMem>;         // [mem8]  -= R8
def SUBmr16  : IM16  <"sub", 0x29, MRMDestMem>, OpSize; // [mem16] -= R16
def SUBmr32  : IM32  <"sub", 0x29, MRMDestMem>;         // [mem32] -= R32
def SUBrm8   : IM8   <"sub", 0x2A, MRMSrcMem >;         // R8  -= [mem8]
def SUBrm16  : IM16  <"sub", 0x2B, MRMSrcMem >, OpSize; // R16 -= [mem16]
def SUBrm32  : IM32  <"sub", 0x2B, MRMSrcMem >;         // R32 -= [mem32]

def SUBri8   : II8   <"sub", 0x80, MRM5r     >,         Pattern<(set R8 , (minus R8 , imm))>;
def SUBri16  : II16  <"sub", 0x81, MRM5r     >, OpSize, Pattern<(set R16, (minus R16, imm))>;
def SUBri32  : II32  <"sub", 0x81, MRM5r     >,         Pattern<(set R32, (minus R32, imm))>;
def SUBmi8   : I8MI  <"sub", 0x80, MRM5m     >;         // [mem8] -= I8
def SUBmi16  : I16MI <"sub", 0x81, MRM5m     >, OpSize; // [mem16] -= I16
def SUBmi32  : I32MI <"sub", 0x81, MRM5m     >;         // [mem32] -= I32

def SUBri16b : II8   <"sub", 0x83, MRM5r     >, OpSize;
def SUBri32b : II8   <"sub", 0x83, MRM5r     >;
def SUBmi16b : IM16I8<"sub", 0x83, MRM5m     >, OpSize; // [mem16] -= I8
def SUBmi32b : IM32I8<"sub", 0x83, MRM5m     >;         // [mem32] -= I8

def SBBrr32  : I     <"sbb", 0x19, MRMDestReg>;         // R32 -= R32+Borrow
def SBBrm32  : IM32  <"sbb", 0x19, MRMSrcMem >;         // R32 -= [mem32]+Borrow
def SBBmr32  : IM32  <"sbb", 0x1B, MRMDestMem>;         // [mem32] -= R32+Borrow

def IMULrr16 : I     <"imul", 0xAF, MRMSrcReg>, TB, OpSize, Pattern<(set R16, (times R16, R16))>;
def IMULrr32 : I     <"imul", 0xAF, MRMSrcReg>, TB        , Pattern<(set R32, (times R32, R32))>;
def IMULrm16 : IM16  <"imul", 0xAF, MRMSrcMem>, TB, OpSize;
def IMULrm32 : IM32  <"imul", 0xAF, MRMSrcMem>, TB        ;

} // end Two Address instructions

// These are suprisingly enough not two address instructions!
def IMULrri16  : II16  <"imul", 0x69, MRMSrcReg>,     OpSize;  // R16 = R16*I16
def IMULrri32  : II32  <"imul", 0x69, MRMSrcReg>;              // R32 = R32*I32
def IMULrri16b : II8   <"imul", 0x6B, MRMSrcReg>,     OpSize;  // R16 = R16*I8
def IMULrri32b : II8   <"imul", 0x6B, MRMSrcReg>;              // R32 = R32*I8
def IMULrmi16  : I16MI <"imul", 0x69, MRMSrcMem>,     OpSize;  // R16 = [mem16]*I16
def IMULrmi32  : I32MI <"imul", 0x69, MRMSrcMem>;              // R32 = [mem32]*I32
def IMULrmi16b : IM16I8<"imul", 0x6B, MRMSrcMem>,     OpSize;  // R16 = [mem16]*I8
def IMULrmi32b : IM32I8<"imul", 0x6B, MRMSrcMem>;              // R32 = [mem32]*I8

//===----------------------------------------------------------------------===//
// Test instructions are just like AND, except they don't generate a result.
def TESTrr8  : I    <"test", 0x84, MRMDestReg>;          // flags = R8  & R8
def TESTrr16 : I    <"test", 0x85, MRMDestReg>, OpSize;  // flags = R16 & R16
def TESTrr32 : I    <"test", 0x85, MRMDestReg>;          // flags = R32 & R32
def TESTmr8  : IM8  <"test", 0x84, MRMDestMem>;          // flags = [mem8]  & R8
def TESTmr16 : IM16 <"test", 0x85, MRMDestMem>, OpSize;  // flags = [mem16] & R16
def TESTmr32 : IM32 <"test", 0x85, MRMDestMem>;          // flags = [mem32] & R32
def TESTrm8  : IM8  <"test", 0x84, MRMSrcMem >;          // flags = R8  & [mem8]
def TESTrm16 : IM16 <"test", 0x85, MRMSrcMem >, OpSize;  // flags = R16 & [mem16]
def TESTrm32 : IM32 <"test", 0x85, MRMSrcMem >;          // flags = R32 & [mem32]

def TESTri8  : II8  <"test", 0xF6, MRM0r     >;          // flags = R8  & imm8
def TESTri16 : II16 <"test", 0xF7, MRM0r     >, OpSize;  // flags = R16 & imm16
def TESTri32 : II32 <"test", 0xF7, MRM0r     >;          // flags = R32 & imm32
def TESTmi8  : I8MI <"test", 0xF6, MRM0m     >;          // flags = [mem8]  & imm8
def TESTmi16 : I16MI<"test", 0xF7, MRM0m     >, OpSize;  // flags = [mem16] & imm16
def TESTmi32 : I32MI<"test", 0xF7, MRM0m     >;          // flags = [mem32] & imm32



// Condition code ops, incl. set if equal/not equal/...
def SAHF     : I  <"sahf" , 0x9E, RawFrm>, Imp<[AH],[]>;  // flags = AH

def SETBr    : I  <"setb" , 0x92, MRM0r>, TB;            // R8 = <  unsign
def SETBm    : IM8<"setb" , 0x92, MRM0m>, TB;            // [mem8] = <  unsign
def SETAEr   : I  <"setae", 0x93, MRM0r>, TB;            // R8 = >= unsign
def SETAEm   : IM8<"setae", 0x93, MRM0m>, TB;            // [mem8] = >= unsign
def SETEr    : I  <"sete" , 0x94, MRM0r>, TB;            // R8 = ==
def SETEm    : IM8<"sete" , 0x94, MRM0m>, TB;            // [mem8] = ==
def SETNEr   : I  <"setne", 0x95, MRM0r>, TB;            // R8 = !=
def SETNEm   : IM8<"setne", 0x95, MRM0m>, TB;            // [mem8] = !=
def SETBEr   : I  <"setbe", 0x96, MRM0r>, TB;            // R8 = <= unsign
def SETBEm   : IM8<"setbe", 0x96, MRM0m>, TB;            // [mem8] = <= unsign
def SETAr    : I  <"seta" , 0x97, MRM0r>, TB;            // R8 = >  signed
def SETAm    : IM8<"seta" , 0x97, MRM0m>, TB;            // [mem8] = >  signed
def SETSr    : I  <"sets" , 0x98, MRM0r>, TB;            // R8 = <sign bit>
def SETSm    : IM8<"sets" , 0x98, MRM0m>, TB;            // [mem8] = <sign bit>
def SETNSr   : I  <"setns", 0x99, MRM0r>, TB;            // R8 = !<sign bit>
def SETNSm   : IM8<"setns", 0x99, MRM0m>, TB;            // [mem8] = !<sign bit>
def SETLr    : I  <"setl" , 0x9C, MRM0r>, TB;            // R8 = <  signed
def SETLm    : IM8<"setl" , 0x9C, MRM0m>, TB;            // [mem8] = <  signed
def SETGEr   : I  <"setge", 0x9D, MRM0r>, TB;            // R8 = >= signed
def SETGEm   : IM8<"setge", 0x9D, MRM0m>, TB;            // [mem8] = >= signed
def SETLEr   : I  <"setle", 0x9E, MRM0r>, TB;            // R8 = <= signed
def SETLEm   : IM8<"setle", 0x9E, MRM0m>, TB;            // [mem8] = <= signed
def SETGr    : I  <"setg" , 0x9F, MRM0r>, TB;            // R8 = <  signed
def SETGm    : IM8<"setg" , 0x9F, MRM0m>, TB;            // [mem8] = <  signed

// Integer comparisons
def CMPrr8  : I    <"cmp", 0x38, MRMDestReg>;              // compare R8, R8
def CMPrr16 : I    <"cmp", 0x39, MRMDestReg>, OpSize;      // compare R16, R16
def CMPrr32 : I    <"cmp", 0x39, MRMDestReg>,              // compare R32, R32
              Pattern<(isVoid (unspec2 R32, R32))>;
def CMPmr8  : IM8  <"cmp", 0x38, MRMDestMem>;              // compare [mem8], R8
def CMPmr16 : IM16 <"cmp", 0x39, MRMDestMem>, OpSize;      // compare [mem16], R16
def CMPmr32 : IM32 <"cmp", 0x39, MRMDestMem>;              // compare [mem32], R32
def CMPrm8  : IM8  <"cmp", 0x3A, MRMSrcMem >;              // compare R8, [mem8]
def CMPrm16 : IM16 <"cmp", 0x3B, MRMSrcMem >, OpSize;      // compare R16, [mem16]
def CMPrm32 : IM32 <"cmp", 0x3B, MRMSrcMem >;              // compare R32, [mem32]
def CMPri8  : II8  <"cmp", 0x80, MRM7r     >;              // compare R8, imm8
def CMPri16 : II16 <"cmp", 0x81, MRM7r     >, OpSize;      // compare R16, imm16
def CMPri32 : II32 <"cmp", 0x81, MRM7r     >;              // compare R32, imm32
def CMPmi8  : I8MI <"cmp", 0x80, MRM7m     >;              // compare [mem8], imm8
def CMPmi16 : I16MI<"cmp", 0x81, MRM7m     >, OpSize;      // compare [mem16], imm16
def CMPmi32 : I32MI<"cmp", 0x81, MRM7m     >;              // compare [mem32], imm32

// Sign/Zero extenders
def MOVSXr16r8 : I   <"movsx", 0xBE, MRMSrcReg>, TB, OpSize; // R16 = signext(R8)
def MOVSXr32r8 : I   <"movsx", 0xBE, MRMSrcReg>, TB;         // R32 = signext(R8)
def MOVSXr32r16: I   <"movsx", 0xBF, MRMSrcReg>, TB;         // R32 = signext(R16)
def MOVSXr16m8 : IM8 <"movsx", 0xBE, MRMSrcMem>, TB, OpSize; // R16 = signext([mem8])
def MOVSXr32m8 : IM8 <"movsx", 0xBE, MRMSrcMem>, TB;         // R32 = signext([mem8])
def MOVSXr32m16: IM16<"movsx", 0xBF, MRMSrcMem>, TB;         // R32 = signext([mem16])

def MOVZXr16r8 : I   <"movzx", 0xB6, MRMSrcReg>, TB, OpSize; // R16 = zeroext(R8)
def MOVZXr32r8 : I   <"movzx", 0xB6, MRMSrcReg>, TB;         // R32 = zeroext(R8)
def MOVZXr32r16: I   <"movzx", 0xB7, MRMSrcReg>, TB;         // R32 = zeroext(R16)
def MOVZXr16m8 : IM8 <"movzx", 0xB6, MRMSrcMem>, TB, OpSize; // R16 = zeroext([mem8])
def MOVZXr32m8 : IM8 <"movzx", 0xB6, MRMSrcMem>, TB;         // R32 = zeroext([mem8])
def MOVZXr32m16: IM16<"movzx", 0xB7, MRMSrcMem>, TB;         // R32 = zeroext([mem16])


//===----------------------------------------------------------------------===//
// Floating point support
//===----------------------------------------------------------------------===//

// FIXME: These need to indicate mod/ref sets for FP regs... & FP 'TOP'

// Floating point instruction templates
class FPInst<string n, bits<8> o, Format F, FPFormat fp, MemType m, ImmType i>
  : X86Inst<n, o, F, m, i> { let FPForm = fp; let FPFormBits = FPForm.Value; }

class FPI<string n, bits<8> o, Format F, FPFormat fp> : FPInst<n, o, F, fp, NoMem, NoImm>;

class FPIM<string n, bits<8> o, Format F, FPFormat fp, MemType m> : FPInst<n, o, F, fp, m, NoImm>;

class FPIM16<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem16>;
class FPIM32<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem32>;
class FPIM64<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem64>;
class FPIM80<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem80>;

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

// Floating point loads & stores...
def FLDrr   : FPI    <"fld"   , 0xC0, AddRegFrm, NotFP>, D9;        // push(ST(i))
def FLDr32  : FPIM32 <"fld"   , 0xD9, MRM0m    , ZeroArgFP>;        // load float
def FLDr64  : FPIM64 <"fld"   , 0xDD, MRM0m    , ZeroArgFP>;        // load double
def FLDr80  : FPIM80 <"fld"   , 0xDB, MRM5m    , ZeroArgFP>;        // load extended
def FILDr16 : FPIM16 <"fild"  , 0xDF, MRM0m    , ZeroArgFP>;        // load signed short
def FILDr32 : FPIM32 <"fild"  , 0xDB, MRM0m    , ZeroArgFP>;        // load signed int
def FILDr64 : FPIM64 <"fild"  , 0xDF, MRM5m    , ZeroArgFP>;        // load signed long

def FSTrr    : FPI    <"fst" , 0xD0, AddRegFrm, NotFP   >, DD;      // ST(i) = ST(0)
def FSTPrr   : FPI    <"fstp", 0xD8, AddRegFrm, NotFP   >, DD;      // ST(i) = ST(0), pop
def FSTr32   : FPIM32 <"fst" , 0xD9, MRM2m    , OneArgFP>;          // store float
def FSTr64   : FPIM64 <"fst" , 0xDD, MRM2m    , OneArgFP>;          // store double
def FSTPr32  : FPIM32 <"fstp", 0xD9, MRM3m    , OneArgFP>;          // store float, pop
def FSTPr64  : FPIM64 <"fstp", 0xDD, MRM3m    , OneArgFP>;          // store double, pop
def FSTPr80  : FPIM80 <"fstp", 0xDB, MRM7m    , OneArgFP>;          // store extended, pop

def FISTr16  : FPIM16 <"fist",    0xDF, MRM2m , OneArgFP>;          // store signed short
def FISTr32  : FPIM32 <"fist",    0xDB, MRM2m , OneArgFP>;          // store signed int
def FISTPr16 : FPIM16 <"fistp",   0xDF, MRM3m , NotFP   >;          // store signed short, pop
def FISTPr32 : FPIM32 <"fistp",   0xDB, MRM3m , NotFP   >;          // store signed int, pop
def FISTPr64 : FPIM64 <"fistpll", 0xDF, MRM7m , OneArgFP>;          // store signed long, pop

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
  bit printImplicitUses = 1;
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
def FNSTSWr8  : I   <"fnstsw" , 0xE0, RawFrm>, DF, Imp<[],[AX]>;   // AX = fp flags
def FNSTCWm16 : IM16<"fnstcw" , 0xD9, MRM7m >;                     // [mem16] = X87 control world
def FLDCWm16  : IM16<"fldcw"  , 0xD9, MRM5m >;                     // X87 control world = [mem16]


//===----------------------------------------------------------------------===//
//  Instruction Expanders
//

def RET_R32 : Expander<(ret R32:$reg),
                       [(MOVrr32 EAX, R32:$reg),
                        (RET)]>;

// FIXME: This should eventually just be implemented by defining a frameidx as a
// value address for a load.
def LOAD_FI16 : Expander<(set R16:$dest, (load frameidx:$fi)),
                         [(MOVrm16 R16:$dest, frameidx:$fi, 1, 0/*NoReg*/, 0)]>;

def LOAD_FI32 : Expander<(set R32:$dest, (load frameidx:$fi)),
                         [(MOVrm32 R32:$dest, frameidx:$fi, 1, 0/*NoReg*/, 0)]>;


def LOAD_R16 : Expander<(set R16:$dest, (load R32:$src)),
                         [(MOVrm16 R16:$dest, R32:$src, 1, 0/*NoReg*/, 0)]>;

def LOAD_R32 : Expander<(set R32:$dest, (load R32:$src)),
                         [(MOVrm32 R32:$dest, R32:$src, 1, 0/*NoReg*/, 0)]>;

def BR_EQ : Expander<(brcond (seteq R32:$a1, R32:$a2),
                             basicblock:$d1, basicblock:$d2),
                     [(CMPrr32 R32:$a1, R32:$a2),
                      (JE basicblock:$d1),
                      (JMP basicblock:$d2)]>;
