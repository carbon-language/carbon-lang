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

// ArgType - This specifies the argument type used by an instruction. This is
// part of the ad-hoc solution used to emit machine instruction encodings by our
// machine code emitter.
class ArgType<bits<3> val> {
  bits<3> Value = val;
}
def NoArg  : ArgType<0>;
def Arg8   : ArgType<1>;
def Arg16  : ArgType<2>;
def Arg32  : ArgType<3>;
def Arg64  : ArgType<4>;   // 64 bit int argument for FILD64
def ArgF32 : ArgType<5>;
def ArgF64 : ArgType<6>;
def ArgF80 : ArgType<6>;

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


class X86Inst<string nam, bits<8> opcod, Format f, ArgType a> : Instruction {
  let Namespace = "X86";

  let Name = nam;
  bits<8> Opcode = opcod;
  Format Form = f;
  bits<5> FormBits = Form.Value;
  ArgType Type = a;
  bits<3> TypeBits = Type.Value;

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
// Instruction list...
//

def PHI : X86Inst<"PHI", 0, Pseudo, NoArg>;          // PHI node...

def NOOP : X86Inst<"nop", 0x90, RawFrm, NoArg>;    // nop

def ADJCALLSTACKDOWN : X86Inst<"ADJCALLSTACKDOWN", 0, Pseudo, NoArg>;
def ADJCALLSTACKUP   : X86Inst<"ADJCALLSTACKUP",   0, Pseudo, NoArg>;
def IMPLICIT_USE     : X86Inst<"IMPLICIT_USE",     0, Pseudo, NoArg>;
def IMPLICIT_DEF     : X86Inst<"IMPLICIT_DEF",     0, Pseudo, NoArg>;
let isTerminator = 1 in
  let Defs = [FP0, FP1, FP2, FP3, FP4, FP5, FP6] in
    def FP_REG_KILL    : X86Inst<"FP_REG_KILL",      0, Pseudo, NoArg>;
//===----------------------------------------------------------------------===//
//  Control Flow Instructions...
//

// Return instruction...
let isTerminator = 1, isReturn = 1 in
  def RET : X86Inst<"ret", 0xC3, RawFrm, NoArg>, Pattern<(retvoid)>;

// All branches are RawFrm, Void, Branch, and Terminators
let isBranch = 1, isTerminator = 1 in
  class IBr<string name, bits<8> opcode> : X86Inst<name, opcode, RawFrm, NoArg>;

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
    def CALLpcrel32 : X86Inst<"call", 0xE8, RawFrm, NoArg>;
    def CALLr32     : X86Inst<"call", 0xFF, MRM2r , Arg32>;
    def CALLm32     : X86Inst<"call", 0xFF, MRM2m , Arg32>;
  }

       
//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//
def LEAVE    : X86Inst<"leave", 0xC9, RawFrm, NoArg>, Imp<[EBP,ESP],[EBP,ESP]>;
def POPr32   : X86Inst<"pop",   0x58, AddRegFrm, Arg32>, Imp<[ESP],[ESP]>;

let isTwoAddress = 1 in                                      // R32 = bswap R32
  def BSWAPr32 : X86Inst<"bswap", 0xC8, AddRegFrm, Arg32>, TB;

def XCHGrr8  : X86Inst<"xchg", 0x86, MRMDestReg, Arg8>;         // xchg R8, R8
def XCHGrr16 : X86Inst<"xchg", 0x87, MRMDestReg, Arg16>, OpSize;// xchg R16, R16
def XCHGrr32 : X86Inst<"xchg", 0x87, MRMDestReg, Arg32>;        // xchg R32, R32
def XCHGmr8  : X86Inst<"xchg", 0x86, MRMDestMem, Arg8>;         // xchg [mem8], R8
def XCHGmr16 : X86Inst<"xchg", 0x87, MRMDestMem, Arg16>, OpSize;// xchg [mem16], R16
def XCHGmr32 : X86Inst<"xchg", 0x87, MRMDestMem, Arg32>;        // xchg [mem32], R32
def XCHGrm8  : X86Inst<"xchg", 0x86, MRMSrcMem , Arg8>;         // xchg R8, [mem8]
def XCHGrm16 : X86Inst<"xchg", 0x87, MRMSrcMem , Arg16>, OpSize;// xchg R16, [mem16]
def XCHGrm32 : X86Inst<"xchg", 0x87, MRMSrcMem , Arg32>;        // xchg R32, [mem32]

def LEAr16 : X86Inst<"lea", 0x8D, MRMSrcMem, Arg16>, OpSize; // R16 = lea [mem]
def LEAr32 : X86Inst<"lea", 0x8D, MRMSrcMem, Arg32>;         // R32 = lea [mem]


def REP_MOVSB : X86Inst<"rep movsb", 0xA4, RawFrm, NoArg>, REP,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>;
def REP_MOVSW : X86Inst<"rep movsw", 0xA5, RawFrm, NoArg>, REP, OpSize,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>;
def REP_MOVSD : X86Inst<"rep movsd", 0xA5, RawFrm, NoArg>, REP,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>;

def REP_STOSB : X86Inst<"rep stosb", 0xAA, RawFrm, NoArg>, REP,
                Imp<[AL,ECX,EDI], [ECX,EDI]>;
def REP_STOSW : X86Inst<"rep stosw", 0xAB, RawFrm, NoArg>, REP, OpSize,
                Imp<[AX,ECX,EDI], [ECX,EDI]>;
def REP_STOSD : X86Inst<"rep stosd", 0xAB, RawFrm, NoArg>, REP,
                Imp<[EAX,ECX,EDI], [ECX,EDI]>;

//===----------------------------------------------------------------------===//
//  Move Instructions...
//
def MOVrr8  : X86Inst<"mov", 0x88, MRMDestReg, Arg8>,          Pattern<(set R8 , R8 )>;
def MOVrr16 : X86Inst<"mov", 0x89, MRMDestReg, Arg16>, OpSize, Pattern<(set R16, R16)>;
def MOVrr32 : X86Inst<"mov", 0x89, MRMDestReg, Arg32>,         Pattern<(set R32, R32)>;
def MOVri8  : X86Inst<"mov", 0xB0, AddRegFrm , Arg8>,          Pattern<(set R8 , imm )>;
def MOVri16 : X86Inst<"mov", 0xB8, AddRegFrm , Arg16>, OpSize, Pattern<(set R16, imm)>;
def MOVri32 : X86Inst<"mov", 0xB8, AddRegFrm , Arg32>,         Pattern<(set R32, imm)>;
def MOVmi8  : X86Inst<"mov", 0xC6, MRM0m     , Arg8>;             // [mem8] = imm8
def MOVmi16 : X86Inst<"mov", 0xC7, MRM0m     , Arg16>, OpSize;    // [mem16] = imm16
def MOVmi32 : X86Inst<"mov", 0xC7, MRM0m     , Arg32>;            // [mem32] = imm32

def MOVrm8  : X86Inst<"mov", 0x8A, MRMSrcMem , Arg8>;             // R8  = [mem8]
def MOVrm16 : X86Inst<"mov", 0x8B, MRMSrcMem , Arg16>, OpSize,    // R16 = [mem16]
              Pattern<(set R16, (load (plus R32, (plus (times imm, R32), imm))))>;
def MOVrm32 : X86Inst<"mov", 0x8B, MRMSrcMem , Arg32>,            // R32 = [mem32]
              Pattern<(set R32, (load (plus R32, (plus (times imm, R32), imm))))>;

def MOVmr8  : X86Inst<"mov", 0x88, MRMDestMem, Arg8>;             // [mem8] = R8
def MOVmr16 : X86Inst<"mov", 0x89, MRMDestMem, Arg16>, OpSize;    // [mem16] = R16
def MOVmr32 : X86Inst<"mov", 0x89, MRMDestMem, Arg32>;            // [mem32] = R32

//===----------------------------------------------------------------------===//
//  Fixed-Register Multiplication and Division Instructions...
//

// Extra precision multiplication
def MULr8  : X86Inst<"mul", 0xF6, MRM4r , Arg8 >, Imp<[AL],[AX]>;               // AL,AH = AL*R8
def MULr16 : X86Inst<"mul", 0xF7, MRM4r , Arg16>, Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*R16
def MULr32 : X86Inst<"mul", 0xF7, MRM4r , Arg32>, Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*R32
def MULm8  : X86Inst<"mul", 0xF6, MRM4m , Arg8 >, Imp<[AL],[AX]>;               // AL,AH = AL*[mem8]
def MULm16 : X86Inst<"mul", 0xF7, MRM4m , Arg16>, Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*[mem16]
def MULm32 : X86Inst<"mul", 0xF7, MRM4m , Arg32>, Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*[mem32]

// unsigned division/remainder
def DIVr8  : X86Inst<"div", 0xF6, MRM6r , Arg8 >, Imp<[AX],[AX]>;               // AX/r8 = AL,AH
def DIVr16 : X86Inst<"div", 0xF7, MRM6r , Arg16>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/r16 = AX,DX
def DIVr32 : X86Inst<"div", 0xF7, MRM6r , Arg32>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/r32 = EAX,EDX
def DIVm8  : X86Inst<"div", 0xF6, MRM6m , Arg8 >, Imp<[AX],[AX]>;               // AX/[mem8] = AL,AH
def DIVm16 : X86Inst<"div", 0xF7, MRM6m , Arg16>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/[mem16] = AX,DX
def DIVm32 : X86Inst<"div", 0xF7, MRM6m , Arg32>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/[mem32] = EAX,EDX

// signed division/remainder
def IDIVr8 : X86Inst<"idiv",0xF6, MRM7r , Arg8 >, Imp<[AX],[AX]>;               // AX/r8 = AL,AH
def IDIVr16: X86Inst<"idiv",0xF7, MRM7r , Arg16>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/r16 = AX,DX
def IDIVr32: X86Inst<"idiv",0xF7, MRM7r , Arg32>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/r32 = EAX,EDX
def IDIVm8 : X86Inst<"idiv",0xF6, MRM7m , Arg8 >, Imp<[AX],[AX]>;               // AX/[mem8] = AL,AH
def IDIVm16: X86Inst<"idiv",0xF7, MRM7m , Arg16>, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/[mem16] = AX,DX
def IDIVm32: X86Inst<"idiv",0xF7, MRM7m , Arg32>, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/[mem32] = EAX,EDX

// Sign-extenders for division
def CBW    : X86Inst<"cbw", 0x98, RawFrm, Arg8 >, Imp<[AL],[AH]>;               // AX = signext(AL)
def CWD    : X86Inst<"cwd", 0x99, RawFrm, Arg8 >, Imp<[AX],[DX]>;               // DX:AX = signext(AX)
def CDQ    : X86Inst<"cdq", 0x99, RawFrm, Arg8 >, Imp<[EAX],[EDX]>;             // EDX:EAX = signext(EAX)

//===----------------------------------------------------------------------===//
//  Two address Instructions...
//
let isTwoAddress = 1 in {  // Define some helper classes to make defs shorter.
  class I2A8 <string n, bits<8> o, Format F> : X86Inst<n, o, F, Arg8>;
  class I2A16<string n, bits<8> o, Format F> : X86Inst<n, o, F, Arg16>;
  class I2A32<string n, bits<8> o, Format F> : X86Inst<n, o, F, Arg32>;
}

// unary instructions
def NEGr8  : I2A8 <"neg", 0xF6, MRM3r >;         // R8  = -R8  = 0-R8
def NEGr16 : I2A16<"neg", 0xF7, MRM3r >, OpSize; // R16 = -R16 = 0-R16
def NEGr32 : I2A32<"neg", 0xF7, MRM3r >;         // R32 = -R32 = 0-R32
def NEGm8  : I2A8 <"neg", 0xF6, MRM3m >;         // [mem8]  = -[mem8]  = 0-[mem8]
def NEGm16 : I2A16<"neg", 0xF7, MRM3m >, OpSize; // [mem16] = -[mem16] = 0-[mem16]
def NEGm32 : I2A32<"neg", 0xF7, MRM3m >;         // [mem32] = -[mem32] = 0-[mem32]

def NOTr8  : I2A8 <"not", 0xF6, MRM2r >;         // R8  = ~R8  = R8^-1
def NOTr16 : I2A16<"not", 0xF7, MRM2r >, OpSize; // R16 = ~R16 = R16^-1
def NOTr32 : I2A32<"not", 0xF7, MRM2r >;         // R32 = ~R32 = R32^-1
def NOTm8  : I2A8 <"not", 0xF6, MRM2m >;         // [mem8]  = ~[mem8]  = [mem8^-1]
def NOTm16 : I2A16<"not", 0xF7, MRM2m >, OpSize; // [mem16] = ~[mem16] = [mem16^-1]
def NOTm32 : I2A32<"not", 0xF7, MRM2m >;         // [mem32] = ~[mem32] = [mem32^-1]

def INCr8  : I2A8 <"inc", 0xFE, MRM0r >;         // ++R8
def INCr16 : I2A16<"inc", 0xFF, MRM0r >, OpSize; // ++R16
def INCr32 : I2A32<"inc", 0xFF, MRM0r >;         // ++R32
def INCm8  : I2A8 <"inc", 0xFE, MRM0m >;         // ++R8
def INCm16 : I2A16<"inc", 0xFF, MRM0m >, OpSize; // ++R16
def INCm32 : I2A32<"inc", 0xFF, MRM0m >;         // ++R32

def DECr8  : I2A8 <"dec", 0xFE, MRM1r >;         // --R8
def DECr16 : I2A16<"dec", 0xFF, MRM1r >, OpSize; // --R16
def DECr32 : I2A32<"dec", 0xFF, MRM1r >;         // --R32
def DECm8  : I2A8 <"dec", 0xFE, MRM1m >;         // --[mem8]
def DECm16 : I2A16<"dec", 0xFF, MRM1m >, OpSize; // --[mem16]
def DECm32 : I2A32<"dec", 0xFF, MRM1m >;         // --[mem32]



// Arithmetic...
def ADDrr8   : I2A8 <"add", 0x00, MRMDestReg>,         Pattern<(set R8 , (plus R8 , R8 ))>;
def ADDrr16  : I2A16<"add", 0x01, MRMDestReg>, OpSize, Pattern<(set R16, (plus R16, R16))>;
def ADDrr32  : I2A32<"add", 0x01, MRMDestReg>,         Pattern<(set R32, (plus R32, R32))>;
def ADDmr8   : I2A8 <"add", 0x00, MRMDestMem>;         // [mem8]  += R8
def ADDmr16  : I2A16<"add", 0x01, MRMDestMem>, OpSize; // [mem16] += R16
def ADDmr32  : I2A32<"add", 0x01, MRMDestMem>;         // [mem32] += R32
def ADDrm8   : I2A8 <"add", 0x02, MRMSrcMem >;         // R8  += [mem8]
def ADDrm16  : I2A16<"add", 0x03, MRMSrcMem >, OpSize; // R16 += [mem16]
def ADDrm32  : I2A32<"add", 0x03, MRMSrcMem >;         // R32 += [mem32]

def ADDri8   : I2A8 <"add", 0x80, MRM0r     >,         Pattern<(set R8 , (plus R8 , imm))>;
def ADDri16  : I2A16<"add", 0x81, MRM0r     >, OpSize, Pattern<(set R16, (plus R16, imm))>;
def ADDri32  : I2A32<"add", 0x81, MRM0r     >,         Pattern<(set R32, (plus R32, imm))>;
def ADDmi8   : I2A8 <"add", 0x80, MRM0m     >;         // [mem8] += I8
def ADDmi16  : I2A16<"add", 0x81, MRM0m     >, OpSize; // [mem16] += I16
def ADDmi32  : I2A32<"add", 0x81, MRM0m     >;         // [mem32] += I32

def ADDri16b : I2A8 <"add", 0x83, MRM0r     >, OpSize;   // ADDri with sign extended 8 bit imm
def ADDri32b : I2A8 <"add", 0x83, MRM0r     >;
def ADDmi16b : I2A8 <"add", 0x83, MRM0m     >, OpSize; // [mem16] += I8
def ADDmi32b : I2A8 <"add", 0x83, MRM0m     >;         // [mem32] += I8

def ADCrr32  : I2A32<"adc", 0x11, MRMDestReg>;         // R32 += R32+Carry
def ADCrm32  : I2A32<"adc", 0x11, MRMSrcMem >;         // R32 += [mem32]+Carry
def ADCmr32  : I2A32<"adc", 0x13, MRMDestMem>;         // [mem32] += R32+Carry

def SUBrr8   : I2A8 <"sub", 0x28, MRMDestReg>,         Pattern<(set R8 , (minus R8 , R8 ))>;
def SUBrr16  : I2A16<"sub", 0x29, MRMDestReg>, OpSize, Pattern<(set R16, (minus R16, R16))>;
def SUBrr32  : I2A32<"sub", 0x29, MRMDestReg>,         Pattern<(set R32, (minus R32, R32))>;
def SUBmr8   : I2A8 <"sub", 0x28, MRMDestMem>;         // [mem8]  -= R8
def SUBmr16  : I2A16<"sub", 0x29, MRMDestMem>, OpSize; // [mem16] -= R16
def SUBmr32  : I2A32<"sub", 0x29, MRMDestMem>;         // [mem32] -= R32
def SUBrm8   : I2A8 <"sub", 0x2A, MRMSrcMem >;         // R8  -= [mem8]
def SUBrm16  : I2A16<"sub", 0x2B, MRMSrcMem >, OpSize; // R16 -= [mem16]
def SUBrm32  : I2A32<"sub", 0x2B, MRMSrcMem >;         // R32 -= [mem32]

def SUBri8   : I2A8 <"sub", 0x80, MRM5r     >,         Pattern<(set R8 , (minus R8 , imm))>;
def SUBri16  : I2A16<"sub", 0x81, MRM5r     >, OpSize, Pattern<(set R16, (minus R16, imm))>;
def SUBri32  : I2A32<"sub", 0x81, MRM5r     >,         Pattern<(set R32, (minus R32, imm))>;
def SUBmi8   : I2A8 <"sub", 0x80, MRM5m     >;         // [mem8] -= I8
def SUBmi16  : I2A16<"sub", 0x81, MRM5m     >, OpSize; // [mem16] -= I16
def SUBmi32  : I2A32<"sub", 0x81, MRM5m     >;         // [mem32] -= I32

def SUBri16b : I2A8 <"sub", 0x83, MRM5r     >, OpSize;
def SUBri32b : I2A8 <"sub", 0x83, MRM5r     >;
def SUBmi16b : I2A8 <"sub", 0x83, MRM5m     >, OpSize; // [mem16] -= I8
def SUBmi32b : I2A8 <"sub", 0x83, MRM5m     >;         // [mem32] -= I8

def SBBrr32  : I2A32<"sbb", 0x19, MRMDestReg>;         // R32 -= R32+Borrow
def SBBrm32  : I2A32<"sbb", 0x19, MRMSrcMem >;         // R32 -= [mem32]+Borrow
def SBBmr32  : I2A32<"sbb", 0x1B, MRMDestMem>;         // [mem32] -= R32+Borrow

def IMULrr16 : I2A16<"imul", 0xAF, MRMSrcReg>, TB, OpSize, Pattern<(set R16, (times R16, R16))>;
def IMULrr32 : I2A32<"imul", 0xAF, MRMSrcReg>, TB        , Pattern<(set R32, (times R32, R32))>;
def IMULrm16 : I2A16<"imul", 0xAF, MRMSrcMem>, TB, OpSize;
def IMULrm32 : I2A32<"imul", 0xAF, MRMSrcMem>, TB        ;


// These are suprisingly enough not two address instructions!
def IMULrri16  : X86Inst<"imul", 0x69, MRMSrcReg, Arg16>,     OpSize;  // R16 = R16*I16
def IMULrri32  : X86Inst<"imul", 0x69, MRMSrcReg, Arg32>;              // R32 = R32*I32
def IMULrri16b : X86Inst<"imul", 0x6B, MRMSrcReg, Arg8 >,     OpSize;  // R16 = R16*I8
def IMULrri32b : X86Inst<"imul", 0x6B, MRMSrcReg, Arg8 >;              // R32 = R32*I8
def IMULrmi16  : X86Inst<"imul", 0x69, MRMSrcMem, Arg16>,     OpSize;  // R16 = [mem16]*I16
def IMULrmi32  : X86Inst<"imul", 0x69, MRMSrcMem, Arg32>;              // R32 = [mem32]*I32
def IMULrmi16b : X86Inst<"imul", 0x6B, MRMSrcMem, Arg8 >,     OpSize;  // R16 = [mem16]*I8
def IMULrmi32b : X86Inst<"imul", 0x6B, MRMSrcMem, Arg8 >;              // R32 = [mem32]*I8



// Logical operators...
def ANDrr8   : I2A8 <"and", 0x20, MRMDestReg>,         Pattern<(set R8 , (and R8 , R8 ))>;
def ANDrr16  : I2A16<"and", 0x21, MRMDestReg>, OpSize, Pattern<(set R16, (and R16, R16))>;
def ANDrr32  : I2A32<"and", 0x21, MRMDestReg>,         Pattern<(set R32, (and R32, R32))>;
def ANDmr8   : I2A8 <"and", 0x20, MRMDestMem>;            // [mem8]  &= R8
def ANDmr16  : I2A16<"and", 0x21, MRMDestMem>, OpSize;    // [mem16] &= R16
def ANDmr32  : I2A32<"and", 0x21, MRMDestMem>;            // [mem32] &= R32
def ANDrm8   : I2A8 <"and", 0x22, MRMSrcMem >;            // R8  &= [mem8]
def ANDrm16  : I2A16<"and", 0x23, MRMSrcMem >, OpSize;    // R16 &= [mem16]
def ANDrm32  : I2A32<"and", 0x23, MRMSrcMem >;            // R32 &= [mem32]

def ANDri8   : I2A8 <"and", 0x80, MRM4r     >,         Pattern<(set R8 , (and R8 , imm))>;
def ANDri16  : I2A16<"and", 0x81, MRM4r     >, OpSize, Pattern<(set R16, (and R16, imm))>;
def ANDri32  : I2A32<"and", 0x81, MRM4r     >,         Pattern<(set R32, (and R32, imm))>;
def ANDmi8   : I2A8 <"and", 0x80, MRM4m     >;            // [mem8]  &= imm8
def ANDmi16  : I2A16<"and", 0x81, MRM4m     >, OpSize;    // [mem16] &= imm16
def ANDmi32  : I2A32<"and", 0x81, MRM4m     >;            // [mem32] &= imm32

def ANDri16b : I2A8 <"and", 0x83, MRM4r     >, OpSize;    // R16 &= imm8
def ANDri32b : I2A8 <"and", 0x83, MRM4r     >;            // R32 &= imm8
def ANDmi16b : I2A8 <"and", 0x83, MRM4m     >, OpSize;    // [mem16] &= imm8
def ANDmi32b : I2A8 <"and", 0x83, MRM4m     >;            // [mem32] &= imm8




def ORrr8    : I2A8 <"or" , 0x08, MRMDestReg>,         Pattern<(set R8 , (or  R8 , R8 ))>;
def ORrr16   : I2A16<"or" , 0x09, MRMDestReg>, OpSize, Pattern<(set R16, (or  R16, R16))>;
def ORrr32   : I2A32<"or" , 0x09, MRMDestReg>,         Pattern<(set R32, (or  R32, R32))>;
def ORmr8    : I2A8 <"or" , 0x08, MRMDestMem>;            // [mem8]  |= R8
def ORmr16   : I2A16<"or" , 0x09, MRMDestMem>, OpSize;    // [mem16] |= R16
def ORmr32   : I2A32<"or" , 0x09, MRMDestMem>;            // [mem32] |= R32
def ORrm8    : I2A8 <"or" , 0x0A, MRMSrcMem >;            // R8  |= [mem8]
def ORrm16   : I2A16<"or" , 0x0B, MRMSrcMem >, OpSize;    // R16 |= [mem16]
def ORrm32   : I2A32<"or" , 0x0B, MRMSrcMem >;            // R32 |= [mem32]

def ORri8    : I2A8 <"or" , 0x80, MRM1r     >,         Pattern<(set R8 , (or  R8 , imm))>;
def ORri16   : I2A16<"or" , 0x81, MRM1r     >, OpSize, Pattern<(set R16, (or  R16, imm))>;
def ORri32   : I2A32<"or" , 0x81, MRM1r     >,         Pattern<(set R32, (or  R32, imm))>;
def ORmi8    : I2A8 <"or" , 0x80, MRM1m     >;            // [mem8]  |= imm8
def ORmi16   : I2A16<"or" , 0x81, MRM1m     >, OpSize;    // [mem16] |= imm16
def ORmi32   : I2A32<"or" , 0x81, MRM1m     >;            // [mem32] |= imm32

def ORri16b  : I2A8 <"or" , 0x83, MRM1r     >, OpSize;    // R16 |= imm8
def ORri32b  : I2A8 <"or" , 0x83, MRM1r     >;            // R32 |= imm8
def ORmi16b  : I2A8 <"or" , 0x83, MRM1m     >, OpSize;    // [mem16] |= imm8
def ORmi32b  : I2A8 <"or" , 0x83, MRM1m     >;            // [mem32] |= imm8


def XORrr8   : I2A8 <"xor", 0x30, MRMDestReg>,         Pattern<(set R8 , (xor R8 , R8 ))>;
def XORrr16  : I2A16<"xor", 0x31, MRMDestReg>, OpSize, Pattern<(set R16, (xor R16, R16))>;
def XORrr32  : I2A32<"xor", 0x31, MRMDestReg>,         Pattern<(set R32, (xor R32, R32))>;
def XORmr8   : I2A8 <"xor", 0x30, MRMDestMem>;            // [mem8]  ^= R8
def XORmr16  : I2A16<"xor", 0x31, MRMDestMem>, OpSize;    // [mem16] ^= R16
def XORmr32  : I2A32<"xor", 0x31, MRMDestMem>;            // [mem32] ^= R32
def XORrm8   : I2A8 <"xor", 0x32, MRMSrcMem >;            // R8  ^= [mem8]
def XORrm16  : I2A16<"xor", 0x33, MRMSrcMem >, OpSize;    // R16 ^= [mem16]
def XORrm32  : I2A32<"xor", 0x33, MRMSrcMem >;            // R32 ^= [mem32]

def XORri8   : I2A8 <"xor", 0x80, MRM6r     >,         Pattern<(set R8 , (xor R8 , imm))>;
def XORri16  : I2A16<"xor", 0x81, MRM6r     >, OpSize, Pattern<(set R16, (xor R16, imm))>;
def XORri32  : I2A32<"xor", 0x81, MRM6r     >,         Pattern<(set R32, (xor R32, imm))>;
def XORmi8   : I2A8 <"xor", 0x80, MRM6m     >;            // [mem8] ^= R8
def XORmi16  : I2A16<"xor", 0x81, MRM6m     >, OpSize;    // [mem16] ^= R16
def XORmi32  : I2A32<"xor", 0x81, MRM6m     >;            // [mem32] ^= R32

def XORri16b : I2A8 <"xor", 0x83, MRM6r     >, OpSize;    // R16 ^= imm8
def XORri32b : I2A8 <"xor", 0x83, MRM6r     >;            // R32 ^= imm8
def XORmi16b : I2A8 <"xor", 0x83, MRM6m     >, OpSize;    // [mem16] ^= imm8
def XORmi32b : I2A8 <"xor", 0x83, MRM6m     >;            // [mem32] ^= imm8

// Test instructions are just like AND, except they don't generate a result.
def TESTrr8  : X86Inst<"test", 0x84, MRMDestReg, Arg8 >;          // flags = R8  & R8
def TESTrr16 : X86Inst<"test", 0x85, MRMDestReg, Arg16>, OpSize;  // flags = R16 & R16
def TESTrr32 : X86Inst<"test", 0x85, MRMDestReg, Arg32>;          // flags = R32 & R32
def TESTmr8  : X86Inst<"test", 0x84, MRMDestMem, Arg8 >;          // flags = [mem8]  & R8
def TESTmr16 : X86Inst<"test", 0x85, MRMDestMem, Arg16>, OpSize;  // flags = [mem16] & R16
def TESTmr32 : X86Inst<"test", 0x85, MRMDestMem, Arg32>;          // flags = [mem32] & R32
def TESTrm8  : X86Inst<"test", 0x84, MRMSrcMem , Arg8 >;          // flags = R8  & [mem8]
def TESTrm16 : X86Inst<"test", 0x85, MRMSrcMem , Arg16>, OpSize;  // flags = R16 & [mem16]
def TESTrm32 : X86Inst<"test", 0x85, MRMSrcMem , Arg32>;          // flags = R32 & [mem32]

def TESTri8  : X86Inst<"test", 0xF6, MRM0r     , Arg8 >;          // flags = R8  & imm8
def TESTri16 : X86Inst<"test", 0xF7, MRM0r     , Arg16>, OpSize;  // flags = R16 & imm16
def TESTri32 : X86Inst<"test", 0xF7, MRM0r     , Arg32>;          // flags = R32 & imm32
def TESTmi8  : X86Inst<"test", 0xF6, MRM0m     , Arg8 >;          // flags = [mem8]  & imm8
def TESTmi16 : X86Inst<"test", 0xF7, MRM0m     , Arg16>, OpSize;  // flags = [mem16] & imm16
def TESTmi32 : X86Inst<"test", 0xF7, MRM0m     , Arg32>;          // flags = [mem32] & imm32

// Shift instructions
class UsesCL { list<Register> Uses = [CL]; bit printImplicitUses = 1; }

def SHLrCL8  : I2A8 <"shl", 0xD2, MRM4r     >        , UsesCL; // R8  <<= cl
def SHLrCL16 : I2A16<"shl", 0xD3, MRM4r     >, OpSize, UsesCL; // R16 <<= cl
def SHLrCL32 : I2A32<"shl", 0xD3, MRM4r     >        , UsesCL; // R32 <<= cl
def SHLmCL8  : I2A8 <"shl", 0xD2, MRM4m     >        , UsesCL; // [mem8]  <<= cl
def SHLmCL16 : I2A16<"shl", 0xD3, MRM4m     >, OpSize, UsesCL; // [mem16] <<= cl
def SHLmCL32 : I2A32<"shl", 0xD3, MRM4m     >        , UsesCL; // [mem32] <<= cl

def SHLri8   : I2A8 <"shl", 0xC0, MRM4r     >;                 // R8  <<= imm8
def SHLri16  : I2A8 <"shl", 0xC1, MRM4r     >, OpSize;         // R16 <<= imm8
def SHLri32  : I2A8 <"shl", 0xC1, MRM4r     >;                 // R32 <<= imm8
def SHLmi8   : I2A8 <"shl", 0xC0, MRM4m     >;                 // [mem8]  <<= imm8
def SHLmi16  : I2A8 <"shl", 0xC1, MRM4m     >, OpSize;         // [mem16] <<= imm8
def SHLmi32  : I2A8 <"shl", 0xC1, MRM4m     >;                 // [mem32] <<= imm8

def SHRrCL8  : I2A8 <"shr", 0xD2, MRM5r     >        , UsesCL; // R8  >>= cl
def SHRrCL16 : I2A16<"shr", 0xD3, MRM5r     >, OpSize, UsesCL; // R16 >>= cl
def SHRrCL32 : I2A32<"shr", 0xD3, MRM5r     >        , UsesCL; // R32 >>= cl
def SHRmCL8  : I2A8 <"shr", 0xD2, MRM5m     >        , UsesCL; // [mem8]  >>= cl
def SHRmCL16 : I2A16<"shr", 0xD3, MRM5m     >, OpSize, UsesCL; // [mem16] >>= cl
def SHRmCL32 : I2A32<"shr", 0xD3, MRM5m     >        , UsesCL; // [mem32] >>= cl

def SHRri8   : I2A8 <"shr", 0xC0, MRM5r     >;                 // R8  >>= imm8
def SHRri16  : I2A8 <"shr", 0xC1, MRM5r     >, OpSize;         // R16 >>= imm8
def SHRri32  : I2A8 <"shr", 0xC1, MRM5r     >;                 // R32 >>= imm8
def SHRmi8   : I2A8 <"shr", 0xC0, MRM5m     >;                 // [mem8]  >>= imm8
def SHRmi16  : I2A8 <"shr", 0xC1, MRM5m     >, OpSize;         // [mem16] >>= imm8
def SHRmi32  : I2A8 <"shr", 0xC1, MRM5m     >;                 // [mem32] >>= imm8

def SARrCL8  : I2A8 <"sar", 0xD2, MRM7r     >        , UsesCL; // R8  >>>= cl
def SARrCL16 : I2A16<"sar", 0xD3, MRM7r     >, OpSize, UsesCL; // R16 >>>= cl
def SARrCL32 : I2A32<"sar", 0xD3, MRM7r     >        , UsesCL; // R32 >>>= cl
def SARmCL8  : I2A8 <"sar", 0xD2, MRM7m     >        , UsesCL; // [mem8]  >>>= cl
def SARmCL16 : I2A16<"sar", 0xD3, MRM7m     >, OpSize, UsesCL; // [mem16] >>>= cl
def SARmCL32 : I2A32<"sar", 0xD3, MRM7m     >        , UsesCL; // [mem32] >>>= cl

def SARri8   : I2A8 <"sar", 0xC0, MRM7r     >;                 // R8  >>>= imm8
def SARri16  : I2A8 <"sar", 0xC1, MRM7r     >, OpSize;         // R16 >>>= imm8
def SARri32  : I2A8 <"sar", 0xC1, MRM7r     >;                 // R32 >>>= imm8
def SARmi8   : I2A8 <"sar", 0xC0, MRM7m     >;                 // [mem8]  >>>= imm8
def SARmi16  : I2A8 <"sar", 0xC1, MRM7m     >, OpSize;         // [mem16] >>>= imm8
def SARmi32  : I2A8 <"sar", 0xC1, MRM7m     >;                 // [mem32] >>>= imm8

def SHLDrrCL32 : I2A32<"shld", 0xA5, MRMDestReg>, TB, UsesCL;   // R32 <<= R32,R32 cl
def SHLDmrCL32 : I2A32<"shld", 0xA5, MRMDestMem>, TB, UsesCL;   // [mem32] <<= [mem32],R32 cl
def SHLDrri32  : I2A8 <"shld", 0xA4, MRMDestReg>, TB;           // R32 <<= R32,R32 imm8
def SHLDmri32  : I2A8 <"shld", 0xA4, MRMDestMem>, TB;           // [mem32] <<= [mem32],R32 imm8

def SHRDrrCL32 : I2A32<"shrd", 0xAD, MRMDestReg>, TB, UsesCL;   // R32 >>= R32,R32 cl
def SHRDmrCL32 : I2A32<"shrd", 0xAD, MRMDestMem>, TB, UsesCL;   // [mem32] >>= [mem32],R32 cl
def SHRDrri32  : I2A8 <"shrd", 0xAC, MRMDestReg>, TB;           // R32 >>= R32,R32 imm8
def SHRDmri32  : I2A8 <"shrd", 0xAC, MRMDestMem>, TB;           // [mem32] >>= [mem32],R32 imm8

// Condition code ops, incl. set if equal/not equal/...
def SAHF     : X86Inst<"sahf" , 0x9E, RawFrm, Arg8>, Imp<[AH],[]>;  // flags = AH

def SETBr    : X86Inst<"setb" , 0x92, MRM0r , Arg8>, TB;            // R8 = <  unsign
def SETBm    : X86Inst<"setb" , 0x92, MRM0m , Arg8>, TB;            // [mem8] = <  unsign
def SETAEr   : X86Inst<"setae", 0x93, MRM0r , Arg8>, TB;            // R8 = >= unsign
def SETAEm   : X86Inst<"setae", 0x93, MRM0m , Arg8>, TB;            // [mem8] = >= unsign
def SETEr    : X86Inst<"sete" , 0x94, MRM0r , Arg8>, TB;            // R8 = ==
def SETEm    : X86Inst<"sete" , 0x94, MRM0m , Arg8>, TB;            // [mem8] = ==
def SETNEr   : X86Inst<"setne", 0x95, MRM0r , Arg8>, TB;            // R8 = !=
def SETNEm   : X86Inst<"setne", 0x95, MRM0m , Arg8>, TB;            // [mem8] = !=
def SETBEr   : X86Inst<"setbe", 0x96, MRM0r , Arg8>, TB;            // R8 = <= unsign
def SETBEm   : X86Inst<"setbe", 0x96, MRM0m , Arg8>, TB;            // [mem8] = <= unsign
def SETAr    : X86Inst<"seta" , 0x97, MRM0r , Arg8>, TB;            // R8 = >  signed
def SETAm    : X86Inst<"seta" , 0x97, MRM0m , Arg8>, TB;            // [mem8] = >  signed
def SETSr    : X86Inst<"sets" , 0x98, MRM0r , Arg8>, TB;            // R8 = <sign bit>
def SETSm    : X86Inst<"sets" , 0x98, MRM0m , Arg8>, TB;            // [mem8] = <sign bit>
def SETNSr   : X86Inst<"setns", 0x99, MRM0r , Arg8>, TB;            // R8 = !<sign bit>
def SETNSm   : X86Inst<"setns", 0x99, MRM0m , Arg8>, TB;            // [mem8] = !<sign bit>
def SETLr    : X86Inst<"setl" , 0x9C, MRM0r , Arg8>, TB;            // R8 = <  signed
def SETLm    : X86Inst<"setl" , 0x9C, MRM0m , Arg8>, TB;            // [mem8] = <  signed
def SETGEr   : X86Inst<"setge", 0x9D, MRM0r , Arg8>, TB;            // R8 = >= signed
def SETGEm   : X86Inst<"setge", 0x9D, MRM0m , Arg8>, TB;            // [mem8] = >= signed
def SETLEr   : X86Inst<"setle", 0x9E, MRM0r , Arg8>, TB;            // R8 = <= signed
def SETLEm   : X86Inst<"setle", 0x9E, MRM0m , Arg8>, TB;            // [mem8] = <= signed
def SETGr    : X86Inst<"setg" , 0x9F, MRM0r , Arg8>, TB;            // R8 = <  signed
def SETGm    : X86Inst<"setg" , 0x9F, MRM0m , Arg8>, TB;            // [mem8] = <  signed

// Conditional moves.  These are modelled as X = cmovXX Y, Z.  Eventually
// register allocated to cmovXX XY, Z
def CMOVErr16 : I2A16<"cmove", 0x44, MRMSrcReg>, TB, OpSize;        // if ==, R16 = R16
def CMOVNErr32: I2A32<"cmovne",0x45, MRMSrcReg>, TB;                // if !=, R32 = R32
def CMOVSrr32 : I2A32<"cmovs", 0x48, MRMSrcReg>, TB;                // if signed, R32 = R32

// Integer comparisons
def CMPrr8  : X86Inst<"cmp", 0x38, MRMDestReg, Arg8 >;              // compare R8, R8
def CMPrr16 : X86Inst<"cmp", 0x39, MRMDestReg, Arg16>, OpSize;      // compare R16, R16
def CMPrr32 : X86Inst<"cmp", 0x39, MRMDestReg, Arg32>,              // compare R32, R32
              Pattern<(isVoid (unspec2 R32, R32))>;
def CMPmr8  : X86Inst<"cmp", 0x38, MRMDestMem, Arg8 >;              // compare [mem8], R8
def CMPmr16 : X86Inst<"cmp", 0x39, MRMDestMem, Arg16>, OpSize;      // compare [mem16], R16
def CMPmr32 : X86Inst<"cmp", 0x39, MRMDestMem, Arg32>;              // compare [mem32], R32
def CMPrm8  : X86Inst<"cmp", 0x3A, MRMSrcMem , Arg8 >;              // compare R8, [mem8]
def CMPrm16 : X86Inst<"cmp", 0x3B, MRMSrcMem , Arg16>, OpSize;      // compare R16, [mem16]
def CMPrm32 : X86Inst<"cmp", 0x3B, MRMSrcMem , Arg32>;              // compare R32, [mem32]
def CMPri8  : X86Inst<"cmp", 0x80, MRM7r     , Arg8 >;              // compare R8, imm8
def CMPri16 : X86Inst<"cmp", 0x81, MRM7r     , Arg16>, OpSize;      // compare R16, imm16
def CMPri32 : X86Inst<"cmp", 0x81, MRM7r     , Arg32>;              // compare R32, imm32
def CMPmi8  : X86Inst<"cmp", 0x80, MRM7m     , Arg8 >;              // compare [mem8], imm8
def CMPmi16 : X86Inst<"cmp", 0x81, MRM7m     , Arg16>, OpSize;      // compare [mem16], imm16
def CMPmi32 : X86Inst<"cmp", 0x81, MRM7m     , Arg32>;              // compare [mem32], imm32

// Sign/Zero extenders
def MOVSXr16r8 : X86Inst<"movsx", 0xBE, MRMSrcReg, Arg8>, TB, OpSize; // R16 = signext(R8)
def MOVSXr32r8 : X86Inst<"movsx", 0xBE, MRMSrcReg, Arg8>, TB;         // R32 = signext(R8)
def MOVSXr32r16: X86Inst<"movsx", 0xBF, MRMSrcReg, Arg16>,TB;         // R32 = signext(R16)
def MOVSXr16m8 : X86Inst<"movsx", 0xBE, MRMSrcMem, Arg8>, TB, OpSize; // R16 = signext([mem8])
def MOVSXr32m8 : X86Inst<"movsx", 0xBE, MRMSrcMem, Arg8>, TB;         // R32 = signext([mem8])
def MOVSXr32m16: X86Inst<"movsx", 0xBF, MRMSrcMem, Arg16>,TB;         // R32 = signext([mem16])

def MOVZXr16r8 : X86Inst<"movzx", 0xB6, MRMSrcReg, Arg8>, TB, OpSize; // R16 = zeroext(R8)
def MOVZXr32r8 : X86Inst<"movzx", 0xB6, MRMSrcReg, Arg8>, TB;         // R32 = zeroext(R8)
def MOVZXr32r16: X86Inst<"movzx", 0xB7, MRMSrcReg, Arg16>,TB;         // R32 = zeroext(R16)
def MOVZXr16m8 : X86Inst<"movzx", 0xB6, MRMSrcMem, Arg8>, TB, OpSize; // R16 = zeroext([mem8])
def MOVZXr32m8 : X86Inst<"movzx", 0xB6, MRMSrcMem, Arg8>, TB;         // R32 = zeroext([mem8])
def MOVZXr32m16: X86Inst<"movzx", 0xB7, MRMSrcMem, Arg16>,TB;         // R32 = zeroext([mem16])


//===----------------------------------------------------------------------===//
// Floating point support
//===----------------------------------------------------------------------===//

// FIXME: These need to indicate mod/ref sets for FP regs... & FP 'TOP'

// Floating point pseudo instructions...
class FPInst<string n, bits<8> o, Format F, ArgType t, FPFormat fp>
  : X86Inst<n, o, F, t> { let FPForm = fp; let FPFormBits = FPForm.Value; }

// Pseudo instructions for floating point.  We use these pseudo instructions
// because they can be expanded by the fp spackifier into one of many different
// forms of instructions for doing these operations.  Until the stackifier runs,
// we prefer to be abstract.
def FpMOV : FPInst<"FMOV", 0, Pseudo, ArgF80, SpecialFP>;   // f1 = fmov f2
def FpADD : FPInst<"FADD", 0, Pseudo, ArgF80, TwoArgFP>;    // f1 = fadd f2, f3
def FpSUB : FPInst<"FSUB", 0, Pseudo, ArgF80, TwoArgFP>;    // f1 = fsub f2, f3
def FpMUL : FPInst<"FMUL", 0, Pseudo, ArgF80, TwoArgFP>;    // f1 = fmul f2, f3
def FpDIV : FPInst<"FDIV", 0, Pseudo, ArgF80, TwoArgFP>;    // f1 = fdiv f2, f3

def FpUCOM : FPInst<"FUCOM", 0, Pseudo, ArgF80, TwoArgFP>;  // FPSW = fucom f1, f2

def FpGETRESULT : FPInst<"FGETRESULT",0, Pseudo, ArgF80, SpecialFP>;  // FPR = ST(0)

def FpSETRESULT : FPInst<"FSETRESULT",0, Pseudo, ArgF80, SpecialFP>;  // ST(0) = FPR

// Floating point loads & stores...
def FLDrr   : FPInst<"fld"   , 0xC0, AddRegFrm, ArgF80, NotFP>, D9;   // push(ST(i))
def FLDr32  : FPInst<"fld"   , 0xD9, MRM0m    , ArgF32, ZeroArgFP>;        // load float
def FLDr64  : FPInst<"fld"   , 0xDD, MRM0m    , ArgF64, ZeroArgFP>;        // load double
def FLDr80  : FPInst<"fld"   , 0xDB, MRM5m    , ArgF80, ZeroArgFP>;        // load extended
def FILDr16 : FPInst<"fild"  , 0xDF, MRM0m    , Arg16 , ZeroArgFP>;        // load signed short
def FILDr32 : FPInst<"fild"  , 0xDB, MRM0m    , Arg32 , ZeroArgFP>;        // load signed int
def FILDr64 : FPInst<"fild"  , 0xDF, MRM5m    , Arg64 , ZeroArgFP>;        // load signed long

def FSTr32   : FPInst<"fst" , 0xD9, MRM2m    , ArgF32, OneArgFP>;          // store float
def FSTr64   : FPInst<"fst" , 0xDD, MRM2m    , ArgF64, OneArgFP>;          // store double
def FSTPr32  : FPInst<"fstp", 0xD9, MRM3m    , ArgF32, OneArgFP>;          // store float, pop
def FSTPr64  : FPInst<"fstp", 0xDD, MRM3m    , ArgF64, OneArgFP>;          // store double, pop
def FSTPr80  : FPInst<"fstp", 0xDB, MRM7m    , ArgF80, OneArgFP>;          // store extended, pop
def FSTrr    : FPInst<"fst" , 0xD0, AddRegFrm, ArgF80, NotFP   >, DD;      // ST(i) = ST(0)
def FSTPrr   : FPInst<"fstp", 0xD8, AddRegFrm, ArgF80, NotFP   >, DD;      // ST(i) = ST(0), pop

def FISTr16  : FPInst<"fist",    0xDF, MRM2m , Arg16 , OneArgFP>;          // store signed short
def FISTr32  : FPInst<"fist",    0xDB, MRM2m , Arg32 , OneArgFP>;          // store signed int
def FISTPr16 : FPInst<"fistp",   0xDF, MRM3m , Arg16 , NotFP   >;          // store signed short, pop
def FISTPr32 : FPInst<"fistp",   0xDB, MRM3m , Arg32 , NotFP   >;          // store signed int, pop
def FISTPr64 : FPInst<"fistpll", 0xDF, MRM7m , Arg64 , OneArgFP>;          // store signed long, pop

def FXCH     : FPInst<"fxch",    0xC8, AddRegFrm, ArgF80, NotFP>, D9;      // fxch ST(i), ST(0)

// Floating point constant loads...
def FLD0 : FPInst<"fldz", 0xEE, RawFrm, ArgF80, ZeroArgFP>, D9;
def FLD1 : FPInst<"fld1", 0xE8, RawFrm, ArgF80, ZeroArgFP>, D9;


// Unary operations...
def FCHS : FPInst<"fchs", 0xE0, RawFrm, ArgF80, OneArgFPRW>, D9;           // f1 = fchs f2

def FTST : FPInst<"ftst", 0xE4, RawFrm, ArgF80, OneArgFP>, D9;             // ftst ST(0)

// Binary arithmetic operations...
class FPST0rInst<string n, bits<8> o>
  : X86Inst<n, o, AddRegFrm, ArgF80>, D8 {
  list<Register> Uses = [ST0];
  list<Register> Defs = [ST0];
}
class FPrST0Inst<string n, bits<8> o>
  : X86Inst<n, o, AddRegFrm, ArgF80>, DC {
  bit printImplicitUses = 1;
  list<Register> Uses = [ST0];
}
class FPrST0PInst<string n, bits<8> o>
  : X86Inst<n, o, AddRegFrm, ArgF80>, DE {
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
def FUCOMr    : X86Inst<"fucom"  , 0xE0, AddRegFrm, ArgF80>, DD, Imp<[ST0],[]>;  // FPSW = compare ST(0) with ST(i)
def FUCOMPr   : X86Inst<"fucomp" , 0xE8, AddRegFrm, ArgF80>, DD, Imp<[ST0],[]>;  // FPSW = compare ST(0) with ST(i), pop
def FUCOMPPr  : X86Inst<"fucompp", 0xE9, RawFrm   , ArgF80>, DA, Imp<[ST0],[]>;  // compare ST(0) with ST(1), pop, pop

// Floating point flag ops
def FNSTSWr8  : X86Inst<"fnstsw" , 0xE0, RawFrm   , ArgF80>, DF, Imp<[],[AX]>;   // AX = fp flags
def FNSTCWm16 : X86Inst<"fnstcw" , 0xD9, MRM7m    , Arg16 >;                     // [mem16] = X87 control world
def FLDCWm16  : X86Inst<"fldcw"  , 0xD9, MRM5m    , Arg16 >;                     // X87 control world = [mem16]


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
