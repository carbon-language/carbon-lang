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

// *mem - Operand definitions for the funky X86 addressing mode operands.
//
def i8mem : Operand<i8> {
  let NumMIOperands = 4;
  let PrintMethod = "printMemoryOperand";
}

def i16mem : Operand<i16> {
  let NumMIOperands = 4;
  let PrintMethod = "printMemoryOperand";
}

def i32mem : Operand<i32> {
  let NumMIOperands = 4;
  let PrintMethod = "printMemoryOperand";
}


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
def Mem16  : MemType<2>;
def Mem32  : MemType<3>;
def Mem64  : MemType<4>;
def Mem80  : MemType<5>;

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
def CompareFP  : FPFormat<5>;
def CondMovFP  : FPFormat<6>;
def SpecialFP  : FPFormat<7>;


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

  //
  // Attributes specific to X86 instructions...
  //
  bit hasOpSizePrefix = 0; // Does this inst have a 0x66 prefix?

  // Flag whether implicit register usage is printed after the instruction.
  bit printImplicitUsesAfter  = 0;

  bits<4> Prefix = 0;       // Which prefix byte does this inst have?
  FPFormat FPForm;          // What flavor of FP instruction is this?
  bits<3> FPFormBits = 0;
}

class Imp<list<Register> uses, list<Register> defs> {
  list<Register> Uses = uses;
  list<Register> Defs = defs;
}

// II - InstructionInfo - this will eventually replace the I class.
class II<dag ops, string AsmStr> {
  dag OperandList = ops;
  string AsmString = AsmStr;
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

class I<bits<8> o, Format f, dag ops, string asm> : X86Inst<"", o, f, NoMem, NoImm>, II<ops, asm>;

class Ii<bits<8> o, Format f, ImmType i> : X86Inst<"", o, f, NoMem, i>;
class Ii8 <bits<8> o, Format f, dag ops, string asm> : Ii<o, f, Imm8 >, II<ops, asm>;
class Ii16<bits<8> o, Format f, dag ops, string asm> : Ii<o, f, Imm16>, II<ops, asm>;
class Ii32<bits<8> o, Format f, dag ops, string asm> : Ii<o, f, Imm32>, II<ops, asm>;

//===----------------------------------------------------------------------===//
// Instruction list...
//

def PHI : I<0, Pseudo, (ops), "PHINODE">;        // PHI node.
def NOOP : I<0x90, RawFrm, (ops), "nop">; // nop

def ADJCALLSTACKDOWN : I<0, Pseudo, (ops), "#ADJCALLSTACKDOWN">;
def ADJCALLSTACKUP   : I<0, Pseudo, (ops), "#ADJCALLSTACKUP">;
def IMPLICIT_USE     : I<0, Pseudo, (ops), "#IMPLICIT_USE">;
def IMPLICIT_DEF     : I<0, Pseudo, (ops), "#IMPLICIT_DEF">;
let isTerminator = 1 in
  let Defs = [FP0, FP1, FP2, FP3, FP4, FP5, FP6] in
    def FP_REG_KILL  : I<0, Pseudo, (ops), "#FP_REG_KILL">;

//===----------------------------------------------------------------------===//
//  Control Flow Instructions...
//

// Return instruction...
let isTerminator = 1, isReturn = 1, isBarrier = 1 in
  def RET : I<0xC3, RawFrm, (ops), "ret">;

// All branches are RawFrm, Void, Branch, and Terminators
let isBranch = 1, isTerminator = 1 in
  class IBr<bits<8> opcode, dag ops, string asm> : I<opcode, RawFrm, ops, asm>;

let isBarrier = 1 in
  def JMP : IBr<0xE9, (ops i32imm:$dst), "jmp $dst">;
def JB  : IBr<0x82, (ops i32imm:$dst), "jb $dst">, TB;
def JAE : IBr<0x83, (ops i32imm:$dst), "jae $dst">, TB;
def JE  : IBr<0x84, (ops i32imm:$dst), "je $dst">, TB;
def JNE : IBr<0x85, (ops i32imm:$dst), "jne $dst">, TB;
def JBE : IBr<0x86, (ops i32imm:$dst), "jbe $dst">, TB;
def JA  : IBr<0x87, (ops i32imm:$dst), "ja $dst">, TB;
def JS  : IBr<0x88, (ops i32imm:$dst), "js $dst">, TB;
def JNS : IBr<0x89, (ops i32imm:$dst), "jns $dst">, TB;
def JL  : IBr<0x8C, (ops i32imm:$dst), "jl $dst">, TB;
def JGE : IBr<0x8D, (ops i32imm:$dst), "jge $dst">, TB;
def JLE : IBr<0x8E, (ops i32imm:$dst), "jle $dst">, TB;
def JG  : IBr<0x8F, (ops i32imm:$dst), "jg $dst">, TB;


//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1 in
  // All calls clobber the non-callee saved registers...
  let Defs = [EAX, ECX, EDX, FP0, FP1, FP2, FP3, FP4, FP5, FP6] in {
    def CALLpcrel32 : X86Inst<"call", 0xE8, RawFrm, NoMem, NoImm>;  // FIXME: 'call' doesn't allow 'OFFSET'
    def CALL32r     : I<0xFF, MRM2r, (ops R32:$dst), "call $dst">;
    def CALL32m     : I<0xFF, MRM2m, (ops i32mem:$dst), "call $dst">;
  }

       
//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//
def LEAVE    : I<0xC9, RawFrm,
                 (ops), "leave">, Imp<[EBP,ESP],[EBP,ESP]>;
def POP32r   : I<0x58, AddRegFrm,
                 (ops R32:$reg), "pop $reg">, Imp<[ESP],[ESP]>;

let isTwoAddress = 1 in                                    // R32 = bswap R32
  def BSWAP32r : I<0xC8, AddRegFrm,
                   (ops R32:$dst, R32:$src), "bswap $dst">, TB;

def XCHG8rr  : I<0x86, MRMDestReg,                    // xchg R8, R8
                 (ops R8:$src1, R8:$src2), "xchg $src1, $src2">;
def XCHG16rr : I<0x87, MRMDestReg,                    // xchg R16, R16
                 (ops R16:$src1, R16:$src2), "xchg $src1, $src2">, OpSize;
def XCHG32rr : I<0x87, MRMDestReg,                    // xchg R32, R32
                 (ops R32:$src1, R32:$src2), "xchg $src1, $src2">;

def XCHG8mr  : I<0x86, MRMDestMem, (ops i8mem:$src1, R8:$src2), "xchg $src1, $src2">;
def XCHG16mr : I<0x87, MRMDestMem, (ops i16mem:$src1, R16:$src2), "xchg $src1, $src2">, OpSize;
def XCHG32mr : I<0x87, MRMDestMem, (ops i32mem:$src1, R32:$src2), "xchg $src1, $src2">;
def XCHG8rm  : I<0x86, MRMSrcMem , (ops R8:$src1, i8mem:$src2), "xchg $src1, $src2">;
def XCHG16rm : I<0x87, MRMSrcMem , (ops R16:$src1, i16mem:$src2), "xchg $src1, $src2">, OpSize;
def XCHG32rm : I<0x87, MRMSrcMem , (ops R32:$src1, i32mem:$src2), "xchg $src1, $src2">;

def LEA16r   : I<0x8D, MRMSrcMem, (ops R16:$dst, i32mem:$src), "lea $dst, $src">, OpSize;
def LEA32r   : I<0x8D, MRMSrcMem, (ops R32:$dst, i32mem:$src), "lea $dst, $src">;


def REP_MOVSB : I<0xA4, RawFrm, (ops), "rep movsb">,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>, REP;
def REP_MOVSW : I<0xA5, RawFrm, (ops), "rep movsw">,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>, REP, OpSize;
def REP_MOVSD : I<0xA5, RawFrm, (ops), "rep movsd">,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>, REP;

def REP_STOSB : I<0xAA, RawFrm, (ops), "rep stosb">,
                Imp<[AL,ECX,EDI], [ECX,EDI]>, REP;
def REP_STOSW : I<0xAB, RawFrm, (ops), "rep stosw">,
                Imp<[AX,ECX,EDI], [ECX,EDI]>, REP, OpSize;
def REP_STOSD : I<0xAB, RawFrm, (ops), "rep stosd">,
                Imp<[EAX,ECX,EDI], [ECX,EDI]>, REP;


//===----------------------------------------------------------------------===//
//  Input/Output Instructions...
//
def IN8rr  : I<0xEC, RawFrm, (ops),
               "in %AL, %DX">,  Imp<[DX], [AL]>;
def IN16rr : I<0xED, RawFrm, (ops),
               "in %AX, %DX">,  Imp<[DX], [AX]>, OpSize;
def IN32rr : I<0xED, RawFrm, (ops),
               "in %EAX, %DX">, Imp<[DX],[EAX]>;

def IN8ri  : Ii16<0xE4, RawFrm, (ops i16imm:$port),
                  "in %AL, $port">,  Imp<[], [AL]>;
def IN16ri : Ii16<0xE5, RawFrm, (ops i16imm:$port),
                  "in %AX, $port">,  Imp<[], [AX]>, OpSize;
def IN32ri : Ii16<0xE5, RawFrm, (ops i16imm:$port),
                  "in %EAX, $port">, Imp<[],[EAX]>;

def OUT8rr  : I<0xEE, RawFrm, (ops),
                "out %DX, %AL">,  Imp<[DX,  AL], []>;
def OUT16rr : I<0xEF, RawFrm, (ops),
                "out %DX, %AX">,  Imp<[DX,  AX], []>, OpSize;
def OUT32rr : I<0xEF, RawFrm, (ops),
                "out %DX, %EAX">, Imp<[DX, EAX], []>;

def OUT8ir  : Ii16<0xE6, RawFrm, (ops i16imm:$port),
                   "out $port, %AL">, Imp<[AL], []>;
def OUT16ir : Ii16<0xE7, RawFrm, (ops i16imm:$port),
                   "out $port, %AX">, Imp<[AX], []>, OpSize;
def OUT32ir : Ii16<0xE7, RawFrm, (ops i16imm:$port),
                   "out $port, %EAX">, Imp<[EAX], []>;

//===----------------------------------------------------------------------===//
//  Move Instructions...
//
def MOV8rr  : I<0x88, MRMDestReg, (ops R8 :$dst, R8 :$src), "mov $dst, $src">;
def MOV16rr : I<0x89, MRMDestReg, (ops R16:$dst, R16:$src), "mov $dst, $src">, OpSize;
def MOV32rr : I<0x89, MRMDestReg, (ops R32:$dst, R32:$src), "mov $dst, $src">;
def MOV8ri  : Ii8 <0xB0, AddRegFrm, (ops R8 :$dst, i8imm :$src), "mov $dst, $src">;
def MOV16ri : Ii16<0xB8, AddRegFrm, (ops R16:$dst, i16imm:$src), "mov $dst, $src">, OpSize;
def MOV32ri : Ii32<0xB8, AddRegFrm, (ops R32:$dst, i32imm:$src), "mov $dst, $src">;
def MOV8mi  : Ii8 <0xC6, MRM0m, (ops i8mem :$dst, i8imm :$src), "mov $dst, $src">;
def MOV16mi : Ii16<0xC7, MRM0m, (ops i16mem:$dst, i16imm:$src), "mov $dst, $src">, OpSize;
def MOV32mi : Ii32<0xC7, MRM0m, (ops i32mem:$dst, i32imm:$src), "mov $dst, $src">;

def MOV8rm  : I<0x8A, MRMSrcMem, (ops R8 :$dst, i8mem :$src), "mov $dst, $src">;
def MOV16rm : I<0x8B, MRMSrcMem, (ops R16:$dst, i16mem:$src), "mov $dst, $src">, OpSize;
def MOV32rm : I<0x8B, MRMSrcMem, (ops R32:$dst, i32mem:$src), "mov $dst, $src">;

def MOV8mr  : I<0x88, MRMDestMem, (ops i8mem :$dst, R8 :$src), "mov $dst, $src">;
def MOV16mr : I<0x89, MRMDestMem, (ops i16mem:$dst, R16:$src), "mov $dst, $src">, OpSize;
def MOV32mr : I<0x89, MRMDestMem, (ops i32mem:$dst, R32:$src), "mov $dst, $src">;

//===----------------------------------------------------------------------===//
//  Fixed-Register Multiplication and Division Instructions...
//

// Extra precision multiplication
def MUL8r  : I<0xF6, MRM4r, (ops R8:$src), "mul $src">,
             Imp<[AL],[AX]>;               // AL,AH = AL*R8
def MUL16r : I<0xF7, MRM4r, (ops R16:$src), "mul $src">,
             Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*R16
def MUL32r : I<0xF7, MRM4r, (ops R32:$src), "mul $src">,
             Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*R32
def MUL8m  : I<0xF6, MRM4m, (ops i8mem :$src),
               "mul $src">, Imp<[AL],[AX]>;               // AL,AH = AL*[mem8]
def MUL16m : I<0xF7, MRM4m, (ops i16mem:$src),
               "mul $src">, Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*[mem16]
def MUL32m : I<0xF7, MRM4m, (ops i32mem:$src),
               "mul $src">, Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*[mem32]

// unsigned division/remainder
def DIV8r  : I<0xF6, MRM6r, (ops R8:$src), "div $src">,
             Imp<[AX],[AX]>;                                         // AX/r8 = AL,AH
def DIV16r : I<0xF7, MRM6r, (ops R16:$src), "div $src">,
             Imp<[AX,DX],[AX,DX]>, OpSize;                           // DX:AX/r16 = AX,DX
def DIV32r : I<0xF7, MRM6r, (ops R32:$src), "div $src">,
             Imp<[EAX,EDX],[EAX,EDX]>;                               // EDX:EAX/r32 = EAX,EDX
def DIV8m  : I<0xF6, MRM6m, (ops i8mem:$src), "div $src">, Imp<[AX],[AX]>;               // AX/[mem8] = AL,AH
def DIV16m : I<0xF7, MRM6m, (ops i16mem:$src), "div $src">, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/[mem16] = AX,DX
def DIV32m : I<0xF7, MRM6m, (ops i32mem:$src), "div $src">, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/[mem32] = EAX,EDX

// Signed division/remainder.
def IDIV8r : I<0xF6, MRM7r, (ops R8:$src), "idiv $src">,
             Imp<[AX],[AX]>;               // AX/r8 = AL,AH
def IDIV16r: I<0xF7, MRM7r, (ops R16:$src), "idiv $src">,
             Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/r16 = AX,DX
def IDIV32r: I<0xF7, MRM7r, (ops R32:$src), "idiv $src">,
             Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/r32 = EAX,EDX
def IDIV8m : I<0xF6, MRM7m, (ops i8mem:$src), "idiv $src">, Imp<[AX],[AX]>;               // AX/[mem8] = AL,AH
def IDIV16m: I<0xF7, MRM7m, (ops i16mem:$src), "idiv $src">, Imp<[AX,DX],[AX,DX]>, OpSize; // DX:AX/[mem16] = AX,DX
def IDIV32m: I<0xF7, MRM7m, (ops i32mem:$src), "idiv $src">, Imp<[EAX,EDX],[EAX,EDX]>;     // EDX:EAX/[mem32] = EAX,EDX

// Sign-extenders for division.
def CBW : I<0x98, RawFrm, (ops), "cbw">, Imp<[AL],[AH]>;   // AX = signext(AL)
def CWD : I<0x99, RawFrm, (ops), "cwd">, Imp<[AX],[DX]>;   // DX:AX = signext(AX)
def CDQ : I<0x99, RawFrm, (ops), "cdq">, Imp<[EAX],[EDX]>; // EDX:EAX = signext(EAX)
          

//===----------------------------------------------------------------------===//
//  Two address Instructions...
//
let isTwoAddress = 1 in {

// Conditional moves
def CMOVB16rr : I<0x42, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovb $dst, $src2">, TB, OpSize;                // if <u, R16 = R16
def CMOVB16rm : I<0x42, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovb $dst, $src2">, TB, OpSize;                // if <u, R16 = [mem16]
def CMOVB32rr : I<0x42, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovb $dst, $src2">, TB;                        // if <u, R32 = R32
def CMOVB32rm : I<0x42, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovb $dst, $src2">, TB;                        // if <u, R32 = [mem32]

def CMOVAE16rr: I<0x43, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovae $dst, $src2">, TB, OpSize;               // if >=u, R16 = R16
def CMOVAE16rm: I<0x43, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovae $dst, $src2">, TB, OpSize;               // if >=u, R16 = [mem16]
def CMOVAE32rr: I<0x43, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovae $dst, $src2">, TB;                       // if >=u, R32 = R32
def CMOVAE32rm: I<0x43, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovae $dst, $src2">, TB;                       // if >=u, R32 = [mem32]

def CMOVE16rr : I<0x44, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmove $dst, $src2">, TB, OpSize;                // if ==, R16 = R16
def CMOVE16rm : I<0x44, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmove $dst, $src2">, TB, OpSize;                // if ==, R16 = [mem16]
def CMOVE32rr : I<0x44, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmove $dst, $src2">, TB;                        // if ==, R32 = R32
def CMOVE32rm : I<0x44, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmove $dst, $src2">, TB;                        // if ==, R32 = [mem32]

def CMOVNE16rr: I<0x45, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovne $dst, $src2">, TB, OpSize;               // if !=, R16 = R16
def CMOVNE16rm: I<0x45, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovne $dst, $src2">, TB, OpSize;        // if !=, R16 = [mem16]
def CMOVNE32rr: I<0x45, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovne $dst, $src2">, TB;                       // if !=, R32 = R32
def CMOVNE32rm: I<0x45, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovne $dst, $src2">, TB;                // if !=, R32 = [mem32]

def CMOVBE16rr: I<0x46, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovbe $dst, $src2">, TB, OpSize;                    // if <=u, R16 = R16
def CMOVBE16rm: I<0x46, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovbe $dst, $src2">, TB, OpSize;        // if <=u, R16 = [mem16]
def CMOVBE32rr: I<0x46, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovbe $dst, $src2">, TB;                       // if <=u, R32 = R32
def CMOVBE32rm: I<0x46, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovbe $dst, $src2">, TB;                // if <=u, R32 = [mem32]

def CMOVA16rr : I<0x47, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmova $dst, $src2">, TB, OpSize;                // if >u, R16 = R16
def CMOVA16rm : I<0x47, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmova $dst, $src2">, TB, OpSize;                // if >u, R16 = [mem16]
def CMOVA32rr : I<0x47, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmova $dst, $src2">, TB;                        // if >u, R32 = R32
def CMOVA32rm : I<0x47, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmova $dst, $src2">, TB;                        // if >u, R32 = [mem32]

def CMOVS16rr : I<0x48, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovs $dst, $src2">, TB, OpSize;                // if signed, R16 = R16
def CMOVS16rm : I<0x48, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovs $dst, $src2">, TB, OpSize;                // if signed, R16 = [mem16]
def CMOVS32rr : I<0x48, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovs $dst, $src2">, TB;                        // if signed, R32 = R32
def CMOVS32rm : I<0x48, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovs $dst, $src2">, TB;                        // if signed, R32 = [mem32]

def CMOVNS16rr: I<0x49, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovns $dst, $src2">, TB, OpSize;               // if !signed, R16 = R16
def CMOVNS16rm: I<0x49, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovns $dst, $src2">, TB, OpSize;                // if !signed, R16 = [mem16]
def CMOVNS32rr: I<0x49, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovns $dst, $src2">, TB;                       // if !signed, R32 = R32
def CMOVNS32rm: I<0x49, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovns $dst, $src2">, TB;                        // if !signed, R32 = [mem32]

def CMOVL16rr : I<0x4C, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovl $dst, $src2">, TB, OpSize;                // if <s, R16 = R16
def CMOVL16rm : I<0x4C, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovl $dst, $src2">, TB, OpSize;                // if <s, R16 = [mem16]
def CMOVL32rr : I<0x4C, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovl $dst, $src2">, TB;                        // if <s, R32 = R32
def CMOVL32rm : I<0x4C, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovl $dst, $src2">, TB;                        // if <s, R32 = [mem32]

def CMOVGE16rr: I<0x4D, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovge $dst, $src2">, TB, OpSize;               // if >=s, R16 = R16
def CMOVGE16rm: I<0x4D, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovge $dst, $src2">, TB, OpSize;               // if >=s, R16 = [mem16]
def CMOVGE32rr: I<0x4D, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovge $dst, $src2">, TB;                       // if >=s, R32 = R32
def CMOVGE32rm: I<0x4D, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovge $dst, $src2">, TB;                       // if >=s, R32 = [mem32]

def CMOVLE16rr: I<0x4E, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovle $dst, $src2">, TB, OpSize;               // if <=s, R16 = R16
def CMOVLE16rm: I<0x4E, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovle $dst, $src2">, TB, OpSize;               // if <=s, R16 = [mem16]
def CMOVLE32rr: I<0x4E, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovle $dst, $src2">, TB;                       // if <=s, R32 = R32
def CMOVLE32rm: I<0x4E, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovle $dst, $src2">, TB;                       // if <=s, R32 = [mem32]

def CMOVG16rr : I<0x4F, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovg $dst, $src2">, TB, OpSize;                // if >s, R16 = R16
def CMOVG16rm : I<0x4F, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovg $dst, $src2">, TB, OpSize;                // if >s, R16 = [mem16]
def CMOVG32rr : I<0x4F, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovg $dst, $src2">, TB;                        // if >s, R32 = R32
def CMOVG32rm : I<0x4F, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovg $dst, $src2">, TB;                        // if >s, R32 = [mem32]

// unary instructions
def NEG8r  : I<0xF6, MRM3r, (ops R8 :$dst, R8 :$src), "neg $dst">;
def NEG16r : I<0xF7, MRM3r, (ops R16:$dst, R16:$src), "neg $dst">, OpSize;
def NEG32r : I<0xF7, MRM3r, (ops R32:$dst, R32:$src), "neg $dst">;
let isTwoAddress = 0 in {
  def NEG8m  : I<0xF6, MRM3m, (ops i8mem :$dst), "neg $dst">;
  def NEG16m : I<0xF7, MRM3m, (ops i16mem:$dst), "neg $dst">, OpSize;
  def NEG32m : I<0xF7, MRM3m, (ops i32mem:$dst), "neg $dst">;
}

def NOT8r  : I<0xF6, MRM2r, (ops R8 :$dst, R8 :$src), "not $dst">;
def NOT16r : I<0xF7, MRM2r, (ops R16:$dst, R16:$src), "not $dst">, OpSize;
def NOT32r : I<0xF7, MRM2r, (ops R32:$dst, R32:$src), "not $dst">;
let isTwoAddress = 0 in {
  def NOT8m  : I<0xF6, MRM2m, (ops i8mem :$dst), "not $dst">;
  def NOT16m : I<0xF7, MRM2m, (ops i16mem:$dst), "not $dst">, OpSize;
  def NOT32m : I<0xF7, MRM2m, (ops i32mem:$dst), "not $dst">;
}

def INC8r  : I<0xFE, MRM0r, (ops R8 :$dst, R8 :$src), "inc $dst">;
def INC16r : I<0xFF, MRM0r, (ops R16:$dst, R16:$src), "inc $dst">, OpSize;
def INC32r : I<0xFF, MRM0r, (ops R32:$dst, R32:$src), "inc $dst">;
let isTwoAddress = 0 in {
  def INC8m  : I<0xFE, MRM0m, (ops i8mem :$dst), "inc $dst">;
  def INC16m : I<0xFF, MRM0m, (ops i16mem:$dst), "inc $dst">, OpSize;
  def INC32m : I<0xFF, MRM0m, (ops i32mem:$dst), "inc $dst">;
}

def DEC8r  : I<0xFE, MRM1r, (ops R8 :$dst, R8 :$src), "dec $dst">;
def DEC16r : I<0xFF, MRM1r, (ops R16:$dst, R16:$src), "dec $dst">, OpSize;
def DEC32r : I<0xFF, MRM1r, (ops R32:$dst, R32:$src), "dec $dst">;

let isTwoAddress = 0 in {
  def DEC8m  : I<0xFE, MRM1m, (ops i8mem :$dst), "dec $dst">;
  def DEC16m : I<0xFF, MRM1m, (ops i16mem:$dst), "dec $dst">, OpSize;
  def DEC32m : I<0xFF, MRM1m, (ops i32mem:$dst), "dec $dst">;
}

// Logical operators...
def AND8rr   : I<0x20, MRMDestReg, (ops R8 :$dst, R8 :$src1, R8 :$src2), "and $dst, $src2">;
def AND16rr  : I<0x21, MRMDestReg, (ops R16:$dst, R16:$src1, R16:$src2), "and $dst, $src2">, OpSize;
def AND32rr  : I<0x21, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2), "and $dst, $src2">;

def AND8rm   : I<0x22, MRMSrcMem , (ops R8 :$dst, R8 :$src1, i8mem :$src2), "and $dst, $src2">;
def AND16rm  : I<0x23, MRMSrcMem , (ops R16:$dst, R16:$src1, i16mem:$src2), "and $dst, $src2">, OpSize;
def AND32rm  : I<0x23, MRMSrcMem , (ops R32:$dst, R32:$src1, i32mem:$src2), "and $dst, $src2">;

def AND8ri   : Ii8 <0x80, MRM4r, (ops R8 :$dst, R8 :$src1, i8imm :$src2),
                    "and $dst, $src2">;
def AND16ri  : Ii16<0x81, MRM4r, (ops R16:$dst, R16:$src1, i16imm:$src2),
                    "and $dst, $src2">, OpSize;
def AND32ri  : Ii32<0x81, MRM4r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "and $dst, $src2">;
def AND16ri8 : Ii8<0x83, MRM4r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "and $dst, $src2" >, OpSize;
def AND32ri8 : Ii8<0x83, MRM4r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "and $dst, $src2">;

let isTwoAddress = 0 in {
  def AND8mr   : I<0x20, MRMDestMem, (ops i8mem :$dst, R8 :$src), "and $dst, $src">;
  def AND16mr  : I<0x21, MRMDestMem, (ops i16mem:$dst, R16:$src), "and $dst, $src">, OpSize;
  def AND32mr  : I<0x21, MRMDestMem, (ops i32mem:$dst, R32:$src), "and $dst, $src">;
  def AND8mi   : Ii8 <0x80, MRM4m, (ops i8mem :$dst, i8imm :$src), "and $dst, $src">;
  def AND16mi  : Ii16<0x81, MRM4m, (ops i16mem:$dst, i16imm:$src), "and $dst, $src">, OpSize;
  def AND32mi  : Ii32<0x81, MRM4m, (ops i32mem:$dst, i32imm:$src), "and $dst, $src">;
  def AND16mi8 : Ii8 <0x83, MRM4m, (ops i16mem:$dst, i8imm :$src), "and $dst, $src">, OpSize;
  def AND32mi8 : Ii8 <0x83, MRM4m, (ops i32mem:$dst, i8imm :$src), "and $dst, $src">;
}


def OR8rr    : I<0x08, MRMDestReg, (ops R8 :$dst, R8 :$src1, R8 :$src2),
                "or $dst, $src2">;
def OR16rr   : I<0x09, MRMDestReg, (ops R16:$dst, R16:$src1, R16:$src2),
                 "or $dst, $src2">, OpSize;
def OR32rr   : I<0x09, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                 "or $dst, $src2">;
def OR8rm    : I<0x0A, MRMSrcMem , (ops R8 :$dst, R8 :$src1, i8mem :$src2),
                 "or $dst, $src2">;
def OR16rm   : I<0x0B, MRMSrcMem , (ops R16:$dst, R16:$src1, i16mem:$src2),
                 "or $dst, $src2">, OpSize;
def OR32rm   : I<0x0B, MRMSrcMem , (ops R32:$dst, R32:$src1, i32mem:$src2),
                 "or $dst, $src2">;

def OR8ri    : Ii8 <0x80, MRM1r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                    "or $dst, $src2">;
def OR16ri   : Ii16<0x81, MRM1r, (ops R16:$dst, R16:$src1, i16imm:$src2),
                    "or $dst, $src2">, OpSize;
def OR32ri   : Ii32<0x81, MRM1r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "or $dst, $src2">;

def OR16ri8  : Ii8<0x83, MRM1r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                   "or $dst, $src2">, OpSize;
def OR32ri8  : Ii8<0x83, MRM1r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "or $dst, $src2">;
let isTwoAddress = 0 in {
  def OR8mr  : I<0x08, MRMDestMem, (ops i8mem:$dst, R8:$src),
                 "or $dst, $src">;
  def OR16mr : I<0x09, MRMDestMem, (ops i16mem:$dst, R16:$src),
                 "or $dst, $src">, OpSize;
  def OR32mr : I<0x09, MRMDestMem, (ops i32mem:$dst, R32:$src),
                 "or $dst, $src">;
  def OR8mi    : Ii8<0x80, MRM1m, (ops i8mem :$dst, i8imm:$src),
                 "or $dst, $src">;
  def OR16mi   : Ii16<0x81, MRM1m, (ops i16mem:$dst, i16imm:$src),
                 "or $dst, $src">, OpSize;
  def OR32mi   : Ii32<0x81, MRM1m, (ops i32mem:$dst, i32imm:$src),
                 "or $dst, $src">;
  def OR16mi8  : Ii8<0x83, MRM1m, (ops i16mem:$dst, i8imm:$src),
                 "or $dst, $src">, OpSize;
  def OR32mi8  : Ii8<0x83, MRM1m, (ops i32mem:$dst, i8imm:$src),
                 "or $dst, $src">;
}


def XOR8rr   : I<0x30, MRMDestReg, (ops R8 :$dst, R8 :$src1, R8 :$src2), "xor $dst, $src2">;
def XOR16rr  : I<0x31, MRMDestReg, (ops R16:$dst, R16:$src1, R16:$src2), "xor $dst, $src2">, OpSize;
def XOR32rr  : I<0x31, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2), "xor $dst, $src2">;
def XOR8rm   : I<0x32, MRMSrcMem , (ops R8 :$dst, R8:$src1, i8mem :$src2), "xor $dst, $src2">;
def XOR16rm  : I<0x33, MRMSrcMem , (ops R16:$dst, R8:$src1, i16mem:$src2), "xor $dst, $src2">, OpSize;
def XOR32rm  : I<0x33, MRMSrcMem , (ops R32:$dst, R8:$src1, i32mem:$src2), "xor $dst, $src2">;

def XOR8ri   : Ii8   <0x80, MRM6r, (ops R8:$dst, R8:$src1, i8imm:$src2), "xor $dst, $src2">;
def XOR16ri  : Ii16  <0x81, MRM6r, (ops R16:$dst, R16:$src1, i16imm:$src2), "xor $dst, $src2">, OpSize;
def XOR32ri  : Ii32  <0x81, MRM6r, (ops R32:$dst, R32:$src1, i32imm:$src2), "xor $dst, $src2">;
def XOR16ri8 : Ii8<0x83, MRM6r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "xor $dst, $src2">, OpSize;
def XOR32ri8 : Ii8<0x83, MRM6r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "xor $dst, $src2">;
let isTwoAddress = 0 in {
  def XOR8mr   : I<0x30, MRMDestMem, (ops i8mem :$dst, R8 :$src), "xor $dst, $src">;
  def XOR16mr  : I<0x31, MRMDestMem, (ops i16mem:$dst, R16:$src), "xor $dst, $src">, OpSize;
  def XOR32mr  : I<0x31, MRMDestMem, (ops i32mem:$dst, R32:$src), "xor $dst, $src">;
  def XOR8mi   : Ii8 <0x80, MRM6m, (ops i8mem :$dst, i8imm :$src), "xor $dst, $src">;
  def XOR16mi  : Ii16<0x81, MRM6m, (ops i16mem:$dst, i16imm:$src), "xor $dst, $src">, OpSize;
  def XOR32mi  : Ii32<0x81, MRM6m, (ops i32mem:$dst, i32imm:$src), "xor $dst, $src">;
  def XOR16mi8 : Ii8 <0x83, MRM6m, (ops i16mem:$dst, i8imm :$src), "xor $dst, $src">, OpSize;
  def XOR32mi8 : Ii8 <0x83, MRM6m, (ops i32mem:$dst, i8imm :$src), "xor $dst, $src">;
}

// Shift instructions
// FIXME: provide shorter instructions when imm8 == 1
def SHL8rCL  : I<0xD2, MRM4r, (ops R8 :$dst, R8 :$src), "shl $dst, %CL">, Imp<[CL],[]>;
def SHL16rCL : I<0xD3, MRM4r, (ops R16:$dst, R16:$src), "shl $dst, %CL">, Imp<[CL],[]>, OpSize;
def SHL32rCL : I<0xD3, MRM4r, (ops R32:$dst, R32:$src), "shl $dst, %CL">, Imp<[CL],[]>;
def SHL8ri   : Ii8<0xC0, MRM4r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                   "shl $dst, $src2">;
def SHL16ri  : Ii8<0xC1, MRM4r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "shl $dst, $src2">, OpSize;
def SHL32ri  : Ii8<0xC1, MRM4r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "shl $dst, $src2">;

let isTwoAddress = 0 in {
  def SHL8mCL  : I<0xD2, MRM4m, (ops i8mem :$dst), "shl $dst, %CL">, Imp<[CL],[]>;
  def SHL16mCL : I<0xD3, MRM4m, (ops i16mem:$dst), "shl $dst, %CL">, Imp<[CL],[]>, OpSize;
  def SHL32mCL : I<0xD3, MRM4m, (ops i32mem:$dst), "shl $dst, %CL">, Imp<[CL],[]>;
  def SHL8mi   : Ii8<0xC0, MRM4m, (ops i8mem :$dst, i8imm:$src), "shl $dst, $src">;
  def SHL16mi  : Ii8<0xC1, MRM4m, (ops i16mem:$dst, i8imm:$src), "shl $dst, $src">, OpSize;
  def SHL32mi  : Ii8<0xC1, MRM4m, (ops i32mem:$dst, i8imm:$src), "shl $dst, $src">;
}

def SHR8rCL  : I<0xD2, MRM5r, (ops R8 :$dst, R8 :$src), "shr $dst, %CL">, Imp<[CL],[]>;
def SHR16rCL : I<0xD3, MRM5r, (ops R16:$dst, R16:$src), "shr $dst, %CL">, Imp<[CL],[]>, OpSize;
def SHR32rCL : I<0xD3, MRM5r, (ops R32:$dst, R32:$src), "shr $dst, %CL">, Imp<[CL],[]>;

def SHR8ri   : Ii8   <0xC0, MRM5r, (ops R8:$dst, R8:$src1, i8imm:$src2), "shr $dst, $src2">;
def SHR16ri  : Ii8   <0xC1, MRM5r, (ops R16:$dst, R16:$src1, i8imm:$src2), "shr $dst, $src2">, OpSize;
def SHR32ri  : Ii8   <0xC1, MRM5r, (ops R32:$dst, R32:$src1, i8imm:$src2), "shr $dst, $src2">;

let isTwoAddress = 0 in {
  def SHR8mCL  : I<0xD2, MRM5m, (ops i8mem :$dst), "shr $dst, %CL">, Imp<[CL],[]>;
  def SHR16mCL : I<0xD3, MRM5m, (ops i16mem:$dst), "shr $dst, %CL">, Imp<[CL],[]>, OpSize;
  def SHR32mCL : I<0xD3, MRM5m, (ops i32mem:$dst), "shr $dst, %CL">, Imp<[CL],[]>;
  def SHR8mi   : Ii8<0xC0, MRM5m, (ops i8mem :$dst, i8imm:$src), "shr $dst, $src">;
  def SHR16mi  : Ii8<0xC1, MRM5m, (ops i16mem:$dst, i8imm:$src), "shr $dst, $src">, OpSize;
  def SHR32mi  : Ii8<0xC1, MRM5m, (ops i32mem:$dst, i8imm:$src), "shr $dst, $src">;
}

def SAR8rCL  : I<0xD2, MRM7r, (ops R8 :$dst, R8 :$src), "sar $dst, %CL">, Imp<[CL],[]>;
def SAR16rCL : I<0xD3, MRM7r, (ops R16:$dst, R16:$src), "sar $dst, %CL">, Imp<[CL],[]>, OpSize;
def SAR32rCL : I<0xD3, MRM7r, (ops R32:$dst, R32:$src), "sar $dst, %CL">, Imp<[CL],[]>;

def SAR8ri   : Ii8<0xC0, MRM7r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                   "sar $dst, $src2">;
def SAR16ri  : Ii8<0xC1, MRM7r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "sar $dst, $src2">, OpSize;
def SAR32ri  : Ii8<0xC1, MRM7r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "sar $dst, $src2">;
let isTwoAddress = 0 in {
  def SAR8mCL  : I<0xD2, MRM7m, (ops i8mem :$dst), "sar $dst, %CL">, Imp<[CL],[]>;
  def SAR16mCL : I<0xD3, MRM7m, (ops i16mem:$dst), "sar $dst, %CL">, Imp<[CL],[]>, OpSize;
  def SAR32mCL : I<0xD3, MRM7m, (ops i32mem:$dst), "sar $dst, %CL">, Imp<[CL],[]>;
  def SAR8mi   : Ii8<0xC0, MRM7m, (ops i8mem :$dst, i8imm:$src), "sar $dst, $src">;
  def SAR16mi  : Ii8<0xC1, MRM7m, (ops i16mem:$dst, i8imm:$src), "sar $dst, $src">, OpSize;
  def SAR32mi  : Ii8<0xC1, MRM7m, (ops i32mem:$dst, i8imm:$src), "sar $dst, $src">;
}

def SHLD32rrCL : I<0xA5, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "shld $dst, $src2, %CL">, Imp<[CL],[]>, TB;
def SHRD32rrCL : I<0xAD, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                   "shrd $dst, $src2, %CL">, Imp<[CL],[]>, TB;
def SHLD32rri8 : Ii8<0xA4, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2, i8imm:$src3),
                     "shld $dst, $src2, $src3">, TB;
def SHRD32rri8 : Ii8<0xAC, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2, i8imm:$src3),
                     "shrd $dst, $src2, $src3">, TB;

let isTwoAddress = 0 in {
  def SHLD32mrCL : I<0xA5, MRMDestMem, (ops i32mem:$dst, R32:$src2),
                  "shld $dst, $src2, %CL">, Imp<[CL],[]>, TB;
  def SHRD32mrCL : I<0xAD, MRMDestMem, (ops i32mem:$dst, R32:$src2),
                  "shrd $dst, $src2, %CL">, Imp<[CL],[]>, TB;
  def SHLD32mri8 : Ii8<0xA4, MRMDestMem, (ops i32mem:$dst, R32:$src2, i8imm:$src3),
                  "shld $dst, $src2, $src3">, TB;
  def SHRD32mri8 : Ii8<0xAC, MRMDestMem, (ops i32mem:$dst, R32:$src2, i8imm:$src3),
                  "shrd $dst, $src2, $src3">, TB;
}


// Arithmetic...
def ADD8rr   : I<0x00, MRMDestReg, (ops R8 :$dst, R8 :$src1, R8 :$src2), "add $dst, $src2">;
def ADD16rr  : I<0x01, MRMDestReg, (ops R16:$dst, R16:$src1, R16:$src2), "add $dst, $src2">, OpSize;
def ADD32rr  : I<0x01, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2), "add $dst, $src2">;
def ADD8rm   : I<0x02, MRMSrcMem, (ops R8 :$dst, R8 :$src1, i8mem :$src2), "add $dst, $src2">;
def ADD16rm  : I<0x03, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2), "add $dst, $src2">, OpSize;
def ADD32rm  : I<0x03, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2), "add $dst, $src2">;

def ADD8ri   : Ii8   <0x80, MRM0r, (ops R8:$dst, R8:$src1, i8imm:$src2), "add $dst, $src2">;
def ADD16ri  : Ii16  <0x81, MRM0r, (ops R16:$dst, R16:$src1, i16imm:$src2), "add $dst, $src2">, OpSize;
def ADD32ri  : Ii32  <0x81, MRM0r, (ops R32:$dst, R32:$src1, i32imm:$src2), "add $dst, $src2">;

def ADD16ri8 : Ii8   <0x83, MRM0r, (ops R16:$dst, R16:$src1, i8imm:$src2), "add $dst, $src2">, OpSize;
def ADD32ri8 : Ii8   <0x83, MRM0r, (ops R32:$dst, R32:$src1, i8imm:$src2), "add $dst, $src2">;

let isTwoAddress = 0 in {
  def ADD8mr   : I<0x00, MRMDestMem, (ops i8mem :$dst, R8 :$src2), "add $dst, $src2">;
  def ADD16mr  : I<0x01, MRMDestMem, (ops i16mem:$dst, R16:$src2), "add $dst, $src2">, OpSize;
  def ADD32mr  : I<0x01, MRMDestMem, (ops i32mem:$dst, R32:$src2), "add $dst, $src2">;
  def ADD8mi   : Ii8 <0x80, MRM0m, (ops i8mem :$dst, i8imm :$src2), "add $dst, $src2">;
  def ADD16mi  : Ii16<0x81, MRM0m, (ops i16mem:$dst, i16imm:$src2), "add $dst, $src2">, OpSize;
  def ADD32mi  : Ii32<0x81, MRM0m, (ops i32mem:$dst, i32imm:$src2), "add $dst, $src2">;
  def ADD16mi8 : Ii8 <0x83, MRM0m, (ops i16mem:$dst, i8imm :$src2), "add $dst, $src2">, OpSize;
  def ADD32mi8 : Ii8 <0x83, MRM0m, (ops i32mem:$dst, i8imm :$src2), "add $dst, $src2">;
}

def ADC32rr  : I<0x11, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2), "adc $dst, $src2">;
def ADC32rm  : I<0x13, MRMSrcMem , (ops R32:$dst, R32:$src1, i32mem:$src2), "adc $dst, $src2">;
def ADC32ri  : Ii32<0x81, MRM2r, (ops R32:$dst, R32:$src1, i32imm:$src2), "adc $dst, $src2">;
def ADC32ri8 : Ii8 <0x83, MRM2r, (ops R32:$dst, R32:$src1, i8imm:$src2), "adc $dst, $src2">;

let isTwoAddress = 0 in {
  def ADC32mr  : I<0x11, MRMDestMem, (ops i32mem:$dst, R32:$src2), "adc $dst, $src2">;
  def ADC32mi  : Ii32<0x81, MRM2m, (ops i32mem:$dst, i32imm:$src2), "adc $dst, $src2">;
  def ADC32mi8 : Ii8 <0x83, MRM2m, (ops i32mem:$dst, i8imm :$src2), "adc $dst, $src2">;
}

def SUB8rr   : I<0x28, MRMDestReg, (ops R8 :$dst, R8 :$src1, R8 :$src2), "sub $dst, $src2">;
def SUB16rr  : I<0x29, MRMDestReg, (ops R16:$dst, R16:$src1, R16:$src2), "sub $dst, $src2">, OpSize;
def SUB32rr  : I<0x29, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2), "sub $dst, $src2">;
def SUB8rm   : I<0x2A, MRMSrcMem, (ops R8 :$dst, R8 :$src1, i8mem :$src2), "sub $dst, $src2">;
def SUB16rm  : I<0x2B, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2), "sub $dst, $src2">, OpSize;
def SUB32rm  : I<0x2B, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2), "sub $dst, $src2">;

def SUB8ri   : Ii8 <0x80, MRM5r, (ops R8:$dst, R8:$src1, i8imm:$src2),
                    "sub $dst, $src2">;
def SUB16ri  : Ii16<0x81, MRM5r, (ops R16:$dst, R16:$src1, i16imm:$src2),
                    "sub $dst, $src2">, OpSize;
def SUB32ri  : Ii32<0x81, MRM5r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "sub $dst, $src2">;
def SUB16ri8 : Ii8<0x83, MRM5r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "sub $dst, $src2">, OpSize;
def SUB32ri8 : Ii8<0x83, MRM5r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "sub $dst, $src2">;
let isTwoAddress = 0 in {
  def SUB8mr   : I<0x28, MRMDestMem, (ops i8mem :$dst, R8 :$src2), "sub $dst, $src2">;
  def SUB16mr  : I<0x29, MRMDestMem, (ops i16mem:$dst, R16:$src2), "sub $dst, $src2">, OpSize;
  def SUB32mr  : I<0x29, MRMDestMem, (ops i32mem:$dst, R32:$src2), "sub $dst, $src2">;
  def SUB8mi   : Ii8 <0x80, MRM5m, (ops i8mem :$dst, i8imm:$src2), "sub $dst, $src2">;
  def SUB16mi  : Ii16<0x81, MRM5m, (ops i16mem:$dst, i16imm:$src2), "sub $dst, $src2">, OpSize;
  def SUB32mi  : Ii32<0x81, MRM5m, (ops i32mem:$dst, i32imm:$src2), "sub $dst, $src2">;
  def SUB16mi8 : Ii8 <0x83, MRM5m, (ops i16mem:$dst, i8imm :$src2), "sub $dst, $src2">, OpSize;
  def SUB32mi8 : Ii8 <0x83, MRM5m, (ops i32mem:$dst, i8imm :$src2), "sub $dst, $src2">;
}

def SBB32rr  : I<0x19, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                 "adc $dst, $src2">;
let isTwoAddress = 0 in {
  def SBB32mr  : I<0x19, MRMDestMem, (ops i32mem:$dst, R32:$src2), "sbb $dst, $src2">;
  def SBB32mi  : Ii32<0x81, MRM3m, (ops i32mem:$dst, i32imm:$src2), "sbb $dst, $src2">;
  def SBB32mi8 : Ii8 <0x83, MRM3m, (ops i32mem:$dst, i8imm :$src2), "sbb $dst, $src2">;
}
def SBB32rm  : I<0x1B, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                    "sbb $dst, $src2">;
def SBB32ri  : Ii32<0x81, MRM3r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "sbb $dst, $src2">;
def SBB32ri8 : Ii8<0x83, MRM3r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "sbb $dst, $src2">;

def IMUL16rr : I<0xAF, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2), "imul $dst, $src2">, TB, OpSize;
def IMUL32rr : I<0xAF, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2), "imul $dst, $src2">, TB;
def IMUL16rm : I<0xAF, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2), "imul $dst, $src2">, TB, OpSize;
def IMUL32rm : I<0xAF, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2), "imul $dst, $src2">, TB        ;

} // end Two Address instructions

// Suprisingly enough, these are not two address instructions!
def IMUL16rri  : Ii16<0x69, MRMSrcReg, (ops R16:$dst, R16:$src1, i16imm:$src2),
                      "imul $dst, $src1, $src2">,     OpSize; // R16 = R16*I16
def IMUL32rri  : Ii32<0x69, MRMSrcReg, (ops R32:$dst, R32:$src1, i32imm:$src2),
                      "imul $dst, $src1, $src2">;             // R32 = R32*I32
def IMUL16rri8 : Ii8<0x6B, MRMSrcReg, (ops R16:$dst, R16:$src1, i8imm:$src2),
                     "imul $dst, $src1, $src2">,     OpSize;  // R16 = R16*I8
def IMUL32rri8 : Ii8<0x6B, MRMSrcReg, (ops R32:$dst, R32:$src1, i8imm:$src2),
                     "imul $dst, $src1, $src2">;           // R32 = R32*I8

def IMUL16rmi  : Ii16<0x69, MRMSrcMem, (ops R32:$dst, i16mem:$src1, i16imm:$src2),
                     "imul $dst, $src1, $src2">,     OpSize;  // R16 = [mem16]*I16
def IMUL32rmi  : Ii32<0x69, MRMSrcMem, (ops R32:$dst, i32mem:$src1, i32imm:$src2),
                     "imul $dst, $src1, $src2">;              // R32 = [mem32]*I32
def IMUL16rmi8 : Ii8<0x6B, MRMSrcMem, (ops R32:$dst, i16mem:$src1, i8imm :$src2),
                     "imul $dst, $src1, $src2">,     OpSize;  // R16 = [mem16]*I8
def IMUL32rmi8 : Ii8<0x6B, MRMSrcMem, (ops R32:$dst, i32mem:$src1, i8imm: $src2),
                     "imul $dst, $src1, $src2">;              // R32 = [mem32]*I8

//===----------------------------------------------------------------------===//
// Test instructions are just like AND, except they don't generate a result.
def TEST8rr  : I<0x84, MRMDestReg, (ops R8:$src1, R8:$src2),
                 "test $src1, $src2">;
def TEST16rr : I<0x85, MRMDestReg, (ops R16:$src1, R16:$src2),
                 "test $src1, $src2">, OpSize;
def TEST32rr : I<0x85, MRMDestReg, (ops R32:$src1, R32:$src2),
                 "test $src1, $src2">;
def TEST8mr  : I<0x84, MRMDestMem, (ops i8mem :$src1, R8 :$src2),
                 "test $src1, $src2">;
def TEST16mr : I<0x85, MRMDestMem, (ops i16mem:$src1, R16:$src2),
                 "test $src1, $src2">, OpSize;
def TEST32mr : I<0x85, MRMDestMem, (ops i32mem:$src1, R32:$src2),
                 "test $src1, $src2">;
def TEST8rm  : I<0x84, MRMSrcMem, (ops R8 :$src1, i8mem :$src2),
                 "test $src1, $src2">;
def TEST16rm : I<0x85, MRMSrcMem, (ops R16:$src1, i16mem:$src2),
                 "test $src1, $src2">, OpSize;
def TEST32rm : I<0x85, MRMSrcMem, (ops R32:$src1, i32mem:$src2),
                 "test $src1, $src2">;

def TEST8ri  : Ii8 <0xF6, MRM0r, (ops R8:$dst, i8imm:$src),
                    "test $dst, $src">;          // flags = R8  & imm8
def TEST16ri : Ii16<0xF7, MRM0r, (ops R16:$dst, i16imm:$src),
                    "test $dst, $src">, OpSize;  // flags = R16 & imm16
def TEST32ri : Ii32<0xF7, MRM0r, (ops R32:$dst, i32imm:$src),
                    "test $dst, $src">;          // flags = R32 & imm32
def TEST8mi  : Ii8 <0xF6, MRM0m, (ops i32mem:$dst, i8imm:$src),
                    "test $dst, $src">;          // flags = [mem8]  & imm8
def TEST16mi : Ii16<0xF7, MRM0m, (ops i16mem:$dst, i16imm:$src),
                    "test $dst, $src">, OpSize;  // flags = [mem16] & imm16
def TEST32mi : Ii32<0xF7, MRM0m, (ops i32mem:$dst, i32imm:$src),
                    "test $dst, $src">;          // flags = [mem32] & imm32



// Condition code ops, incl. set if equal/not equal/...
def SAHF     : I<0x9E, RawFrm, (ops), "sahf">, Imp<[AH],[]>;  // flags = AH
def LAHF     : I<0x9F, RawFrm, (ops), "lahf">, Imp<[],[AH]>;  // AH = flags

def SETBr    : I<0x92, MRM0r, (ops R8   :$dst), "setb $dst">, TB;    // R8 = <  unsign
def SETBm    : I<0x92, MRM0m, (ops i8mem:$dst), "setb $dst">, TB;            // [mem8] = <  unsign
def SETAEr   : I<0x93, MRM0r, (ops R8   :$dst), "setae $dst">, TB;   // R8 = >= unsign
def SETAEm   : I<0x93, MRM0m, (ops i8mem:$dst), "setae $dst">, TB;            // [mem8] = >= unsign
def SETEr    : I<0x94, MRM0r, (ops R8   :$dst), "sete $dst">, TB;    // R8 = ==
def SETEm    : I<0x94, MRM0m, (ops i8mem:$dst), "sete $dst">, TB;            // [mem8] = ==
def SETNEr   : I<0x95, MRM0r, (ops R8   :$dst), "setne $dst">, TB;   // R8 = !=
def SETNEm   : I<0x95, MRM0m, (ops i8mem:$dst), "setne $dst">, TB;            // [mem8] = !=
def SETBEr   : I<0x96, MRM0r, (ops R8   :$dst), "setbe $dst">, TB;   // R8 = <= unsign
def SETBEm   : I<0x96, MRM0m, (ops i8mem:$dst), "setbe $dst">, TB;            // [mem8] = <= unsign
def SETAr    : I<0x97, MRM0r, (ops R8   :$dst), "seta $dst">, TB;    // R8 = >  signed
def SETAm    : I<0x97, MRM0m, (ops i8mem:$dst), "seta $dst">, TB;            // [mem8] = >  signed
def SETSr    : I<0x98, MRM0r, (ops R8   :$dst), "sets $dst">, TB;    // R8 = <sign bit>
def SETSm    : I<0x98, MRM0m, (ops i8mem:$dst), "sets $dst">, TB;            // [mem8] = <sign bit>
def SETNSr   : I<0x99, MRM0r, (ops R8   :$dst), "setns $dst">, TB;   // R8 = !<sign bit>
def SETNSm   : I<0x99, MRM0m, (ops i8mem:$dst), "setns $dst">, TB;            // [mem8] = !<sign bit>
def SETPr    : I<0x9A, MRM0r, (ops R8   :$dst), "setp $dst">, TB;    // R8 = parity
def SETPm    : I<0x9A, MRM0m, (ops i8mem:$dst), "setp $dst">, TB;            // [mem8] = parity
def SETLr    : I<0x9C, MRM0r, (ops R8   :$dst), "setl $dst">, TB;    // R8 = <  signed
def SETLm    : I<0x9C, MRM0m, (ops i8mem:$dst), "setl $dst">, TB;            // [mem8] = <  signed
def SETGEr   : I<0x9D, MRM0r, (ops R8   :$dst), "setge $dst">, TB;   // R8 = >= signed
def SETGEm   : I<0x9D, MRM0m, (ops i8mem:$dst), "setge $dst">, TB;            // [mem8] = >= signed
def SETLEr   : I<0x9E, MRM0r, (ops R8   :$dst), "setle $dst">, TB;   // R8 = <= signed
def SETLEm   : I<0x9E, MRM0m, (ops i8mem:$dst), "setle $dst">, TB;            // [mem8] = <= signed
def SETGr    : I<0x9F, MRM0r, (ops R8   :$dst), "setg $dst">, TB;    // R8 = <  signed
def SETGm    : I<0x9F, MRM0m, (ops i8mem:$dst), "setg $dst">, TB;            // [mem8] = <  signed

// Integer comparisons
def CMP8rr  : I<0x38, MRMDestReg, (ops R8 :$src1, R8 :$src2), "cmp $src1, $src2">;
def CMP16rr : I<0x39, MRMDestReg, (ops R16:$src1, R16:$src2), "cmp $src1, $src2">, OpSize;
def CMP32rr : I<0x39, MRMDestReg, (ops R32:$src1, R32:$src2), "cmp $src1, $src2">;
def CMP8mr  : I<0x38, MRMDestMem, (ops i8mem :$src1, R8 :$src2), "cmp $src1, $src2">;
def CMP16mr : I<0x39, MRMDestMem, (ops i16mem:$src1, R16:$src2), "cmp $src1, $src2">, OpSize;
def CMP32mr : I<0x39, MRMDestMem, (ops i32mem:$src1, R32:$src2), "cmp $src1, $src2">;
def CMP8rm  : I<0x3A, MRMSrcMem , (ops R8 :$src1, i8mem :$src2), "cmp $src1, $src2">;
def CMP16rm : I<0x3B, MRMSrcMem , (ops R16:$src1, i16mem:$src2), "cmp $src1, $src2">, OpSize;
def CMP32rm : I<0x3B, MRMSrcMem , (ops R32:$src1, i32mem:$src2), "cmp $src1, $src2">;
def CMP8ri  : Ii8 <0x80, MRM7r, (ops R16:$dst, i8imm:$src), "cmp $dst, $src">;
def CMP16ri : Ii16<0x81, MRM7r, (ops R16:$dst, i16imm:$src), "cmp $dst, $src">, OpSize;
def CMP32ri : Ii32<0x81, MRM7r, (ops R32:$dst, i32imm:$src), "cmp $dst, $src">;
def CMP8mi  : Ii8 <0x80, MRM7m, (ops i8mem :$dst, i8imm :$src), "cmp $dst, $src">;
def CMP16mi : Ii16<0x81, MRM7m, (ops i16mem:$dst, i16imm:$src), "cmp $dst, $src">, OpSize;
def CMP32mi : Ii32<0x81, MRM7m, (ops i32mem:$dst, i32imm:$src), "cmp $dst, $src">;

// Sign/Zero extenders
def MOVSX16rr8 : I<0xBE, MRMSrcReg, (ops R16:$dst, R8 :$src), "movsx $dst, $src">, TB, OpSize;
def MOVSX32rr8 : I<0xBE, MRMSrcReg, (ops R32:$dst, R8 :$src), "movsx $dst, $src">, TB;
def MOVSX32rr16: I<0xBF, MRMSrcReg, (ops R32:$dst, R16:$src), "movsx $dst, $src">, TB;
def MOVSX16rm8 : I<0xBE, MRMSrcMem, (ops R16:$dst, i8mem :$src), "movsx $dst, $src">, TB, OpSize;
def MOVSX32rm8 : I<0xBE, MRMSrcMem, (ops R32:$dst, i8mem :$src), "movsx $dst, $src">, TB;
def MOVSX32rm16: I<0xBF, MRMSrcMem, (ops R32:$dst, i16mem:$src), "movsx $dst, $src">, TB;

def MOVZX16rr8 : I<0xB6, MRMSrcReg, (ops R16:$dst, R8 :$src), "movzx $dst, $src">, TB, OpSize;
def MOVZX32rr8 : I<0xB6, MRMSrcReg, (ops R32:$dst, R8 :$src), "movzx $dst, $src">, TB;
def MOVZX32rr16: I<0xB7, MRMSrcReg, (ops R32:$dst, R16:$src), "movzx $dst, $src">, TB;
def MOVZX16rm8 : I<0xB6, MRMSrcMem, (ops R16:$dst, i8mem :$src), "movzx $dst, $src">, TB, OpSize;
def MOVZX32rm8 : I<0xB6, MRMSrcMem, (ops R32:$dst, i8mem :$src), "movzx $dst, $src">, TB;
def MOVZX32rm16: I<0xB7, MRMSrcMem, (ops R32:$dst, i16mem:$src), "movzx $dst, $src">, TB;


//===----------------------------------------------------------------------===//
// Floating point support
//===----------------------------------------------------------------------===//

// FIXME: These need to indicate mod/ref sets for FP regs... & FP 'TOP'

// Floating point instruction templates
class FPInst<string n, bits<8> o, Format F, FPFormat fp, MemType m, ImmType i>
  : X86Inst<n, o, F, m, i> { let FPForm = fp; let FPFormBits = FPForm.Value; }

class FPI<bits<8> o, Format F, FPFormat fp, dag ops, string asm>
  : FPInst<"", o, F, fp, NoMem, NoImm>, II<ops, asm>;

class FPIM<string n, bits<8> o, Format F, FPFormat fp, MemType m> : FPInst<n, o, F, fp, m, NoImm>;

class FPI16m<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem16>;
class FPI32m<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem32>;
class FPI64m<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem64>;
class FPI80m<string n, bits<8> o, Format F, FPFormat fp> : FPIM<n, o, F, fp, Mem80>;

// Pseudo instructions for floating point.  We use these pseudo instructions
// because they can be expanded by the fp spackifier into one of many different
// forms of instructions for doing these operations.  Until the stackifier runs,
// we prefer to be abstract.
def FpMOV : FPI<0, Pseudo, SpecialFP, (ops RFP, RFP), "">;   // f1 = fmov f2
def FpADD : FPI<0, Pseudo, TwoArgFP , (ops RFP, RFP, RFP), "">;    // f1 = fadd f2, f3
def FpSUB : FPI<0, Pseudo, TwoArgFP , (ops RFP, RFP, RFP), "">;    // f1 = fsub f2, f3
def FpMUL : FPI<0, Pseudo, TwoArgFP , (ops RFP, RFP, RFP), "">;    // f1 = fmul f2, f3
def FpDIV : FPI<0, Pseudo, TwoArgFP , (ops RFP, RFP, RFP), "">;    // f1 = fdiv f2, f3

def FpGETRESULT : FPI<0, Pseudo, SpecialFP, (ops RFP), "">;  // FPR = ST(0)
def FpSETRESULT : FPI<0, Pseudo, SpecialFP, (ops RFP), "">;  // ST(0) = FPR

// FADD reg, mem: Before stackification, these are represented by: R1 = FADD* R2, [mem]
def FADD32m  : FPI32m<"fadd",  0xD8, MRM0m, OneArgFPRW>;    // ST(0) = ST(0) + [mem32real]
def FADD64m  : FPI64m<"fadd",  0xDC, MRM0m, OneArgFPRW>;    // ST(0) = ST(0) + [mem64real]
def FIADD16m : FPI16m<"fiadd", 0xDE, MRM0m, OneArgFPRW>;    // ST(0) = ST(0) + [mem16int]
def FIADD32m : FPI32m<"fiadd", 0xDA, MRM0m, OneArgFPRW>;    // ST(0) = ST(0) + [mem32int]

// FMUL reg, mem: Before stackification, these are represented by: R1 = FMUL* R2, [mem]
def FMUL32m  : FPI32m<"fmul",  0xD8, MRM1m, OneArgFPRW>;    // ST(0) = ST(0) * [mem32real]
def FMUL64m  : FPI64m<"fmul",  0xDC, MRM1m, OneArgFPRW>;    // ST(0) = ST(0) * [mem64real]
def FIMUL16m : FPI16m<"fimul", 0xDE, MRM1m, OneArgFPRW>;    // ST(0) = ST(0) * [mem16int]
def FIMUL32m : FPI32m<"fimul", 0xDA, MRM1m, OneArgFPRW>;    // ST(0) = ST(0) * [mem32int]

// FSUB reg, mem: Before stackification, these are represented by: R1 = FSUB* R2, [mem]
def FSUB32m  : FPI32m<"fsub",  0xD8, MRM4m, OneArgFPRW>;    // ST(0) = ST(0) - [mem32real]
def FSUB64m  : FPI64m<"fsub",  0xDC, MRM4m, OneArgFPRW>;    // ST(0) = ST(0) - [mem64real]
def FISUB16m : FPI16m<"fisub", 0xDE, MRM4m, OneArgFPRW>;    // ST(0) = ST(0) - [mem16int]
def FISUB32m : FPI32m<"fisub", 0xDA, MRM4m, OneArgFPRW>;    // ST(0) = ST(0) - [mem32int]

// FSUBR reg, mem: Before stackification, these are represented by: R1 = FSUBR* R2, [mem]
// Note that the order of operands does not reflect the operation being performed.
def FSUBR32m  : FPI32m<"fsubr",  0xD8, MRM5m, OneArgFPRW>;  // ST(0) = [mem32real] - ST(0)
def FSUBR64m  : FPI64m<"fsubr",  0xDC, MRM5m, OneArgFPRW>;  // ST(0) = [mem64real] - ST(0)
def FISUBR16m : FPI16m<"fisubr", 0xDE, MRM5m, OneArgFPRW>;  // ST(0) = [mem16int] - ST(0)
def FISUBR32m : FPI32m<"fisubr", 0xDA, MRM5m, OneArgFPRW>;  // ST(0) = [mem32int] - ST(0)

// FDIV reg, mem: Before stackification, these are represented by: R1 = FDIV* R2, [mem]
def FDIV32m  : FPI32m<"fdiv",  0xD8, MRM6m, OneArgFPRW>;    // ST(0) = ST(0) / [mem32real]
def FDIV64m  : FPI64m<"fdiv",  0xDC, MRM6m, OneArgFPRW>;    // ST(0) = ST(0) / [mem64real]
def FIDIV16m : FPI16m<"fidiv", 0xDE, MRM6m, OneArgFPRW>;    // ST(0) = ST(0) / [mem16int]
def FIDIV32m : FPI32m<"fidiv", 0xDA, MRM6m, OneArgFPRW>;    // ST(0) = ST(0) / [mem32int]

// FDIVR reg, mem: Before stackification, these are represented by: R1 = FDIVR* R2, [mem]
// Note that the order of operands does not reflect the operation being performed.
def FDIVR32m  : FPI32m<"fdivr",  0xD8, MRM7m, OneArgFPRW>;  // ST(0) = [mem32real] / ST(0)
def FDIVR64m  : FPI64m<"fdivr",  0xDC, MRM7m, OneArgFPRW>;  // ST(0) = [mem64real] / ST(0)
def FIDIVR16m : FPI16m<"fidivr", 0xDE, MRM7m, OneArgFPRW>;  // ST(0) = [mem16int] / ST(0)
def FIDIVR32m : FPI32m<"fidivr", 0xDA, MRM7m, OneArgFPRW>;  // ST(0) = [mem32int] / ST(0)


// Floating point cmovs...
let isTwoAddress = 1, Uses = [ST0], Defs = [ST0] in {
  def FCMOVB  : FPI<0xC0, AddRegFrm, CondMovFP,
                    (ops RST:$op), "fcmovb %ST(0), $op">, DA;
  def FCMOVBE : FPI<0xD0, AddRegFrm, CondMovFP,
                    (ops RST:$op), "fcmovbe %ST(0), $op">, DA;
  def FCMOVE  : FPI<0xC8, AddRegFrm, CondMovFP,
                    (ops RST:$op), "fcmove %ST(0), $op">, DA;
  def FCMOVAE : FPI<0xC0, AddRegFrm, CondMovFP,
                    (ops RST:$op), "fcmovae %ST(0), $op">, DB;
  def FCMOVA  : FPI<0xD0, AddRegFrm, CondMovFP,
                    (ops RST:$op), "fcmova %ST(0), $op">, DB;
  def FCMOVNE : FPI<0xC8, AddRegFrm, CondMovFP,
                    (ops RST:$op), "fcmovne %ST(0), $op">, DB;
}

// Floating point loads & stores...
let Name = "fld" in
def FLDrr   : FPI<0xC0, AddRegFrm, NotFP, (ops RST:$op), "fld $op">, D9;
def FLD32m  : FPI32m <"fld"   , 0xD9, MRM0m    , ZeroArgFP>;        // load float
def FLD64m  : FPI64m <"fld"   , 0xDD, MRM0m    , ZeroArgFP>;        // load double
def FLD80m  : FPI80m <"fld"   , 0xDB, MRM5m    , ZeroArgFP>;        // load extended
def FILD16m : FPI16m <"fild"  , 0xDF, MRM0m    , ZeroArgFP>;        // load signed short
def FILD32m : FPI32m <"fild"  , 0xDB, MRM0m    , ZeroArgFP>;        // load signed int
def FILD64m : FPI64m <"fild"  , 0xDF, MRM5m    , ZeroArgFP>;        // load signed long

def FSTrr  : FPI<0xD0, AddRegFrm, NotFP, (ops RST:$op), "fst $op">, DD;      // ST(i) = ST(0)
def FSTPrr : FPI<0xD8, AddRegFrm, NotFP, (ops RST:$op), "fstp $op">, DD;     // ST(i) = ST(0), pop
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

def FXCH     : FPI<0xC8, AddRegFrm, NotFP, (ops RST:$op), "fxch $op">, D9;      // fxch ST(i), ST(0)

// Floating point constant loads...
def FLD0 : FPI<0xEE, RawFrm, ZeroArgFP, (ops), "fldz">, D9;
def FLD1 : FPI<0xE8, RawFrm, ZeroArgFP, (ops), "fld1">, D9;


// Unary operations...
def FCHS : FPI<0xE0, RawFrm, OneArgFPRW, (ops), "fchs">, D9;           // f1 = fchs f2
def FTST : FPI<0xE4, RawFrm, OneArgFP, (ops), "ftst">, D9;             // ftst ST(0)

// Binary arithmetic operations...
class FPST0rInst<bits<8> o, dag ops, string asm> : I<o, AddRegFrm, ops, asm>, D8 {
  list<Register> Uses = [ST0];
  list<Register> Defs = [ST0];
}
class FPrST0Inst<bits<8> o, dag ops, string asm> : I<o, AddRegFrm, ops, asm>, DC {
  list<Register> Uses = [ST0];
}
class FPrST0PInst<bits<8> o, dag ops, string asm> : I<o, AddRegFrm, ops, asm>, DE {
  list<Register> Uses = [ST0];
}

def FADDST0r   : FPST0rInst <0xC0, (ops RST:$op), "fadd $op">;
def FADDrST0   : FPrST0Inst <0xC0, (ops RST:$op), "fadd $op, %ST(0)">;
def FADDPrST0  : FPrST0PInst<0xC0, (ops RST:$op), "faddp $op">;

def FSUBRST0r  : FPST0rInst <0xE8, (ops RST:$op), "fsubr $op">;
def FSUBrST0   : FPrST0Inst <0xE8, (ops RST:$op), "fsub $op, %ST(0)">;
def FSUBPrST0  : FPrST0PInst<0xE8, (ops RST:$op), "fsubp $op">;

def FSUBST0r   : FPST0rInst <0xE0, (ops RST:$op), "fsub $op">;
def FSUBRrST0  : FPrST0Inst <0xE0, (ops RST:$op), "fsubr $op, %ST(0)">;
def FSUBRPrST0 : FPrST0PInst<0xE0, (ops RST:$op), "fsubrp $op">;

def FMULST0r   : FPST0rInst <0xC8, (ops RST:$op), "fmul $op">;
def FMULrST0   : FPrST0Inst <0xC8, (ops RST:$op), "fmul $op, %ST(0)">;
def FMULPrST0  : FPrST0PInst<0xC8, (ops RST:$op), "fmulp $op">;

def FDIVRST0r  : FPST0rInst <0xF8, (ops RST:$op), "fdivr $op">;
def FDIVrST0   : FPrST0Inst <0xF8, (ops RST:$op), "fdiv $op, %ST(0)">;
def FDIVPrST0  : FPrST0PInst<0xF8, (ops RST:$op), "fdivp $op">;

def FDIVST0r   : FPST0rInst <0xF0, (ops RST:$op), "fdiv $op">;           // ST(0) = ST(0) / ST(i)
def FDIVRrST0  : FPrST0Inst <0xF0, (ops RST:$op), "fdivr $op, %ST(0)">;  // ST(i) = ST(0) / ST(i)
def FDIVRPrST0 : FPrST0PInst<0xF0, (ops RST:$op), "fdivrp $op">;         // ST(i) = ST(0) / ST(i), pop

// Floating point compares
def FUCOMr    : FPI<0xE0, AddRegFrm, CompareFP, (ops RST:$reg),
                    "fucom $reg">, DD, Imp<[ST0],[]>;          // FPSW = compare ST(0) with ST(i)
def FUCOMPr   : I<0xE8, AddRegFrm, (ops RST:$reg),
                  "fucomp $reg">, DD, Imp<[ST0],[]>;  // FPSW = compare ST(0) with ST(i), pop
def FUCOMPPr  : I<0xE9, RawFrm, (ops),
                  "fucompp">, DA, Imp<[ST0],[]>;  // compare ST(0) with ST(1), pop, pop

def FUCOMIr  : FPI<0xE8, AddRegFrm, CompareFP, (ops RST:$reg),
                   "fucomi %ST(0), $reg">, DB, Imp<[ST0],[]>; // CC = compare ST(0) with ST(i)
def FUCOMIPr : I<0xE8, AddRegFrm, (ops RST:$reg),
                 "fucomip %ST(0), $reg">, DF, Imp<[ST0],[]>;  // CC = compare ST(0) with ST(i), pop


// Floating point flag ops
def FNSTSW8r  : I<0xE0, RawFrm, (ops), "fnstsw">, DF, Imp<[],[AX]>;   // AX = fp flags

def FNSTCW16m : I<0xD9, MRM7m, (ops i16mem:$dst), "fnstcw $dst">; // [mem16] = X87 control world
def FLDCW16m  : I<0xD9, MRM5m, (ops i16mem:$dst), "fldcw $dst">;  // X87 control world = [mem16]
