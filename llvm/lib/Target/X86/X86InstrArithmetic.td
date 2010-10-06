//===- X86InstrArithmetic.td - Integer Arithmetic Instrs ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the integer arithmetic instructions in the X86
// architecture.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LEA - Load Effective Address

let neverHasSideEffects = 1 in
def LEA16r   : I<0x8D, MRMSrcMem,
                 (outs GR16:$dst), (ins i32mem:$src),
                 "lea{w}\t{$src|$dst}, {$dst|$src}", []>, OpSize;
let isReMaterializable = 1 in
def LEA32r   : I<0x8D, MRMSrcMem,
                 (outs GR32:$dst), (ins i32mem:$src),
                 "lea{l}\t{$src|$dst}, {$dst|$src}",
                 [(set GR32:$dst, lea32addr:$src)]>, Requires<[In32BitMode]>;

def LEA64_32r : I<0x8D, MRMSrcMem,
                  (outs GR32:$dst), (ins lea64_32mem:$src),
                  "lea{l}\t{$src|$dst}, {$dst|$src}",
                  [(set GR32:$dst, lea32addr:$src)]>, Requires<[In64BitMode]>;

let isReMaterializable = 1 in
def LEA64r   : RI<0x8D, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                  "lea{q}\t{$src|$dst}, {$dst|$src}",
                  [(set GR64:$dst, lea64addr:$src)]>;



//===----------------------------------------------------------------------===//
//  Fixed-Register Multiplication and Division Instructions.
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
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX], neverHasSideEffects = 1 in
def MUL64r : RI<0xF7, MRM4r, (outs), (ins GR64:$src),
                "mul{q}\t$src", []>;         // RAX,RDX = RAX*GR64

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
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX], neverHasSideEffects = 1 in
def MUL64m : RI<0xF7, MRM4m, (outs), (ins i64mem:$src),
                "mul{q}\t$src", []>;         // RAX,RDX = RAX*[mem64]
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
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX], neverHasSideEffects = 1 in
def IMUL64r : RI<0xF7, MRM5r, (outs), (ins GR64:$src), "imul{q}\t$src", []>;
              // RAX,RDX = RAX*GR64

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
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX], neverHasSideEffects = 1 in
def IMUL64m : RI<0xF7, MRM5m, (outs), (ins i64mem:$src),
                 "imul{q}\t$src", []>;         // RAX,RDX = RAX*[mem64]
}
} // neverHasSideEffects


let Defs = [EFLAGS] in {
let Constraints = "$src1 = $dst" in {

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
def IMUL64rr : RI<0xAF, MRMSrcReg, (outs GR64:$dst),
                                   (ins GR64:$src1, GR64:$src2),
                  "imul{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86smul_flag GR64:$src1, GR64:$src2))]>, TB;
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
def IMUL64rm : RI<0xAF, MRMSrcMem, (outs GR64:$dst),
                                   (ins GR64:$src1, i64mem:$src2),
                  "imul{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86smul_flag GR64:$src1, (load addr:$src2)))]>, TB;
} // Constraints = "$src1 = $dst"

} // Defs = [EFLAGS]

// Suprisingly enough, these are not two address instructions!
let Defs = [EFLAGS] in {
// Register-Integer Signed Integer Multiply
def IMUL16rri  : Ii16<0x69, MRMSrcReg,                      // GR16 = GR16*I16
                      (outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                      "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR16:$dst, EFLAGS, 
                            (X86smul_flag GR16:$src1, imm:$src2))]>, OpSize;
def IMUL16rri8 : Ii8<0x6B, MRMSrcReg,                       // GR16 = GR16*I8
                     (outs GR16:$dst), (ins GR16:$src1, i16i8imm:$src2),
                     "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR16:$dst, EFLAGS,
                           (X86smul_flag GR16:$src1, i16immSExt8:$src2))]>,
                 OpSize;
def IMUL32rri  : Ii32<0x69, MRMSrcReg,                      // GR32 = GR32*I32
                      (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                      "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR32:$dst, EFLAGS,
                            (X86smul_flag GR32:$src1, imm:$src2))]>;
def IMUL32rri8 : Ii8<0x6B, MRMSrcReg,                       // GR32 = GR32*I8
                     (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                     "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR32:$dst, EFLAGS,
                           (X86smul_flag GR32:$src1, i32immSExt8:$src2))]>;
def IMUL64rri32 : RIi32<0x69, MRMSrcReg,                    // GR64 = GR64*I32
                        (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                        "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                       [(set GR64:$dst, EFLAGS,
                             (X86smul_flag GR64:$src1, i64immSExt32:$src2))]>;
def IMUL64rri8 : RIi8<0x6B, MRMSrcReg,                      // GR64 = GR64*I8
                      (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                      "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR64:$dst, EFLAGS,
                            (X86smul_flag GR64:$src1, i64immSExt8:$src2))]>;


// Memory-Integer Signed Integer Multiply
def IMUL16rmi  : Ii16<0x69, MRMSrcMem,                     // GR16 = [mem16]*I16
                      (outs GR16:$dst), (ins i16mem:$src1, i16imm:$src2),
                      "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR16:$dst, EFLAGS,
                            (X86smul_flag (load addr:$src1), imm:$src2))]>,
                 OpSize;
def IMUL16rmi8 : Ii8<0x6B, MRMSrcMem,                       // GR16 = [mem16]*I8
                     (outs GR16:$dst), (ins i16mem:$src1, i16i8imm :$src2),
                     "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR16:$dst, EFLAGS,
                           (X86smul_flag (load addr:$src1),
                                         i16immSExt8:$src2))]>, OpSize;
def IMUL32rmi  : Ii32<0x69, MRMSrcMem,                     // GR32 = [mem32]*I32
                      (outs GR32:$dst), (ins i32mem:$src1, i32imm:$src2),
                      "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR32:$dst, EFLAGS,
                            (X86smul_flag (load addr:$src1), imm:$src2))]>;
def IMUL32rmi8 : Ii8<0x6B, MRMSrcMem,                       // GR32 = [mem32]*I8
                     (outs GR32:$dst), (ins i32mem:$src1, i32i8imm: $src2),
                     "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR32:$dst, EFLAGS,
                           (X86smul_flag (load addr:$src1),
                                         i32immSExt8:$src2))]>;
def IMUL64rmi32 : RIi32<0x69, MRMSrcMem,                   // GR64 = [mem64]*I32
                        (outs GR64:$dst), (ins i64mem:$src1, i64i32imm:$src2),
                        "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                        [(set GR64:$dst, EFLAGS,
                              (X86smul_flag (load addr:$src1),
                                            i64immSExt32:$src2))]>;
def IMUL64rmi8 : RIi8<0x6B, MRMSrcMem,                      // GR64 = [mem64]*I8
                      (outs GR64:$dst), (ins i64mem:$src1, i64i8imm: $src2),
                      "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR64:$dst, EFLAGS,
                            (X86smul_flag (load addr:$src1),
                                          i64immSExt8:$src2))]>;
} // Defs = [EFLAGS]




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
// RDX:RAX/r64 = RAX,RDX
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX,RDX] in
def DIV64r : RI<0xF7, MRM6r, (outs), (ins GR64:$src),
                "div{q}\t$src", []>;

let mayLoad = 1 in {
let Defs = [AL,EFLAGS,AX], Uses = [AX] in
def DIV8m  : I<0xF6, MRM6m, (outs), (ins i8mem:$src),   // AX/[mem8] = AL,AH
               "div{b}\t$src", []>;
let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
def DIV16m : I<0xF7, MRM6m, (outs), (ins i16mem:$src),  // DX:AX/[mem16] = AX,DX
               "div{w}\t$src", []>, OpSize;
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in    // EDX:EAX/[mem32] = EAX,EDX
def DIV32m : I<0xF7, MRM6m, (outs), (ins i32mem:$src),
               "div{l}\t$src", []>;
// RDX:RAX/[mem64] = RAX,RDX
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX,RDX] in
def DIV64m : RI<0xF7, MRM6m, (outs), (ins i64mem:$src),
                "div{q}\t$src", []>;
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
// RDX:RAX/r64 = RAX,RDX
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX,RDX] in
def IDIV64r: RI<0xF7, MRM7r, (outs), (ins GR64:$src),
                "idiv{q}\t$src", []>;
               
let mayLoad = 1, mayLoad = 1 in {
let Defs = [AL,EFLAGS,AX], Uses = [AX] in
def IDIV8m : I<0xF6, MRM7m, (outs), (ins i8mem:$src),   // AX/[mem8] = AL,AH
               "idiv{b}\t$src", []>;
let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
def IDIV16m: I<0xF7, MRM7m, (outs), (ins i16mem:$src),  // DX:AX/[mem16] = AX,DX
               "idiv{w}\t$src", []>, OpSize;
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in    // EDX:EAX/[mem32] = EAX,EDX
def IDIV32m: I<0xF7, MRM7m, (outs), (ins i32mem:$src), 
               "idiv{l}\t$src", []>;
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX,RDX] in // RDX:RAX/[mem64] = RAX,RDX
def IDIV64m: RI<0xF7, MRM7m, (outs), (ins i64mem:$src),
                "idiv{q}\t$src", []>;
}

//===----------------------------------------------------------------------===//
//  Two address Instructions.
//

// unary instructions
let CodeSize = 2 in {
let Defs = [EFLAGS] in {
let Constraints = "$src1 = $dst" in {
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
def NEG64r : RI<0xF7, MRM3r, (outs GR64:$dst), (ins GR64:$src1), "neg{q}\t$dst",
                [(set GR64:$dst, (ineg GR64:$src1)),
                 (implicit EFLAGS)]>;
} // Constraints = "$src1 = $dst"

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
def NEG64m : RI<0xF7, MRM3m, (outs), (ins i64mem:$dst), "neg{q}\t$dst",
                [(store (ineg (loadi64 addr:$dst)), addr:$dst),
                 (implicit EFLAGS)]>;
} // Defs = [EFLAGS]


// Note: NOT does not set EFLAGS!

let Constraints = "$src1 = $dst" in {
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
def NOT64r : RI<0xF7, MRM2r, (outs GR64:$dst), (ins GR64:$src1), "not{q}\t$dst",
                [(set GR64:$dst, (not GR64:$src1))]>;
}
} // Constraints = "$src1 = $dst"

def NOT8m  : I<0xF6, MRM2m, (outs), (ins i8mem :$dst),
               "not{b}\t$dst",
               [(store (not (loadi8 addr:$dst)), addr:$dst)]>;
def NOT16m : I<0xF7, MRM2m, (outs), (ins i16mem:$dst),
               "not{w}\t$dst",
               [(store (not (loadi16 addr:$dst)), addr:$dst)]>, OpSize;
def NOT32m : I<0xF7, MRM2m, (outs), (ins i32mem:$dst),
               "not{l}\t$dst",
               [(store (not (loadi32 addr:$dst)), addr:$dst)]>;
def NOT64m : RI<0xF7, MRM2m, (outs), (ins i64mem:$dst), "not{q}\t$dst",
                [(store (not (loadi64 addr:$dst)), addr:$dst)]>;
} // CodeSize

// TODO: inc/dec is slow for P4, but fast for Pentium-M.
let Defs = [EFLAGS] in {
let Constraints = "$src1 = $dst" in {
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
def INC64r : RI<0xFF, MRM0r, (outs GR64:$dst), (ins GR64:$src1), "inc{q}\t$dst",
                [(set GR64:$dst, EFLAGS, (X86inc_flag GR64:$src1))]>;
} // isConvertibleToThreeAddress = 1, CodeSize = 1


// In 64-bit mode, single byte INC and DEC cannot be encoded.
let isConvertibleToThreeAddress = 1, CodeSize = 2 in {
// Can transform into LEA.
def INC64_16r : I<0xFF, MRM0r, (outs GR16:$dst), (ins GR16:$src1), 
                  "inc{w}\t$dst",
                  [(set GR16:$dst, EFLAGS, (X86inc_flag GR16:$src1))]>,
                OpSize, Requires<[In64BitMode]>;
def INC64_32r : I<0xFF, MRM0r, (outs GR32:$dst), (ins GR32:$src1), 
                  "inc{l}\t$dst",
                  [(set GR32:$dst, EFLAGS, (X86inc_flag GR32:$src1))]>,
                Requires<[In64BitMode]>;
def DEC64_16r : I<0xFF, MRM1r, (outs GR16:$dst), (ins GR16:$src1), 
                  "dec{w}\t$dst",
                  [(set GR16:$dst, EFLAGS, (X86dec_flag GR16:$src1))]>,
                OpSize, Requires<[In64BitMode]>;
def DEC64_32r : I<0xFF, MRM1r, (outs GR32:$dst), (ins GR32:$src1), 
                  "dec{l}\t$dst",
                  [(set GR32:$dst, EFLAGS, (X86dec_flag GR32:$src1))]>,
                Requires<[In64BitMode]>;
} // isConvertibleToThreeAddress = 1, CodeSize = 2

} // Constraints = "$src1 = $dst"

let CodeSize = 2 in {
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
  def INC64m : RI<0xFF, MRM0m, (outs), (ins i64mem:$dst), "inc{q}\t$dst",
                  [(store (add (loadi64 addr:$dst), 1), addr:$dst),
                   (implicit EFLAGS)]>;
                   
// These are duplicates of their 32-bit counterparts. Only needed so X86 knows
// how to unfold them.
// FIXME: What is this for??
def INC64_16m : I<0xFF, MRM0m, (outs), (ins i16mem:$dst), "inc{w}\t$dst",
                  [(store (add (loadi16 addr:$dst), 1), addr:$dst),
                    (implicit EFLAGS)]>,
                OpSize, Requires<[In64BitMode]>;
def INC64_32m : I<0xFF, MRM0m, (outs), (ins i32mem:$dst), "inc{l}\t$dst",
                  [(store (add (loadi32 addr:$dst), 1), addr:$dst),
                    (implicit EFLAGS)]>,
                Requires<[In64BitMode]>;
def DEC64_16m : I<0xFF, MRM1m, (outs), (ins i16mem:$dst), "dec{w}\t$dst",
                  [(store (add (loadi16 addr:$dst), -1), addr:$dst),
                    (implicit EFLAGS)]>,
                OpSize, Requires<[In64BitMode]>;
def DEC64_32m : I<0xFF, MRM1m, (outs), (ins i32mem:$dst), "dec{l}\t$dst",
                  [(store (add (loadi32 addr:$dst), -1), addr:$dst),
                    (implicit EFLAGS)]>,
                Requires<[In64BitMode]>;
} // CodeSize = 2

let Constraints = "$src1 = $dst" in {
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
def DEC64r : RI<0xFF, MRM1r, (outs GR64:$dst), (ins GR64:$src1), "dec{q}\t$dst",
                [(set GR64:$dst, EFLAGS, (X86dec_flag GR64:$src1))]>;
} // CodeSize = 2
} // Constraints = "$src1 = $dst"


let CodeSize = 2 in {
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
  def DEC64m : RI<0xFF, MRM1m, (outs), (ins i64mem:$dst), "dec{q}\t$dst",
                  [(store (add (loadi64 addr:$dst), -1), addr:$dst),
                   (implicit EFLAGS)]>;
} // CodeSize = 2
} // Defs = [EFLAGS]

/// X86TypeInfo - This is a bunch of information that describes relevant X86
/// information about value types.  For example, it can tell you what the
/// register class and preferred load to use.
class X86TypeInfo<ValueType vt, string instrsuffix, RegisterClass regclass,
                  PatFrag loadnode, X86MemOperand memoperand> {
  /// VT - This is the value type itself.
  ValueType VT = vt;
  
  /// InstrSuffix - This is the suffix used on instructions with this type.  For
  /// example, i8 -> "b", i16 -> "w", i32 -> "l", i64 -> "q".
  string InstrSuffix = instrsuffix;
  
  /// RegClass - This is the register class associated with this type.  For
  /// example, i8 -> GR8, i16 -> GR16, i32 -> GR32, i64 -> GR64.
  RegisterClass RegClass = regclass;
  
  /// LoadNode - This is the load node associated with this type.  For
  /// example, i8 -> loadi8, i16 -> loadi16, i32 -> loadi32, i64 -> loadi64.
  PatFrag LoadNode = loadnode;
  
  /// MemOperand - This is the memory operand associated with this type.  For
  /// example, i8 -> i8mem, i16 -> i16mem, i32 -> i32mem, i64 -> i64mem.
  X86MemOperand MemOperand = memoperand;
}

def Xi8  : X86TypeInfo<i8 , "b", GR8 , loadi8 , i8mem>;
def Xi16 : X86TypeInfo<i16, "w", GR16, loadi16, i16mem>;
def Xi32 : X86TypeInfo<i32, "l", GR32, loadi32, i32mem>;
def Xi64 : X86TypeInfo<i64, "q", GR64, loadi64, i64mem>;


class BinOpRR<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
              SDNode opnode, Format format>
  : I<opcode, format,
      (outs typeinfo.RegClass:$dst),
      (ins typeinfo.RegClass:$src1, typeinfo.RegClass:$src2),
      !strconcat(mnemonic, "{", typeinfo.InstrSuffix,
                 "}\t{$src2, $dst|$dst, $src2}"),
      [(set typeinfo.RegClass:$dst, EFLAGS,
            (opnode typeinfo.RegClass:$src1, typeinfo.RegClass:$src2))]>;


class BinOpRM<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
              SDNode opnode>
  : I<opcode, MRMSrcMem,
      (outs typeinfo.RegClass:$dst),
      (ins typeinfo.RegClass:$src1, typeinfo.MemOperand:$src2),
      !strconcat(mnemonic, "{", typeinfo.InstrSuffix,
                 "}\t{$src2, $dst|$dst, $src2}"),
      [(set typeinfo.RegClass:$dst, EFLAGS,
            (opnode typeinfo.RegClass:$src1, (typeinfo.LoadNode addr:$src2)))]>;


// Logical operators.
let Defs = [EFLAGS] in {
let Constraints = "$src1 = $dst" in {

let isCommutable = 1 in {   // X = AND Y, Z   --> X = AND Z, Y
def AND8rr  : BinOpRR<0x20, "and", Xi8 , X86and_flag, MRMDestReg>;
def AND16rr : BinOpRR<0x21, "and", Xi16, X86and_flag, MRMDestReg>, OpSize;
def AND32rr : BinOpRR<0x21, "and", Xi32, X86and_flag, MRMDestReg>;
def AND64rr : BinOpRR<0x21, "and", Xi64, X86and_flag, MRMDestReg>, REX_W;
} // isCommutable


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
def AND64rr_REV : RI<0x23, MRMSrcReg, (outs GR64:$dst), 
                     (ins GR64:$src1, GR64:$src2),
                     "and{q}\t{$src2, $dst|$dst, $src2}", []>;
}

def AND8rm   : BinOpRM<0x22, "and", Xi8 , X86and_flag>;
def AND16rm  : BinOpRM<0x23, "and", Xi16, X86and_flag>, OpSize;
def AND32rm  : BinOpRM<0x23, "and", Xi32, X86and_flag>;
def AND64rm  : BinOpRM<0x23, "and", Xi64, X86and_flag>, REX_W;

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
def AND64ri32  : RIi32<0x81, MRM4r, 
                       (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                       "and{q}\t{$src2, $dst|$dst, $src2}",
                       [(set GR64:$dst, EFLAGS,
                             (X86and_flag GR64:$src1, i64immSExt32:$src2))]>;

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
def AND64ri8 : RIi8<0x83, MRM4r, 
                    (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                    "and{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, EFLAGS,
                          (X86and_flag GR64:$src1, i64immSExt8:$src2))]>;
} // Constraints = "$src1 = $dst"

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
def AND64mr  : RI<0x21, MRMDestMem,
                  (outs), (ins i64mem:$dst, GR64:$src),
                  "and{q}\t{$src, $dst|$dst, $src}",
                  [(store (and (load addr:$dst), GR64:$src), addr:$dst),
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
def AND64mi32  : RIi32<0x81, MRM4m,
                       (outs), (ins i64mem:$dst, i64i32imm:$src),
                       "and{q}\t{$src, $dst|$dst, $src}",
             [(store (and (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst),
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
def AND64mi8 : RIi8<0x83, MRM4m,
                    (outs), (ins i64mem:$dst, i64i8imm :$src),
                    "and{q}\t{$src, $dst|$dst, $src}",
                 [(store (and (load addr:$dst), i64immSExt8:$src), addr:$dst),
                  (implicit EFLAGS)]>;

// FIXME: Implicitly modifiers AL.
def AND8i8 : Ii8<0x24, RawFrm, (outs), (ins i8imm:$src),
                 "and{b}\t{$src, %al|%al, $src}", []>;
def AND16i16 : Ii16<0x25, RawFrm, (outs), (ins i16imm:$src),
                    "and{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def AND32i32 : Ii32<0x25, RawFrm, (outs), (ins i32imm:$src),
                    "and{l}\t{$src, %eax|%eax, $src}", []>;
def AND64i32 : RIi32<0x25, RawFrm, (outs), (ins i64i32imm:$src),
                     "and{q}\t{$src, %rax|%rax, $src}", []>;

let Constraints = "$src1 = $dst" in {

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
def OR64rr   : RI<0x09, MRMDestReg, (outs GR64:$dst), 
                  (ins GR64:$src1, GR64:$src2),
                  "or{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86or_flag GR64:$src1, GR64:$src2))]>;
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
def OR64rr_REV : RI<0x0B, MRMSrcReg, (outs GR64:$dst), 
                    (ins GR64:$src1, GR64:$src2),
                    "or{q}\t{$src2, $dst|$dst, $src2}", []>;
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
def OR64rm   : RI<0x0B, MRMSrcMem , (outs GR64:$dst),
                  (ins GR64:$src1, i64mem:$src2),
                  "or{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86or_flag GR64:$src1, (load addr:$src2)))]>;

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
def OR64ri32 : RIi32<0x81, MRM1r, (outs GR64:$dst),
                     (ins GR64:$src1, i64i32imm:$src2),
                     "or{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86or_flag GR64:$src1, i64immSExt32:$src2))]>;

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
def OR64ri8  : RIi8<0x83, MRM1r, (outs GR64:$dst),
                    (ins GR64:$src1, i64i8imm:$src2),
                    "or{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, EFLAGS,
                         (X86or_flag GR64:$src1, i64immSExt8:$src2))]>;
} // Constraints = "$src1 = $dst"

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
def OR64mr : RI<0x09, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                "or{q}\t{$src, $dst|$dst, $src}",
                [(store (or (load addr:$dst), GR64:$src), addr:$dst),
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
def OR64mi32 : RIi32<0x81, MRM1m, (outs), (ins i64mem:$dst, i64i32imm:$src),
                     "or{q}\t{$src, $dst|$dst, $src}",
              [(store (or (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst),
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
def OR64mi8  : RIi8<0x83, MRM1m, (outs), (ins i64mem:$dst, i64i8imm:$src),
                    "or{q}\t{$src, $dst|$dst, $src}",
                  [(store (or (load addr:$dst), i64immSExt8:$src), addr:$dst),
                   (implicit EFLAGS)]>;

def OR8i8 : Ii8 <0x0C, RawFrm, (outs), (ins i8imm:$src),
                 "or{b}\t{$src, %al|%al, $src}", []>;
def OR16i16 : Ii16 <0x0D, RawFrm, (outs), (ins i16imm:$src),
                    "or{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def OR32i32 : Ii32 <0x0D, RawFrm, (outs), (ins i32imm:$src),
                    "or{l}\t{$src, %eax|%eax, $src}", []>;
def OR64i32 : RIi32<0x0D, RawFrm, (outs), (ins i64i32imm:$src),
                    "or{q}\t{$src, %rax|%rax, $src}", []>;


let Constraints = "$src1 = $dst" in {

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
  def XOR64rr  : RI<0x31, MRMDestReg,  (outs GR64:$dst), 
                    (ins GR64:$src1, GR64:$src2), 
                    "xor{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, EFLAGS,
                          (X86xor_flag GR64:$src1, GR64:$src2))]>;
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
def XOR64rr_REV : RI<0x33, MRMSrcReg, (outs GR64:$dst), 
                     (ins GR64:$src1, GR64:$src2),
                    "xor{q}\t{$src2, $dst|$dst, $src2}", []>;
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
def XOR64rm  : RI<0x33, MRMSrcMem, (outs GR64:$dst), 
                  (ins GR64:$src1, i64mem:$src2), 
                  "xor{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86xor_flag GR64:$src1, (load addr:$src2)))]>;

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
def XOR64ri32 : RIi32<0x81, MRM6r, 
                      (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2), 
                      "xor{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, EFLAGS,
                            (X86xor_flag GR64:$src1, i64immSExt32:$src2))]>;

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
def XOR64ri8 : RIi8<0x83, MRM6r,  (outs GR64:$dst), 
                    (ins GR64:$src1, i64i8imm:$src2),
                    "xor{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, EFLAGS,
                          (X86xor_flag GR64:$src1, i64immSExt8:$src2))]>;
} // Constraints = "$src1 = $dst"


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
def XOR64mr  : RI<0x31, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                  "xor{q}\t{$src, $dst|$dst, $src}",
                  [(store (xor (load addr:$dst), GR64:$src), addr:$dst),
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
def XOR64mi32 : RIi32<0x81, MRM6m, (outs), (ins i64mem:$dst, i64i32imm:$src),
                      "xor{q}\t{$src, $dst|$dst, $src}",
             [(store (xor (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst),
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
def XOR64mi8 : RIi8<0x83, MRM6m, (outs), (ins i64mem:$dst, i64i8imm :$src),
                    "xor{q}\t{$src, $dst|$dst, $src}",
                 [(store (xor (load addr:$dst), i64immSExt8:$src), addr:$dst),
                  (implicit EFLAGS)]>;

def XOR8i8   : Ii8 <0x34, RawFrm, (outs), (ins i8imm:$src),
                    "xor{b}\t{$src, %al|%al, $src}", []>;
def XOR16i16 : Ii16<0x35, RawFrm, (outs), (ins i16imm:$src),
                    "xor{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def XOR32i32 : Ii32<0x35, RawFrm, (outs), (ins i32imm:$src),
                    "xor{l}\t{$src, %eax|%eax, $src}", []>;
def XOR64i32 : RIi32<0x35, RawFrm, (outs), (ins i64i32imm:$src),
                     "xor{q}\t{$src, %rax|%rax, $src}", []>;
} // Defs = [EFLAGS]


// Arithmetic.
let Defs = [EFLAGS] in {
let Constraints = "$src1 = $dst" in {
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
def ADD64rr    : RI<0x01, MRMDestReg, (outs GR64:$dst), 
                    (ins GR64:$src1, GR64:$src2),
                    "add{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, EFLAGS,
                          (X86add_flag GR64:$src1, GR64:$src2))]>;
} // end isConvertibleToThreeAddress
} // end isCommutable

// These are alternate spellings for use by the disassembler, we mark them as
// code gen only to ensure they aren't matched by the assembler.
let isCodeGenOnly = 1 in {
  def ADD8rr_alt: I<0x02, MRMSrcReg,
                    (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                   "add{b}\t{$src2, $dst|$dst, $src2}", []>;
  def ADD16rr_alt: I<0x03, MRMSrcReg,
                    (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                    "add{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize;
  def ADD32rr_alt: I<0x03, MRMSrcReg,
                     (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                    "add{l}\t{$src2, $dst|$dst, $src2}", []>;
  def ADD64rr_alt : RI<0x03, MRMSrcReg,
                       (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
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
def ADD64rm     : RI<0x03, MRMSrcMem, (outs GR64:$dst), 
                     (ins GR64:$src1, i64mem:$src2),
                     "add{q}\t{$src2, $dst|$dst, $src2}",
                     [(set GR64:$dst, EFLAGS,
                           (X86add_flag GR64:$src1, (load addr:$src2)))]>;

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
def ADD64ri8  : RIi8<0x83, MRM0r, (outs GR64:$dst), 
                     (ins GR64:$src1, i64i8imm:$src2),
                     "add{q}\t{$src2, $dst|$dst, $src2}",
                     [(set GR64:$dst, EFLAGS,
                           (X86add_flag GR64:$src1, i64immSExt8:$src2))]>;
def ADD64ri32 : RIi32<0x81, MRM0r, (outs GR64:$dst), 
                      (ins GR64:$src1, i64i32imm:$src2),
                      "add{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, EFLAGS,
                            (X86add_flag GR64:$src1, i64immSExt32:$src2))]>;
}
} // Constraints = "$src1 = $dst"

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
def ADD64mr  : RI<0x01, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                  "add{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (add (load addr:$dst), GR64:$src2), addr:$dst),
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
def ADD64mi32 : RIi32<0x81, MRM0m, (outs), (ins i64mem:$dst, i64i32imm :$src2),
                      "add{q}\t{$src2, $dst|$dst, $src2}",
               [(store (add (load addr:$dst), i64immSExt32:$src2), addr:$dst),
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
def ADD64mi8 : RIi8<0x83, MRM0m, (outs), (ins i64mem:$dst, i64i8imm :$src2),
                    "add{q}\t{$src2, $dst|$dst, $src2}",
                [(store (add (load addr:$dst), i64immSExt8:$src2), addr:$dst),
                 (implicit EFLAGS)]>;

// addition to rAX
def ADD8i8 : Ii8<0x04, RawFrm, (outs), (ins i8imm:$src),
                 "add{b}\t{$src, %al|%al, $src}", []>;
def ADD16i16 : Ii16<0x05, RawFrm, (outs), (ins i16imm:$src),
                    "add{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def ADD32i32 : Ii32<0x05, RawFrm, (outs), (ins i32imm:$src),
                    "add{l}\t{$src, %eax|%eax, $src}", []>;
def ADD64i32 : RIi32<0x05, RawFrm, (outs), (ins i64i32imm:$src),
                     "add{q}\t{$src, %rax|%rax, $src}", []>;

let Uses = [EFLAGS] in {
let Constraints = "$src1 = $dst" in {
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
def ADC64rr  : RI<0x11, MRMDestReg, (outs GR64:$dst), 
                  (ins GR64:$src1, GR64:$src2),
                  "adc{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (adde GR64:$src1, GR64:$src2))]>;
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
def ADC64rr_REV : RI<0x13, MRMSrcReg , (outs GR32:$dst), 
                     (ins GR64:$src1, GR64:$src2),
                    "adc{q}\t{$src2, $dst|$dst, $src2}", []>;
}

def ADC8rm   : I<0x12, MRMSrcMem ,
                 (outs GR8:$dst), (ins GR8:$src1, i8mem:$src2),
                 "adc{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, (adde GR8:$src1, (load addr:$src2)))]>;
def ADC16rm  : I<0x13, MRMSrcMem , (outs GR16:$dst),
                                   (ins GR16:$src1, i16mem:$src2),
                 "adc{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (adde GR16:$src1, (load addr:$src2)))]>,
                 OpSize;
def ADC32rm  : I<0x13, MRMSrcMem ,
                 (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                 "adc{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (adde GR32:$src1, (load addr:$src2)))]>;
def ADC64rm  : RI<0x13, MRMSrcMem , (outs GR64:$dst), 
                  (ins GR64:$src1, i64mem:$src2),
                  "adc{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (adde GR64:$src1, (load addr:$src2)))]>;
def ADC8ri   : Ii8<0x80, MRM2r, (outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                    "adc{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, (adde GR8:$src1, imm:$src2))]>;
def ADC16ri  : Ii16<0x81, MRM2r,
                    (outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
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
def ADC64ri32 : RIi32<0x81, MRM2r, (outs GR64:$dst), 
                      (ins GR64:$src1, i64i32imm:$src2),
                      "adc{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (adde GR64:$src1, i64immSExt32:$src2))]>;
def ADC64ri8 : RIi8<0x83, MRM2r, (outs GR64:$dst), 
                    (ins GR64:$src1, i64i8imm:$src2),
                    "adc{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (adde GR64:$src1, i64immSExt8:$src2))]>;
} // Constraints = "$src1 = $dst"

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
def ADC64mr  : RI<0x11, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                  "adc{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (adde (load addr:$dst), GR64:$src2), addr:$dst)]>;
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

def ADC64mi32 : RIi32<0x81, MRM2m, (outs), (ins i64mem:$dst, i64i32imm:$src2),
                      "adc{q}\t{$src2, $dst|$dst, $src2}",
                 [(store (adde (load addr:$dst), i64immSExt32:$src2), 
                  addr:$dst)]>;
def ADC64mi8 : RIi8<0x83, MRM2m, (outs), (ins i64mem:$dst, i64i8imm :$src2),
                    "adc{q}\t{$src2, $dst|$dst, $src2}",
                 [(store (adde (load addr:$dst), i64immSExt8:$src2), 
                  addr:$dst)]>;

def ADC8i8 : Ii8<0x14, RawFrm, (outs), (ins i8imm:$src),
                 "adc{b}\t{$src, %al|%al, $src}", []>;
def ADC16i16 : Ii16<0x15, RawFrm, (outs), (ins i16imm:$src),
                    "adc{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def ADC32i32 : Ii32<0x15, RawFrm, (outs), (ins i32imm:$src),
                    "adc{l}\t{$src, %eax|%eax, $src}", []>;
def ADC64i32 : RIi32<0x15, RawFrm, (outs), (ins i64i32imm:$src),
                     "adc{q}\t{$src, %rax|%rax, $src}", []>;
} // Uses = [EFLAGS]

let Constraints = "$src1 = $dst" in {

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
def SUB64rr  : RI<0x29, MRMDestReg, (outs GR64:$dst), 
                  (ins GR64:$src1, GR64:$src2),
                  "sub{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86sub_flag GR64:$src1, GR64:$src2))]>;

let isCodeGenOnly = 1 in {
def SUB8rr_REV : I<0x2A, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                   "sub{b}\t{$src2, $dst|$dst, $src2}", []>;
def SUB16rr_REV : I<0x2B, MRMSrcReg, (outs GR16:$dst), 
                    (ins GR16:$src1, GR16:$src2),
                    "sub{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize;
def SUB32rr_REV : I<0x2B, MRMSrcReg, (outs GR32:$dst), 
                    (ins GR32:$src1, GR32:$src2),
                    "sub{l}\t{$src2, $dst|$dst, $src2}", []>;
def SUB64rr_REV : RI<0x2B, MRMSrcReg, (outs GR64:$dst), 
                     (ins GR64:$src1, GR64:$src2),
                     "sub{q}\t{$src2, $dst|$dst, $src2}", []>;
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
def SUB64rm  : RI<0x2B, MRMSrcMem, (outs GR64:$dst), 
                  (ins GR64:$src1, i64mem:$src2),
                  "sub{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS, 
                        (X86sub_flag GR64:$src1, (load addr:$src2)))]>;

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
def SUB64ri32 : RIi32<0x81, MRM5r, (outs GR64:$dst),
                                   (ins GR64:$src1, i64i32imm:$src2),
                      "sub{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, EFLAGS,
                            (X86sub_flag GR64:$src1, i64immSExt32:$src2))]>;
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
def SUB64ri8 : RIi8<0x83, MRM5r, (outs GR64:$dst),
                                 (ins GR64:$src1, i64i8imm:$src2),
                    "sub{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, EFLAGS,
                          (X86sub_flag GR64:$src1, i64immSExt8:$src2))]>;
} // Constraints = "$src1 = $dst"

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
def SUB64mr  : RI<0x29, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2), 
                  "sub{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (sub (load addr:$dst), GR64:$src2), addr:$dst),
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
def SUB64mi32 : RIi32<0x81, MRM5m, (outs), (ins i64mem:$dst, i64i32imm:$src2),
                      "sub{q}\t{$src2, $dst|$dst, $src2}",
                      [(store (sub (load addr:$dst), i64immSExt32:$src2),
                              addr:$dst),
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
def SUB64mi8 : RIi8<0x83, MRM5m, (outs), (ins i64mem:$dst, i64i8imm :$src2), 
                    "sub{q}\t{$src2, $dst|$dst, $src2}",
                    [(store (sub (load addr:$dst), i64immSExt8:$src2),
                            addr:$dst),
                     (implicit EFLAGS)]>;

def SUB8i8 : Ii8<0x2C, RawFrm, (outs), (ins i8imm:$src),
                 "sub{b}\t{$src, %al|%al, $src}", []>;
def SUB16i16 : Ii16<0x2D, RawFrm, (outs), (ins i16imm:$src),
                    "sub{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def SUB32i32 : Ii32<0x2D, RawFrm, (outs), (ins i32imm:$src),
                    "sub{l}\t{$src, %eax|%eax, $src}", []>;
def SUB64i32 : RIi32<0x2D, RawFrm, (outs), (ins i64i32imm:$src),
                     "sub{q}\t{$src, %rax|%rax, $src}", []>;

let Uses = [EFLAGS] in {
let Constraints = "$src1 = $dst" in {
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
def SBB64rr    : RI<0x19, MRMDestReg, (outs GR64:$dst), 
                    (ins GR64:$src1, GR64:$src2),
                    "sbb{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sube GR64:$src1, GR64:$src2))]>;
} // Constraints = "$src1 = $dst"


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
def SBB64mr  : RI<0x19, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2), 
                  "sbb{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (sube (load addr:$dst), GR64:$src2), addr:$dst)]>;

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
def SBB64mi32 : RIi32<0x81, MRM3m, (outs), (ins i64mem:$dst, i64i32imm:$src2), 
                      "sbb{q}\t{$src2, $dst|$dst, $src2}",
              [(store (sube (load addr:$dst), i64immSExt32:$src2), addr:$dst)]>;
def SBB64mi8 : RIi8<0x83, MRM3m, (outs), (ins i64mem:$dst, i64i8imm :$src2), 
                    "sbb{q}\t{$src2, $dst|$dst, $src2}",
               [(store (sube (load addr:$dst), i64immSExt8:$src2), addr:$dst)]>;

def SBB8i8 : Ii8<0x1C, RawFrm, (outs), (ins i8imm:$src),
                 "sbb{b}\t{$src, %al|%al, $src}", []>;
def SBB16i16 : Ii16<0x1D, RawFrm, (outs), (ins i16imm:$src),
                    "sbb{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def SBB32i32 : Ii32<0x1D, RawFrm, (outs), (ins i32imm:$src),
                    "sbb{l}\t{$src, %eax|%eax, $src}", []>;
def SBB64i32 : RIi32<0x1D, RawFrm, (outs), (ins i64i32imm:$src),
                     "sbb{q}\t{$src, %rax|%rax, $src}", []>;

let Constraints = "$src1 = $dst" in {

let isCodeGenOnly = 1 in {
def SBB8rr_REV : I<0x1A, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                   "sbb{b}\t{$src2, $dst|$dst, $src2}", []>;
def SBB16rr_REV : I<0x1B, MRMSrcReg, (outs GR16:$dst), 
                    (ins GR16:$src1, GR16:$src2),
                    "sbb{w}\t{$src2, $dst|$dst, $src2}", []>, OpSize;
def SBB32rr_REV : I<0x1B, MRMSrcReg, (outs GR32:$dst), 
                    (ins GR32:$src1, GR32:$src2),
                    "sbb{l}\t{$src2, $dst|$dst, $src2}", []>;
def SBB64rr_REV : RI<0x1B, MRMSrcReg, (outs GR64:$dst), 
                     (ins GR64:$src1, GR64:$src2),
                     "sbb{q}\t{$src2, $dst|$dst, $src2}", []>;
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
def SBB64rm  : RI<0x1B, MRMSrcMem, (outs GR64:$dst), 
                  (ins GR64:$src1, i64mem:$src2),
                  "sbb{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (sube GR64:$src1, (load addr:$src2)))]>;
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
def SBB64ri32 : RIi32<0x81, MRM3r, (outs GR64:$dst), 
                      (ins GR64:$src1, i64i32imm:$src2),
                      "sbb{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (sube GR64:$src1, i64immSExt32:$src2))]>;
def SBB64ri8 : RIi8<0x83, MRM3r, (outs GR64:$dst), 
                    (ins GR64:$src1, i64i8imm:$src2),
                    "sbb{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sube GR64:$src1, i64immSExt8:$src2))]>;

} // Constraints = "$src1 = $dst"
} // Uses = [EFLAGS]
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
def TEST64rr : RI<0x85, MRMSrcReg, (outs), (ins GR64:$src1, GR64:$src2),
                  "test{q}\t{$src2, $src1|$src1, $src2}",
                  [(set EFLAGS, (X86cmp (and GR64:$src1, GR64:$src2), 0))]>;
}

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
def TEST64rm : RI<0x85, MRMSrcMem, (outs), (ins GR64:$src1, i64mem:$src2),
                  "test{q}\t{$src2, $src1|$src1, $src2}",
                  [(set EFLAGS, (X86cmp (and GR64:$src1, (loadi64 addr:$src2)),
                    0))]>;

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
def TEST64ri32 : RIi32<0xF7, MRM0r, (outs),
                                        (ins GR64:$src1, i64i32imm:$src2),
                       "test{q}\t{$src2, $src1|$src1, $src2}",
                     [(set EFLAGS, (X86cmp (and GR64:$src1, i64immSExt32:$src2),
                      0))]>;

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
def TEST64mi32 : RIi32<0xF7, MRM0m, (outs),
                                        (ins i64mem:$src1, i64i32imm:$src2),
                       "test{q}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp (and (loadi64 addr:$src1),
                                           i64immSExt32:$src2), 0))]>;

def TEST8i8  : Ii8<0xA8, RawFrm, (outs), (ins i8imm:$src),
                   "test{b}\t{$src, %al|%al, $src}", []>;
def TEST16i16 : Ii16<0xA9, RawFrm, (outs), (ins i16imm:$src),
                     "test{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def TEST32i32 : Ii32<0xA9, RawFrm, (outs), (ins i32imm:$src),
                     "test{l}\t{$src, %eax|%eax, $src}", []>;
def TEST64i32 : RIi32<0xa9, RawFrm, (outs), (ins i64i32imm:$src),
                      "test{q}\t{$src, %rax|%rax, $src}", []>;

} // Defs = [EFLAGS]


//===----------------------------------------------------------------------===//
// Integer comparisons

let Defs = [EFLAGS] in {

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
def CMP64rr : RI<0x39, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
                 "cmp{q}\t{$src2, $src1|$src1, $src2}",
                 [(set EFLAGS, (X86cmp GR64:$src1, GR64:$src2))]>;

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
def CMP64mr : RI<0x39, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
                 "cmp{q}\t{$src2, $src1|$src1, $src2}",
                 [(set EFLAGS, (X86cmp (loadi64 addr:$src1), GR64:$src2))]>;
                 
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
def CMP64rm : RI<0x3B, MRMSrcMem, (outs), (ins GR64:$src1, i64mem:$src2),
                 "cmp{q}\t{$src2, $src1|$src1, $src2}",
                 [(set EFLAGS, (X86cmp GR64:$src1, (loadi64 addr:$src2)))]>;

// These are alternate spellings for use by the disassembler, we mark them as
// code gen only to ensure they aren't matched by the assembler.
let isCodeGenOnly = 1 in {
  def CMP8rr_alt : I<0x3A, MRMSrcReg, (outs), (ins GR8:$src1, GR8:$src2),
                    "cmp{b}\t{$src2, $src1|$src1, $src2}", []>;
  def CMP16rr_alt : I<0x3B, MRMSrcReg, (outs), (ins GR16:$src1, GR16:$src2),
                     "cmp{w}\t{$src2, $src1|$src1, $src2}", []>, OpSize;
  def CMP32rr_alt : I<0x3B, MRMSrcReg, (outs), (ins GR32:$src1, GR32:$src2),
                     "cmp{l}\t{$src2, $src1|$src1, $src2}", []>;
  def CMP64rr_alt : RI<0x3B, MRMSrcReg, (outs), (ins GR64:$src1, GR64:$src2),
                      "cmp{q}\t{$src2, $src1|$src1, $src2}", []>;
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
def CMP64ri32 : RIi32<0x81, MRM7r, (outs), (ins GR64:$src1, i64i32imm:$src2),
                      "cmp{q}\t{$src2, $src1|$src1, $src2}",
                      [(set EFLAGS, (X86cmp GR64:$src1, i64immSExt32:$src2))]>;

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
def CMP64mi32 : RIi32<0x81, MRM7m, (outs),
                                       (ins i64mem:$src1, i64i32imm:$src2),
                      "cmp{q}\t{$src2, $src1|$src1, $src2}",
                      [(set EFLAGS, (X86cmp (loadi64 addr:$src1),
                                            i64immSExt32:$src2))]>;

def CMP16ri8 : Ii8<0x83, MRM7r,
                   (outs), (ins GR16:$src1, i16i8imm:$src2),
                   "cmp{w}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp GR16:$src1, i16immSExt8:$src2))]>,
                    OpSize;
def CMP32ri8 : Ii8<0x83, MRM7r,
                   (outs), (ins GR32:$src1, i32i8imm:$src2),
                   "cmp{l}\t{$src2, $src1|$src1, $src2}",
                   [(set EFLAGS, (X86cmp GR32:$src1, i32immSExt8:$src2))]>;
def CMP64ri8 : RIi8<0x83, MRM7r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                    "cmp{q}\t{$src2, $src1|$src1, $src2}",
                    [(set EFLAGS, (X86cmp GR64:$src1, i64immSExt8:$src2))]>;

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
def CMP64mi8 : RIi8<0x83, MRM7m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                    "cmp{q}\t{$src2, $src1|$src1, $src2}",
                    [(set EFLAGS, (X86cmp (loadi64 addr:$src1),
                                          i64immSExt8:$src2))]>;

def CMP8i8 : Ii8<0x3C, RawFrm, (outs), (ins i8imm:$src),
                 "cmp{b}\t{$src, %al|%al, $src}", []>;
def CMP16i16 : Ii16<0x3D, RawFrm, (outs), (ins i16imm:$src),
                    "cmp{w}\t{$src, %ax|%ax, $src}", []>, OpSize;
def CMP32i32 : Ii32<0x3D, RawFrm, (outs), (ins i32imm:$src),
                    "cmp{l}\t{$src, %eax|%eax, $src}", []>;
def CMP64i32 : RIi32<0x3D, RawFrm, (outs), (ins i64i32imm:$src),
                     "cmp{q}\t{$src, %rax|%rax, $src}", []>;

} // Defs = [EFLAGS]
