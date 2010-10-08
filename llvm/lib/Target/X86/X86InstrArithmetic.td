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
                  PatFrag loadnode, X86MemOperand memoperand, ImmType immkind,
                  Operand immoperand, SDPatternOperator immoperator,
                  Operand imm8operand, SDPatternOperator imm8operator,
                  bit hasOddOpcode, bit hasOpSizePrefix, bit hasREX_WPrefix> {
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
  
  /// ImmEncoding - This is the encoding of an immediate of this type.  For
  /// example, i8 -> Imm8, i16 -> Imm16, i32 -> Imm32.  Note that i64 -> Imm32
  /// since the immediate fields of i64 instructions is a 32-bit sign extended
  /// value.
  ImmType ImmEncoding = immkind;
  
  /// ImmOperand - This is the operand kind of an immediate of this type.  For
  /// example, i8 -> i8imm, i16 -> i16imm, i32 -> i32imm.  Note that i64 ->
  /// i64i32imm since the immediate fields of i64 instructions is a 32-bit sign
  /// extended value.
  Operand ImmOperand = immoperand;
  
  /// ImmOperator - This is the operator that should be used to match an
  /// immediate of this kind in a pattern (e.g. imm, or i64immSExt32).
  SDPatternOperator ImmOperator = immoperator;
  
  /// Imm8Operand - This is the operand kind to use for an imm8 of this type.
  /// For example, i8 -> <invalid>, i16 -> i16i8imm, i32 -> i32i8imm.  This is
  /// only used for instructions that have a sign-extended imm8 field form.
  Operand Imm8Operand = imm8operand;
  
  /// Imm8Operator - This is the operator that should be used to match an 8-bit
  /// sign extended immediate of this kind in a pattern (e.g. imm16immSExt8).
  SDPatternOperator Imm8Operator = imm8operator;
  
  /// HasOddOpcode - This bit is true if the instruction should have an odd (as
  /// opposed to even) opcode.  Operations on i8 are usually even, operations on
  /// other datatypes are odd.
  bit HasOddOpcode = hasOddOpcode;
  
  /// HasOpSizePrefix - This bit is set to true if the instruction should have
  /// the 0x66 operand size prefix.  This is set for i16 types.
  bit HasOpSizePrefix = hasOpSizePrefix;
  
  /// HasREX_WPrefix - This bit is set to true if the instruction should have
  /// the 0x40 REX prefix.  This is set for i64 types.
  bit HasREX_WPrefix = hasREX_WPrefix;
}

def invalid_node : SDNode<"<<invalid_node>>", SDTIntLeaf,[],"<<invalid_node>>">;


def Xi8  : X86TypeInfo<i8 , "b", GR8 , loadi8 , i8mem ,
                       Imm8 , i8imm ,    imm,          i8imm   , invalid_node,
                       0, 0, 0>;
def Xi16 : X86TypeInfo<i16, "w", GR16, loadi16, i16mem,
                       Imm16, i16imm,    imm,          i16i8imm, i16immSExt8,
                       1, 1, 0>;
def Xi32 : X86TypeInfo<i32, "l", GR32, loadi32, i32mem,
                       Imm32, i32imm,    imm,          i32i8imm, i32immSExt8,
                       1, 0, 0>;
def Xi64 : X86TypeInfo<i64, "q", GR64, loadi64, i64mem,
                       Imm32, i64i32imm, i64immSExt32, i64i8imm, i64immSExt8,
                       1, 0, 1>;

/// ITy - This instruction base class takes the type info for the instruction.
/// Using this, it:
/// 1. Concatenates together the instruction mnemonic with the appropriate
///    suffix letter, a tab, and the arguments.
/// 2. Infers whether the instruction should have a 0x66 prefix byte.
/// 3. Infers whether the instruction should have a 0x40 REX_W prefix.
/// 4. Infers whether the low bit of the opcode should be 0 (for i8 operations)
///    or 1 (for i16,i32,i64 operations).
class ITy<bits<8> opcode, Format f, X86TypeInfo typeinfo, dag outs, dag ins, 
          string mnemonic, string args, list<dag> pattern>
  : I<{opcode{7}, opcode{6}, opcode{5}, opcode{4},
       opcode{3}, opcode{2}, opcode{1}, typeinfo.HasOddOpcode },
      f, outs, ins, 
      !strconcat(mnemonic, "{", typeinfo.InstrSuffix, "}\t", args), pattern> {

  // Infer instruction prefixes from type info.
  let hasOpSizePrefix = typeinfo.HasOpSizePrefix;
  let hasREX_WPrefix  = typeinfo.HasREX_WPrefix;
}

// BinOpRR - Instructions like "add reg, reg, reg".
class BinOpRR<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
              dag outlist, list<dag> pattern, Format f = MRMDestReg>
  : ITy<opcode, f, typeinfo, outlist,
        (ins typeinfo.RegClass:$src1, typeinfo.RegClass:$src2),
        mnemonic, "{$src2, $src1|$src1, $src2}", pattern>;

// BinOpRR_R - Instructions like "add reg, reg, reg", where the pattern has
// just a regclass (no eflags) as a result.
class BinOpRR_R<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                SDNode opnode>
  : BinOpRR<opcode, mnemonic, typeinfo, (outs typeinfo.RegClass:$dst),
            [(set typeinfo.RegClass:$dst,
                  (opnode typeinfo.RegClass:$src1, typeinfo.RegClass:$src2))]>;

// BinOpRR_F - Instructions like "cmp reg, Reg", where the pattern has
// just a EFLAGS as a result.
class BinOpRR_F<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                SDPatternOperator opnode, Format f = MRMDestReg>
  : BinOpRR<opcode, mnemonic, typeinfo, (outs),
            [(set EFLAGS,
                  (opnode typeinfo.RegClass:$src1, typeinfo.RegClass:$src2))],
            f>;

// BinOpRR_RF - Instructions like "add reg, reg, reg", where the pattern has
// both a regclass and EFLAGS as a result.
class BinOpRR_RF<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                 SDNode opnode>
  : BinOpRR<opcode, mnemonic, typeinfo, (outs typeinfo.RegClass:$dst),
            [(set typeinfo.RegClass:$dst, EFLAGS,
                  (opnode typeinfo.RegClass:$src1, typeinfo.RegClass:$src2))]>;

// BinOpRR_Rev - Instructions like "add reg, reg, reg" (reversed encoding).
class BinOpRR_Rev<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo>
  : ITy<opcode, MRMSrcReg, typeinfo,
        (outs typeinfo.RegClass:$dst),
        (ins typeinfo.RegClass:$src1, typeinfo.RegClass:$src2),
        mnemonic, "{$src2, $dst|$dst, $src2}", []> {
  // The disassembler should know about this, but not the asmparser.
  let isCodeGenOnly = 1;
}

// BinOpRM - Instructions like "add reg, reg, [mem]".
class BinOpRM<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
              dag outlist, list<dag> pattern>
  : ITy<opcode, MRMSrcMem, typeinfo, outlist,
        (ins typeinfo.RegClass:$src1, typeinfo.MemOperand:$src2),
        mnemonic, "{$src2, $src1|$src1, $src2}", pattern>;

// BinOpRM_R - Instructions like "add reg, reg, [mem]".
class BinOpRM_R<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
              SDNode opnode>
  : BinOpRM<opcode, mnemonic, typeinfo, (outs typeinfo.RegClass:$dst),
            [(set typeinfo.RegClass:$dst,
            (opnode typeinfo.RegClass:$src1, (typeinfo.LoadNode addr:$src2)))]>;

// BinOpRM_F - Instructions like "cmp reg, [mem]".
class BinOpRM_F<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
              SDPatternOperator opnode>
  : BinOpRM<opcode, mnemonic, typeinfo, (outs),
            [(set EFLAGS,
            (opnode typeinfo.RegClass:$src1, (typeinfo.LoadNode addr:$src2)))]>;

// BinOpRM_RF - Instructions like "add reg, reg, [mem]".
class BinOpRM_RF<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                 SDNode opnode>
  : BinOpRM<opcode, mnemonic, typeinfo, (outs typeinfo.RegClass:$dst),
            [(set typeinfo.RegClass:$dst, EFLAGS,
            (opnode typeinfo.RegClass:$src1, (typeinfo.LoadNode addr:$src2)))]>;

// BinOpRI - Instructions like "add reg, reg, imm".
class BinOpRI<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
              Format f, dag outlist, list<dag> pattern>
  : ITy<opcode, f, typeinfo, outlist,
        (ins typeinfo.RegClass:$src1, typeinfo.ImmOperand:$src2),
        mnemonic, "{$src2, $src1|$src1, $src2}", pattern> {
  let ImmT = typeinfo.ImmEncoding;
}

// BinOpRI_R - Instructions like "add reg, reg, imm".
class BinOpRI_R<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                SDNode opnode, Format f>
  : BinOpRI<opcode, mnemonic, typeinfo, f, (outs typeinfo.RegClass:$dst),
            [(set typeinfo.RegClass:$dst,
                (opnode typeinfo.RegClass:$src1, typeinfo.ImmOperator:$src2))]>;

// BinOpRI_F - Instructions like "cmp reg, imm".
class BinOpRI_F<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                SDPatternOperator opnode, Format f>
  : BinOpRI<opcode, mnemonic, typeinfo, f, (outs),
            [(set EFLAGS,
                (opnode typeinfo.RegClass:$src1, typeinfo.ImmOperator:$src2))]>;

// BinOpRI_RF - Instructions like "add reg, reg, imm".
class BinOpRI_RF<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                 SDNode opnode, Format f>
  : BinOpRI<opcode, mnemonic, typeinfo, f, (outs typeinfo.RegClass:$dst),
            [(set typeinfo.RegClass:$dst, EFLAGS, 
                (opnode typeinfo.RegClass:$src1, typeinfo.ImmOperator:$src2))]>;

// BinOpRI8 - Instructions like "add reg, reg, imm8".
class BinOpRI8<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
               Format f, dag outlist, list<dag> pattern>
  : ITy<opcode, f, typeinfo, outlist,
        (ins typeinfo.RegClass:$src1, typeinfo.Imm8Operand:$src2),
        mnemonic, "{$src2, $src1|$src1, $src2}", pattern> {
  let ImmT = Imm8; // Always 8-bit immediate.
}

// BinOpRI8_R - Instructions like "add reg, reg, imm8".
class BinOpRI8_R<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                  SDNode opnode, Format f>
  : BinOpRI8<opcode, mnemonic, typeinfo, f, (outs typeinfo.RegClass:$dst),
             [(set typeinfo.RegClass:$dst,
               (opnode typeinfo.RegClass:$src1, typeinfo.Imm8Operator:$src2))]>;
               
// BinOpRI8_F - Instructions like "cmp reg, imm8".
class BinOpRI8_F<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                  SDNode opnode, Format f>
  : BinOpRI8<opcode, mnemonic, typeinfo, f, (outs),
             [(set EFLAGS,
               (opnode typeinfo.RegClass:$src1, typeinfo.Imm8Operator:$src2))]>;

// BinOpRI8_RF - Instructions like "add reg, reg, imm8".
class BinOpRI8_RF<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                  SDNode opnode, Format f>
  : BinOpRI8<opcode, mnemonic, typeinfo, f, (outs typeinfo.RegClass:$dst),
             [(set typeinfo.RegClass:$dst, EFLAGS,
               (opnode typeinfo.RegClass:$src1, typeinfo.Imm8Operator:$src2))]>;

// BinOpMR - Instructions like "add [mem], reg".
class BinOpMR<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
              list<dag> pattern>
  : ITy<opcode, MRMDestMem, typeinfo,
        (outs), (ins typeinfo.MemOperand:$dst, typeinfo.RegClass:$src),
        mnemonic, "{$src, $dst|$dst, $src}", pattern>;

// BinOpMR_RMW - Instructions like "add [mem], reg".
class BinOpMR_RMW<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                  SDNode opnode>
  : BinOpMR<opcode, mnemonic, typeinfo,
          [(store (opnode (load addr:$dst), typeinfo.RegClass:$src), addr:$dst),
           (implicit EFLAGS)]>;

// BinOpMR_F - Instructions like "cmp [mem], reg".
class BinOpMR_F<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
                  SDNode opnode>
  : BinOpMR<opcode, mnemonic, typeinfo,
            [(set EFLAGS, (opnode (load addr:$dst), typeinfo.RegClass:$src))]>;

// BinOpMI - Instructions like "add [mem], imm".
class BinOpMI<string mnemonic, X86TypeInfo typeinfo,
              Format f, list<dag> pattern, bits<8> opcode = 0x80>
  : ITy<opcode, f, typeinfo,
        (outs), (ins typeinfo.MemOperand:$dst, typeinfo.ImmOperand:$src),
        mnemonic, "{$src, $dst|$dst, $src}", pattern> {
  let ImmT = typeinfo.ImmEncoding;
}

// BinOpMI_RMW - Instructions like "add [mem], imm".
class BinOpMI_RMW<string mnemonic, X86TypeInfo typeinfo,
                  SDNode opnode, Format f>
  : BinOpMI<mnemonic, typeinfo, f, 
            [(store (opnode (typeinfo.VT (load addr:$dst)),
                            typeinfo.ImmOperator:$src), addr:$dst),
             (implicit EFLAGS)]>;

// BinOpMI_F - Instructions like "cmp [mem], imm".
class BinOpMI_F<string mnemonic, X86TypeInfo typeinfo,
                SDPatternOperator opnode, Format f, bits<8> opcode = 0x80>
  : BinOpMI<mnemonic, typeinfo, f, 
            [(set EFLAGS, (opnode (typeinfo.VT (load addr:$dst)),
                                               typeinfo.ImmOperator:$src))],
            opcode>;

// BinOpMI8 - Instructions like "add [mem], imm8".
class BinOpMI8<string mnemonic, X86TypeInfo typeinfo,
               Format f, list<dag> pattern>
  : ITy<0x82, f, typeinfo,
        (outs), (ins typeinfo.MemOperand:$dst, typeinfo.Imm8Operand:$src),
        mnemonic, "{$src, $dst|$dst, $src}", pattern> {
  let ImmT = Imm8; // Always 8-bit immediate.
}

// BinOpMI8_RMW - Instructions like "add [mem], imm8".
class BinOpMI8_RMW<string mnemonic, X86TypeInfo typeinfo,
                   SDNode opnode, Format f>
  : BinOpMI8<mnemonic, typeinfo, f,
             [(store (opnode (load addr:$dst),
                             typeinfo.Imm8Operator:$src), addr:$dst),
              (implicit EFLAGS)]>;

// BinOpMI8_F - Instructions like "cmp [mem], imm8".
class BinOpMI8_F<string mnemonic, X86TypeInfo typeinfo,
                 SDNode opnode, Format f>
  : BinOpMI8<mnemonic, typeinfo, f,
             [(set EFLAGS, (opnode (load addr:$dst),
                                   typeinfo.Imm8Operator:$src))]>;

// BinOpAI - Instructions like "add %eax, %eax, imm".
class BinOpAI<bits<8> opcode, string mnemonic, X86TypeInfo typeinfo,
              Register areg>
  : ITy<opcode, RawFrm, typeinfo,
        (outs), (ins typeinfo.ImmOperand:$src),
        mnemonic, !strconcat("{$src, %", areg.AsmName, "|%",
                               areg.AsmName, ", $src}"), []> {
  let ImmT = typeinfo.ImmEncoding;
  let Uses = [areg];
  let Defs = [areg];
}

/// ArithBinOp_RF - This is an arithmetic binary operator where the pattern is
/// defined with "(set GPR:$dst, EFLAGS, (...".
///
/// It would be nice to get rid of the second and third argument here, but
/// tblgen can't handle dependent type references aggressively enough: PR8330
multiclass ArithBinOp_RF<bits<8> BaseOpc, bits<8> BaseOpc2, bits<8> BaseOpc4,
                         string mnemonic, Format RegMRM, Format MemMRM,
                         SDNode opnodeflag, SDNode opnode,
                         bit CommutableRR, bit ConvertibleToThreeAddress> {
  let Defs = [EFLAGS] in {
    let Constraints = "$src1 = $dst" in {
      let isCommutable = CommutableRR,
          isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
        def #NAME#8rr  : BinOpRR_RF<BaseOpc, mnemonic, Xi8 , opnodeflag>;
        def #NAME#16rr : BinOpRR_RF<BaseOpc, mnemonic, Xi16, opnodeflag>;
        def #NAME#32rr : BinOpRR_RF<BaseOpc, mnemonic, Xi32, opnodeflag>;
        def #NAME#64rr : BinOpRR_RF<BaseOpc, mnemonic, Xi64, opnodeflag>;
      } // isCommutable

      def #NAME#8rr_REV  : BinOpRR_Rev<BaseOpc2, mnemonic, Xi8>;
      def #NAME#16rr_REV : BinOpRR_Rev<BaseOpc2, mnemonic, Xi16>;
      def #NAME#32rr_REV : BinOpRR_Rev<BaseOpc2, mnemonic, Xi32>;
      def #NAME#64rr_REV : BinOpRR_Rev<BaseOpc2, mnemonic, Xi64>;

      def #NAME#8rm   : BinOpRM_RF<BaseOpc2, mnemonic, Xi8 , opnodeflag>;
      def #NAME#16rm  : BinOpRM_RF<BaseOpc2, mnemonic, Xi16, opnodeflag>;
      def #NAME#32rm  : BinOpRM_RF<BaseOpc2, mnemonic, Xi32, opnodeflag>;
      def #NAME#64rm  : BinOpRM_RF<BaseOpc2, mnemonic, Xi64, opnodeflag>;

      let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
        // NOTE: These are order specific, we want the ri8 forms to be listed
        // first so that they are slightly preferred to the ri forms.
        def #NAME#16ri8 : BinOpRI8_RF<0x82, mnemonic, Xi16, opnodeflag, RegMRM>;
        def #NAME#32ri8 : BinOpRI8_RF<0x82, mnemonic, Xi32, opnodeflag, RegMRM>;
        def #NAME#64ri8 : BinOpRI8_RF<0x82, mnemonic, Xi64, opnodeflag, RegMRM>;

        def #NAME#8ri   : BinOpRI_RF<0x80, mnemonic, Xi8 , opnodeflag, RegMRM>;
        def #NAME#16ri  : BinOpRI_RF<0x80, mnemonic, Xi16, opnodeflag, RegMRM>;
        def #NAME#32ri  : BinOpRI_RF<0x80, mnemonic, Xi32, opnodeflag, RegMRM>;
        def #NAME#64ri32: BinOpRI_RF<0x80, mnemonic, Xi64, opnodeflag, RegMRM>;
      }
    } // Constraints = "$src1 = $dst"

    def #NAME#8mr    : BinOpMR_RMW<BaseOpc, mnemonic, Xi8 , opnode>;
    def #NAME#16mr   : BinOpMR_RMW<BaseOpc, mnemonic, Xi16, opnode>;
    def #NAME#32mr   : BinOpMR_RMW<BaseOpc, mnemonic, Xi32, opnode>;
    def #NAME#64mr   : BinOpMR_RMW<BaseOpc, mnemonic, Xi64, opnode>;

    // NOTE: These are order specific, we want the mi8 forms to be listed
    // first so that they are slightly preferred to the mi forms.
    def #NAME#16mi8  : BinOpMI8_RMW<mnemonic, Xi16, opnode, MemMRM>;
    def #NAME#32mi8  : BinOpMI8_RMW<mnemonic, Xi32, opnode, MemMRM>;
    def #NAME#64mi8  : BinOpMI8_RMW<mnemonic, Xi64, opnode, MemMRM>;
                       
    def #NAME#8mi    : BinOpMI_RMW<mnemonic, Xi8 , opnode, MemMRM>;
    def #NAME#16mi   : BinOpMI_RMW<mnemonic, Xi16, opnode, MemMRM>;
    def #NAME#32mi   : BinOpMI_RMW<mnemonic, Xi32, opnode, MemMRM>;
    def #NAME#64mi32 : BinOpMI_RMW<mnemonic, Xi64, opnode, MemMRM>;

    def #NAME#8i8   : BinOpAI<BaseOpc4, mnemonic, Xi8 , AL>;
    def #NAME#16i16 : BinOpAI<BaseOpc4, mnemonic, Xi16, AX>;
    def #NAME#32i32 : BinOpAI<BaseOpc4, mnemonic, Xi32, EAX>;
    def #NAME#64i32 : BinOpAI<BaseOpc4, mnemonic, Xi64, RAX>;
  }                          
}

/// ArithBinOp_R - This is an arithmetic binary operator where the pattern is
/// defined with "(set GPR:$dst, (...".  It would be really nice to find a way
/// to factor this with the other ArithBinOp_*.
///
multiclass ArithBinOp_R<bits<8> BaseOpc, bits<8> BaseOpc2, bits<8> BaseOpc4,
                        string mnemonic, Format RegMRM, Format MemMRM,
                        SDNode opnode,
                        bit CommutableRR, bit ConvertibleToThreeAddress> {
  let Defs = [EFLAGS] in {
    let Constraints = "$src1 = $dst" in {
      let isCommutable = CommutableRR,
          isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
        def #NAME#8rr  : BinOpRR_R<BaseOpc, mnemonic, Xi8 , opnode>;
        def #NAME#16rr : BinOpRR_R<BaseOpc, mnemonic, Xi16, opnode>;
        def #NAME#32rr : BinOpRR_R<BaseOpc, mnemonic, Xi32, opnode>;
        def #NAME#64rr : BinOpRR_R<BaseOpc, mnemonic, Xi64, opnode>;
      } // isCommutable

      def #NAME#8rr_REV  : BinOpRR_Rev<BaseOpc2, mnemonic, Xi8>;
      def #NAME#16rr_REV : BinOpRR_Rev<BaseOpc2, mnemonic, Xi16>;
      def #NAME#32rr_REV : BinOpRR_Rev<BaseOpc2, mnemonic, Xi32>;
      def #NAME#64rr_REV : BinOpRR_Rev<BaseOpc2, mnemonic, Xi64>;

      def #NAME#8rm   : BinOpRM_R<BaseOpc2, mnemonic, Xi8 , opnode>;
      def #NAME#16rm  : BinOpRM_R<BaseOpc2, mnemonic, Xi16, opnode>;
      def #NAME#32rm  : BinOpRM_R<BaseOpc2, mnemonic, Xi32, opnode>;
      def #NAME#64rm  : BinOpRM_R<BaseOpc2, mnemonic, Xi64, opnode>;

      let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
        // NOTE: These are order specific, we want the ri8 forms to be listed
        // first so that they are slightly preferred to the ri forms.
        def #NAME#16ri8 : BinOpRI8_R<0x82, mnemonic, Xi16, opnode, RegMRM>;
        def #NAME#32ri8 : BinOpRI8_R<0x82, mnemonic, Xi32, opnode, RegMRM>;
        def #NAME#64ri8 : BinOpRI8_R<0x82, mnemonic, Xi64, opnode, RegMRM>;

        def #NAME#8ri   : BinOpRI_R<0x80, mnemonic, Xi8 , opnode, RegMRM>;
        def #NAME#16ri  : BinOpRI_R<0x80, mnemonic, Xi16, opnode, RegMRM>;
        def #NAME#32ri  : BinOpRI_R<0x80, mnemonic, Xi32, opnode, RegMRM>;
        def #NAME#64ri32: BinOpRI_R<0x80, mnemonic, Xi64, opnode, RegMRM>;
      }
    } // Constraints = "$src1 = $dst"

    def #NAME#8mr    : BinOpMR_RMW<BaseOpc, mnemonic, Xi8 , opnode>;
    def #NAME#16mr   : BinOpMR_RMW<BaseOpc, mnemonic, Xi16, opnode>;
    def #NAME#32mr   : BinOpMR_RMW<BaseOpc, mnemonic, Xi32, opnode>;
    def #NAME#64mr   : BinOpMR_RMW<BaseOpc, mnemonic, Xi64, opnode>;

    // NOTE: These are order specific, we want the mi8 forms to be listed
    // first so that they are slightly preferred to the mi forms.
    def #NAME#16mi8  : BinOpMI8_RMW<mnemonic, Xi16, opnode, MemMRM>;
    def #NAME#32mi8  : BinOpMI8_RMW<mnemonic, Xi32, opnode, MemMRM>;
    def #NAME#64mi8  : BinOpMI8_RMW<mnemonic, Xi64, opnode, MemMRM>;
                       
    def #NAME#8mi    : BinOpMI_RMW<mnemonic, Xi8 , opnode, MemMRM>;
    def #NAME#16mi   : BinOpMI_RMW<mnemonic, Xi16, opnode, MemMRM>;
    def #NAME#32mi   : BinOpMI_RMW<mnemonic, Xi32, opnode, MemMRM>;
    def #NAME#64mi32 : BinOpMI_RMW<mnemonic, Xi64, opnode, MemMRM>;

    def #NAME#8i8   : BinOpAI<BaseOpc4, mnemonic, Xi8 , AL>;
    def #NAME#16i16 : BinOpAI<BaseOpc4, mnemonic, Xi16, AX>;
    def #NAME#32i32 : BinOpAI<BaseOpc4, mnemonic, Xi32, EAX>;
    def #NAME#64i32 : BinOpAI<BaseOpc4, mnemonic, Xi64, RAX>;
  }                          
}

/// ArithBinOp_F - This is an arithmetic binary operator where the pattern is
/// defined with "(set EFLAGS, (...".  It would be really nice to find a way
/// to factor this with the other ArithBinOp_*.
///
multiclass ArithBinOp_F<bits<8> BaseOpc, bits<8> BaseOpc2, bits<8> BaseOpc4,
                        string mnemonic, Format RegMRM, Format MemMRM,
                        SDNode opnode,
                        bit CommutableRR, bit ConvertibleToThreeAddress> {
  let Defs = [EFLAGS] in {
    let isCommutable = CommutableRR,
        isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
      def #NAME#8rr  : BinOpRR_F<BaseOpc, mnemonic, Xi8 , opnode>;
      def #NAME#16rr : BinOpRR_F<BaseOpc, mnemonic, Xi16, opnode>;
      def #NAME#32rr : BinOpRR_F<BaseOpc, mnemonic, Xi32, opnode>;
      def #NAME#64rr : BinOpRR_F<BaseOpc, mnemonic, Xi64, opnode>;
    } // isCommutable

    def #NAME#8rr_REV  : BinOpRR_Rev<BaseOpc2, mnemonic, Xi8>;
    def #NAME#16rr_REV : BinOpRR_Rev<BaseOpc2, mnemonic, Xi16>;
    def #NAME#32rr_REV : BinOpRR_Rev<BaseOpc2, mnemonic, Xi32>;
    def #NAME#64rr_REV : BinOpRR_Rev<BaseOpc2, mnemonic, Xi64>;

    def #NAME#8rm   : BinOpRM_F<BaseOpc2, mnemonic, Xi8 , opnode>;
    def #NAME#16rm  : BinOpRM_F<BaseOpc2, mnemonic, Xi16, opnode>;
    def #NAME#32rm  : BinOpRM_F<BaseOpc2, mnemonic, Xi32, opnode>;
    def #NAME#64rm  : BinOpRM_F<BaseOpc2, mnemonic, Xi64, opnode>;

    let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
      // NOTE: These are order specific, we want the ri8 forms to be listed
      // first so that they are slightly preferred to the ri forms.
      def #NAME#16ri8 : BinOpRI8_F<0x82, mnemonic, Xi16, opnode, RegMRM>;
      def #NAME#32ri8 : BinOpRI8_F<0x82, mnemonic, Xi32, opnode, RegMRM>;
      def #NAME#64ri8 : BinOpRI8_F<0x82, mnemonic, Xi64, opnode, RegMRM>;
      
      def #NAME#8ri   : BinOpRI_F<0x80, mnemonic, Xi8 , opnode, RegMRM>;
      def #NAME#16ri  : BinOpRI_F<0x80, mnemonic, Xi16, opnode, RegMRM>;
      def #NAME#32ri  : BinOpRI_F<0x80, mnemonic, Xi32, opnode, RegMRM>;
      def #NAME#64ri32: BinOpRI_F<0x80, mnemonic, Xi64, opnode, RegMRM>;
    }

    def #NAME#8mr    : BinOpMR_F<BaseOpc, mnemonic, Xi8 , opnode>;
    def #NAME#16mr   : BinOpMR_F<BaseOpc, mnemonic, Xi16, opnode>;
    def #NAME#32mr   : BinOpMR_F<BaseOpc, mnemonic, Xi32, opnode>;
    def #NAME#64mr   : BinOpMR_F<BaseOpc, mnemonic, Xi64, opnode>;

    // NOTE: These are order specific, we want the mi8 forms to be listed
    // first so that they are slightly preferred to the mi forms.
    def #NAME#16mi8  : BinOpMI8_F<mnemonic, Xi16, opnode, MemMRM>;
    def #NAME#32mi8  : BinOpMI8_F<mnemonic, Xi32, opnode, MemMRM>;
    def #NAME#64mi8  : BinOpMI8_F<mnemonic, Xi64, opnode, MemMRM>;
                       
    def #NAME#8mi    : BinOpMI_F<mnemonic, Xi8 , opnode, MemMRM>;
    def #NAME#16mi   : BinOpMI_F<mnemonic, Xi16, opnode, MemMRM>;
    def #NAME#32mi   : BinOpMI_F<mnemonic, Xi32, opnode, MemMRM>;
    def #NAME#64mi32 : BinOpMI_F<mnemonic, Xi64, opnode, MemMRM>;

    def #NAME#8i8   : BinOpAI<BaseOpc4, mnemonic, Xi8 , AL>;
    def #NAME#16i16 : BinOpAI<BaseOpc4, mnemonic, Xi16, AX>;
    def #NAME#32i32 : BinOpAI<BaseOpc4, mnemonic, Xi32, EAX>;
    def #NAME#64i32 : BinOpAI<BaseOpc4, mnemonic, Xi64, RAX>;
  }                          
}


defm AND : ArithBinOp_RF<0x20, 0x22, 0x24, "and", MRM4r, MRM4m,
                         X86and_flag, and, 1, 0>;
defm OR  : ArithBinOp_RF<0x08, 0x0A, 0x0C, "or", MRM1r, MRM1m,
                         X86or_flag, or, 1, 0>;
defm XOR : ArithBinOp_RF<0x30, 0x32, 0x34, "xor", MRM6r, MRM6m,
                         X86xor_flag, xor, 1, 0>;
defm ADD : ArithBinOp_RF<0x00, 0x02, 0x04, "add", MRM0r, MRM0m,
                         X86add_flag, add, 1, 1>;
defm SUB : ArithBinOp_RF<0x28, 0x2A, 0x2C, "sub", MRM5r, MRM5m,
                         X86sub_flag, sub, 0, 0>;

// Arithmetic.
let Uses = [EFLAGS] in {
  // FIXME: Delete ArithBinOp_R if these switch off adde/sube.
  defm ADC : ArithBinOp_R<0x10, 0x12, 0x14, "adc", MRM2r, MRM2m, adde, 1, 0>;
  defm SBB : ArithBinOp_R<0x18, 0x1A, 0x1C, "sbb", MRM3r, MRM3m, sube, 0, 0>;
}



defm CMP : ArithBinOp_F<0x38, 0x3A, 0x3C, "cmp", MRM7r, MRM7m, X86cmp, 0, 0>;


//===----------------------------------------------------------------------===//
// Semantically, test instructions are similar like AND, except they don't
// generate a result.  From an encoding perspective, they are very different:
// they don't have all the usual imm8 and REV forms, and are encoded into a
// different space.
def X86testpat : PatFrag<(ops node:$lhs, node:$rhs),
                         (X86cmp (and_su node:$lhs, node:$rhs), 0)>;

let Defs = [EFLAGS] in {
  let isCommutable = 1 in {
    def TEST8rr  : BinOpRR_F<0x84, "test", Xi8 , X86testpat, MRMSrcReg>;
    def TEST16rr : BinOpRR_F<0x84, "test", Xi16, X86testpat, MRMSrcReg>;
    def TEST32rr : BinOpRR_F<0x84, "test", Xi32, X86testpat, MRMSrcReg>;
    def TEST64rr : BinOpRR_F<0x84, "test", Xi64, X86testpat, MRMSrcReg>;
  } // isCommutable

  def TEST8rm    : BinOpRM_F<0x84, "test", Xi8 , X86testpat>;
  def TEST16rm   : BinOpRM_F<0x84, "test", Xi16, X86testpat>;
  def TEST32rm   : BinOpRM_F<0x84, "test", Xi32, X86testpat>;
  def TEST64rm   : BinOpRM_F<0x84, "test", Xi64, X86testpat>;

  def TEST8ri    : BinOpRI_F<0xF6, "test", Xi8 , X86testpat, MRM0r>;
  def TEST16ri   : BinOpRI_F<0xF6, "test", Xi16, X86testpat, MRM0r>;
  def TEST32ri   : BinOpRI_F<0xF6, "test", Xi32, X86testpat, MRM0r>;
  def TEST64ri32 : BinOpRI_F<0xF6, "test", Xi64, X86testpat, MRM0r>;

  def TEST8mi    : BinOpMI_F<"test", Xi8 , X86testpat, MRM0m, 0xF6>;
  def TEST16mi   : BinOpMI_F<"test", Xi16, X86testpat, MRM0m, 0xF6>;
  def TEST32mi   : BinOpMI_F<"test", Xi32, X86testpat, MRM0m, 0xF6>;
  def TEST64mi32 : BinOpMI_F<"test", Xi64, X86testpat, MRM0m, 0xF6>;
                     
  def TEST8i8    : BinOpAI<0xA8, "test", Xi8 , AL>;
  def TEST16i16  : BinOpAI<0xA8, "test", Xi16, AX>;
  def TEST32i32  : BinOpAI<0xA8, "test", Xi32, EAX>;
  def TEST64i32  : BinOpAI<0xA8, "test", Xi64, RAX>;
}                          

