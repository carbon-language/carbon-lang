//===- X86InstrSystem.td - System Instructions -------------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 instructions that are generally used in
// privileged modes.  These are not typically used by the compiler, but are
// supported for the assembler and disassembler.
//
//===----------------------------------------------------------------------===//

let Defs = [RAX, RDX] in
  def RDTSC : I<0x31, RawFrm, (outs), (ins), "rdtsc", [(X86rdtsc)]>, TB;

let Defs = [RAX, RCX, RDX] in
  def RDTSCP : I<0x01, MRM_F9, (outs), (ins), "rdtscp", []>, TB;

// CPU flow control instructions

let isTerminator = 1, isBarrier = 1, hasCtrlDep = 1 in {
  def TRAP    : I<0x0B, RawFrm, (outs), (ins), "ud2", [(trap)]>, TB;
  def UD2B    : I<0xB9, RawFrm, (outs), (ins), "ud2b", []>, TB;
}

def HLT : I<0xF4, RawFrm, (outs), (ins), "hlt", []>;
def RSM : I<0xAA, RawFrm, (outs), (ins), "rsm", []>, TB;

// Interrupt and SysCall Instructions.
let Uses = [EFLAGS] in
  def INTO : I<0xce, RawFrm, (outs), (ins), "into", []>;
def INT3 : I<0xcc, RawFrm, (outs), (ins), "int3",
              [(int_x86_int (i8 3))]>;

// The long form of "int $3" turns into int3 as a size optimization.
// FIXME: This doesn't work because InstAlias can't match immediate constants.
//def : InstAlias<"int\t$3", (INT3)>;


def INT : Ii8<0xcd, RawFrm, (outs), (ins i8imm:$trap), "int\t$trap",
              [(int_x86_int imm:$trap)]>;


def SYSCALL  : I<0x05, RawFrm, (outs), (ins), "syscall", []>, TB;
def SYSRETL  : I<0x07, RawFrm, (outs), (ins), "sysretl", []>, TB;
def SYSRETQ  :RI<0x07, RawFrm, (outs), (ins), "sysretq", []>, TB,
               Requires<[In64BitMode]>;

def SYSENTER : I<0x34, RawFrm, (outs), (ins), "sysenter", []>, TB;
                 
def SYSEXIT   : I<0x35, RawFrm, (outs), (ins), "sysexit", []>, TB,
                Requires<[In32BitMode]>;
def SYSEXIT64 :RI<0x35, RawFrm, (outs), (ins), "sysexit", []>, TB,
                Requires<[In64BitMode]>;

def IRET16 : I<0xcf, RawFrm, (outs), (ins), "iretw", []>, OpSize;
def IRET32 : I<0xcf, RawFrm, (outs), (ins), "iret{l|d}", []>;
def IRET64 : RI<0xcf, RawFrm, (outs), (ins), "iretq", []>,
             Requires<[In64BitMode]>;


//===----------------------------------------------------------------------===//
//  Input/Output Instructions.
//
let Defs = [AL], Uses = [DX] in
def IN8rr  : I<0xEC, RawFrm, (outs), (ins),
               "in{b}\t{%dx, %al|AL, DX}", []>;
let Defs = [AX], Uses = [DX] in
def IN16rr : I<0xED, RawFrm, (outs), (ins),
               "in{w}\t{%dx, %ax|AX, DX}", []>,  OpSize;
let Defs = [EAX], Uses = [DX] in
def IN32rr : I<0xED, RawFrm, (outs), (ins),
               "in{l}\t{%dx, %eax|EAX, DX}", []>;

let Defs = [AL] in
def IN8ri  : Ii8<0xE4, RawFrm, (outs), (ins i8imm:$port),
                  "in{b}\t{$port, %al|AL, $port}", []>;
let Defs = [AX] in
def IN16ri : Ii8<0xE5, RawFrm, (outs), (ins i8imm:$port),
                  "in{w}\t{$port, %ax|AX, $port}", []>, OpSize;
let Defs = [EAX] in
def IN32ri : Ii8<0xE5, RawFrm, (outs), (ins i8imm:$port),
                  "in{l}\t{$port, %eax|EAX, $port}", []>;

let Uses = [DX, AL] in
def OUT8rr  : I<0xEE, RawFrm, (outs), (ins),
                "out{b}\t{%al, %dx|DX, AL}", []>;
let Uses = [DX, AX] in
def OUT16rr : I<0xEF, RawFrm, (outs), (ins),
                "out{w}\t{%ax, %dx|DX, AX}", []>, OpSize;
let Uses = [DX, EAX] in
def OUT32rr : I<0xEF, RawFrm, (outs), (ins),
                "out{l}\t{%eax, %dx|DX, EAX}", []>;

let Uses = [AL] in
def OUT8ir  : Ii8<0xE6, RawFrm, (outs), (ins i8imm:$port),
                   "out{b}\t{%al, $port|$port, AL}", []>;
let Uses = [AX] in
def OUT16ir : Ii8<0xE7, RawFrm, (outs), (ins i8imm:$port),
                   "out{w}\t{%ax, $port|$port, AX}", []>, OpSize;
let Uses = [EAX] in
def OUT32ir : Ii8<0xE7, RawFrm, (outs), (ins i8imm:$port),
                   "out{l}\t{%eax, $port|$port, EAX}", []>;

def IN8  : I<0x6C, RawFrm, (outs), (ins), "ins{b}", []>;
def IN16 : I<0x6D, RawFrm, (outs), (ins), "ins{w}", []>,  OpSize;
def IN32 : I<0x6D, RawFrm, (outs), (ins), "ins{l}", []>;

//===----------------------------------------------------------------------===//
// Moves to and from debug registers

def MOV32rd : I<0x21, MRMDestReg, (outs GR32:$dst), (ins DEBUG_REG:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>, TB;
def MOV64rd : I<0x21, MRMDestReg, (outs GR64:$dst), (ins DEBUG_REG:$src),
                "mov{q}\t{$src, $dst|$dst, $src}", []>, TB;
                
def MOV32dr : I<0x23, MRMSrcReg, (outs DEBUG_REG:$dst), (ins GR32:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>, TB;
def MOV64dr : I<0x23, MRMSrcReg, (outs DEBUG_REG:$dst), (ins GR64:$src),
                "mov{q}\t{$src, $dst|$dst, $src}", []>, TB;

//===----------------------------------------------------------------------===//
// Moves to and from control registers

def MOV32rc : I<0x20, MRMDestReg, (outs GR32:$dst), (ins CONTROL_REG:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>, TB;
def MOV64rc : I<0x20, MRMDestReg, (outs GR64:$dst), (ins CONTROL_REG:$src),
                "mov{q}\t{$src, $dst|$dst, $src}", []>, TB;
                
def MOV32cr : I<0x22, MRMSrcReg, (outs CONTROL_REG:$dst), (ins GR32:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>, TB;
def MOV64cr : I<0x22, MRMSrcReg, (outs CONTROL_REG:$dst), (ins GR64:$src),
                "mov{q}\t{$src, $dst|$dst, $src}", []>, TB;

//===----------------------------------------------------------------------===//
// Segment override instruction prefixes

def CS_PREFIX : I<0x2E, RawFrm, (outs), (ins), "cs", []>;
def SS_PREFIX : I<0x36, RawFrm, (outs), (ins), "ss", []>;
def DS_PREFIX : I<0x3E, RawFrm, (outs), (ins), "ds", []>;
def ES_PREFIX : I<0x26, RawFrm, (outs), (ins), "es", []>;
def FS_PREFIX : I<0x64, RawFrm, (outs), (ins), "fs", []>;
def GS_PREFIX : I<0x65, RawFrm, (outs), (ins), "gs", []>;


//===----------------------------------------------------------------------===//
// Moves to and from segment registers.
//

def MOV16rs : I<0x8C, MRMDestReg, (outs GR16:$dst), (ins SEGMENT_REG:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32rs : I<0x8C, MRMDestReg, (outs GR32:$dst), (ins SEGMENT_REG:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;
def MOV64rs : RI<0x8C, MRMDestReg, (outs GR64:$dst), (ins SEGMENT_REG:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}", []>;

def MOV16ms : I<0x8C, MRMDestMem, (outs i16mem:$dst), (ins SEGMENT_REG:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32ms : I<0x8C, MRMDestMem, (outs i32mem:$dst), (ins SEGMENT_REG:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;
def MOV64ms : RI<0x8C, MRMDestMem, (outs i64mem:$dst), (ins SEGMENT_REG:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}", []>;

def MOV16sr : I<0x8E, MRMSrcReg, (outs SEGMENT_REG:$dst), (ins GR16:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32sr : I<0x8E, MRMSrcReg, (outs SEGMENT_REG:$dst), (ins GR32:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;
def MOV64sr : RI<0x8E, MRMSrcReg, (outs SEGMENT_REG:$dst), (ins GR64:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}", []>;

def MOV16sm : I<0x8E, MRMSrcMem, (outs SEGMENT_REG:$dst), (ins i16mem:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32sm : I<0x8E, MRMSrcMem, (outs SEGMENT_REG:$dst), (ins i32mem:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;
def MOV64sm : RI<0x8E, MRMSrcMem, (outs SEGMENT_REG:$dst), (ins i64mem:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}", []>;

//===----------------------------------------------------------------------===//
// Segmentation support instructions.

def SWAPGS : I<0x01, MRM_F8, (outs), (ins), "swapgs", []>, TB;

def LAR16rm : I<0x02, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src), 
                "lar{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def LAR16rr : I<0x02, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                "lar{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;

// i16mem operand in LAR32rm and GR32 operand in LAR32rr is not a typo.
def LAR32rm : I<0x02, MRMSrcMem, (outs GR32:$dst), (ins i16mem:$src), 
                "lar{l}\t{$src, $dst|$dst, $src}", []>, TB;
def LAR32rr : I<0x02, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                "lar{l}\t{$src, $dst|$dst, $src}", []>, TB;
// i16mem operand in LAR64rm and GR32 operand in LAR32rr is not a typo.
def LAR64rm : RI<0x02, MRMSrcMem, (outs GR64:$dst), (ins i16mem:$src), 
                 "lar{q}\t{$src, $dst|$dst, $src}", []>, TB;
def LAR64rr : RI<0x02, MRMSrcReg, (outs GR64:$dst), (ins GR32:$src),
                 "lar{q}\t{$src, $dst|$dst, $src}", []>, TB;

def LSL16rm : I<0x03, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                "lsl{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize; 
def LSL16rr : I<0x03, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                "lsl{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def LSL32rm : I<0x03, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                "lsl{l}\t{$src, $dst|$dst, $src}", []>, TB; 
def LSL32rr : I<0x03, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                "lsl{l}\t{$src, $dst|$dst, $src}", []>, TB;
def LSL64rm : RI<0x03, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                 "lsl{q}\t{$src, $dst|$dst, $src}", []>, TB; 
def LSL64rr : RI<0x03, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                 "lsl{q}\t{$src, $dst|$dst, $src}", []>, TB;

def INVLPG : I<0x01, MRM7m, (outs), (ins i8mem:$addr), "invlpg\t$addr", []>, TB;

def STR16r : I<0x00, MRM1r, (outs GR16:$dst), (ins),
               "str{w}\t{$dst}", []>, TB, OpSize;
def STR32r : I<0x00, MRM1r, (outs GR32:$dst), (ins),
               "str{l}\t{$dst}", []>, TB;
def STR64r : RI<0x00, MRM1r, (outs GR64:$dst), (ins),
                "str{q}\t{$dst}", []>, TB;
def STRm   : I<0x00, MRM1m, (outs i16mem:$dst), (ins),
               "str{w}\t{$dst}", []>, TB;

def LTRr : I<0x00, MRM3r, (outs), (ins GR16:$src),
             "ltr{w}\t{$src}", []>, TB;
def LTRm : I<0x00, MRM3m, (outs), (ins i16mem:$src),
             "ltr{w}\t{$src}", []>, TB;
             
def PUSHCS16 : I<0x0E, RawFrm, (outs), (ins),
                 "push{w}\t{%cs|CS}", []>, Requires<[In32BitMode]>, OpSize;
def PUSHCS32 : I<0x0E, RawFrm, (outs), (ins),
                 "push{l}\t{%cs|CS}", []>, Requires<[In32BitMode]>;
def PUSHSS16 : I<0x16, RawFrm, (outs), (ins),
                 "push{w}\t{%ss|SS}", []>, Requires<[In32BitMode]>, OpSize;
def PUSHSS32 : I<0x16, RawFrm, (outs), (ins),
                 "push{l}\t{%ss|SS}", []>, Requires<[In32BitMode]>;
def PUSHDS16 : I<0x1E, RawFrm, (outs), (ins),
                 "push{w}\t{%ds|DS}", []>, Requires<[In32BitMode]>, OpSize;
def PUSHDS32 : I<0x1E, RawFrm, (outs), (ins),
                 "push{l}\t{%ds|DS}", []>, Requires<[In32BitMode]>;
def PUSHES16 : I<0x06, RawFrm, (outs), (ins),
                 "push{w}\t{%es|ES}", []>, Requires<[In32BitMode]>, OpSize;
def PUSHES32 : I<0x06, RawFrm, (outs), (ins),
                 "push{l}\t{%es|ES}", []>, Requires<[In32BitMode]>;
                 
def PUSHFS16 : I<0xa0, RawFrm, (outs), (ins),
                 "push{w}\t{%fs|FS}", []>, OpSize, TB;
def PUSHFS32 : I<0xa0, RawFrm, (outs), (ins),
                 "push{l}\t{%fs|FS}", []>, TB, Requires<[In32BitMode]>;
def PUSHGS16 : I<0xa8, RawFrm, (outs), (ins),
                 "push{w}\t{%gs|GS}", []>, OpSize, TB;
def PUSHGS32 : I<0xa8, RawFrm, (outs), (ins),
                 "push{l}\t{%gs|GS}", []>, TB, Requires<[In32BitMode]>;

def PUSHFS64 : I<0xa0, RawFrm, (outs), (ins),
                 "push{q}\t{%fs|FS}", []>, TB;
def PUSHGS64 : I<0xa8, RawFrm, (outs), (ins),
                 "push{q}\t{%gs|GS}", []>, TB;

// No "pop cs" instruction.
def POPSS16 : I<0x17, RawFrm, (outs), (ins),
                "pop{w}\t{%ss|SS}", []>, OpSize, Requires<[In32BitMode]>;
def POPSS32 : I<0x17, RawFrm, (outs), (ins),
                "pop{l}\t{%ss|SS}", []>        , Requires<[In32BitMode]>;
                
def POPDS16 : I<0x1F, RawFrm, (outs), (ins),
                "pop{w}\t{%ds|DS}", []>, OpSize, Requires<[In32BitMode]>;
def POPDS32 : I<0x1F, RawFrm, (outs), (ins),
                "pop{l}\t{%ds|DS}", []>        , Requires<[In32BitMode]>;
                
def POPES16 : I<0x07, RawFrm, (outs), (ins),
                "pop{w}\t{%es|ES}", []>, OpSize, Requires<[In32BitMode]>;
def POPES32 : I<0x07, RawFrm, (outs), (ins),
                "pop{l}\t{%es|ES}", []>        , Requires<[In32BitMode]>;
                
def POPFS16 : I<0xa1, RawFrm, (outs), (ins),
                "pop{w}\t{%fs|FS}", []>, OpSize, TB;
def POPFS32 : I<0xa1, RawFrm, (outs), (ins),
                "pop{l}\t{%fs|FS}", []>, TB    , Requires<[In32BitMode]>;
def POPFS64 : I<0xa1, RawFrm, (outs), (ins),
                "pop{q}\t{%fs|FS}", []>, TB;
                
def POPGS16 : I<0xa9, RawFrm, (outs), (ins),
                "pop{w}\t{%gs|GS}", []>, OpSize, TB;
def POPGS32 : I<0xa9, RawFrm, (outs), (ins),
                "pop{l}\t{%gs|GS}", []>, TB    , Requires<[In32BitMode]>;
def POPGS64 : I<0xa9, RawFrm, (outs), (ins),
                "pop{q}\t{%gs|GS}", []>, TB;
                 

def LDS16rm : I<0xc5, MRMSrcMem, (outs GR16:$dst), (ins opaque32mem:$src),
                "lds{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def LDS32rm : I<0xc5, MRMSrcMem, (outs GR32:$dst), (ins opaque48mem:$src),
                "lds{l}\t{$src, $dst|$dst, $src}", []>;
                
def LSS16rm : I<0xb2, MRMSrcMem, (outs GR16:$dst), (ins opaque32mem:$src),
                "lss{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def LSS32rm : I<0xb2, MRMSrcMem, (outs GR32:$dst), (ins opaque48mem:$src),
                "lss{l}\t{$src, $dst|$dst, $src}", []>, TB;
def LSS64rm : RI<0xb2, MRMSrcMem, (outs GR64:$dst), (ins opaque80mem:$src),
                 "lss{q}\t{$src, $dst|$dst, $src}", []>, TB;
                
def LES16rm : I<0xc4, MRMSrcMem, (outs GR16:$dst), (ins opaque32mem:$src),
                "les{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def LES32rm : I<0xc4, MRMSrcMem, (outs GR32:$dst), (ins opaque48mem:$src),
                "les{l}\t{$src, $dst|$dst, $src}", []>;
                
def LFS16rm : I<0xb4, MRMSrcMem, (outs GR16:$dst), (ins opaque32mem:$src),
                "lfs{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def LFS32rm : I<0xb4, MRMSrcMem, (outs GR32:$dst), (ins opaque48mem:$src),
                "lfs{l}\t{$src, $dst|$dst, $src}", []>, TB;
def LFS64rm : RI<0xb4, MRMSrcMem, (outs GR64:$dst), (ins opaque80mem:$src),
                 "lfs{q}\t{$src, $dst|$dst, $src}", []>, TB;
                
def LGS16rm : I<0xb5, MRMSrcMem, (outs GR16:$dst), (ins opaque32mem:$src),
                "lgs{w}\t{$src, $dst|$dst, $src}", []>, TB, OpSize;
def LGS32rm : I<0xb5, MRMSrcMem, (outs GR32:$dst), (ins opaque48mem:$src),
                "lgs{l}\t{$src, $dst|$dst, $src}", []>, TB;
                
def LGS64rm : RI<0xb5, MRMSrcMem, (outs GR64:$dst), (ins opaque80mem:$src),
                 "lgs{q}\t{$src, $dst|$dst, $src}", []>, TB;


def VERRr : I<0x00, MRM4r, (outs), (ins GR16:$seg),
              "verr\t$seg", []>, TB;
def VERRm : I<0x00, MRM4m, (outs), (ins i16mem:$seg),
              "verr\t$seg", []>, TB;
def VERWr : I<0x00, MRM5r, (outs), (ins GR16:$seg),
              "verw\t$seg", []>, TB;
def VERWm : I<0x00, MRM5m, (outs), (ins i16mem:$seg),
              "verw\t$seg", []>, TB;

//===----------------------------------------------------------------------===//
// Descriptor-table support instructions

def SGDT16m : I<0x01, MRM0m, (outs opaque48mem:$dst), (ins),
              "sgdtw\t$dst", []>, TB, OpSize, Requires<[In32BitMode]>;
def SGDTm : I<0x01, MRM0m, (outs opaque48mem:$dst), (ins),
              "sgdt\t$dst", []>, TB;
def SIDT16m : I<0x01, MRM1m, (outs opaque48mem:$dst), (ins),
              "sidtw\t$dst", []>, TB, OpSize, Requires<[In32BitMode]>;
def SIDTm : I<0x01, MRM1m, (outs opaque48mem:$dst), (ins),
              "sidt\t$dst", []>, TB;
def SLDT16r : I<0x00, MRM0r, (outs GR16:$dst), (ins),
                "sldt{w}\t$dst", []>, TB, OpSize;
def SLDT16m : I<0x00, MRM0m, (outs i16mem:$dst), (ins),
                "sldt{w}\t$dst", []>, TB;
def SLDT32r : I<0x00, MRM0r, (outs GR32:$dst), (ins),
                "sldt{l}\t$dst", []>, TB;
                
// LLDT is not interpreted specially in 64-bit mode because there is no sign
//   extension.
def SLDT64r : RI<0x00, MRM0r, (outs GR64:$dst), (ins),
                 "sldt{q}\t$dst", []>, TB;
def SLDT64m : RI<0x00, MRM0m, (outs i16mem:$dst), (ins),
                 "sldt{q}\t$dst", []>, TB;

def LGDT16m : I<0x01, MRM2m, (outs), (ins opaque48mem:$src),
              "lgdtw\t$src", []>, TB, OpSize, Requires<[In32BitMode]>;
def LGDTm : I<0x01, MRM2m, (outs), (ins opaque48mem:$src),
              "lgdt\t$src", []>, TB;
def LIDT16m : I<0x01, MRM3m, (outs), (ins opaque48mem:$src),
              "lidtw\t$src", []>, TB, OpSize, Requires<[In32BitMode]>;
def LIDTm : I<0x01, MRM3m, (outs), (ins opaque48mem:$src),
              "lidt\t$src", []>, TB;
def LLDT16r : I<0x00, MRM2r, (outs), (ins GR16:$src),
                "lldt{w}\t$src", []>, TB;
def LLDT16m : I<0x00, MRM2m, (outs), (ins i16mem:$src),
                "lldt{w}\t$src", []>, TB;
                
//===----------------------------------------------------------------------===//
// Specialized register support
def WRMSR : I<0x30, RawFrm, (outs), (ins), "wrmsr", []>, TB;
def RDMSR : I<0x32, RawFrm, (outs), (ins), "rdmsr", []>, TB;
def RDPMC : I<0x33, RawFrm, (outs), (ins), "rdpmc", []>, TB;

def SMSW16r : I<0x01, MRM4r, (outs GR16:$dst), (ins), 
                "smsw{w}\t$dst", []>, OpSize, TB;
def SMSW32r : I<0x01, MRM4r, (outs GR32:$dst), (ins), 
                "smsw{l}\t$dst", []>, TB;
// no m form encodable; use SMSW16m
def SMSW64r : RI<0x01, MRM4r, (outs GR64:$dst), (ins), 
                 "smsw{q}\t$dst", []>, TB;

// For memory operands, there is only a 16-bit form
def SMSW16m : I<0x01, MRM4m, (outs i16mem:$dst), (ins),
                "smsw{w}\t$dst", []>, TB;

def LMSW16r : I<0x01, MRM6r, (outs), (ins GR16:$src),
                "lmsw{w}\t$src", []>, TB;
def LMSW16m : I<0x01, MRM6m, (outs), (ins i16mem:$src),
                "lmsw{w}\t$src", []>, TB;
                
def CPUID : I<0xA2, RawFrm, (outs), (ins), "cpuid", []>, TB;

//===----------------------------------------------------------------------===//
// Cache instructions
def INVD : I<0x08, RawFrm, (outs), (ins), "invd", []>, TB;
def WBINVD : I<0x09, RawFrm, (outs), (ins), "wbinvd", []>, TB;

//===----------------------------------------------------------------------===//
// XSAVE instructions
let Defs = [RDX, RAX], Uses = [RCX] in
  def XGETBV : I<0x01, MRM_D0, (outs), (ins), "xgetbv", []>, TB;

let Uses = [RDX, RAX, RCX] in
  def XSETBV : I<0x01, MRM_D1, (outs), (ins), "xsetbv", []>, TB;

let Uses = [RDX, RAX] in {
  def XSAVE : I<0xAE, MRM4m, (outs opaque512mem:$dst), (ins),
               "xsave\t$dst", []>, TB;
  def XSAVE64 : I<0xAE, MRM4m, (outs opaque512mem:$dst), (ins),
                 "xsaveq\t$dst", []>, TB, REX_W, Requires<[In64BitMode]>;
  def XRSTOR : I<0xAE, MRM5m, (outs), (ins opaque512mem:$dst),
               "xrstor\t$dst", []>, TB;
  def XRSTOR64 : I<0xAE, MRM5m, (outs), (ins opaque512mem:$dst),
                 "xrstorq\t$dst", []>, TB, REX_W, Requires<[In64BitMode]>;
  def XSAVEOPT : I<0xAE, MRM6m, (outs opaque512mem:$dst), (ins),
                  "xsaveopt\t$dst", []>, TB;
  def XSAVEOPT64 : I<0xAE, MRM6m, (outs opaque512mem:$dst), (ins),
                    "xsaveoptq\t$dst", []>, TB, REX_W, Requires<[In64BitMode]>;
}

//===----------------------------------------------------------------------===//
// VIA PadLock crypto instructions
let Defs = [RAX, RDI], Uses = [RDX, RDI] in
  def XSTORE : I<0xc0, RawFrm, (outs), (ins), "xstore", []>, A7;

def : InstAlias<"xstorerng", (XSTORE)>;

let Defs = [RSI, RDI], Uses = [RBX, RDX, RSI, RDI] in {
  def XCRYPTECB : I<0xc8, RawFrm, (outs), (ins), "xcryptecb", []>, A7;
  def XCRYPTCBC : I<0xd0, RawFrm, (outs), (ins), "xcryptcbc", []>, A7;
  def XCRYPTCTR : I<0xd8, RawFrm, (outs), (ins), "xcryptctr", []>, A7;
  def XCRYPTCFB : I<0xe0, RawFrm, (outs), (ins), "xcryptcfb", []>, A7;
  def XCRYPTOFB : I<0xe8, RawFrm, (outs), (ins), "xcryptofb", []>, A7;
}

let Defs = [RAX, RSI, RDI], Uses = [RAX, RSI, RDI] in {
  def XSHA1 : I<0xc8, RawFrm, (outs), (ins), "xsha1", []>, A6;
  def XSHA256 : I<0xd0, RawFrm, (outs), (ins), "xsha256", []>, A6;
}
let Defs = [RAX, RDX, RSI], Uses = [RAX, RSI] in
  def MONTMUL : I<0xc0, RawFrm, (outs), (ins), "montmul", []>, A6;

//===----------------------------------------------------------------------===//
// FS/GS Base Instructions
let Predicates = [In64BitMode] in {
  def RDFSBASE : I<0xAE, MRM0r, (outs GR32:$dst), (ins),
                   "rdfsbase{l}\t$dst", []>, TB, XS;
  def RDFSBASE64 : RI<0xAE, MRM0r, (outs GR64:$dst), (ins),
                     "rdfsbase{q}\t$dst", []>, TB, XS;
  def RDGSBASE : I<0xAE, MRM1r, (outs GR32:$dst), (ins),
                   "rdgsbase{l}\t$dst", []>, TB, XS;
  def RDGSBASE64 : RI<0xAE, MRM1r, (outs GR64:$dst), (ins),
                     "rdgsbase{q}\t$dst", []>, TB, XS;
  def WRFSBASE : I<0xAE, MRM2r, (outs), (ins GR32:$dst),
                   "wrfsbase{l}\t$dst", []>, TB, XS;
  def WRFSBASE64 : RI<0xAE, MRM2r, (outs), (ins GR64:$dst),
                   "wrfsbase{q}\t$dst", []>, TB, XS;
  def WRGSBASE : I<0xAE, MRM3r, (outs), (ins GR32:$dst),
                   "wrgsbase{l}\t$dst", []>, TB, XS;
  def WRGSBASE64 : RI<0xAE, MRM3r, (outs), (ins GR64:$dst),
                    "wrgsbase{q}\t$dst", []>, TB, XS;
}

//===----------------------------------------------------------------------===//
// INVPCID Instruction
def INVPCID32 : I<0x82, MRMSrcMem, (outs), (ins GR32:$src1, i128mem:$src2),
                "invpcid {$src2, $src1|$src1, $src2}", []>, OpSize, T8,
                Requires<[In32BitMode]>;
def INVPCID64 : I<0x82, MRMSrcMem, (outs), (ins GR64:$src1, i128mem:$src2),
                "invpcid {$src2, $src1|$src1, $src2}", []>, OpSize, T8,
                Requires<[In64BitMode]>;
