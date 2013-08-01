//===- AArch64RegisterInfo.td - ARM Register defs ----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains declarations that describe the AArch64 register file
//
//===----------------------------------------------------------------------===//

let Namespace = "AArch64" in {
def sub_128 : SubRegIndex<128>;
def sub_64 : SubRegIndex<64>;
def sub_32 : SubRegIndex<32>;
def sub_16 : SubRegIndex<16>;
def sub_8  : SubRegIndex<8>;

// The VPR registers are handled as sub-registers of FPR equivalents, but
// they're really the same thing. We give this concept a special index.
def sub_alias : SubRegIndex<128>;
}

// Registers are identified with 5-bit ID numbers.
class AArch64Reg<bits<16> enc, string n> : Register<n> {
  let HWEncoding = enc;
  let Namespace = "AArch64";
}

class AArch64RegWithSubs<bits<16> enc, string n, list<Register> subregs = [],
                         list<SubRegIndex> inds = []>
      : AArch64Reg<enc, n> {
  let SubRegs = subregs;
  let SubRegIndices = inds;
}

//===----------------------------------------------------------------------===//
//  Integer registers: w0-w30, wzr, wsp, x0-x30, xzr, sp
//===----------------------------------------------------------------------===//

foreach Index = 0-30 in {
  def W#Index : AArch64Reg< Index, "w"#Index>, DwarfRegNum<[Index]>;
}

def WSP : AArch64Reg<31, "wsp">, DwarfRegNum<[31]>;
def WZR : AArch64Reg<31, "wzr">;

// Could be combined with previous loop, but this way leaves w and x registers
// consecutive as LLVM register numbers, which makes for easier debugging.
foreach Index = 0-30 in {
  def X#Index : AArch64RegWithSubs<Index, "x"#Index,
                                   [!cast<Register>("W"#Index)], [sub_32]>,
                DwarfRegNum<[Index]>;
}

def XSP : AArch64RegWithSubs<31, "sp", [WSP], [sub_32]>, DwarfRegNum<[31]>;
def XZR : AArch64RegWithSubs<31, "xzr", [WZR], [sub_32]>;

// Most instructions treat register 31 as zero for reads and a black-hole for
// writes.

// Note that the order of registers is important for the Disassembler here:
// tablegen uses it to form MCRegisterClass::getRegister, which we assume can
// take an encoding value.
def GPR32 : RegisterClass<"AArch64", [i32], 32,
                          (add (sequence "W%u", 0, 30), WZR)> {
}

def GPR64 : RegisterClass<"AArch64", [i64], 64,
                          (add (sequence "X%u", 0, 30), XZR)> {
}

def GPR32nowzr : RegisterClass<"AArch64", [i32], 32,
                               (sequence "W%u", 0, 30)> {
}

def GPR64noxzr : RegisterClass<"AArch64", [i64], 64,
                               (sequence "X%u", 0, 30)> {
}

// For tail calls, we can't use callee-saved registers or the structure-return
// register, as they are supposed to be live across function calls and may be
// clobbered by the epilogue.
def tcGPR64 : RegisterClass<"AArch64", [i64], 64,
                            (add (sequence "X%u", 0, 7),
                                 (sequence "X%u", 9, 18))> {
}


// Certain addressing-useful instructions accept sp directly. Again the order of
// registers is important to the Disassembler.
def GPR32wsp : RegisterClass<"AArch64", [i32], 32,
                             (add (sequence "W%u", 0, 30), WSP)> {
}

def GPR64xsp : RegisterClass<"AArch64", [i64], 64,
                             (add (sequence "X%u", 0, 30), XSP)> {
}

// Some aliases *only* apply to SP (e.g. MOV uses different encoding for SP and
// non-SP variants). We can't use a bare register in those patterns because
// TableGen doesn't like it, so we need a class containing just stack registers
def Rxsp : RegisterClass<"AArch64", [i64], 64,
                         (add XSP)> {
}

def Rwsp : RegisterClass<"AArch64", [i32], 32,
                         (add WSP)> {
}

//===----------------------------------------------------------------------===//
//  Scalar registers in the vector unit:
//  b0-b31, h0-h31, s0-s31, d0-d31, q0-q31
//===----------------------------------------------------------------------===//

foreach Index = 0-31 in {
  def B # Index : AArch64Reg< Index, "b" # Index>,
                  DwarfRegNum<[!add(Index, 64)]>;

  def H # Index : AArch64RegWithSubs<Index, "h" # Index,
                                     [!cast<Register>("B" # Index)], [sub_8]>,
                  DwarfRegNum<[!add(Index, 64)]>;

  def S # Index : AArch64RegWithSubs<Index, "s" # Index,
                                     [!cast<Register>("H" # Index)], [sub_16]>,
                  DwarfRegNum<[!add(Index, 64)]>;

  def D # Index : AArch64RegWithSubs<Index, "d" # Index,
                                     [!cast<Register>("S" # Index)], [sub_32]>,
                  DwarfRegNum<[!add(Index, 64)]>;

  def Q # Index : AArch64RegWithSubs<Index, "q" # Index,
                                     [!cast<Register>("D" # Index)], [sub_64]>,
                  DwarfRegNum<[!add(Index, 64)]>;
}


def FPR8 : RegisterClass<"AArch64", [i8], 8,
                          (sequence "B%u", 0, 31)> {
}

def FPR16 : RegisterClass<"AArch64", [f16], 16,
                          (sequence "H%u", 0, 31)> {
}

def FPR32 : RegisterClass<"AArch64", [f32], 32,
                          (sequence "S%u", 0, 31)> {
}

def FPR64 : RegisterClass<"AArch64", [f64], 64,
                          (sequence "D%u", 0, 31)> {
}

def FPR128 : RegisterClass<"AArch64", [f128], 128,
                          (sequence "Q%u", 0, 31)> {
}


//===----------------------------------------------------------------------===//
//  Vector registers:
//===----------------------------------------------------------------------===//

// NEON registers simply specify the overall vector, and it's expected that
// Instructions will individually specify the acceptable data layout. In
// principle this leaves two approaches open:
//   + An operand, giving a single ADDvvv instruction (for example). This turns
//     out to be unworkable in the assembly parser (without every Instruction
//     having a "cvt" function, at least) because the constraints can't be
//     properly enforced. It also complicates specifying patterns since each
//     instruction will accept many types.
//  + A bare token (e.g. ".2d"). This means the AsmParser has to know specific
//    details about NEON registers, but simplifies most other details.
//
// The second approach was taken.

foreach Index = 0-31 in {
  def V # Index  : AArch64RegWithSubs<Index, "v" # Index,
                                      [!cast<Register>("Q" # Index)],
                                      [sub_alias]>,
            DwarfRegNum<[!add(Index, 64)]>;
}

// These two classes contain the same registers, which should be reasonably
// sensible for MC and allocation purposes, but allows them to be treated
// separately for things like stack spilling.
def VPR64 : RegisterClass<"AArch64", [v2f32, v2i32, v4i16, v8i8, v1i64], 64,
                          (sequence "V%u", 0, 31)>;

def VPR128 : RegisterClass<"AArch64",
                           [v2f64, v2i64, v4f32, v4i32, v8i16, v16i8], 128,
                           (sequence "V%u", 0, 31)>;

// Flags register
def NZCV : Register<"nzcv"> {
  let Namespace = "AArch64";
}

def FlagClass : RegisterClass<"AArch64", [i32], 32, (add NZCV)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}
