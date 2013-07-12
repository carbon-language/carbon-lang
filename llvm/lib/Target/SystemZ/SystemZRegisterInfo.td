//==- SystemZRegisterInfo.td - SystemZ register definitions -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Class definitions.
//===----------------------------------------------------------------------===//

class SystemZReg<string n> : Register<n> {
  let Namespace = "SystemZ";
}

class SystemZRegWithSubregs<string n, list<Register> subregs>
  : RegisterWithSubRegs<n, subregs> {
  let Namespace = "SystemZ";
}

let Namespace = "SystemZ" in {
def subreg_32bit  : SubRegIndex<32>; // could also be named "subreg_high32"
// Indices are used in a variety of ways, so don't set an Offset.
def subreg_high   : SubRegIndex<64, -1>;
def subreg_low    : SubRegIndex<64, -1>;
def subreg_low32  : ComposedSubRegIndex<subreg_low, subreg_32bit>;
}

// Define a register class that contains values of type TYPE and an
// associated operand called NAME.  SIZE is the size and alignment
// of the registers and REGLIST is the list of individual registers.
multiclass SystemZRegClass<string name, ValueType type, int size, dag regList> {
  def AsmOperand : AsmOperandClass {
    let Name = name;
    let ParserMethod = "parse"##name;
    let RenderMethod = "addRegOperands";
  }
  def Bit : RegisterClass<"SystemZ", [type], size, regList> {
    let Size = size;
  }
  def "" : RegisterOperand<!cast<RegisterClass>(name##"Bit")> {
    let ParserMatchClass = !cast<AsmOperandClass>(name##"AsmOperand");
  }
}

//===----------------------------------------------------------------------===//
// General-purpose registers
//===----------------------------------------------------------------------===//

// Lower 32 bits of one of the 16 64-bit general-purpose registers
class GPR32<bits<16> num, string n> : SystemZReg<n> {
  let HWEncoding = num;
}

// One of the 16 64-bit general-purpose registers.
class GPR64<bits<16> num, string n, GPR32 low>
 : SystemZRegWithSubregs<n, [low]> {
  let HWEncoding = num;
  let SubRegIndices = [subreg_32bit];
}

// 8 even-odd pairs of GPR64s.
class GPR128<bits<16> num, string n, GPR64 high, GPR64 low>
 : SystemZRegWithSubregs<n, [high, low]> {
  let HWEncoding = num;
  let SubRegIndices = [subreg_high, subreg_low];
}

// General-purpose registers
foreach I = 0-15 in {
  def R#I#W : GPR32<I, "r"#I>;
  def R#I#D : GPR64<I, "r"#I, !cast<GPR32>("R"#I#"W")>, DwarfRegNum<[I]>;
}

foreach I = [0, 2, 4, 6, 8, 10, 12, 14] in {
  def R#I#Q : GPR128<I, "r"#I, !cast<GPR64>("R"#I#"D"),
                     !cast<GPR64>("R"#!add(I, 1)#"D")>;
}

/// Allocate the callee-saved R6-R13 backwards. That way they can be saved
/// together with R14 and R15 in one prolog instruction.
defm GR32 : SystemZRegClass<"GR32", i32, 32, (add (sequence "R%uW",  0, 5),
                                                  (sequence "R%uW", 15, 6))>;
defm GR64 : SystemZRegClass<"GR64", i64, 64, (add (sequence "R%uD",  0, 5),
                                                  (sequence "R%uD", 15, 6))>;

// The architecture doesn't really have any i128 support, so model the
// register pairs as untyped instead.
defm GR128 : SystemZRegClass<"GR128", untyped, 128, (add R0Q, R2Q, R4Q,
                                                         R12Q, R10Q, R8Q, R6Q,
                                                         R14Q)>;

// Base and index registers.  Everything except R0, which in an address
// context evaluates as 0.
defm ADDR32 : SystemZRegClass<"ADDR32", i32, 32, (sub GR32Bit, R0W)>;
defm ADDR64 : SystemZRegClass<"ADDR64", i64, 64, (sub GR64Bit, R0D)>;

// Not used directly, but needs to exist for ADDR32 and ADDR64 subregs
// of a GR128.
defm ADDR128 : SystemZRegClass<"ADDR128", untyped, 128, (sub GR128Bit, R0Q)>;

//===----------------------------------------------------------------------===//
// Floating-point registers
//===----------------------------------------------------------------------===//

// Lower 32 bits of one of the 16 64-bit floating-point registers
class FPR32<bits<16> num, string n> : SystemZReg<n> {
  let HWEncoding = num;
}

// One of the 16 64-bit floating-point registers
class FPR64<bits<16> num, string n, FPR32 low>
 : SystemZRegWithSubregs<n, [low]> {
  let HWEncoding = num;
  let SubRegIndices = [subreg_32bit];
}

// 8 pairs of FPR64s, with a one-register gap inbetween.
class FPR128<bits<16> num, string n, FPR64 high, FPR64 low>
 : SystemZRegWithSubregs<n, [high, low]> {
  let HWEncoding = num;
  let SubRegIndices = [subreg_high, subreg_low];
}

// Floating-point registers
foreach I = 0-15 in {
  def F#I#S : FPR32<I, "f"#I>;
  def F#I#D : FPR64<I, "f"#I, !cast<FPR32>("F"#I#"S")>,
              DwarfRegNum<[!add(I, 16)]>;
}

foreach I = [0, 1, 4, 5, 8, 9, 12, 13] in {
  def F#I#Q  : FPR128<I, "f"#I, !cast<FPR64>("F"#I#"D"),
                     !cast<FPR64>("F"#!add(I, 2)#"D")>;
}

// There's no store-multiple instruction for FPRs, so we're not fussy
// about the order in which call-saved registers are allocated.
defm FP32  : SystemZRegClass<"FP32", f32, 32, (sequence "F%uS", 0, 15)>;
defm FP64  : SystemZRegClass<"FP64", f64, 64, (sequence "F%uD", 0, 15)>;
defm FP128 : SystemZRegClass<"FP128", f128, 128, (add F0Q, F1Q, F4Q, F5Q,
                                                      F8Q, F9Q, F12Q, F13Q)>;

//===----------------------------------------------------------------------===//
// Other registers
//===----------------------------------------------------------------------===//

// The 2-bit condition code field of the PSW.  Every register named in an
// inline asm needs a class associated with it.
def CC : SystemZReg<"cc">;
def CCRegs : RegisterClass<"SystemZ", [i32], 32, (add CC)>;
