//===- X86RegisterInfo.td - Describe the X86 Register File ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 Register file, defining the registers themselves,
// aliases between the registers, and the register classes built out of the
// registers.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Register definitions...
//
let Namespace = "X86" in {

  // In the register alias definitions below, we define which registers alias
  // which others.  We only specify which registers the small registers alias,
  // because the register file generator is smart enough to figure out that
  // AL aliases AX if we tell it that AX aliased AL (for example).

  // 32-bit registers
  def EAX : Register<"EAX">; def ECX : Register<"ECX">;
  def EDX : Register<"EDX">; def EBX : Register<"EBX">;
  def ESP : Register<"ESP">; def EBP : Register<"EBP">;
  def ESI : Register<"ESI">; def EDI : Register<"EDI">;
  
  // 16-bit registers
  def AX : RegisterGroup<"AX", [EAX]>; def CX : RegisterGroup<"CX", [ECX]>;
  def DX : RegisterGroup<"DX", [EDX]>; def BX : RegisterGroup<"BX", [EBX]>;
  def SP : RegisterGroup<"SP", [ESP]>; def BP : RegisterGroup<"BP", [EBP]>;
  def SI : RegisterGroup<"SI", [ESI]>; def DI : RegisterGroup<"DI", [EDI]>;
  
  // 8-bit registers
  def AL : RegisterGroup<"AL", [AX,EAX]>; def CL : RegisterGroup<"CL",[CX,ECX]>;
  def DL : RegisterGroup<"DL", [DX,EDX]>; def BL : RegisterGroup<"BL",[BX,EBX]>;
  def AH : RegisterGroup<"AH", [AX,EAX]>; def CH : RegisterGroup<"CH",[CX,ECX]>;
  def DH : RegisterGroup<"DH", [DX,EDX]>; def BH : RegisterGroup<"BH",[BX,EBX]>;
  
  // Pseudo Floating Point registers
  def FP0 : Register<"FP0">; def FP1 : Register<"FP1">;
  def FP2 : Register<"FP2">; def FP3 : Register<"FP3">;
  def FP4 : Register<"FP4">; def FP5 : Register<"FP5">;
  def FP6 : Register<"FP6">; 

  // Floating point stack registers
  def ST0 : Register<"ST(0)">; def ST1 : Register<"ST(1)">;
  def ST2 : Register<"ST(2)">; def ST3 : Register<"ST(3)">;
  def ST4 : Register<"ST(4)">; def ST5 : Register<"ST(5)">;
  def ST6 : Register<"ST(6)">; def ST7 : Register<"ST(7)">; 
  
  // Flags, Segment registers, etc...
}

//===----------------------------------------------------------------------===//
// Register Class Definitions... now that we have all of the pieces, define the
// top-level register classes.  The order specified in the register list is
// implicitly defined to be the register allocation order.
//

// List AL,CL,DL before AH,CH,DH, as X86 processors often suffer from false
// dependences between upper and lower parts of the register.  BL and BH are
// last because they are call clobbered. Both Athlon and P4 chips suffer this
// issue.
def R8  : RegisterClass<i8,  8, [AL, CL, DL, AH, CH, DH, BL, BH]>;

def R16 : RegisterClass<i16, 16, [AX, CX, DX, SI, DI, BX, BP, SP]> {
  let Methods = [{
    iterator allocation_order_end(MachineFunction &MF) const {
      if (hasFP(MF))     // Does the function dedicate EBP to being a frame ptr?
        return end()-2;  // If so, don't allocate SP or BP
      else
        return end()-1;  // If not, just don't allocate SP
    }
  }];
}

def R32 : RegisterClass<i32, 32, [EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP]> {
  let Methods = [{
    iterator allocation_order_end(MachineFunction &MF) const {
      if (hasFP(MF))     // Does the function dedicate EBP to being a frame ptr?
        return end()-2;  // If so, don't allocate ESP or EBP
      else
        return end()-1;  // If not, just don't allocate ESP
    }
  }];
}

// FIXME: This sets up the floating point register files as though they are f64
// values, though they really are f80 values.  This will cause us to spill
// values as 64-bit quantities instead of 80-bit quantities, which is much much
// faster on common hardware.  In reality, this should be controlled by a
// command line option or something.

def RFP : RegisterClass<f64, 32, [FP0, FP1, FP2, FP3, FP4, FP5, FP6]>;

// Floating point stack registers (these are not allocatable by the
// register allocator - the floating point stackifier is responsible
// for transforming FPn allocations to STn registers)
def RST : RegisterClass<f64, 32, [ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7]> {
  let Methods = [{
    iterator allocation_order_end(MachineFunction &MF) const {
      return begin();
    }
  }];
}
