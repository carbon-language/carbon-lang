//===- X86RegisterInfo.td - Describe the X86 Register File ------*- C++ -*-===//
//
// This file describes the X86 Register file, defining the registers themselves,
// aliases between the registers, and the register classes built out of the
// registers.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Register definitions...
//
set Namespace = "X86" in {
  // 32-bit registers
  def EAX : Register; def ECX : Register;
  def EDX : Register; def EBX : Register;
  def ESP : Register; def EBP : Register;
  def ESI : Register; def EDI : Register;
  
  // 16-bit registers
  def AX : Register; def CX : Register;
  def DX : Register; def BX : Register;
  def SP : Register; def BP : Register;
  def SI : Register; def DI : Register;
  
  // 8-bit registers
  def AL : Register; def CL : Register;
  def DL : Register; def BL : Register;
  def AH : Register; def CH : Register;
  def DH : Register; def BH : Register;
  
  // Pseudo Floating Point registers
  def FP0 : Register; def FP1 : Register;
  def FP2 : Register; def FP3 : Register;
  def FP4 : Register; def FP5 : Register;
  def FP6 : Register; 

  // Floating point stack registers
  def ST0 : NamedReg<"ST(0)">; def ST1 : NamedReg<"ST(1)">;
  def ST2 : NamedReg<"ST(2)">; def ST3 : NamedReg<"ST(3)">;
  def ST4 : NamedReg<"ST(4)">; def ST5 : NamedReg<"ST(5)">;
  def ST6 : NamedReg<"ST(6)">; def ST7 : NamedReg<"ST(7)">; 
  
  // Flags, Segment registers, etc...
  
  // This is a slimy hack to make it possible to say that flags are clobbered...
  // Ideally we'd model instructions based on which particular flag(s) they
  // could clobber. 
  def EFLAGS : Register;
}

//===----------------------------------------------------------------------===//
// Register alias definitions... define which registers alias which others.  We
// only specify which registers the small registers alias, because the register
// file generator is smart enough to figure out that AL aliases AX if we tell it
// that AX aliases AL (for example).
//
def : RegisterAliases<AL, [AX, EAX]>; def : RegisterAliases<BL, [BX, EBX]>;
def : RegisterAliases<CL, [CX, ECX]>; def : RegisterAliases<DL, [DX, EDX]>;
def : RegisterAliases<AH, [AX, EAX]>; def : RegisterAliases<BH, [BX, EBX]>;
def : RegisterAliases<CH, [CX, ECX]>; def : RegisterAliases<DH, [DX, EDX]>;

def : RegisterAliases<AX, [EAX]>;     def : RegisterAliases<BX, [EBX]>;
def : RegisterAliases<CX, [ECX]>;     def : RegisterAliases<DX, [EDX]>;
def : RegisterAliases<SI, [ESI]>;     def : RegisterAliases<DI, [EDI]>;
def : RegisterAliases<SP, [ESP]>;     def : RegisterAliases<BP, [EBP]>;

//===----------------------------------------------------------------------===//
// Register Class Definitions... now that we have all of the pieces, define the
// top-level register classes.  The order specified in the register list is
// implicitly defined to be the register allocation order.
//
def r8  : RegisterClass<i8,  1, [AL, CL, DL, BL, AH, CH, DH, BH]>;
def r16 : RegisterClass<i16, 2, [AX, CX, DX, BX, SI, DI, BP, SP]> {
  set Methods = [{
    iterator allocation_order_end(MachineFunction &MF) const {
      if (hasFP(MF))     // Does the function dedicate EBP to being a frame ptr?
	return end()-2;  // If so, don't allocate SP or BP
      else
	return end()-1;  // If not, just don't allocate SP
    }
  }];
}

def r32 : RegisterClass<i32, 4, [EAX, ECX, EDX, EBX, ESI, EDI, EBP, ESP]> {
  set Methods = [{
    iterator allocation_order_end(MachineFunction &MF) const {
      if (hasFP(MF))     // Does the function dedicate EBP to being a frame ptr?
        return end()-2;  // If so, don't allocate ESP or EBP
      else
        return end()-1;  // If not, just don't allocate ESP
    }
  }];
}

def rFP : RegisterClass<f80, 4, [FP0, FP1, FP2, FP3, FP4, FP5, FP6]>;

// Registers which cannot be allocated... and are thus left unnamed.
def : RegisterClass<f80, 4, [ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7]>;
def : RegisterClass<i16, 2, [EFLAGS]>;

