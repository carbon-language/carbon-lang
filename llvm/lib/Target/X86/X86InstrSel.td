//===- X86InstrSel.td - Describe the X86 Instruction Selector -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This file is used by the instruction selector and is used to map
//  LLVM instructions to a corresponding set of machine instructions
//
//
//===----------------------------------------------------------------------===//

include "../Target.td"

  // ret br switch invoke unwind
  // add sub mul div rem setcc (eq ne lt gt le ge)
  // and or xor sbl sbr
  // malloc free alloca load store
  // getelementptr phi cast call vanext vaarg


class InstrSubclass<int v> {
  int Value = v;
}

// provide a grouping of InstrSubclasses
class InstrSubclassCollection<string pre, string post, list<InstrSubclass> l> {
  list<InstrSubclass> List = l;
  string PreCode = pre;
  string PostCode = post;
}

// virtual registers
let Namespace = "Virtual" in {
  def DestReg   : Register;
  def DestRegp1 : Register; // DestReg+1
  def Op0Reg    : Register;
  def Op0Regp1  : Register; // Op0Reg+1
  def Op1Reg    : Register;
  def Op1Regp1  : Register; // Op1Reg+1
  def TmpReg01  : Register;
  def TmpReg01p1: Register;
  def TmpReg02  : Register;
  def TmpReg02p1: Register;
  def NullReg   : Register; // represents no register
}

class InstrClass<string n, string fn, string pre, string post, list<InstrSubclassCollection> l> {
  string FunctionName = fn;
  string InstructionName = n;
  string PreCode = pre;
  string PostCode = post;

  // add lists of what subclasses this InstrClass supports
  list<InstrSubclassCollection> Supports = l;
}


// helper class to build BMI instruction
class addedOperand<string op> {
  string Name = op;
}

def MBI_Reg          : addedOperand<"Reg">;
def MBI_GlobalAddr   : addedOperand<"GlobalAddr">;
def MBI_ZImm         : addedOperand<"ZImm">;

def MBI_DirectMem    : addedOperand<"DirectMem">;
def MBI_RegOffset    : addedOperand<"RegOffset">;
def MBI_FrameRef     : addedOperand<"FrameRef">;
def MBI_ConstPoolRef : addedOperand<"ConstPoolRef">;
def MBI_CCReg        : addedOperand<"CCReg">;
def MBI_RegDef       : addedOperand<"RegDef">;
def MBI_PCDisp       : addedOperand<"PCDisp">;
def MBI_MReg         : addedOperand<"MReg">;
def MBI_SImm         : addedOperand<"SImm">;
def MBI_MBB          : addedOperand<"MBB">;
def MBI_FrameIndex   : addedOperand<"FrameIndex">;
def MBI_ConstPoolIdx : addedOperand<"ConstPoolIdx">;
def MBI_ExternSymbol : addedOperand<"ExternSymbol">;


class TargInstr<string n, int numops, list<addedOperand> lops> {

  string Name = n; // make this a reference to the Instruction class
  string Namespace;
  int NumOperands = numops;
  list<addedOperand> Params = lops; // will this work for mem-mem instrs, destination is implicitly a register

} //TargInstr


let Namespace = "Virtual" in {
  // virtual instructions for creating registers
  def CreateRegInt  : TargInstr<"CreateRegInt",0,[]>;
  def CreateRegByte : TargInstr<"CreateRegByte",0,[]>;
  def CreateRegShort: TargInstr<"CreateRegShort",0,[]>;
  def CreateRegLong : TargInstr<"CreateRegLong",0,[]>;
  
  def CreateRegUInt  : TargInstr<"CreateRegUInt",0,[]>;
  def CreateRegUByte : TargInstr<"CreateRegUByte",0,[]>;
  def CreateRegUShort: TargInstr<"CreateRegUShort",0,[]>;
  def CreateRegULong : TargInstr<"CreateRegULong",0,[]>;
  
  def CreateRegFloat  : TargInstr<"CreateRegFloat",0,[]>;
  def CreateRegDouble : TargInstr<"CreateRegDouble",0,[]>;
  def CreateRegPointer: TargInstr<"CreateRegPointer",0,[]>;
  
  def NullInstr       : TargInstr<"NullInstruction",0,[]>; // ignored
}

class TargInstrSet<InstrClass c, string pre, string post, list<InstrSubclass>  l,  list<TargInstr>  targs, list<list<string>> r> {
  InstrClass Class = c;

  list<InstrSubclass> List = l;
  list<TargInstr> Instructions = targs;
  list< list<string> > Operands = r; // generalized for all operand types

  string PreCode = pre;
  string PostCode = post;
}



// --------------------------------------------------------------------
// Begin architecture-specific information
// --------------------------------------------------------------------

include "X86RegisterInfo.td"
include "X86InstrSelInfo.td"


// set up the subclasses of instructions
// value is what the subclass's case value
def cByte  : InstrSubclass<1>;
def cInt   : InstrSubclass<2>;
def cShort : InstrSubclass<3>;
def cFP    : InstrSubclass<4>;
def cLong  : InstrSubclass<5>;

def EQ     : InstrSubclass<1>;
def NE     : InstrSubclass<2>;
def LT     : InstrSubclass<3>;
def GE     : InstrSubclass<4>;
def GT     : InstrSubclass<5>;
def LE     : InstrSubclass<6>;

def True   : InstrSubclass<1>;
def False  : InstrSubclass<0>;

def NegOne : InstrSubclass<-1>;
def PosOne : InstrSubclass<1>;
def Cons   :InstrSubclass<2>;
def Other  : InstrSubclass<0>;

def Regular: InstrSubclass<0>;
def Zero   : InstrSubclass<0>;

def NoOps  : InstrSubclass<0>;
def Ops    : InstrSubclass<1>;

def Signed   : InstrSubclass<0>;
def Unsigned : InstrSubclass<1>;

def SuccessorIsNextBB : InstrSubclass<0>;
def SuccessorIsNotNextBB: InstrSubclass<1>;

def Cond: InstrSubclass<0>;
def Uncond: InstrSubclass<1>;

def ConstTwo: InstrSubclass<2>;
def ConstNotTwo: InstrSubclass<3>;

def Reg: InstrSubclass<6>;


// group subclasses, specify how to determine instruction subclass
def OperandSize: InstrSubclassCollection<"unsigned OperandSize = getClassB(Op0Val->getType());","", [cByte, cShort, cInt, cFP, cLong]>;
def Conditional: InstrSubclassCollection<"unsigned Conditional;", "", [EQ, NE, LT, GE, GT, LE]>;
def SpecialCase: InstrSubclassCollection<"unsigned SpecialCase;", "", [True, False]>;

//TODO write funcs for separating out special cases
def AddSpecialCases: InstrSubclassCollection<"Subclasses AddSpecialCases = Other;", "", [NegOne,PosOne,Cons,Other]>;
def SubSpecialCases: InstrSubclassCollection<"unsigned SubSpecialCases = Other;", "", [NegOne,PosOne,Cons,Other]>;
def XorSpecialCases: InstrSubclassCollection<"unsigned XorSpecialCases = Other;","", [NegOne,Other]>;



// Instruction subclasses, as defined in llvm/Instructions.h:
// iTerminators.h : ReturnInst, BranchInst, SwitchInst, InvokeInst, UnwindInst
// iPHINode.h     : PHINode, 
// iOperators.h   : SetCondInst, 
// iMemory.h      : AllocationInst, MallocInst, AllocaInst, FreeInst, LoadInst, StoreInst, GetElementPtrInst, 
// iOther.h       : CastInst, CallInst, ShiftInst, VANextInst, VAArgInst, 
// InstrTypes.h   : TerminatorInst, BinaryOperator

// general classes of LLVM instructions, and the subclass sets that apply to them
def add : InstrClass<"BinaryOperator","Add","","",[AddSpecialCases, OperandSize]>;
def sub : InstrClass<"BinaryOperator","Sub","","",[SubSpecialCases, OperandSize]>;
// def mul : InstrClass<"BinaryOperator","Mul","","",[OperandSize]>;
// def div : InstrClass<"BinaryOperator","Div","","",[OperandSize]>;
// def rem : InstrClass<"BinaryOperator","Rem","","",[OperandSize]>;
def logic_and : InstrClass<"BinaryOperator","And","","",[OperandSize]>;
def logic_or  : InstrClass<"BinaryOperator","Or","","",[OperandSize]>;
def logic_xor : InstrClass<"BinaryOperator","Xor","","",[XorSpecialCases, OperandSize]>;
// //def mov : InstrClass<"ShiftInst","Mov","","",[OperandSize]>;

// def setcc: InstrClass<"SetCondInst","SetCondInst","","",[OperandSize]>;
// def branch  : InstrClass<"BranchInst","BranchInst","","",[Conditional]>;

// def retrn : InstrClass<"ReturnInst","ReturnInst","","",[]>;

// definition of machine instructions for instruction subclasses


// ADD

def ADD_Other_cByte     : TargInstrSet<add,"","",[Other,cByte],[CreateRegByte,ADDrr8],[["EAX"],["EAX","Op0Reg","Op1Reg"]]>;
def ADD_Other_cShort    : TargInstrSet<add,"","",[Other,cShort],[ADDrr16],[["DestReg","Op0Reg","Op1Reg"]]>;
def ADD_Other_cInt      : TargInstrSet<add,"","",[Other,cInt],[ADDrr32],[["DestReg","Op0Reg","Op1Reg"]]>;
def ADD_Other_cLong     : TargInstrSet<add,"","",[Other,cLong],[ADDrr32,ADCrr32],[["DestReg","Op0Reg","Op1Reg"],["DestReg","Op0Reg","Op1Reg"]]>;
def ADD_Other_cFP       : TargInstrSet<add,"","",[Other,cFP],[FpADD],[["DestReg","Op0Reg","Op1Reg"]]>;
def ADD_Constant_cByte  : TargInstrSet<add,"","",[Cons,cByte],[ADDri8],[["DestReg","Op0Reg","Op1Reg"]]>;
def ADD_Constant_cShort :TargInstrSet<add,"","",[Cons,cShort],[ADDri16],[["DestReg","Op0Reg","Op1Reg"]]>;
def ADD_Constant_cInt   :TargInstrSet<add,"","",[Cons,cInt],[ADDri32],[["DestReg","Op0Reg","Op1Reg"]]>;
def ADD_PosOne_cByte    : TargInstrSet<add,"","",[PosOne,cByte],[INCr8],[["DestReg","Op0Reg"]]>;
def ADD_PosOne_cShort   : TargInstrSet<add,"","",[PosOne,cShort],[INCr16],[["DestReg","Op0Reg"]]>;
def ADD_PosOne_cInt     : TargInstrSet<add,"","",[PosOne,cInt],[INCr32],[["DestReg","Op0Reg"]]>;
def ADD_NegOne_cByte    : TargInstrSet<add,"","",[NegOne,cByte],[DECr8],[["DestReg","Op0Reg"]]>;
def ADD_NegOne_cShort   : TargInstrSet<add,"","",[NegOne,cShort],[DECr16],[["DestReg","Op0Reg"]]>;
def ADD_NegOne_cInt     : TargInstrSet<add,"","",[NegOne,cInt],[DECr32],[["DestReg","Op0Reg"]]>;


// SUBTRACT

def SUB_Other_cByte     : TargInstrSet<sub,"","",[Other,cByte],[CreateRegByte,SUBrr8],[["EAX"],["EAX","Op0Reg","Op1Reg"]]>;
def SUB_Other_cShort    : TargInstrSet<sub,"","",[Other,cShort],[SUBrr16],[["DestReg","Op0Reg","Op1Reg"]]>;
def SUB_Other_cInt      : TargInstrSet<sub,"","",[Other,cInt],[SUBrr32],[["DestReg","Op0Reg","Op1Reg"]]>;
def SUB_Other_cLong     : TargInstrSet<sub,"","",[Other,cLong],[SUBrr32,SUBrr32],[["DestReg","Op0Reg","Op1Reg"],["DestReg","Op0Reg","Op1Reg"]]>;
def SUB_Other_cFP       : TargInstrSet<sub,"","",[Other,cFP],[FpSUB],[["DestReg","Op0Reg","Op1Reg"]]>;
def SUB_Constant_cByte  : TargInstrSet<sub,"","",[Cons,cByte],[SUBri8],[["DestReg","Op0Reg","Op1Reg"]]>;
def SUB_Constant_cShort :TargInstrSet<sub,"","",[Cons,cShort],[SUBri16],[["DestReg","Op0Reg","Op1Reg"]]>;
def SUB_Constant_cInt   :TargInstrSet<sub,"","",[Cons,cInt],[SUBri32],[["DestReg","Op0Reg","Op1Reg"]]>;
def SUB_PosOne_cByte    : TargInstrSet<sub,"","",[PosOne,cByte],[DECr8],[["DestReg","Op0Reg"]]>;
def SUB_PosOne_cShort   : TargInstrSet<sub,"","",[PosOne,cShort],[DECr16],[["DestReg","Op0Reg"]]>;
def SUB_PosOne_cInt     : TargInstrSet<sub,"","",[PosOne,cInt],[DECr32],[["DestReg","Op0Reg"]]>;
def SUB_NegOne_cByte    : TargInstrSet<sub,"","",[NegOne,cByte],[INCr8],[["DestReg","Op0Reg"]]>;
def SUB_NegOne_cShort   : TargInstrSet<sub,"","",[NegOne,cShort],[INCr16],[["DestReg","Op0Reg"]]>;
def SUB_NegOne_cInt     : TargInstrSet<sub,"","",[NegOne,cInt],[INCr32],[["DestReg","Op0Reg"]]>;


// def SHIFT_S_L_Const_Byte     : 
// def SHIFT_S_L_Const_Short    : 
// def SHIFT_S_L_Const_Int      : 
// def SHIFT_S_L_Const_SHLDIR32 : 
// def SHIFT_S_L_Reg_Byte       : 
// def SHIFT_S_L_Reg_Short      : 
// def SHIFT_S_L_Reg_Int        : 
// def SHIFT_S_R_Const_Byte     : 
// def SHIFT_S_R_Const_Short    :
// def SHIFT_S_R_Const_Int      : 
// def SHIFT_S_R_Const_SHRDIR32 : 
// def SHIFT_S_R_Reg_Byte       : 
// def SHIFT_S_R_Reg_Short      : 
// def SHIFT_S_R_Reg_Int        : 
// def SHIFT_U_L_Const_Byte     : 
// def SHIFT_U_L_Const_Short    : 
// def SHIFT_U_L_Const_Int      : 
// def SHIFT_U_L_Const_SHLDIR32 : 
// def SHIFT_U_L_Reg_Byte       : 
// def SHIFT_U_L_Reg_Short      : 
// def SHIFT_U_L_Reg_Int        : 
// def SHIFT_U_R_Const_Byte     : 
// def SHIFT_U_R_Const_Short    : 
// def SHIFT_U_R_Const_Int      : 
// def SHIFT_U_R_Const_SHRDIR32 : 
// def SHIFT_U_R_Reg_Byte       : 
// def SHIFT_U_R_Reg_Short      : 
// def SHIFT_U_R_Reg_Int        : 


// def CMP_i_Byte  : TargInstrSet<setcc,"","",[Constant,cByte],[CMPri8],[["DestReg","Op0Reg","Op1Val"]]>;
// def CMP_i_Short : TargInstrSet<setcc,"","",[Constant,cShort],[CMPri16],[["DestReg","Op0Reg","Op1Val"]]>;
// def CMP_i_Int   : TargInstrSet<setcc,"","",[Constant,cInt],[CMPri32],[["DestReg","Op0Reg","Op1Val"]]>;
// //def CMP_i_Long  : // not supported
// def CMP_z_Byte  : TargInstrSet<setcc,"ConstantInt *CI = dyn_cast<ConstantInt>(Op1); uint64_t Op1v = cast<ConstantInt>(CI)->getRawValue(); Op1v &= (1ULL << (8 << Class)) - 1;","",[Zero,cByte],[TESTrr8],[["DestReg","Op0Reg","Op0Reg"]]>;
// def CMP_z_Short : TargInstrSet<setcc,"ConstantInt *CI = dyn_cast<ConstantInt>(Op1); uint64_t Op1v = cast<ConstantInt>(CI)->getRawValue(); Op1v &= (1ULL << (8 << Class)) - 1;","",[Zero,cShort],[TESTrr16],[["DestReg","Op0Reg","Op0Reg"]]>;
// def CMP_z_Int   : TargInstrSet<setcc,"ConstantInt *CI = dyn_cast<ConstantInt>(Op1); uint64_t Op1v = cast<ConstantInt>(CI)->getRawValue(); Op1v &= (1ULL << (8 << Class)) - 1;","",[Zero,cInt],[TESTrr32],[["DestReg","Op0Reg","Op0Reg"]]>;
// //def CMP_z_Long  : // not supported
// def CMP_r_Byte  : TargInstrSet<setcc,"","",[Regular,cByte],[CMPrr8],[["NullReg","Op0Reg","Op1Reg"]]>;
// def CMP_r_Short : TargInstrSet<setcc,"","",[Regular,cShort],[CMPrr16],[["NullReg","Op0Reg","Op1Reg"]]>;
// def CMP_r_Int   : TargInstrSet<setcc,"","",[Regular,cInt],[CMPrr32],[["NullReg","Op0Reg","Op1Reg"]]>;
// //def CMP_r_Long  : // two cases of long, depending on num of operands
// def CMP_r_FP    : TargInstrSet<setcc,"","",[Regular,cFP],[FpUCOM,FNSTSWr8,SAHF],[["NullReg","Op0Reg","Op1Reg"],["NullReg"],["NullReg"]]>;


// def RET_NoOps    : TargInstrSet<retrn,"","",[NoOps],[FP_REG_KILL,RET],[["NullReg"],["NullReg"]]>;
// def RET_Op_Byte  : TargInstrSet<retrn,"promote32(X86::EAX, ValueRecord(Op0Reg, Op0Val->getType()));","",[Ops,cByte],[IMPLICIT_USE],[["NullReg","EAX","ESP"]]>;
// def RET_Op_Short : TargInstrSet<retrn,"promote32(X86::EAX, ValueRecord(Op0Reg, Op0Val->getType()));","",[Ops,cShort],[IMPLICIT_USE],[["NullReg","EAX","ESP"]]>;
// def RET_Op_Int   : TargInstrSet<retrn,"promote32(X86::EAX, ValueRecord(Op0Reg, Op0Val->getType()));","",[Ops,cInt],[IMPLICIT_USE],[["NullReg","EAX","ESP"]]>;
// def RET_Op_FP    : TargInstrSet<retrn,"","",[Ops,cFP],[FpSETRESULT,IMPLICIT_USE],[["NullReg","Op0Reg"],["NullReg","STO","ESP"]]>;
// def RET_Op_Long  : TargInstrSet<retrn,"","",[Ops,cLong],[MOVrr32,MOVrr32,IMPLICIT_USE],[["EAX","Op0Reg"],["EDX","Op0Reg+1"],["NullReg","EAX","EDX","ESP"]]>;


// def BR_Uncond    : TargInstrSet<branch,"","",[Uncond],[FP_REG_KILL,JMP],[["NullReg"],["NullReg","I.getSuccessor(0);"]]>;
// def BR_Cond_S_EQ : 
// def BR_Cond_S_NE
// def BR_Cond_S_B
// def BR_Cond_S_AE
// def BR_Cond_S_A
// def BR_Cond_S_BE
// def BR_Cond_U_EQ
// def BR_Cond_U_NE
// def BR_Cond_U_LT
// def BR_Cond_U_GE
// def BR_Cond_U_GT
// def BR_Cond_U_LE
// def BR_Cond_U_S
// def BR_Cond_U_NS


// def CALL_Byte
// def CALL_Short
// def CALL_Int
// def CALL_Long
// def CALL_FP


def AND_Byte  : TargInstrSet<logic_and,"","",[cByte],[ANDrr8],[["DestReg","Op0Reg","Op1Reg"]]>;
def AND_Short : TargInstrSet<logic_and,"","",[cShort],[ANDrr16],[["DestReg","Op0Reg","Op1Reg"]]>;
def AND_Int   : TargInstrSet<logic_and,"","",[cInt],[ANDrr32],[["DestReg","Op0Reg","Op1Reg"]]>;
def AND_Long  : TargInstrSet<logic_and,"","",[cLong],[ANDrr32,ANDrr32],[["DestReg","Op0Reg","Op1Reg"],["DestReg+1","Op0Reg+1","Op1Reg+1"]]>;


def OR_Byte  : TargInstrSet<logic_or,"","",[cByte],[ORrr8],[["DestReg","Op0Reg","Op1Reg"]]>;
def OR_Short : TargInstrSet<logic_or,"","",[cShort],[ORrr16],[["DestReg","Op0Reg","Op1Reg"]]>;
def OR_Int   : TargInstrSet<logic_or,"","",[cInt],[ORrr32],[["DestReg","Op0Reg","Op1Reg"]]>;
def OR_Long  : TargInstrSet<logic_or,"","",[cLong],[ORrr32,ORrr32],[["DestReg","Op0Reg","Op1Reg"],["DestReg+1","Op0Reg+1","Op1Reg+1"]]>;


def XOR_NegOne_Byte : TargInstrSet<logic_xor,"","",[NegOne,cByte],[NOTr8],[["DestReg","Op0Reg"]]>;
def XOR_NegOne_Short: TargInstrSet<logic_xor,"","",[NegOne,cShort],[NOTr16],[["DestReg","Op0Reg"]]>;
def XOR_NegOne_Int  : TargInstrSet<logic_xor,"","",[NegOne,cInt],[NOTr32],[["DestReg","Op0Reg"]]>;
//def XOR_NegOne_Long :  // not supported (treat as regular long XOR)
def XOR_Other_Byte  : TargInstrSet<logic_xor,"","",[Other,cByte],[XORrr8],[["DestReg","Op0Reg","Op1Reg"]]>;
def XOR_Other_Short : TargInstrSet<logic_xor,"","",[Other,cByte],[XORrr16],[["DestReg","Op0Reg","Op1Reg"]]>;
def XOR_Other_Int   : TargInstrSet<logic_xor,"","",[Other,cByte],[XORrr32],[["DestReg","Op0Reg","Op1Reg"]]>;
def XOR_Other_Long  : TargInstrSet<logic_xor,"","",[Other,cLong],[XORrr32,XORrr32],[["DestReg","Op0Reg","Op1Reg"],["DestReg+1","Op0Reg+1","Op1Reg+1"]]>;



// // TODO support for arbitrary values as operand arguments
// //      or make ConstantInt versions as well-known variables
// // ConstantInt, unsigned Val
// def MUL_ConstTwo_Byte     : TargInstrSet<mul,"ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(1)); unsigned Val = (unsigned)CI->getRawValue(); unsigned Shift = ExactLog2(Val);","",[ConstTwo,cByte],[SHLir32],[["DestReg","Op0Reg","Shift-1"]]>;
// def MUL_ConstTwo_Short    : TargInstrSet<mul,"ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(1)); unsigned Val = (unsigned)CI->getRawValue(); unsigned Shift = ExactLog2(Val);","",[ConstTwo,cShort],[SHLir32],[["DestReg","Op0Reg","Shift-1"]]>;
// def MUL_ConstTwo_Int      : TargInstrSet<mul,"ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(1)); unsigned Val = (unsigned)CI->getRawValue(); unsigned Shift = ExactLog2(Val);","",[ConstTwo,cInt],[SHLir32],[["DestReg","Op0Reg","Shift-1"]]>;
// def MUL_ConstNotTwo_Byte  : TargInstrSet<mul,"","",[ConstNotTwo,cByte],[CreateRegByte,MOVir8,MOVrr8,MULr8,MOVrr8],[["tmpReg"],["tmpReg","ConstRHS"],["AL","Op0Reg"],["NullReg","Op1Reg"],["DestReg","AL"]]>; //TODO fixme, define ConstRHS
// def MUL_ConstNotTwo_Short : TargInstrSet<mul,"","",[ConstNotTwo,cShort],[IMULri16],[["DestReg","Op0Reg","Val"]]>;
// def MUL_ConstNotTwo_Int   : TargInstrSet<mul,"","",[ConstNotTwo,cInt],[IMULri32],[["DestReg","Op0Reg","Val"]]>;
// //def MUL_ConstNotTwo_Long  :
// //def MUL_ConstNotTwo_FP    :
// def MUL_Reg_Byte          : TargInstrSet<mul,"","",[Reg,cByte],[MOVrr8,MULr8,MOVrr8],[["AL","Op0Reg"],["NullReg","Op1Reg"],["DestReg","AL"]]>;
// def MUL_Reg_Short         : TargInstrSet<mul,"","",[Reg,cShort],[IMULrr16],[["DestReg","Op0Reg","Op1Reg"]]>;
// def MUL_Reg_Int           : TargInstrSet<mul,"","",[Reg,cInt],[IMULrr32],[["DestReg","Op0Reg","Op1Reg"]]>;
// //def MUL_Reg_Long          : // not supported (assert)
// def MUL_Reg_FP            : TargInstrSet<mul,"","",[Reg,cFP],[FpMUL],[["DestReg","Op0Reg","Op1Reg"]]>;

// def STO_Byte
// def STO_Short
// def STO_Int
// def STO_FP
// def STO_Long


// DIVREM

// CAST

// PHI

// LOAD ARGS TO VIRTUAL REGS

// COPY CONSTANT

//TODO
//  handling various operand options
//  allowing for arbitrary code or operand arguments
//  generate skeletons for selector function
//  
