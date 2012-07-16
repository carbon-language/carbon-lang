//==- AMDILFormats.td - AMDIL Instruction Formats ----*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
//===--------------------------------------------------------------------===//
include "AMDILTokenDesc.td"

//===--------------------------------------------------------------------===//
// The parent IL instruction class that inherits the Instruction class. This
// class sets the corresponding namespace, the out and input dag lists the
// pattern to match to and the string to print out for the assembly printer.
//===--------------------------------------------------------------------===//
class ILFormat<ILOpCode op, dag outs, dag ins, string asmstr, list<dag> pattern>
: Instruction {

     let Namespace = "AMDGPU";
     dag OutOperandList = outs;
     dag InOperandList = ins;
     ILOpCode operation = op;
     let Pattern = pattern;
     let AsmString = !strconcat(asmstr, "\n");
     let isPseudo = 1;
     let Itinerary = NullALU;
     bit hasIEEEFlag = 0;
     bit hasZeroOpFlag = 0;
}

//===--------------------------------------------------------------------===//
// Class that has one input parameters and one output parameter.
// The basic pattern for this class is "Opcode Dst, Src0" and
// handles the unary math operators.
// It sets the binary token ILSrc, ILSrcMod, ILRelAddr and ILSrc and ILSrcMod
// if the addressing is register relative for input and output register 0.
//===--------------------------------------------------------------------===//
class OneInOneOut<ILOpCode op, dag outs, dag ins,
      string asmstr, list<dag> pattern>
      : ILFormat<op, outs, ins, asmstr, pattern>
{
     ILDst       dst_reg;
     ILDstMod    dst_mod;
     ILRelAddr   dst_rel;
     ILSrc       dst_reg_rel;
     ILSrcMod    dst_reg_rel_mod;
     ILSrc       src0_reg;
     ILSrcMod    src0_mod;
     ILRelAddr   src0_rel;
     ILSrc       src0_reg_rel;
     ILSrcMod    src0_reg_rel_mod;
}

//===--------------------------------------------------------------------===//
// This class is similiar to the UnaryOp class, however, there is no
// result value to assign.
//===--------------------------------------------------------------------===//
class UnaryOpNoRet<ILOpCode op, dag outs, dag ins,
      string asmstr, list<dag> pattern>
      : ILFormat<op, outs, ins, asmstr, pattern>
{
     ILSrc       src0_reg;
     ILSrcMod    src0_mod;
     ILRelAddr   src0_rel;
     ILSrc       src0_reg_rel;
     ILSrcMod    src0_reg_rel_mod;
}

//===--------------------------------------------------------------------===//
// Set of classes that have two input parameters and one output parameter.
// The basic pattern for this class is "Opcode Dst, Src0, Src1" and
// handles the binary math operators and comparison operations.
// It sets the binary token ILSrc, ILSrcMod, ILRelAddr and ILSrc and ILSrcMod
// if the addressing is register relative for input register 1.
//===--------------------------------------------------------------------===//
class TwoInOneOut<ILOpCode op, dag outs, dag ins,
      string asmstr, list<dag> pattern>
      : OneInOneOut<op, outs, ins, asmstr, pattern>
{
     ILSrc       src1_reg;
     ILSrcMod    src1_mod;
     ILRelAddr   src1_rel;
     ILSrc       src1_reg_rel;
     ILSrcMod    src1_reg_rel_mod;
}

//===--------------------------------------------------------------------===//
// Similiar to the UnaryOpNoRet class, but takes as arguments two input
// operands. Used mainly for barrier instructions on PC platform.
//===--------------------------------------------------------------------===//
class BinaryOpNoRet<ILOpCode op, dag outs, dag ins,
      string asmstr, list<dag> pattern>
      : UnaryOpNoRet<op, outs, ins, asmstr, pattern>
{
     ILSrc       src1_reg;
     ILSrcMod    src1_mod;
     ILRelAddr   src1_rel;
     ILSrc       src1_reg_rel;
     ILSrcMod    src1_reg_rel_mod;
}

//===--------------------------------------------------------------------===//
// Set of classes that have three input parameters and one output parameter.
// The basic pattern for this class is "Opcode Dst, Src0, Src1, Src2" and
// handles the mad and conditional mov instruction.
// It sets the binary token ILSrc, ILSrcMod, ILRelAddr and ILSrc and ILSrcMod
// if the addressing is register relative.
// This class is the parent class of TernaryOp
//===--------------------------------------------------------------------===//
class ThreeInOneOut<ILOpCode op, dag outs, dag ins,
      string asmstr, list<dag> pattern>
      : TwoInOneOut<op, outs, ins, asmstr, pattern> {
           ILSrc       src2_reg;
           ILSrcMod    src2_mod;
           ILRelAddr   src2_rel;
           ILSrc       src2_reg_rel;
           ILSrcMod    src2_reg_rel_mod;
      }

//===--------------------------------------------------------------------===//
// Intrinsic classes
// Generic versions of the above classes but for Target specific intrinsics
// instead of SDNode patterns.
//===--------------------------------------------------------------------===//
let TargetPrefix = "AMDIL", isTarget = 1 in {
     class VoidIntLong :
          Intrinsic<[llvm_i64_ty], [], []>;
     class VoidIntInt :
          Intrinsic<[llvm_i32_ty], [], []>;
     class VoidIntBool :
          Intrinsic<[llvm_i32_ty], [], []>;
     class UnaryIntInt :
          Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>], [IntrNoMem]>;
     class UnaryIntFloat :
          Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
     class ConvertIntFTOI :
          Intrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty], [IntrNoMem]>;
     class ConvertIntITOF :
          Intrinsic<[llvm_anyfloat_ty], [llvm_anyint_ty], [IntrNoMem]>;
     class UnaryIntNoRetInt :
          Intrinsic<[], [llvm_anyint_ty], []>;
     class UnaryIntNoRetFloat :
          Intrinsic<[], [llvm_anyfloat_ty], []>;
     class BinaryIntInt :
          Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
     class BinaryIntFloat :
          Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
     class BinaryIntNoRetInt :
          Intrinsic<[], [llvm_anyint_ty, LLVMMatchType<0>], []>;
     class BinaryIntNoRetFloat :
          Intrinsic<[], [llvm_anyfloat_ty, LLVMMatchType<0>], []>;
     class TernaryIntInt :
          Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>,
          LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
     class TernaryIntFloat :
          Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>,
          LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
     class QuaternaryIntInt :
          Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>,
          LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
     class UnaryAtomicInt :
          Intrinsic<[llvm_i32_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrReadWriteArgMem]>;
     class BinaryAtomicInt :
          Intrinsic<[llvm_i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], [IntrReadWriteArgMem]>;
     class TernaryAtomicInt :
          Intrinsic<[llvm_i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty]>;
     class UnaryAtomicIntNoRet :
          Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty], [IntrReadWriteArgMem]>;
     class BinaryAtomicIntNoRet :
          Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], [IntrReadWriteArgMem]>;
     class TernaryAtomicIntNoRet :
          Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrReadWriteArgMem]>;
}
