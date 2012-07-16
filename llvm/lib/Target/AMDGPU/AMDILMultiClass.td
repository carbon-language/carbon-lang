//===-- AMDILMultiClass.td - AMDIL Multiclass defs ---*- tablegen -*-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
// Multiclass that handles branch instructions
multiclass BranchConditional<SDNode Op> {
    def _i32 : ILFormat<IL_OP_IFC, (outs),
  (ins brtarget:$target, GPRI32:$src0),
        "; i32 Pseudo branch instruction",
  [(Op bb:$target, GPRI32:$src0)]>;
    def _f32 : ILFormat<IL_OP_IFC, (outs),
  (ins brtarget:$target, GPRF32:$src0),
        "; f32 Pseudo branch instruction",
  [(Op bb:$target, GPRF32:$src0)]>;
}

// Multiclass that handles memory store operations
multiclass GTRUNCSTORE<string asm> {
  def _i32i8 : OneInOneOut<IL_OP_MOV, (outs), (ins GPRI32:$val, MEMI32:$ptr),
      !strconcat(asm, " $val $ptr"),
      [(global_i8trunc_store GPRI32:$val, ADDR:$ptr)]>;
  def _i32i16 : OneInOneOut<IL_OP_MOV, (outs), (ins GPRI32:$val, MEMI32:$ptr),
      !strconcat(asm, " $val $ptr"),
      [(global_i16trunc_store GPRI32:$val, ADDR:$ptr)]>;
}

// Multiclass that handles memory store operations
multiclass LTRUNCSTORE<string asm> {
  def _i32i8 : OneInOneOut<IL_OP_MOV, (outs), (ins GPRI32:$val, MEMI32:$ptr),
      !strconcat(asm, " $val $ptr"),
      [(local_i8trunc_store GPRI32:$val, ADDR:$ptr)]>;
  def _i32i16 : OneInOneOut<IL_OP_MOV, (outs), (ins GPRI32:$val, MEMI32:$ptr),
      !strconcat(asm, " $val $ptr"),
      [(local_i16trunc_store GPRI32:$val, ADDR:$ptr)]>;
}

// Multiclass that handles memory store operations
multiclass PTRUNCSTORE<string asm> {
  def _i32i8 : OneInOneOut<IL_OP_MOV, (outs), (ins GPRI32:$val, MEMI32:$ptr),
      !strconcat(asm, " $val $ptr"),
      [(private_i8trunc_store GPRI32:$val, ADDR:$ptr)]>;
  def _i32i16 : OneInOneOut<IL_OP_MOV, (outs), (ins GPRI32:$val, MEMI32:$ptr),
      !strconcat(asm, " $val $ptr"),
      [(private_i16trunc_store GPRI32:$val, ADDR:$ptr)]>;
}

// Multiclass that handles memory store operations
multiclass RTRUNCSTORE<string asm> {
  def _i32i8 : OneInOneOut<IL_OP_MOV, (outs), (ins GPRI32:$val, MEMI32:$ptr),
      !strconcat(asm, " $val $ptr"),
      [(region_i8trunc_store GPRI32:$val, ADDR:$ptr)]>;
  def _i32i16 : OneInOneOut<IL_OP_MOV, (outs), (ins GPRI32:$val, MEMI32:$ptr),
      !strconcat(asm, " $val $ptr"),
      [(region_i16trunc_store GPRI32:$val, ADDR:$ptr)]>;
}


// Multiclass that handles memory store operations
multiclass STORE<string asm, PatFrag OpNode> {
  def _i32 : OneInOneOut<IL_OP_MOV, (outs), (ins GPRI32:$val, MEMI32:$ptr),
      !strconcat(asm, " $val $ptr"),
      [(OpNode GPRI32:$val, ADDR:$ptr)]>;
  def _f32 : OneInOneOut<IL_OP_MOV, (outs), (ins GPRF32:$val, MEMI32:$ptr),
      !strconcat(asm, " $val $ptr"),
      [(OpNode GPRF32:$val, ADDR:$ptr)]>;
}

// Multiclass that handles load operations
multiclass LOAD<string asm, PatFrag OpNode> {
  def _i32 : OneInOneOut<IL_OP_MOV, (outs GPRI32:$dst), (ins MEMI32:$ptr),
      !strconcat(asm, " $dst $ptr"),
      [(set GPRI32:$dst, (OpNode ADDR:$ptr))]>;
  def _f32 : OneInOneOut<IL_OP_MOV, (outs GPRF32:$dst), (ins MEMI32:$ptr),
      !strconcat(asm, " $dst $ptr"),
      [(set GPRF32:$dst, (OpNode ADDR:$ptr))]>;
}

// Only scalar types should generate flow control
multiclass BranchInstr<ILOpCode opc> {
  def _i32 : UnaryOpNoRet<opc, (outs), (ins GPRI32:$src),
      !strconcat(opc.Text, " $src"), []>;
  def _f32 : UnaryOpNoRet<opc, (outs), (ins GPRF32:$src),
      !strconcat(opc.Text, " $src"), []>;
}
// Only scalar types should generate flow control
multiclass BranchInstr2<ILOpCode opc> {
  def _i32 : BinaryOpNoRet<opc, (outs), (ins GPRI32:$src0, GPRI32:$src1),
      !strconcat(opc.Text, " $src0, $src1"), []>;
  def _f32 : BinaryOpNoRet<opc, (outs), (ins GPRF32:$src0, GPRF32:$src1),
      !strconcat(opc.Text, " $src0, $src1"), []>;
}
