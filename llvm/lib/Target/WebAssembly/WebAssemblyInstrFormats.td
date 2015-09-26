// WebAssemblyInstrFormats.td - WebAssembly Instruction Formats -*- tblgen -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly instruction format definitions.
///
//===----------------------------------------------------------------------===//

// WebAssembly Instruction Format.
class WebAssemblyInst<string cstr> : Instruction {
  field bits<0> Inst; // Instruction encoding.
  let Namespace   = "WebAssembly";
  let Pattern     = [];
  let Constraints = cstr;
}

// Normal instructions.
class I<dag oops, dag iops, list<dag> pattern, string cstr = "">
    : WebAssemblyInst<cstr> {
  dag OutOperandList = oops;
  dag InOperandList  = iops;
  let Pattern        = pattern;
}

// Unary and binary instructions, for the local types that WebAssembly supports.
multiclass UnaryInt<SDNode node> {
  def _I32 : I<(outs I32:$dst), (ins I32:$src),
               [(set I32:$dst, (node I32:$src))]>;
  def _I64 : I<(outs I64:$dst), (ins I64:$src),
               [(set I64:$dst, (node I64:$src))]>;
}
multiclass BinaryInt<SDNode node> {
  def _I32 : I<(outs I32:$dst), (ins I32:$lhs, I32:$rhs),
               [(set I32:$dst, (node I32:$lhs, I32:$rhs))]>;
  def _I64 : I<(outs I64:$dst), (ins I64:$lhs, I64:$rhs),
               [(set I64:$dst, (node I64:$lhs, I64:$rhs))]>;
}
multiclass UnaryFP<SDNode node> {
  def _F32 : I<(outs F32:$dst), (ins F32:$src),
               [(set F32:$dst, (node F32:$src))]>;
  def _F64 : I<(outs F64:$dst), (ins F64:$src),
               [(set F64:$dst, (node F64:$src))]>;
}
multiclass BinaryFP<SDNode node> {
  def _F32 : I<(outs F32:$dst), (ins F32:$lhs, F32:$rhs),
               [(set F32:$dst, (node F32:$lhs, F32:$rhs))]>;
  def _F64 : I<(outs F64:$dst), (ins F64:$lhs, F64:$rhs),
               [(set F64:$dst, (node F64:$lhs, F64:$rhs))]>;
}
multiclass ComparisonInt<CondCode cond> {
  def _I32 : I<(outs I32:$dst), (ins I32:$lhs, I32:$rhs),
               [(set I32:$dst, (setcc I32:$lhs, I32:$rhs, cond))]>;
  def _I64 : I<(outs I32:$dst), (ins I64:$lhs, I64:$rhs),
               [(set I32:$dst, (setcc I64:$lhs, I64:$rhs, cond))]>;
}
multiclass ComparisonFP<CondCode cond> {
  def _F32 : I<(outs I32:$dst), (ins F32:$lhs, F32:$rhs),
               [(set I32:$dst, (setcc F32:$lhs, F32:$rhs, cond))]>;
  def _F64 : I<(outs I32:$dst), (ins F64:$lhs, F64:$rhs),
               [(set I32:$dst, (setcc F64:$lhs, F64:$rhs, cond))]>;
}
