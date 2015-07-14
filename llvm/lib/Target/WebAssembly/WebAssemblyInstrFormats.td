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

// WebAssembly Instruction Format
class WebAssemblyInst<string cstr> : Instruction {
  field bits<0> Inst; // Instruction encoding.
  let Namespace   = "WebAssembly";
  let Pattern     = [];
  let Constraints = cstr;
}

// Normal instructions
class I<dag oops, dag iops, list<dag> pattern, string cstr = "">
    : WebAssemblyInst<cstr> {
  dag OutOperandList = oops;
  dag InOperandList  = iops;
  let Pattern        = pattern;
}

// Unary and binary instructions, for the local types that WebAssembly supports.
multiclass UnaryInt<SDNode node> {
  def _I32 : I<(outs Int32:$dst), (ins Int32:$src),
               [(set Int32:$dst, (node Int32:$src))]>;
  def _I64 : I<(outs Int64:$dst), (ins Int64:$src),
               [(set Int64:$dst, (node Int64:$src))]>;
}
multiclass BinaryInt<SDNode node> {
  def _I32 : I<(outs Int32:$dst), (ins Int32:$lhs, Int32:$rhs),
               [(set Int32:$dst, (node Int32:$lhs, Int32:$rhs))]>;
  def _I64 : I<(outs Int64:$dst), (ins Int64:$lhs, Int64:$rhs),
               [(set Int64:$dst, (node Int64:$lhs, Int64:$rhs))]>;
}
multiclass UnaryFP<SDNode node> {
  def _F32 : I<(outs Float32:$dst), (ins Float32:$src),
               [(set Float32:$dst, (node Float32:$src))]>;
  def _F64 : I<(outs Float64:$dst), (ins Float64:$src),
               [(set Float64:$dst, (node Float64:$src))]>;
}
multiclass BinaryFP<SDNode node> {
  def _F32 : I<(outs Float32:$dst), (ins Float32:$lhs, Float32:$rhs),
               [(set Float32:$dst, (node Float32:$lhs, Float32:$rhs))]>;
  def _F64 : I<(outs Float64:$dst), (ins Float64:$lhs, Float64:$rhs),
               [(set Float64:$dst, (node Float64:$lhs, Float64:$rhs))]>;
}
