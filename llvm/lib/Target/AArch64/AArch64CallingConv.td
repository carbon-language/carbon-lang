//==-- AArch64CallingConv.td - Calling Conventions for ARM ----*- tblgen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for AArch64 architecture.
//===----------------------------------------------------------------------===//


// The AArch64 Procedure Call Standard is unfortunately specified at a slightly
// higher level of abstraction than LLVM's target interface presents. In
// particular, it refers (like other ABIs, in fact) directly to
// structs. However, generic LLVM code takes the liberty of lowering structure
// arguments to the component fields before we see them.
//
// As a result, the obvious direct map from LLVM IR to PCS concepts can't be
// implemented, so the goals of this calling convention are, in decreasing
// priority order:
//     1. Expose *some* way to express the concepts required to implement the
//        generic PCS from a front-end.
//     2. Provide a sane ABI for pure LLVM.
//     3. Follow the generic PCS as closely as is naturally possible.
//
// The suggested front-end implementation of PCS features is:
//     * Integer, float and vector arguments of all sizes which end up in
//       registers are passed and returned via the natural LLVM type.
//     * Structure arguments with size <= 16 bytes are passed and returned in
//       registers as similar integer or composite types. For example:
//       [1 x i64], [2 x i64] or [1 x i128] (if alignment 16 needed).
//     * HFAs in registers follow rules similar to small structs: appropriate
//       composite types.
//     * Structure arguments with size > 16 bytes are passed via a pointer,
//       handled completely by the front-end.
//     * Structure return values > 16 bytes via an sret pointer argument.
//     * Other stack-based arguments (not large structs) are passed using byval
//       pointers. Padding arguments are added beforehand to guarantee a large
//       struct doesn't later use integer registers.
//
// N.b. this means that it is the front-end's responsibility (if it cares about
// PCS compliance) to check whether enough registers are available for an
// argument when deciding how to pass it.

class CCIfAlign<int Align, CCAction A>:
  CCIf<"ArgFlags.getOrigAlign() == " # Align, A>;

def CC_A64_APCS : CallingConv<[
  // SRet is an LLVM-specific concept, so it takes precedence over general ABI
  // concerns. However, this rule will be used by C/C++ frontends to implement
  // structure return.
  CCIfSRet<CCAssignToReg<[X8]>>,

  // Put ByVal arguments directly on the stack. Minimum size and alignment of a
  // slot is 64-bit.
  CCIfByVal<CCPassByVal<8, 8>>,

  // Canonicalise the various types that live in different floating-point
  // registers. This makes sense because the PCS does not distinguish Short
  // Vectors and Floating-point types.
  CCIfType<[v2i8], CCBitConvertToType<f16>>,
  CCIfType<[v4i8, v2i16], CCBitConvertToType<f32>>,
  CCIfType<[v8i8, v4i16, v2i32, v2f32], CCBitConvertToType<f64>>,
  CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
           CCBitConvertToType<f128>>,

  // PCS: "C.1: If the argument is a Half-, Single-, Double- or Quad- precision
  // Floating-point or Short Vector Type and the NSRN is less than 8, then the
  // argument is allocated to the least significant bits of register
  // v[NSRN]. The NSRN is incremented by one. The argument has now been
  // allocated."
  CCIfType<[f16],  CCAssignToReg<[B0, B1, B2, B3, B4, B5, B6, B7]>>,
  CCIfType<[f32],  CCAssignToReg<[S0, S1, S2, S3, S4, S5, S6, S7]>>,
  CCIfType<[f64],  CCAssignToReg<[D0, D1, D2, D3, D4, D5, D6, D7]>>,
  CCIfType<[f128], CCAssignToReg<[Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,

  // PCS: "C.2: If the argument is an HFA and there are sufficient unallocated
  // SIMD and Floating-point registers (NSRN - number of elements < 8), then the
  // argument is allocated to SIMD and Floating-point registers (with one
  // register per element of the HFA). The NSRN is incremented by the number of
  // registers used. The argument has now been allocated."
  //
  // N.b. As above, this rule is the responsibility of the front-end.

  // "C.3: If the argument is an HFA then the NSRN is set to 8 and the size of
  // the argument is rounded up to the nearest multiple of 8 bytes."
  //
  // "C.4: If the argument is an HFA, a Quad-precision Floating-point or Short
  // Vector Type then the NSAA is rounded up to the larger of 8 or the Natural
  // Alignment of the Argument's type."
  //
  // It is expected that these will be satisfied by adding dummy arguments to
  // the prototype.

  // PCS: "C.5: If the argument is a Half- or Single- precision Floating-point
  // type then the size of the argument is set to 8 bytes. The effect is as if
  // the argument had been copied to the least significant bits of a 64-bit
  // register and the remaining bits filled with unspecified values."
  CCIfType<[f16, f32], CCPromoteToType<f64>>,

  // PCS: "C.6: If the argument is an HFA, a Half-, Single-, Double- or Quad-
  // precision Floating-point or Short Vector Type, then the argument is copied
  // to memory at the adjusted NSAA. The NSAA is incremented by the size of the
  // argument. The argument has now been allocated."
  CCIfType<[f64], CCAssignToStack<8, 8>>,
  CCIfType<[f128], CCAssignToStack<16, 16>>,

  // PCS: "C.7: If the argument is an Integral Type, the size of the argument is
  // less than or equal to 8 bytes and the NGRN is less than 8, the argument is
  // copied to the least significant bits of x[NGRN]. The NGRN is incremented by
  // one. The argument has now been allocated."

  // First we implement C.8 and C.9 (128-bit types get even registers). i128 is
  // represented as two i64s, the first one being split. If we delayed this
  // operation C.8 would never be reached.
  CCIfType<[i64],
        CCIfSplit<CCAssignToRegWithShadow<[X0, X2, X4, X6], [X0, X1, X3, X5]>>>,

  // Note: the promotion also implements C.14.
  CCIfType<[i8, i16, i32], CCPromoteToType<i64>>,

  // And now the real implementation of C.7
  CCIfType<[i64], CCAssignToReg<[X0, X1, X2, X3, X4, X5, X6, X7]>>,

  // PCS: "C.8: If the argument has an alignment of 16 then the NGRN is rounded
  // up to the next even number."
  //
  // "C.9: If the argument is an Integral Type, the size of the argument is
  // equal to 16 and the NGRN is less than 7, the argument is copied to x[NGRN]
  // and x[NGRN+1], x[NGRN] shall contain the lower addressed double-word of the
  // memory representation of the argument. The NGRN is incremented by two. The
  // argument has now been allocated."
  //
  // Subtlety here: what if alignment is 16 but it is not an integral type? All
  // floating-point types have been allocated already, which leaves composite
  // types: this is why a front-end may need to produce i128 for a struct <= 16
  // bytes.

  // PCS: "C.10 If the argument is a Composite Type and the size in double-words
  // of the argument is not more than 8 minus NGRN, then the argument is copied
  // into consecutive general-purpose registers, starting at x[NGRN]. The
  // argument is passed as though it had been loaded into the registers from a
  // double-word aligned address with an appropriate sequence of LDR
  // instructions loading consecutive registers from memory (the contents of any
  // unused parts of the registers are unspecified by this standard). The NGRN
  // is incremented by the number of registers used. The argument has now been
  // allocated."
  //
  // Another one that's the responsibility of the front-end (sigh).

  // PCS: "C.11: The NGRN is set to 8."
  CCCustom<"CC_AArch64NoMoreRegs">,

  // PCS: "C.12: The NSAA is rounded up to the larger of 8 or the Natural
  // Alignment of the argument's type."
  //
  // PCS: "C.13: If the argument is a composite type then the argument is copied
  // to memory at the adjusted NSAA. The NSAA is by the size of the
  // argument. The argument has now been allocated."
  //
  // Note that the effect of this corresponds to a memcpy rather than register
  // stores so that the struct ends up correctly addressable at the adjusted
  // NSAA.

  // PCS: "C.14: If the size of the argument is less than 8 bytes then the size
  // of the argument is set to 8 bytes. The effect is as if the argument was
  // copied to the least significant bits of a 64-bit register and the remaining
  // bits filled with unspecified values."
  //
  // Integer types were widened above. Floating-point and composite types have
  // already been allocated completely. Nothing to do.

  // PCS: "C.15: The argument is copied to memory at the adjusted NSAA. The NSAA
  // is incremented by the size of the argument. The argument has now been
  // allocated."
  CCIfType<[i64], CCIfSplit<CCAssignToStack<8, 16>>>,
  CCIfType<[i64], CCAssignToStack<8, 8>>

]>;

// According to the PCS, X19-X30 are callee-saved, however only the low 64-bits
// of vector registers (8-15) are callee-saved. The order here is is picked up
// by PrologEpilogInserter.cpp to allocate stack slots, starting from top of
// stack upon entry. This gives the customary layout of x30 at [sp-8], x29 at
// [sp-16], ...
def CSR_PCS : CalleeSavedRegs<(add (sequence "X%u", 30, 19),
                                   (sequence "D%u", 15, 8))>;


// TLS descriptor calls are extremely restricted in their changes, to allow
// optimisations in the (hopefully) more common fast path where no real action
// is needed. They actually have to preserve all registers, except for the
// unavoidable X30 and the return register X0.
def TLSDesc : CalleeSavedRegs<(add (sequence "X%u", 29, 1),
                                   (sequence "Q%u", 31, 0))>;
