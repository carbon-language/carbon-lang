/*===-- llvm_ocaml.h - LLVM Ocaml Glue --------------------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file was developed by Gordon Henriksen and is distributed under the   *|
|* University of Illinois Open Source License. See LICENSE.TXT for details.   *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file glues LLVM's ocaml interface to its C interface. These functions *|
|* are by and large transparent wrappers to the corresponding C functions.    *|
|*                                                                            *|
|* Note that these functions intentionally take liberties with the CAMLparamX *|
|* macros, since most of the parameters are not GC heap objects.              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c/Core.h"
#include "caml/alloc.h"
#include "caml/custom.h"
#include "caml/mlvalues.h"
#include "caml/memory.h"
#include "llvm/Config/config.h" 


/*===-- Modules -----------------------------------------------------------===*/

/* string -> llmodule */
CAMLprim LLVMModuleRef llvm_create_module(value ModuleID) {
  return LLVMModuleCreateWithName(String_val(ModuleID));
}

/* llmodule -> unit */
CAMLprim value llvm_dispose_module(LLVMModuleRef M) {
  LLVMDisposeModule(M);
  return Val_unit;
}

/* string -> lltype -> llmodule -> bool */
CAMLprim value llvm_add_type_name(value Name, LLVMTypeRef Ty, LLVMModuleRef M) {
  int res = LLVMAddTypeName(M, String_val(Name), Ty);
  return Val_bool(res == 0);
}

/* string -> llmodule -> unit */
CAMLprim value llvm_delete_type_name(value Name, LLVMModuleRef M) {
  LLVMDeleteTypeName(M, String_val(Name));
  return Val_unit;
}


/*===-- Types -------------------------------------------------------------===*/

/* lltype -> type_kind */
CAMLprim value llvm_classify_type(LLVMTypeRef Ty) {
  return Val_int(LLVMGetTypeKind(Ty));
}

/* lltype -> lltype -> unit */
CAMLprim value llvm_refine_abstract_type(LLVMTypeRef ConcreteTy,
                                         LLVMTypeRef AbstractTy) {
  LLVMRefineAbstractType(AbstractTy, ConcreteTy);
  return Val_unit;
}

/*--... Operations on integer types ........................................--*/

/* unit -> lltype */
CAMLprim LLVMTypeRef llvm_i1_type (value Unit) { return LLVMInt1Type();  }
CAMLprim LLVMTypeRef llvm_i8_type (value Unit) { return LLVMInt8Type();  }
CAMLprim LLVMTypeRef llvm_i16_type(value Unit) { return LLVMInt16Type(); }
CAMLprim LLVMTypeRef llvm_i32_type(value Unit) { return LLVMInt32Type(); }
CAMLprim LLVMTypeRef llvm_i64_type(value Unit) { return LLVMInt64Type(); }

/* int -> lltype */
CAMLprim LLVMTypeRef llvm_make_integer_type(value Width) {
  return LLVMCreateIntType(Int_val(Width));
}

/* lltype -> int */
CAMLprim value llvm_integer_bitwidth(LLVMTypeRef IntegerTy) {
  return Val_int(LLVMGetIntTypeWidth(IntegerTy));
}

/*--... Operations on real types ...........................................--*/

/* unit -> lltype */
CAMLprim LLVMTypeRef llvm_float_type(value Unit) {
  return LLVMFloatType();
}

/* unit -> lltype */
CAMLprim LLVMTypeRef llvm_double_type(value Unit) {
  return LLVMDoubleType();
}

/* unit -> lltype */
CAMLprim LLVMTypeRef llvm_x86fp80_type(value Unit) {
  return LLVMX86FP80Type();
}

/* unit -> lltype */
CAMLprim LLVMTypeRef llvm_fp128_type(value Unit) {
  return LLVMFP128Type();
}

/* unit -> lltype */
CAMLprim LLVMTypeRef llvm_ppc_fp128_type(value Unit) {
  return LLVMPPCFP128Type();
}

/*--... Operations on function types .......................................--*/

/* lltype -> lltype array -> bool -> lltype */
CAMLprim LLVMTypeRef llvm_make_function_type(LLVMTypeRef RetTy, value ParamTys,
                                             value IsVarArg) {
  return LLVMCreateFunctionType(RetTy, (LLVMTypeRef *) ParamTys,
                                Wosize_val(ParamTys),
                                Bool_val(IsVarArg));
}

/* lltype -> bool */
CAMLprim value llvm_is_var_arg(LLVMTypeRef FunTy) {
  return Val_bool(LLVMIsFunctionVarArg(FunTy));
}

/* lltype -> lltype */
CAMLprim LLVMTypeRef llvm_return_type(LLVMTypeRef FunTy) {
  return LLVMGetReturnType(FunTy);
}

/* lltype -> lltype array */
CAMLprim value llvm_param_types(LLVMTypeRef FunTy) {
  value Tys = alloc(LLVMCountParamTypes(FunTy), 0);
  LLVMGetParamTypes(FunTy, (LLVMTypeRef *) Tys);
  return Tys;
}

/*--... Operations on struct types .........................................--*/

/* lltype array -> bool -> lltype */
CAMLprim value llvm_make_struct_type(value ElementTypes, value Packed) {
  return (value) LLVMCreateStructType((LLVMTypeRef *) ElementTypes,
                                      Wosize_val(ElementTypes),
                                      Bool_val(Packed));
}

/* lltype -> lltype array */
CAMLprim value llvm_element_types(LLVMTypeRef StructTy) {
  value Tys = alloc(LLVMCountStructElementTypes(StructTy), 0);
  LLVMGetStructElementTypes(StructTy, (LLVMTypeRef *) Tys);
  return Tys;
}

/* lltype -> bool */
CAMLprim value llvm_is_packed(LLVMTypeRef StructTy) {
  return Val_bool(LLVMIsPackedStruct(StructTy));
}

/*--... Operations on array, pointer, and vector types .....................--*/

/* lltype -> int -> lltype */
CAMLprim value llvm_make_array_type(LLVMTypeRef ElementTy, value Count) {
  return (value) LLVMCreateArrayType(ElementTy, Int_val(Count));
}

/* lltype -> lltype */
CAMLprim LLVMTypeRef llvm_make_pointer_type(LLVMTypeRef ElementTy) {
  return LLVMCreatePointerType(ElementTy);
}

/* lltype -> int -> lltype */
CAMLprim LLVMTypeRef llvm_make_vector_type(LLVMTypeRef ElementTy, value Count) {
  return LLVMCreateVectorType(ElementTy, Int_val(Count));
}

/* lltype -> lltype */
CAMLprim LLVMTypeRef llvm_element_type(LLVMTypeRef Ty) {
  return LLVMGetElementType(Ty);
}

/* lltype -> int */
CAMLprim value llvm_array_length(LLVMTypeRef ArrayTy) {
  return Val_int(LLVMGetArrayLength(ArrayTy));
}

/* lltype -> int */
CAMLprim value llvm_vector_size(LLVMTypeRef VectorTy) {
  return Val_int(LLVMGetVectorSize(VectorTy));
}

/*--... Operations on other types ..........................................--*/

/* unit -> lltype */
CAMLprim LLVMTypeRef llvm_void_type (value Unit) { return LLVMVoidType();  }
CAMLprim LLVMTypeRef llvm_label_type(value Unit) { return LLVMLabelType(); }

/* unit -> lltype */
CAMLprim LLVMTypeRef llvm_make_opaque_type(value Unit) {
  return LLVMCreateOpaqueType();
}


/*===-- VALUES ------------------------------------------------------------===*/

/* llvalue -> lltype */
CAMLprim LLVMTypeRef llvm_type_of(LLVMValueRef Val) {
  return LLVMTypeOf(Val);
}

/* llvalue -> string */
CAMLprim value llvm_value_name(LLVMValueRef Val) {
  return copy_string(LLVMGetValueName(Val));
}

/* string -> llvalue -> unit */
CAMLprim value llvm_set_value_name(value Name, LLVMValueRef Val) {
  LLVMSetValueName(Val, String_val(Name));
  return Val_unit;
}

/* llvalue -> unit */
CAMLprim value llvm_dump_value(LLVMValueRef Val) {
  LLVMDumpValue(Val);
  return Val_unit;
}

/*--... Operations on constants of (mostly) any type .......................--*/

/* llvalue -> bool */
CAMLprim value llvm_is_constant(LLVMValueRef Val) {
  return Val_bool(LLVMIsConstant(Val));
}

/* llvalue -> bool */
CAMLprim value llvm_is_null(LLVMValueRef Val) {
  return Val_bool(LLVMIsNull(Val));
}

/* llvalue -> bool */
CAMLprim value llvm_is_undef(LLVMValueRef Val) {
  return Val_bool(LLVMIsUndef(Val));
}

/*--... Operations on scalar constants .....................................--*/

/* lltype -> int -> bool -> llvalue */
CAMLprim LLVMValueRef llvm_make_int_constant(LLVMTypeRef IntTy, value N,
                                             value SExt) {
  /* GCC warns if we use the ternary operator. */
  unsigned long long N2;
  if (Bool_val(SExt))
    N2 = (value) Int_val(N);
  else
    N2 = (mlsize_t) Int_val(N);
  
  return LLVMGetIntConstant(IntTy, N2, Bool_val(SExt));
}

/* lltype -> Int64.t -> bool -> llvalue */
CAMLprim LLVMValueRef llvm_make_int64_constant(LLVMTypeRef IntTy, value N,
                                               value SExt) {
  return LLVMGetIntConstant(IntTy, Int64_val(N), Bool_val(SExt));
}

/* lltype -> float -> llvalue */
CAMLprim LLVMValueRef llvm_make_real_constant(LLVMTypeRef RealTy, value N) {
  return LLVMGetRealConstant(RealTy, Double_val(N));
}

/*--... Operations on composite constants ..................................--*/

/* string -> bool -> llvalue */
CAMLprim LLVMValueRef llvm_make_string_constant(value Str, value NullTerminate) {
  return LLVMGetStringConstant(String_val(Str), string_length(Str),
                               Bool_val(NullTerminate) == 0);
}

/* lltype -> llvalue array -> llvalue */
CAMLprim LLVMValueRef llvm_make_array_constant(LLVMTypeRef ElementTy,
                                               value ElementVals) {
  return LLVMGetArrayConstant(ElementTy, (LLVMValueRef*) Op_val(ElementVals),
                              Wosize_val(ElementVals));
}

/* llvalue array -> bool -> llvalue */
CAMLprim LLVMValueRef llvm_make_struct_constant(value ElementVals,
                                                value Packed) {
  return LLVMGetStructConstant((LLVMValueRef *) Op_val(ElementVals),
                               Wosize_val(ElementVals), Bool_val(Packed));
}

/* llvalue array -> llvalue */
CAMLprim value llvm_make_vector_constant(value ElementVals) {
  return (value) LLVMGetVectorConstant((LLVMValueRef*) Op_val(ElementVals),
                                       Wosize_val(ElementVals));
}

/*--... Operations on global variables, functions, and aliases (globals) ...--*/

/* llvalue -> bool */
CAMLprim value llvm_is_declaration(LLVMValueRef Global) {
  return Val_bool(LLVMIsDeclaration(Global));
}

/* llvalue -> linkage */
CAMLprim value llvm_linkage(LLVMValueRef Global) {
  return Val_int(LLVMGetLinkage(Global));
}

/* linkage -> llvalue -> unit */
CAMLprim value llvm_set_linkage(value Linkage, LLVMValueRef Global) {
  LLVMSetLinkage(Global, Int_val(Linkage));
  return Val_unit;
}

/* llvalue -> string */
CAMLprim value llvm_section(LLVMValueRef Global) {
  return copy_string(LLVMGetSection(Global));
}

/* string -> llvalue -> unit */
CAMLprim value llvm_set_section(value Section, LLVMValueRef Global) {
  LLVMSetSection(Global, String_val(Section));
  return Val_unit;
}

/* llvalue -> visibility */
CAMLprim value llvm_visibility(LLVMValueRef Global) {
  return Val_int(LLVMGetVisibility(Global));
}

/* visibility -> llvalue -> unit */
CAMLprim value llvm_set_visibility(value Viz, LLVMValueRef Global) {
  LLVMSetVisibility(Global, Int_val(Viz));
  return Val_unit;
}

/* llvalue -> int */
CAMLprim value llvm_alignment(LLVMValueRef Global) {
  return Val_int(LLVMGetAlignment(Global));
}

/* int -> llvalue -> unit */
CAMLprim value llvm_set_alignment(value Bytes, LLVMValueRef Global) {
  LLVMSetAlignment(Global, Int_val(Bytes));
  return Val_unit;
}

/*--... Operations on global variables .....................................--*/

/* lltype -> string -> llmodule -> llvalue */
CAMLprim LLVMValueRef llvm_declare_global(LLVMTypeRef Ty, value Name,
                                          LLVMModuleRef M) {
  return LLVMAddGlobal(M, Ty, String_val(Name));
}

/* string -> llvalue -> llmodule -> llvalue */
CAMLprim LLVMValueRef llvm_define_global(value Name, LLVMValueRef Initializer,
                                         LLVMModuleRef M) {
  LLVMValueRef GlobalVar = LLVMAddGlobal(M, LLVMTypeOf(Initializer),
                                         String_val(Name));
  LLVMSetInitializer(GlobalVar, Initializer);
  return GlobalVar;
}

/* llvalue -> unit */
CAMLprim value llvm_delete_global(LLVMValueRef GlobalVar) {
  LLVMDeleteGlobal(GlobalVar);
  return Val_unit;
}

/* llvalue -> llvalue -> unit */
CAMLprim value llvm_set_initializer(LLVMValueRef ConstantVal,
                                    LLVMValueRef GlobalVar) {
  LLVMSetInitializer(GlobalVar, ConstantVal);
  return Val_unit;
}

/* llvalue -> unit */
CAMLprim value llvm_remove_initializer(LLVMValueRef GlobalVar) {
  LLVMSetInitializer(GlobalVar, NULL);
  return Val_unit;
}

/* llvalue -> bool */
CAMLprim value llvm_is_thread_local(LLVMValueRef GlobalVar) {
  return Val_bool(LLVMIsThreadLocal(GlobalVar));
}

/* bool -> llvalue -> unit */
CAMLprim value llvm_set_thread_local(value IsThreadLocal,
                                     LLVMValueRef GlobalVar) {
  LLVMSetThreadLocal(GlobalVar, Bool_val(IsThreadLocal));
  return Val_unit;
}

/*--... Operations on functions ............................................--*/

/* string -> lltype -> llmodule -> llvalue */
CAMLprim LLVMValueRef llvm_declare_function(value Name, LLVMTypeRef Ty,
                                            LLVMModuleRef M) {
  return LLVMAddFunction(M, String_val(Name), Ty);
}

/* string -> lltype -> llmodule -> llvalue */
CAMLprim LLVMValueRef llvm_define_function(value Name, LLVMTypeRef Ty,
                                           LLVMModuleRef M) {
  LLVMValueRef Fn = LLVMAddFunction(M, String_val(Name), Ty);
  LLVMAppendBasicBlock(Fn, "entry");
  return Fn;
}

/* llvalue -> unit */
CAMLprim value llvm_delete_function(LLVMValueRef Fn) {
  LLVMDeleteFunction(Fn);
  return Val_unit;
}

/* llvalue -> int -> llvalue */
CAMLprim LLVMValueRef llvm_param(LLVMValueRef Fn, value Index) {
  return LLVMGetParam(Fn, Int_val(Index));
}

/* llvalue -> int -> llvalue */
CAMLprim value llvm_params(LLVMValueRef Fn, value Index) {
  value Params = alloc(LLVMCountParams(Fn), 0);
  LLVMGetParams(Fn, (LLVMValueRef *) Op_val(Params));
  return Params;
}

/* llvalue -> bool */
CAMLprim value llvm_is_intrinsic(LLVMValueRef Fn) {
  return Val_bool(LLVMGetIntrinsicID(Fn));
}

/* llvalue -> int */
CAMLprim value llvm_function_call_conv(LLVMValueRef Fn) {
  return Val_int(LLVMGetFunctionCallConv(Fn));
}

/* int -> llvalue -> unit */
CAMLprim value llvm_set_function_call_conv(value Id, LLVMValueRef Fn) {
  LLVMSetFunctionCallConv(Fn, Int_val(Id));
  return Val_unit;
}

/*--... Operations on basic blocks .........................................--*/

/* llvalue -> llbasicblock array */
CAMLprim value llvm_basic_blocks(LLVMValueRef Fn) {
  value MLArray = alloc(LLVMCountBasicBlocks(Fn), 0);
  LLVMGetBasicBlocks(Fn, (LLVMBasicBlockRef *) Op_val(MLArray));
  return MLArray;
}

/* llbasicblock -> unit */
CAMLprim value llvm_delete_block(LLVMBasicBlockRef BB) {
  LLVMDeleteBasicBlock(BB);
  return Val_unit;
}

/* string -> llvalue -> llbasicblock */
CAMLprim LLVMBasicBlockRef llvm_append_block(value Name, LLVMValueRef Fn) {
  return LLVMAppendBasicBlock(Fn, String_val(Name));
}

/* string -> llbasicblock -> llbasicblock */
CAMLprim LLVMBasicBlockRef llvm_insert_block(value Name, LLVMBasicBlockRef BB) {
  return LLVMInsertBasicBlock(BB, String_val(Name));
}

/* llvalue -> bool */
CAMLprim value llvm_value_is_block(LLVMValueRef Val) {
  return Val_bool(LLVMValueIsBasicBlock(Val));
}


/*===-- Instruction builders ----------------------------------------------===*/

#define Builder_val(v)  (*(LLVMBuilderRef *)(Data_custom_val(v)))

void llvm_finalize_builder(value B) {
  LLVMDisposeBuilder(Builder_val(B));
}

static struct custom_operations builder_ops = {
  (char *) "LLVMBuilder",
  llvm_finalize_builder,
  custom_compare_default,
  custom_hash_default,
  custom_serialize_default,
  custom_deserialize_default
};

/* llvalue -> llbuilder */
CAMLprim value llvm_builder_before(LLVMValueRef Inst) {
  value V;
  LLVMBuilderRef B = LLVMCreateBuilder();
  LLVMPositionBuilderBefore(B, Inst);
  V = alloc_custom(&builder_ops, sizeof(LLVMBuilderRef), 0, 1);
  Builder_val(V) = B;
  return V;
}

/* llbasicblock -> llbuilder */
CAMLprim value llvm_builder_at_end(LLVMBasicBlockRef BB) {
  value V;
  LLVMBuilderRef B = LLVMCreateBuilder();
  LLVMPositionBuilderAtEnd(B, BB);
  V = alloc_custom(&builder_ops, sizeof(LLVMBuilderRef), 0, 1);
  Builder_val(V) = B;
  return V;
}

/* llvalue -> llbuilder -> unit */
CAMLprim value llvm_position_before(LLVMValueRef Inst, value B) {
  LLVMPositionBuilderBefore(Builder_val(B), Inst);
  return Val_unit;
}

/* llbasicblock -> llbuilder -> unit */
CAMLprim value llvm_position_at_end(LLVMBasicBlockRef BB, value B) {
  LLVMPositionBuilderAtEnd(Builder_val(B), BB);
  return Val_unit;
}

/*--... Terminators ........................................................--*/

/* llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_ret_void(value B) {
  return LLVMBuildRetVoid(Builder_val(B));
}

/* llvalue -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_ret(LLVMValueRef Val, value B) {
  return LLVMBuildRet(Builder_val(B), Val);
}

/* llbasicblock -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_br(LLVMBasicBlockRef BB, value B) {
  return LLVMBuildBr(Builder_val(B), BB);
}

/* llvalue -> llbasicblock -> llbasicblock -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_cond_br(LLVMValueRef If,
                                         LLVMBasicBlockRef Then,
                                         LLVMBasicBlockRef Else,
                                         value B) {
  return LLVMBuildCondBr(Builder_val(B), If, Then, Else);
}

/* llvalue -> llbasicblock -> int -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_switch(LLVMValueRef Of,
                                        LLVMBasicBlockRef Else,
                                        value EstimatedCount,
                                        value B) {
  return LLVMBuildSwitch(Builder_val(B), Of, Else, Int_val(EstimatedCount));
}

/* llvalue -> llvalue array -> llbasicblock -> llbasicblock -> string ->
   llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_invoke_nat(LLVMValueRef Fn, value Args,
                                            LLVMBasicBlockRef Then,
                                            LLVMBasicBlockRef Catch,
                                            value Name, value B) {
  return LLVMBuildInvoke(Builder_val(B), Fn, (LLVMValueRef *) Op_val(Args),
                         Wosize_val(Args), Then, Catch, String_val(Name));
}

/* llvalue -> llvalue array -> llbasicblock -> llbasicblock -> string ->
   llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_invoke_bc(value Args[], int NumArgs) {
  return llvm_build_invoke_nat((LLVMValueRef) Args[0], Args[1],
                               (LLVMBasicBlockRef) Args[2],
                               (LLVMBasicBlockRef) Args[3],
                               Args[4], Args[5]);
}

/* llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_unwind(value B) {
  return LLVMBuildUnwind(Builder_val(B));
}

/* llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_unreachable(value B) {
  return LLVMBuildUnreachable(Builder_val(B));
}

/*--... Arithmetic .........................................................--*/

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_add(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildAdd(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_sub(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildSub(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_mul(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildMul(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_udiv(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildUDiv(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_sdiv(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildSDiv(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fdiv(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildFDiv(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_urem(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildURem(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_srem(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildSRem(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_frem(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildFRem(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_shl(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildShl(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_lshr(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildLShr(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_ashr(LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildAShr(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_and(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildAnd(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_or(LLVMValueRef LHS, LLVMValueRef RHS,
                                    value Name, value B) {
  return LLVMBuildOr(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_xor(LLVMValueRef LHS, LLVMValueRef RHS,
                                     value Name, value B) {
  return LLVMBuildXor(Builder_val(B), LHS, RHS, String_val(Name));
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_neg(LLVMValueRef X,
                                     value Name, value B) {
  return LLVMBuildNeg(Builder_val(B), X, String_val(Name));
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_not(LLVMValueRef X,
                                     value Name, value B) {
  return LLVMBuildNot(Builder_val(B), X, String_val(Name));
}

/*--... Memory .............................................................--*/

/* lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_malloc(LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildMalloc(Builder_val(B), Ty, String_val(Name));
}

/* lltype -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_array_malloc(LLVMTypeRef Ty, LLVMValueRef Size,
                                              value Name, value B) {
  return LLVMBuildArrayMalloc(Builder_val(B), Ty, Size, String_val(Name));
}

/* lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_alloca(LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildAlloca(Builder_val(B), Ty, String_val(Name));
}

/* lltype -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_array_alloca(LLVMTypeRef Ty, LLVMValueRef Size,
                                              value Name, value B) {
  return LLVMBuildArrayAlloca(Builder_val(B), Ty, Size, String_val(Name));
}

/* llvalue -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_free(LLVMValueRef Pointer, value B) {
  return LLVMBuildFree(Builder_val(B), Pointer);
}

/* llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_load(LLVMValueRef Pointer,
                                      value Name, value B) {
  return LLVMBuildLoad(Builder_val(B), Pointer, String_val(Name));
}

/* llvalue -> llvalue -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_store(LLVMValueRef Value, LLVMValueRef Pointer,
                                       value B) {
  return LLVMBuildStore(Builder_val(B), Value, Pointer);
}

/* llvalue -> llvalue array -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_gep(LLVMValueRef Pointer, value Indices,
                                     value Name, value B) {
  return LLVMBuildGEP(Builder_val(B), Pointer,
                      (LLVMValueRef *) Op_val(Indices), Wosize_val(Indices),
                      String_val(Name));
}

/*--... Casts ..............................................................--*/

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_trunc(LLVMValueRef X, LLVMTypeRef Ty,
                                       value Name, value B) {
  return LLVMBuildTrunc(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_zext(LLVMValueRef X, LLVMTypeRef Ty,
                                      value Name, value B) {
  return LLVMBuildZExt(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_sext(LLVMValueRef X, LLVMTypeRef Ty,
                                      value Name, value B) {
  return LLVMBuildSExt(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fptoui(LLVMValueRef X, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildFPToUI(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fptosi(LLVMValueRef X, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildFPToSI(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_uitofp(LLVMValueRef X, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildUIToFP(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_sitofp(LLVMValueRef X, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildSIToFP(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fptrunc(LLVMValueRef X, LLVMTypeRef Ty,
                                         value Name, value B) {
  return LLVMBuildFPTrunc(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fpext(LLVMValueRef X, LLVMTypeRef Ty,
                                       value Name, value B) {
  return LLVMBuildFPExt(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_prttoint(LLVMValueRef X, LLVMTypeRef Ty,
                                          value Name, value B) {
  return LLVMBuildPtrToInt(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_inttoptr(LLVMValueRef X, LLVMTypeRef Ty,
                                          value Name, value B) {
  return LLVMBuildIntToPtr(Builder_val(B), X, Ty, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_bitcast(LLVMValueRef X, LLVMTypeRef Ty,
                                         value Name, value B) {
  return LLVMBuildBitCast(Builder_val(B), X, Ty, String_val(Name));
}

/*--... Comparisons ........................................................--*/

/* int_predicate -> llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_icmp(value Pred,
                                      LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildICmp(Builder_val(B), Int_val(Pred) + LLVMIntEQ, LHS, RHS,
                       String_val(Name));
}

/* real_predicate -> llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_fcmp(value Pred,
                                      LLVMValueRef LHS, LLVMValueRef RHS,
                                      value Name, value B) {
  return LLVMBuildFCmp(Builder_val(B), Int_val(Pred), LHS, RHS,
                       String_val(Name));
}

/*--... Miscellaneous instructions .........................................--*/

/* lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_phi(LLVMTypeRef Ty,
                                     value Name, value B) {
  return LLVMBuildPhi(Builder_val(B), Ty, String_val(Name));
}

/* llvalue -> llvalue array -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_call(LLVMValueRef Fn, value Params,
                                      value Name, value B) {
  return LLVMBuildCall(Builder_val(B), Fn, (LLVMValueRef *) Op_val(Params),
                       Wosize_val(Params), String_val(Name));
}

/* llvalue -> llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_select(LLVMValueRef If,
                                        LLVMValueRef Then, LLVMValueRef Else,
                                        value Name, value B) {
  return LLVMBuildSelect(Builder_val(B), If, Then, Else, String_val(Name));
}

/* llvalue -> lltype -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_va_arg(LLVMValueRef List, LLVMTypeRef Ty,
                                        value Name, value B) {
  return LLVMBuildVAArg(Builder_val(B), List, Ty, String_val(Name));
}

/* llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_extractelement(LLVMValueRef Vec,
                                                LLVMValueRef Idx,
                                                value Name, value B) {
  return LLVMBuildExtractElement(Builder_val(B), Vec, Idx, String_val(Name));
}

/* llvalue -> llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_insertelement(LLVMValueRef Vec,
                                               LLVMValueRef Element,
                                               LLVMValueRef Idx,
                                               value Name, value B) {
  return LLVMBuildInsertElement(Builder_val(B), Vec, Element, Idx, 
                                String_val(Name));
}

/* llvalue -> llvalue -> llvalue -> string -> llbuilder -> llvalue */
CAMLprim LLVMValueRef llvm_build_shufflevector(LLVMValueRef V1, LLVMValueRef V2,
                                               LLVMValueRef Mask,
                                               value Name, value B) {
  return LLVMBuildShuffleVector(Builder_val(B), V1, V2, Mask, String_val(Name));
}

