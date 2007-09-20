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
#include "caml/mlvalues.h"
#include "caml/memory.h"
#include "llvm/Config/config.h" 
#include <stdio.h>
#ifdef HAVE_ALLOCA_H
#include <alloca.h>
#endif


/*===-- Modules -----------------------------------------------------------===*/

/* string -> llmodule */
CAMLprim value llvm_create_module(value ModuleID) {
  return (value) LLVMModuleCreateWithName(String_val(ModuleID));
}

/* llmodule -> unit */
CAMLprim value llvm_dispose_module(value M) {
  LLVMDisposeModule((LLVMModuleRef) M);
  return Val_unit;
}

/* string -> lltype -> llmodule -> bool */
CAMLprim value llvm_add_type_name(value Name, value Ty, value M) {
  int res = LLVMAddTypeName((LLVMModuleRef) M,
                            String_val(Name), (LLVMTypeRef) Ty);
  return Val_bool(res == 0);
}


/*===-- Types -------------------------------------------------------------===*/

/* lltype -> type_kind */
CAMLprim value llvm_classify_type(value Ty) {
  return Val_int(LLVMGetTypeKind((LLVMTypeRef) Ty));
}

/* lltype -> lltype -> unit */
CAMLprim value llvm_refine_abstract_type(value ConcreteTy, value AbstractTy) {
  LLVMRefineAbstractType((LLVMTypeRef) AbstractTy, (LLVMTypeRef) ConcreteTy);
  return (value) Val_unit;
}

/*--... Operations on integer types ........................................--*/

/* unit -> lltype */
CAMLprim value llvm_i1_type (value Unit) { return (value) LLVMInt1Type();  }
CAMLprim value llvm_i8_type (value Unit) { return (value) LLVMInt8Type();  }
CAMLprim value llvm_i16_type(value Unit) { return (value) LLVMInt16Type(); }
CAMLprim value llvm_i32_type(value Unit) { return (value) LLVMInt32Type(); }
CAMLprim value llvm_i64_type(value Unit) { return (value) LLVMInt64Type(); }

/* int -> lltype */
CAMLprim value llvm_make_integer_type(value Width) {
  return (value) LLVMCreateIntegerType(Int_val(Width));
}

/* lltype -> int */
CAMLprim value llvm_integer_bitwidth(value IntegerTy) {
  return Val_int(LLVMGetIntegerTypeWidth((LLVMTypeRef) IntegerTy));
}

/*--... Operations on real types ...........................................--*/

/* unit -> lltype */
CAMLprim value llvm_float_type(value Unit) {
  return (value) LLVMFloatType();
}

/* unit -> lltype */
CAMLprim value llvm_double_type(value Unit) {
  return (value) LLVMDoubleType();
}

/* unit -> lltype */
CAMLprim value llvm_x86fp80_type(value Unit) {
  return (value) LLVMX86FP80Type();
}

/* unit -> lltype */
CAMLprim value llvm_fp128_type(value Unit) {
  return (value) LLVMFP128Type();
}

/* unit -> lltype */
CAMLprim value llvm_ppc_fp128_type(value Unit) {
  return (value) LLVMPPCFP128Type();
}

/*--... Operations on function types .......................................--*/

/* lltype -> lltype array -> bool -> lltype */
CAMLprim value llvm_make_function_type(value RetTy, value ParamTys,
                                       value IsVarArg) {
  return (value) LLVMCreateFunctionType((LLVMTypeRef) RetTy,
                                        (LLVMTypeRef *) ParamTys,
                                        Wosize_val(ParamTys),
                                        Bool_val(IsVarArg));
}

/* lltype -> bool */
CAMLprim value llvm_is_var_arg(value FunTy) {
  return Val_bool(LLVMIsFunctionVarArg((LLVMTypeRef) FunTy));
}

/* lltype -> lltype */
CAMLprim value llvm_return_type(value FunTy) {
  return (value) LLVMGetFunctionReturnType((LLVMTypeRef) FunTy);
}

/* lltype -> lltype array */
CAMLprim value llvm_param_types(value FunTy) {
  unsigned Count = LLVMGetFunctionParamCount((LLVMTypeRef) FunTy);
  LLVMTypeRef *FunTys = alloca(Count * sizeof(LLVMTypeRef));
  
  /* copy into an ocaml array */
  unsigned i;
  value ParamTys = caml_alloc(Count, 0);
  
  LLVMGetFunctionParamTypes((LLVMTypeRef) FunTy, FunTys);
  for (i = 0; i != Count; ++i)
    Store_field(ParamTys, i, (value) FunTys[i]);
  
  return ParamTys;
}

/*--... Operations on struct types .........................................--*/

/* lltype array -> bool -> lltype */
CAMLprim value llvm_make_struct_type(value ElementTypes, value Packed) {
  return (value) LLVMCreateStructType((LLVMTypeRef *) ElementTypes,
                                      Wosize_val(ElementTypes),
                                      Bool_val(Packed));
}

/* lltype -> lltype array */
CAMLprim value llvm_element_types(value StructTy) {
  unsigned Count = LLVMGetStructElementCount((LLVMTypeRef) StructTy);
  LLVMTypeRef *Tys = alloca(Count * sizeof(LLVMTypeRef));
  
  /* copy into an ocaml array */
  unsigned i;
  value ElementTys = caml_alloc(Count, 0);
  
  LLVMGetStructElementTypes((LLVMTypeRef) StructTy, Tys);
  for (i = 0; i != Count; ++i)
    Store_field(ElementTys, i, (value) Tys[i]);
  
  return ElementTys;
}

CAMLprim value llvm_is_packed(value StructTy) {
  return Val_bool(LLVMIsPackedStruct((LLVMTypeRef) StructTy));
}

/*--... Operations on array, pointer, and vector types .....................--*/

/* lltype -> int -> lltype */
CAMLprim value llvm_make_array_type(value ElementTy, value Count) {
  return (value) LLVMCreateArrayType((LLVMTypeRef) ElementTy, Int_val(Count));
}

/* lltype -> lltype */
CAMLprim value llvm_make_pointer_type(value ElementTy) {
  return (value) LLVMCreatePointerType((LLVMTypeRef) ElementTy);
}

/* lltype -> int -> lltype */
CAMLprim value llvm_make_vector_type(value ElementTy, value Count) {
  return (value) LLVMCreateVectorType((LLVMTypeRef) ElementTy, Int_val(Count));
}

/* lltype -> lltype */
CAMLprim value llvm_element_type(value Ty) {
  return (value) LLVMGetElementType((LLVMTypeRef) Ty);
}

/* lltype -> int */
CAMLprim value llvm_array_length(value ArrayTy) {
  return Val_int(LLVMGetArrayLength((LLVMTypeRef) ArrayTy));
}

/* lltype -> int */
CAMLprim value llvm_vector_size(value VectorTy) {
  return Val_int(LLVMGetVectorSize((LLVMTypeRef) VectorTy));
}

/*--... Operations on other types ..........................................--*/

/* unit -> lltype */
CAMLprim value llvm_void_type (value Unit) { return (value) LLVMVoidType();  }
CAMLprim value llvm_label_type(value Unit) { return (value) LLVMLabelType(); }

/* unit -> lltype */
CAMLprim value llvm_make_opaque_type(value Unit) {
  return (value) LLVMCreateOpaqueType();
}


/*===-- VALUES ------------------------------------------------------------===*/

/* llvalue -> lltype */
CAMLprim value llvm_type_of(value Val) {
  return (value) LLVMGetTypeOfValue((LLVMValueRef) Val);
}

/* llvalue -> string */
CAMLprim value llvm_value_name(value Val) {
  return caml_copy_string(LLVMGetValueName((LLVMValueRef) Val));
}

/* string -> llvalue -> unit */
CAMLprim value llvm_set_value_name(value Name, value Val) {
  LLVMSetValueName((LLVMValueRef) Val, String_val(Name));
  return Val_unit;
}

/*--... Operations on constants of (mostly) any type .......................--*/

/* lltype -> llvalue */
CAMLprim value llvm_make_null(value Ty) {
  return (value) LLVMGetNull((LLVMTypeRef) Ty);
}

/* lltype -> llvalue */
CAMLprim value llvm_make_all_ones(value Ty) {
  return (value) LLVMGetAllOnes((LLVMTypeRef) Ty);
}

/* lltype -> llvalue */
CAMLprim value llvm_make_undef(value Ty) {
  return (value) LLVMGetUndef((LLVMTypeRef) Ty);
}

/* llvalue -> bool */
CAMLprim value llvm_is_constant(value Ty) {
  return Val_bool(LLVMIsConstant((LLVMValueRef) Ty));
}

/* llvalue -> bool */
CAMLprim value llvm_is_null(value Val) {
  return Val_bool(LLVMIsNull((LLVMValueRef) Val));
}

/* llvalue -> bool */
CAMLprim value llvm_is_undef(value Ty) {
  return Val_bool(LLVMIsUndef((LLVMValueRef) Ty));
}

/*--... Operations on scalar constants .....................................--*/

/* lltype -> int -> bool -> llvalue */
CAMLprim value llvm_make_int_constant(value IntTy, value N, value SExt) {
  /* GCC warns if we use the ternary operator. */
  unsigned long long N2;
  if (Bool_val(SExt))
    N2 = (value) Int_val(N);
  else
    N2 = (mlsize_t) Int_val(N);
  
  return (value) LLVMGetIntConstant((LLVMTypeRef) IntTy, N2, Bool_val(SExt));
}

/* lltype -> Int64.t -> bool -> llvalue */
CAMLprim value llvm_make_int64_constant(value IntTy, value N, value SExt) {
  return (value) LLVMGetIntConstant((LLVMTypeRef) IntTy, Int64_val(N),
                                    Bool_val(SExt));
}

/* lltype -> float -> llvalue */
CAMLprim value llvm_make_real_constant(value RealTy, value N) {
  return (value) LLVMGetRealConstant((LLVMTypeRef) RealTy, Double_val(N));
}

/*--... Operations on composite constants ..................................--*/

/* string -> bool -> llvalue */
CAMLprim value llvm_make_string_constant(value Str, value NullTerminate) {
  return (value) LLVMGetStringConstant(String_val(Str),
                                       caml_string_length(Str),
                                       Bool_val(NullTerminate) == 0);
}

/* lltype -> llvalue array -> llvalue */
CAMLprim value llvm_make_array_constant(value ElementTy, value ElementVals) {
  return (value) LLVMGetArrayConstant((LLVMTypeRef) ElementTy,
                                      (LLVMValueRef*) Op_val(ElementVals),
                                      Wosize_val(ElementVals));
}

/* llvalue array -> bool -> llvalue */
CAMLprim value llvm_make_struct_constant(value ElementVals, value Packed) {
  return (value) LLVMGetStructConstant((LLVMValueRef*) Op_val(ElementVals),
                                       Wosize_val(ElementVals),
                                       Bool_val(Packed));
}

/* llvalue array -> llvalue */
CAMLprim value llvm_make_vector_constant(value ElementVals) {
  return (value) LLVMGetVectorConstant((LLVMValueRef*) Op_val(ElementVals),
                                       Wosize_val(ElementVals));
}

/*--... Operations on global variables, functions, and aliases (globals) ...--*/

/* llvalue -> bool */
CAMLprim value llvm_is_declaration(value Global) {
  return Val_bool(LLVMIsDeclaration((LLVMValueRef) Global));
}

/* llvalue -> linkage */
CAMLprim value llvm_linkage(value Global) {
  return Val_int(LLVMGetLinkage((LLVMValueRef) Global));
}

/* linkage -> llvalue -> unit */
CAMLprim value llvm_set_linkage(value Linkage, value Global) {
  LLVMSetLinkage((LLVMValueRef) Global, Int_val(Linkage));
  return Val_unit;
}

/* llvalue -> string */
CAMLprim value llvm_section(value Global) {
  return caml_copy_string(LLVMGetSection((LLVMValueRef) Global));
}

/* string -> llvalue -> unit */
CAMLprim value llvm_set_section(value Section, value Global) {
  LLVMSetSection((LLVMValueRef) Global, String_val(Section));
  return Val_unit;
}

/* llvalue -> visibility */
CAMLprim value llvm_visibility(value Global) {
  return Val_int(LLVMGetVisibility((LLVMValueRef) Global));
}

/* visibility -> llvalue -> unit */
CAMLprim value llvm_set_visibility(value Viz, value Global) {
  LLVMSetVisibility((LLVMValueRef) Global, Int_val(Viz));
  return Val_unit;
}

/* llvalue -> int */
CAMLprim value llvm_alignment(value Global) {
  return Val_int(LLVMGetAlignment((LLVMValueRef) Global));
}

/* int -> llvalue -> unit */
CAMLprim value llvm_set_alignment(value Bytes, value Global) {
  LLVMSetAlignment((LLVMValueRef) Global, Int_val(Bytes));
  return Val_unit;
}

/*--... Operations on global variables .....................................--*/

/* lltype -> string -> llmodule -> llvalue */
CAMLprim value llvm_add_global(value Ty, value Name, value M) {
  return (value) LLVMAddGlobal((LLVMModuleRef) M,
                               (LLVMTypeRef) Ty, String_val(Name));
}

/* lltype -> string -> llmodule -> llvalue */
CAMLprim value llvm_declare_global(value Ty, value Name, value M) {
  return (value) LLVMAddGlobal((LLVMModuleRef) M,
                               (LLVMTypeRef) Ty, String_val(Name));
}

/* string -> llvalue -> llmodule -> llvalue */
CAMLprim value llvm_define_global(value Name, value ConstantVal, value M) {
  LLVMValueRef Initializer = (LLVMValueRef) ConstantVal;
  LLVMValueRef GlobalVar = LLVMAddGlobal((LLVMModuleRef) M,
                                         LLVMGetTypeOfValue(Initializer),
                                         String_val(Name));
  LLVMSetInitializer(GlobalVar, Initializer);
  return (value) GlobalVar;
}

/* llvalue -> unit */
CAMLprim value llvm_delete_global(value GlobalVar) {
  LLVMDeleteGlobal((LLVMValueRef) GlobalVar);
  return Val_unit;
}

/* llvalue -> llvalue */
CAMLprim value llvm_global_initializer(value GlobalVar) {
  return (value) LLVMGetInitializer((LLVMValueRef) GlobalVar);
}

/* llvalue -> llvalue -> unit */
CAMLprim value llvm_set_initializer(value ConstantVal, value GlobalVar) {
  LLVMSetInitializer((LLVMValueRef) GlobalVar, (LLVMValueRef) ConstantVal);
  return Val_unit;
}

/* llvalue -> unit */
CAMLprim value llvm_remove_initializer(value GlobalVar) {
  LLVMSetInitializer((LLVMValueRef) GlobalVar, NULL);
  return Val_unit;
}

/* llvalue -> bool */
CAMLprim value llvm_is_thread_local(value GlobalVar) {
  return Val_bool(LLVMIsThreadLocal((LLVMValueRef) GlobalVar));
}

/* bool -> llvalue -> unit */
CAMLprim value llvm_set_thread_local(value IsThreadLocal, value GlobalVar) {
  LLVMSetThreadLocal((LLVMValueRef) GlobalVar, Bool_val(IsThreadLocal));
  return Val_unit;
}
