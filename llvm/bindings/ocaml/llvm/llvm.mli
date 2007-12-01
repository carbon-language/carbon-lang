(*===-- tools/ml/llvm.ml - LLVM Ocaml Interface ---------------------------===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by Gordon Henriksen and is distributed under the
 * University of Illinois Open Source License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===
 *
 * This interface provides an ocaml API for the LLVM intermediate
 * representation, the classes in the VMCore library.
 *
 *===----------------------------------------------------------------------===*)


(* These abstract types correlate directly to the LLVM VMCore classes. *)

(** The top-level container for all other LLVM Intermediate Representation (IR)
    objects. See the [llvm::Module] class. **)
type llmodule

(** Each value in the LLVM IR has a type, an instance of [lltype]. See the
    [llvm::Type] class. **)
type lltype 

(** When building recursive types using [refine_type], [lltype] values may
    become invalid; use [lltypehandle] to resolve this problem. See the
    [llvm::AbstractTypeHolder] class. **)
type lltypehandle

(** Any value in the LLVM IR. Functions, instructions, global variables,
    constants, and much more are all [llvalues]. See the [llvm::Value] class.
    This type covers a wide range of subclasses. **)
type llvalue

(** A basic block in LLVM IR. See the [llvm::BasicBlock] class. **)
type llbasicblock

(** Used to generate instructions in the LLVM IR. See the [llvm::LLVMBuilder]
    class. **)
type llbuilder

(** The kind of an [lltype], the result of [classify_type ty]. See the 
    [llvm::Type::TypeID] enumeration. **)
type type_kind =
  Void_type
| Float_type
| Double_type
| X86fp80_type
| Fp128_type
| Ppc_fp128_type
| Label_type
| Integer_type
| Function_type
| Struct_type
| Array_type
| Pointer_type
| Opaque_type
| Vector_type

(** The linkage of a global value, accessed with [linkage gv] and
    [set_linkage l gv]. See [llvm::GlobalValue::LinkageTypes]. **)
type linkage =
  External_linkage
| Link_once_linkage
| Weak_linkage
| Appending_linkage
| Internal_linkage
| Dllimport_linkage
| Dllexport_linkage
| External_weak_linkage
| Ghost_linkage

(** The linker visibility of a global value, accessed with [visibility gv] and
    [set_visibility v gv]. See [llvm::GlobalValue::VisibilityTypes]. **)
type visibility =
  Default_visibility
| Hidden_visibility
| Protected_visibility

(* The following calling convention values may be accessed with
   [function_call_conv f] and [set_function_call_conv conv f]. Calling
   conventions are open-ended. *)
val ccc : int             (** [ccc] is the C calling convention. **)
val fastcc : int          (** [fastcc] is the calling convention to allow LLVM
                              maximum optimization opportunities. Use only with
                              internal linkage. **)
val coldcc : int          (** [coldcc] is the calling convention for
                              callee-save. **)
val x86_stdcallcc : int   (** [x86_stdcallcc] is the familiar stdcall calling
                              convention from C. **)
val x86_fastcallcc : int  (** [x86_fastcallcc] is the familiar fastcall calling
                              convention from C. **)

(** The predicate for an integer comparison ([icmp]) instruction.
    See the [llvm::ICmpInst::Predicate] enumeration. **)
type int_predicate =
  Icmp_eq
| Icmp_ne
| Icmp_ugt
| Icmp_uge
| Icmp_ult
| Icmp_ule
| Icmp_sgt
| Icmp_sge
| Icmp_slt
| Icmp_sle

(** The predicate for a floating-point comparison ([fcmp]) instruction.
    See the [llvm::FCmpInst::Predicate] enumeration. **)
type real_predicate =
  Fcmp_false
| Fcmp_oeq
| Fcmp_ogt
| Fcmp_oge
| Fcmp_olt
| Fcmp_ole
| Fcmp_one
| Fcmp_ord
| Fcmp_uno
| Fcmp_ueq
| Fcmp_ugt
| Fcmp_uge
| Fcmp_ult
| Fcmp_ule
| Fcmp_une
| Fcmp_true


(*===-- Modules -----------------------------------------------------------===*)

(** [create_module id] creates a module with the supplied module ID. Modules are
    not garbage collected; it is mandatory to call [dispose_module m] to free
    memory. See the constructor [llvm::Module::Module]. *)
external create_module : string -> llmodule = "llvm_create_module"

(** [dispose_module m] destroys a module [m] and all of the IR objects it
    contained. All references to subordinate objects are invalidated;
    referencing them will invoke undefined behavior. See the destructor
    [llvm::Module::~Module]. **)
external dispose_module : llmodule -> unit = "llvm_dispose_module"

(** [define_type_name name ty m] adds a named type to the module's symbol table.
    Returns [true] if successful. If such a name already exists, then no entry
    is added and [false] is returned. See the [llvm::Module::addTypeName]
    method. **)
external define_type_name : string -> lltype -> llmodule -> bool
                          = "llvm_add_type_name"

(** [delete_type_name name] removes a type name from the module's symbol
    table. *)
external delete_type_name : string -> llmodule -> unit
                          = "llvm_delete_type_name"


(*===-- Types -------------------------------------------------------------===*)

(** [classify_type ty] returns the [type_kind] corresponding to the type [ty].
    See the method [llvm::Type::getTypeID]. **)
external classify_type : lltype -> type_kind = "llvm_classify_type"

(** [string_of_lltype ty] returns a string describing the type [ty]. **)
val string_of_lltype : lltype -> string

(*--... Operations on integer types ........................................--*)

(** The 1-bit integer type. See [llvm::Type::Int1Ty]. **)
val i1_type : lltype 

(** The 8-bit integer type. See [llvm::Type::Int8Ty]. **)
val i8_type : lltype 

(** The 16-bit integer type. See [llvm::Type::Int16Ty]. **)
val i16_type : lltype

(** The 32-bit integer type. See [llvm::Type::Int32Ty]. **)
val i32_type : lltype

(** The 64-bit integer type. See [llvm::Type::Int64Ty]. **)
val i64_type : lltype

(** [integer_type n] returns an integer type of bitwidth [n].
    See the method [llvm::IntegerType::get]. **)
external integer_type : int -> lltype = "llvm_integer_type"

(** [integer_bitwidth ty] returns the number of bits in the integer type [ty]..
    See the method [llvm::IntegerType::getBitWidth]. **)
external integer_bitwidth : lltype -> int = "llvm_integer_bitwidth"

(*--... Operations on real types ...........................................--*)

(** The IEEE 32-bit floating point type. See [llvm::Type::FloatTy]. **)
val float_type : lltype

(** The IEEE 64-bit floating point type. See [llvm::Type::DoubleTy]. **)
val double_type : lltype

(** The x87 80-bit floating point type. See [llvm::Type::X86_FP80Ty]. **)
val x86fp80_type : lltype

(** The IEEE 128-bit floating point type. See [llvm::Type::FP128Ty]. **)
val fp128_type : lltype

(** The PowerPC 128-bit floating point type. See [llvm::Type::PPC_FP128Ty]. **)
val ppc_fp128_type : lltype

(*--... Operations on function types .......................................--*)

(** [function_type ret_ty param_tys] returns the function type returning
    [ret_ty] and taking [param_tys] as parameters.
    See the method [llvm::FunctionType::get]. **)
external function_type : lltype -> lltype array -> lltype = "llvm_function_type"

(** [va_arg_function_type ret_ty param_tys] is just like
    [function_type ret_ty param_tys] except that it returns the function type
    which also takes a variable number of arguments.
    See the method [llvm::FunctionType::get]. **)
external var_arg_function_type : lltype -> lltype array -> lltype
                               = "llvm_var_arg_function_type"

(** [is_var_arg fty] returns [true] if [fty] is a varargs function type, [false]
    otherwise. See the method [llvm::FunctionType::isVarArg]. **)
external is_var_arg : lltype -> bool = "llvm_is_var_arg"

(** [return_type fty] gets the return type of the function type [fty].
    See the method [llvm::FunctionType::getReturnType]. **)
external return_type : lltype -> lltype = "LLVMGetReturnType"

(** [param_types fty] gets the parameter types of the function type [fty].
    See the method [llvm::FunctionType::getParamType]. **)
external param_types : lltype -> lltype array = "llvm_param_types"

(*--... Operations on struct types .........................................--*)

(** [struct_type tys] returns the structure type containing in the types in the
    array [tys]. See the method [llvm::StructType::get]. **)
external struct_type : lltype array -> lltype = "llvm_struct_type"

(** [struct_type tys] returns the packed structure type containing in the types
    in the array [tys]. See the method [llvm::StructType::get]. **)
external packed_struct_type : lltype array -> lltype = "llvm_packed_struct_type"

(** [element_types sty] returns the constituent types of the struct type [sty].
    See the method [llvm::StructType::getElementType]. **)
external element_types : lltype -> lltype array = "llvm_element_types"

(** [is_packed sty] returns [true] if the structure type [sty] is packed,
    [false] otherwise. See the method [llvm::StructType::isPacked]. **)
external is_packed : lltype -> bool = "llvm_is_packed"

(*--... Operations on pointer, vector, and array types .....................--*)

(** [array_type ty n] returns the array type containing [n] elements of type
    [ty]. See the method [llvm::ArrayType::get]. **)
external array_type : lltype -> int -> lltype = "llvm_array_type"

(** [pointer_type ty] returns the pointer type referencing objects of type
    [ty]. See the method [llvm::PointerType::get]. **)
external pointer_type : lltype -> lltype = "LLVMPointerType"

(** [vector_type ty n] returns the array type containing [n] elements of the
    primitive type [ty]. See the method [llvm::ArrayType::get]. **)
external vector_type : lltype -> int -> lltype = "llvm_vector_type"

(** [element_type ty] returns the element type of the pointer, vector, or array
    type [ty]. See the method [llvm::SequentialType::get]. **)
external element_type : lltype -> lltype = "LLVMGetElementType"

(** [element_type aty] returns the element count of the array type [aty].
    See the method [llvm::ArrayType::getNumElements]. **)
external array_length : lltype -> int = "llvm_array_length"

(** [element_type ty] returns the element count of the vector type [ty].
    See the method [llvm::VectorType::getNumElements]. **)
external vector_size : lltype -> int = "llvm_vector_size"

(*--... Operations on other types ..........................................--*)

(** [opaque_type ()] creates a new opaque type distinct from any other.
    Opaque types are useful for building recursive types in combination with
    [refine_type opaque_ty ty].
    See [llvm::OpaqueType::get]. **)
external opaque_type : unit -> lltype = "llvm_opaque_type"

(** [void_type] is the type of a function which does not return any value.
    See [llvm::Type::VoidTy]. **)
val void_type : lltype

(** [label_type] is the type of a basic block. See [llvm::Type::LabelTy]. **)
val label_type : lltype

(*--... Operations on type handles .........................................--*)

(** [handle_to_type ty] creates a handle to the type [ty]. If [ty] is later
    refined as a result of a call to [refine_type], the handle will be updated;
    any bare [lltype] references will become invalid.
    See the class [llvm::PATypeHolder]. **)
external handle_to_type : lltype -> lltypehandle = "llvm_handle_to_type"

(** [type_of_handle tyh] resolves the type handle [tyh].
    See the method [llvm::PATypeHolder::get()]. **)
external type_of_handle : lltypehandle -> lltype = "llvm_type_of_handle"

(** [refine_type opaque_ty ty] replaces the abstract type [opaque_ty] with the
    concrete type [ty] in all users. Warning: This may invalidate [lltype]
    values! Use [lltypehandle] to manipulate potentially abstract types. See the
    method [llvm::Type::refineAbstractType]. **)
external refine_type : lltype -> lltype -> unit = "llvm_refine_type"


(*===-- Values ------------------------------------------------------------===*)

(** [type_of v] returns the type of the value [v].
    See the method [llvm::Value::getType]. **)
external type_of : llvalue -> lltype = "llvm_type_of"

(** [value_name v] returns the name of the value [v]. For global values, this is
    the symbol name. For instructions and basic blocks, it is the SSA register
    name. It is meaningless for constants.
    See the method [llvm::Value::getName]. **)
external value_name : llvalue -> string = "llvm_value_name"

(** [set_value_name n v] sets the name of the value [v] to [n]. See the method 
    [llvm::Value::setName]. **)
external set_value_name : string -> llvalue -> unit = "llvm_set_value_name"

(** [dump_value v] prints the .ll representation of the value [v] to standard
    error. See the method [llvm::Value::dump]. **)
external dump_value : llvalue -> unit = "llvm_dump_value"

(*--... Operations on constants of (mostly) any type .......................--*)

(** [is_constant v] returns [true] if the value [v] is a constant, [false]
    otherwise. Similar to [llvm::isa<Constant>]. **)
external is_constant : llvalue -> bool = "llvm_is_constant"

(** [const_null ty] returns the constant null (zero) of the type [ty].
    See the method [llvm::Constant::getNullValue]. **)
external const_null : lltype -> llvalue = "LLVMConstNull"

(** [const_all_ones ty] returns the constant '-1' of the integer or vector type
    [ty]. See the method [llvm::Constant::getAllOnesValue]. **)
external const_all_ones : (*int|vec*)lltype -> llvalue = "LLVMConstAllOnes"

(** [undef ty] returns the undefined value of the type [ty].
    See the method [llvm::UndefValue::get]. **)
external undef : lltype -> llvalue = "LLVMGetUndef"

(** [is_null v] returns [true] if the value [v] is the null (zero) value.
    See the method [llvm::Constant::isNullValue]. **)
external is_null : llvalue -> bool = "llvm_is_null"

(** [is_undef v] returns [true] if the value [v] is an undefined value, [false]
    otherwise. Similar to [llvm::isa<UndefValue>]. **)
external is_undef : llvalue -> bool = "llvm_is_undef"

(*--... Operations on scalar constants .....................................--*)

(** [const_int ty i] returns the integer constant of type [ty] and value [i].
    See the method [llvm::ConstantInt::get]. **)
external const_int : lltype -> int -> llvalue = "llvm_const_int"

(** [const_of_int64 ty i] returns the integer constant of type [ty] and value
    [i]. See the method [llvm::ConstantInt::get]. **)
external const_of_int64 : lltype -> Int64.t -> bool -> llvalue
                        = "llvm_const_of_int64"

(** [const_float ty n] returns the floating point constant of type [ty] and
    value [n]. See the method [llvm::ConstantInt::get]. **)
external const_float : lltype -> float -> llvalue = "llvm_const_float"

(*--... Operations on composite constants ..................................--*)

(** [const_string s] returns the constant [i8] array with the values of the
    characters in the string [s]. The array is not null-terminated (but see
    [const_stringz]). This value can in turn be used as the initializer for a
    global variable. See the method [llvm::ConstantArray::get]. **)
external const_string : string -> llvalue = "llvm_const_string"

(** [const_stringz s] returns the constant [i8] array with the values of the
    characters in the string [s] and a null terminator. This value can in turn
    be used as the initializer for a global variable.
    See the method [llvm::ConstantArray::get]. **)
external const_stringz : string -> llvalue = "llvm_const_stringz"

(** [const_array ty elts] returns the constant array of type
    [array_type ty (Array.length elts)] and containing the values [elts].
    This value can in turn be used as the initializer for a global variable.
    See the method [llvm::ConstantArray::get]. **)
external const_array : lltype -> llvalue array -> llvalue = "llvm_const_array"

(** [const_struct elts] returns the structured constant of type
    [struct_type (Array.map type_of elts)] and containing the values [elts].
    This value can in turn be used as the initializer for a global variable.
    See the method [llvm::ConstantStruct::get]. **)
external const_struct : llvalue array -> llvalue = "llvm_const_struct"

(** [const_packed_struct elts] returns the structured constant of type
    [packed_struct_type (Array.map type_of elts)] and containing the values
    [elts]. This value can in turn be used as the initializer for a global
    variable. See the method [llvm::ConstantStruct::get]. **)
external const_packed_struct : llvalue array -> llvalue
                             = "llvm_const_packed_struct"

(** [const_vector elts] returns the vector constant of type
    [vector_type (type_of elts.(0)) (Array.length elts)] and containing the
    values [elts]. See the method [llvm::ConstantVector::get]. **)
external const_vector : llvalue array -> llvalue = "llvm_const_vector"

(*--... Constant expressions ...............................................--*)

(** [size_of ty] returns the sizeof constant for the type [ty]. This is
    equivalent to [const_ptrtoint (const_gep (const_null (pointer_type ty))
    (const_int i64_type 1)) i64_type], but considerably more readable.
    See the method [llvm::ConstantExpr::getSizeOf]. **)
external size_of : lltype -> llvalue = "LLVMSizeOf"

(** [const_neg c] returns the arithmetic negation of the constant [c].
    See the method [llvm::ConstantExpr::getNeg]. **)
external const_neg : llvalue -> llvalue = "LLVMConstNeg"

(** [const_not c] returns the bitwise inverse of the constant [c].
    See the method [llvm::ConstantExpr::getNot]. **)
external const_not : llvalue -> llvalue = "LLVMConstNot"

(** [const_add c1 c2] returns the constant sum of two constants.
    See the method [llvm::ConstantExpr::getAdd]. **)
external const_add : llvalue -> llvalue -> llvalue = "LLVMConstAdd"

(** [const_sub c1 c2] returns the constant difference, [c1 - c2], of two
    constants. See the method [llvm::ConstantExpr::getSub]. **)
external const_sub : llvalue -> llvalue -> llvalue = "LLVMConstSub"

(** [const_mul c1 c2] returns the constant product of two constants.
    See the method [llvm::ConstantExpr::getMul]. **)
external const_mul : llvalue -> llvalue -> llvalue = "LLVMConstMul"

(** [const_udiv c1 c2] returns the constant quotient [c1 / c2] of two unsigned
    integer constants.
    See the method [llvm::ConstantExpr::getUDiv]. **)
external const_udiv : llvalue -> llvalue -> llvalue = "LLVMConstUDiv"

(** [const_sdiv c1 c2] returns the constant quotient [c1 / c2] of two signed
    integer constants.
    See the method [llvm::ConstantExpr::]. **)
external const_sdiv : llvalue -> llvalue -> llvalue = "LLVMConstSDiv"

(** [const_fdiv c1 c2] returns the constant quotient [c1 / c2] of two floating
    point constants.
    See the method [llvm::ConstantExpr::getFDiv]. **)
external const_fdiv : llvalue -> llvalue -> llvalue = "LLVMConstFDiv"

(** [const_udiv c1 c2] returns the constant remainder [c1 MOD c2] of two
    unsigned integer constants.
    See the method [llvm::ConstantExpr::getURem]. **)
external const_urem : llvalue -> llvalue -> llvalue = "LLVMConstURem"

(** [const_sdiv c1 c2] returns the constant remainder [c1 MOD c2] of two
    signed integer constants.
    See the method [llvm::ConstantExpr::getSRem]. **)
external const_srem : llvalue -> llvalue -> llvalue = "LLVMConstSRem"

(** [const_frem c1 c2] returns the constant remainder [c1 MOD c2] of two
    signed floating point constants.
    See the method [llvm::ConstantExpr::getFRem]. **)
external const_frem : llvalue -> llvalue -> llvalue = "LLVMConstFRem"

(** [const_and c1 c2] returns the constant bitwise [AND] of two integer
    constants.
    See the method [llvm::ConstantExpr::getAnd]. **)
external const_and : llvalue -> llvalue -> llvalue = "LLVMConstAnd"

(** [const_or c1 c2] returns the constant bitwise [OR] of two integer
    constants.
    See the method [llvm::ConstantExpr::getOr]. **)
external const_or : llvalue -> llvalue -> llvalue = "LLVMConstOr"

(** [const_xor c1 c2] returns the constant bitwise [XOR] of two integer
    constants.
    See the method [llvm::ConstantExpr::getXor]. **)
external const_xor : llvalue -> llvalue -> llvalue = "LLVMConstXor"

(** [const_icmp pred c1 c2] returns the constant comparison of two integer
    constants, [c1 pred c2].
    See the method [llvm::ConstantExpr::getICmp]. **)
external const_icmp : int_predicate -> llvalue -> llvalue -> llvalue
                    = "llvm_const_icmp"

(** [const_fcmp pred c1 c2] returns the constant comparison of two floating
    point constants, [c1 pred c2].
    See the method [llvm::ConstantExpr::getFCmp]. **)
external const_fcmp : real_predicate -> llvalue -> llvalue -> llvalue
                    = "llvm_const_fcmp"

(** [const_shl c1 c2] returns the constant integer [c1] left-shifted by the
    constant integer [c2].
    See the method [llvm::ConstantExpr::getShl]. **)
external const_shl : llvalue -> llvalue -> llvalue = "LLVMConstShl"

(** [const_lshr c1 c2] returns the constant integer [c1] right-shifted by the
    constant integer [c2] with zero extension.
    See the method [llvm::ConstantExpr::getLShr]. **)
external const_lshr : llvalue -> llvalue -> llvalue = "LLVMConstLShr"

(** [const_ashr c1 c2] returns the constant integer [c1] right-shifted by the
    constant integer [c2] with sign extension.
    See the method [llvm::ConstantExpr::getAShr]. **)
external const_ashr : llvalue -> llvalue -> llvalue = "LLVMConstAShr"

(** [const_gep pc indices] returns the constant [getElementPtr] of [p1] with the
    constant integers indices from the array [indices].
    See the method [llvm::ConstantExpr::getGetElementPtr]. **)
external const_gep : llvalue -> llvalue array -> llvalue = "llvm_const_gep"

(** [const_trunc c ty] returns the constant truncation of integer constant [c]
    to the smaller integer type [ty].
    See the method [llvm::ConstantExpr::getTrunc]. **)
external const_trunc : llvalue -> lltype -> llvalue = "LLVMConstTrunc"

(** [const_sext c ty] returns the constant sign extension of integer constant
    [c] to the larger integer type [ty].
    See the method [llvm::ConstantExpr::getSExt]. **)
external const_sext : llvalue -> lltype -> llvalue = "LLVMConstSExt"

(** [const_zext c ty] returns the constant zero extension of integer constant
    [c] to the larger integer type [ty].
    See the method [llvm::ConstantExpr::getZExt]. **)
external const_zext : llvalue -> lltype -> llvalue = "LLVMConstZExt"

(** [const_fptrunc c ty] returns the constant truncation of floating point
    constant [c] to the smaller floating point type [ty].
    See the method [llvm::ConstantExpr::getFPTrunc]. **)
external const_fptrunc : llvalue -> lltype -> llvalue = "LLVMConstFPTrunc"

(** [const_fpext c ty] returns the constant extension of floating point constant
    [c] to the larger floating point type [ty].
    See the method [llvm::ConstantExpr::getFPExt]. **)
external const_fpext : llvalue -> lltype -> llvalue = "LLVMConstFPExt"

(** [const_uitofp c ty] returns the constant floating point conversion of
    unsigned integer constant [c] to the floating point type [ty].
    See the method [llvm::ConstantExpr::getUIToFP]. **)
external const_uitofp : llvalue -> lltype -> llvalue = "LLVMConstUIToFP"

(** [const_sitofp c ty] returns the constant floating point conversion of
    signed integer constant [c] to the floating point type [ty].
    See the method [llvm::ConstantExpr::getSIToFP]. **)
external const_sitofp : llvalue -> lltype -> llvalue = "LLVMConstSIToFP"

(** [const_fptoui c ty] returns the constant unsigned integer conversion of
    floating point constant [c] to integer type [ty].
    See the method [llvm::ConstantExpr::getFPToUI]. **)
external const_fptoui : llvalue -> lltype -> llvalue = "LLVMConstFPToUI"

(** [const_fptoui c ty] returns the constant unsigned integer conversion of
    floating point constant [c] to integer type [ty].
    See the method [llvm::ConstantExpr::getFPToSI]. **)
external const_fptosi : llvalue -> lltype -> llvalue = "LLVMConstFPToSI"

(** [const_ptrtoint c ty] returns the constant integer conversion of
    pointer constant [c] to integer type [ty].
    See the method [llvm::ConstantExpr::getPtrToInt]. **)
external const_ptrtoint : llvalue -> lltype -> llvalue = "LLVMConstPtrToInt"

(** [const_inttoptr c ty] returns the constant pointer conversion of
    integer constant [c] to pointer type [ty].
    See the method [llvm::ConstantExpr::getIntToPtr]. **)
external const_inttoptr : llvalue -> lltype -> llvalue = "LLVMConstIntToPtr"

(** [const_bitcast c ty] returns the constant bitwise conversion of constant [c]
    to type [ty] of equal size.
    See the method [llvm::ConstantExpr::getBitCast]. **)
external const_bitcast : llvalue -> lltype -> llvalue = "LLVMConstBitCast"

(** [const_select cond t f] returns the constant conditional which returns value
    [t] if the boolean constant [cond] is true and the value [f] otherwise.
    See the method [llvm::ConstantExpr::getSelect]. **)
external const_select : llvalue -> llvalue -> llvalue -> llvalue
                      = "LLVMConstSelect"

(** [const_extractelement vec i] returns the constant [i]th element of
    constant vector [vec]. [i] must be a constant [i32] value unsigned less than
    the size of the vector.
    See the method [llvm::ConstantExpr::getExtractElement]. **)
external const_extractelement : llvalue -> llvalue -> llvalue
                              = "LLVMConstExtractElement"

(** [const_insertelement vec v i] returns the constant vector with the same
    elements as constant vector [v] but the [i]th element replaced by the
    constant [v]. [v] must be a constant value with the type of the vector
    elements. [i] must be a constant [i32] value unsigned less than the size
    of the vector.
    See the method [llvm::ConstantExpr::getInsertElement]. **)
external const_insertelement : llvalue -> llvalue -> llvalue -> llvalue
                             = "LLVMConstInsertElement"

(** [const_shufflevector a b mask] returns a constant [shufflevector].
    See the LLVM Language Reference for details on the [sufflevector]
    instruction.
    See the method [llvm::ConstantExpr::getShuffleVector]. **)
external const_shufflevector : llvalue -> llvalue -> llvalue -> llvalue
                             = "LLVMConstShuffleVector"

(*--... Operations on global variables, functions, and aliases (globals) ...--*)

(** [is_declaration g] returns [true] if the global value [g] is a declaration
    only. Returns [false] otherwise.
    See the method [llvm::GlobalValue::isDeclaration]. **)
external is_declaration : llvalue -> bool = "llvm_is_declaration"

(** [linkage g] returns the linkage of the global value [g].
    See the method [llvm::GlobalValue::getLinkage]. **)
external linkage : llvalue -> linkage = "llvm_linkage"

(** [set_linkage l g] sets the linkage of the global value [g] to [l].
    See the method [llvm::GlobalValue::setLinkage]. **)
external set_linkage : linkage -> llvalue -> unit = "llvm_set_linkage"

(** [section g] returns the linker section of the global value [g].
    See the method [llvm::GlobalValue::getSection]. **)
external section : llvalue -> string = "llvm_section"

(** [set_section s g] sets the linker section of the global value [g] to [s].
    See the method [llvm::GlobalValue::setSection]. **)
external set_section : string -> llvalue -> unit = "llvm_set_section"

(** [visibility g] returns the linker visibility of the global value [g].
    See the method [llvm::GlobalValue::getVisibility]. **)
external visibility : llvalue -> visibility = "llvm_visibility"

(** [set_visibility v g] sets the linker visibility of the global value [g] to
    [v]. See the method [llvm::GlobalValue::setVisibility]. **)
external set_visibility : visibility -> llvalue -> unit = "llvm_set_visibility"

(** [alignment g] returns the required alignment of the global value [g].
    See the method [llvm::GlobalValue::getAlignment]. **)
external alignment : llvalue -> int = "llvm_alignment"

(** [set_alignment n g] sets the required alignment of the global value [g] to
    [n] bytes. See the method [llvm::GlobalValue::setAlignment]. **)
external set_alignment : int -> llvalue -> unit = "llvm_set_alignment"

(*--... Operations on global variables .....................................--*)

(** [declare_global ty name m] returns a new global variable of type [ty] and
    with name [name] in module [m]. If such a global variable already exists,
    it is returned. If the type of the existing global differs, then a bitcast
    to [ty] is returned. **)
external declare_global : lltype -> string -> llmodule -> llvalue
                        = "llvm_declare_global"

(** [define_global name init m] returns a new global with name [name] and
    initializer [init] in module [m]. If the named global already exists, it is
    renamed.
    See the constructor of [llvm::GlobalVariable]. **)
external define_global : string -> llvalue -> llmodule -> llvalue
                       = "llvm_define_global"

(** [lookup_global name m] returns [Some g] if a global variable with name
    [name] exists in module [m]. If no such global exists, returns [None].
    See the [llvm::GlobalVariable] constructor. **)
external lookup_global : string -> llmodule -> llvalue option
                       = "llvm_lookup_global"

(** [delete_global gv] destroys the global variable [gv].
    See the method [llvm::GlobalVariable::eraseFromParent]. **)
external delete_global : llvalue -> unit = "llvm_delete_global"

(** [is_global_constant gv] returns [true] if the global variabile [gv] is a
    constant. Returns [false] otherwise.
    See the method [llvm::GlobalVariable::isConstant]. **)
external is_global_constant : llvalue -> bool = "llvm_is_global_constant"

(** [set_global_constant c gv] sets the global variable [gv] to be a constant if
    [c] is [true] and not if [c] is [false].
    See the method [llvm::GlobalVariable::setConstant]. **)
external set_global_constant : bool -> llvalue -> unit
                             = "llvm_set_global_constant"

(** [has_initializer gv] returns [true] if the global variable [gv] has an
    initializer and [false] otherwise.
    See the method [llvm::GlobalVariable::hasInitializer]. **)
external has_initializer : llvalue -> bool = "llvm_has_initializer"

(** [global_initializer gv] returns the initializer for the global variable
    [gv]. See the method [llvm::GlobalVariable::getInitializer]. **)
external global_initializer : llvalue -> llvalue = "LLVMGetInitializer"

(** [set_initializer c gv] sets the initializer for the global variable
    [gv] to the constant [c].
    See the method [llvm::GlobalVariable::setInitializer]. **)
external set_initializer : llvalue -> llvalue -> unit = "llvm_set_initializer"

(** [remove_initializer gv] unsets the initializer for the global variable
    [gv].
    See the method [llvm::GlobalVariable::setInitializer]. **)
external remove_initializer : llvalue -> unit = "llvm_remove_initializer"

(** [is_thread_local gv] returns [true] if the global variable [gv] is
    thread-local and [false] otherwise.
    See the method [llvm::GlobalVariable::isThreadLocal]. **)
external is_thread_local : llvalue -> bool = "llvm_is_thread_local"

(** [set_thread_local c gv] sets the global variable [gv] to be thread local if
    [c] is [true] and not otherwise.
    See the method [llvm::GlobalVariable::setThreadLocal]. **)
external set_thread_local : bool -> llvalue -> unit = "llvm_set_thread_local"

(*--... Operations on functions ............................................--*)

(** [declare_function name ty m] returns a new function of type [ty] and
    with name [name] in module [m]. If such a function already exists,
    it is returned. If the type of the existing function differs, then a bitcast
    to [ty] is returned. **)
external declare_function : string -> lltype -> llmodule -> llvalue
                          = "llvm_declare_function"

(** [define_function name ty m] creates a new function with name [name] and
    type [ty] in module [m]. If the named function already exists, it is
    renamed. An entry basic block is created in the function.
    See the constructor of [llvm::GlobalVariable]. **)
external define_function : string -> lltype -> llmodule -> llvalue
                         = "llvm_define_function"

(** [lookup_function name m] returns [Some f] if a function with name
    [name] exists in module [m]. If no such function exists, returns [None].
    See the method [llvm::Module] constructor. **)
external lookup_function : string -> llmodule -> llvalue option
                         = "llvm_lookup_function"

(** [delete_function f] destroys the function [f].
    See the method [llvm::Function::eraseFromParent]. **)
external delete_function : llvalue -> unit = "llvm_delete_function"

(** [params f] returns the parameters of function [f].
    See the method [llvm::Function::getArgumentList]. **)
external params : llvalue -> llvalue array = "llvm_params"

(** [param f n] returns the [n]th parameter of function [f].
    See the method [llvm::Function::getArgumentList]. **)
external param : llvalue -> int -> llvalue = "llvm_param"

(** [is_intrinsic f] returns true if the function [f] is an intrinsic.
    See the method [llvm::Function::isIntrinsic]. **)
external is_intrinsic : llvalue -> bool = "llvm_is_intrinsic"

(** [function_call_conv f] returns the calling convention of the function [f].
    See the method [llvm::Function::getCallingConv]. **)
external function_call_conv : llvalue -> int = "llvm_function_call_conv"

(** [set_function_call_conv cc f] sets the calling convention of the function
    [f] to the calling convention numbered [cc].
    See the method [llvm::Function::setCallingConv]. **)
external set_function_call_conv : int -> llvalue -> unit
                                = "llvm_set_function_call_conv"

(*--... Operations on basic blocks .........................................--*)

(** [basic_blocks fn] returns the basic blocks of the function [f].
    See the method [llvm::Function::getBasicBlockList]. **)
external basic_blocks : llvalue -> llbasicblock array = "llvm_basic_blocks"

(** [entry_block fn] returns the entry basic block of the function [f].
    See the method [llvm::Function::getEntryBlock]. **)
external entry_block : llvalue -> llbasicblock = "LLVMGetEntryBasicBlock"

(** [delete_block bb] deletes the basic block [bb].
    See the method [llvm::BasicBlock::eraseFromParent]. **)
external delete_block : llbasicblock -> unit = "llvm_delete_block"

(** [append_block name f] creates a new basic block named [name] at the end of
    function [f].
    See the constructor of [llvm::BasicBlock]. **)
external append_block : string -> llvalue -> llbasicblock = "llvm_append_block"

(** [insert_block name bb] creates a new basic block named [name] before the
    basic block [bb].
    See the constructor of [llvm::BasicBlock]. **)
external insert_block : string -> llbasicblock -> llbasicblock
                      = "llvm_insert_block"

(** [value_of_block bb] losslessly casts [bb] to an [llvalue]. **)
external value_of_block : llbasicblock -> llvalue = "LLVMBasicBlockAsValue"

(** [value_is_block v] returns [true] if the value [v] is a basic block and
    [false] otherwise.
    Similar to [llvm::isa<BasicBlock>]. **)
external value_is_block : llvalue -> bool = "llvm_value_is_block"

(** [block_of_value v] losslessly casts [v] to an [llbasicblock]. **)
external block_of_value : llvalue -> llbasicblock = "LLVMValueAsBasicBlock"

(*--... Operations on phi nodes ............................................--*)

(** [add_incoming (v, bb) pn] adds the value [v] to the phi node [pn] for use
    with branches from [bb]. See the method [llvm::PHINode::addIncoming]. **)
external add_incoming : (llvalue * llbasicblock) -> llvalue -> unit
                      = "llvm_add_incoming"

(** [incoming pn] returns the list of value-block pairs for phi node [pn].
    See the method [llvm::PHINode::getIncomingValue]. **)
external incoming : llvalue -> (llvalue * llbasicblock) list = "llvm_incoming"


(*===-- Instruction builders ----------------------------------------------===*)

(** [builder_before ins] creates an instruction builder positioned before the
    instruction [isn]. See the constructor for [llvm::LLVMBuilder]. **)
external builder_before : llvalue -> llbuilder = "llvm_builder_before"

(** [builder_at_end bb] creates an instruction builder positioned at the end of
    the basic block [bb]. See the constructor for [llvm::LLVMBuilder]. **)
external builder_at_end : llbasicblock -> llbuilder = "llvm_builder_at_end"

(** [position_before ins b] moves the instruction builder [b] to before the
    instruction [isn]. See the method [llvm::LLVMBuilder::SetInsertPoint]. **)
external position_before : llvalue -> llbuilder -> unit = "llvm_position_before"

(** [position_at_end bb b] moves the instruction builder [b] to the end of the
    basic block [bb]. See the method [llvm::LLVMBuilder::SetInsertPoint]. **)
external position_at_end : llbasicblock -> llbuilder -> unit
                         = "llvm_position_at_end"

(*--... Terminators ........................................................--*)

(** [build_ret_void b] creates a
    [ret void]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateRetVoid]. **)
external build_ret_void : llbuilder -> llvalue = "llvm_build_ret_void"

(** [build_ret v b] creates a
    [ret %v]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateRet]. **)
external build_ret : llvalue -> llbuilder -> llvalue = "llvm_build_ret"

(** [build_br bb b] creates a
    [b %bb]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateBr]. **)
external build_br : llbasicblock -> llbuilder -> llvalue = "llvm_build_br"

(** [build_cond_br cond tbb fbb b] creates a
    [b %cond, %tbb, %fbb]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateCondBr]. **)
external build_cond_br : llvalue -> llbasicblock -> llbasicblock -> llbuilder ->
                         llvalue = "llvm_build_cond_br"

(** [build_switch case elsebb b] creates an empty
    [switch %case, %elsebb]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSwitch]. **)
external build_switch : llvalue -> llbasicblock -> int -> llbuilder -> llvalue
                      = "llvm_build_switch"

(** [build_invoke fn args tobb unwindbb name b] creates an
    [%name = invoke %fn(args) to %tobb unwind %unwindbb]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateInvoke]. **)
external build_invoke : llvalue -> llvalue array -> llbasicblock ->
                        llbasicblock -> string -> llbuilder -> llvalue
                      = "llvm_build_invoke_bc" "llvm_build_invoke_nat"

(** [build_unwind b] creates an
    [unwind]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUnwind]. **)
external build_unwind : llbuilder -> llvalue = "llvm_build_unwind"

(** [build_unreachable b] creates an
    [unreachable]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUnwind]. **)
external build_unreachable : llbuilder -> llvalue = "llvm_build_unreachable"

(*--... Arithmetic .........................................................--*)

(** [build_add x y name b] creates a
    [%name = add %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAdd]. **)
external build_add : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_add"

(** [build_sub x y name b] creates a
    [%name = sub %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSub]. **)
external build_sub : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_sub"

(** [build_mul x y name b] creates a
    [%name = mul %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateMul]. **)
external build_mul : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_mul"

(** [build_udiv x y name b] creates a
    [%name = udiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUDiv]. **)
external build_udiv : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_udiv"

(** [build_sdiv x y name b] creates a
    [%name = sdiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSDiv]. **)
external build_sdiv : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_sdiv"

(** [build_fdiv x y name b] creates a
    [%name = fdiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFDiv]. **)
external build_fdiv : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_fdiv"

(** [build_urem x y name b] creates a
    [%name = urem %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateURem]. **)
external build_urem : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_urem"

(** [build_SRem x y name b] creates a
    [%name = srem %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSRem]. **)
external build_srem : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_srem"

(** [build_frem x y name b] creates a
    [%name = frem %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFRem]. **)
external build_frem : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_frem"

(** [build_shl x y name b] creates a
    [%name = shl %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateShl]. **)
external build_shl : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_shl"

(** [build_lshr x y name b] creates a
    [%name = lshr %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateLShr]. **)
external build_lshr : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_lshr"

(** [build_ashr x y name b] creates a
    [%name = ashr %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAShr]. **)
external build_ashr : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_ashr"

(** [build_and x y name b] creates a
    [%name = and %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAnd]. **)
external build_and : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_and"

(** [build_or x y name b] creates a
    [%name = or %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateOr]. **)
external build_or : llvalue -> llvalue -> string -> llbuilder -> llvalue
                  = "llvm_build_or"

(** [build_xor x y name b] creates a
    [%name = xor %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateXor]. **)
external build_xor : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_xor"

(** [build_neg x name b] creates a
    [%name = sub 0, %x]
    instruction at the position specified by the instruction builder [b].
    [-0.0] is used for floating point types to compute the correct sign.
    See the method [llvm::LLVMBuilder::CreateNeg]. **)
external build_neg : llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_neg"

(** [build_xor x name b] creates a
    [%name = xor %x, -1]
    instruction at the position specified by the instruction builder [b].
    [-1] is the correct "all ones" value for the type of [x].
    See the method [llvm::LLVMBuilder::CreateXor]. **)
external build_not : llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_not"

(*--... Memory .............................................................--*)

(** [build_malloc ty name b] creates a
    [%name = malloc %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAlloca]. **)
external build_malloc : lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_malloc"

(** [build_array_malloc ty n name b] creates a
    [%name = malloc %ty, %n]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateMalloc]. **)
external build_array_malloc : lltype -> llvalue -> string -> llbuilder ->
                              llvalue = "llvm_build_array_malloc"

(** [build_alloca ty name b] creates a
    [%name = alloca %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAlloca]. **)
external build_alloca : lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_alloca"

(** [build_array_alloca ty n name b] creates a
    [%name = alloca %ty, %n]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAlloca]. **)
external build_array_alloca : lltype -> llvalue -> string -> llbuilder ->
                              llvalue = "llvm_build_array_alloca"

(** [build_free v b] creates a
    [free %v]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFree]. **)
external build_free : llvalue -> llbuilder -> llvalue = "llvm_build_free"

(** [build_load v name b] creates a
    [%name = load %v]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateLoad]. **)
external build_load : llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_load"

(** [build_store v p b] creates a
    [store %v, %p]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateStore]. **)
external build_store : llvalue -> llvalue -> llbuilder -> llvalue
                     = "llvm_build_store"

(** [build_store p indices name b] creates a
    [%name = gep %p, indices...]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateGetElementPtr]. **)
external build_gep : llvalue -> llvalue array -> string -> llbuilder -> llvalue
                   = "llvm_build_gep"

(*--... Casts ..............................................................--*)

(** [build_trunc v ty name b] creates a
    [%name = trunc %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateTrunc]. **)
external build_trunc : llvalue -> lltype -> string -> llbuilder -> llvalue
                     = "llvm_build_trunc"

(** [build_zext v ty name b] creates a
    [%name = zext %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateZExt]. **)
external build_zext : llvalue -> lltype -> string -> llbuilder -> llvalue
                    = "llvm_build_zext"

(** [build_sext v ty name b] creates a
    [%name = sext %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSExt]. **)
external build_sext : llvalue -> lltype -> string -> llbuilder -> llvalue
                    = "llvm_build_sext"

(** [build_fptoui v ty name b] creates a
    [%name = fptoui %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPToUI]. **)
external build_fptoui : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_fptoui"

(** [build_fptosi v ty name b] creates a
    [%name = fptosi %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPToSI]. **)
external build_fptosi : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_fptosi"

(** [build_uitofp v ty name b] creates a
    [%name = uitofp %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUIToFP]. **)
external build_uitofp : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_uitofp"

(** [build_sitofp v ty name b] creates a
    [%name = sitofp %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSIToFP]. **)
external build_sitofp : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_sitofp"

(** [build_fptrunc v ty name b] creates a
    [%name = fptrunc %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPTrunc]. **)
external build_fptrunc : llvalue -> lltype -> string -> llbuilder -> llvalue
                       = "llvm_build_fptrunc"

(** [build_fpext v ty name b] creates a
    [%name = fpext %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPExt]. **)
external build_fpext : llvalue -> lltype -> string -> llbuilder -> llvalue
                     = "llvm_build_fpext"

(** [build_ptrtoint v ty name b] creates a
    [%name = prtotint %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreatePtrToInt]. **)
external build_ptrtoint : llvalue -> lltype -> string -> llbuilder -> llvalue
                        = "llvm_build_prttoint"

(** [build_inttoptr v ty name b] creates a
    [%name = inttoptr %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateIntToPtr]. **)
external build_inttoptr : llvalue -> lltype -> string -> llbuilder -> llvalue
                        = "llvm_build_inttoptr"

(** [build_bitcast v ty name b] creates a
    [%name = bitcast %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateBitcast]. **)
external build_bitcast : llvalue -> lltype -> string -> llbuilder -> llvalue
                       = "llvm_build_bitcast"

(*--... Comparisons ........................................................--*)

(** [build_icmp pred x y name b] creates a
    [%name = icmp %pred %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateICmp]. **)
external build_icmp : int_predicate -> llvalue -> llvalue -> string ->
                      llbuilder -> llvalue = "llvm_build_icmp"

(** [build_fcmp pred x y name b] creates a
    [%name = fcmp %pred %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFCmp]. **)
external build_fcmp : real_predicate -> llvalue -> llvalue -> string ->
                      llbuilder -> llvalue = "llvm_build_fcmp"

(*--... Miscellaneous instructions .........................................--*)

(** [build_phi incoming name b] creates a
    [%name = phi %incoming]
    instruction at the position specified by the instruction builder [b].
    [incoming] is a list of [(llvalue, llbasicblock)] tuples.
    See the method [llvm::LLVMBuilder::CreatePHI]. **)
external build_phi : (llvalue * llbasicblock) list -> string -> llbuilder ->
                     llvalue = "llvm_build_phi"

(** [build_call fn args name b] creates a
    [%name = call %fn(args...)]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateCall]. **)
external build_call : llvalue -> llvalue array -> string -> llbuilder -> llvalue
                    = "llvm_build_call"

(** [build_select cond thenv elsev name b] creates a
    [%name = select %cond, %thenv, %elsev]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSelect]. **)
external build_select : llvalue -> llvalue -> llvalue -> string -> llbuilder ->
                        llvalue = "llvm_build_select"

(** [build_va_arg valist argty name b] creates a
    [%name = va_arg %valist, %argty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateVAArg]. **)
external build_va_arg : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_va_arg"

(** [build_extractelement vec i name b] creates a
    [%name = extractelement %vec, %i]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateExtractElement]. **)
external build_extractelement : llvalue -> llvalue -> string -> llbuilder ->
                                llvalue = "llvm_build_extractelement"

(** [build_insertelement vec elt i name b] creates a
    [%name = insertelement %vec, %elt, %i]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateInsertElement]. **)
external build_insertelement : llvalue -> llvalue -> llvalue -> string ->
                               llbuilder -> llvalue = "llvm_build_insertelement"

(** [build_shufflevector veca vecb mask name b] creates a
    [%name = shufflevector %veca, %vecb, %mask]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateShuffleVector]. **)
external build_shufflevector : llvalue -> llvalue -> llvalue -> string ->
                               llbuilder -> llvalue = "llvm_build_shufflevector"
