(*===-- llvm/llvm.mli - LLVM Ocaml Interface -------------------------------===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Core API.

    This interface provides an ocaml API for the LLVM intermediate
    representation, the classes in the VMCore library. *)


(** {6 Abstract types}

    These abstract types correlate directly to the LLVM VMCore classes. *)

(** The top-level container for all LLVM global data. See the
    [llvm::LLVMContext] class. *)
type llcontext

(** The top-level container for all other LLVM Intermediate Representation (IR)
    objects. See the [llvm::Module] class. *)
type llmodule

(** Each value in the LLVM IR has a type, an instance of [lltype]. See the
    [llvm::Type] class. *)
type lltype

(** When building recursive types using {!refine_type}, [lltype] values may
    become invalid; use [lltypehandle] to resolve this problem. See the
    [llvm::AbstractTypeHolder] class. *)
type lltypehandle

(** Any value in the LLVM IR. Functions, instructions, global variables,
    constants, and much more are all [llvalues]. See the [llvm::Value] class.
    This type covers a wide range of subclasses. *)
type llvalue

(** Used to store users and usees of values. See the [llvm::Use] class. *)
type lluse

(** A basic block in LLVM IR. See the [llvm::BasicBlock] class. *)
type llbasicblock

(** Used to generate instructions in the LLVM IR. See the [llvm::LLVMBuilder]
    class. *)
type llbuilder

(** Used to efficiently handle large buffers of read-only binary data.
    See the [llvm::MemoryBuffer] class. *)
type llmemorybuffer

(** The kind of an [lltype], the result of [classify_type ty]. See the
    [llvm::Type::TypeID] enumeration. *)
module TypeKind : sig
  type t =
    Void
  | Float
  | Double
  | X86fp80
  | Fp128
  | Ppc_fp128
  | Label
  | Integer
  | Function
  | Struct
  | Array
  | Pointer
  | Opaque
  | Vector
  | Metadata
  | Union
end

(** The linkage of a global value, accessed with {!linkage} and
    {!set_linkage}. See [llvm::GlobalValue::LinkageTypes]. *)
module Linkage : sig
  type t =
    External
  | Available_externally
  | Link_once
  | Link_once_odr
  | Weak
  | Weak_odr
  | Appending
  | Internal
  | Private
  | Dllimport
  | Dllexport
  | External_weak
  | Ghost
  | Common
  | Linker_private
end

(** The linker visibility of a global value, accessed with {!visibility} and
    {!set_visibility}. See [llvm::GlobalValue::VisibilityTypes]. *)
module Visibility : sig
  type t =
    Default
  | Hidden
  | Protected
end

(** The following calling convention values may be accessed with
    {!function_call_conv} and {!set_function_call_conv}. Calling
    conventions are open-ended. *)
module CallConv : sig
  val c : int             (** [c] is the C calling convention. *)
  val fast : int          (** [fast] is the calling convention to allow LLVM
                              maximum optimization opportunities. Use only with
                              internal linkage. *)
  val cold : int          (** [cold] is the calling convention for
                              callee-save. *)
  val x86_stdcall : int   (** [x86_stdcall] is the familiar stdcall calling
                              convention from C. *)
  val x86_fastcall : int  (** [x86_fastcall] is the familiar fastcall calling
                              convention from C. *)
end

module Attribute : sig
  type t =
  | Zext
  | Sext
  | Noreturn
  | Inreg
  | Structret
  | Nounwind
  | Noalias
  | Byval
  | Nest
  | Readnone
  | Readonly
  | Noinline
  | Alwaysinline
  | Optsize
  | Ssp
  | Sspreq
  | Nocapture
  | Noredzone
  | Noimplicitfloat
  | Naked
  | Inlinehint
end

(** The predicate for an integer comparison ([icmp]) instruction.
    See the [llvm::ICmpInst::Predicate] enumeration. *)
module Icmp : sig
  type t =
  | Eq
  | Ne
  | Ugt
  | Uge
  | Ult
  | Ule
  | Sgt
  | Sge
  | Slt
  | Sle
end

(** The predicate for a floating-point comparison ([fcmp]) instruction.
    See the [llvm::FCmpInst::Predicate] enumeration. *)
module Fcmp : sig
  type t =
  | False
  | Oeq
  | Ogt
  | Oge
  | Olt
  | Ole
  | One
  | Ord
  | Uno
  | Ueq
  | Ugt
  | Uge
  | Ult
  | Ule
  | Une
  | True
end


(** {6 Iteration} *)

(** [Before b] and [At_end a] specify positions from the start of the ['b] list
    of [a]. [llpos] is used to specify positions in and for forward iteration
    through the various value lists maintained by the LLVM IR. *)
type ('a, 'b) llpos =
| At_end of 'a
| Before of 'b

(** [After b] and [At_start a] specify positions from the end of the ['b] list
    of [a]. [llrev_pos] is used for reverse iteration through the various value
    lists maintained by the LLVM IR. *)
type ('a, 'b) llrev_pos =
| At_start of 'a
| After of 'b


(** {6 Exceptions} *)

exception IoError of string


(** {6 Contexts} *)

(** [create_context ()] creates a context for storing the "global" state in
    LLVM. See the constructor [llvm::LLVMContext]. *)
external create_context : unit -> llcontext = "llvm_create_context"

(** [destroy_context ()] destroys a context. See the destructor
    [llvm::LLVMContext::~LLVMContext]. *)
external dispose_context : llcontext -> unit = "llvm_dispose_context"

(** See the function [llvm::getGlobalContext]. *)
external global_context : unit -> llcontext = "llvm_global_context"

(** [mdkind_id context name] returns the MDKind ID that corresponds to the
    name [name] in the context [context].  See the function
    [llvm::LLVMContext::getMDKindID]. *)
external mdkind_id : llcontext -> string -> int = "llvm_mdkind_id"


(** {6 Modules} *)

(** [create_module context id] creates a module with the supplied module ID in
    the context [context].  Modules are not garbage collected; it is mandatory
    to call {!dispose_module} to free memory. See the constructor
    [llvm::Module::Module]. *)
external create_module : llcontext -> string -> llmodule = "llvm_create_module"

(** [dispose_module m] destroys a module [m] and all of the IR objects it
    contained. All references to subordinate objects are invalidated;
    referencing them will invoke undefined behavior. See the destructor
    [llvm::Module::~Module]. *)
external dispose_module : llmodule -> unit = "llvm_dispose_module"

(** [target_triple m] is the target specifier for the module [m], something like
    [i686-apple-darwin8]. See the method [llvm::Module::getTargetTriple]. *)
external target_triple: llmodule -> string
                      = "llvm_target_triple"

(** [target_triple triple m] changes the target specifier for the module [m] to
    the string [triple]. See the method [llvm::Module::setTargetTriple]. *)
external set_target_triple: string -> llmodule -> unit
                          = "llvm_set_target_triple"

(** [data_layout m] is the data layout specifier for the module [m], something
    like [e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-...-a0:0:64-f80:128:128]. See the
    method [llvm::Module::getDataLayout]. *)
external data_layout: llmodule -> string
                    = "llvm_data_layout"

(** [set_data_layout s m] changes the data layout specifier for the module [m]
    to the string [s]. See the method [llvm::Module::setDataLayout]. *)
external set_data_layout: string -> llmodule -> unit
                        = "llvm_set_data_layout"

(** [define_type_name name ty m] adds a named type to the module's symbol table.
    Returns [true] if successful. If such a name already exists, then no entry
    is added and [false] is returned. See the [llvm::Module::addTypeName]
    method. *)
external define_type_name : string -> lltype -> llmodule -> bool
                          = "llvm_add_type_name"

(** [delete_type_name name] removes a type name from the module's symbol
    table. *)
external delete_type_name : string -> llmodule -> unit
                          = "llvm_delete_type_name"

(** [type_by_name m n] returns the type in the module [m] named [n], or [None]
    if it does not exist. See the method [llvm::Module::getTypeByName]. *)
external type_by_name : llmodule -> string -> lltype option
                      = "llvm_type_by_name"

(** [dump_module m] prints the .ll representation of the module [m] to standard
    error. See the method [llvm::Module::dump]. *)
external dump_module : llmodule -> unit = "llvm_dump_module"


(** {6 Types} *)

(** [classify_type ty] returns the {!TypeKind.t} corresponding to the type [ty].
    See the method [llvm::Type::getTypeID]. *)
external classify_type : lltype -> TypeKind.t = "llvm_classify_type"

(** [type_context ty] returns the {!llcontext} corresponding to the type [ty].
    See the method [llvm::Type::getContext]. *)
external type_context : lltype -> llcontext = "llvm_type_context"

(** [string_of_lltype ty] returns a string describing the type [ty]. *)
val string_of_lltype : lltype -> string

(** {7 Operations on integer types} *)

(** [i1_type c] returns an integer type of bitwidth 1 in the context [c]. See
    [llvm::Type::Int1Ty]. *)
external i1_type : llcontext -> lltype = "llvm_i1_type"

(** [i8_type c] returns an integer type of bitwidth 8 in the context [c]. See
    [llvm::Type::Int8Ty]. *)
external i8_type : llcontext -> lltype = "llvm_i8_type"

(** [i16_type c] returns an integer type of bitwidth 16 in the context [c]. See
    [llvm::Type::Int16Ty]. *)
external i16_type : llcontext -> lltype = "llvm_i16_type"

(** [i32_type c] returns an integer type of bitwidth 32 in the context [c]. See
    [llvm::Type::Int32Ty]. *)
external i32_type : llcontext -> lltype = "llvm_i32_type"

(** [i64_type c] returns an integer type of bitwidth 64 in the context [c]. See
    [llvm::Type::Int64Ty]. *)
external i64_type : llcontext -> lltype = "llvm_i64_type"

(** [integer_type c n] returns an integer type of bitwidth [n] in the context
    [c]. See the method [llvm::IntegerType::get]. *)
external integer_type : llcontext -> int -> lltype = "llvm_integer_type"

(** [integer_bitwidth c ty] returns the number of bits in the integer type [ty]
    in the context [c].  See the method [llvm::IntegerType::getBitWidth]. *)
external integer_bitwidth : lltype -> int = "llvm_integer_bitwidth"


(** {7 Operations on real types} *)

(** [float_type c] returns the IEEE 32-bit floating point type in the context
    [c]. See [llvm::Type::FloatTy]. *)
external float_type : llcontext -> lltype = "llvm_float_type"

(** [double_type c] returns the IEEE 64-bit floating point type in the context
    [c]. See [llvm::Type::DoubleTy]. *)
external double_type : llcontext -> lltype = "llvm_double_type"

(** [x86fp80_type c] returns the x87 80-bit floating point type in the context
    [c]. See [llvm::Type::X86_FP80Ty]. *)
external x86fp80_type : llcontext -> lltype = "llvm_x86fp80_type"

(** [fp128_type c] returns the IEEE 128-bit floating point type in the context
    [c]. See [llvm::Type::FP128Ty]. *)
external fp128_type : llcontext -> lltype = "llvm_fp128_type"

(** [ppc_fp128_type c] returns the PowerPC 128-bit floating point type in the
    context [c]. See [llvm::Type::PPC_FP128Ty]. *)
external ppc_fp128_type : llcontext -> lltype = "llvm_ppc_fp128_type"


(** {7 Operations on function types} *)

(** [function_type ret_ty param_tys] returns the function type returning
    [ret_ty] and taking [param_tys] as parameters.
    See the method [llvm::FunctionType::get]. *)
external function_type : lltype -> lltype array -> lltype = "llvm_function_type"

(** [va_arg_function_type ret_ty param_tys] is just like
    [function_type ret_ty param_tys] except that it returns the function type
    which also takes a variable number of arguments.
    See the method [llvm::FunctionType::get]. *)
external var_arg_function_type : lltype -> lltype array -> lltype
                               = "llvm_var_arg_function_type"

(** [is_var_arg fty] returns [true] if [fty] is a varargs function type, [false]
    otherwise. See the method [llvm::FunctionType::isVarArg]. *)
external is_var_arg : lltype -> bool = "llvm_is_var_arg"

(** [return_type fty] gets the return type of the function type [fty].
    See the method [llvm::FunctionType::getReturnType]. *)
external return_type : lltype -> lltype = "LLVMGetReturnType"

(** [param_types fty] gets the parameter types of the function type [fty].
    See the method [llvm::FunctionType::getParamType]. *)
external param_types : lltype -> lltype array = "llvm_param_types"


(** {7 Operations on struct types} *)

(** [struct_type context tys] returns the structure type in the context
    [context] containing in the types in the array [tys]. See the method
    [llvm::StructType::get]. *)
external struct_type : llcontext -> lltype array -> lltype
                     = "llvm_struct_type"

(** [packed_struct_type context ys] returns the packed structure type in the
    context [context] containing in the types in the array [tys]. See the method
    [llvm::StructType::get]. *)
external packed_struct_type : llcontext -> lltype array -> lltype
                            = "llvm_packed_struct_type"

(** [struct_element_types sty] returns the constituent types of the struct type
    [sty]. See the method [llvm::StructType::getElementType]. *)
external struct_element_types : lltype -> lltype array
                              = "llvm_struct_element_types"

(** [is_packed sty] returns [true] if the structure type [sty] is packed,
    [false] otherwise. See the method [llvm::StructType::isPacked]. *)
external is_packed : lltype -> bool = "llvm_is_packed"


(** {7 Operations on union types} *)

(** [union_type context tys] returns the union type in the context [context]
    containing the types in the array [tys]. See the method
    [llvm::UnionType::get] *)
external union_type : llcontext -> lltype array -> lltype = "llvm_union_type"

(** [union_element_types uty] returns the constituent types of the union type
    [uty]. See the method [llvm::UnionType::getElementType]. *)
external union_element_types : lltype -> lltype array
                             = "llvm_union_element_types"


(** {7 Operations on pointer, vector, and array types} *)

(** [array_type ty n] returns the array type containing [n] elements of type
    [ty]. See the method [llvm::ArrayType::get]. *)
external array_type : lltype -> int -> lltype = "llvm_array_type"

(** [pointer_type ty] returns the pointer type referencing objects of type
    [ty] in the default address space (0).
    See the method [llvm::PointerType::getUnqual]. *)
external pointer_type : lltype -> lltype = "llvm_pointer_type"

(** [qualified_pointer_type ty as] returns the pointer type referencing objects
    of type [ty] in address space [as].
    See the method [llvm::PointerType::get]. *)
external qualified_pointer_type : lltype -> int -> lltype
                                = "llvm_qualified_pointer_type"

(** [vector_type ty n] returns the array type containing [n] elements of the
    primitive type [ty]. See the method [llvm::ArrayType::get]. *)
external vector_type : lltype -> int -> lltype = "llvm_vector_type"

(** [element_type ty] returns the element type of the pointer, vector, or array
    type [ty]. See the method [llvm::SequentialType::get]. *)
external element_type : lltype -> lltype = "LLVMGetElementType"

(** [element_type aty] returns the element count of the array type [aty].
    See the method [llvm::ArrayType::getNumElements]. *)
external array_length : lltype -> int = "llvm_array_length"

(** [address_space pty] returns the address space qualifier of the pointer type
    [pty]. See the method [llvm::PointerType::getAddressSpace]. *)
external address_space : lltype -> int = "llvm_address_space"

(** [element_type ty] returns the element count of the vector type [ty].
    See the method [llvm::VectorType::getNumElements]. *)
external vector_size : lltype -> int = "llvm_vector_size"


(** {7 Operations on other types} *)

(** [opaque_type c] creates a new opaque type distinct from any other in the
    context [c]. Opaque types are useful for building recursive types in
    combination with {!refine_type}. See [llvm::OpaqueType::get]. *)
external opaque_type : llcontext -> lltype = "llvm_opaque_type"

(** [void_type c] creates a type of a function which does not return any
    value in the context [c]. See [llvm::Type::VoidTy]. *)
external void_type : llcontext -> lltype = "llvm_void_type"

(** [label_type c] creates a type of a basic block in the context [c]. See
    [llvm::Type::LabelTy]. *)
external label_type : llcontext -> lltype = "llvm_label_type"

(** {7 Operations on type handles} *)

(** [handle_to_type ty] creates a handle to the type [ty]. If [ty] is later
    refined as a result of a call to {!refine_type}, the handle will be updated;
    any bare [lltype] references will become invalid.
    See the class [llvm::PATypeHolder]. *)
external handle_to_type : lltype -> lltypehandle = "llvm_handle_to_type"

(** [type_of_handle tyh] resolves the type handle [tyh].
    See the method [llvm::PATypeHolder::get()]. *)
external type_of_handle : lltypehandle -> lltype = "llvm_type_of_handle"

(** [refine_type opaque_ty ty] replaces the abstract type [opaque_ty] with the
    concrete type [ty] in all users. Warning: This may invalidate {!lltype}
    values! Use {!lltypehandle} to manipulate potentially abstract types. See
    the method [llvm::Type::refineAbstractType]. *)
external refine_type : lltype -> lltype -> unit = "llvm_refine_type"


(* {6 Values} *)

(** [type_of v] returns the type of the value [v].
    See the method [llvm::Value::getType]. *)
external type_of : llvalue -> lltype = "llvm_type_of"

(** [value_name v] returns the name of the value [v]. For global values, this is
    the symbol name. For instructions and basic blocks, it is the SSA register
    name. It is meaningless for constants.
    See the method [llvm::Value::getName]. *)
external value_name : llvalue -> string = "llvm_value_name"

(** [set_value_name n v] sets the name of the value [v] to [n]. See the method
    [llvm::Value::setName]. *)
external set_value_name : string -> llvalue -> unit = "llvm_set_value_name"

(** [dump_value v] prints the .ll representation of the value [v] to standard
    error. See the method [llvm::Value::dump]. *)
external dump_value : llvalue -> unit = "llvm_dump_value"

(** [replace_all_uses_with old new] replaces all uses of the value [old]
 * with the value [new]. See the method [llvm::Value::replaceAllUsesWith]. *)
external replace_all_uses_with : llvalue -> llvalue -> unit
                               = "LLVMReplaceAllUsesWith"


(* {6 Uses} *)

(** [use_begin v] returns the first position in the use list for the value [v].
    [use_begin] and [use_succ] can e used to iterate over the use list in order.
    See the method [llvm::Value::use_begin]. *)
external use_begin : llvalue -> lluse option = "llvm_use_begin"

(** [use_succ u] returns the use list position succeeding [u].
    See the method [llvm::use_value_iterator::operator++]. *)
external use_succ : lluse -> lluse option = "llvm_use_succ"

(** [user u] returns the user of the use [u].
    See the method [llvm::Use::getUser]. *)
external user : lluse -> llvalue = "llvm_user"

(** [used_value u] returns the usee of the use [u].
    See the method [llvm::Use::getUsedValue]. *)
external used_value : lluse -> llvalue = "llvm_used_value"

(** [iter_uses f v] applies function [f] to each of the users of the value [v]
    in order. Tail recursive. *)
val iter_uses : (lluse -> unit) -> llvalue -> unit

(** [fold_left_uses f init v] is [f (... (f init u1) ...) uN] where
    [u1,...,uN] are the users of the value [v]. Tail recursive. *)
val fold_left_uses : ('a -> lluse -> 'a) -> 'a -> llvalue -> 'a

(** [fold_right_uses f v init] is [f u1 (... (f uN init) ...)] where
    [u1,...,uN] are the users of the value [v]. Not tail recursive. *)
val fold_right_uses : (lluse -> 'a -> 'a) -> llvalue -> 'a -> 'a


(* {6 Users} *)

(** [operand v i] returns the operand at index [i] for the value [v]. See the
    method [llvm::User::getOperand]. *)
external operand : llvalue -> int -> llvalue = "llvm_operand"


(** {7 Operations on constants of (mostly) any type} *)

(** [is_constant v] returns [true] if the value [v] is a constant, [false]
    otherwise. Similar to [llvm::isa<Constant>]. *)
external is_constant : llvalue -> bool = "llvm_is_constant"

(** [const_null ty] returns the constant null (zero) of the type [ty].
    See the method [llvm::Constant::getNullValue]. *)
external const_null : lltype -> llvalue = "LLVMConstNull"

(** [const_all_ones ty] returns the constant '-1' of the integer or vector type
    [ty]. See the method [llvm::Constant::getAllOnesValue]. *)
external const_all_ones : (*int|vec*)lltype -> llvalue = "LLVMConstAllOnes"

(** [const_pointer_null ty] returns the constant null (zero) pointer of the type
    [ty]. See the method [llvm::ConstantPointerNull::get]. *)
external const_pointer_null : lltype -> llvalue = "LLVMConstPointerNull"

(** [undef ty] returns the undefined value of the type [ty].
    See the method [llvm::UndefValue::get]. *)
external undef : lltype -> llvalue = "LLVMGetUndef"

(** [is_null v] returns [true] if the value [v] is the null (zero) value.
    See the method [llvm::Constant::isNullValue]. *)
external is_null : llvalue -> bool = "llvm_is_null"

(** [is_undef v] returns [true] if the value [v] is an undefined value, [false]
    otherwise. Similar to [llvm::isa<UndefValue>]. *)
external is_undef : llvalue -> bool = "llvm_is_undef"


(** {7 Operations on instructions} *)

(** [has_metadata i] returns whether or not the instruction [i] has any
    metadata attached to it. See the function
    [llvm::Instruction::hasMetadata]. *)
external has_metadata : llvalue -> bool = "llvm_has_metadata"

(** [metadata i kind] optionally returns the metadata associated with the
    kind [kind] in the instruction [i] See the function
    [llvm::Instruction::getMetadata]. *)
external metadata : llvalue -> int -> llvalue option = "llvm_metadata"

(** [set_metadata i kind md] sets the metadata [md] of kind [kind] in the
    instruction [i]. See the function [llvm::Instruction::setMetadata]. *)
external set_metadata : llvalue -> int -> llvalue -> unit = "llvm_set_metadata"

(** [clear_metadata i kind] clears the metadata of kind [kind] in the
    instruction [i]. See the function [llvm::Instruction::setMetadata]. *)
external clear_metadata : llvalue -> int -> unit = "llvm_clear_metadata"


(** {7 Operations on metadata} *)

(** [mdstring c s] returns the MDString of the string [s] in the context [c].
    See the method [llvm::MDNode::get]. *)
external mdstring : llcontext -> string -> llvalue = "llvm_mdstring"

(** [mdnode c elts] returns the MDNode containing the values [elts] in the
    context [c].
    See the method [llvm::MDNode::get]. *)
external mdnode : llcontext -> llvalue array -> llvalue = "llvm_mdnode"


(** {7 Operations on scalar constants} *)

(** [const_int ty i] returns the integer constant of type [ty] and value [i].
    See the method [llvm::ConstantInt::get]. *)
external const_int : lltype -> int -> llvalue = "llvm_const_int"

(** [const_of_int64 ty i] returns the integer constant of type [ty] and value
    [i]. See the method [llvm::ConstantInt::get]. *)
external const_of_int64 : lltype -> Int64.t -> bool -> llvalue
                        = "llvm_const_of_int64"

(** [const_int_of_string ty s r] returns the integer constant of type [ty] and
 * value [s], with the radix [r]. See the method [llvm::ConstantInt::get]. *)
external const_int_of_string : lltype -> string -> int -> llvalue
                   = "llvm_const_int_of_string"

(** [const_float ty n] returns the floating point constant of type [ty] and
    value [n]. See the method [llvm::ConstantFP::get]. *)
external const_float : lltype -> float -> llvalue = "llvm_const_float"

(** [const_float_of_string ty s] returns the floating point constant of type
    [ty] and value [n]. See the method [llvm::ConstantFP::get]. *)
external const_float_of_string : lltype -> string -> llvalue
                               = "llvm_const_float_of_string"


(** {7 Operations on composite constants} *)

(** [const_string c s] returns the constant [i8] array with the values of the
    characters in the string [s] in the context [c]. The array is not 
    null-terminated (but see {!const_stringz}). This value can in turn be used
    as the initializer for a global variable. See the method
    [llvm::ConstantArray::get]. *)
external const_string : llcontext -> string -> llvalue = "llvm_const_string"

(** [const_stringz c s] returns the constant [i8] array with the values of the
    characters in the string [s] and a null terminator in the context [c]. This
    value can in turn be used as the initializer for a global variable.
    See the method [llvm::ConstantArray::get]. *)
external const_stringz : llcontext -> string -> llvalue = "llvm_const_stringz"

(** [const_array ty elts] returns the constant array of type
    [array_type ty (Array.length elts)] and containing the values [elts].
    This value can in turn be used as the initializer for a global variable.
    See the method [llvm::ConstantArray::get]. *)
external const_array : lltype -> llvalue array -> llvalue = "llvm_const_array"

(** [const_struct context elts] returns the structured constant of type
    [struct_type (Array.map type_of elts)] and containing the values [elts]
    in the context [context]. This value can in turn be used as the initializer
    for a global variable. See the method [llvm::ConstantStruct::get]. *)
external const_struct : llcontext -> llvalue array -> llvalue
                      = "llvm_const_struct"

(** [const_packed_struct context elts] returns the structured constant of
    type {!packed_struct_type} [(Array.map type_of elts)] and containing the
    values [elts] in the context [context]. This value can in turn be used as
    the initializer for a global variable. See the method
    [llvm::ConstantStruct::get]. *)
external const_packed_struct : llcontext -> llvalue array -> llvalue
                             = "llvm_const_packed_struct"

(** [const_vector elts] returns the vector constant of type
    [vector_type (type_of elts.(0)) (Array.length elts)] and containing the
    values [elts]. See the method [llvm::ConstantVector::get]. *)
external const_vector : llvalue array -> llvalue = "llvm_const_vector"

(** [const_union ty v] returns the union constant of type [union_type tys] and
    containing the value [v]. See the method [llvm::ConstantUnion::get]. *)
external const_union : lltype -> llvalue -> llvalue = "LLVMConstUnion"


(** {7 Constant expressions} *)

(** [align_of ty] returns the alignof constant for the type [ty]. This is
    equivalent to [const_ptrtoint (const_gep (const_null (pointer_type {i8,ty}))
    (const_int i32_type 0) (const_int i32_type 1)) i32_type], but considerably
    more readable.  See the method [llvm::ConstantExpr::getAlignOf]. *)
external align_of : lltype -> llvalue = "LLVMAlignOf"

(** [size_of ty] returns the sizeof constant for the type [ty]. This is
    equivalent to [const_ptrtoint (const_gep (const_null (pointer_type ty))
    (const_int i32_type 1)) i64_type], but considerably more readable.
    See the method [llvm::ConstantExpr::getSizeOf]. *)
external size_of : lltype -> llvalue = "LLVMSizeOf"

(** [const_neg c] returns the arithmetic negation of the constant [c].
    See the method [llvm::ConstantExpr::getNeg]. *)
external const_neg : llvalue -> llvalue = "LLVMConstNeg"

(** [const_nsw_neg c] returns the arithmetic negation of the constant [c] with
    no signed wrapping. The result is undefined if the negation overflows.
    See the method [llvm::ConstantExpr::getNSWNeg]. *)
external const_nsw_neg : llvalue -> llvalue = "LLVMConstNSWNeg"

(** [const_nuw_neg c] returns the arithmetic negation of the constant [c] with
    no unsigned wrapping. The result is undefined if the negation overflows.
    See the method [llvm::ConstantExpr::getNUWNeg]. *)
external const_nuw_neg : llvalue -> llvalue = "LLVMConstNUWNeg"

(** [const_fneg c] returns the arithmetic negation of the constant float [c].
    See the method [llvm::ConstantExpr::getFNeg]. *)
external const_fneg : llvalue -> llvalue = "LLVMConstFNeg"

(** [const_not c] returns the bitwise inverse of the constant [c].
    See the method [llvm::ConstantExpr::getNot]. *)
external const_not : llvalue -> llvalue = "LLVMConstNot"

(** [const_add c1 c2] returns the constant sum of two constants.
    See the method [llvm::ConstantExpr::getAdd]. *)
external const_add : llvalue -> llvalue -> llvalue = "LLVMConstAdd"

(** [const_nsw_add c1 c2] returns the constant sum of two constants with no
    signed wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWAdd]. *)
external const_nsw_add : llvalue -> llvalue -> llvalue = "LLVMConstNSWAdd"

(** [const_nuw_add c1 c2] returns the constant sum of two constants with no
    unsigned wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWAdd]. *)
external const_nuw_add : llvalue -> llvalue -> llvalue = "LLVMConstNUWAdd"

(** [const_fadd c1 c2] returns the constant sum of two constant floats.
    See the method [llvm::ConstantExpr::getFAdd]. *)
external const_fadd : llvalue -> llvalue -> llvalue = "LLVMConstFAdd"

(** [const_sub c1 c2] returns the constant difference, [c1 - c2], of two
    constants. See the method [llvm::ConstantExpr::getSub]. *)
external const_sub : llvalue -> llvalue -> llvalue = "LLVMConstSub"

(** [const_nsw_sub c1 c2] returns the constant difference of two constants with
    no signed wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWSub]. *)
external const_nsw_sub : llvalue -> llvalue -> llvalue = "LLVMConstNSWSub"

(** [const_nuw_sub c1 c2] returns the constant difference of two constants with
    no unsigned wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWSub]. *)
external const_nuw_sub : llvalue -> llvalue -> llvalue = "LLVMConstNUWSub"

(** [const_fsub c1 c2] returns the constant difference, [c1 - c2], of two
    constant floats. See the method [llvm::ConstantExpr::getFSub]. *)
external const_fsub : llvalue -> llvalue -> llvalue = "LLVMConstFSub"

(** [const_mul c1 c2] returns the constant product of two constants.
    See the method [llvm::ConstantExpr::getMul]. *)
external const_mul : llvalue -> llvalue -> llvalue = "LLVMConstMul"

(** [const_nsw_mul c1 c2] returns the constant product of two constants with
    no signed wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWMul]. *)
external const_nsw_mul : llvalue -> llvalue -> llvalue = "LLVMConstNSWMul"

(** [const_nuw_mul c1 c2] returns the constant product of two constants with
    no unsigned wrapping. The result is undefined if the sum overflows.
    See the method [llvm::ConstantExpr::getNSWMul]. *)
external const_nuw_mul : llvalue -> llvalue -> llvalue = "LLVMConstNUWMul"

(** [const_fmul c1 c2] returns the constant product of two constants floats.
    See the method [llvm::ConstantExpr::getFMul]. *)
external const_fmul : llvalue -> llvalue -> llvalue = "LLVMConstFMul"

(** [const_udiv c1 c2] returns the constant quotient [c1 / c2] of two unsigned
    integer constants.
    See the method [llvm::ConstantExpr::getUDiv]. *)
external const_udiv : llvalue -> llvalue -> llvalue = "LLVMConstUDiv"

(** [const_sdiv c1 c2] returns the constant quotient [c1 / c2] of two signed
    integer constants.
    See the method [llvm::ConstantExpr::getSDiv]. *)
external const_sdiv : llvalue -> llvalue -> llvalue = "LLVMConstSDiv"

(** [const_exact_sdiv c1 c2] returns the constant quotient [c1 / c2] of two
    signed integer constants. The result is undefined if the result is rounded
    or overflows. See the method [llvm::ConstantExpr::getExactSDiv]. *)
external const_exact_sdiv : llvalue -> llvalue -> llvalue = "LLVMConstExactSDiv"

(** [const_fdiv c1 c2] returns the constant quotient [c1 / c2] of two floating
    point constants.
    See the method [llvm::ConstantExpr::getFDiv]. *)
external const_fdiv : llvalue -> llvalue -> llvalue = "LLVMConstFDiv"

(** [const_urem c1 c2] returns the constant remainder [c1 MOD c2] of two
    unsigned integer constants.
    See the method [llvm::ConstantExpr::getURem]. *)
external const_urem : llvalue -> llvalue -> llvalue = "LLVMConstURem"

(** [const_srem c1 c2] returns the constant remainder [c1 MOD c2] of two
    signed integer constants.
    See the method [llvm::ConstantExpr::getSRem]. *)
external const_srem : llvalue -> llvalue -> llvalue = "LLVMConstSRem"

(** [const_frem c1 c2] returns the constant remainder [c1 MOD c2] of two
    signed floating point constants.
    See the method [llvm::ConstantExpr::getFRem]. *)
external const_frem : llvalue -> llvalue -> llvalue = "LLVMConstFRem"

(** [const_and c1 c2] returns the constant bitwise [AND] of two integer
    constants.
    See the method [llvm::ConstantExpr::getAnd]. *)
external const_and : llvalue -> llvalue -> llvalue = "LLVMConstAnd"

(** [const_or c1 c2] returns the constant bitwise [OR] of two integer
    constants.
    See the method [llvm::ConstantExpr::getOr]. *)
external const_or : llvalue -> llvalue -> llvalue = "LLVMConstOr"

(** [const_xor c1 c2] returns the constant bitwise [XOR] of two integer
    constants.
    See the method [llvm::ConstantExpr::getXor]. *)
external const_xor : llvalue -> llvalue -> llvalue = "LLVMConstXor"

(** [const_icmp pred c1 c2] returns the constant comparison of two integer
    constants, [c1 pred c2].
    See the method [llvm::ConstantExpr::getICmp]. *)
external const_icmp : Icmp.t -> llvalue -> llvalue -> llvalue
                    = "llvm_const_icmp"

(** [const_fcmp pred c1 c2] returns the constant comparison of two floating
    point constants, [c1 pred c2].
    See the method [llvm::ConstantExpr::getFCmp]. *)
external const_fcmp : Fcmp.t -> llvalue -> llvalue -> llvalue
                    = "llvm_const_fcmp"

(** [const_shl c1 c2] returns the constant integer [c1] left-shifted by the
    constant integer [c2].
    See the method [llvm::ConstantExpr::getShl]. *)
external const_shl : llvalue -> llvalue -> llvalue = "LLVMConstShl"

(** [const_lshr c1 c2] returns the constant integer [c1] right-shifted by the
    constant integer [c2] with zero extension.
    See the method [llvm::ConstantExpr::getLShr]. *)
external const_lshr : llvalue -> llvalue -> llvalue = "LLVMConstLShr"

(** [const_ashr c1 c2] returns the constant integer [c1] right-shifted by the
    constant integer [c2] with sign extension.
    See the method [llvm::ConstantExpr::getAShr]. *)
external const_ashr : llvalue -> llvalue -> llvalue = "LLVMConstAShr"

(** [const_gep pc indices] returns the constant [getElementPtr] of [p1] with the
    constant integers indices from the array [indices].
    See the method [llvm::ConstantExpr::getGetElementPtr]. *)
external const_gep : llvalue -> llvalue array -> llvalue = "llvm_const_gep"

(** [const_in_bounds_gep pc indices] returns the constant [getElementPtr] of [p1]
    with the constant integers indices from the array [indices].
    See the method [llvm::ConstantExpr::getInBoundsGetElementPtr]. *)
external const_in_bounds_gep : llvalue -> llvalue array -> llvalue
                            = "llvm_const_in_bounds_gep"

(** [const_trunc c ty] returns the constant truncation of integer constant [c]
    to the smaller integer type [ty].
    See the method [llvm::ConstantExpr::getTrunc]. *)
external const_trunc : llvalue -> lltype -> llvalue = "LLVMConstTrunc"

(** [const_sext c ty] returns the constant sign extension of integer constant
    [c] to the larger integer type [ty].
    See the method [llvm::ConstantExpr::getSExt]. *)
external const_sext : llvalue -> lltype -> llvalue = "LLVMConstSExt"

(** [const_zext c ty] returns the constant zero extension of integer constant
    [c] to the larger integer type [ty].
    See the method [llvm::ConstantExpr::getZExt]. *)
external const_zext : llvalue -> lltype -> llvalue = "LLVMConstZExt"

(** [const_fptrunc c ty] returns the constant truncation of floating point
    constant [c] to the smaller floating point type [ty].
    See the method [llvm::ConstantExpr::getFPTrunc]. *)
external const_fptrunc : llvalue -> lltype -> llvalue = "LLVMConstFPTrunc"

(** [const_fpext c ty] returns the constant extension of floating point constant
    [c] to the larger floating point type [ty].
    See the method [llvm::ConstantExpr::getFPExt]. *)
external const_fpext : llvalue -> lltype -> llvalue = "LLVMConstFPExt"

(** [const_uitofp c ty] returns the constant floating point conversion of
    unsigned integer constant [c] to the floating point type [ty].
    See the method [llvm::ConstantExpr::getUIToFP]. *)
external const_uitofp : llvalue -> lltype -> llvalue = "LLVMConstUIToFP"

(** [const_sitofp c ty] returns the constant floating point conversion of
    signed integer constant [c] to the floating point type [ty].
    See the method [llvm::ConstantExpr::getSIToFP]. *)
external const_sitofp : llvalue -> lltype -> llvalue = "LLVMConstSIToFP"

(** [const_fptoui c ty] returns the constant unsigned integer conversion of
    floating point constant [c] to integer type [ty].
    See the method [llvm::ConstantExpr::getFPToUI]. *)
external const_fptoui : llvalue -> lltype -> llvalue = "LLVMConstFPToUI"

(** [const_fptoui c ty] returns the constant unsigned integer conversion of
    floating point constant [c] to integer type [ty].
    See the method [llvm::ConstantExpr::getFPToSI]. *)
external const_fptosi : llvalue -> lltype -> llvalue = "LLVMConstFPToSI"

(** [const_ptrtoint c ty] returns the constant integer conversion of
    pointer constant [c] to integer type [ty].
    See the method [llvm::ConstantExpr::getPtrToInt]. *)
external const_ptrtoint : llvalue -> lltype -> llvalue = "LLVMConstPtrToInt"

(** [const_inttoptr c ty] returns the constant pointer conversion of
    integer constant [c] to pointer type [ty].
    See the method [llvm::ConstantExpr::getIntToPtr]. *)
external const_inttoptr : llvalue -> lltype -> llvalue = "LLVMConstIntToPtr"

(** [const_bitcast c ty] returns the constant bitwise conversion of constant [c]
    to type [ty] of equal size.
    See the method [llvm::ConstantExpr::getBitCast]. *)
external const_bitcast : llvalue -> lltype -> llvalue = "LLVMConstBitCast"

(** [const_zext_or_bitcast c ty] returns a constant zext or bitwise cast
    conversion of constant [c] to type [ty].
    See the method [llvm::ConstantExpr::getZExtOrBitCast]. *)
external const_zext_or_bitcast : llvalue -> lltype -> llvalue
                               = "LLVMConstZExtOrBitCast"

(** [const_sext_or_bitcast c ty] returns a constant sext or bitwise cast
    conversion of constant [c] to type [ty].
    See the method [llvm::ConstantExpr::getSExtOrBitCast]. *)
external const_sext_or_bitcast : llvalue -> lltype -> llvalue
                               = "LLVMConstSExtOrBitCast"

(** [const_trunc_or_bitcast c ty] returns a constant trunc or bitwise cast
    conversion of constant [c] to type [ty].
    See the method [llvm::ConstantExpr::getTruncOrBitCast]. *)
external const_trunc_or_bitcast : llvalue -> lltype -> llvalue
                                = "LLVMConstTruncOrBitCast"

(** [const_pointercast c ty] returns a constant bitcast or a pointer-to-int
    cast conversion of constant [c] to type [ty] of equal size.
    See the method [llvm::ConstantExpr::getPointerCast]. *)
external const_pointercast : llvalue -> lltype -> llvalue
                           = "LLVMConstPointerCast"

(** [const_intcast c ty] returns a constant zext, bitcast, or trunc for integer
    -> integer casts of constant [c] to type [ty].
    See the method [llvm::ConstantExpr::getIntCast]. *)
external const_intcast : llvalue -> lltype -> llvalue
                       = "LLVMConstIntCast"

(** [const_fpcast c ty] returns a constant fpext, bitcast, or fptrunc for fp ->
    fp casts of constant [c] to type [ty].
    See the method [llvm::ConstantExpr::getFPCast]. *)
external const_fpcast : llvalue -> lltype -> llvalue
                      = "LLVMConstFPCast"

(** [const_select cond t f] returns the constant conditional which returns value
    [t] if the boolean constant [cond] is true and the value [f] otherwise.
    See the method [llvm::ConstantExpr::getSelect]. *)
external const_select : llvalue -> llvalue -> llvalue -> llvalue
                      = "LLVMConstSelect"

(** [const_extractelement vec i] returns the constant [i]th element of
    constant vector [vec]. [i] must be a constant [i32] value unsigned less than
    the size of the vector.
    See the method [llvm::ConstantExpr::getExtractElement]. *)
external const_extractelement : llvalue -> llvalue -> llvalue
                              = "LLVMConstExtractElement"

(** [const_insertelement vec v i] returns the constant vector with the same
    elements as constant vector [v] but the [i]th element replaced by the
    constant [v]. [v] must be a constant value with the type of the vector
    elements. [i] must be a constant [i32] value unsigned less than the size
    of the vector.
    See the method [llvm::ConstantExpr::getInsertElement]. *)
external const_insertelement : llvalue -> llvalue -> llvalue -> llvalue
                             = "LLVMConstInsertElement"

(** [const_shufflevector a b mask] returns a constant [shufflevector].
    See the LLVM Language Reference for details on the [sufflevector]
    instruction.
    See the method [llvm::ConstantExpr::getShuffleVector]. *)
external const_shufflevector : llvalue -> llvalue -> llvalue -> llvalue
                             = "LLVMConstShuffleVector"

(** [const_extractvalue agg idxs] returns the constant [idxs]th value of
    constant aggregate [agg]. Each [idxs] must be less than the size of the
    aggregate.  See the method [llvm::ConstantExpr::getExtractValue]. *)
external const_extractvalue : llvalue -> int array -> llvalue
                            = "llvm_const_extractvalue"

(** [const_insertvalue agg val idxs] inserts the value [val] in the specified
    indexs [idxs] in the aggegate [agg]. Each [idxs] must be less than the size
    of the aggregate. See the method [llvm::ConstantExpr::getInsertValue]. *)
external const_insertvalue : llvalue -> llvalue -> int array -> llvalue
                           = "llvm_const_insertvalue"

(** [const_inline_asm ty asm con side align] inserts a inline assembly string.
    See the method [llvm::InlineAsm::get]. *)
external const_inline_asm : lltype -> string -> string -> bool -> bool ->
                            llvalue
                          = "llvm_const_inline_asm"

(** [block_address f bb] returns the address of the basic block [bb] in the
    function [f]. See the method [llvm::BasicBlock::get]. *)
external block_address : llvalue -> llbasicblock -> llvalue = "LLVMBlockAddress"


(** {7 Operations on global variables, functions, and aliases (globals)} *)

(** [global_parent g] is the enclosing module of the global value [g].
    See the method [llvm::GlobalValue::getParent]. *)
external global_parent : llvalue -> llmodule = "LLVMGetGlobalParent"

(** [is_declaration g] returns [true] if the global value [g] is a declaration
    only. Returns [false] otherwise.
    See the method [llvm::GlobalValue::isDeclaration]. *)
external is_declaration : llvalue -> bool = "llvm_is_declaration"

(** [linkage g] returns the linkage of the global value [g].
    See the method [llvm::GlobalValue::getLinkage]. *)
external linkage : llvalue -> Linkage.t = "llvm_linkage"

(** [set_linkage l g] sets the linkage of the global value [g] to [l].
    See the method [llvm::GlobalValue::setLinkage]. *)
external set_linkage : Linkage.t -> llvalue -> unit = "llvm_set_linkage"

(** [section g] returns the linker section of the global value [g].
    See the method [llvm::GlobalValue::getSection]. *)
external section : llvalue -> string = "llvm_section"

(** [set_section s g] sets the linker section of the global value [g] to [s].
    See the method [llvm::GlobalValue::setSection]. *)
external set_section : string -> llvalue -> unit = "llvm_set_section"

(** [visibility g] returns the linker visibility of the global value [g].
    See the method [llvm::GlobalValue::getVisibility]. *)
external visibility : llvalue -> Visibility.t = "llvm_visibility"

(** [set_visibility v g] sets the linker visibility of the global value [g] to
    [v]. See the method [llvm::GlobalValue::setVisibility]. *)
external set_visibility : Visibility.t -> llvalue -> unit
                        = "llvm_set_visibility"

(** [alignment g] returns the required alignment of the global value [g].
    See the method [llvm::GlobalValue::getAlignment]. *)
external alignment : llvalue -> int = "llvm_alignment"

(** [set_alignment n g] sets the required alignment of the global value [g] to
    [n] bytes. See the method [llvm::GlobalValue::setAlignment]. *)
external set_alignment : int -> llvalue -> unit = "llvm_set_alignment"


(** {7 Operations on global variables} *)

(** [declare_global ty name m] returns a new global variable of type [ty] and
    with name [name] in module [m] in the default address space (0). If such a
    global variable already exists, it is returned. If the type of the existing
    global differs, then a bitcast to [ty] is returned. *)
external declare_global : lltype -> string -> llmodule -> llvalue
                        = "llvm_declare_global"

(** [declare_qualified_global ty name as m] returns a new global variable of
    type [ty] and with name [name] in module [m] in the address space [as]. If
    such a global variable already exists, it is returned. If the type of the
    existing global differs, then a bitcast to [ty] is returned. *)
external declare_qualified_global : lltype -> string -> int -> llmodule ->
                                    llvalue
                                  = "llvm_declare_qualified_global"

(** [define_global name init m] returns a new global with name [name] and
    initializer [init] in module [m] in the default address space (0). If the
    named global already exists, it is renamed.
    See the constructor of [llvm::GlobalVariable]. *)
external define_global : string -> llvalue -> llmodule -> llvalue
                       = "llvm_define_global"

(** [define_qualified_global name init as m] returns a new global with name
    [name] and initializer [init] in module [m] in the address space [as]. If
    the named global already exists, it is renamed.
    See the constructor of [llvm::GlobalVariable]. *)
external define_qualified_global : string -> llvalue -> int -> llmodule ->
                                   llvalue
                                 = "llvm_define_qualified_global"

(** [lookup_global name m] returns [Some g] if a global variable with name
    [name] exists in module [m]. If no such global exists, returns [None].
    See the [llvm::GlobalVariable] constructor. *)
external lookup_global : string -> llmodule -> llvalue option
                       = "llvm_lookup_global"

(** [delete_global gv] destroys the global variable [gv].
    See the method [llvm::GlobalVariable::eraseFromParent]. *)
external delete_global : llvalue -> unit = "llvm_delete_global"

(** [global_begin m] returns the first position in the global variable list of
    the module [m]. [global_begin] and [global_succ] can be used to iterate
    over the global list in order.
    See the method [llvm::Module::global_begin]. *)
external global_begin : llmodule -> (llmodule, llvalue) llpos
                      = "llvm_global_begin"

(** [global_succ gv] returns the global variable list position succeeding
    [Before gv].
    See the method [llvm::Module::global_iterator::operator++]. *)
external global_succ : llvalue -> (llmodule, llvalue) llpos
                     = "llvm_global_succ"

(** [iter_globals f m] applies function [f] to each of the global variables of
    module [m] in order. Tail recursive. *)
val iter_globals : (llvalue -> unit) -> llmodule -> unit

(** [fold_left_globals f init m] is [f (... (f init g1) ...) gN] where
    [g1,...,gN] are the global variables of module [m]. Tail recursive. *)
val fold_left_globals : ('a -> llvalue -> 'a) -> 'a -> llmodule -> 'a

(** [global_end m] returns the last position in the global variable list of the
    module [m]. [global_end] and [global_pred] can be used to iterate over the
    global list in reverse.
    See the method [llvm::Module::global_end]. *)
external global_end : llmodule -> (llmodule, llvalue) llrev_pos
                    = "llvm_global_end"

(** [global_pred gv] returns the global variable list position preceding
    [After gv].
    See the method [llvm::Module::global_iterator::operator--]. *)
external global_pred : llvalue -> (llmodule, llvalue) llrev_pos
                     = "llvm_global_pred"

(** [rev_iter_globals f m] applies function [f] to each of the global variables
    of module [m] in reverse order. Tail recursive. *)
val rev_iter_globals : (llvalue -> unit) -> llmodule -> unit

(** [fold_right_globals f m init] is [f g1 (... (f gN init) ...)] where
    [g1,...,gN] are the global variables of module [m]. Tail recursive. *)
val fold_right_globals : (llvalue -> 'a -> 'a) -> llmodule -> 'a -> 'a

(** [is_global_constant gv] returns [true] if the global variabile [gv] is a
    constant. Returns [false] otherwise.
    See the method [llvm::GlobalVariable::isConstant]. *)
external is_global_constant : llvalue -> bool = "llvm_is_global_constant"

(** [set_global_constant c gv] sets the global variable [gv] to be a constant if
    [c] is [true] and not if [c] is [false].
    See the method [llvm::GlobalVariable::setConstant]. *)
external set_global_constant : bool -> llvalue -> unit
                             = "llvm_set_global_constant"

(** [global_initializer gv] returns the initializer for the global variable
    [gv]. See the method [llvm::GlobalVariable::getInitializer]. *)
external global_initializer : llvalue -> llvalue = "LLVMGetInitializer"

(** [set_initializer c gv] sets the initializer for the global variable
    [gv] to the constant [c].
    See the method [llvm::GlobalVariable::setInitializer]. *)
external set_initializer : llvalue -> llvalue -> unit = "llvm_set_initializer"

(** [remove_initializer gv] unsets the initializer for the global variable
    [gv].
    See the method [llvm::GlobalVariable::setInitializer]. *)
external remove_initializer : llvalue -> unit = "llvm_remove_initializer"

(** [is_thread_local gv] returns [true] if the global variable [gv] is
    thread-local and [false] otherwise.
    See the method [llvm::GlobalVariable::isThreadLocal]. *)
external is_thread_local : llvalue -> bool = "llvm_is_thread_local"

(** [set_thread_local c gv] sets the global variable [gv] to be thread local if
    [c] is [true] and not otherwise.
    See the method [llvm::GlobalVariable::setThreadLocal]. *)
external set_thread_local : bool -> llvalue -> unit = "llvm_set_thread_local"


(** {7 Operations on aliases} *)

(** [add_alias m t a n] inserts an alias in the module [m] with the type [t] and
    the aliasee [a] with the name [n].
    See the constructor for [llvm::GlobalAlias]. *)
external add_alias : llmodule -> lltype -> llvalue -> string -> llvalue
                   = "llvm_add_alias"


(** {7 Operations on functions} *)

(** [declare_function name ty m] returns a new function of type [ty] and
    with name [name] in module [m]. If such a function already exists,
    it is returned. If the type of the existing function differs, then a bitcast
    to [ty] is returned. *)
external declare_function : string -> lltype -> llmodule -> llvalue
                          = "llvm_declare_function"

(** [define_function name ty m] creates a new function with name [name] and
    type [ty] in module [m]. If the named function already exists, it is
    renamed. An entry basic block is created in the function.
    See the constructor of [llvm::GlobalVariable]. *)
external define_function : string -> lltype -> llmodule -> llvalue
                         = "llvm_define_function"

(** [lookup_function name m] returns [Some f] if a function with name
    [name] exists in module [m]. If no such function exists, returns [None].
    See the method [llvm::Module] constructor. *)
external lookup_function : string -> llmodule -> llvalue option
                         = "llvm_lookup_function"

(** [delete_function f] destroys the function [f].
    See the method [llvm::Function::eraseFromParent]. *)
external delete_function : llvalue -> unit = "llvm_delete_function"

(** [function_begin m] returns the first position in the function list of the
    module [m]. [function_begin] and [function_succ] can be used to iterate over
    the function list in order.
    See the method [llvm::Module::begin]. *)
external function_begin : llmodule -> (llmodule, llvalue) llpos
                        = "llvm_function_begin"

(** [function_succ gv] returns the function list position succeeding
    [Before gv].
    See the method [llvm::Module::iterator::operator++]. *)
external function_succ : llvalue -> (llmodule, llvalue) llpos
                       = "llvm_function_succ"

(** [iter_functions f m] applies function [f] to each of the functions of module
    [m] in order. Tail recursive. *)
val iter_functions : (llvalue -> unit) -> llmodule -> unit

(** [fold_left_function f init m] is [f (... (f init f1) ...) fN] where
    [f1,...,fN] are the functions of module [m]. Tail recursive. *)
val fold_left_functions : ('a -> llvalue -> 'a) -> 'a -> llmodule -> 'a

(** [function_end m] returns the last position in the function list of
    the module [m]. [function_end] and [function_pred] can be used to iterate
    over the function list in reverse.
    See the method [llvm::Module::end]. *)
external function_end : llmodule -> (llmodule, llvalue) llrev_pos
                      = "llvm_function_end"

(** [function_pred gv] returns the function list position preceding [After gv].
    See the method [llvm::Module::iterator::operator--]. *)
external function_pred : llvalue -> (llmodule, llvalue) llrev_pos
                       = "llvm_function_pred"

(** [rev_iter_functions f fn] applies function [f] to each of the functions of
    module [m] in reverse order. Tail recursive. *)
val rev_iter_functions : (llvalue -> unit) -> llmodule -> unit

(** [fold_right_functions f m init] is [f (... (f init fN) ...) f1] where
    [f1,...,fN] are the functions of module [m]. Tail recursive. *)
val fold_right_functions : (llvalue -> 'a -> 'a) -> llmodule -> 'a -> 'a

(** [is_intrinsic f] returns true if the function [f] is an intrinsic.
    See the method [llvm::Function::isIntrinsic]. *)
external is_intrinsic : llvalue -> bool = "llvm_is_intrinsic"

(** [function_call_conv f] returns the calling convention of the function [f].
    See the method [llvm::Function::getCallingConv]. *)
external function_call_conv : llvalue -> int = "llvm_function_call_conv"

(** [set_function_call_conv cc f] sets the calling convention of the function
    [f] to the calling convention numbered [cc].
    See the method [llvm::Function::setCallingConv]. *)
external set_function_call_conv : int -> llvalue -> unit
                                = "llvm_set_function_call_conv"

(** [gc f] returns [Some name] if the function [f] has a garbage
    collection algorithm specified and [None] otherwise.
    See the method [llvm::Function::getGC]. *)
external gc : llvalue -> string option = "llvm_gc"

(** [set_gc gc f] sets the collection algorithm for the function [f] to
    [gc]. See the method [llvm::Function::setGC]. *)
external set_gc : string option -> llvalue -> unit = "llvm_set_gc"

(** [add_function_attr f a] adds attribute [a] to the return type of function
    [f]. *)
external add_function_attr : llvalue -> Attribute.t -> unit
                           = "llvm_add_function_attr"

(** [remove_function_attr f a] removes attribute [a] from the return type of
    function [f]. *)
external remove_function_attr : llvalue -> Attribute.t -> unit
                              = "llvm_remove_function_attr"

(** {7 Operations on params} *)

(** [params f] returns the parameters of function [f].
    See the method [llvm::Function::getArgumentList]. *)
external params : llvalue -> llvalue array = "llvm_params"

(** [param f n] returns the [n]th parameter of function [f].
    See the method [llvm::Function::getArgumentList]. *)
external param : llvalue -> int -> llvalue = "llvm_param"

(** [param_parent p] returns the parent function that owns the parameter.
    See the method [llvm::Argument::getParent]. *)
external param_parent : llvalue -> llvalue = "LLVMGetParamParent"

(** [param_begin f] returns the first position in the parameter list of the
    function [f]. [param_begin] and [param_succ] can be used to iterate over
    the parameter list in order.
    See the method [llvm::Function::arg_begin]. *)
external param_begin : llvalue -> (llvalue, llvalue) llpos = "llvm_param_begin"

(** [param_succ bb] returns the parameter list position succeeding
    [Before bb].
    See the method [llvm::Function::arg_iterator::operator++]. *)
external param_succ : llvalue -> (llvalue, llvalue) llpos = "llvm_param_succ"

(** [iter_params f fn] applies function [f] to each of the parameters
    of function [fn] in order. Tail recursive. *)
val iter_params : (llvalue -> unit) -> llvalue -> unit

(** [fold_left_params f init fn] is [f (... (f init b1) ...) bN] where
    [b1,...,bN] are the parameters of function [fn]. Tail recursive. *)
val fold_left_params : ('a -> llvalue -> 'a) -> 'a -> llvalue -> 'a

(** [param_end f] returns the last position in the parameter list of
    the function [f]. [param_end] and [param_pred] can be used to iterate
    over the parameter list in reverse.
    See the method [llvm::Function::arg_end]. *)
external param_end : llvalue -> (llvalue, llvalue) llrev_pos = "llvm_param_end"

(** [param_pred gv] returns the function list position preceding [After gv].
    See the method [llvm::Function::arg_iterator::operator--]. *)
external param_pred : llvalue -> (llvalue, llvalue) llrev_pos
                    = "llvm_param_pred"

(** [rev_iter_params f fn] applies function [f] to each of the parameters
    of function [fn] in reverse order. Tail recursive. *)
val rev_iter_params : (llvalue -> unit) -> llvalue -> unit

(** [fold_right_params f fn init] is [f (... (f init bN) ...) b1] where
    [b1,...,bN] are the parameters of function [fn]. Tail recursive. *)
val fold_right_params : (llvalue -> 'a -> 'a) -> llvalue -> 'a -> 'a

(** [add_param p a] adds attribute [a] to parameter [p]. *)
external add_param_attr : llvalue -> Attribute.t -> unit = "llvm_add_param_attr"

(** [remove_param_attr p a] removes attribute [a] from parameter [p]. *)
external remove_param_attr : llvalue -> Attribute.t -> unit
                           = "llvm_remove_param_attr"

(** [set_param_alignment p a] set the alignment of parameter [p] to [a]. *)
external set_param_alignment : llvalue -> int -> unit
                             = "llvm_set_param_alignment"

(** {7 Operations on basic blocks} *)

(** [basic_blocks fn] returns the basic blocks of the function [f].
    See the method [llvm::Function::getBasicBlockList]. *)
external basic_blocks : llvalue -> llbasicblock array = "llvm_basic_blocks"

(** [entry_block fn] returns the entry basic block of the function [f].
    See the method [llvm::Function::getEntryBlock]. *)
external entry_block : llvalue -> llbasicblock = "LLVMGetEntryBasicBlock"

(** [delete_block bb] deletes the basic block [bb].
    See the method [llvm::BasicBlock::eraseFromParent]. *)
external delete_block : llbasicblock -> unit = "llvm_delete_block"

(** [append_block c name f] creates a new basic block named [name] at the end of
    function [f] in the context [c].
    See the constructor of [llvm::BasicBlock]. *)
external append_block : llcontext -> string -> llvalue -> llbasicblock
                      = "llvm_append_block"

(** [insert_block c name bb] creates a new basic block named [name] before the
    basic block [bb] in the context [c].
    See the constructor of [llvm::BasicBlock]. *)
external insert_block : llcontext -> string -> llbasicblock -> llbasicblock
                      = "llvm_insert_block"

(** [block_parent bb] returns the parent function that owns the basic block.
    See the method [llvm::BasicBlock::getParent]. *)
external block_parent : llbasicblock -> llvalue = "LLVMGetBasicBlockParent"

(** [block_begin f] returns the first position in the basic block list of the
    function [f]. [block_begin] and [block_succ] can be used to iterate over
    the basic block list in order.
    See the method [llvm::Function::begin]. *)
external block_begin : llvalue -> (llvalue, llbasicblock) llpos
                     = "llvm_block_begin"

(** [block_succ bb] returns the basic block list position succeeding
    [Before bb].
    See the method [llvm::Function::iterator::operator++]. *)
external block_succ : llbasicblock -> (llvalue, llbasicblock) llpos
                    = "llvm_block_succ"

(** [iter_blocks f fn] applies function [f] to each of the basic blocks
    of function [fn] in order. Tail recursive. *)
val iter_blocks : (llbasicblock -> unit) -> llvalue -> unit

(** [fold_left_blocks f init fn] is [f (... (f init b1) ...) bN] where
    [b1,...,bN] are the basic blocks of function [fn]. Tail recursive. *)
val fold_left_blocks : ('a -> llbasicblock -> 'a) -> 'a -> llvalue -> 'a

(** [block_end f] returns the last position in the basic block list of
    the function [f]. [block_end] and [block_pred] can be used to iterate
    over the basic block list in reverse.
    See the method [llvm::Function::end]. *)
external block_end : llvalue -> (llvalue, llbasicblock) llrev_pos
                   = "llvm_block_end"

(** [block_pred gv] returns the function list position preceding [After gv].
    See the method [llvm::Function::iterator::operator--]. *)
external block_pred : llbasicblock -> (llvalue, llbasicblock) llrev_pos
                    = "llvm_block_pred"

(** [rev_iter_blocks f fn] applies function [f] to each of the basic blocks
    of function [fn] in reverse order. Tail recursive. *)
val rev_iter_blocks : (llbasicblock -> unit) -> llvalue -> unit

(** [fold_right_blocks f fn init] is [f (... (f init bN) ...) b1] where
    [b1,...,bN] are the basic blocks of function [fn]. Tail recursive. *)
val fold_right_blocks : (llbasicblock -> 'a -> 'a) -> llvalue -> 'a -> 'a

(** [value_of_block bb] losslessly casts [bb] to an [llvalue]. *)
external value_of_block : llbasicblock -> llvalue = "LLVMBasicBlockAsValue"

(** [value_is_block v] returns [true] if the value [v] is a basic block and
    [false] otherwise.
    Similar to [llvm::isa<BasicBlock>]. *)
external value_is_block : llvalue -> bool = "llvm_value_is_block"

(** [block_of_value v] losslessly casts [v] to an [llbasicblock]. *)
external block_of_value : llvalue -> llbasicblock = "LLVMValueAsBasicBlock"


(** {7 Operations on instructions} *)

(** [instr_parent i] is the enclosing basic block of the instruction [i].
    See the method [llvm::Instruction::getParent]. *)
external instr_parent : llvalue -> llbasicblock = "LLVMGetInstructionParent"

(** [instr_begin bb] returns the first position in the instruction list of the
    basic block [bb]. [instr_begin] and [instr_succ] can be used to iterate over
    the instruction list in order.
    See the method [llvm::BasicBlock::begin]. *)
external instr_begin : llbasicblock -> (llbasicblock, llvalue) llpos
                     = "llvm_instr_begin"

(** [instr_succ i] returns the instruction list position succeeding [Before i].
    See the method [llvm::BasicBlock::iterator::operator++]. *)
external instr_succ : llvalue -> (llbasicblock, llvalue) llpos
                     = "llvm_instr_succ"

(** [iter_instrs f bb] applies function [f] to each of the instructions of basic
    block [bb] in order. Tail recursive. *)
val iter_instrs: (llvalue -> unit) -> llbasicblock -> unit

(** [fold_left_instrs f init bb] is [f (... (f init g1) ...) gN] where
    [g1,...,gN] are the instructions of basic block [bb]. Tail recursive. *)
val fold_left_instrs: ('a -> llvalue -> 'a) -> 'a -> llbasicblock -> 'a

(** [instr_end bb] returns the last position in the instruction list of the
    basic block [bb]. [instr_end] and [instr_pred] can be used to iterate over
    the instruction list in reverse.
    See the method [llvm::BasicBlock::end]. *)
external instr_end : llbasicblock -> (llbasicblock, llvalue) llrev_pos
                     = "llvm_instr_end"

(** [instr_pred i] returns the instruction list position preceding [After i].
    See the method [llvm::BasicBlock::iterator::operator--]. *)
external instr_pred : llvalue -> (llbasicblock, llvalue) llrev_pos
                     = "llvm_instr_pred"

(** [fold_right_instrs f bb init] is [f (... (f init fN) ...) f1] where
    [f1,...,fN] are the instructions of basic block [bb]. Tail recursive. *)
val fold_right_instrs: (llvalue -> 'a -> 'a) -> llbasicblock -> 'a -> 'a


(** {7 Operations on call sites} *)

(** [instruction_call_conv ci] is the calling convention for the call or invoke
    instruction [ci], which may be one of the values from the module
    {!CallConv}. See the method [llvm::CallInst::getCallingConv] and
    [llvm::InvokeInst::getCallingConv]. *)
external instruction_call_conv: llvalue -> int
                              = "llvm_instruction_call_conv"

(** [set_instruction_call_conv cc ci] sets the calling convention for the call
    or invoke instruction [ci] to the integer [cc], which can be one of the
    values from the module {!CallConv}.
    See the method [llvm::CallInst::setCallingConv]
    and [llvm::InvokeInst::setCallingConv]. *)
external set_instruction_call_conv: int -> llvalue -> unit
                                  = "llvm_set_instruction_call_conv"

(** [add_instruction_param_attr ci i a] adds attribute [a] to the [i]th
    parameter of the call or invoke instruction [ci]. [i]=0 denotes the return
    value. *)
external add_instruction_param_attr : llvalue -> int -> Attribute.t -> unit
  = "llvm_add_instruction_param_attr"

(** [remove_instruction_param_attr ci i a] removes attribute [a] from the
    [i]th parameter of the call or invoke instruction [ci]. [i]=0 denotes the
    return value. *)
external remove_instruction_param_attr : llvalue -> int -> Attribute.t -> unit
  = "llvm_remove_instruction_param_attr"

(** {Operations on call instructions (only)} *)

(** [is_tail_call ci] is [true] if the call instruction [ci] is flagged as
    eligible for tail call optimization, [false] otherwise.
    See the method [llvm::CallInst::isTailCall]. *)
external is_tail_call : llvalue -> bool = "llvm_is_tail_call"

(** [set_tail_call tc ci] flags the call instruction [ci] as eligible for tail
    call optimization if [tc] is [true], clears otherwise.
    See the method [llvm::CallInst::setTailCall]. *)
external set_tail_call : bool -> llvalue -> unit = "llvm_set_tail_call"

(** {7 Operations on phi nodes} *)

(** [add_incoming (v, bb) pn] adds the value [v] to the phi node [pn] for use
    with branches from [bb]. See the method [llvm::PHINode::addIncoming]. *)
external add_incoming : (llvalue * llbasicblock) -> llvalue -> unit
                      = "llvm_add_incoming"

(** [incoming pn] returns the list of value-block pairs for phi node [pn].
    See the method [llvm::PHINode::getIncomingValue]. *)
external incoming : llvalue -> (llvalue * llbasicblock) list = "llvm_incoming"



(** {6 Instruction builders} *)

(** [builder context] creates an instruction builder with no position in
    the context [context]. It is invalid to use this builder until its position
    is set with {!position_before} or {!position_at_end}. See the constructor
    for [llvm::LLVMBuilder]. *)
external builder : llcontext -> llbuilder = "llvm_builder"

(** [builder_at ip] creates an instruction builder positioned at [ip].
    See the constructor for [llvm::LLVMBuilder]. *)
val builder_at : llcontext -> (llbasicblock, llvalue) llpos -> llbuilder

(** [builder_before ins] creates an instruction builder positioned before the
    instruction [isn]. See the constructor for [llvm::LLVMBuilder]. *)
val builder_before : llcontext -> llvalue -> llbuilder

(** [builder_at_end bb] creates an instruction builder positioned at the end of
    the basic block [bb]. See the constructor for [llvm::LLVMBuilder]. *)
val builder_at_end : llcontext -> llbasicblock -> llbuilder

(** [position_builder ip bb] moves the instruction builder [bb] to the position
    [ip].
    See the constructor for [llvm::LLVMBuilder]. *)
external position_builder : (llbasicblock, llvalue) llpos -> llbuilder -> unit
                          = "llvm_position_builder"

(** [position_before ins b] moves the instruction builder [b] to before the
    instruction [isn]. See the method [llvm::LLVMBuilder::SetInsertPoint]. *)
val position_before : llvalue -> llbuilder -> unit

(** [position_at_end bb b] moves the instruction builder [b] to the end of the
    basic block [bb]. See the method [llvm::LLVMBuilder::SetInsertPoint]. *)
val position_at_end : llbasicblock -> llbuilder -> unit

(** [insertion_block b] returns the basic block that the builder [b] is
    positioned to insert into. Raises [Not_Found] if the instruction builder is
    uninitialized.
    See the method [llvm::LLVMBuilder::GetInsertBlock]. *)
external insertion_block : llbuilder -> llbasicblock = "llvm_insertion_block"

(** [insert_into_builder i name b] inserts the specified instruction [i] at the
    position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::Insert]. *)
external insert_into_builder : llvalue -> string -> llbuilder -> unit
                             = "llvm_insert_into_builder"

(** {7 Metadata} *)

(** [set_current_debug_location b md] sets the current debug location [md] in
    the builder [b].
    See the method [llvm::IRBuilder::SetDebugLocation]. *)
external set_current_debug_location : llbuilder -> llvalue -> unit
                                    = "llvm_set_current_debug_location"

(** [clear_current_debug_location b] clears the current debug location in the
    builder [b]. *)
external clear_current_debug_location : llbuilder -> unit
                                      = "llvm_clear_current_debug_location"

(** [current_debug_location b] returns the current debug location, or None
    if none is currently set.
    See the method [llvm::IRBuilder::GetDebugLocation]. *)
external current_debug_location : llbuilder -> llvalue option
                                = "llvm_current_debug_location"

(** [set_inst_debug_location b i] sets the current debug location of the builder
    [b] to the instruction [i].
    See the method [llvm::IRBuilder::SetInstDebugLocation]. *)
external set_inst_debug_location : llbuilder -> llvalue -> unit
                                 = "llvm_set_inst_debug_location"

(** {7 Terminators} *)

(** [build_ret_void b] creates a
    [ret void]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateRetVoid]. *)
external build_ret_void : llbuilder -> llvalue = "llvm_build_ret_void"

(** [build_ret v b] creates a
    [ret %v]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateRet]. *)
external build_ret : llvalue -> llbuilder -> llvalue = "llvm_build_ret"

(** [build_aggregate_ret vs b] creates a
    [ret {...} { %v1, %v2, ... } ]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAggregateRet]. *)
external build_aggregate_ret : llvalue array -> llbuilder -> llvalue
                             = "llvm_build_aggregate_ret"

(** [build_br bb b] creates a
    [br %bb]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateBr]. *)
external build_br : llbasicblock -> llbuilder -> llvalue = "llvm_build_br"

(** [build_cond_br cond tbb fbb b] creates a
    [br %cond, %tbb, %fbb]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateCondBr]. *)
external build_cond_br : llvalue -> llbasicblock -> llbasicblock -> llbuilder ->
                         llvalue = "llvm_build_cond_br"

(** [build_switch case elsebb count b] creates an empty
    [switch %case, %elsebb]
    instruction at the position specified by the instruction builder [b] with
    space reserved for [count] cases.
    See the method [llvm::LLVMBuilder::CreateSwitch]. *)
external build_switch : llvalue -> llbasicblock -> int -> llbuilder -> llvalue
                      = "llvm_build_switch"

(** [add_case sw onval bb] causes switch instruction [sw] to branch to [bb]
    when its input matches the constant [onval].
    See the method [llvm::SwitchInst::addCase]. **)
external add_case : llvalue -> llvalue -> llbasicblock -> unit
                  = "llvm_add_case"

(** [build_indirect_br addr count b] creates a
    [indirectbr %addr]
    instruction at the position specified by the instruction builder [b] with
    space reserved for [count] destinations.
    See the method [llvm::LLVMBuilder::CreateIndirectBr]. *)
external build_indirect_br : llvalue -> int -> llbuilder -> llvalue
                           = "llvm_build_indirect_br"

(** [add_destination br bb] adds the basic block [bb] as a possible branch
    location for the indirectbr instruction [br].
    See the method [llvm::IndirectBrInst::addDestination]. **)
external add_destination : llvalue -> llbasicblock -> unit
                         = "llvm_add_destination"

(** [build_invoke fn args tobb unwindbb name b] creates an
    [%name = invoke %fn(args) to %tobb unwind %unwindbb]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateInvoke]. *)
external build_invoke : llvalue -> llvalue array -> llbasicblock ->
                        llbasicblock -> string -> llbuilder -> llvalue
                      = "llvm_build_invoke_bc" "llvm_build_invoke_nat"

(** [build_unwind b] creates an
    [unwind]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUnwind]. *)
external build_unwind : llbuilder -> llvalue = "llvm_build_unwind"

(** [build_unreachable b] creates an
    [unreachable]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUnwind]. *)
external build_unreachable : llbuilder -> llvalue = "llvm_build_unreachable"


(** {7 Arithmetic} *)

(** [build_add x y name b] creates a
    [%name = add %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAdd]. *)
external build_add : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_add"

(** [build_nsw_add x y name b] creates a
    [%name = nsw add %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNSWAdd]. *)
external build_nsw_add : llvalue -> llvalue -> string -> llbuilder -> llvalue
                      = "llvm_build_nsw_add"

(** [build_nuw_add x y name b] creates a
    [%name = nuw add %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNUWAdd]. *)
external build_nuw_add : llvalue -> llvalue -> string -> llbuilder -> llvalue
                      = "llvm_build_nuw_add"

(** [build_fadd x y name b] creates a
    [%name = fadd %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFAdd]. *)
external build_fadd : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_fadd"

(** [build_sub x y name b] creates a
    [%name = sub %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSub]. *)
external build_sub : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_sub"

(** [build_nsw_sub x y name b] creates a
    [%name = nsw sub %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNSWSub]. *)
external build_nsw_sub : llvalue -> llvalue -> string -> llbuilder -> llvalue
                       = "llvm_build_nsw_sub"

(** [build_nuw_sub x y name b] creates a
    [%name = nuw sub %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNUWSub]. *)
external build_nuw_sub : llvalue -> llvalue -> string -> llbuilder -> llvalue
                       = "llvm_build_nuw_sub"

(** [build_fsub x y name b] creates a
    [%name = fsub %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFSub]. *)
external build_fsub : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_fsub"

(** [build_mul x y name b] creates a
    [%name = mul %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateMul]. *)
external build_mul : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_mul"

(** [build_nsw_mul x y name b] creates a
    [%name = nsw mul %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNSWMul]. *)
external build_nsw_mul : llvalue -> llvalue -> string -> llbuilder -> llvalue
                       = "llvm_build_nsw_mul"

(** [build_nuw_mul x y name b] creates a
    [%name = nuw mul %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateNUWMul]. *)
external build_nuw_mul : llvalue -> llvalue -> string -> llbuilder -> llvalue
                       = "llvm_build_nuw_mul"

(** [build_fmul x y name b] creates a
    [%name = fmul %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFMul]. *)
external build_fmul : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_fmul"

(** [build_udiv x y name b] creates a
    [%name = udiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUDiv]. *)
external build_udiv : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_udiv"

(** [build_sdiv x y name b] creates a
    [%name = sdiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSDiv]. *)
external build_sdiv : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_sdiv"

(** [build_exact_sdiv x y name b] creates a
    [%name = exact sdiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateExactSDiv]. *)
external build_exact_sdiv : llvalue -> llvalue -> string -> llbuilder -> llvalue
                          = "llvm_build_exact_sdiv"

(** [build_fdiv x y name b] creates a
    [%name = fdiv %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFDiv]. *)
external build_fdiv : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_fdiv"

(** [build_urem x y name b] creates a
    [%name = urem %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateURem]. *)
external build_urem : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_urem"

(** [build_SRem x y name b] creates a
    [%name = srem %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSRem]. *)
external build_srem : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_srem"

(** [build_frem x y name b] creates a
    [%name = frem %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFRem]. *)
external build_frem : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_frem"

(** [build_shl x y name b] creates a
    [%name = shl %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateShl]. *)
external build_shl : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_shl"

(** [build_lshr x y name b] creates a
    [%name = lshr %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateLShr]. *)
external build_lshr : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_lshr"

(** [build_ashr x y name b] creates a
    [%name = ashr %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAShr]. *)
external build_ashr : llvalue -> llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_ashr"

(** [build_and x y name b] creates a
    [%name = and %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAnd]. *)
external build_and : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_and"

(** [build_or x y name b] creates a
    [%name = or %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateOr]. *)
external build_or : llvalue -> llvalue -> string -> llbuilder -> llvalue
                  = "llvm_build_or"

(** [build_xor x y name b] creates a
    [%name = xor %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateXor]. *)
external build_xor : llvalue -> llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_xor"

(** [build_neg x name b] creates a
    [%name = sub 0, %x]
    instruction at the position specified by the instruction builder [b].
    [-0.0] is used for floating point types to compute the correct sign.
    See the method [llvm::LLVMBuilder::CreateNeg]. *)
external build_neg : llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_neg"

(** [build_nsw_neg x name b] creates a
    [%name = nsw sub 0, %x]
    instruction at the position specified by the instruction builder [b].
    [-0.0] is used for floating point types to compute the correct sign.
    See the method [llvm::LLVMBuilder::CreateNeg]. *)
external build_nsw_neg : llvalue -> string -> llbuilder -> llvalue
                       = "llvm_build_nsw_neg"

(** [build_nuw_neg x name b] creates a
    [%name = nuw sub 0, %x]
    instruction at the position specified by the instruction builder [b].
    [-0.0] is used for floating point types to compute the correct sign.
    See the method [llvm::LLVMBuilder::CreateNeg]. *)
external build_nuw_neg : llvalue -> string -> llbuilder -> llvalue
                       = "llvm_build_nuw_neg"

(** [build_fneg x name b] creates a
    [%name = fsub 0, %x]
    instruction at the position specified by the instruction builder [b].
    [-0.0] is used for floating point types to compute the correct sign.
    See the method [llvm::LLVMBuilder::CreateFNeg]. *)
external build_fneg : llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_fneg"

(** [build_xor x name b] creates a
    [%name = xor %x, -1]
    instruction at the position specified by the instruction builder [b].
    [-1] is the correct "all ones" value for the type of [x].
    See the method [llvm::LLVMBuilder::CreateXor]. *)
external build_not : llvalue -> string -> llbuilder -> llvalue
                   = "llvm_build_not"


(** {7 Memory} *)

(** [build_alloca ty name b] creates a
    [%name = alloca %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAlloca]. *)
external build_alloca : lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_alloca"

(** [build_array_alloca ty n name b] creates a
    [%name = alloca %ty, %n]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateAlloca]. *)
external build_array_alloca : lltype -> llvalue -> string -> llbuilder ->
                              llvalue = "llvm_build_array_alloca"

(** [build_load v name b] creates a
    [%name = load %v]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateLoad]. *)
external build_load : llvalue -> string -> llbuilder -> llvalue
                    = "llvm_build_load"

(** [build_store v p b] creates a
    [store %v, %p]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateStore]. *)
external build_store : llvalue -> llvalue -> llbuilder -> llvalue
                     = "llvm_build_store"

(** [build_gep p indices name b] creates a
    [%name = getelementptr %p, indices...]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateGetElementPtr]. *)
external build_gep : llvalue -> llvalue array -> string -> llbuilder -> llvalue
                   = "llvm_build_gep"

(** [build_in_bounds_gep p indices name b] creates a
    [%name = gelementptr inbounds %p, indices...]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateInBoundsGetElementPtr]. *)
external build_in_bounds_gep : llvalue -> llvalue array -> string -> llbuilder ->
                               llvalue = "llvm_build_in_bounds_gep"

(** [build_struct_gep p idx name b] creates a
    [%name = getelementptr %p, 0, idx]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateStructGetElementPtr]. *)
external build_struct_gep : llvalue -> int -> string -> llbuilder ->
                            llvalue = "llvm_build_struct_gep"

(** [build_global_string str name b] creates a series of instructions that adds
    a global string at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateGlobalString]. *)
external build_global_string : string -> string -> llbuilder -> llvalue
                             = "llvm_build_global_string"

(** [build_global_stringptr str name b] creates a series of instructions that
    adds a global string pointer at the position specified by the instruction
    builder [b].
    See the method [llvm::LLVMBuilder::CreateGlobalStringPtr]. *)
external build_global_stringptr : string -> string -> llbuilder -> llvalue
                                = "llvm_build_global_stringptr"


(** {7 Casts} *)

(** [build_trunc v ty name b] creates a
    [%name = trunc %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateTrunc]. *)
external build_trunc : llvalue -> lltype -> string -> llbuilder -> llvalue
                     = "llvm_build_trunc"

(** [build_zext v ty name b] creates a
    [%name = zext %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateZExt]. *)
external build_zext : llvalue -> lltype -> string -> llbuilder -> llvalue
                    = "llvm_build_zext"

(** [build_sext v ty name b] creates a
    [%name = sext %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSExt]. *)
external build_sext : llvalue -> lltype -> string -> llbuilder -> llvalue
                    = "llvm_build_sext"

(** [build_fptoui v ty name b] creates a
    [%name = fptoui %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPToUI]. *)
external build_fptoui : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_fptoui"

(** [build_fptosi v ty name b] creates a
    [%name = fptosi %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPToSI]. *)
external build_fptosi : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_fptosi"

(** [build_uitofp v ty name b] creates a
    [%name = uitofp %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateUIToFP]. *)
external build_uitofp : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_uitofp"

(** [build_sitofp v ty name b] creates a
    [%name = sitofp %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSIToFP]. *)
external build_sitofp : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_sitofp"

(** [build_fptrunc v ty name b] creates a
    [%name = fptrunc %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPTrunc]. *)
external build_fptrunc : llvalue -> lltype -> string -> llbuilder -> llvalue
                       = "llvm_build_fptrunc"

(** [build_fpext v ty name b] creates a
    [%name = fpext %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPExt]. *)
external build_fpext : llvalue -> lltype -> string -> llbuilder -> llvalue
                     = "llvm_build_fpext"

(** [build_ptrtoint v ty name b] creates a
    [%name = prtotint %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreatePtrToInt]. *)
external build_ptrtoint : llvalue -> lltype -> string -> llbuilder -> llvalue
                        = "llvm_build_prttoint"

(** [build_inttoptr v ty name b] creates a
    [%name = inttoptr %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateIntToPtr]. *)
external build_inttoptr : llvalue -> lltype -> string -> llbuilder -> llvalue
                        = "llvm_build_inttoptr"

(** [build_bitcast v ty name b] creates a
    [%name = bitcast %p to %ty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateBitCast]. *)
external build_bitcast : llvalue -> lltype -> string -> llbuilder -> llvalue
                       = "llvm_build_bitcast"

(** [build_zext_or_bitcast v ty name b] creates a zext or bitcast
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateZExtOrBitCast]. *)
external build_zext_or_bitcast : llvalue -> lltype -> string -> llbuilder ->
                                 llvalue = "llvm_build_zext_or_bitcast"

(** [build_sext_or_bitcast v ty name b] creates a sext or bitcast
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSExtOrBitCast]. *)
external build_sext_or_bitcast : llvalue -> lltype -> string -> llbuilder ->
                                 llvalue = "llvm_build_sext_or_bitcast"

(** [build_trunc_or_bitcast v ty name b] creates a trunc or bitcast
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateZExtOrBitCast]. *)
external build_trunc_or_bitcast : llvalue -> lltype -> string -> llbuilder ->
                                  llvalue = "llvm_build_trunc_or_bitcast"

(** [build_pointercast v ty name b] creates a bitcast or pointer-to-int
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreatePointerCast]. *)
external build_pointercast : llvalue -> lltype -> string -> llbuilder -> llvalue
                           = "llvm_build_pointercast"

(** [build_intcast v ty name b] creates a zext, bitcast, or trunc
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateIntCast]. *)
external build_intcast : llvalue -> lltype -> string -> llbuilder -> llvalue
                       = "llvm_build_intcast"

(** [build_fpcast v ty name b] creates a fpext, bitcast, or fptrunc
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFPCast]. *)
external build_fpcast : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_fpcast"


(** {7 Comparisons} *)

(** [build_icmp pred x y name b] creates a
    [%name = icmp %pred %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateICmp]. *)
external build_icmp : Icmp.t -> llvalue -> llvalue -> string ->
                      llbuilder -> llvalue = "llvm_build_icmp"

(** [build_fcmp pred x y name b] creates a
    [%name = fcmp %pred %x, %y]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateFCmp]. *)
external build_fcmp : Fcmp.t -> llvalue -> llvalue -> string ->
                      llbuilder -> llvalue = "llvm_build_fcmp"


(** {7 Miscellaneous instructions} *)

(** [build_phi incoming name b] creates a
    [%name = phi %incoming]
    instruction at the position specified by the instruction builder [b].
    [incoming] is a list of [(llvalue, llbasicblock)] tuples.
    See the method [llvm::LLVMBuilder::CreatePHI]. *)
external build_phi : (llvalue * llbasicblock) list -> string -> llbuilder ->
                     llvalue = "llvm_build_phi"

(** [build_call fn args name b] creates a
    [%name = call %fn(args...)]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateCall]. *)
external build_call : llvalue -> llvalue array -> string -> llbuilder -> llvalue
                    = "llvm_build_call"

(** [build_select cond thenv elsev name b] creates a
    [%name = select %cond, %thenv, %elsev]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateSelect]. *)
external build_select : llvalue -> llvalue -> llvalue -> string -> llbuilder ->
                        llvalue = "llvm_build_select"

(** [build_va_arg valist argty name b] creates a
    [%name = va_arg %valist, %argty]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateVAArg]. *)
external build_va_arg : llvalue -> lltype -> string -> llbuilder -> llvalue
                      = "llvm_build_va_arg"

(** [build_extractelement vec i name b] creates a
    [%name = extractelement %vec, %i]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateExtractElement]. *)
external build_extractelement : llvalue -> llvalue -> string -> llbuilder ->
                                llvalue = "llvm_build_extractelement"

(** [build_insertelement vec elt i name b] creates a
    [%name = insertelement %vec, %elt, %i]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateInsertElement]. *)
external build_insertelement : llvalue -> llvalue -> llvalue -> string ->
                               llbuilder -> llvalue = "llvm_build_insertelement"

(** [build_shufflevector veca vecb mask name b] creates a
    [%name = shufflevector %veca, %vecb, %mask]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateShuffleVector]. *)
external build_shufflevector : llvalue -> llvalue -> llvalue -> string ->
                               llbuilder -> llvalue = "llvm_build_shufflevector"

(** [build_insertvalue agg idx name b] creates a
    [%name = extractvalue %agg, %idx]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateExtractValue]. *)
external build_extractvalue : llvalue -> int -> string -> llbuilder -> llvalue
                            = "llvm_build_extractvalue"

(** [build_insertvalue agg val idx name b] creates a
    [%name = insertvalue %agg, %val, %idx]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateInsertValue]. *)
external build_insertvalue : llvalue -> llvalue -> int -> string -> llbuilder ->
                             llvalue = "llvm_build_insertvalue"

(** [build_is_null val name b] creates a
    [%name = icmp eq %val, null]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateIsNull]. *)
external build_is_null : llvalue -> string -> llbuilder -> llvalue
                       = "llvm_build_is_null"

(** [build_is_not_null val name b] creates a
    [%name = icmp ne %val, null]
    instruction at the position specified by the instruction builder [b].
    See the method [llvm::LLVMBuilder::CreateIsNotNull]. *)
external build_is_not_null : llvalue -> string -> llbuilder -> llvalue
                           = "llvm_build_is_not_null"

(** [build_ptrdiff lhs rhs name b] creates a series of instructions that measure
    the difference between two pointer values at the position specified by the
    instruction builder [b].
    See the method [llvm::LLVMBuilder::CreatePtrDiff]. *)
external build_ptrdiff : llvalue -> llvalue -> string -> llbuilder -> llvalue
                       = "llvm_build_ptrdiff"


(** {6 Memory buffers} *)

module MemoryBuffer : sig
  (** [of_file p] is the memory buffer containing the contents of the file at
      path [p]. If the file could not be read, then [IoError msg] is
      raised. *)
  external of_file : string -> llmemorybuffer = "llvm_memorybuffer_of_file"
  
  (** [stdin ()] is the memory buffer containing the contents of standard input.
      If standard input is empty, then [IoError msg] is raised. *)
  external of_stdin : unit -> llmemorybuffer = "llvm_memorybuffer_of_stdin"
  
  (** Disposes of a memory buffer. *)
  external dispose : llmemorybuffer -> unit = "llvm_memorybuffer_dispose"
end


(** {6 Pass Managers} *)

module PassManager : sig
  (**  *)
  type 'a t
  type any = [ `Module | `Function ]
  
  (** [PassManager.create ()] constructs a new whole-module pass pipeline. This
      type of pipeline is suitable for link-time optimization and whole-module
      transformations.
      See the constructor of [llvm::PassManager]. *)
  external create : unit -> [ `Module ] t = "llvm_passmanager_create"
  
  (** [PassManager.create_function m] constructs a new function-by-function
      pass pipeline over the module [m]. It does not take ownership of [m].
      This type of pipeline is suitable for code generation and JIT compilation
      tasks.
      See the constructor of [llvm::FunctionPassManager]. *)
  external create_function : llmodule -> [ `Function ] t
                           = "LLVMCreateFunctionPassManager"
  
  (** [run_module m pm] initializes, executes on the module [m], and finalizes
      all of the passes scheduled in the pass manager [pm]. Returns [true] if
      any of the passes modified the module, [false] otherwise.
      See the [llvm::PassManager::run] method. *)
  external run_module : llmodule -> [ `Module ] t -> bool
                      = "llvm_passmanager_run_module"
  
  (** [initialize fpm] initializes all of the function passes scheduled in the
      function pass manager [fpm]. Returns [true] if any of the passes modified
      the module, [false] otherwise.
      See the [llvm::FunctionPassManager::doInitialization] method. *)
  external initialize : [ `Function ] t -> bool = "llvm_passmanager_initialize"
  
  (** [run_function f fpm] executes all of the function passes scheduled in the
      function pass manager [fpm] over the function [f]. Returns [true] if any
      of the passes modified [f], [false] otherwise.
      See the [llvm::FunctionPassManager::run] method. *)
  external run_function : llvalue -> [ `Function ] t -> bool
                        = "llvm_passmanager_run_function"
  
  (** [finalize fpm] finalizes all of the function passes scheduled in in the
      function pass manager [fpm]. Returns [true] if any of the passes
      modified the module, [false] otherwise.
      See the [llvm::FunctionPassManager::doFinalization] method. *)
  external finalize : [ `Function ] t -> bool = "llvm_passmanager_finalize"
  
  (** Frees the memory of a pass pipeline. For function pipelines, does not free
      the module.
      See the destructor of [llvm::BasePassManager]. *)
  external dispose : [< any ] t -> unit = "llvm_passmanager_dispose"
end
