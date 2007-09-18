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
type llmodule
type lltype
type llvalue

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

type visibility =
  Default_visibility
| Hidden_visibility
| Protected_visibility


(*===-- Modules -----------------------------------------------------------===*)

(* Creates a module with the supplied module ID. Modules are not garbage
   collected; it is mandatory to call dispose_module to free memory. *)
external create_module : string -> llmodule = "llvm_create_module"

(* Disposes a module. All references to subordinate objects are invalidated;
   referencing them will invoke undefined behavior. *)
external dispose_module : llmodule -> unit = "llvm_dispose_module"

(* Adds a named type to the module's symbol table. Returns true if successful.
   If such a name already exists, then no entry is added and returns false. *)
external add_type_name : string -> lltype -> llmodule -> bool
                       = "llvm_add_type_name"


(*===-- Types -------------------------------------------------------------===*)
external classify_type : lltype -> type_kind = "llvm_classify_type"
external refine_abstract_type : lltype -> lltype -> unit
                              = "llvm_refine_abstract_type"
val string_of_lltype : lltype -> string

(*--... Operations on integer types ........................................--*)
val i1_type : lltype
val i8_type : lltype
val i16_type : lltype
val i32_type : lltype
val i64_type : lltype
external make_integer_type : int -> lltype = "llvm_make_integer_type"
external integer_bitwidth : lltype -> int = "llvm_integer_bitwidth"

(*--... Operations on real types ...........................................--*)
val float_type : lltype
val double_type : lltype
val x86fp80_type : lltype
val fp128_type : lltype
val ppc_fp128_type : lltype

(*--... Operations on function types .......................................--*)
(* FIXME: handle parameter attributes                                         *)
external make_function_type : lltype -> lltype array -> bool -> lltype
                            = "llvm_make_function_type"
external is_var_arg : lltype -> bool = "llvm_is_var_arg"
external return_type : lltype -> lltype = "llvm_return_type"
external param_types : lltype -> lltype array = "llvm_param_types"

(*--... Operations on struct types .........................................--*)
external make_struct_type : lltype array -> bool -> lltype
                          = "llvm_make_struct_type"
external element_types : lltype -> lltype array = "llvm_element_types"
external is_packed : lltype -> bool = "llvm_is_packed"

(*--... Operations on pointer, vector, and array types .....................--*)
external make_array_type : lltype -> int -> lltype = "llvm_make_array_type"
external make_pointer_type : lltype -> lltype = "llvm_make_pointer_type"
external make_vector_type : lltype -> int -> lltype = "llvm_make_vector_type"

external element_type : lltype -> lltype = "llvm_element_type"
external array_length : lltype -> int = "llvm_array_length"
external vector_size : lltype -> int = "llvm_vector_size"

(*--... Operations on other types ..........................................--*)
external make_opaque_type : unit -> lltype = "llvm_make_opaque_type"
val void_type : lltype
val label_type : lltype


(*===-- Values ------------------------------------------------------------===*)
external type_of : llvalue -> lltype = "llvm_type_of"
external value_name : llvalue -> string = "llvm_value_name"
external set_value_name : string -> llvalue -> unit = "llvm_set_value_name"

(*--... Operations on constants of (mostly) any type .......................--*)
external is_constant : llvalue -> bool = "llvm_is_constant"
external make_null : lltype -> llvalue = "llvm_make_null"
external make_all_ones : (*int|vec*)lltype -> llvalue = "llvm_make_all_ones"
external make_undef : lltype -> llvalue = "llvm_make_undef"
external is_null : llvalue -> bool = "llvm_is_null"
external is_undef : llvalue -> bool = "llvm_is_undef"

(*--... Operations on scalar constants .....................................--*)
external make_int_constant : lltype -> int -> bool -> llvalue
                           = "llvm_make_int_constant"
external make_int64_constant : lltype -> Int64.t -> bool -> llvalue
                             = "llvm_make_int64_constant"
external make_real_constant : lltype -> float -> llvalue
                            = "llvm_make_real_constant"

(*--... Operations on composite constants ..................................--*)
external make_string_constant : string -> bool -> llvalue
                              = "llvm_make_string_constant"
external make_array_constant : lltype -> llvalue array -> llvalue
                             = "llvm_make_array_constant"
external make_struct_constant : llvalue array -> bool -> llvalue
                              = "llvm_make_struct_constant"
external make_vector_constant : llvalue array -> llvalue
                              = "llvm_make_vector_constant"

(*--... Operations on global variables, functions, and aliases (globals) ...--*)
external is_declaration : llvalue -> bool = "llvm_is_declaration"
external linkage : llvalue -> linkage = "llvm_linkage"
external set_linkage : linkage -> llvalue -> unit = "llvm_set_linkage"
external section : llvalue -> string = "llvm_section"
external set_section : string -> llvalue -> unit = "llvm_set_section"
external visibility : llvalue -> visibility = "llvm_visibility"
external set_visibility : visibility -> llvalue -> unit = "llvm_set_visibility"
external alignment : llvalue -> int = "llvm_alignment"
external set_alignment : int -> llvalue -> unit = "llvm_set_alignment"

(*--... Operations on global variables .....................................--*)
external declare_global : lltype -> string -> llmodule -> llvalue
                        = "llvm_declare_global"
external define_global : string -> llvalue -> llmodule -> llvalue
                       = "llvm_define_global"
external delete_global : llvalue -> unit = "llvm_delete_global"
external global_initializer : llvalue -> llvalue = "llvm_global_initializer"
external set_initializer : llvalue -> llvalue -> unit = "llvm_set_initializer"
external remove_initializer : llvalue -> unit = "llvm_remove_initializer"
external is_thread_local : llvalue -> bool = "llvm_is_thread_local"
external set_thread_local : bool -> llvalue -> unit = "llvm_set_thread_local"

