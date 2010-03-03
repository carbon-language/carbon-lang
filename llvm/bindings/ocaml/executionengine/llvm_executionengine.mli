(*===-- llvm_executionengine.mli - LLVM Ocaml Interface ---------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** JIT Interpreter.

    This interface provides an ocaml API for LLVM execution engine (JIT/
    interpreter), the classes in the ExecutionEngine library. *)

exception Error of string

module GenericValue: sig
  (** [GenericValue.t] is a boxed union type used to portably pass arguments to
      and receive values from the execution engine. It supports only a limited
      selection of types; for more complex argument types, it is necessary to
      generate a stub function by hand or to pass parameters by reference.
      See the struct [llvm::GenericValue]. *)
  type t
  
  (** [of_float fpty n] boxes the float [n] in a float-valued generic value
      according to the floating point type [fpty]. See the fields
      [llvm::GenericValue::DoubleVal] and [llvm::GenericValue::FloatVal]. *)
  external of_float : Llvm.lltype -> float -> t = "llvm_genericvalue_of_float"
  
  (** [of_pointer v] boxes the pointer value [v] in a generic value. See the
      field [llvm::GenericValue::PointerVal]. *)
  external of_pointer : 'a -> t = "llvm_genericvalue_of_value"
  
  (** [of_int32 n w] boxes the int32 [i] in a generic value with the bitwidth
      [w]. See the field [llvm::GenericValue::IntVal]. *)
  external of_int32 : Llvm.lltype -> int32 -> t = "llvm_genericvalue_of_int32"
  
  (** [of_int n w] boxes the int [i] in a generic value with the bitwidth
      [w]. See the field [llvm::GenericValue::IntVal]. *)
  external of_int : Llvm.lltype -> int -> t = "llvm_genericvalue_of_int"
  
  (** [of_natint n w] boxes the native int [i] in a generic value with the
      bitwidth [w]. See the field [llvm::GenericValue::IntVal]. *)
  external of_nativeint : Llvm.lltype -> nativeint -> t
                        = "llvm_genericvalue_of_nativeint"

  (** [of_int64 n w] boxes the int64 [i] in a generic value with the bitwidth
      [w]. See the field [llvm::GenericValue::IntVal]. *)
  external of_int64 : Llvm.lltype -> int64 -> t = "llvm_genericvalue_of_int64"

  (** [as_float fpty gv] unboxes the floating point-valued generic value [gv] of
      floating point type [fpty]. See the fields [llvm::GenericValue::DoubleVal]
      and [llvm::GenericValue::FloatVal]. *)
  external as_float : Llvm.lltype -> t -> float = "llvm_genericvalue_as_float"
  
  (** [as_pointer gv] unboxes the pointer-valued generic value [gv]. See the
      field [llvm::GenericValue::PointerVal]. *)
  external as_pointer : t -> 'a = "llvm_genericvalue_as_value"
  
  (** [as_int32 gv] unboxes the integer-valued generic value [gv] as an [int32].
      Is invalid if [gv] has a bitwidth greater than 32 bits. See the field
      [llvm::GenericValue::IntVal]. *)
  external as_int32 : t -> int32 = "llvm_genericvalue_as_int32"
  
  (** [as_int gv] unboxes the integer-valued generic value [gv] as an [int].
      Is invalid if [gv] has a bitwidth greater than the host bit width (but the
      most significant bit may be lost). See the field
      [llvm::GenericValue::IntVal]. *)
  external as_int : t -> int = "llvm_genericvalue_as_int"
  
  (** [as_natint gv] unboxes the integer-valued generic value [gv] as a
      [nativeint]. Is invalid if [gv] has a bitwidth greater than
      [nativeint]. See the field [llvm::GenericValue::IntVal]. *)
  external as_nativeint : t -> nativeint = "llvm_genericvalue_as_int"
  
  (** [as_int64 gv] returns the integer-valued generic value [gv] as an [int64].
      Is invalid if [gv] has a bitwidth greater than [int64]. See the field
      [llvm::GenericValue::IntVal]. *)
  external as_int64 : t -> int64 = "llvm_genericvalue_as_int64"
end


module ExecutionEngine: sig
  (** An execution engine is either a JIT compiler or an interpreter, capable of
      directly loading an LLVM module and executing its functions without first
      invoking a static compiler and generating a native executable. *)
  type t
  
  (** [create m] creates a new execution engine, taking ownership of the
      module [m] if successful. Creates a JIT if possible, else falls back to an
      interpreter. Raises [Error msg] if an error occurrs. The execution engine
      is not garbage collected and must be destroyed with [dispose ee].
      See the function [llvm::EngineBuilder::create]. *)
  external create : Llvm.llmodule -> t = "llvm_ee_create"
  
  (** [create_interpreter m] creates a new interpreter, taking ownership of the
      module [m] if successful. Raises [Error msg] if an error occurrs. The
      execution engine is not garbage collected and must be destroyed with
      [dispose ee].
      See the function [llvm::EngineBuilder::create]. *)
  external create_interpreter : Llvm.llmodule -> t = "llvm_ee_create_interpreter"
  
  (** [create_jit m optlevel] creates a new JIT (just-in-time compiler), taking
      ownership of the module [m] if successful with the desired optimization
      level [optlevel]. Raises [Error msg] if an error occurrs. The execution
      engine is not garbage collected and must be destroyed with [dispose ee].
      See the function [llvm::EngineBuilder::create]. *)
  external create_jit : Llvm.llmodule -> int -> t = "llvm_ee_create_jit"

  (** [dispose ee] releases the memory used by the execution engine and must be
      invoked to avoid memory leaks. *)
  external dispose : t -> unit = "llvm_ee_dispose"
  
  (** [add_module m ee] adds the module [m] to the execution engine [ee]. *)
  external add_module : Llvm.llmodule -> t -> unit = "llvm_ee_add_mp"
  
  (** [remove_module m ee] removes the module [m] from the execution engine
      [ee], disposing of [m] and the module referenced by [mp]. Raises
      [Error msg] if an error occurs. *)
  external remove_module : Llvm.llmodule -> t -> Llvm.llmodule
                         = "llvm_ee_remove_module"
  
  (** [find_function n ee] finds the function named [n] defined in any of the
      modules owned by the execution engine [ee]. Returns [None] if the function
      is not found and [Some f] otherwise. *)
  external find_function : string -> t -> Llvm.llvalue option
                         = "llvm_ee_find_function"
  
  (** [run_function f args ee] synchronously executes the function [f] with the
      arguments [args], which must be compatible with the parameter types. *)
  external run_function : Llvm.llvalue -> GenericValue.t array -> t ->
                     GenericValue.t
                   = "llvm_ee_run_function"
  
  (** [run_static_ctors ee] executes the static constructors of each module in
      the execution engine [ee]. *)
  external run_static_ctors : t -> unit = "llvm_ee_run_static_ctors"
  
  (** [run_static_dtors ee] executes the static destructors of each module in
      the execution engine [ee]. *)
  external run_static_dtors : t -> unit = "llvm_ee_run_static_dtors"
  
  (** [run_function_as_main f args env ee] executes the function [f] as a main
      function, passing it [argv] and [argc] according to the string array
      [args], and [envp] as specified by the array [env]. Returns the integer
      return value of the function. *)
  external run_function_as_main : Llvm.llvalue -> string array ->
                                  (string * string) array -> t -> int
                                = "llvm_ee_run_function_as_main"
  
  (** [free_machine_code f ee] releases the memory in the execution engine [ee]
      used to store the machine code for the function [f]. *)
  external free_machine_code : Llvm.llvalue -> t -> unit
                             = "llvm_ee_free_machine_code"

  (** [target_data ee] is the target data owned by the execution engine
      [ee]. *)
  external target_data : t -> Llvm_target.TargetData.t
                       = "LLVMGetExecutionEngineTargetData"
end

external initialize_native_target : unit -> bool
                                  = "llvm_initialize_native_target"
