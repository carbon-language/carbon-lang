(*===-- llvm_executionengine.mli - LLVM OCaml Interface -------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** JIT Interpreter.

    This interface provides an OCaml API for LLVM execution engine (JIT/
    interpreter), the classes in the [ExecutionEngine] library. *)

exception Error of string

(** [initialize ()] initializes the backend corresponding to the host.
    Returns [true] if initialization is successful; [false] indicates
    that there is no such backend or it is unable to emit object code
    via MCJIT. *)
val initialize : unit -> bool

(** An execution engine is either a JIT compiler or an interpreter, capable of
    directly loading an LLVM module and executing its functions without first
    invoking a static compiler and generating a native executable. *)
type llexecutionengine

(** MCJIT compiler options. See [llvm::TargetOptions]. *)
type llcompileroptions = {
  opt_level: int;
  code_model: Llvm_target.CodeModel.t;
  no_framepointer_elim: bool;
  enable_fast_isel: bool;
}

(** Default MCJIT compiler options:
    [{ opt_level = 0; code_model = CodeModel.JIT_default;
       no_framepointer_elim = false; enable_fast_isel = false }] *)
val default_compiler_options : llcompileroptions

(** [create m optlevel] creates a new MCJIT just-in-time compiler, taking
    ownership of the module [m] if successful with the desired optimization
    level [optlevel]. Raises [Error msg] if an error occurrs. The execution
    engine is not garbage collected and must be destroyed with [dispose ee].

    Run {!initialize} before using this function.

    See the function [llvm::EngineBuilder::create]. *)
val create : ?options:llcompileroptions -> Llvm.llmodule -> llexecutionengine

(** [dispose ee] releases the memory used by the execution engine and must be
    invoked to avoid memory leaks. *)
val dispose : llexecutionengine -> unit

(** [add_module m ee] adds the module [m] to the execution engine [ee]. *)
val add_module : Llvm.llmodule -> llexecutionengine -> unit

(** [remove_module m ee] removes the module [m] from the execution engine
    [ee]. Raises [Error msg] if an error occurs. *)
val remove_module : Llvm.llmodule -> llexecutionengine -> unit

(** [run_static_ctors ee] executes the static constructors of each module in
    the execution engine [ee]. *)
val run_static_ctors : llexecutionengine -> unit

(** [run_static_dtors ee] executes the static destructors of each module in
    the execution engine [ee]. *)
val run_static_dtors : llexecutionengine -> unit

(** [data_layout ee] is the data layout of the execution engine [ee]. *)
val data_layout : llexecutionengine -> Llvm_target.DataLayout.t

(** [add_global_mapping gv ptr ee] tells the execution engine [ee] that
    the global [gv] is at the specified location [ptr], which must outlive
    [gv] and [ee].
    All uses of [gv] in the compiled code will refer to [ptr]. *)
val add_global_mapping : Llvm.llvalue -> 'a Ctypes.ptr -> llexecutionengine -> unit

(** [get_pointer_to_global gv typ ee] returns the value of the global
    variable [gv] in the execution engine [ee] as type [typ], which may
    be a pointer type (e.g. [int ptr typ]) for global variables or
    a function (e.g. [(int -> int) typ]) type for functions, and which
    will be live as long as [gv] and [ee] are. *)
val get_pointer_to_global : Llvm.llvalue -> 'a Ctypes.typ -> llexecutionengine -> 'a
