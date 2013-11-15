(*===-- llvm_target.mli - LLVM OCaml Interface -----------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Target Information.

    This interface provides an OCaml API for LLVM target information,
    the classes in the Target library. *)

module Endian : sig
  type t =
  | Big
  | Little
end

module DataLayout : sig
  type t

  (** [of_string rep] parses the data layout string representation [rep].
      See the constructor [llvm::DataLayout::DataLayout]. *)
  val of_string : string -> t

  (** [as_string dl] is the string representation of the data layout [dl].
      See the method [llvm::DataLayout::getStringRepresentation]. *)
  val as_string : t -> string

  (** [add_to_pass_manager dl pm] adds the target data [dl] to
      the pass manager [pm].
      See the method [llvm::PassManagerBase::add]. *)
  val add_to_pass_manager : [<Llvm.PassManager.any] Llvm.PassManager.t ->
                            t -> unit

  (** Returns the byte order of a target, either [Endian.Big] or
      [Endian.Little].
      See the method [llvm::DataLayout::isLittleEndian]. *)
  val byte_order : t -> Endian.t

  (** Returns the pointer size in bytes for a target.
      See the method [llvm::DataLayout::getPointerSize]. *)
  val pointer_size : t -> int

  (** Returns the integer type that is the same size as a pointer on a target.
      See the method [llvm::DataLayout::getIntPtrType]. *)
  val intptr_type : Llvm.llcontext -> t -> Llvm.lltype

  (** Returns the pointer size in bytes for a target in a given address space.
      See the method [llvm::DataLayout::getPointerSize]. *)
  val qualified_pointer_size : int -> t -> int

  (** Returns the integer type that is the same size as a pointer on a target
      in a given address space.
      See the method [llvm::DataLayout::getIntPtrType]. *)
  val qualified_intptr_type : Llvm.llcontext -> int -> t -> Llvm.lltype

  (** Computes the size of a type in bits for a target.
      See the method [llvm::DataLayout::getTypeSizeInBits]. *)
  val size_in_bits : Llvm.lltype -> t -> Int64.t

  (** Computes the storage size of a type in bytes for a target.
      See the method [llvm::DataLayout::getTypeStoreSize]. *)
  val store_size : Llvm.lltype -> t -> Int64.t

  (** Computes the ABI size of a type in bytes for a target.
      See the method [llvm::DataLayout::getTypeAllocSize]. *)
  val abi_size : Llvm.lltype -> t -> Int64.t

  (** Computes the ABI alignment of a type in bytes for a target.
      See the method [llvm::DataLayout::getTypeABISize]. *)
  val abi_align : Llvm.lltype -> t -> int

  (** Computes the call frame alignment of a type in bytes for a target.
      See the method [llvm::DataLayout::getTypeABISize]. *)
  val stack_align : Llvm.lltype -> t -> int

  (** Computes the preferred alignment of a type in bytes for a target.
      See the method [llvm::DataLayout::getTypeABISize]. *)
  val preferred_align : Llvm.lltype -> t -> int

  (** Computes the preferred alignment of a global variable in bytes for
      a target. See the method [llvm::DataLayout::getPreferredAlignment]. *)
  val preferred_align_of_global : Llvm.llvalue -> t -> int

  (** Computes the structure element that contains the byte offset for a target.
      See the method [llvm::StructLayout::getElementContainingOffset]. *)
  val element_at_offset : Llvm.lltype -> Int64.t -> t -> int

  (** Computes the byte offset of the indexed struct element for a target.
      See the method [llvm::StructLayout::getElementContainingOffset]. *)
  val offset_of_element : Llvm.lltype -> int -> t -> Int64.t
end
