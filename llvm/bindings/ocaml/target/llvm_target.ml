(*===-- llvm_target.ml - LLVM OCaml Interface ------------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

module Endian = struct
  type t =
  | Big
  | Little
end

module DataLayout = struct
  type t

  external of_string : string -> t = "llvm_datalayout_of_string"
  external as_string : t -> string = "llvm_datalayout_as_string"
  external add_to_pass_manager : [<Llvm.PassManager.any]
                                 Llvm.PassManager.t -> t -> unit
                               = "llvm_datalayout_add_to_pass_manager"
  external byte_order : t -> Endian.t = "llvm_datalayout_byte_order"
  external pointer_size : t -> int = "llvm_datalayout_pointer_size"
  external intptr_type : Llvm.llcontext -> t -> Llvm.lltype
                       = "llvm_datalayout_intptr_type"
  external qualified_pointer_size : int -> t -> int
                                  = "llvm_datalayout_qualified_pointer_size"
  external qualified_intptr_type : Llvm.llcontext -> int -> t -> Llvm.lltype
                                 = "llvm_datalayout_qualified_intptr_type"
  external size_in_bits : Llvm.lltype -> t -> Int64.t
                        = "llvm_datalayout_size_in_bits"
  external store_size : Llvm.lltype -> t -> Int64.t
                      = "llvm_datalayout_store_size"
  external abi_size : Llvm.lltype -> t -> Int64.t
                    = "llvm_datalayout_abi_size"
  external abi_align : Llvm.lltype -> t -> int
                     = "llvm_datalayout_abi_align"
  external stack_align : Llvm.lltype -> t -> int
                       = "llvm_datalayout_stack_align"
  external preferred_align : Llvm.lltype -> t -> int
                           = "llvm_datalayout_preferred_align"
  external preferred_align_of_global : Llvm.llvalue -> t -> int
                                   = "llvm_datalayout_preferred_align_of_global"
  external element_at_offset : Llvm.lltype -> Int64.t -> t -> int
                             = "llvm_datalayout_element_at_offset"
  external offset_of_element : Llvm.lltype -> int -> t -> Int64.t
                             = "llvm_datalayout_offset_of_element"
end

