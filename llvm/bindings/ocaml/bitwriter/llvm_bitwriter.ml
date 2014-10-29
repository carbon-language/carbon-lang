(*===-- llvm_bitwriter.ml - LLVM OCaml Interface --------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===
 *
 * This interface provides an OCaml API for the LLVM intermediate
 * representation, the classes in the VMCore library.
 *
 *===----------------------------------------------------------------------===*)

external write_bitcode_file
  : Llvm.llmodule -> string -> bool
  = "llvm_write_bitcode_file"

external write_bitcode_to_fd
  : ?unbuffered:bool -> Llvm.llmodule -> Unix.file_descr -> bool
  = "llvm_write_bitcode_to_fd"

external write_bitcode_to_memory_buffer
  : Llvm.llmodule -> Llvm.llmemorybuffer
  = "llvm_write_bitcode_to_memory_buffer"

let output_bitcode ?unbuffered channel m =
  write_bitcode_to_fd ?unbuffered m (Unix.descr_of_out_channel channel)
