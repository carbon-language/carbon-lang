(*===-- llvm_bitreader.mli - LLVM Ocaml Interface ---------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by Gordon Henriksen and is distributed under the
 * University of Illinois Open Source License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===
 *
 * This interface provides an ocaml API for the LLVM bitcode reader, the
 * classes in the Bitreader library.
 *
 *===----------------------------------------------------------------------===*)


type bitreader_result =
| Bitreader_success of Llvm.llmodule
| Bitreader_failure of string


(** [read_bitcode_file path] reads the bitcode for module [m] from the file at
    [path]. Returns [Reader_success m] if successful, and [Reader_failure msg]
    otherwise, where [msg] is a description of the error encountered. **)
external read_bitcode_file : string -> bitreader_result
                           = "llvm_read_bitcode_file"
