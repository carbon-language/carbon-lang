(*===-- llvm_bitreader.ml - LLVM Ocaml Interface ----------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by Gordon Henriksen and is distributed under the
 * University of Illinois Open Source License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)


type bitreader_result =
| Bitreader_success of Llvm.llmodule
| Bitreader_failure of string


external read_bitcode_file : string -> bitreader_result
                           = "llvm_read_bitcode_file"
