(*===-- llvm_irreader.ml - LLVM OCaml Interface ---------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)


exception Error of string

external register_exns : exn -> unit = "llvm_register_irreader_exns"
let _ = register_exns (Error "")

external parse_ir : Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule
                  = "llvm_parse_ir"
