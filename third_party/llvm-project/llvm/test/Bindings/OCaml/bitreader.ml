(* RUN: rm -rf %t && mkdir -p %t && cp %s %t/bitreader.ml
 * RUN: %ocamlc -g -w +A -package llvm.bitreader -package llvm.bitwriter -linkpkg %t/bitreader.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc
 * RUN: %ocamlopt -g -w +A -package llvm.bitreader -package llvm.bitwriter -linkpkg %t/bitreader.ml -o %t/executable
 * RUN: %t/executable %t/bitcode.bc
 * RUN: llvm-dis < %t/bitcode.bc
 * XFAIL: vg_leak
 *)

(* Note that this takes a moment to link, so it's best to keep the number of
   individual tests low. *)

let context = Llvm.global_context ()

let diagnostic_handler _ = ()

let test x = if not x then exit 1 else ()

let _ =
  Llvm.set_diagnostic_handler context (Some diagnostic_handler);

  let fn = Sys.argv.(1) in
  let m = Llvm.create_module context "ocaml_test_module" in

  test (Llvm_bitwriter.write_bitcode_file m fn);

  Llvm.dispose_module m;

  (* parse_bitcode *)
  begin
    let mb = Llvm.MemoryBuffer.of_file fn in
    begin try
      let m = Llvm_bitreader.parse_bitcode context mb in
      Llvm.dispose_module m
    with x ->
      Llvm.MemoryBuffer.dispose mb;
      raise x
    end
  end;

  (* MemoryBuffer.of_file *)
  test begin try
    let mb = Llvm.MemoryBuffer.of_file (fn ^ ".bogus") in
    Llvm.MemoryBuffer.dispose mb;
    false
  with Llvm.IoError _ ->
    true
  end;

  (* get_module *)
  begin
    let mb = Llvm.MemoryBuffer.of_file fn in
    let m = begin try
      Llvm_bitreader.get_module context mb
    with x ->
      Llvm.MemoryBuffer.dispose mb;
      raise x
    end in
    Llvm.dispose_module m
  end;

  (* corrupt the bitcode *)
  let fn = fn ^ ".txt" in
  begin let oc = open_out fn in
    output_string oc "not a bitcode file\n";
    close_out oc
  end;

  (* test get_module exceptions *)
  test begin
    try
      let mb = Llvm.MemoryBuffer.of_file fn in
      let m = begin try
        Llvm_bitreader.get_module context mb
      with x ->
        Llvm.MemoryBuffer.dispose mb;
        raise x
      end in
      Llvm.dispose_module m;
      false
    with Llvm_bitreader.Error _ ->
      true
  end
