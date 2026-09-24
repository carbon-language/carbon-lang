// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/language_server/testdata/position/sem_ir_text.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/language_server/testdata/position/sem_ir_text.carbon

// Navigation within the SemIR in a test file's `// CHECK:STDOUT:` lines.
//
// The split below is the document under test, not part of this test: `CHECK`
// lines only form expectations inside the `AUTOUPDATE-SPLIT`, and the
// document's own `// ---` marker is written as `[[@0x2f]]/ ---` so that it
// doesn't split this file.

// --- sem_ir_text.carbon
[[@0x2f]]/ --- a.carbon
library "a";

class C {}
class D {}

//@dump-sem-ir-begin
fn F[T: type](x: T) {}
//@dump-sem-ir-end

fn Run(c: C, d: D) {
  F(c);
  F(d);
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d = wrapper_binding_pattern x, %x.param_patt.91d [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %x.param_patt.0f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.953: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt.0f9 [concrete]
// CHECK:STDOUT:   %x.param_patt.23b: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.c51: %pattern_type.d8d = wrapper_binding_pattern x, %x.param_patt.23b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc7_16.1: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %x.param_patt.loc7_16.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc7_16.1: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc7_16.1 [symbolic = %x.patt.loc7_16.2 (constants.%x.patt.260)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_9.1: type = splice_block %.loc7_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc7_7.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc7_7.1 (%T) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_7.2: type) {
// CHECK:STDOUT:   %T.patt.loc7_7.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc7_7.1 [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %x.param_patt.loc7_16.2: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %x.param_patt.loc7_16.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc7_16.2: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc7_16.2 [symbolic = %x.patt.loc7_16.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc7_7.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc7_7.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %x.param_patt.loc7_16.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc7_16.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %x.param_patt.loc7_16.2 => constants.%x.param_patt.0f9
// CHECK:STDOUT:   %x.patt.loc7_16.2 => constants.%x.patt.953
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d8d
// CHECK:STDOUT:   %x.param_patt.loc7_16.2 => constants.%x.param_patt.23b
// CHECK:STDOUT:   %x.patt.loc7_16.2 => constants.%x.patt.c51
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }

// --- STDIN
[[@LSP-CALL:initialize:"capabilities": {}]]
[[@LSP-NOTIFY:textDocument/didOpen:
  "textDocument": {
    "uri": "file:/sem_ir_text.carbon",
    "languageId": "carbon-testdata",
    "text": "FROM_FILE_SPLIT"
  }
]]
// An operand with an explicit scope resolves in that scope.
// Line 42, character 71 is the `constants.%F` in:
//   // CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 42, "character": 71}
]]
// A type annotation resolves in `constants`, not the enclosing `file` scope.
// Line 42, character 29 is the `%F.type` in:
//   // CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 42, "character": 29}
]]
// Inside a `fn_decl @F` the scope is `@F`, not the lexically enclosing `file`.
// This name is also defined in a generic and has a `locN_M` suffix, so it has
// specific values and a source location too. Each specific value is shown as the
// definition of the constant it names, not just the name.
// Line 43, character 102 is the `%T.patt.loc7_7.2` in:
//   // CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 43, "character": 102}
]]
// Line 43, character 102 is the `%T.patt.loc7_7.2` in:
//   // CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
[[@LSP-CALL:textDocument/definition:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 43, "character": 102}
]]
// Line 43, character 102 is the `%T.patt.loc7_7.2` in:
//   // CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
[[@LSP-CALL:textDocument/declaration:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 43, "character": 102}
]]
// Line 43, character 102 is the `%T.patt.loc7_7.2` in:
//   // CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
[[@LSP-CALL:textDocument/implementation:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 43, "character": 102}
]]
// Line 43, character 102 is the `%T.patt.loc7_7.2` in:
//   // CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
[[@LSP-CALL:textDocument/references:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 43, "character": 102},
  "context": {"includeDeclaration": true}
]]
// The left side of a `specific` row names an instruction of the generic.
// Line 76, character 20 is the `%T.loc7_7.1` in:
//   // CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 76, "character": 20}
]]
// Line 76, character 20 is the `%T.loc7_7.1` in:
//   // CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
[[@LSP-CALL:textDocument/definition:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 76, "character": 20}
]]
// A definition that opens a brace group runs to its closing brace, and is shown
// in full.
// Line 47, character 22 is the `%.loc7_9.1` in:
//   // CHECK:STDOUT:     %.loc7_9.1: type = splice_block %.loc7_9.2 [concrete = type] {
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 47, "character": 22}
]]
// A scope keyword is not an operand.
// Line 17, character 18 is the `constants` in:
//   // CHECK:STDOUT: constants {
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 17, "character": 18}
]]
// Test files aren't compiled, so the Carbon source has no information.
// Line 7, character 3 is the `F[` in:
//   fn F[T: type](x: T) {}
[[@LSP-CALL:textDocument/hover:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 7, "character": 3}
]]
// Line 7, character 3 is the `F[` in:
//   fn F[T: type](x: T) {}
[[@LSP-CALL:textDocument/definition:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "position": {"line": 7, "character": 3}
]]
// Test files have no parse tree, so these produce empty results rather than
// crashing.
[[@LSP-CALL:textDocument/documentSymbol:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"}
]]
[[@LSP-CALL:textDocument/formatting:
  "textDocument": {"uri": "file:/sem_ir_text.carbon"},
  "options": {"tabSize": 2, "insertSpaces": true}
]]
[[@LSP-CALL:shutdown]]
[[@LSP-NOTIFY:exit]]

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: Content-Length: 432{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 1,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "capabilities": {
// CHECK:STDOUT:       "declarationProvider": true,
// CHECK:STDOUT:       "definitionProvider": true,
// CHECK:STDOUT:       "documentFormattingProvider": true,
// CHECK:STDOUT:       "documentSymbolProvider": true,
// CHECK:STDOUT:       "hoverProvider": true,
// CHECK:STDOUT:       "implementationProvider": true,
// CHECK:STDOUT:       "positionEncoding": "utf-16",
// CHECK:STDOUT:       "referencesProvider": true,
// CHECK:STDOUT:       "textDocumentSync": 2,
// CHECK:STDOUT:       "typeDefinitionProvider": true
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 151{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "method": "textDocument/publishDiagnostics",
// CHECK:STDOUT:   "params": {
// CHECK:STDOUT:     "diagnostics": [],
// CHECK:STDOUT:     "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 346{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 2,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "contents": {
// CHECK:STDOUT:       "kind": "markdown",
// CHECK:STDOUT:       "value": "```semir\nconstants.%F: %F.type = struct_value () [concrete]\n```\n"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     "range": {
// CHECK:STDOUT:       "end": {
// CHECK:STDOUT:         "character": 73,
// CHECK:STDOUT:         "line": 42
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "start": {
// CHECK:STDOUT:         "character": 61,
// CHECK:STDOUT:         "line": 42
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 343{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 3,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "contents": {
// CHECK:STDOUT:       "kind": "markdown",
// CHECK:STDOUT:       "value": "```semir\nconstants.%F.type: type = fn_type @F [concrete]\n```\n"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     "range": {
// CHECK:STDOUT:       "end": {
// CHECK:STDOUT:         "character": 35,
// CHECK:STDOUT:         "line": 42
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "start": {
// CHECK:STDOUT:         "character": 28,
// CHECK:STDOUT:         "line": 42
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 847{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 4,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "contents": {
// CHECK:STDOUT:       "kind": "markdown",
// CHECK:STDOUT:       "value": "```semir\n@F.%T.patt.loc7_7.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]\n```\n\nSpecific values:\n\n```semir\nspecific @F(constants.%T) =>\n  constants.%T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]\nspecific @F(constants.%C) =>\n  constants.%T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]\nspecific @F(constants.%D) =>\n  constants.%T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]\n```\n\nSource:\n\n```carbon\nfn F[T: type](x: T) {}\n```\n"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     "range": {
// CHECK:STDOUT:       "end": {
// CHECK:STDOUT:         "character": 117,
// CHECK:STDOUT:         "line": 43
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "start": {
// CHECK:STDOUT:         "character": 101,
// CHECK:STDOUT:         "line": 43
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 285{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 5,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 35,
// CHECK:STDOUT:           "line": 59
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 59
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 281{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 6,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 6,
// CHECK:STDOUT:           "line": 7
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 6,
// CHECK:STDOUT:           "line": 7
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 753{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 7,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 35,
// CHECK:STDOUT:           "line": 75
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 75
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 35,
// CHECK:STDOUT:           "line": 83
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 83
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 35,
// CHECK:STDOUT:           "line": 94
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 94
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 1458{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 8,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 117,
// CHECK:STDOUT:           "line": 43
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 101,
// CHECK:STDOUT:           "line": 43
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 35,
// CHECK:STDOUT:           "line": 59
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 59
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 115,
// CHECK:STDOUT:           "line": 59
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 99,
// CHECK:STDOUT:           "line": 59
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 35,
// CHECK:STDOUT:           "line": 75
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 75
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 35,
// CHECK:STDOUT:           "line": 83
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 83
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 35,
// CHECK:STDOUT:           "line": 94
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 94
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 715{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 9,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "contents": {
// CHECK:STDOUT:       "kind": "markdown",
// CHECK:STDOUT:       "value": "```semir\n@F.%T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]\n```\n\nSpecific values:\n\n```semir\nspecific @F(constants.%T) =>\n  constants.%T: type = symbolic_binding T, 0 [symbolic]\nspecific @F(constants.%C) =>\n  constants.%C: type = class_type @C [concrete]\nspecific @F(constants.%D) =>\n  constants.%D: type = class_type @D [concrete]\n```\n\nSource:\n\n```carbon\nfn F[T: type](x: T) {}\n```\n"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     "range": {
// CHECK:STDOUT:       "end": {
// CHECK:STDOUT:         "character": 30,
// CHECK:STDOUT:         "line": 76
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "start": {
// CHECK:STDOUT:         "character": 19,
// CHECK:STDOUT:         "line": 76
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 286{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 10,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": [
// CHECK:STDOUT:     {
// CHECK:STDOUT:       "range": {
// CHECK:STDOUT:         "end": {
// CHECK:STDOUT:           "character": 30,
// CHECK:STDOUT:           "line": 60
// CHECK:STDOUT:         },
// CHECK:STDOUT:         "start": {
// CHECK:STDOUT:           "character": 19,
// CHECK:STDOUT:           "line": 60
// CHECK:STDOUT:         }
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "uri": "file:///sem_ir_text.carbon"
// CHECK:STDOUT:     }
// CHECK:STDOUT:   ]
// CHECK:STDOUT: }Content-Length: 567{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 11,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "contents": {
// CHECK:STDOUT:       "kind": "markdown",
// CHECK:STDOUT:       "value": "```semir\n@F.%.loc7_9.1: type = splice_block %.loc7_9.2 [concrete = type] {\n  %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]\n  %.loc7_9.2: type = type_literal type [concrete = type]\n}\n```\n\nSource:\n\n```carbon\nfn F[T: type](x: T) {}\n```\n"
// CHECK:STDOUT:     },
// CHECK:STDOUT:     "range": {
// CHECK:STDOUT:       "end": {
// CHECK:STDOUT:         "character": 31,
// CHECK:STDOUT:         "line": 47
// CHECK:STDOUT:       },
// CHECK:STDOUT:       "start": {
// CHECK:STDOUT:         "character": 21,
// CHECK:STDOUT:         "line": 47
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 74{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 12,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "contents": null
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 74{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 13,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": {
// CHECK:STDOUT:     "contents": null
// CHECK:STDOUT:   }
// CHECK:STDOUT: }Content-Length: 50{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 14,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": []
// CHECK:STDOUT: }Content-Length: 50{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 15,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": []
// CHECK:STDOUT: }Content-Length: 50{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 16,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": []
// CHECK:STDOUT: }Content-Length: 52{{\r}}
// CHECK:STDOUT: {{\r}}
// CHECK:STDOUT: {
// CHECK:STDOUT:   "id": 17,
// CHECK:STDOUT:   "jsonrpc": "2.0",
// CHECK:STDOUT:   "result": null
// CHECK:STDOUT: }
