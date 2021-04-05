// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A token as recognized by the lexical analyzer.
///
/// - Note: TokenID does not include the token's text, except inasmuch as it may
///   be implied (e.g. for keywords, which only match one string.)
typealias TokenID = CarbonParser.CitronTokenCode

struct Token_: Hashable {
  init(_ kind: TokenID, _ text: String) {
    self.kind = kind
    self.text = text
  }
  let kind: TokenID
  let text: String
}

extension Token_: CustomStringConvertible {
  var description: String {
    "Token_(.\(kind), \(String(reflecting: text)))"
  }
}

typealias Token = AST<Token_>
