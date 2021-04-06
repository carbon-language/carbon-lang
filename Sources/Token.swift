// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Identifies a textual pattern matched by the lexical analyzer.
///
/// - Note: TokenID does not include the token's text, except inasmuch as it may
///   be implied (e.g. for keywords, which only match one string.)
typealias TokenID = CarbonParser.CitronTokenCode

/// A symbol recognized by the lexical analyzer.
typealias Token = AST<Token_>

/// The body of a `Token`.
struct Token_: Hashable {
  /// Creates an instance of the given token kind and content.
  init(_ kind: TokenID, _ content: String) {
    self.kind = kind
    self.text = content
  }

  /// The lexical analyzer pattern that matched this token.
  let kind: TokenID

  /// The textual content of this token.
  let text: String
}

extension Token_: CustomStringConvertible {
  /// A textual description of `self`.
  var description: String {
    "Token_(.\(kind), \(String(reflecting: text)))"
  }
}

