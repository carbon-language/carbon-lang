// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Identifies a textual pattern matched by the lexical analyzer.
///
/// - Note: TokenID does not include the token's text, except inasmuch as it may
///   be implied (e.g. for keywords, which only match one string.)
typealias TokenID = CarbonParser.CitronTokenCode

/// A symbol recognized by the lexical analyzer.
struct Token: AST {
  /// Creates an instance of the given token kind and content.
  init(_ kind: TokenID, _ content: String, _ site: Site) {
    self.kind = kind
    self.text = content
    self.site = site
  }

  /// The lexical analyzer pattern that matched this token.
  let kind: TokenID

  /// The textual content of this token.
  let text: String

  let site: Site
}

extension Token: CustomStringConvertible {
  /// A textual description of `self`.
  var description: String {
    "Token(.\(kind), \(String(reflecting: text)), \(String(reflecting: site)))"
  }
}

