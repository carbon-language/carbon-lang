/// A token as recognized by the lexical analyzer.
///
/// - Note: TokenID does not include the token's text, except inasmuch as it may
///   be implied (e.g. for keywords, which only match one string.)
typealias TokenID = CarbonParser.CitronTokenCode

struct Token: Hashable {
  init(_ kind: TokenID, _ text: String) {
    self.kind = kind
    self.text = text
  }
  let kind: TokenID
  let text: String
}

extension Token: CustomStringConvertible {
  var description: String {
    "Token(.\(kind), \(String(reflecting: text)))"
  }
}

