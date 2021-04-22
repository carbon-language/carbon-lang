// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// An error produced at compile-time.
struct CompileError: Error {
  /// An additional informative note to go with the error message.
  typealias Note = (message: String, site: ASTSite)

  /// A human-readable description of the problem.
  let message: String
  /// Where to point in the source code
  let site: ASTSite
  /// Any additional notes
  let notes: [Note]

  /// Creates an instance with the given properties.
  init(_ message: String, at site: ASTSite, notes: [Note] = []) {
    self.message = message
    self.site = site
    self.notes = notes
  }
}

extension CompileError: CustomStringConvertible {
  /// String representation that, if printed at the beginning of the line,
  /// should be recognized by IDEs.
  var description: String {
    return (
      ["\(site): \(message)"] + notes.enumerated().lazy.map {
        (i, n) in "\(n.site): note(\(i)): \(n.message)"
      }).joined(separator: "\n")
  }
}


/// This will be thrown from executable program construction if there's a
/// failure.
typealias ErrorLog = [CompileError]
extension ErrorLog: Error {}
