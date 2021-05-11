// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// An error produced at compile-time.
struct CompileError: Error, Equatable {
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

  static func == (l: Self, r: Self) -> Bool {
    l.message == r.message && l.site == r.site
    && l.notes.lazy.map(\.message) == r.notes.lazy.map(\.message)
      && l.notes.lazy.map(\.site) == r.notes.lazy.map(\.site)
  }
}

extension CompileError: CustomStringConvertible {
  /// String representation that, if printed at the beginning of the line,
  /// should be recognized by IDEs.
  var description: String {
    return (
      ["\(site): error: \(message)"] + notes.enumerated().lazy.map {
        (i, n) in "\(n.site): note(\(i)): \(n.message)"
      }).joined(separator: "\n")
  }
}

extension CompileError {
  /// Returns `l` offset by `r`.
  static func + (l: Self, r: SourcePosition.Offset) -> Self {
    Self(
      l.message, at: ASTSite(devaluing: l.site.region + r),
      notes: l.notes.map {
        ($0.message, ASTSite(devaluing: $0.site.region + r))
      })
  }

  /// Returns `r` offset by `l`.
  static func + (l: SourcePosition.Offset, r: Self) -> Self {
    Self(
      r.message, at: ASTSite(devaluing: r.site.region + l),
      notes: r.notes.map {
        ($0.message, ASTSite(devaluing: $0.site.region + l))
      })
  }
}

/// This will be thrown from executable program construction if there's a
/// failure.
/*
struct ErrorLog {
  var contents: [CompileError]
}
 */
typealias ErrorLog = [CompileError]
extension ErrorLog: Error {}

