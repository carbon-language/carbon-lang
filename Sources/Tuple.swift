// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Identifier for a tuple element or parameter to a particular function type.
///
/// Fields of carbon tuples may be named or positional.
enum FieldID: Hashable {
  /// field identified by label (syntactically, ".<label> = ...")
  case label(Identifier)
  /// field identified by its offset in the sequence of positional fields.
  case position(Int)
}

typealias TupleValue = [FieldID: Value]
typealias TupleType = [FieldID: Type]

extension TupleValue: CarbonInterpreter.Value {
  var type: Type {
    .tuple(self.mapValues { $0.type })
  }
}

extension TupleSyntax {
  /// Returns a form of this AST node that is agnostic to the ordering of
  /// labeled fields, adding an error to errors if there are any duplicate
  /// fields.
  func fields(
    reportingDuplicatesIn errors: inout ErrorLog
  ) -> [FieldID: Payload] {
    var r: [FieldID: Payload] = [:]
    var positionalCount = 0
    for e in elements {
      let key: FieldID
        = e.label.map { .label($0) } ?? .position(positionalCount)
      if case .position = key { positionalCount += 1 }
      if let other = r[key] {
        errors.append(
          CompileError(
            "Duplicate label \(e.label!)", at: e.label!.site,
            notes: [("other definition", other.site)]))
      }
      else {
        r[key] = e.payload
      }
    }
    return r
  }
}
