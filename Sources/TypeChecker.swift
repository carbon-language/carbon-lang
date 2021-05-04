// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

var UNIMPLEMENTED: Never { fatalError("unimplemented") }

struct TypeChecker {
  init(_ program: ExecutableProgram) {
    self.program = program

    for d in program.ast {
      checkNominalTypeBody(d)
    }
    /*
    for d in program.ast {
      checkFunctionSignature(d)
    }
    for d in program.ast {
      checkFunctionBody(d)
    }
    */
  }

  private let program: ExecutableProgram

  private(set) var types = Dictionary<AnyASTIdentity, Type>()

  /// Return type of function currently being checked, if any.
  private var returnType: Type?

  /// A record of the collected errors.
  var errors: ErrorLog = []
}

private extension TypeChecker {
  /// Adds an error at the site of `offender` to the error log.
  mutating func error<Node: AST>(
    _ offender: Node, _ message: String , notes: [CompileError.Note] = []
  ) {
    errors.append(CompileError(message, at: offender.site, notes: notes))
  }
}

private extension TypeChecker {
  /// Typechecks `d`, recording any errors in `self.errors`
  mutating func checkNominalTypeBody(_ d: TopLevelDeclaration) {
    switch d {
    case let .struct(s): checkBody(s)
    case let .choice(c): checkBody(c)
    case .function, .initialization: ()
    }
  }

  mutating func checkBody(_ s: StructDefinition) {
    for m in s.members {
      types[m.identity] = evaluate(m.type)
    }
  }

  mutating func checkBody(_ c: ChoiceDefinition) {
    for a in c.alternatives {
      types[a.identity] = .error
      UNIMPLEMENTED
    }
  }

  /// Returns the type defined by `t` or `.error` if `d` doesn't define a type.
  mutating func evaluate(_ e: TypeExpression) -> Type {
    let v = evaluate(e.body)
    if let r = (v as? Type) { return r }

    // If the value is a tuple, check that all its elements are types.
    if let elements = (v as? TupleValue) {
      let typeElements = elements.compactMapValues { $0 as? Type }
      if typeElements.count == elements.count {
        return .tuple(typeElements)
      }
    }

    error(e, "Not a type expression (value has type \(v.type)).")
    return .error
  }

  mutating func evaluate(_ e: Expression) -> Value {
    // Temporarily evaluating the easy subset of type expressions until we have
    // an interpreter.
    switch e {
    case let .name(v):
      if let r = Type(program.definition[v]!) {
        return r
      }
      UNIMPLEMENTED
    case .getField(_, _, _): UNIMPLEMENTED
    case .index(target: _, offset: _, _): UNIMPLEMENTED
    case let .integerLiteral(r, _): return r
    case let .booleanLiteral(r, _): return r
    case let .tupleLiteral(t):
      if let e = t.duplicateLabelError { errors.append(e) }
      return t.fields.mapValues { self.evaluate($0) }
    case .unaryOperator(operation: _, operand: _, _): UNIMPLEMENTED
    case .binaryOperator(operation: _, lhs: _, rhs: _, _): UNIMPLEMENTED
    case .functionCall(_): UNIMPLEMENTED
    case .intType: return Type.int
    case .boolType: return Type.bool
    case .typeType: return Type.type
    case let .functionType(f):
      if let e = f.parameters.duplicateLabelError { errors.append(e) }
      let p = evaluate(TypeExpression(f.parameters)).tuple!
      return Type.function(
        parameterTypes: p, returnType: evaluate(f.returnType))
    }
  }}
