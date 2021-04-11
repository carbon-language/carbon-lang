// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// The address of a (global or local) symbol, or temporary, with respect
/// to the currently-executing function context.
struct RelativeAddress: Hashable {
  init(_ base: Context, _ offset: Int) {
    self.base = base
    self.offset = offset
  }

  /// Symbolic base of a `RelativeAddress`.
  enum Context {
    /// The address is to be resolved globally.
    case global 

    /// The address is in the currently-executing function's frame.
    ///
    /// Parameters, local variable, or temporaries  are `base`d here.
    case local  

    /// The address is in the frame of a function that is about to be called.
    ///
    /// The subexpressions in a function call's argument list are `base`d here.
    case callee 
    
    // expect to add "this-relative" for lambdas and method access.
  }
  
  let base: Context
  let offset: Int
  
  /// Returns the absolute address corresponding to `self` in `i`.
  func resolved(in i: Interpreter) -> Address {
    switch base {
    case .global: return offset
    case .local: return offset + i.functionContext.frameBase
    case .callee: return offset + i.functionContext.calleeFrameBase
    }
  }
}

