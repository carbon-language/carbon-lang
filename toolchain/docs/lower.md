# Lower

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Mangling](#mangling)
    -   [Examples](#examples)

<!-- tocstop -->

## Overview

Lowering takes the SemIR and produces LLVM IR. At present, this is done in a
single pass, although it's possible we may need to do a second pass so that we
can first generate type information for function arguments.

Lowering is done per `SemIR::InstBlock`. This minimizes changes to the
`IRBuilder` insertion point, something that is both expensive and potentially
fragile.

## Mangling

Part of lowering is choosing deterministically unique identifiers for each
lowered entity to use in platform object files. Any feature of an entity (such
as parent namespaces or overloaded function parameters) that would create a
distinct entity must be included in some way in the generated identifier.

The current rudimentary name mangling scheme is as follows:

-   As a special case, `Main.Run` is emitted as `main`.

Otherwise the resulting name consists of:

1.  `_C`
2.  The unqualified function name (function name mangling is the only thing
    implemented at the moment).
3.  `.`
4.  If the function being mangled is a member of:
    -   an `impl`, then add:
        1.  The implementing type, per the scope mangling.
        2.  `:`
        3.  The interface type, per the scope mangling.
    -   a type or namespace, then add:
        1.  The scope, per the scope mangling.

The scope mangling scheme is as follows:

1.  The unqualified name of the type or namespace.
2.  If the type or namespace is within another type or namespace:
    1.  `.`
    2.  The enclosing scope, per the scope mangling.
3.  `.`
4.  The package name.

### Examples

```carbon
package P1;
interface Interface {
  fn Op[self: Self]();
}
```

```carbon
namespace NameSpace;
class NameSpace.Implementation {
  // Mangled as:
  // `_COp.Implementation.NameSpace.Main:Interface.P1`
  impl as P1.Interface {
    fn Op[self: Self]() {
    }
  }
}
// Mangled as `main`.
fn Run() {
  var v: NameSpace.Implementation;
  v.(P1.Interface.Op)();
}
```
