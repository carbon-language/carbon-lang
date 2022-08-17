# Comments

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Use case](#use-cases)
    -   [Comment format](#comment-format)
    -   [Implementation-comments](#implementation-comments)
    -   [Disabling Code](#disabling-code)
    -   [Reserved Comments](#reserved-comments)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

## Overview

A _comment_ is a lexical element beginning with the characters `//` and running
to the end of the line. We have no mechanism for physical line continuation, so
a trailing `\` does not extend a comment to subsequent lines. For example:

```carbon
// This is a comment. Next line is also a valid comment.
//

This is not a comment because it does not starts with '//'.
//This is not a comment because there is no whitespace after '//'.
var Int: x; // error, because there are non-whitespace characters before '//'.
```

## Details

### Use cases

Comments serve a variety of purposes in existing programming languages. The
primary use cases are:

-   _Documentation_: human-readable commentary explaining to users and future
    maintainers of an API what function it performs and how to use it. Such
    comments are typically attached to function declarations, class definitions,
    public member declarations, at file scope, and similar levels of granularity
    in an API.

    ```
    /// A container for a collection of connected widgets.
    class WidgetAssembly {
      /// Improve the appearance of the assembly if possible.
      void decorate(bool repaint_all = false);

      // ...
    };
    ```

-   _Implementation comments_: human-readable commentary explaining intent and
    mechanism to future readers or maintainers of code, or summarizing the
    behavior of code to avoid readers or maintainers needing to read it in
    detail. Such comments are typically used when such details may not be
    readily apparent from the code itself or may require non-trivial work to
    infer, and tend to be short.

    ```
    void WidgetAssembly::decorate(bool repaint_all) {
      // ...

      // Paint all the widgets that have been changed since last time.
      for (auto &w : widgets) {
        if (repaint_all || w.modified > last_foo)
          w.paint();
      }
      last_decorate = now();

      // ...
    }
    ```

-   _Syntactic disambiguation comments_: comments that contain code or
    pseudocode intended to allow the human reader to more easily parse the code
    in the same way that the compiler does.

    ```
    void WidgetAssembly::decorate(bool repaint_all /*= false*/) {
    // ...

    /*static*/ std::unique_ptr<WidgetAssembly> WidgetAssembly::make() {
    // ...

    assembly.decorate(/*repaint_all=*/true);
    // ...

    }  // end namespace WidgetLibrary
    ```

-   _Disabled code_: comments that contain regions of code that have been
    disabled, because the code is incomplete or incorrect, or in order to
    isolate a problem while debugging, or as reference material for a change in
    progress. It is often considered bad practice to check such comments into
    version control.


### Comment format

Carbon language provides only one kind of comments unlike C/C++, that starts
with `//` and runs to the end of the line. No code is permitted prior to a
comment on the same line, and the `//` introducing the comment is required to be
followed by whitespace.

This comment syntax can be used as implementation comments and to disable code.
The documentation use case is to be covered by a separate mechanism specially
design for the purpose. The syntactic disambiguation use case is not covered,
with the intent that the language syntax is designed in a way that avoids this
use case.

### Implementation comments

The implementation comments can be added just above a particular line/block of
code to describe the intent.For example:

```carbon
// The next line declares an integer variable 'x'.
var Int: x;
```

The multiline/long comments can be written by breaking the comments into
multiple lines and appending `//` at the start of each line. Ensure there is a
whitespace after each `//`. For example:

```carbon
// This is an example of a multiline comments in the Carbon language. This block
// of code declares a function that takes two integers and returns their sum, as
// well as prints the output.
fn add(var a: i32, var b: i32) -> i32 {
    return a + b;
}
```

## Disabling code

To disable a line of code, simply append `//` and a whitespace before that line.
For example:

```carbon
// var Int: x;
```

This syntax can also be used to comment multiple line or block of code. Note
that Carbon language does not provides any syntax for block comments. This is
to reduce ambiguity and confusion. for example:

```carbon
// fn disabledCode(var x: i32){
//     Print("This is a disabled code.");
// }
```

### Reserved comments

Comments in which the `//` characters are not followed by whitespace are
reserved for future extension. Anticipated possible extensions are block
comments, documentation comments, and code folding region markers.

## Alternative considered


## References

-   Proposal
    [#198: Comments](https://github.com/carbon-language/carbon-lang/pull/198)