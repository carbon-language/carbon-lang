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

## Alternative considered

