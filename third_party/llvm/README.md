# Extracted LLVM code

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This directory holds code extracted from the LLVM project and used in various
places in Carbon. This mostly happens when some logic isn't readily exposed in a
library-suitable API, or needs excessive customization.

We separate the baseline code and customizations here as these are really
derived from the LLVM project. However, both Carbon and LLVM use the same
license, so linking these isn't a problem. We also use the standard Carbon
banner at the top of these files as the license is accurate and the collection
and arrangement of the code are part of Carbon. The code itself should be
assumed to be derived from LLVM.

Whenever possible, we should eventually refactor the upstream code until the
logic can be exposed and then we can depend on it rather than extracting it
here.
